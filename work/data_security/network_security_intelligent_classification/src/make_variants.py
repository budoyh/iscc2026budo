from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw" / "data"
MODELS = ROOT / "models"
SUBMISSIONS = ROOT / "submissions"
UPLOAD_READY = ROOT / "upload_ready"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create calibrated submission variants.")
    parser.add_argument("--run-ids", nargs="+", required=True)
    parser.add_argument("--weights", nargs="*", type=float, default=None)
    parser.add_argument("--name-prefix", required=True)
    parser.add_argument(
        "--soft-alphas",
        nargs="*",
        type=float,
        default=[1.0],
        help="Strengths for soft prior adjustment. 0 keeps original probabilities; 1 fully applies the prior factor.",
    )
    return parser.parse_args()


def normalize_weights(run_ids: list[str], weights: list[float] | None) -> np.ndarray:
    if not weights:
        return np.full(len(run_ids), 1.0 / len(run_ids), dtype=np.float64)
    if len(weights) != len(run_ids):
        raise ValueError("weights count must match run ids")
    weights_arr = np.asarray(weights, dtype=np.float64)
    return weights_arr / weights_arr.sum()


def load_blend(run_ids: list[str], weights: np.ndarray) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    classes = None
    oof = None
    test = None
    true = None
    for run_id, weight in zip(run_ids, weights):
        run_dir = MODELS / run_id
        meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        current_classes = meta["classes"]
        if classes is None:
            classes = current_classes
            current_oof = np.load(run_dir / "oof_proba.npy")
            current_test = np.load(run_dir / "test_proba.npy")
            oof = np.zeros_like(current_oof)
            test = np.zeros_like(current_test)
            true = pd.read_csv(run_dir / "oof_predictions.csv")["true_label"].to_numpy()
        elif current_classes != classes:
            raise ValueError("all runs must have identical class ordering")
        oof += weight * np.load(run_dir / "oof_proba.npy")
        test += weight * np.load(run_dir / "test_proba.npy")
    assert classes is not None and oof is not None and test is not None and true is not None
    return classes, oof, test, true


def rounded_counts(prior: np.ndarray, total: int) -> np.ndarray:
    raw = prior / prior.sum() * total
    counts = np.floor(raw).astype(int)
    remain = total - counts.sum()
    if remain > 0:
        order = np.argsort(raw - counts)[::-1]
        counts[order[:remain]] += 1
    return counts


def quota_assign(proba: np.ndarray, target_counts: np.ndarray) -> np.ndarray:
    eps = 1e-15
    logp = np.log(np.clip(proba, eps, 1.0))
    pred = logp.argmax(axis=1)
    counts = np.bincount(pred, minlength=proba.shape[1])

    for _ in range(proba.shape[1] * 3):
        deficits = np.where(counts < target_counts)[0]
        if len(deficits) == 0:
            break
        changed = False
        for target in deficits:
            need = int(target_counts[target] - counts[target])
            if need <= 0:
                continue
            donor_ok = counts[pred] > target_counts[pred]
            candidates = np.where((pred != target) & donor_ok)[0]
            if len(candidates) == 0:
                continue
            current = pred[candidates]
            gain = logp[candidates, target] - logp[candidates, current]
            take = candidates[np.argsort(gain)[::-1][:need]]
            for idx in take:
                old = pred[idx]
                if old == target or counts[old] <= target_counts[old]:
                    continue
                pred[idx] = target
                counts[old] -= 1
                counts[target] += 1
                changed = True
        if not changed:
            break
    return pred


def soft_prior_adjust(proba: np.ndarray, target_prior: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    eps = 1e-15
    current_prior = np.clip(proba.mean(axis=0), eps, 1.0)
    factor = (target_prior / current_prior) ** alpha
    adjusted = proba * factor
    adjusted /= adjusted.sum(axis=1, keepdims=True)
    return adjusted.argmax(axis=1)


def write_submission(name: str, ids: pd.Series, labels: np.ndarray, classes: list[str], sample: pd.DataFrame) -> dict:
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    UPLOAD_READY.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"id": ids, "label": labels})
    path = SUBMISSIONS / f"{name}.csv"
    upload_path = UPLOAD_READY / f"{name}.csv"
    df.to_csv(path, index=False, encoding="utf-8", lineterminator="\r\n")
    df.to_csv(upload_path, index=False, encoding="utf-8", lineterminator="\r\n")
    report = {
        "path": str(path),
        "upload_path": str(upload_path),
        "row_count": int(len(df)),
        "columns_ok": df.columns.tolist() == ["id", "label"],
        "id_order_ok": df["id"].equals(sample["id"]),
        "null_count": int(df.isna().sum().sum()),
        "duplicate_id_count": int(df["id"].duplicated().sum()),
        "labels_ok": bool(set(df["label"].unique()).issubset(set(classes))),
        "label_counts": df["label"].value_counts().sort_index().to_dict(),
    }
    path.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    weights = normalize_weights(args.run_ids, args.weights)
    classes, oof, test, true = load_blend(args.run_ids, weights)
    class_array = np.asarray(classes)

    train = pd.read_csv(RAW / "train_data.csv")
    test_frame = pd.read_csv(RAW / "test_data.csv")
    sample = pd.read_csv(RAW / "sample_submission.csv")

    base_oof = class_array[oof.argmax(axis=1)]
    base_score = float(f1_score(true, base_oof, average="macro"))
    print(json.dumps({"base_oof_macro_f1": base_score, "run_ids": args.run_ids, "weights": weights.tolist()}))

    train_prior = train["label"].value_counts(normalize=True).reindex(classes).to_numpy()
    uniform_prior = np.full(len(classes), 1.0 / len(classes))

    variants = {
        "quota_train_prior": train_prior,
        "quota_uniform_prior": uniform_prior,
    }

    soft_variants = {
        "soft_train_prior": train_prior,
        "soft_uniform_prior": uniform_prior,
    }

    for alpha in args.soft_alphas:
        alpha_tag = str(alpha).replace(".", "p").replace("-", "m")
        for variant_name, prior in soft_variants.items():
            pred_idx = soft_prior_adjust(test, prior, alpha=alpha)
            labels = class_array[pred_idx]
            report = write_submission(
                name=f"{args.name_prefix}_{variant_name}_a{alpha_tag}",
                ids=sample["id"],
                labels=labels,
                classes=classes,
                sample=sample,
            )
            report["target_prior"] = dict(zip(classes, prior.astype(float).tolist()))
            report["alpha"] = alpha
            print(json.dumps(report, ensure_ascii=False))

    for variant_name, prior in variants.items():
        counts = rounded_counts(prior, len(test_frame))
        pred_idx = quota_assign(test, counts)
        labels = class_array[pred_idx]
        report = write_submission(
            name=f"{args.name_prefix}_{variant_name}",
            ids=sample["id"],
            labels=labels,
            classes=classes,
            sample=sample,
        )
        report["target_counts"] = dict(zip(classes, counts.astype(int).tolist()))
        print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
