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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write one validated submission from a probability blend.")
    parser.add_argument("--run-ids", nargs="+", required=True)
    parser.add_argument("--weights", nargs="+", type=float, required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--output-dir", default="upload_ready_main")
    parser.add_argument("--soft-prior", choices=["none", "uniform", "train"], default="uniform")
    parser.add_argument("--soft-alpha", type=float, default=1.0)
    return parser.parse_args()


def load_blend(run_ids: list[str], weights: np.ndarray) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    classes: list[str] | None = None
    oof: np.ndarray | None = None
    test: np.ndarray | None = None
    true_labels: np.ndarray | None = None

    for run_id, weight in zip(run_ids, weights):
        run_dir = MODELS / run_id
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        current_classes = metadata["classes"]
        current_oof = np.load(run_dir / "oof_proba.npy")
        current_test = np.load(run_dir / "test_proba.npy")

        if classes is None:
            classes = current_classes
            oof = np.zeros_like(current_oof, dtype=np.float64)
            test = np.zeros_like(current_test, dtype=np.float64)
            true_labels = pd.read_csv(run_dir / "oof_predictions.csv")["true_label"].to_numpy()
        elif current_classes != classes:
            raise ValueError(f"class order mismatch for {run_id}")

        oof += weight * current_oof
        test += weight * current_test

    assert classes is not None and oof is not None and test is not None and true_labels is not None
    return classes, oof, test, true_labels


def apply_soft_prior(proba: np.ndarray, target_prior: np.ndarray, alpha: float) -> np.ndarray:
    current_prior = np.clip(proba.mean(axis=0), 1e-15, 1.0)
    adjusted = proba * ((target_prior / current_prior) ** alpha)
    adjusted /= adjusted.sum(axis=1, keepdims=True)
    return adjusted


def validate(submission: pd.DataFrame, sample: pd.DataFrame, classes: set[str]) -> dict[str, object]:
    return {
        "row_count": int(len(submission)),
        "sample_row_count": int(len(sample)),
        "columns_ok": submission.columns.tolist() == ["id", "label"],
        "id_order_ok": submission["id"].equals(sample["id"]),
        "null_count": int(submission.isna().sum().sum()),
        "duplicate_id_count": int(submission["id"].duplicated().sum()),
        "labels_ok": bool(set(submission["label"].unique()).issubset(classes)),
        "label_counts": submission["label"].value_counts().sort_index().to_dict(),
    }


def main() -> None:
    args = parse_args()
    if len(args.run_ids) != len(args.weights):
        raise ValueError("--weights length must match --run-ids length")

    weights = np.asarray(args.weights, dtype=np.float64)
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("weights must be non-negative and sum to a positive value")
    weights = weights / weights.sum()

    train = pd.read_csv(RAW / "train_data.csv")
    test = pd.read_csv(RAW / "test_data.csv")
    sample = pd.read_csv(RAW / "sample_submission.csv")
    classes, oof, test_proba, true_labels = load_blend(args.run_ids, weights)

    if args.soft_prior != "none":
        if args.soft_prior == "uniform":
            target_prior = np.full(len(classes), 1.0 / len(classes), dtype=np.float64)
        else:
            target_prior = train["label"].value_counts(normalize=True).reindex(classes).to_numpy(dtype=np.float64)
        oof = apply_soft_prior(oof, target_prior, args.soft_alpha)
        test_proba = apply_soft_prior(test_proba, target_prior, args.soft_alpha)

    class_array = np.asarray(classes)
    oof_pred = class_array[oof.argmax(axis=1)]
    test_pred = class_array[test_proba.argmax(axis=1)]
    local_score = float(f1_score(true_labels, oof_pred, average="macro"))

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)

    submission = pd.DataFrame({"id": test["id"], "label": test_pred})
    output_path = output_dir / f"{args.output_name}.csv"
    mirror_path = SUBMISSIONS / f"{args.output_name}.csv"
    submission.to_csv(output_path, index=False, encoding="utf-8", lineterminator="\r\n")
    submission.to_csv(mirror_path, index=False, encoding="utf-8", lineterminator="\r\n")

    report = validate(submission, sample, set(classes))
    report.update(
        {
            "output_path": str(output_path),
            "mirror_path": str(mirror_path),
            "local_macro_f1": local_score,
            "run_ids": args.run_ids,
            "weights": weights.tolist(),
            "soft_prior": args.soft_prior,
            "soft_alpha": args.soft_alpha,
        }
    )
    output_path.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    mirror_path.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
