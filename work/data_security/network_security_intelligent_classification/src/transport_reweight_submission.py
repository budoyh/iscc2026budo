from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw" / "data"
MODELS = ROOT / "models"
SUBMISSIONS = ROOT / "submissions"


PATTERN_FEATURES = {
    "pattern_transition_ratio",
    "pattern_diversity_ratio",
    "pattern_switch_frequency",
    "pattern_mix_density",
    "pattern_change_ratio",
    "pattern_concentration",
}


GROUPS = [
    ["class_0", "class_10", "class_11"],
    ["class_2", "class_7"],
    ["class_4", "class_8"],
    ["class_5", "class_6"],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reweight neighboring class probabilities by stable-space transport distance.")
    parser.add_argument("--blend-report", required=True)
    parser.add_argument("--output-dir", default="upload_ready_transport")
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--margin-threshold", type=float, default=0.35)
    parser.add_argument("--advantage-threshold", type=float, default=0.02)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--max-change-rate", type=float, default=0.035)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def apply_uniform_prior(proba: np.ndarray, alpha: float) -> np.ndarray:
    target = np.full(proba.shape[1], 1.0 / proba.shape[1], dtype=np.float64)
    current = np.clip(proba.mean(axis=0), 1e-15, 1.0)
    out = proba * ((target / current) ** alpha)
    out /= out.sum(axis=1, keepdims=True)
    return out


def load_blend(report_path: Path, classes: list[str]) -> tuple[np.ndarray, np.ndarray, dict]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    weights = np.asarray(report["weights"], dtype=np.float64)
    weights /= weights.sum()
    oof = None
    test = None
    for run_id, weight in zip(report["run_ids"], weights):
        meta = json.loads((MODELS / run_id / "metadata.json").read_text(encoding="utf-8"))
        if meta["classes"] != classes:
            raise ValueError(f"class order mismatch for {run_id}")
        current_oof = np.load(MODELS / run_id / "oof_proba.npy")
        current_test = np.load(MODELS / run_id / "test_proba.npy")
        if oof is None:
            oof = np.zeros_like(current_oof, dtype=np.float64)
            test = np.zeros_like(current_test, dtype=np.float64)
        oof += weight * current_oof
        test += weight * current_test
    assert oof is not None and test is not None
    if report.get("soft_prior") == "uniform":
        oof = apply_uniform_prior(oof, float(report.get("soft_alpha", 1.0)))
        test = apply_uniform_prior(test, float(report.get("soft_alpha", 1.0)))
    return oof, test, report


def stable_features(train: pd.DataFrame) -> list[str]:
    return [col for col in train.columns if col not in {"id", "label"} and col not in PATTERN_FEATURES]


def class_distances(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    y: np.ndarray,
    class_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler().fit(pd.concat([train[features], test[features]], ignore_index=True))
    z_train = scaler.transform(train[features]).astype(np.float32)
    z_test = scaler.transform(test[features]).astype(np.float32)
    means = np.zeros((class_count, z_train.shape[1]), dtype=np.float32)
    stds = np.zeros((class_count, z_train.shape[1]), dtype=np.float32)
    for cls in range(class_count):
        cls_z = z_train[y == cls]
        means[cls] = cls_z.mean(axis=0)
        stds[cls] = np.maximum(cls_z.std(axis=0), 0.25)
    train_dist = np.zeros((len(train), class_count), dtype=np.float32)
    test_dist = np.zeros((len(test), class_count), dtype=np.float32)
    for cls in range(class_count):
        train_dist[:, cls] = np.sqrt((((z_train - means[cls]) / stds[cls]) ** 2).mean(axis=1))
        test_dist[:, cls] = np.sqrt((((z_test - means[cls]) / stds[cls]) ** 2).mean(axis=1))
    return train_dist, test_dist


def reweight(
    proba: np.ndarray,
    distances: np.ndarray,
    classes: list[str],
    margin_threshold: float,
    advantage_threshold: float,
    beta: float,
    max_change_rate: float,
) -> tuple[np.ndarray, dict[str, object]]:
    out = proba.copy()
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    index_groups = [[class_to_idx[name] for name in group] for group in GROUPS]
    pred = proba.argmax(axis=1)
    sorted_proba = np.sort(proba, axis=1)
    margin = sorted_proba[:, -1] - sorted_proba[:, -2]
    candidates = []
    for group in index_groups:
        group_arr = np.asarray(group, dtype=np.int64)
        in_group = np.isin(pred, group_arr)
        if not np.any(in_group):
            continue
        group_dist = distances[:, group_arr]
        nearest_pos = group_dist.argmin(axis=1)
        nearest = group_arr[nearest_pos]
        pred_dist = distances[np.arange(len(proba)), pred]
        nearest_dist = distances[np.arange(len(proba)), nearest]
        advantage = pred_dist - nearest_dist
        mask = in_group & (nearest != pred) & (advantage > advantage_threshold) & (margin < margin_threshold)
        idx = np.where(mask)[0]
        if len(idx):
            candidates.append((idx, nearest[idx], advantage[idx], margin[idx]))
    if candidates:
        all_idx = np.concatenate([item[0] for item in candidates])
        all_nearest = np.concatenate([item[1] for item in candidates])
        all_advantage = np.concatenate([item[2] for item in candidates])
        all_margin = np.concatenate([item[3] for item in candidates])
        order = np.argsort(all_advantage - all_margin)[::-1]
        take_count = min(len(order), int(round(max_change_rate * len(proba))))
        chosen = order[:take_count]
        for idx, target, advantage in zip(all_idx[chosen], all_nearest[chosen], all_advantage[chosen]):
            factor = float(np.exp(beta * advantage))
            out[idx, target] *= factor
            out[idx] /= out[idx].sum()
    changed = int((out.argmax(axis=1) != pred).sum())
    return out, {
        "candidate_count": int(sum(len(item[0]) for item in candidates)) if candidates else 0,
        "changed_count": changed,
        "changed_rate": changed / len(proba),
        "label_counts": dict(zip(classes, np.bincount(out.argmax(axis=1), minlength=len(classes)).astype(int).tolist())),
    }


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
    train = pd.read_csv(RAW / "train_data.csv")
    test = pd.read_csv(RAW / "test_data.csv")
    sample = pd.read_csv(RAW / "sample_submission.csv")
    encoder = LabelEncoder()
    y = encoder.fit_transform(train["label"]).astype(np.int64)
    classes = encoder.classes_.tolist()
    class_array = np.asarray(classes)
    oof, test_proba, report = load_blend(Path(args.blend_report), classes)
    features = stable_features(train)
    train_dist, test_dist = class_distances(train, test, features, y, len(classes))
    oof_adj, oof_info = reweight(
        oof,
        train_dist,
        classes,
        args.margin_threshold,
        args.advantage_threshold,
        args.beta,
        args.max_change_rate,
    )
    test_adj, test_info = reweight(
        test_proba,
        test_dist,
        classes,
        args.margin_threshold,
        args.advantage_threshold,
        args.beta,
        args.max_change_rate,
    )
    base_oof = float(f1_score(y, oof.argmax(axis=1), average="macro"))
    adjusted_oof = float(f1_score(y, oof_adj.argmax(axis=1), average="macro"))
    payload = {
        "base_oof_macro_f1": base_oof,
        "adjusted_oof_macro_f1": adjusted_oof,
        "oof_info": oof_info,
        "test_info": test_info,
        "features": features,
        "groups": GROUPS,
        "blend_report": str(args.blend_report),
        "margin_threshold": args.margin_threshold,
        "advantage_threshold": args.advantage_threshold,
        "beta": args.beta,
        "max_change_rate": args.max_change_rate,
    }
    if args.dry_run:
        print(json.dumps(payload, ensure_ascii=False))
        return

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame({"id": sample["id"], "label": class_array[test_adj.argmax(axis=1)]})
    output_path = output_dir / f"{args.output_name}.csv"
    mirror_path = SUBMISSIONS / f"{args.output_name}.csv"
    submission.to_csv(output_path, index=False, encoding="utf-8", lineterminator="\r\n")
    submission.to_csv(mirror_path, index=False, encoding="utf-8", lineterminator="\r\n")
    payload.update(validate(submission, sample, set(classes)))
    payload["output_path"] = str(output_path)
    payload["mirror_path"] = str(mirror_path)
    output_path.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    mirror_path.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
