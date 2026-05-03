from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw" / "data"
MODELS = ROOT / "models"
SUBMISSIONS = ROOT / "submissions"

FEATURE_GROUPS = {
    "template": [
        "behavior_template_time",
        "behavior_template_volume",
        "behavior_template_activity",
        "behavior_template_control",
        "behavior_compactness_score",
    ],
    "top10": [
        "behavior_template_activity",
        "behavior_template_control",
        "behavior_template_volume",
        "control_signal_intensity",
        "protocol_variation_level",
        "behavior_template_time",
        "behavior_compactness_score",
        "payload_unit_mean",
        "direction_rate_gap",
        "direction_gap_dispersion",
    ],
}


@dataclass(frozen=True)
class ViewSpec:
    name: str
    k: int
    purity: float
    min_count: int


def parse_view_spec(text: str, default_min_count: int) -> ViewSpec:
    parts = text.split(":")
    if len(parts) not in {3, 4}:
        raise ValueError(f"view spec must be name:k:purity[:min_count], got {text}")
    name = parts[0]
    if name not in FEATURE_GROUPS:
        raise ValueError(f"unknown view {name}; choices={sorted(FEATURE_GROUPS)}")
    return ViewSpec(
        name=name,
        k=int(parts[1]),
        purity=float(parts[2]),
        min_count=int(parts[3]) if len(parts) == 4 else default_min_count,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="High-purity template cluster override for a base blend.")
    parser.add_argument("--base-report", default="upload_ready_breakthrough4/c04_c03_drop_pattern_50.json")
    parser.add_argument("--view-spec", nargs="+", default=["template:768:0.97:20"])
    parser.add_argument("--mode", choices=["single", "consensus"], default="single")
    parser.add_argument("--min-count", type=int, default=20)
    parser.add_argument("--base-max-conf", type=float, default=1.0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output-name", default=None)
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return ROOT / path


def apply_uniform_prior(proba: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    target = np.full(proba.shape[1], 1.0 / proba.shape[1], dtype=np.float64)
    current = np.clip(proba.mean(axis=0), 1e-15, 1.0)
    out = proba * ((target / current) ** alpha)
    out /= out.sum(axis=1, keepdims=True)
    return out


def load_run(run_id: str, classes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    run_dir = MODELS / run_id
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata["classes"] != classes:
        raise ValueError(f"class order mismatch for {run_id}")
    return np.load(run_dir / "oof_proba.npy").astype(np.float64), np.load(run_dir / "test_proba.npy").astype(np.float64)


def load_report(path: Path, classes: list[str]) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    weights = np.asarray(report["weights"], dtype=np.float64)
    weights /= weights.sum()
    oof = None
    test = None
    for run_id, weight in zip(report["run_ids"], weights):
        current_oof, current_test = load_run(run_id, classes)
        if oof is None:
            oof = np.zeros_like(current_oof, dtype=np.float64)
            test = np.zeros_like(current_test, dtype=np.float64)
        oof += weight * current_oof
        test += weight * current_test
    assert oof is not None and test is not None
    if report.get("soft_prior") == "uniform":
        alpha = float(report.get("soft_alpha", 1.0))
        oof = apply_uniform_prior(oof, alpha)
        test = apply_uniform_prior(test, alpha)
    return oof, test, report


def cluster_rule(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_apply: np.ndarray,
    spec: ViewSpec,
    seed: int,
    num_classes: int,
) -> tuple[np.ndarray, dict[str, float | int]]:
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_apply_scaled = scaler.transform(x_apply)
    model = MiniBatchKMeans(
        n_clusters=spec.k,
        batch_size=4096,
        n_init=5,
        max_iter=250,
        random_state=seed,
        reassignment_ratio=0.0,
    )
    train_cluster = model.fit_predict(x_train_scaled)
    apply_cluster = model.predict(x_apply_scaled)

    cluster_pred = np.full(spec.k, -1, dtype=np.int64)
    cluster_purity = np.zeros(spec.k, dtype=np.float64)
    cluster_count = np.bincount(train_cluster, minlength=spec.k)
    trusted = np.zeros(spec.k, dtype=bool)
    for cluster_id in range(spec.k):
        idx = train_cluster == cluster_id
        count = int(idx.sum())
        if count < spec.min_count:
            continue
        counts = np.bincount(y_train[idx], minlength=num_classes)
        majority = int(counts.argmax())
        purity = float(counts[majority] / count)
        cluster_purity[cluster_id] = purity
        if purity >= spec.purity:
            cluster_pred[cluster_id] = majority
            trusted[cluster_id] = True

    pred = cluster_pred[apply_cluster]
    stats: dict[str, float | int] = {
        "trusted_clusters": int(trusted.sum()),
        "trusted_train_rows": int(cluster_count[trusted].sum()),
        "mean_trusted_purity": float(cluster_purity[trusted].mean()) if np.any(trusted) else 0.0,
        "applied_rows": int(np.sum(pred >= 0)),
    }
    return pred, stats


def combine_view_predictions(view_preds: list[np.ndarray], mode: str) -> np.ndarray:
    if len(view_preds) == 1:
        return view_preds[0].copy()
    stacked = np.vstack(view_preds)
    out = np.full(stacked.shape[1], -1, dtype=np.int64)
    if mode == "single":
        # Use the first trusted view in the provided priority order.
        for row in stacked:
            fill = (out < 0) & (row >= 0)
            out[fill] = row[fill]
        return out
    valid = np.all(stacked >= 0, axis=0)
    agree = valid & np.all(stacked == stacked[0:1], axis=0)
    out[agree] = stacked[0, agree]
    return out


def validate_submission(submission: pd.DataFrame, sample: pd.DataFrame, classes: set[str]) -> dict[str, object]:
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
    specs = [parse_view_spec(text, args.min_count) for text in args.view_spec]
    train = pd.read_csv(RAW / "train_data.csv")
    test = pd.read_csv(RAW / "test_data.csv")
    sample = pd.read_csv(RAW / "sample_submission.csv")
    encoder = LabelEncoder()
    y = encoder.fit_transform(train["label"])
    classes = encoder.classes_.tolist()
    base_oof, base_test, base_report = load_report(resolve_path(args.base_report), classes)
    class_array = np.asarray(classes)
    base_oof_pred = base_oof.argmax(axis=1)
    base_test_pred = base_test.argmax(axis=1)
    base_oof_conf = base_oof.max(axis=1)
    base_test_conf = base_test.max(axis=1)

    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof_rule = np.full(len(train), -1, dtype=np.int64)
    fold_view_stats: list[dict[str, object]] = []
    for fold, (tr_idx, va_idx) in enumerate(cv.split(train, y), start=1):
        fold_preds = []
        for spec in specs:
            features = FEATURE_GROUPS[spec.name]
            pred, stats = cluster_rule(
                train.loc[tr_idx, features].to_numpy(dtype=np.float32),
                y[tr_idx],
                train.loc[va_idx, features].to_numpy(dtype=np.float32),
                spec,
                args.seed * 100 + fold,
                len(classes),
            )
            fold_preds.append(pred)
            stats.update({"fold": fold, "view": spec.name, "k": spec.k, "purity": spec.purity, "min_count": spec.min_count})
            fold_view_stats.append(stats)
        oof_rule[va_idx] = combine_view_predictions(fold_preds, args.mode)

    base_valid_pred = base_oof_pred.copy()
    rule_mask = (oof_rule >= 0) & (base_oof_conf <= args.base_max_conf)
    changed_mask = rule_mask & (oof_rule != base_valid_pred)
    override_pred = base_valid_pred.copy()
    override_pred[rule_mask] = oof_rule[rule_mask]
    validation = {
        "base_oof_macro_f1": float(f1_score(y, base_valid_pred, average="macro")),
        "override_oof_macro_f1": float(f1_score(y, override_pred, average="macro")),
        "rule_coverage": int(rule_mask.sum()),
        "rule_accuracy": float(np.mean(oof_rule[rule_mask] == y[rule_mask])) if np.any(rule_mask) else None,
        "changed_oof_rows": int(changed_mask.sum()),
        "changed_rule_accuracy": float(np.mean(oof_rule[changed_mask] == y[changed_mask])) if np.any(changed_mask) else None,
        "changed_base_accuracy": float(np.mean(base_valid_pred[changed_mask] == y[changed_mask])) if np.any(changed_mask) else None,
        "mode": args.mode,
        "base_max_conf": args.base_max_conf,
        "view_specs": [spec.__dict__ for spec in specs],
        "fold_view_stats": fold_view_stats,
    }

    full_view_preds = []
    full_view_stats: list[dict[str, object]] = []
    for spec in specs:
        features = FEATURE_GROUPS[spec.name]
        pred, stats = cluster_rule(
            train[features].to_numpy(dtype=np.float32),
            y,
            test[features].to_numpy(dtype=np.float32),
            spec,
            args.seed,
            len(classes),
        )
        full_view_preds.append(pred)
        stats.update({"view": spec.name, "k": spec.k, "purity": spec.purity, "min_count": spec.min_count})
        full_view_stats.append(stats)
    test_rule = combine_view_predictions(full_view_preds, args.mode)
    test_rule_mask = (test_rule >= 0) & (base_test_conf <= args.base_max_conf)
    changed_test_mask = test_rule_mask & (test_rule != base_test_pred)
    test_pred = base_test_pred.copy()
    test_pred[test_rule_mask] = test_rule[test_rule_mask]

    report = {
        "base_report": str(resolve_path(args.base_report)),
        "base_output_path": base_report.get("output_path"),
        "validation": validation,
        "test_rule_coverage": int(test_rule_mask.sum()),
        "changed_test_rows": int(changed_test_mask.sum()),
        "full_view_stats": full_view_stats,
        "changed_test_label_flow": pd.crosstab(
            class_array[base_test_pred[changed_test_mask]],
            class_array[test_rule[changed_test_mask]],
            rownames=["base"],
            colnames=["override"],
        ).to_dict(),
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.output_dir and args.output_name:
        output_dir = ROOT / args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        SUBMISSIONS.mkdir(parents=True, exist_ok=True)
        submission = pd.DataFrame({"id": test["id"], "label": class_array[test_pred]})
        output_path = output_dir / f"{args.output_name}.csv"
        mirror_path = SUBMISSIONS / f"{args.output_name}.csv"
        submission.to_csv(output_path, index=False, encoding="utf-8", lineterminator="\r\n")
        submission.to_csv(mirror_path, index=False, encoding="utf-8", lineterminator="\r\n")
        report.update(
            {
                "output_path": str(output_path),
                "mirror_path": str(mirror_path),
                "submission_validation": validate_submission(submission, sample, set(classes)),
            }
        )
        output_path.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        mirror_path.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"output_path": str(output_path), "submission_validation": report["submission_validation"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
