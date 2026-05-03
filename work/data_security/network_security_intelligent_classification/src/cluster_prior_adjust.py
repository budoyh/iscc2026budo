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
    "pattern6": [
        "pattern_transition_ratio",
        "pattern_diversity_ratio",
        "pattern_switch_frequency",
        "pattern_mix_density",
        "pattern_change_ratio",
        "pattern_concentration",
    ],
}


@dataclass
class ClusterState:
    apply_cluster: np.ndarray
    labeled_cluster: np.ndarray
    trusted_apply_mask: np.ndarray
    cluster_count: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster-conditional soft prior adjustment.")
    parser.add_argument("--base-report", default="upload_ready_breakthrough4/c04_c03_drop_pattern_50.json")
    parser.add_argument("--features", choices=sorted(FEATURE_GROUPS), default="template")
    parser.add_argument("--k", type=int, default=384)
    parser.add_argument("--alpha-grid", nargs="+", type=float, default=[0.15, 0.25, 0.35, 0.5])
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--min-count", type=int, default=25)
    parser.add_argument("--smooth", type=float, default=30.0)
    parser.add_argument("--ratio-clip", type=float, default=2.0)
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


def fit_transductive_clusters(
    x_labeled: np.ndarray,
    x_apply: np.ndarray,
    k: int,
    min_count: int,
    seed: int,
) -> ClusterState:
    x_all = np.vstack([x_labeled, x_apply]).astype(np.float32)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_all)
    model = MiniBatchKMeans(
        n_clusters=k,
        batch_size=4096,
        n_init=5,
        max_iter=250,
        random_state=seed,
        reassignment_ratio=0.0,
    )
    cluster_all = model.fit_predict(x_scaled)
    labeled_cluster = cluster_all[: len(x_labeled)]
    apply_cluster = cluster_all[len(x_labeled) :]
    cluster_count = np.bincount(labeled_cluster, minlength=k)
    trusted = cluster_count >= min_count
    return ClusterState(
        apply_cluster=apply_cluster,
        labeled_cluster=labeled_cluster,
        trusted_apply_mask=trusted[apply_cluster],
        cluster_count=cluster_count,
    )


def adjust_proba(
    proba_apply: np.ndarray,
    state: ClusterState,
    y_labeled: np.ndarray,
    proba_labeled: np.ndarray,
    alpha: float,
    smooth: float,
    ratio_clip: float,
    num_classes: int,
) -> np.ndarray:
    out = proba_apply.copy()
    global_true = np.bincount(y_labeled, minlength=num_classes).astype(np.float64)
    global_true /= global_true.sum()
    global_pred = proba_labeled.mean(axis=0)
    for cluster_id, count in enumerate(state.cluster_count):
        if count <= 0:
            continue
        apply_mask = state.apply_cluster == cluster_id
        if not np.any(apply_mask):
            continue
        labeled_mask = state.labeled_cluster == cluster_id
        if not np.any(labeled_mask):
            continue
        true_counts = np.bincount(y_labeled[labeled_mask], minlength=num_classes).astype(np.float64)
        true_prior = (true_counts + smooth * global_true) / (true_counts.sum() + smooth)
        pred_prior = (proba_labeled[labeled_mask].sum(axis=0) + smooth * global_pred) / (labeled_mask.sum() + smooth)
        ratio = np.clip(true_prior / np.clip(pred_prior, 1e-12, None), 1.0 / ratio_clip, ratio_clip) ** alpha
        out[apply_mask] *= ratio
        out[apply_mask] /= out[apply_mask].sum(axis=1, keepdims=True)
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
    train = pd.read_csv(RAW / "train_data.csv")
    test = pd.read_csv(RAW / "test_data.csv")
    sample = pd.read_csv(RAW / "sample_submission.csv")
    encoder = LabelEncoder()
    y = encoder.fit_transform(train["label"])
    classes = encoder.classes_.tolist()
    class_array = np.asarray(classes)
    base_oof, base_test, base_report = load_report(resolve_path(args.base_report), classes)
    features = FEATURE_GROUPS[args.features]
    x_train = train[features].to_numpy(dtype=np.float32)
    x_test = test[features].to_numpy(dtype=np.float32)

    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    adjusted_by_alpha = {alpha: np.zeros_like(base_oof) for alpha in args.alpha_grid}
    coverage = np.zeros(len(train), dtype=bool)
    for fold, (tr_idx, va_idx) in enumerate(cv.split(x_train, y), start=1):
        state = fit_transductive_clusters(
            x_train[tr_idx],
            x_train[va_idx],
            args.k,
            args.min_count,
            args.seed * 100 + fold,
        )
        coverage[va_idx] = state.trusted_apply_mask
        for alpha in args.alpha_grid:
            adjusted_by_alpha[alpha][va_idx] = adjust_proba(
                base_oof[va_idx],
                state,
                y[tr_idx],
                base_oof[tr_idx],
                alpha,
                args.smooth,
                args.ratio_clip,
                len(classes),
            )

    base_pred = base_oof.argmax(axis=1)
    rows = []
    for alpha, adjusted in adjusted_by_alpha.items():
        pred = adjusted.argmax(axis=1)
        rows.append(
            {
                "alpha": alpha,
                "base_oof_macro_f1": float(f1_score(y, base_pred, average="macro")),
                "adjusted_oof_macro_f1": float(f1_score(y, pred, average="macro")),
                "changed_oof_rows": int(np.sum(pred != base_pred)),
                "covered_oof_rows": int(coverage.sum()),
                "changed_accuracy": float(np.mean(pred[pred != base_pred] == y[pred != base_pred])) if np.any(pred != base_pred) else None,
                "base_changed_accuracy": float(np.mean(base_pred[pred != base_pred] == y[pred != base_pred])) if np.any(pred != base_pred) else None,
            }
        )
    print(pd.DataFrame(rows).sort_values("adjusted_oof_macro_f1", ascending=False).round(6).to_string(index=False))

    chosen_alpha = args.alpha
    if chosen_alpha is None and rows:
        chosen_alpha = max(rows, key=lambda row: row["adjusted_oof_macro_f1"])["alpha"]
    state = fit_transductive_clusters(x_train, x_test, args.k, args.min_count, args.seed)
    adjusted_test = adjust_proba(
        base_test,
        state,
        y,
        base_oof,
        float(chosen_alpha),
        args.smooth,
        args.ratio_clip,
        len(classes),
    )
    test_pred = adjusted_test.argmax(axis=1)
    base_test_pred = base_test.argmax(axis=1)
    report = {
        "base_report": str(resolve_path(args.base_report)),
        "base_output_path": base_report.get("output_path"),
        "features": args.features,
        "k": args.k,
        "min_count": args.min_count,
        "smooth": args.smooth,
        "ratio_clip": args.ratio_clip,
        "chosen_alpha": chosen_alpha,
        "grid": rows,
        "test_covered_rows": int(state.trusted_apply_mask.sum()),
        "changed_test_rows": int(np.sum(test_pred != base_test_pred)),
        "label_counts": pd.Series(class_array[test_pred]).value_counts().sort_index().to_dict(),
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
