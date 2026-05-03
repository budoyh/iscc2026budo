from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw" / "data"
SUBMISSIONS = ROOT / "submissions"

PAIRWISE_FEATURES = {
    "template_top10": [
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
    "drop_pattern": [
        "behavior_template_time",
        "behavior_template_volume",
        "behavior_template_activity",
        "behavior_template_control",
        "behavior_compactness_score",
        "protocol_mix_entropy",
        "traffic_rate_mean",
        "volume_rate_mean",
        "payload_unit_mean",
        "traffic_rate_upper_q90",
        "traffic_rate_dispersion",
        "volume_rate_dispersion",
        "payload_unit_dispersion",
        "direction_gap_dispersion",
        "direction_rate_gap",
        "control_signal_intensity",
        "protocol_variation_level",
        "early_rate_mean",
        "mid_rate_mean",
        "late_rate_mean",
        "early_volume_mean",
        "mid_volume_mean",
        "late_volume_mean",
        "early_payload_mean",
        "mid_payload_mean",
        "late_payload_mean",
        "rate_shift_score",
        "volume_shift_score",
        "payload_shift_score",
        "traffic_rate_spike_ratio",
        "volume_rate_spike_ratio",
        "traffic_peak_position",
        "volume_peak_position",
        "traffic_jump_mean",
        "traffic_jump_max",
        "high_rate_ratio",
        "high_volume_ratio",
        "sparse_payload_ratio",
        "traffic_irregularity",
        "volume_irregularity",
        "payload_irregularity",
        "traffic_trend_score",
        "volume_trend_score",
        "payload_trend_score",
    ],
}


def load_cluster_module():
    path = ROOT / "src" / "cluster_prior_adjust.py"
    spec = importlib.util.spec_from_file_location("cluster_prior_adjust_local", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load cluster_prior_adjust.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gate cluster-prior label changes with local pairwise classifiers.")
    parser.add_argument("--base-report", default="upload_ready_breakthrough5/c10_c04_patternshuffle_s43_25.json")
    parser.add_argument("--cluster-features", choices=["template", "top10"], default="template")
    parser.add_argument("--cluster-k", type=int, default=768)
    parser.add_argument("--cluster-alpha", type=float, default=1.2)
    parser.add_argument("--cluster-min-count", type=int, default=25)
    parser.add_argument("--cluster-smooth", type=float, default=15.0)
    parser.add_argument("--cluster-ratio-clip", type=float, default=1.8)
    parser.add_argument("--pairwise-features", choices=sorted(PAIRWISE_FEATURES), default="template_top10")
    parser.add_argument("--pair-threshold", type=float, default=0.5)
    parser.add_argument("--pairs", nargs="+", required=True)
    parser.add_argument("--post-cluster-features", choices=["none", "template", "top10", "pattern6"], default="none")
    parser.add_argument("--post-cluster-k", type=int, default=512)
    parser.add_argument("--post-cluster-alpha", type=float, default=0.2)
    parser.add_argument("--post-cluster-min-count", type=int, default=25)
    parser.add_argument("--post-cluster-smooth", type=float, default=20.0)
    parser.add_argument("--post-cluster-ratio-clip", type=float, default=1.8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output-name", default=None)
    return parser.parse_args()


def parse_pairs(pair_texts: list[str], label_to_idx: dict[str, int]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    ordered: list[tuple[int, int]] = []
    unordered_set: set[tuple[int, int]] = set()
    for text in pair_texts:
        if ">" in text:
            left, right = text.split(">", 1)
        elif ":" in text:
            left, right = text.split(":", 1)
        else:
            raise ValueError(f"pair must be base>adjusted or base:adjusted, got {text}")
        base_idx = label_to_idx[left]
        adjusted_idx = label_to_idx[right]
        ordered.append((base_idx, adjusted_idx))
        unordered_set.add(tuple(sorted((base_idx, adjusted_idx))))
    return ordered, sorted(unordered_set)


def pair_prefers(proba_for_high: np.ndarray, pair_key: tuple[int, int], cls: int, threshold: float) -> np.ndarray:
    low, high = pair_key
    if cls == low:
        return proba_for_high <= (1.0 - threshold)
    if cls == high:
        return proba_for_high >= threshold
    raise ValueError(f"class {cls} not in pair {pair_key}")


def train_pair_models(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y: np.ndarray,
    unordered_pairs: list[tuple[int, int]],
    seed: int,
) -> tuple[dict[tuple[int, int], np.ndarray], dict[tuple[int, int], np.ndarray]]:
    pair_oof: dict[tuple[int, int], np.ndarray] = {}
    pair_test: dict[tuple[int, int], np.ndarray] = {}
    for pair_index, (low, high) in enumerate(unordered_pairs):
        mask = (y == low) | (y == high)
        indices = np.flatnonzero(mask)
        yy = (y[indices] == high).astype(np.int64)
        xx = x_train[indices]
        oof = np.zeros(len(y), dtype=np.float64)
        test_proba = np.zeros(len(x_test), dtype=np.float64)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + pair_index * 37)
        for fold, (tr_idx, va_idx) in enumerate(cv.split(xx, yy), start=1):
            model = LGBMClassifier(
                objective="binary",
                n_estimators=500,
                learning_rate=0.04,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.88,
                colsample_bytree=0.88,
                reg_lambda=1.2,
                random_state=seed + pair_index * 101 + fold,
                n_jobs=-1,
                verbosity=-1,
            )
            model.fit(xx[tr_idx], yy[tr_idx])
            oof[indices[va_idx]] = model.predict_proba(xx[va_idx])[:, 1]
            test_proba += model.predict_proba(x_test)[:, 1] / 5.0
        pair_oof[(low, high)] = oof
        pair_test[(low, high)] = test_proba
    return pair_oof, pair_test


def build_cluster_adjusted(
    mod,
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    base_oof: np.ndarray,
    base_test: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    cluster_features = mod.FEATURE_GROUPS[args.cluster_features]
    x_cluster = train[cluster_features].to_numpy(dtype=np.float32)
    x_test_cluster = test[cluster_features].to_numpy(dtype=np.float32)
    adjusted_oof = np.zeros_like(base_oof)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    for fold, (tr_idx, va_idx) in enumerate(cv.split(x_cluster, y), start=1):
        state = mod.fit_transductive_clusters(
            x_cluster[tr_idx],
            x_cluster[va_idx],
            args.cluster_k,
            args.cluster_min_count,
            args.seed * 100 + fold,
        )
        adjusted_oof[va_idx] = mod.adjust_proba(
            base_oof[va_idx],
            state,
            y[tr_idx],
            base_oof[tr_idx],
            args.cluster_alpha,
            args.cluster_smooth,
            args.cluster_ratio_clip,
            base_oof.shape[1],
        )
    state = mod.fit_transductive_clusters(
        x_cluster,
        x_test_cluster,
        args.cluster_k,
        args.cluster_min_count,
        args.seed,
    )
    adjusted_test = mod.adjust_proba(
        base_test,
        state,
        y,
        base_oof,
        args.cluster_alpha,
        args.cluster_smooth,
        args.cluster_ratio_clip,
        base_oof.shape[1],
    )
    return adjusted_oof, adjusted_test


def apply_cluster_prior_to_probs(
    mod,
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    oof: np.ndarray,
    test_proba: np.ndarray,
    features_name: str,
    k: int,
    alpha: float,
    min_count: int,
    smooth: float,
    ratio_clip: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    features = mod.FEATURE_GROUPS[features_name]
    x_train = train[features].to_numpy(dtype=np.float32)
    x_test = test[features].to_numpy(dtype=np.float32)
    adjusted_oof = np.zeros_like(oof)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (tr_idx, va_idx) in enumerate(cv.split(x_train, y), start=1):
        state = mod.fit_transductive_clusters(
            x_train[tr_idx],
            x_train[va_idx],
            k,
            min_count,
            seed * 100 + fold,
        )
        adjusted_oof[va_idx] = mod.adjust_proba(
            oof[va_idx],
            state,
            y[tr_idx],
            oof[tr_idx],
            alpha,
            smooth,
            ratio_clip,
            oof.shape[1],
        )
    state = mod.fit_transductive_clusters(x_train, x_test, k, min_count, seed)
    adjusted_test = mod.adjust_proba(
        test_proba,
        state,
        y,
        oof,
        alpha,
        smooth,
        ratio_clip,
        oof.shape[1],
    )
    return adjusted_oof, adjusted_test


def apply_pair_gate(
    base_proba: np.ndarray,
    adjusted_proba: np.ndarray,
    pair_proba: dict[tuple[int, int], np.ndarray],
    ordered_pairs: list[tuple[int, int]],
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    base_pred = base_proba.argmax(axis=1)
    adjusted_pred = adjusted_proba.argmax(axis=1)
    out = base_proba.copy()
    keep = np.zeros(len(base_pred), dtype=bool)
    for base_cls, adjusted_cls in ordered_pairs:
        key = tuple(sorted((base_cls, adjusted_cls)))
        prefers_adjusted = pair_prefers(pair_proba[key], key, adjusted_cls, threshold)
        mask = (base_pred == base_cls) & (adjusted_pred == adjusted_cls) & prefers_adjusted
        keep |= mask
    out[keep] = adjusted_proba[keep]
    return out, keep


def main() -> None:
    args = parse_args()
    mod = load_cluster_module()
    train = pd.read_csv(RAW / "train_data.csv")
    test = pd.read_csv(RAW / "test_data.csv")
    sample = pd.read_csv(RAW / "sample_submission.csv")
    encoder = LabelEncoder()
    y = encoder.fit_transform(train["label"])
    classes = encoder.classes_.tolist()
    class_array = np.asarray(classes)
    base_oof, base_test, _ = mod.load_report(mod.resolve_path(args.base_report), classes)
    ordered_pairs, unordered_pairs = parse_pairs(args.pairs, {label: i for i, label in enumerate(classes)})
    adjusted_oof, adjusted_test = build_cluster_adjusted(mod, train, test, y, base_oof, base_test, args)

    pair_features = PAIRWISE_FEATURES[args.pairwise_features]
    x_pair = train[pair_features].to_numpy(dtype=np.float32)
    x_test_pair = test[pair_features].to_numpy(dtype=np.float32)
    pair_oof, pair_test = train_pair_models(x_pair, x_test_pair, y, unordered_pairs, args.seed)
    gated_oof, oof_keep = apply_pair_gate(base_oof, adjusted_oof, pair_oof, ordered_pairs, args.pair_threshold)
    gated_test, test_keep = apply_pair_gate(base_test, adjusted_test, pair_test, ordered_pairs, args.pair_threshold)
    final_oof = gated_oof
    final_test = gated_test
    post_report = None
    if args.post_cluster_features != "none":
        post_oof, post_test = apply_cluster_prior_to_probs(
            mod,
            train,
            test,
            y,
            gated_oof,
            gated_test,
            args.post_cluster_features,
            args.post_cluster_k,
            args.post_cluster_alpha,
            args.post_cluster_min_count,
            args.post_cluster_smooth,
            args.post_cluster_ratio_clip,
            args.seed + 777,
        )
        gated_pred_for_post = gated_oof.argmax(axis=1)
        post_pred = post_oof.argmax(axis=1)
        post_changed = post_pred != gated_pred_for_post
        gated_test_pred_for_post = gated_test.argmax(axis=1)
        post_test_pred = post_test.argmax(axis=1)
        post_test_changed = post_test_pred != gated_test_pred_for_post
        post_report = {
            "post_cluster_features": args.post_cluster_features,
            "post_cluster_k": args.post_cluster_k,
            "post_cluster_alpha": args.post_cluster_alpha,
            "post_cluster_min_count": args.post_cluster_min_count,
            "post_cluster_smooth": args.post_cluster_smooth,
            "post_cluster_ratio_clip": args.post_cluster_ratio_clip,
            "post_oof_macro_f1": float(f1_score(y, post_pred, average="macro")),
            "post_changed_oof_rows": int(post_changed.sum()),
            "post_changed_test_rows": int(post_test_changed.sum()),
            "post_changed_oof_new_acc": float(np.mean(post_pred[post_changed] == y[post_changed])) if np.any(post_changed) else None,
            "post_changed_oof_before_acc": float(np.mean(gated_pred_for_post[post_changed] == y[post_changed])) if np.any(post_changed) else None,
        }
        final_oof = post_oof
        final_test = post_test

    base_pred = base_oof.argmax(axis=1)
    gated_pred = gated_oof.argmax(axis=1)
    final_pred = final_oof.argmax(axis=1)
    report = {
        "base_report": str(mod.resolve_path(args.base_report)),
        "cluster_features": args.cluster_features,
        "cluster_k": args.cluster_k,
        "cluster_alpha": args.cluster_alpha,
        "cluster_min_count": args.cluster_min_count,
        "cluster_smooth": args.cluster_smooth,
        "cluster_ratio_clip": args.cluster_ratio_clip,
        "pairwise_features": args.pairwise_features,
        "pair_threshold": args.pair_threshold,
        "pairs": args.pairs,
        "base_oof_macro_f1": float(f1_score(y, base_pred, average="macro")),
        "gated_oof_macro_f1": float(f1_score(y, gated_pred, average="macro")),
        "final_oof_macro_f1": float(f1_score(y, final_pred, average="macro")),
        "changed_oof_rows": int(oof_keep.sum()),
        "changed_test_rows": int(test_keep.sum()),
        "changed_oof_new_acc": float(np.mean(gated_pred[oof_keep] == y[oof_keep])) if np.any(oof_keep) else None,
        "changed_oof_base_acc": float(np.mean(base_pred[oof_keep] == y[oof_keep])) if np.any(oof_keep) else None,
        "post_report": post_report,
        "final_changed_oof_rows_vs_base": int(np.sum(final_pred != base_pred)),
        "final_changed_test_rows_vs_base": int(np.sum(final_test.argmax(axis=1) != base_test.argmax(axis=1))),
        "test_label_counts": pd.Series(class_array[final_test.argmax(axis=1)]).value_counts().sort_index().to_dict(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.output_dir and args.output_name:
        output_dir = ROOT / args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        SUBMISSIONS.mkdir(parents=True, exist_ok=True)
        submission = pd.DataFrame({"id": test["id"], "label": class_array[final_test.argmax(axis=1)]})
        output_path = output_dir / f"{args.output_name}.csv"
        mirror_path = SUBMISSIONS / f"{args.output_name}.csv"
        submission.to_csv(output_path, index=False, encoding="utf-8", lineterminator="\r\n")
        submission.to_csv(mirror_path, index=False, encoding="utf-8", lineterminator="\r\n")
        report.update(
            {
                "output_path": str(output_path),
                "mirror_path": str(mirror_path),
                "submission_validation": mod.validate_submission(submission, sample, set(classes)),
            }
        )
        output_path.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        mirror_path.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"output_path": str(output_path), "submission_validation": report["submission_validation"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
