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


FEATURE_SETS = {
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
FEATURE_SETS["triad_combo"] = list(dict.fromkeys(FEATURE_SETS["template_top10"] + FEATURE_SETS["drop_pattern"]))


def load_pairwise_module():
    path = ROOT / "src" / "pairwise_gate_submission.py"
    spec = importlib.util.spec_from_file_location("pairwise_gate_local", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load pairwise_gate_submission.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local tri-class specialist proposal.")
    parser.add_argument("--base-report", default="upload_ready_breakthrough5/c10_c04_patternshuffle_s43_25.json")
    parser.add_argument("--cluster-features", choices=["template", "top10"], default="template")
    parser.add_argument("--cluster-k", type=int, default=768)
    parser.add_argument("--cluster-alpha", type=float, default=1.2)
    parser.add_argument("--cluster-min-count", type=int, default=25)
    parser.add_argument("--cluster-smooth", type=float, default=15.0)
    parser.add_argument("--cluster-ratio-clip", type=float, default=1.8)
    parser.add_argument("--gate-pairwise-features", choices=["template_top10", "drop_pattern"], default="template_top10")
    parser.add_argument("--gate-pairs", nargs="+", required=True)
    parser.add_argument("--gate-thresholds", nargs="+", required=True)
    parser.add_argument("--pre-source")
    parser.add_argument("--pre-target")
    parser.add_argument("--pre-threshold", type=float, default=None)
    parser.add_argument("--triad", nargs=3, default=["class_0", "class_10", "class_11"])
    parser.add_argument("--features", choices=sorted(FEATURE_SETS), default="triad_combo")
    parser.add_argument("--threshold-grid", nargs="+", type=float, default=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8])
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--margin-grid", nargs="+", type=float, default=[0.0])
    parser.add_argument("--margin", type=float, default=None)
    parser.add_argument("--flow-whitelist", nargs="*", default=[])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output-name", default=None)
    return parser.parse_args()


def parse_gate_thresholds(
    pair_texts: list[str],
    threshold_texts: list[str],
    label_to_idx: dict[str, int],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], dict[tuple[int, int], float | None]]:
    if len(pair_texts) != len(threshold_texts):
        raise ValueError("--gate-pairs and --gate-thresholds must have the same length")
    ordered: list[tuple[int, int]] = []
    unordered_set: set[tuple[int, int]] = set()
    thresholds: dict[tuple[int, int], float | None] = {}
    for pair_text, threshold_text in zip(pair_texts, threshold_texts):
        left, right = pair_text.replace(">", ":").split(":", 1)
        pair = (label_to_idx[left], label_to_idx[right])
        ordered.append(pair)
        unordered_set.add(tuple(sorted(pair)))
        thresholds[pair] = None if threshold_text.lower() in {"off", "none"} else float(threshold_text)
    return ordered, sorted(unordered_set), thresholds


def apply_gate_thresholds(
    pmod,
    base_proba: np.ndarray,
    adjusted_proba: np.ndarray,
    pair_proba: dict[tuple[int, int], np.ndarray],
    ordered_pairs: list[tuple[int, int]],
    thresholds: dict[tuple[int, int], float | None],
) -> np.ndarray:
    base_pred = base_proba.argmax(axis=1)
    adjusted_pred = adjusted_proba.argmax(axis=1)
    out = base_proba.copy()
    keep = np.zeros(len(base_pred), dtype=bool)
    for base_cls, adjusted_cls in ordered_pairs:
        threshold = thresholds[(base_cls, adjusted_cls)]
        if threshold is None:
            continue
        key = tuple(sorted((base_cls, adjusted_cls)))
        prefers_adjusted = pmod.pair_prefers(pair_proba[key], key, adjusted_cls, threshold)
        keep |= (base_pred == base_cls) & (adjusted_pred == adjusted_cls) & prefers_adjusted
    out[keep] = adjusted_proba[keep]
    return out


def train_binary_pair_full_oof(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y: np.ndarray,
    source_idx: int,
    target_idx: int,
    seed: int,
    pmod,
) -> tuple[np.ndarray, np.ndarray]:
    pair_key = tuple(sorted((source_idx, target_idx)))
    high = pair_key[1]
    oof = np.zeros(len(y), dtype=np.float64)
    test_proba = np.zeros(len(x_test), dtype=np.float64)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (tr_idx, va_idx) in enumerate(cv.split(x_train, y), start=1):
        pair_train = tr_idx[(y[tr_idx] == source_idx) | (y[tr_idx] == target_idx)]
        yy = (y[pair_train] == high).astype(np.int64)
        model = LGBMClassifier(
            objective="binary",
            n_estimators=700,
            learning_rate=0.035,
            num_leaves=31,
            min_child_samples=18,
            subsample=0.88,
            colsample_bytree=0.88,
            reg_lambda=1.2,
            random_state=seed + fold * 101,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(x_train[pair_train], yy)
        oof[va_idx] = model.predict_proba(x_train[va_idx])[:, 1]
        test_proba += model.predict_proba(x_test)[:, 1] / 5.0
    return oof, test_proba


def apply_binary_preproposal(
    pmod,
    current_pred: np.ndarray,
    pair_proba: np.ndarray,
    source_idx: int,
    target_idx: int,
    threshold: float,
) -> np.ndarray:
    key = tuple(sorted((source_idx, target_idx)))
    candidate = (current_pred == source_idx) & pmod.pair_prefers(pair_proba, key, target_idx, threshold)
    out = current_pred.copy()
    out[candidate] = target_idx
    return out


def train_triad_full_oof(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y: np.ndarray,
    triad_indices: list[int],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    triad_set = set(triad_indices)
    triad_to_local = {cls: idx for idx, cls in enumerate(triad_indices)}
    oof = np.zeros((len(y), len(triad_indices)), dtype=np.float64)
    test_proba = np.zeros((len(x_test), len(triad_indices)), dtype=np.float64)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (tr_idx, va_idx) in enumerate(cv.split(x_train, y), start=1):
        local_train = np.asarray([idx for idx in tr_idx if y[idx] in triad_set], dtype=np.int64)
        yy = np.asarray([triad_to_local[int(cls)] for cls in y[local_train]], dtype=np.int64)
        model = LGBMClassifier(
            objective="multiclass",
            num_class=len(triad_indices),
            n_estimators=850,
            learning_rate=0.03,
            num_leaves=39,
            min_child_samples=18,
            subsample=0.88,
            colsample_bytree=0.9,
            reg_lambda=1.4,
            random_state=seed + fold * 211,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(x_train[local_train], yy)
        oof[va_idx] = model.predict_proba(x_train[va_idx])
        test_proba += model.predict_proba(x_test) / 5.0
    return oof, test_proba


def apply_triad(
    current_pred: np.ndarray,
    triad_proba: np.ndarray,
    triad_indices: list[int],
    threshold: float,
    margin: float,
    flow_whitelist: set[tuple[int, int]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    triad_array = np.asarray(triad_indices, dtype=np.int64)
    triad_pred = triad_array[triad_proba.argmax(axis=1)]
    sorted_proba = np.sort(triad_proba, axis=1)
    conf = sorted_proba[:, -1]
    top_margin = sorted_proba[:, -1] - sorted_proba[:, -2]
    in_triad = np.isin(current_pred, triad_array)
    changed = in_triad & (triad_pred != current_pred) & (conf >= threshold) & (top_margin >= margin)
    if flow_whitelist is not None:
        flow_keep = np.zeros(len(current_pred), dtype=bool)
        for src, dst in flow_whitelist:
            flow_keep |= (current_pred == src) & (triad_pred == dst)
        changed &= flow_keep
    out = current_pred.copy()
    out[changed] = triad_pred[changed]
    return out, changed


def main() -> None:
    args = parse_args()
    pmod = load_pairwise_module()
    cmod = pmod.load_cluster_module()
    train = pd.read_csv(RAW / "train_data.csv")
    test = pd.read_csv(RAW / "test_data.csv")
    sample = pd.read_csv(RAW / "sample_submission.csv")
    encoder = LabelEncoder()
    y = encoder.fit_transform(train["label"])
    classes = encoder.classes_.tolist()
    class_array = np.asarray(classes)
    label_to_idx = {label: idx for idx, label in enumerate(classes)}

    base_oof, base_test, _ = cmod.load_report(cmod.resolve_path(args.base_report), classes)
    ordered_pairs, unordered_pairs, gate_thresholds = parse_gate_thresholds(args.gate_pairs, args.gate_thresholds, label_to_idx)
    adjusted_oof, adjusted_test = pmod.build_cluster_adjusted(cmod, train, test, y, base_oof, base_test, args)
    gate_features = pmod.PAIRWISE_FEATURES[args.gate_pairwise_features]
    x_gate = train[gate_features].to_numpy(dtype=np.float32)
    x_test_gate = test[gate_features].to_numpy(dtype=np.float32)
    pair_oof, pair_test = pmod.train_pair_models(x_gate, x_test_gate, y, unordered_pairs, args.seed)
    current_oof = apply_gate_thresholds(pmod, base_oof, adjusted_oof, pair_oof, ordered_pairs, gate_thresholds)
    current_test = apply_gate_thresholds(pmod, base_test, adjusted_test, pair_test, ordered_pairs, gate_thresholds)
    current_pred = current_oof.argmax(axis=1)
    current_test_pred = current_test.argmax(axis=1)

    if args.pre_source and args.pre_target and args.pre_threshold is not None:
        source_idx = label_to_idx[args.pre_source]
        target_idx = label_to_idx[args.pre_target]
        pre_oof, pre_test = train_binary_pair_full_oof(
            x_gate,
            x_test_gate,
            y,
            source_idx,
            target_idx,
            args.seed + 4099,
            pmod,
        )
        current_pred = apply_binary_preproposal(pmod, current_pred, pre_oof, source_idx, target_idx, args.pre_threshold)
        current_test_pred = apply_binary_preproposal(pmod, current_test_pred, pre_test, source_idx, target_idx, args.pre_threshold)

    triad_indices = [label_to_idx[label] for label in args.triad]
    flow_whitelist = None
    if args.flow_whitelist:
        flow_whitelist = set()
        for text in args.flow_whitelist:
            left, right = text.replace(">", ":").split(":", 1)
            flow_whitelist.add((label_to_idx[left], label_to_idx[right]))
    feature_names = FEATURE_SETS[args.features]
    x_train = train[feature_names].to_numpy(dtype=np.float32)
    x_test = test[feature_names].to_numpy(dtype=np.float32)
    triad_oof, triad_test = train_triad_full_oof(x_train, x_test, y, triad_indices, args.seed + 707)

    rows = []
    threshold_grid = [args.threshold] if args.threshold is not None else args.threshold_grid
    margin_grid = [args.margin] if args.margin is not None else args.margin_grid
    for threshold in threshold_grid:
        for margin in margin_grid:
            pred, mask = apply_triad(current_pred, triad_oof, triad_indices, threshold, margin, flow_whitelist)
            test_pred, test_mask = apply_triad(current_test_pred, triad_test, triad_indices, threshold, margin, flow_whitelist)
            rows.append(
                {
                    "threshold": threshold,
                    "margin": margin,
                    "score": float(f1_score(y, pred, average="macro")),
                    "changed_oof_rows": int(mask.sum()),
                    "changed_test_rows": int(test_mask.sum()),
                    "new_acc": float(np.mean(pred[mask] == y[mask])) if np.any(mask) else None,
                    "before_acc": float(np.mean(current_pred[mask] == y[mask])) if np.any(mask) else None,
                }
            )
    best = max(rows, key=lambda row: row["score"])
    chosen_threshold = float(best["threshold"])
    chosen_margin = float(best["margin"])
    final_pred, final_mask = apply_triad(current_pred, triad_oof, triad_indices, chosen_threshold, chosen_margin, flow_whitelist)
    final_test_pred, final_test_mask = apply_triad(current_test_pred, triad_test, triad_indices, chosen_threshold, chosen_margin, flow_whitelist)

    pair_details = []
    for src in triad_indices:
        for dst in triad_indices:
            if src == dst:
                continue
            mask = final_mask & (current_pred == src) & (final_pred == dst)
            test_mask = final_test_mask & (current_test_pred == src) & (final_test_pred == dst)
            if np.any(mask) or np.any(test_mask):
                pair_details.append(
                    {
                        "flow": f"{classes[src]}:{classes[dst]}",
                        "changed_oof_rows": int(mask.sum()),
                        "changed_test_rows": int(test_mask.sum()),
                        "new_acc": float(np.mean(final_pred[mask] == y[mask])) if np.any(mask) else None,
                        "before_acc": float(np.mean(current_pred[mask] == y[mask])) if np.any(mask) else None,
                    }
                )

    report = {
        "base_report": str(cmod.resolve_path(args.base_report)),
        "features": args.features,
        "triad": args.triad,
        "flow_whitelist": args.flow_whitelist,
        "pre_source": args.pre_source,
        "pre_target": args.pre_target,
        "pre_threshold": args.pre_threshold,
        "grid": rows,
        "chosen_threshold": chosen_threshold,
        "chosen_margin": chosen_margin,
        "current_oof_macro_f1": float(f1_score(y, current_pred, average="macro")),
        "final_oof_macro_f1": float(f1_score(y, final_pred, average="macro")),
        "changed_oof_rows": int(final_mask.sum()),
        "changed_test_rows": int(final_test_mask.sum()),
        "changed_oof_new_acc": float(np.mean(final_pred[final_mask] == y[final_mask])) if np.any(final_mask) else None,
        "changed_oof_before_acc": float(np.mean(current_pred[final_mask] == y[final_mask])) if np.any(final_mask) else None,
        "flow_details": pair_details,
        "test_label_counts": pd.Series(class_array[final_test_pred]).value_counts().sort_index().to_dict(),
    }
    print(pd.DataFrame(rows).sort_values("score", ascending=False).head(30).round(6).to_string(index=False))
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.output_dir and args.output_name:
        output_dir = ROOT / args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        SUBMISSIONS.mkdir(parents=True, exist_ok=True)
        submission = pd.DataFrame({"id": test["id"], "label": class_array[final_test_pred]})
        output_path = output_dir / f"{args.output_name}.csv"
        mirror_path = SUBMISSIONS / f"{args.output_name}.csv"
        submission.to_csv(output_path, index=False, encoding="utf-8", lineterminator="\r\n")
        submission.to_csv(mirror_path, index=False, encoding="utf-8", lineterminator="\r\n")
        report.update(
            {
                "output_path": str(output_path),
                "mirror_path": str(mirror_path),
                "submission_validation": cmod.validate_submission(submission, sample, set(classes)),
            }
        )
        output_path.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        mirror_path.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"output_path": str(output_path), "submission_validation": report["submission_validation"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
