from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw" / "data"
SUBMISSIONS = ROOT / "submissions"


def load_pairwise_module():
    path = ROOT / "src" / "pairwise_gate_submission.py"
    spec = importlib.util.spec_from_file_location("pairwise_gate_local", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load pairwise_gate_submission.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Coordinate search per-pair gate thresholds.")
    parser.add_argument("--base-report", default="upload_ready_breakthrough5/c10_c04_patternshuffle_s43_25.json")
    parser.add_argument("--cluster-features", choices=["template", "top10"], default="template")
    parser.add_argument("--cluster-k", type=int, default=768)
    parser.add_argument("--cluster-alpha", type=float, default=1.2)
    parser.add_argument("--cluster-min-count", type=int, default=25)
    parser.add_argument("--cluster-smooth", type=float, default=15.0)
    parser.add_argument("--cluster-ratio-clip", type=float, default=1.8)
    parser.add_argument("--pairwise-features", choices=["template_top10", "drop_pattern"], default="template_top10")
    parser.add_argument("--initial-threshold", type=float, default=0.5)
    parser.add_argument("--threshold-grid", nargs="+", default=["off", "0.45", "0.50", "0.55", "0.60", "0.65", "0.70"])
    parser.add_argument("--max-iter", type=int, default=4)
    parser.add_argument("--pairs", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output-name", default=None)
    return parser.parse_args()


def parse_threshold_grid(values: list[str]) -> list[float | None]:
    grid: list[float | None] = []
    for value in values:
        if value.lower() in {"off", "none"}:
            grid.append(None)
        else:
            grid.append(float(value))
    return grid


def apply_gate(
    pmod,
    base_proba: np.ndarray,
    adjusted_proba: np.ndarray,
    pair_proba: dict[tuple[int, int], np.ndarray],
    ordered_pairs: list[tuple[int, int]],
    thresholds: dict[tuple[int, int], float | None],
) -> tuple[np.ndarray, np.ndarray, dict[tuple[int, int], np.ndarray]]:
    base_pred = base_proba.argmax(axis=1)
    adjusted_pred = adjusted_proba.argmax(axis=1)
    out = base_proba.copy()
    keep = np.zeros(len(base_pred), dtype=bool)
    pair_masks: dict[tuple[int, int], np.ndarray] = {}
    for base_cls, adjusted_cls in ordered_pairs:
        threshold = thresholds[(base_cls, adjusted_cls)]
        if threshold is None:
            mask = np.zeros(len(base_pred), dtype=bool)
        else:
            key = tuple(sorted((base_cls, adjusted_cls)))
            prefers_adjusted = pmod.pair_prefers(pair_proba[key], key, adjusted_cls, threshold)
            mask = (base_pred == base_cls) & (adjusted_pred == adjusted_cls) & prefers_adjusted
        pair_masks[(base_cls, adjusted_cls)] = mask
        keep |= mask
    out[keep] = adjusted_proba[keep]
    return out, keep, pair_masks


def score_thresholds(
    pmod,
    y: np.ndarray,
    base_oof: np.ndarray,
    adjusted_oof: np.ndarray,
    pair_oof: dict[tuple[int, int], np.ndarray],
    ordered_pairs: list[tuple[int, int]],
    thresholds: dict[tuple[int, int], float | None],
) -> float:
    out, _, _ = apply_gate(pmod, base_oof, adjusted_oof, pair_oof, ordered_pairs, thresholds)
    return float(f1_score(y, out.argmax(axis=1), average="macro"))


def threshold_text(value: float | None) -> str:
    return "off" if value is None else f"{value:.2f}"


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

    base_oof, base_test, _ = cmod.load_report(cmod.resolve_path(args.base_report), classes)
    ordered_pairs, unordered_pairs = pmod.parse_pairs(args.pairs, {label: i for i, label in enumerate(classes)})
    adjusted_oof, adjusted_test = pmod.build_cluster_adjusted(cmod, train, test, y, base_oof, base_test, args)
    x_pair = train[pmod.PAIRWISE_FEATURES[args.pairwise_features]].to_numpy(dtype=np.float32)
    x_test_pair = test[pmod.PAIRWISE_FEATURES[args.pairwise_features]].to_numpy(dtype=np.float32)
    pair_oof, pair_test = pmod.train_pair_models(x_pair, x_test_pair, y, unordered_pairs, args.seed)

    grid = parse_threshold_grid(args.threshold_grid)
    thresholds = {pair: args.initial_threshold for pair in ordered_pairs}
    best_score = score_thresholds(pmod, y, base_oof, adjusted_oof, pair_oof, ordered_pairs, thresholds)
    history = [{"iteration": 0, "score": best_score, "thresholds": {str(pair): threshold_text(value) for pair, value in thresholds.items()}}]
    for iteration in range(1, args.max_iter + 1):
        improved = False
        for pair in ordered_pairs:
            current_value = thresholds[pair]
            pair_best_value = current_value
            pair_best_score = best_score
            for candidate in grid:
                thresholds[pair] = candidate
                score = score_thresholds(pmod, y, base_oof, adjusted_oof, pair_oof, ordered_pairs, thresholds)
                if score > pair_best_score + 1e-12:
                    pair_best_score = score
                    pair_best_value = candidate
            thresholds[pair] = pair_best_value
            if pair_best_score > best_score + 1e-12:
                improved = True
                best_score = pair_best_score
        history.append({"iteration": iteration, "score": best_score, "thresholds": {str(pair): threshold_text(value) for pair, value in thresholds.items()}})
        if not improved:
            break

    gated_oof, oof_keep, oof_pair_masks = apply_gate(pmod, base_oof, adjusted_oof, pair_oof, ordered_pairs, thresholds)
    gated_test, test_keep, test_pair_masks = apply_gate(pmod, base_test, adjusted_test, pair_test, ordered_pairs, thresholds)
    base_pred = base_oof.argmax(axis=1)
    gated_pred = gated_oof.argmax(axis=1)
    pair_details = []
    for base_cls, adjusted_cls in ordered_pairs:
        mask = oof_pair_masks[(base_cls, adjusted_cls)]
        test_mask = test_pair_masks[(base_cls, adjusted_cls)]
        pair_details.append(
            {
                "pair": f"{classes[base_cls]}:{classes[adjusted_cls]}",
                "threshold": threshold_text(thresholds[(base_cls, adjusted_cls)]),
                "changed_oof_rows": int(mask.sum()),
                "changed_test_rows": int(test_mask.sum()),
                "new_acc": float(np.mean(gated_pred[mask] == y[mask])) if np.any(mask) else None,
                "base_acc": float(np.mean(base_pred[mask] == y[mask])) if np.any(mask) else None,
            }
        )

    report = {
        "base_report": str(cmod.resolve_path(args.base_report)),
        "cluster_features": args.cluster_features,
        "cluster_k": args.cluster_k,
        "cluster_alpha": args.cluster_alpha,
        "cluster_min_count": args.cluster_min_count,
        "cluster_smooth": args.cluster_smooth,
        "cluster_ratio_clip": args.cluster_ratio_clip,
        "pairwise_features": args.pairwise_features,
        "threshold_grid": [threshold_text(value) for value in grid],
        "history": history,
        "thresholds": {f"{classes[a]}:{classes[b]}": threshold_text(value) for (a, b), value in thresholds.items()},
        "base_oof_macro_f1": float(f1_score(y, base_pred, average="macro")),
        "gated_oof_macro_f1": float(f1_score(y, gated_pred, average="macro")),
        "changed_oof_rows": int(oof_keep.sum()),
        "changed_test_rows": int(test_keep.sum()),
        "changed_oof_new_acc": float(np.mean(gated_pred[oof_keep] == y[oof_keep])) if np.any(oof_keep) else None,
        "changed_oof_base_acc": float(np.mean(base_pred[oof_keep] == y[oof_keep])) if np.any(oof_keep) else None,
        "pair_details": pair_details,
        "test_label_counts": pd.Series(class_array[gated_test.argmax(axis=1)]).value_counts().sort_index().to_dict(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.output_dir and args.output_name:
        output_dir = ROOT / args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        SUBMISSIONS.mkdir(parents=True, exist_ok=True)
        submission = pd.DataFrame({"id": test["id"], "label": class_array[gated_test.argmax(axis=1)]})
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
