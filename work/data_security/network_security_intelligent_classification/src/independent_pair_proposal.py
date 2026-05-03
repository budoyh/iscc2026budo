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


def load_pairwise_module():
    path = ROOT / "src" / "pairwise_gate_submission.py"
    spec = importlib.util.spec_from_file_location("pairwise_gate_local", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load pairwise_gate_submission.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add independent pairwise proposals beyond cluster-prior candidates.")
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
    parser.add_argument("--proposal-source", required=True)
    parser.add_argument("--proposal-target", required=True)
    parser.add_argument("--proposal-features", choices=["template_top10", "drop_pattern"], default="template_top10")
    parser.add_argument("--proposal-threshold-grid", nargs="+", type=float, default=[0.8, 0.85, 0.9, 0.95])
    parser.add_argument("--proposal-threshold", type=float, default=None)
    parser.add_argument("--source-current-only", action="store_true", help="Require the current gated prediction to be the proposal source.")
    parser.add_argument("--source-base-only", action="store_true", help="Require the original base prediction to be the proposal source.")
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
        if ":" in pair_text:
            left, right = pair_text.split(":", 1)
        elif ">" in pair_text:
            left, right = pair_text.split(">", 1)
        else:
            raise ValueError(f"bad pair {pair_text!r}")
        pair = (label_to_idx[left], label_to_idx[right])
        ordered.append(pair)
        unordered_set.add(tuple(sorted(pair)))
        thresholds[pair] = None if threshold_text.lower() in {"off", "none"} else float(threshold_text)
    return ordered, sorted(unordered_set), thresholds


def pair_prefers_target(pmod, proba_for_high: np.ndarray, source_idx: int, target_idx: int, threshold: float) -> np.ndarray:
    key = tuple(sorted((source_idx, target_idx)))
    return pmod.pair_prefers(proba_for_high, key, target_idx, threshold)


def train_binary_pair_full_oof(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y: np.ndarray,
    source_idx: int,
    target_idx: int,
    seed: int,
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


def apply_proposal_labels(
    current_pred: np.ndarray,
    base_pred: np.ndarray,
    pair_proba: np.ndarray,
    source_idx: int,
    target_idx: int,
    threshold: float,
    pmod,
    source_current_only: bool,
    source_base_only: bool,
) -> tuple[np.ndarray, np.ndarray]:
    candidate = pair_prefers_target(pmod, pair_proba, source_idx, target_idx, threshold)
    if source_current_only or not source_base_only:
        candidate &= current_pred == source_idx
    if source_base_only:
        candidate &= base_pred == source_idx
    out = current_pred.copy()
    out[candidate] = target_idx
    return out, candidate


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

    source_idx = label_to_idx[args.proposal_source]
    target_idx = label_to_idx[args.proposal_target]
    proposal_features = pmod.PAIRWISE_FEATURES[args.proposal_features]
    x_prop = train[proposal_features].to_numpy(dtype=np.float32)
    x_test_prop = test[proposal_features].to_numpy(dtype=np.float32)
    proposal_oof, proposal_test = train_binary_pair_full_oof(
        x_prop,
        x_test_prop,
        y,
        source_idx,
        target_idx,
        args.seed + 4099,
    )

    base_pred = base_oof.argmax(axis=1)
    current_pred = current_oof.argmax(axis=1)
    current_test_pred = current_test.argmax(axis=1)
    rows = []
    for threshold in args.proposal_threshold_grid:
        pred, mask = apply_proposal_labels(
            current_pred,
            base_pred,
            proposal_oof,
            source_idx,
            target_idx,
            threshold,
            pmod,
            args.source_current_only,
            args.source_base_only,
        )
        _, test_mask = apply_proposal_labels(
            current_test_pred,
            base_test.argmax(axis=1),
            proposal_test,
            source_idx,
            target_idx,
            threshold,
            pmod,
            args.source_current_only,
            args.source_base_only,
        )
        rows.append(
            {
                "threshold": threshold,
                "score": float(f1_score(y, pred, average="macro")),
                "changed_oof_rows": int(mask.sum()),
                "changed_test_rows": int(test_mask.sum()),
                "new_acc": float(np.mean(pred[mask] == y[mask])) if np.any(mask) else None,
                "before_acc": float(np.mean(current_pred[mask] == y[mask])) if np.any(mask) else None,
            }
        )
    chosen_threshold = args.proposal_threshold
    if chosen_threshold is None:
        chosen_threshold = max(rows, key=lambda row: row["score"])["threshold"]
    final_pred, final_mask = apply_proposal_labels(
        current_pred,
        base_pred,
        proposal_oof,
        source_idx,
        target_idx,
        chosen_threshold,
        pmod,
        args.source_current_only,
        args.source_base_only,
    )
    final_test_pred, final_test_mask = apply_proposal_labels(
        current_test_pred,
        base_test.argmax(axis=1),
        proposal_test,
        source_idx,
        target_idx,
        chosen_threshold,
        pmod,
        args.source_current_only,
        args.source_base_only,
    )
    report = {
        "base_report": str(cmod.resolve_path(args.base_report)),
        "gate_pairs": args.gate_pairs,
        "gate_thresholds": args.gate_thresholds,
        "proposal_source": args.proposal_source,
        "proposal_target": args.proposal_target,
        "proposal_features": args.proposal_features,
        "source_current_only": args.source_current_only,
        "source_base_only": args.source_base_only,
        "grid": rows,
        "chosen_threshold": chosen_threshold,
        "base_oof_macro_f1": float(f1_score(y, base_pred, average="macro")),
        "current_oof_macro_f1": float(f1_score(y, current_pred, average="macro")),
        "final_oof_macro_f1": float(f1_score(y, final_pred, average="macro")),
        "proposal_changed_oof_rows": int(final_mask.sum()),
        "proposal_changed_test_rows": int(final_test_mask.sum()),
        "proposal_new_acc": float(np.mean(final_pred[final_mask] == y[final_mask])) if np.any(final_mask) else None,
        "proposal_before_acc": float(np.mean(current_pred[final_mask] == y[final_mask])) if np.any(final_mask) else None,
        "test_label_counts": pd.Series(class_array[final_test_pred]).value_counts().sort_index().to_dict(),
    }
    print(pd.DataFrame(rows).sort_values("score", ascending=False).round(6).to_string(index=False))
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
