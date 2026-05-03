from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw" / "data"
SUBMISSIONS = ROOT / "submissions"


def load_cluster_module():
    path = ROOT / "src" / "cluster_prior_adjust.py"
    spec = importlib.util.spec_from_file_location("cluster_prior_adjust_local", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load cluster_prior_adjust.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search filters for cluster-prior changed rows.")
    parser.add_argument("--base-report", default="upload_ready_breakthrough5/c10_c04_patternshuffle_s43_25.json")
    parser.add_argument("--features", choices=["template", "top10"], default="template")
    parser.add_argument("--k", type=int, default=768)
    parser.add_argument("--alpha", type=float, default=1.2)
    parser.add_argument("--min-count", type=int, default=25)
    parser.add_argument("--smooth", type=float, default=15.0)
    parser.add_argument("--ratio-clip", type=float, default=1.8)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--base-conf-max", type=float, default=None)
    parser.add_argument("--base-margin-max", type=float, default=None)
    parser.add_argument("--adj-conf-min", type=float, default=None)
    parser.add_argument("--adj-margin-min", type=float, default=None)
    parser.add_argument("--pair-whitelist", nargs="*", default=[])
    return parser.parse_args()


def top_margin(proba: np.ndarray) -> np.ndarray:
    part = np.partition(proba, -2, axis=1)
    return part[:, -1] - part[:, -2]


def apply_filter(
    base_proba: np.ndarray,
    adjusted_proba: np.ndarray,
    base_conf_max: float | None,
    base_margin_max: float | None,
    adj_conf_min: float | None,
    adj_margin_min: float | None,
    pair_whitelist: set[tuple[int, int]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    base_pred = base_proba.argmax(axis=1)
    adj_pred = adjusted_proba.argmax(axis=1)
    changed = adj_pred != base_pred
    keep = changed.copy()
    base_conf = base_proba.max(axis=1)
    adj_conf = adjusted_proba.max(axis=1)
    base_margin = top_margin(base_proba)
    adj_margin = top_margin(adjusted_proba)
    if base_conf_max is not None:
        keep &= base_conf <= base_conf_max
    if base_margin_max is not None:
        keep &= base_margin <= base_margin_max
    if adj_conf_min is not None:
        keep &= adj_conf >= adj_conf_min
    if adj_margin_min is not None:
        keep &= adj_margin >= adj_margin_min
    if pair_whitelist is not None:
        pair_keep = np.zeros_like(keep)
        for base_cls, adj_cls in pair_whitelist:
            pair_keep |= (base_pred == base_cls) & (adj_pred == adj_cls)
        keep &= pair_keep
    out = base_proba.copy()
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
    features = mod.FEATURE_GROUPS[args.features]
    x_train = train[features].to_numpy(dtype=np.float32)
    x_test = test[features].to_numpy(dtype=np.float32)

    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    adjusted_oof = np.zeros_like(base_oof)
    for fold, (tr_idx, va_idx) in enumerate(cv.split(x_train, y), start=1):
        state = mod.fit_transductive_clusters(
            x_train[tr_idx],
            x_train[va_idx],
            args.k,
            args.min_count,
            args.seed * 100 + fold,
        )
        adjusted_oof[va_idx] = mod.adjust_proba(
            base_oof[va_idx],
            state,
            y[tr_idx],
            base_oof[tr_idx],
            args.alpha,
            args.smooth,
            args.ratio_clip,
            len(classes),
        )

    base_pred = base_oof.argmax(axis=1)
    adj_pred = adjusted_oof.argmax(axis=1)
    changed = adj_pred != base_pred
    base_conf = base_oof.max(axis=1)
    adj_conf = adjusted_oof.max(axis=1)
    base_margin = top_margin(base_oof)
    adj_margin = top_margin(adjusted_oof)

    rows = []
    base_conf_grid = [None, 0.95, 0.90, 0.85, 0.80, 0.70]
    base_margin_grid = [None, 0.90, 0.80, 0.70, 0.60, 0.50]
    adj_margin_grid = [None, 0.02, 0.04, 0.06, 0.08, 0.10]
    for bcm in base_conf_grid:
        for bmm in base_margin_grid:
            for amm in adj_margin_grid:
                filt = changed.copy()
                if bcm is not None:
                    filt &= base_conf <= bcm
                if bmm is not None:
                    filt &= base_margin <= bmm
                if amm is not None:
                    filt &= adj_margin >= amm
                pred = base_pred.copy()
                pred[filt] = adj_pred[filt]
                rows.append(
                    {
                        "base_conf_max": bcm,
                        "base_margin_max": bmm,
                        "adj_margin_min": amm,
                        "changed_rows": int(filt.sum()),
                        "score": float(f1_score(y, pred, average="macro")),
                        "new_acc": float(np.mean(adj_pred[filt] == y[filt])) if np.any(filt) else None,
                        "base_acc": float(np.mean(base_pred[filt] == y[filt])) if np.any(filt) else None,
                    }
                )
    frame = pd.DataFrame(rows).sort_values(["score", "changed_rows"], ascending=[False, True])
    print(frame.head(40).round(6).to_string(index=False))

    pair_whitelist = None
    if args.pair_whitelist:
        label_to_idx = {label: idx for idx, label in enumerate(classes)}
        pair_whitelist = set()
        for pair in args.pair_whitelist:
            if ">" in pair:
                left, right = pair.split(">", 1)
            elif ":" in pair:
                left, right = pair.split(":", 1)
            else:
                raise ValueError(f"pair must be base>adj or base:adj, got {pair}")
            pair_whitelist.add((label_to_idx[left], label_to_idx[right]))

    selected_oof, selected_mask = apply_filter(
        base_oof,
        adjusted_oof,
        args.base_conf_max,
        args.base_margin_max,
        args.adj_conf_min,
        args.adj_margin_min,
        pair_whitelist,
    )
    print(
        json.dumps(
            {
                "selected_score": float(f1_score(y, selected_oof.argmax(axis=1), average="macro")),
                "selected_changed_oof_rows": int(selected_mask.sum()),
                "selected_new_acc": float(np.mean(selected_oof.argmax(axis=1)[selected_mask] == y[selected_mask]))
                if np.any(selected_mask)
                else None,
                "selected_base_acc": float(np.mean(base_pred[selected_mask] == y[selected_mask]))
                if np.any(selected_mask)
                else None,
            },
            ensure_ascii=False,
        )
    )

    if args.output_dir and args.output_name:
        state = mod.fit_transductive_clusters(x_train, x_test, args.k, args.min_count, args.seed)
        adjusted_test = mod.adjust_proba(
            base_test,
            state,
            y,
            base_oof,
            args.alpha,
            args.smooth,
            args.ratio_clip,
            len(classes),
        )
        selected_test, selected_test_mask = apply_filter(
            base_test,
            adjusted_test,
            args.base_conf_max,
            args.base_margin_max,
            args.adj_conf_min,
            args.adj_margin_min,
            pair_whitelist,
        )
        output_dir = ROOT / args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        SUBMISSIONS.mkdir(parents=True, exist_ok=True)
        submission = pd.DataFrame({"id": test["id"], "label": class_array[selected_test.argmax(axis=1)]})
        output_path = output_dir / f"{args.output_name}.csv"
        mirror_path = SUBMISSIONS / f"{args.output_name}.csv"
        submission.to_csv(output_path, index=False, encoding="utf-8", lineterminator="\r\n")
        submission.to_csv(mirror_path, index=False, encoding="utf-8", lineterminator="\r\n")
        report = {
            "output_path": str(output_path),
            "mirror_path": str(mirror_path),
            "base_report": str(mod.resolve_path(args.base_report)),
            "features": args.features,
            "k": args.k,
            "alpha": args.alpha,
            "min_count": args.min_count,
            "smooth": args.smooth,
            "ratio_clip": args.ratio_clip,
            "base_conf_max": args.base_conf_max,
            "base_margin_max": args.base_margin_max,
            "adj_conf_min": args.adj_conf_min,
            "adj_margin_min": args.adj_margin_min,
            "pair_whitelist": args.pair_whitelist,
            "local_score": float(f1_score(y, selected_oof.argmax(axis=1), average="macro")),
            "changed_oof_rows": int(selected_mask.sum()),
            "changed_test_rows": int(selected_test_mask.sum()),
            "submission_validation": mod.validate_submission(submission, sample, set(classes)),
        }
        output_path.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        mirror_path.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
