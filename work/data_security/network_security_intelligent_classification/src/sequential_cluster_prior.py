from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw" / "data"
SUBMISSIONS = ROOT / "submissions"


@dataclass(frozen=True)
class Stage:
    features: str
    k: int
    alpha: float
    min_count: int
    smooth: float
    ratio_clip: float


def load_cluster_module():
    path = ROOT / "src" / "cluster_prior_adjust.py"
    spec = importlib.util.spec_from_file_location("cluster_prior_adjust_local", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load cluster_prior_adjust.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_stage(text: str) -> Stage:
    parts = text.split(":")
    if len(parts) != 6:
        raise ValueError(
            "--stage must be features:k:alpha:min_count:smooth:ratio_clip, "
            f"got {text!r}"
        )
    return Stage(
        features=parts[0],
        k=int(parts[1]),
        alpha=float(parts[2]),
        min_count=int(parts[3]),
        smooth=float(parts[4]),
        ratio_clip=float(parts[5]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply several cluster-prior stages in sequence.")
    parser.add_argument("--base-report", default="upload_ready_breakthrough5/c10_c04_patternshuffle_s43_25.json")
    parser.add_argument(
        "--stage",
        action="append",
        required=True,
        help="features:k:alpha:min_count:smooth:ratio_clip, e.g. template:768:1.2:25:15:1.8",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output-name", default=None)
    return parser.parse_args()


def top_margin(proba: np.ndarray) -> np.ndarray:
    part = np.partition(proba, -2, axis=1)
    return part[:, -1] - part[:, -2]


def apply_stage(
    mod,
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    oof: np.ndarray,
    test_proba: np.ndarray,
    stage: Stage,
    folds: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    feature_names = mod.FEATURE_GROUPS[stage.features]
    x_train = train[feature_names].to_numpy(dtype=np.float32)
    x_test = test[feature_names].to_numpy(dtype=np.float32)

    adjusted_oof = np.zeros_like(oof)
    coverage = np.zeros(len(train), dtype=bool)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (tr_idx, va_idx) in enumerate(cv.split(x_train, y), start=1):
        state = mod.fit_transductive_clusters(
            x_train[tr_idx],
            x_train[va_idx],
            stage.k,
            stage.min_count,
            seed * 100 + fold,
        )
        coverage[va_idx] = state.trusted_apply_mask
        adjusted_oof[va_idx] = mod.adjust_proba(
            oof[va_idx],
            state,
            y[tr_idx],
            oof[tr_idx],
            stage.alpha,
            stage.smooth,
            stage.ratio_clip,
            oof.shape[1],
        )

    state = mod.fit_transductive_clusters(x_train, x_test, stage.k, stage.min_count, seed)
    adjusted_test = mod.adjust_proba(
        test_proba,
        state,
        y,
        oof,
        stage.alpha,
        stage.smooth,
        stage.ratio_clip,
        oof.shape[1],
    )

    before_pred = oof.argmax(axis=1)
    after_pred = adjusted_oof.argmax(axis=1)
    changed = after_pred != before_pred
    before_test_pred = test_proba.argmax(axis=1)
    after_test_pred = adjusted_test.argmax(axis=1)
    changed_test = after_test_pred != before_test_pred
    report = {
        "features": stage.features,
        "k": stage.k,
        "alpha": stage.alpha,
        "min_count": stage.min_count,
        "smooth": stage.smooth,
        "ratio_clip": stage.ratio_clip,
        "before_oof_macro_f1": float(f1_score(y, before_pred, average="macro")),
        "after_oof_macro_f1": float(f1_score(y, after_pred, average="macro")),
        "changed_oof_rows": int(changed.sum()),
        "covered_oof_rows": int(coverage.sum()),
        "changed_oof_new_acc": float(np.mean(after_pred[changed] == y[changed])) if np.any(changed) else None,
        "changed_oof_before_acc": float(np.mean(before_pred[changed] == y[changed])) if np.any(changed) else None,
        "changed_oof_before_margin_mean": float(top_margin(oof)[changed].mean()) if np.any(changed) else None,
        "changed_test_rows": int(changed_test.sum()),
        "test_covered_rows": int(state.trusted_apply_mask.sum()),
    }
    return adjusted_oof, adjusted_test, report


def main() -> None:
    args = parse_args()
    mod = load_cluster_module()
    stages = [parse_stage(text) for text in args.stage]
    unknown = [stage.features for stage in stages if stage.features not in mod.FEATURE_GROUPS]
    if unknown:
        raise ValueError(f"unknown feature groups: {unknown}; available={sorted(mod.FEATURE_GROUPS)}")

    train = pd.read_csv(RAW / "train_data.csv")
    test = pd.read_csv(RAW / "test_data.csv")
    sample = pd.read_csv(RAW / "sample_submission.csv")
    encoder = LabelEncoder()
    y = encoder.fit_transform(train["label"])
    classes = encoder.classes_.tolist()
    class_array = np.asarray(classes)

    base_oof, base_test, _ = mod.load_report(mod.resolve_path(args.base_report), classes)
    current_oof = base_oof
    current_test = base_test
    stage_reports = []
    for index, stage in enumerate(stages, start=1):
        current_oof, current_test, stage_report = apply_stage(
            mod,
            train,
            test,
            y,
            current_oof,
            current_test,
            stage,
            args.folds,
            args.seed + (index - 1) * 1000,
        )
        stage_report["stage_index"] = index
        stage_reports.append(stage_report)
        print(json.dumps(stage_report, ensure_ascii=False))

    base_pred = base_oof.argmax(axis=1)
    final_pred = current_oof.argmax(axis=1)
    base_test_pred = base_test.argmax(axis=1)
    final_test_pred = current_test.argmax(axis=1)
    report = {
        "base_report": str(mod.resolve_path(args.base_report)),
        "stages": [stage.__dict__ for stage in stages],
        "stage_reports": stage_reports,
        "base_oof_macro_f1": float(f1_score(y, base_pred, average="macro")),
        "final_oof_macro_f1": float(f1_score(y, final_pred, average="macro")),
        "changed_oof_rows_vs_base": int(np.sum(final_pred != base_pred)),
        "changed_test_rows_vs_base": int(np.sum(final_test_pred != base_test_pred)),
        "test_label_counts": pd.Series(class_array[final_test_pred]).value_counts().sort_index().to_dict(),
    }
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
                "submission_validation": mod.validate_submission(submission, sample, set(classes)),
            }
        )
        output_path.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        mirror_path.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"output_path": str(output_path), "submission_validation": report["submission_validation"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
