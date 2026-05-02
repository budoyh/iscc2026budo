from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor

from common import ROOT, TEST_PATH, TRAIN_PATH, empty_prediction, validate_submission
from predict_torch_boundary_ensemble import predict_one as predict_boundary_one
from predict_torch_hybrid_ensemble import predict_one as predict_hybrid_one
from train_torch_boundary import log_softmax_1d
from train_torch_sequence import LABELS


SUBMISSION_PATH = ROOT / "submissions" / "s6.csv"
SUMMARY_PATH = ROOT / "logs" / "span_ranker_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a candidate span ranker on full labels and predict test spans.")
    parser.add_argument("--boundary-artifact", action="append", required=True)
    parser.add_argument("--hybrid-artifact", action="append", required=True)
    parser.add_argument("--boundary-weight", type=float, default=0.40)
    parser.add_argument("--hybrid-weight", type=float, default=0.60)
    parser.add_argument("--train-path", type=str, default=str(TRAIN_PATH))
    parser.add_argument("--test-path", type=str, default=str(TEST_PATH))
    parser.add_argument("--output-path", type=str, default=str(SUBMISSION_PATH))
    parser.add_argument("--summary-path", type=str, default=str(SUMMARY_PATH))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--force-count", type=int, default=3214)
    parser.add_argument("--variant-count", action="append", type=int, default=[])
    parser.add_argument("--confidence-output", type=str, default="")
    parser.add_argument("--top-per-label", type=int, default=5)
    parser.add_argument("--top-per-length", type=int, default=2)
    parser.add_argument("--sample-neg-ratio", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def add_weighted(
    summed: dict[int, dict[str, np.ndarray]],
    outputs: dict[int, dict[str, np.ndarray]],
    weight: float,
) -> None:
    for row_id, output in outputs.items():
        if row_id not in summed:
            summed[row_id] = {key: value.copy() * weight for key, value in output.items()}
        else:
            for key in summed[row_id]:
                summed[row_id][key] += output[key] * weight


def predict_mixed_outputs(args: argparse.Namespace, df: pd.DataFrame, device: torch.device) -> dict[int, dict[str, np.ndarray]]:
    summed: dict[int, dict[str, np.ndarray]] = {}
    per_boundary = args.boundary_weight / len(args.boundary_artifact)
    for artifact_path in args.boundary_artifact:
        payload = torch.load(artifact_path, map_location="cpu", weights_only=False)
        add_weighted(summed, predict_boundary_one(payload, df, args.batch_size, device), per_boundary)
    per_hybrid = args.hybrid_weight / len(args.hybrid_artifact)
    for artifact_path in args.hybrid_artifact:
        payload = torch.load(artifact_path, map_location="cpu", weights_only=False)
        add_weighted(summed, predict_hybrid_one(payload, df, args.batch_size, device), per_hybrid)
    total_weight = args.boundary_weight + args.hybrid_weight
    return {
        row_id: {
            "line_logits": output["line_logits"] / total_weight,
            "start_logits": output["start_logits"] / total_weight,
            "end_logits": output["end_logits"] / total_weight,
        }
        for row_id, output in summed.items()
    }


def build_length_priors(train_df: pd.DataFrame) -> dict[str, dict[int, float]]:
    counts: dict[str, Counter[int]] = defaultdict(Counter)
    for row in train_df[train_df["has_anomaly"] == 1].itertuples(index=False):
        span_len = int(row.primary_end_idx) - int(row.primary_start_idx) + 1
        counts[str(row.primary_anomaly_type)][span_len] += 1
    priors: dict[str, dict[int, float]] = {}
    for label_name in LABELS[1:]:
        total = sum(counts[label_name].values()) + 8
        priors[label_name] = {span_len: math.log((counts[label_name][span_len] + 1) / total) for span_len in range(3, 11)}
    return priors


def candidate_features(
    row: object,
    output: dict[str, np.ndarray],
    priors: dict[str, dict[int, float]],
    top_per_label: int,
    top_per_length: int,
    include_target: bool,
) -> tuple[np.ndarray, np.ndarray | None, list[tuple[int, int, str, float]]]:
    logits = output["line_logits"]
    line_count = len(logits)
    log_probs = logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)
    none_scores = log_probs[:, 0]
    start_log_probs = log_softmax_1d(output["start_logits"])
    end_log_probs = log_softmax_1d(output["end_logits"])
    feature_rows: list[list[float]] = []
    targets: list[float] = []
    metas: list[tuple[int, int, str, float]] = []

    true_anomaly = include_target and int(row.has_anomaly) == 1
    true_start = int(row.primary_start_idx) if true_anomaly else -1
    true_end = int(row.primary_end_idx) if true_anomaly else -1
    true_type = str(row.primary_anomaly_type) if true_anomaly else "none"

    for label_id, label_name in enumerate(LABELS[1:], start=1):
        diff = log_probs[:, label_id] - none_scores
        prefix = np.concatenate([[0.0], np.cumsum(diff)])
        label_candidates: list[tuple[float, int, int, int, float, float, float, float, float, float, float]] = []
        for span_len in range(3, 11):
            if span_len > line_count:
                break
            starts = np.arange(0, line_count - span_len + 1)
            ends = starts + span_len - 1
            sum_diff = prefix[span_len:] - prefix[:-span_len]
            raw_score = sum_diff + 1.5 * (start_log_probs[starts] + end_log_probs[ends])
            if len(raw_score) == 0:
                continue
            top_k = min(top_per_length, len(raw_score))
            top_idx = np.argpartition(raw_score, -top_k)[-top_k:]
            for idx in top_idx:
                start = int(starts[idx])
                end = int(ends[idx])
                window_label = log_probs[start : end + 1, label_id]
                window_none = none_scores[start : end + 1]
                label_candidates.append(
                    (
                        float(raw_score[idx]),
                        start,
                        end,
                        span_len,
                        float(sum_diff[idx]),
                        float(start_log_probs[start]),
                        float(end_log_probs[end]),
                        float(priors[label_name][span_len]),
                        float(window_label.mean()),
                        float(window_label.max()),
                        float(window_none.mean()),
                    )
                )
        for rank, item in enumerate(sorted(label_candidates, reverse=True)[:top_per_label]):
            raw_score, start, end, span_len, sum_diff, start_lp, end_lp, prior, win_mean, win_max, none_mean = item
            feature_rows.append(
                [
                    raw_score,
                    raw_score + 0.6 * prior,
                    raw_score + prior,
                    sum_diff,
                    start_lp,
                    end_lp,
                    prior,
                    float(span_len),
                    float(label_id),
                    start / line_count,
                    end / line_count,
                    (start + end) / (2 * line_count),
                    float(rank),
                    win_mean,
                    win_max,
                    none_mean,
                    raw_score - (raw_score + prior),
                    win_mean - none_mean,
                    float(line_count),
                ]
            )
            metas.append((start, end, label_name, raw_score))
            if include_target:
                if true_anomaly and label_name == true_type:
                    intersection = max(0, min(true_end, end) - max(true_start, start) + 1)
                    union = max(true_end, end) - min(true_start, start) + 1
                    targets.append(intersection / union if union else 0.0)
                else:
                    targets.append(0.0)

    return (
        np.asarray(feature_rows, dtype=np.float32),
        np.asarray(targets, dtype=np.float32) if include_target else None,
        metas,
    )


def build_training_matrix(
    train_df: pd.DataFrame,
    outputs: dict[int, dict[str, np.ndarray]],
    priors: dict[str, dict[int, float]],
    top_per_label: int,
    top_per_length: int,
    sample_neg_ratio: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    feature_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    for row in train_df.itertuples(index=False):
        features, targets, _ = candidate_features(
            row,
            outputs[int(row.id)],
            priors,
            top_per_label,
            top_per_length,
            include_target=True,
        )
        if len(features) == 0 or targets is None:
            continue
        feature_parts.append(features)
        target_parts.append(targets)
    features = np.vstack(feature_parts)
    targets = np.concatenate(target_parts)
    positive_idx = np.where(targets > 0)[0]
    zero_idx = np.where(targets == 0)[0]
    rng = np.random.default_rng(seed)
    if len(zero_idx) > len(positive_idx) * sample_neg_ratio:
        zero_idx = rng.choice(zero_idx, size=len(positive_idx) * sample_neg_ratio, replace=False)
    keep_idx = np.concatenate([positive_idx, zero_idx])
    rng.shuffle(keep_idx)
    return features[keep_idx], targets[keep_idx]


def train_rankers(features: np.ndarray, targets: np.ndarray, seed: int) -> list[object]:
    models: list[object] = [
        HistGradientBoostingRegressor(
            max_iter=260,
            learning_rate=0.045,
            max_leaf_nodes=31,
            l2_regularization=0.04,
            random_state=seed,
        ),
        ExtraTreesRegressor(
            n_estimators=220,
            max_depth=18,
            min_samples_leaf=4,
            n_jobs=-1,
            random_state=seed + 1,
        ),
    ]
    for model in models:
        model.fit(features, targets)
    return models


def apply_force_count(submission: pd.DataFrame, confidences: np.ndarray, force_count: int) -> pd.DataFrame:
    output = submission.copy()
    if force_count >= 0:
        keep = set(np.argsort(confidences)[::-1][:force_count])
        for idx in range(len(output)):
            if idx not in keep:
                output.loc[idx] = empty_prediction(int(output.loc[idx, "id"]))
    return output


def predict_raw_submission(
    test_df: pd.DataFrame,
    outputs: dict[int, dict[str, np.ndarray]],
    priors: dict[str, dict[int, float]],
    models: list[object],
    top_per_label: int,
    top_per_length: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    rows: list[dict[str, object]] = []
    confidences: list[float] = []
    for row in test_df.itertuples(index=False):
        features, _, metas = candidate_features(
            row,
            outputs[int(row.id)],
            priors,
            top_per_label,
            top_per_length,
            include_target=False,
        )
        if len(features) == 0:
            rows.append(empty_prediction(int(row.id)))
            confidences.append(-1.0)
            continue
        scores = np.mean([model.predict(features) for model in models], axis=0)
        best_idx = int(np.argmax(scores))
        start, end, label_name, _ = metas[best_idx]
        confidences.append(float(scores[best_idx]))
        rows.append(
            {
                "id": int(row.id),
                "has_anomaly": 1,
                "primary_start_idx": int(start),
                "primary_end_idx": int(end),
                "primary_anomaly_type": label_name,
                "all_spans": f"{int(start)}|{int(end)}|{label_name}",
            }
        )
    submission = pd.DataFrame(rows)
    confidences_array = np.asarray(confidences, dtype=np.float32)
    return submission, confidences_array


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)
    priors = build_length_priors(train_df)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Predicting train outputs...", flush=True)
    train_outputs = predict_mixed_outputs(args, train_df, device)
    print("Predicting test outputs...", flush=True)
    test_outputs = predict_mixed_outputs(args, test_df, device)
    print("Building training candidates...", flush=True)
    features, targets = build_training_matrix(
        train_df,
        train_outputs,
        priors,
        top_per_label=args.top_per_label,
        top_per_length=args.top_per_length,
        sample_neg_ratio=args.sample_neg_ratio,
        seed=args.seed,
    )
    print(f"Training rankers on {features.shape[0]} candidates...", flush=True)
    models = train_rankers(features, targets, args.seed)
    raw_submission, confidences = predict_raw_submission(
        test_df,
        test_outputs,
        priors,
        models,
        top_per_label=args.top_per_label,
        top_per_length=args.top_per_length,
    )
    submission = apply_force_count(raw_submission, confidences, args.force_count)
    errors = validate_submission(submission, expected_rows=len(test_df))
    if errors:
        raise SystemExit("submission validation failed: " + " | ".join(errors))

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False, encoding="utf-8")
    variant_paths: list[str] = []
    for force_count in args.variant_count:
        variant_submission = apply_force_count(raw_submission, confidences, force_count)
        errors = validate_submission(variant_submission, expected_rows=len(test_df))
        if errors:
            raise SystemExit(f"variant {force_count} validation failed: " + " | ".join(errors))
        variant_path = output_path.with_name(f"{output_path.stem}_c{force_count}.csv")
        variant_submission.to_csv(variant_path, index=False, encoding="utf-8")
        variant_paths.append(str(variant_path))
    if args.confidence_output:
        confidence_path = Path(args.confidence_output)
        confidence_path.parent.mkdir(parents=True, exist_ok=True)
        confidence_df = raw_submission.copy()
        confidence_df["ranker_confidence"] = confidences
        confidence_df.to_csv(confidence_path, index=False, encoding="utf-8")
    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "output_path": str(output_path),
                "force_count": int(args.force_count),
                "train_candidates": int(features.shape[0]),
                "top_per_label": int(args.top_per_label),
                "top_per_length": int(args.top_per_length),
                "target_mean": float(targets.mean()),
                "confidence_quantiles": {
                    str(q): float(np.quantile(confidences, q)) for q in [0.1, 0.25, 0.5, 0.75, 0.9]
                },
                "anomaly_count": int(submission["has_anomaly"].sum()),
                "variant_paths": variant_paths,
                "confidence_output": args.confidence_output,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Submission saved to: {output_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"Anomaly count: {int(submission['has_anomaly'].sum())} / {len(submission)}")
    print("Validation: OK")


if __name__ == "__main__":
    main()
