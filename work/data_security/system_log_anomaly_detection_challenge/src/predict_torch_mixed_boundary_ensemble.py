from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import ROOT, TEST_PATH, TRAIN_PATH, empty_prediction, validate_submission
from predict_torch_boundary_ensemble import predict_one as predict_boundary_one
from predict_torch_hybrid_ensemble import predict_one as predict_hybrid_one
from train_torch_boundary import decode_frame, log_softmax_1d
from train_torch_sequence import LABELS


SUBMISSION_PATH = ROOT / "submissions" / "submission_torch_mixed_boundary_ensemble.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blend proven word-boundary models with hybrid char+word boundary models.")
    parser.add_argument("--boundary-artifact", action="append", default=[])
    parser.add_argument("--hybrid-artifact", action="append", default=[])
    parser.add_argument("--boundary-weight", type=float, default=0.60)
    parser.add_argument("--hybrid-weight", type=float, default=0.40)
    parser.add_argument("--test-path", type=str, default=str(TEST_PATH))
    parser.add_argument("--output-path", type=str, default=str(SUBMISSION_PATH))
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--threshold", type=float, default=-4.0)
    parser.add_argument("--decode-boundary-weight", type=float, default=1.2)
    parser.add_argument("--length-prior-weight", type=float, default=0.0)
    parser.add_argument("--train-path", type=str, default=str(TRAIN_PATH))
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


def build_length_priors(train_path: str) -> dict[str, dict[int, float]]:
    train_df = pd.read_csv(train_path)
    counts: dict[str, Counter[int]] = defaultdict(Counter)
    for row in train_df[train_df["has_anomaly"] == 1].itertuples(index=False):
        span_len = int(row.primary_end_idx) - int(row.primary_start_idx) + 1
        counts[str(row.primary_anomaly_type)][span_len] += 1
    priors: dict[str, dict[int, float]] = {}
    for label_name in LABELS[1:]:
        total = sum(counts[label_name].values()) + 8
        priors[label_name] = {span_len: math.log((counts[label_name][span_len] + 1) / total) for span_len in range(3, 11)}
    return priors


def decode_row_with_length_prior(
    row_id: int,
    output: dict[str, np.ndarray],
    threshold: float,
    boundary_weight: float,
    length_priors: dict[str, dict[int, float]],
    length_prior_weight: float,
) -> dict[str, object]:
    logits = output["line_logits"]
    if len(logits) == 0:
        return empty_prediction(row_id)
    log_probs = logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)
    none_scores = log_probs[:, 0]
    start_log_probs = log_softmax_1d(output["start_logits"])
    end_log_probs = log_softmax_1d(output["end_logits"])
    line_count = len(logits)
    best: tuple[float, int, int, str] | None = None
    for label_id, label_name in enumerate(LABELS):
        if label_id == 0:
            continue
        diff = log_probs[:, label_id] - none_scores
        prefix = np.concatenate([[0.0], np.cumsum(diff)])
        for span_len in range(3, 11):
            if span_len > line_count:
                break
            starts = np.arange(0, line_count - span_len + 1)
            ends = starts + span_len - 1
            scores = (
                prefix[span_len:]
                - prefix[:-span_len]
                + boundary_weight * (start_log_probs[starts] + end_log_probs[ends])
                + length_prior_weight * length_priors[label_name][span_len]
            )
            best_idx = int(np.argmax(scores))
            candidate = (float(scores[best_idx]), -int(starts[best_idx]), int(ends[best_idx]), label_name)
            if best is None or candidate > best:
                best = candidate
    if best is None or best[0] < threshold:
        return empty_prediction(row_id)
    _, neg_start, end, label_name = best
    start = -neg_start
    return {
        "id": row_id,
        "has_anomaly": 1,
        "primary_start_idx": int(start),
        "primary_end_idx": int(end),
        "primary_anomaly_type": label_name,
        "all_spans": f"{int(start)}|{int(end)}|{label_name}",
    }


def decode_frame_with_length_prior(
    df: pd.DataFrame,
    outputs: dict[int, dict[str, np.ndarray]],
    threshold: float,
    boundary_weight: float,
    length_priors: dict[str, dict[int, float]],
    length_prior_weight: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            decode_row_with_length_prior(
                int(row.id),
                outputs[int(row.id)],
                threshold,
                boundary_weight,
                length_priors,
                length_prior_weight,
            )
            for row in df.itertuples(index=False)
        ]
    )


def main() -> None:
    args = parse_args()
    if not args.boundary_artifact and not args.hybrid_artifact:
        raise SystemExit("at least one artifact is required")
    test_df = pd.read_csv(args.test_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    summed: dict[int, dict[str, np.ndarray]] = {}
    total_weight = 0.0
    if args.boundary_artifact:
        per_model_weight = args.boundary_weight / len(args.boundary_artifact)
        for artifact_path in args.boundary_artifact:
            payload = torch.load(artifact_path, map_location="cpu", weights_only=False)
            add_weighted(summed, predict_boundary_one(payload, test_df, args.batch_size, device), per_model_weight)
            total_weight += per_model_weight
    if args.hybrid_artifact:
        per_model_weight = args.hybrid_weight / len(args.hybrid_artifact)
        for artifact_path in args.hybrid_artifact:
            payload = torch.load(artifact_path, map_location="cpu", weights_only=False)
            add_weighted(summed, predict_hybrid_one(payload, test_df, args.batch_size, device), per_model_weight)
            total_weight += per_model_weight

    averaged = {
        row_id: {
            "line_logits": output["line_logits"] / total_weight,
            "start_logits": output["start_logits"] / total_weight,
            "end_logits": output["end_logits"] / total_weight,
        }
        for row_id, output in summed.items()
    }
    if args.length_prior_weight > 0:
        submission = decode_frame_with_length_prior(
            test_df,
            averaged,
            args.threshold,
            args.decode_boundary_weight,
            build_length_priors(args.train_path),
            args.length_prior_weight,
        )
    else:
        submission = decode_frame(test_df, averaged, args.threshold, args.decode_boundary_weight)
    errors = validate_submission(submission, expected_rows=len(test_df))
    if errors:
        raise SystemExit("submission validation failed: " + " | ".join(errors))

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Submission saved to: {output_path}")
    print(f"Boundary artifacts: {len(args.boundary_artifact)} weight={args.boundary_weight}")
    print(f"Hybrid artifacts: {len(args.hybrid_artifact)} weight={args.hybrid_weight}")
    print(f"Threshold: {args.threshold}")
    print(f"Decode boundary weight: {args.decode_boundary_weight}")
    print(f"Length prior weight: {args.length_prior_weight}")
    print(f"Anomaly count: {int(submission['has_anomaly'].sum())} / {len(submission)}")
    print("Validation: OK")


if __name__ == "__main__":
    main()
