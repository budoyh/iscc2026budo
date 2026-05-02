from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from common import ROOT, TEST_PATH, validate_submission
from train_torch_boundary import BoundaryDataset, BoundarySequenceTagger, collate_boundary, decode_frame, log_softmax_1d
from train_torch_sequence import LABELS, tokenize


SUBMISSION_PATH = ROOT / "submissions" / "submission_torch_boundary_ensemble_fuzzy.csv"
ALPHA_PATTERN = re.compile(r"^[a-z_]{4,}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate boundary ensemble predictions with fuzzy OOV token correction.")
    parser.add_argument("--artifact-path", action="append", required=True)
    parser.add_argument("--test-path", type=str, default=str(TEST_PATH))
    parser.add_argument("--output-path", type=str, default=str(SUBMISSION_PATH))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=-4.0)
    parser.add_argument("--decode-boundary-weight", type=float, default=1.2)
    return parser.parse_args()


def bounded_damerau_levenshtein(left: str, right: str, cutoff: int) -> int:
    if abs(len(left) - len(right)) > cutoff:
        return cutoff + 1
    previous_previous: list[int] | None = None
    previous = list(range(len(right) + 1))
    for i, char_left in enumerate(left, start=1):
        current = [i] + [0] * len(right)
        row_min = current[0]
        for j, char_right in enumerate(right, start=1):
            cost = 0 if char_left == char_right else 1
            value = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + cost,
            )
            if (
                previous_previous is not None
                and i > 1
                and j > 1
                and char_left == right[j - 2]
                and left[i - 2] == char_right
            ):
                value = min(value, previous_previous[j - 2] + 1)
            current[j] = value
            row_min = min(row_min, value)
        if row_min > cutoff:
            return cutoff + 1
        previous_previous, previous = previous, current
    return previous[-1]


def build_candidate_index(vocab: dict[str, int]) -> dict[tuple[str, int], list[str]]:
    index: dict[tuple[str, int], list[str]] = {}
    for token in vocab:
        if token.startswith("<") or not ALPHA_PATTERN.match(token):
            continue
        index.setdefault((token[0], len(token)), []).append(token)
    return index


def correct_token(
    token: str,
    vocab: dict[str, int],
    candidate_index: dict[tuple[str, int], list[str]],
    cache: dict[str, str],
    corrections: Counter[tuple[str, str]],
) -> str:
    if token in vocab:
        return token
    if token in cache:
        return cache[token]
    corrected = token
    if ALPHA_PATTERN.match(token):
        cutoff = 1 if len(token) <= 5 else 2 if len(token) <= 10 else 3
        candidates: list[str] = []
        for length in range(max(4, len(token) - cutoff), len(token) + cutoff + 1):
            candidates.extend(candidate_index.get((token[0], length), []))
        best: tuple[int, float, str] | None = None
        for candidate in candidates:
            distance = bounded_damerau_levenshtein(token, candidate, cutoff)
            if distance > cutoff:
                continue
            ratio = distance / max(len(token), len(candidate))
            current = (distance, ratio, candidate)
            if best is None or current < best:
                best = current
        if best is not None and best[1] <= 0.28:
            corrected = best[2]
            corrections[(token, corrected)] += 1
    cache[token] = corrected
    return corrected


def encode_line_fuzzy(
    line: str,
    vocab: dict[str, int],
    max_tokens: int,
    candidate_index: dict[tuple[str, int], list[str]],
    cache: dict[str, str],
    corrections: Counter[tuple[str, str]],
) -> list[int]:
    ids = [
        vocab.get(correct_token(token, vocab, candidate_index, cache, corrections), 1)
        for token in tokenize(line)[:max_tokens]
    ]
    return ids or [1]


def encode_frame_fuzzy(
    df: pd.DataFrame,
    vocab: dict[str, int],
    max_tokens: int,
    candidate_index: dict[tuple[str, int], list[str]],
    corrections: Counter[tuple[str, str]],
) -> list[dict[str, object]]:
    cache: dict[str, str] = {}
    encoded: list[dict[str, object]] = []
    for row in df.itertuples(index=False):
        lines = str(row.log_text).split("\n")
        encoded.append(
            {
                "id": int(row.id),
                "tokens": [
                    encode_line_fuzzy(line, vocab, max_tokens, candidate_index, cache, corrections)
                    for line in lines
                ],
                "line_count": len(lines),
            }
        )
    return encoded


@torch.no_grad()
def predict_one(
    payload: dict[str, object],
    test_df: pd.DataFrame,
    batch_size: int,
    device: torch.device,
    corrections: Counter[tuple[str, str]],
) -> dict[int, dict[str, np.ndarray]]:
    model = BoundarySequenceTagger(
        vocab_size=len(payload["vocab"]),
        embed_dim=int(payload["embed_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
        label_count=len(LABELS),
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    candidate_index = build_candidate_index(payload["vocab"])
    items = encode_frame_fuzzy(test_df, payload["vocab"], int(payload["max_tokens"]), candidate_index, corrections)
    loader = DataLoader(BoundaryDataset(items), batch_size=batch_size, shuffle=False, collate_fn=collate_boundary)
    outputs: dict[int, dict[str, np.ndarray]] = {}
    for batch in loader:
        line_logits, start_logits, end_logits = model(batch["tokens"].to(device), batch["token_mask"].to(device))
        line_logits_np = line_logits.detach().cpu().numpy()
        line_log_probs = line_logits_np - np.logaddexp.reduce(line_logits_np, axis=2, keepdims=True)
        start_np = start_logits.detach().cpu().numpy()
        end_np = end_logits.detach().cpu().numpy()
        line_mask = batch["line_mask"].numpy()
        for bidx, row_id in enumerate(batch["ids"]):
            mask = line_mask[bidx]
            outputs[int(row_id)] = {
                "line_logits": line_log_probs[bidx, mask],
                "start_logits": log_softmax_1d(start_np[bidx, mask]),
                "end_logits": log_softmax_1d(end_np[bidx, mask]),
            }
    return outputs


def main() -> None:
    args = parse_args()
    test_df = pd.read_csv(args.test_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    summed: dict[int, dict[str, np.ndarray]] = {}
    corrections: Counter[tuple[str, str]] = Counter()
    for artifact_path in args.artifact_path:
        payload = torch.load(artifact_path, map_location="cpu", weights_only=False)
        pred = predict_one(payload, test_df, args.batch_size, device, corrections)
        for row_id, output in pred.items():
            if row_id not in summed:
                summed[row_id] = {key: value.copy() for key, value in output.items()}
            else:
                for key in summed[row_id]:
                    summed[row_id][key] += output[key]

    model_count = len(args.artifact_path)
    averaged = {
        row_id: {
            "line_logits": output["line_logits"] / model_count,
            "start_logits": output["start_logits"] / model_count,
            "end_logits": output["end_logits"] / model_count,
        }
        for row_id, output in summed.items()
    }
    submission = decode_frame(test_df, averaged, args.threshold, args.decode_boundary_weight)
    errors = validate_submission(submission, expected_rows=len(test_df))
    if errors:
        raise SystemExit("submission validation failed: " + " | ".join(errors))

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Submission saved to: {output_path}")
    print(f"Artifacts: {model_count}")
    print(f"Threshold: {args.threshold}")
    print(f"Decode boundary weight: {args.decode_boundary_weight}")
    print(f"Anomaly count: {int(submission['has_anomaly'].sum())} / {len(submission)}")
    print("Top fuzzy corrections:")
    for (source, target), count in corrections.most_common(40):
        print(f"  {source} -> {target}: {count}")
    print("Validation: OK")


if __name__ == "__main__":
    main()
