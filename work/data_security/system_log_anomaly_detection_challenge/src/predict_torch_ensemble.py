from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from common import ROOT, TEST_PATH, validate_submission
from train_torch_sequence import LABELS, LogDataset, SequenceTagger, collate_batch, decode_frame, encode_frame


SUBMISSION_PATH = ROOT / "submissions" / "submission_torch_sequence_ensemble.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a submission by averaging PyTorch sequence tagger outputs.")
    parser.add_argument("--artifact-path", action="append", required=True)
    parser.add_argument("--test-path", type=str, default=str(TEST_PATH))
    parser.add_argument("--output-path", type=str, default=str(SUBMISSION_PATH))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=-2.0)
    return parser.parse_args()


@torch.no_grad()
def predict_log_probs(payload: dict[str, object], test_df: pd.DataFrame, batch_size: int, device: torch.device) -> dict[int, np.ndarray]:
    model = SequenceTagger(
        vocab_size=len(payload["vocab"]),
        embed_dim=int(payload["embed_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
        label_count=len(LABELS),
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    items = encode_frame(test_df, payload["vocab"], int(payload["max_tokens"]), include_labels=False)
    loader = DataLoader(LogDataset(items), batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    outputs: dict[int, np.ndarray] = {}
    for batch in loader:
        logits = model(batch["tokens"].to(device), batch["token_mask"].to(device)).detach().cpu().numpy()
        logits = logits - np.logaddexp.reduce(logits, axis=2, keepdims=True)
        line_mask = batch["line_mask"].numpy()
        for bidx, row_id in enumerate(batch["ids"]):
            outputs[int(row_id)] = logits[bidx, line_mask[bidx]]
    return outputs


def main() -> None:
    args = parse_args()
    test_df = pd.read_csv(args.test_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    summed: dict[int, np.ndarray] = {}
    for artifact_path in args.artifact_path:
        payload = torch.load(artifact_path, map_location="cpu", weights_only=False)
        pred = predict_log_probs(payload, test_df, args.batch_size, device)
        for row_id, log_probs in pred.items():
            if row_id not in summed:
                summed[row_id] = log_probs.copy()
            else:
                summed[row_id] += log_probs

    count = len(args.artifact_path)
    averaged = {row_id: values / count for row_id, values in summed.items()}
    submission = decode_frame(test_df, averaged, args.threshold)
    errors = validate_submission(submission, expected_rows=len(test_df))
    if errors:
        raise SystemExit("submission validation failed: " + " | ".join(errors))

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Submission saved to: {output_path}")
    print(f"Artifacts: {count}")
    print(f"Threshold: {args.threshold}")
    print(f"Anomaly count: {int(submission['has_anomaly'].sum())} / {len(submission)}")
    print("Validation: OK")


if __name__ == "__main__":
    main()
