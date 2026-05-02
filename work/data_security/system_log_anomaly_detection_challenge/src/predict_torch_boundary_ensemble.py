from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from common import ROOT, TEST_PATH, validate_submission
from train_torch_boundary import BoundaryDataset, BoundarySequenceTagger, collate_boundary, decode_frame, log_softmax_1d
from train_torch_sequence import LABELS, encode_frame


SUBMISSION_PATH = ROOT / "submissions" / "submission_torch_boundary_ensemble.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an ensemble submission with boundary-aware PyTorch models.")
    parser.add_argument("--artifact-path", action="append", required=True)
    parser.add_argument("--test-path", type=str, default=str(TEST_PATH))
    parser.add_argument("--output-path", type=str, default=str(SUBMISSION_PATH))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=-4.0)
    parser.add_argument("--decode-boundary-weight", type=float, default=1.2)
    return parser.parse_args()


@torch.no_grad()
def predict_one(payload: dict[str, object], test_df: pd.DataFrame, batch_size: int, device: torch.device) -> dict[int, dict[str, np.ndarray]]:
    model = BoundarySequenceTagger(
        vocab_size=len(payload["vocab"]),
        embed_dim=int(payload["embed_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
        label_count=len(LABELS),
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    items = encode_frame(test_df, payload["vocab"], int(payload["max_tokens"]), include_labels=False)
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
    for artifact_path in args.artifact_path:
        payload = torch.load(artifact_path, map_location="cpu", weights_only=False)
        pred = predict_one(payload, test_df, args.batch_size, device)
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
    print("Validation: OK")


if __name__ == "__main__":
    main()
