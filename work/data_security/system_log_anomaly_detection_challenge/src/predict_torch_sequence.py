from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from common import ROOT, TEST_PATH, validate_submission
from train_torch_sequence import (
    ARTIFACT_PATH,
    LABELS,
    LogDataset,
    SequenceTagger,
    collate_batch,
    decode_frame,
    encode_frame,
)


SUBMISSION_PATH = ROOT / "submissions" / "submission_torch_sequence.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a submission with the PyTorch sequence tagger.")
    parser.add_argument("--test-path", type=str, default=str(TEST_PATH))
    parser.add_argument("--artifact-path", type=str, default=str(ARTIFACT_PATH))
    parser.add_argument("--output-path", type=str, default=str(SUBMISSION_PATH))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=None)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    payload = torch.load(args.artifact_path, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SequenceTagger(
        vocab_size=len(payload["vocab"]),
        embed_dim=int(payload["embed_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
        label_count=len(LABELS),
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    test_df = pd.read_csv(args.test_path)
    items = encode_frame(test_df, payload["vocab"], int(payload["max_tokens"]), include_labels=False)
    loader = DataLoader(LogDataset(items), batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    logits_by_id: dict[int, object] = {}
    for batch in loader:
        logits = model(batch["tokens"].to(device), batch["token_mask"].to(device)).detach().cpu().numpy()
        line_mask = batch["line_mask"].numpy()
        for bidx, row_id in enumerate(batch["ids"]):
            logits_by_id[int(row_id)] = logits[bidx, line_mask[bidx]]

    threshold = float(payload["threshold"] if args.threshold is None else args.threshold)
    submission = decode_frame(test_df, logits_by_id, threshold)
    errors = validate_submission(submission, expected_rows=len(test_df))
    if errors:
        raise SystemExit("submission validation failed: " + " | ".join(errors))

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Submission saved to: {output_path}")
    print(f"Threshold: {threshold}")
    print(f"Anomaly count: {int(submission['has_anomaly'].sum())} / {len(submission)}")
    print("Validation: OK")


if __name__ == "__main__":
    main()
