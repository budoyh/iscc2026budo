from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from common import ROOT, TRAIN_PATH
from train_torch_boundary import (
    BoundaryDataset,
    BoundarySequenceTagger,
    add_boundary_targets,
    collate_boundary,
    train_epoch,
)
from train_torch_sequence import LABELS, build_vocab, class_weights, encode_frame, set_seed


ARTIFACT_PATH = ROOT / "models" / "torch_boundary_full_model.pt"
METRICS_PATH = ROOT / "logs" / "torch_boundary_full_metrics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the boundary-aware PyTorch model on all rows.")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-vocab", type=int, default=50000)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--embed-dim", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--boundary-loss-weight", type=float, default=0.35)
    parser.add_argument("--threshold", type=float, default=-4.0)
    parser.add_argument("--decode-boundary-weight", type=float, default=1.2)
    parser.add_argument("--train-path", type=str, default=str(TRAIN_PATH))
    parser.add_argument("--artifact-path", type=str, default=str(ARTIFACT_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    train_df = pd.read_csv(args.train_path)
    vocab = build_vocab(train_df, args.max_vocab)
    items = add_boundary_targets(encode_frame(train_df, vocab, args.max_tokens, include_labels=True))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BoundarySequenceTagger(len(vocab), args.embed_dim, args.hidden_dim, len(LABELS)).to(device)
    line_criterion = nn.CrossEntropyLoss(weight=class_weights(items, device), ignore_index=-100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loader = DataLoader(BoundaryDataset(items), batch_size=args.batch_size, shuffle=True, collate_fn=collate_boundary, num_workers=0)

    history = []
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, loader, optimizer, line_criterion, args.boundary_loss_weight, device)
        record = {"epoch": epoch, "loss": float(loss)}
        history.append(record)
        print(json.dumps(record), flush=True)

    output_path = Path(args.artifact_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "vocab": vocab,
            "labels": LABELS,
            "max_tokens": args.max_tokens,
            "embed_dim": args.embed_dim,
            "hidden_dim": args.hidden_dim,
            "threshold": float(args.threshold),
            "decode_boundary_weight": float(args.decode_boundary_weight),
            "boundary_loss_weight": float(args.boundary_loss_weight),
            "history": history,
        },
        output_path,
    )
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(
        json.dumps(
            {
                "history": history,
                "threshold": args.threshold,
                "decode_boundary_weight": args.decode_boundary_weight,
                "mean_train_loss": float(np.mean([item["loss"] for item in history])),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Artifact saved to: {output_path}")


if __name__ == "__main__":
    main()
