from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, Dataset

from common import ROOT, TRAIN_PATH, compute_score, empty_prediction
from train_torch_sequence import (
    LABELS,
    build_vocab,
    class_weights,
    encode_frame,
    set_seed,
)


ARTIFACT_PATH = ROOT / "models" / "torch_boundary_model.pt"
METRICS_PATH = ROOT / "logs" / "torch_boundary_metrics.json"
VAL_PRED_PATH = ROOT / "processed" / "torch_boundary_val_predictions.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a boundary-aware PyTorch sequence tagger.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-vocab", type=int, default=50000)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--embed-dim", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--boundary-loss-weight", type=float, default=0.35)
    parser.add_argument("--train-path", type=str, default=str(TRAIN_PATH))
    return parser.parse_args()


def add_boundary_targets(items: list[dict[str, object]]) -> list[dict[str, object]]:
    for item in items:
        labels = item["labels"]
        start = -1
        end = -1
        for idx, label in enumerate(labels):
            if label != 0:
                start = idx
                break
        for idx in range(len(labels) - 1, -1, -1):
            if labels[idx] != 0:
                end = idx
                break
        item["start_idx"] = start
        item["end_idx"] = end
    return items


class BoundaryDataset(Dataset):
    def __init__(self, items: list[dict[str, object]]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, object]:
        return self.items[idx]


def collate_boundary(batch: list[dict[str, object]]) -> dict[str, torch.Tensor | list[int]]:
    max_lines = max(int(item["line_count"]) for item in batch)
    max_tokens = max(max(len(line) for line in item["tokens"]) for item in batch)
    token_tensor = torch.zeros((len(batch), max_lines, max_tokens), dtype=torch.long)
    token_mask = torch.zeros((len(batch), max_lines, max_tokens), dtype=torch.float32)
    line_mask = torch.zeros((len(batch), max_lines), dtype=torch.bool)
    label_tensor = torch.full((len(batch), max_lines), -100, dtype=torch.long)
    start_targets = torch.full((len(batch),), -100, dtype=torch.long)
    end_targets = torch.full((len(batch),), -100, dtype=torch.long)
    ids: list[int] = []

    for bidx, item in enumerate(batch):
        ids.append(int(item["id"]))
        for lidx, token_ids in enumerate(item["tokens"]):
            token_tensor[bidx, lidx, : len(token_ids)] = torch.tensor(token_ids, dtype=torch.long)
            token_mask[bidx, lidx, : len(token_ids)] = 1.0
            line_mask[bidx, lidx] = True
        if "labels" in item:
            labels = item["labels"]
            label_tensor[bidx, : len(labels)] = torch.tensor(labels, dtype=torch.long)
            if int(item["start_idx"]) >= 0:
                start_targets[bidx] = int(item["start_idx"])
                end_targets[bidx] = int(item["end_idx"])

    return {
        "ids": ids,
        "tokens": token_tensor,
        "token_mask": token_mask,
        "line_mask": line_mask,
        "labels": label_tensor,
        "start_targets": start_targets,
        "end_targets": end_targets,
    }


class BoundarySequenceTagger(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, label_count: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(0.20)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True, dropout=0.15)
        self.classifier = nn.Linear(hidden_dim * 2, label_count)
        self.start_head = nn.Linear(hidden_dim * 2, 1)
        self.end_head = nn.Linear(hidden_dim * 2, 1)

    def forward(self, tokens: torch.Tensor, token_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.embedding(tokens)
        denom = token_mask.sum(dim=2).clamp_min(1.0).unsqueeze(-1)
        line_emb = (emb * token_mask.unsqueeze(-1)).sum(dim=2) / denom
        line_emb = self.dropout(line_emb)
        seq_out, _ = self.gru(line_emb)
        seq_out = self.dropout(seq_out)
        return self.classifier(seq_out), self.start_head(seq_out).squeeze(-1), self.end_head(seq_out).squeeze(-1)


def masked_boundary_loss(logits: torch.Tensor, targets: torch.Tensor, line_mask: torch.Tensor) -> torch.Tensor:
    valid = targets >= 0
    if not valid.any():
        return logits.sum() * 0.0
    masked_logits = logits.masked_fill(~line_mask, -1e4)
    return nn.functional.cross_entropy(masked_logits[valid], targets[valid])


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    line_criterion: nn.Module,
    boundary_weight: float,
    device: torch.device,
) -> float:
    model.train()
    losses: list[float] = []
    for batch in loader:
        tokens = batch["tokens"].to(device)
        token_mask = batch["token_mask"].to(device)
        labels = batch["labels"].to(device)
        line_mask = batch["line_mask"].to(device)
        start_targets = batch["start_targets"].to(device)
        end_targets = batch["end_targets"].to(device)
        optimizer.zero_grad(set_to_none=True)
        line_logits, start_logits, end_logits = model(tokens, token_mask)
        line_loss = line_criterion(line_logits.view(-1, len(LABELS)), labels.view(-1))
        start_loss = masked_boundary_loss(start_logits, start_targets, line_mask)
        end_loss = masked_boundary_loss(end_logits, end_targets, line_mask)
        loss = line_loss + boundary_weight * (start_loss + end_loss)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


@torch.no_grad()
def predict_outputs(model: nn.Module, items: list[dict[str, object]], batch_size: int, device: torch.device) -> dict[int, dict[str, np.ndarray]]:
    model.eval()
    loader = DataLoader(BoundaryDataset(items), batch_size=batch_size, shuffle=False, collate_fn=collate_boundary)
    outputs: dict[int, dict[str, np.ndarray]] = {}
    for batch in loader:
        line_logits, start_logits, end_logits = model(batch["tokens"].to(device), batch["token_mask"].to(device))
        line_logits_np = line_logits.detach().cpu().numpy()
        start_np = start_logits.detach().cpu().numpy()
        end_np = end_logits.detach().cpu().numpy()
        line_mask = batch["line_mask"].numpy()
        for bidx, row_id in enumerate(batch["ids"]):
            mask = line_mask[bidx]
            outputs[int(row_id)] = {
                "line_logits": line_logits_np[bidx, mask],
                "start_logits": start_np[bidx, mask],
                "end_logits": end_np[bidx, mask],
            }
    return outputs


def log_softmax_1d(values: np.ndarray) -> np.ndarray:
    max_value = float(np.max(values))
    return values - max_value - np.log(np.exp(values - max_value).sum())


def decode_row(row_id: int, output: dict[str, np.ndarray], threshold: float, boundary_weight: float) -> dict[str, object]:
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
            sums = prefix[span_len:] - prefix[:-span_len]
            starts = np.arange(0, line_count - span_len + 1)
            ends = starts + span_len - 1
            scores = sums + boundary_weight * (start_log_probs[starts] + end_log_probs[ends])
            best_idx = int(np.argmax(scores))
            score = float(scores[best_idx])
            start = int(starts[best_idx])
            end = int(ends[best_idx])
            candidate = (score, -start, end, label_name)
            if best is None or candidate > best:
                best = candidate
    if best is None or best[0] < threshold:
        return empty_prediction(row_id)
    score, neg_start, end, label_name = best
    start = -neg_start
    return {
        "id": row_id,
        "has_anomaly": 1,
        "primary_start_idx": int(start),
        "primary_end_idx": int(end),
        "primary_anomaly_type": label_name,
        "all_spans": f"{int(start)}|{int(end)}|{label_name}",
    }


def decode_frame(df: pd.DataFrame, outputs: dict[int, dict[str, np.ndarray]], threshold: float, boundary_weight: float) -> pd.DataFrame:
    return pd.DataFrame([decode_row(int(row.id), outputs[int(row.id)], threshold, boundary_weight) for row in df.itertuples(index=False)])


def tune_decode(valid_df: pd.DataFrame, outputs: dict[int, dict[str, np.ndarray]]) -> tuple[float, float, dict[str, float]]:
    best: tuple[float, float, dict[str, float]] | None = None
    for boundary_weight in [0.0, 0.4, 0.8, 1.2]:
        for threshold in [-4.0, -2.0, 0.0, 2.0]:
            pred = decode_frame(valid_df, outputs, float(threshold), float(boundary_weight))
            score = compute_score(valid_df, pred)
            if best is None or score["final_score"] > best[2]["final_score"]:
                best = (float(threshold), float(boundary_weight), score)
    assert best is not None
    return best


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    train_df = pd.read_csv(args.train_path)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=args.seed)
    train_idx, valid_idx = next(splitter.split(train_df, train_df["primary_anomaly_type"]))
    fit_df = train_df.iloc[train_idx].reset_index(drop=True)
    valid_df = train_df.iloc[valid_idx].reset_index(drop=True)
    vocab = build_vocab(fit_df, args.max_vocab)
    fit_items = add_boundary_targets(encode_frame(fit_df, vocab, args.max_tokens, include_labels=True))
    valid_items = add_boundary_targets(encode_frame(valid_df, vocab, args.max_tokens, include_labels=True))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BoundarySequenceTagger(len(vocab), args.embed_dim, args.hidden_dim, len(LABELS)).to(device)
    line_criterion = nn.CrossEntropyLoss(weight=class_weights(fit_items, device), ignore_index=-100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loader = DataLoader(BoundaryDataset(fit_items), batch_size=args.batch_size, shuffle=True, collate_fn=collate_boundary, num_workers=0)
    history: list[dict[str, object]] = []
    best_payload: dict[str, object] | None = None

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, loader, optimizer, line_criterion, args.boundary_loss_weight, device)
        outputs = predict_outputs(model, valid_items, args.batch_size, device)
        threshold, decode_boundary_weight, score = tune_decode(valid_df, outputs)
        record = {
            "epoch": epoch,
            "loss": loss,
            "threshold": threshold,
            "decode_boundary_weight": decode_boundary_weight,
            **score,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if best_payload is None or score["final_score"] > best_payload["score"]["final_score"]:
            best_payload = {
                "epoch": epoch,
                "threshold": threshold,
                "decode_boundary_weight": decode_boundary_weight,
                "score": score,
                "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            }

    assert best_payload is not None
    model.load_state_dict(best_payload["state_dict"])
    outputs = predict_outputs(model, valid_items, args.batch_size, device)
    val_pred = decode_frame(valid_df, outputs, float(best_payload["threshold"]), float(best_payload["decode_boundary_weight"]))

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": best_payload["state_dict"],
            "vocab": vocab,
            "labels": LABELS,
            "max_tokens": args.max_tokens,
            "embed_dim": args.embed_dim,
            "hidden_dim": args.hidden_dim,
            "threshold": float(best_payload["threshold"]),
            "decode_boundary_weight": float(best_payload["decode_boundary_weight"]),
            "boundary_loss_weight": args.boundary_loss_weight,
            "history": history,
            "best_score": best_payload["score"],
        },
        ARTIFACT_PATH,
    )
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps({"history": history, "best": best_payload["score"]}, ensure_ascii=False, indent=2), encoding="utf-8")
    VAL_PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    val_pred.to_csv(VAL_PRED_PATH, index=False, encoding="utf-8")
    print(f"Artifact saved to: {ARTIFACT_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"Validation predictions saved to: {VAL_PRED_PATH}")


if __name__ == "__main__":
    main()
