from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, Dataset

from common import ANOMALY_TYPES, ROOT, TRAIN_PATH, compute_score, empty_prediction, normalize_line


LABELS = ["none"] + ANOMALY_TYPES
LABEL_TO_ID = {name: idx for idx, name in enumerate(LABELS)}
ID_TO_LABEL = {idx: name for name, idx in LABEL_TO_ID.items()}
TOKEN_PATTERN = re.compile(r"<addr>|<path>|[a-z_]+|num|[=*:/().,-]+")

ARTIFACT_PATH = ROOT / "models" / "torch_sequence_model.pt"
METRICS_PATH = ROOT / "logs" / "torch_sequence_metrics.json"
VAL_PRED_PATH = ROOT / "processed" / "torch_sequence_val_predictions.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PyTorch line-sequence tagger.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-vocab", type=int, default=50000)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--embed-dim", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-path", type=str, default=str(TRAIN_PATH))
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tokenize(line: str) -> list[str]:
    return TOKEN_PATTERN.findall(normalize_line(line))


def parse_labels(row: object, line_count: int) -> list[int]:
    labels = [LABEL_TO_ID["none"]] * line_count
    if int(row.has_anomaly) == 1:
        label = LABEL_TO_ID[str(row.primary_anomaly_type)]
        for idx in range(int(row.primary_start_idx), int(row.primary_end_idx) + 1):
            labels[idx] = label
    return labels


def build_vocab(train_rows: pd.DataFrame, max_vocab: int) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for text in train_rows["log_text"]:
        for line in str(text).split("\n"):
            counter.update(tokenize(line))
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, _ in counter.most_common(max_vocab - len(vocab)):
        vocab[token] = len(vocab)
    return vocab


def encode_line(line: str, vocab: dict[str, int], max_tokens: int) -> list[int]:
    ids = [vocab.get(token, 1) for token in tokenize(line)[:max_tokens]]
    return ids or [1]


def encode_frame(df: pd.DataFrame, vocab: dict[str, int], max_tokens: int, include_labels: bool) -> list[dict[str, object]]:
    encoded: list[dict[str, object]] = []
    for row in df.itertuples(index=False):
        lines = str(row.log_text).split("\n")
        item = {
            "id": int(row.id),
            "tokens": [encode_line(line, vocab, max_tokens) for line in lines],
            "line_count": len(lines),
        }
        if include_labels:
            item["labels"] = parse_labels(row, len(lines))
        encoded.append(item)
    return encoded


class LogDataset(Dataset):
    def __init__(self, items: list[dict[str, object]]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, object]:
        return self.items[idx]


def collate_batch(batch: list[dict[str, object]]) -> dict[str, torch.Tensor | list[int]]:
    max_lines = max(int(item["line_count"]) for item in batch)
    max_tokens = max(max(len(line) for line in item["tokens"]) for item in batch)
    token_tensor = torch.zeros((len(batch), max_lines, max_tokens), dtype=torch.long)
    token_mask = torch.zeros((len(batch), max_lines, max_tokens), dtype=torch.float32)
    line_mask = torch.zeros((len(batch), max_lines), dtype=torch.bool)
    label_tensor = torch.full((len(batch), max_lines), -100, dtype=torch.long)
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

    return {
        "ids": ids,
        "tokens": token_tensor,
        "token_mask": token_mask,
        "line_mask": line_mask,
        "labels": label_tensor,
    }


class SequenceTagger(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, label_count: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(0.20)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True, dropout=0.15)
        self.classifier = nn.Linear(hidden_dim * 2, label_count)

    def forward(self, tokens: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(tokens)
        denom = token_mask.sum(dim=2).clamp_min(1.0).unsqueeze(-1)
        line_emb = (emb * token_mask.unsqueeze(-1)).sum(dim=2) / denom
        line_emb = self.dropout(line_emb)
        seq_out, _ = self.gru(line_emb)
        return self.classifier(self.dropout(seq_out))


def class_weights(items: list[dict[str, object]], device: torch.device) -> torch.Tensor:
    counts = Counter()
    for item in items:
        counts.update(item["labels"])
    weights = []
    for idx in range(len(LABELS)):
        value = 1.0 / math.sqrt(counts[idx] + 1)
        weights.append(value)
    weights = np.array(weights, dtype=np.float32)
    weights = weights / weights.mean()
    weights[0] = min(weights[0], 0.20)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device) -> float:
    model.train()
    losses: list[float] = []
    for batch in loader:
        tokens = batch["tokens"].to(device)
        token_mask = batch["token_mask"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(tokens, token_mask)
        loss = criterion(logits.view(-1, len(LABELS)), labels.view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


@torch.no_grad()
def predict_logits(model: nn.Module, items: list[dict[str, object]], batch_size: int, device: torch.device) -> dict[int, np.ndarray]:
    model.eval()
    loader = DataLoader(LogDataset(items), batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    out: dict[int, np.ndarray] = {}
    for batch in loader:
        logits = model(batch["tokens"].to(device), batch["token_mask"].to(device)).detach().cpu().numpy()
        line_mask = batch["line_mask"].numpy()
        for bidx, row_id in enumerate(batch["ids"]):
            out[int(row_id)] = logits[bidx, line_mask[bidx]]
    return out


def decode_row(row_id: int, logits: np.ndarray, threshold: float) -> dict[str, object]:
    if len(logits) == 0:
        return empty_prediction(row_id)
    log_probs = logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)
    none_scores = log_probs[:, 0]
    best: tuple[float, int, int, str] | None = None
    line_count = len(logits)
    for label_id, label_name in ID_TO_LABEL.items():
        if label_id == 0:
            continue
        diff = log_probs[:, label_id] - none_scores
        prefix = np.concatenate([[0.0], np.cumsum(diff)])
        for span_len in range(3, 11):
            if span_len > line_count:
                break
            scores = prefix[span_len:] - prefix[:-span_len]
            start = int(np.argmax(scores))
            score = float(scores[start])
            candidate = (score, -start, start + span_len - 1, label_name)
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


def decode_frame(df: pd.DataFrame, logits_by_id: dict[int, np.ndarray], threshold: float) -> pd.DataFrame:
    return pd.DataFrame([decode_row(int(row.id), logits_by_id[int(row.id)], threshold) for row in df.itertuples(index=False)])


def tune_threshold(valid_df: pd.DataFrame, logits_by_id: dict[int, np.ndarray]) -> tuple[float, dict[str, float]]:
    thresholds = np.linspace(-2.0, 8.0, 41)
    best_threshold = float(thresholds[0])
    best_score: dict[str, float] | None = None
    for threshold in thresholds:
        pred = decode_frame(valid_df, logits_by_id, float(threshold))
        score = compute_score(valid_df, pred)
        if best_score is None or score["final_score"] > best_score["final_score"]:
            best_score = score
            best_threshold = float(threshold)
    assert best_score is not None
    return best_threshold, best_score


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    train_df = pd.read_csv(args.train_path)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=args.seed)
    train_idx, valid_idx = next(splitter.split(train_df, train_df["primary_anomaly_type"]))
    fit_df = train_df.iloc[train_idx].reset_index(drop=True)
    valid_df = train_df.iloc[valid_idx].reset_index(drop=True)

    vocab = build_vocab(fit_df, args.max_vocab)
    fit_items = encode_frame(fit_df, vocab, args.max_tokens, include_labels=True)
    valid_items = encode_frame(valid_df, vocab, args.max_tokens, include_labels=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SequenceTagger(len(vocab), args.embed_dim, args.hidden_dim, len(LABELS)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(fit_items, device), ignore_index=-100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    loader = DataLoader(LogDataset(fit_items), batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch, num_workers=0)
    history: list[dict[str, float]] = []
    best_payload: dict[str, object] | None = None

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, loader, optimizer, criterion, device)
        logits_by_id = predict_logits(model, valid_items, args.batch_size, device)
        threshold, score = tune_threshold(valid_df, logits_by_id)
        history.append({"epoch": epoch, "loss": loss, "threshold": threshold, **score})
        print(json.dumps(history[-1], ensure_ascii=False), flush=True)
        if best_payload is None or score["final_score"] > best_payload["score"]["final_score"]:
            best_payload = {
                "epoch": epoch,
                "threshold": threshold,
                "score": score,
                "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            }

    assert best_payload is not None
    model.load_state_dict(best_payload["state_dict"])
    logits_by_id = predict_logits(model, valid_items, args.batch_size, device)
    val_pred = decode_frame(valid_df, logits_by_id, float(best_payload["threshold"]))

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
            "history": history,
            "best_score": best_payload["score"],
        },
        ARTIFACT_PATH,
    )
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps({"history": history, "best": best_payload["score"], "threshold": best_payload["threshold"]}, ensure_ascii=False, indent=2), encoding="utf-8")
    VAL_PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    val_pred.to_csv(VAL_PRED_PATH, index=False, encoding="utf-8")

    print(f"Artifact saved to: {ARTIFACT_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"Validation predictions saved to: {VAL_PRED_PATH}")


if __name__ == "__main__":
    main()
