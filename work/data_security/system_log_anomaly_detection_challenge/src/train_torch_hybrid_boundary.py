from __future__ import annotations

import argparse
import json
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

from common import ROOT, TEST_PATH, TRAIN_PATH, compute_score, empty_prediction, normalize_line
from train_torch_boundary import decode_frame, masked_boundary_loss, tune_decode
from train_torch_sequence import LABELS, LABEL_TO_ID, class_weights, set_seed


ARTIFACT_PATH = ROOT / "models" / "torch_hybrid_boundary_model.pt"
METRICS_PATH = ROOT / "logs" / "torch_hybrid_boundary_metrics.json"
VAL_PRED_PATH = ROOT / "processed" / "torch_hybrid_boundary_val_predictions.csv"

TOKEN_PATTERN = re.compile(r"<addr>|<path>|[a-z_]+|num|[=*:/().,-]+")
ALPHA_PATTERN = re.compile(r"^[a-z_]{4,}$")
PROTECTED_TOKENS = {"<addr>", "<path>", "num", "seg", "id"}

AUGMENT_REPLACEMENTS: dict[str, list[str]] = {
    "accepted": ["approved", "acceptde", "accetped"],
    "accepting": ["acceptign", "accetping"],
    "action": ["aciton"],
    "anomaly": ["anomlay", "aonmaly", "anoamly"],
    "arrived": ["arrievd"],
    "attempted": ["attemptde"],
    "backward": ["backawrd", "bakcward"],
    "baseline": ["basleine"],
    "because": ["beacuse"],
    "before": ["befoer"],
    "blockinfo": ["blockifno"],
    "compatible": ["compaitble", "comaptible"],
    "confirmed": ["confirmde", "conifrmed"],
    "contract": ["conrtact", "cotnract", "contratc"],
    "delivery": ["delivrey"],
    "despite": ["desptie"],
    "detected": ["detetced", "dteected"],
    "differed": ["difefred"],
    "drift": ["dirft", "shift"],
    "entered": ["enetred"],
    "envelope": ["enveloep"],
    "escalation": ["ecsalation", "esaclation"],
    "exceeded": ["ecxeeded"],
    "exhausted": ["exhautsed", "exhausetd", "ehxausted"],
    "exposed": ["exopsed", "epxosed"],
    "failure": ["faiulre", "failrue"],
    "field": ["fiedl", "filed"],
    "handoff": ["hadnoff"],
    "immediate": ["immeidate", "imemdiate"],
    "integrity": ["integriyt", "integirty"],
    "interval": ["interavl"],
    "invalid": ["invlaid", "ivnalid"],
    "latency": ["latenyc", "latecny"],
    "local": ["loacl"],
    "metadata": ["metdaata"],
    "outside": ["outisde", "oustide"],
    "parameter": ["paraemter", "parmaeter"],
    "placement": ["plaecment"],
    "precursor": ["precrusor", "prceursor"],
    "profile": ["proifle", "proflie"],
    "raised": ["raisde"],
    "reconcile": ["reocncile", "recnocile"],
    "recovery": ["recoevry"],
    "rejected": ["rejetced"],
    "remained": ["remanied", "remaiend"],
    "requests": ["requetss"],
    "required": ["reuqired"],
    "response": ["resopnse", "resposne", "rseponse"],
    "restored": ["restoerd"],
    "retry": ["re-attempt", "rerty", "retrie"],
    "rising": ["rsiing", "risign"],
    "rollback": ["rlolback"],
    "runtime": ["rutnime"],
    "scheduled": ["schedulde", "scehduled"],
    "schema": ["schmea"],
    "sealed": ["selaed"],
    "sequence": ["sequenec"],
    "serve": ["sreve"],
    "slower": ["solwer"],
    "stabilization": ["stbailization", "stabliization"],
    "stage": ["staeg", "stgae"],
    "stream": ["strema"],
    "sustained": ["ssutained", "sustanied"],
    "transition": ["trnasition"],
    "trying": ["trynig", "triyng"],
    "unexpectedly": ["unexpectedyl"],
    "unresolved": ["urnesolved", "unresolevd"],
    "upstream": ["upsteram"],
    "validated": ["verified", "vreified", "verfiied"],
    "verification": ["verificaiton", "verifciation", "verifictaion"],
    "verify": ["validation", "validated", "verified", "veirfy"],
    "within": ["withni", "witihn", "wtihin", "wihtin"],
    "write": ["wirte"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a char+word boundary-aware PyTorch sequence tagger.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--max-vocab", type=int, default=60000)
    parser.add_argument("--max-tokens", type=int, default=56)
    parser.add_argument("--max-chars", type=int, default=220)
    parser.add_argument("--word-dim", type=int, default=96)
    parser.add_argument("--char-dim", type=int, default=32)
    parser.add_argument("--char-channels", type=int, default=48)
    parser.add_argument("--line-dim", type=int, default=192)
    parser.add_argument("--hidden-dim", type=int, default=144)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--boundary-loss-weight", type=float, default=0.40)
    parser.add_argument("--threshold", type=float, default=-4.0)
    parser.add_argument("--decode-boundary-weight", type=float, default=1.2)
    parser.add_argument("--augment-copies", type=int, default=1)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--train-path", type=str, default=str(TRAIN_PATH))
    parser.add_argument("--test-path", type=str, default=str(TEST_PATH))
    parser.add_argument("--artifact-path", type=str, default=str(ARTIFACT_PATH))
    parser.add_argument("--metrics-path", type=str, default=str(METRICS_PATH))
    return parser.parse_args()


def tokenize_model(line: str) -> list[str]:
    tokens = TOKEN_PATTERN.findall(normalize_line(line))
    return tokens or ["<unk>"]


def parse_labels(row: object, line_count: int) -> list[int]:
    labels = [LABEL_TO_ID["none"]] * line_count
    if int(row.has_anomaly) == 1:
        label = LABEL_TO_ID[str(row.primary_anomaly_type)]
        for idx in range(int(row.primary_start_idx), int(row.primary_end_idx) + 1):
            labels[idx] = label
    return labels


def boundary_targets(labels: list[int]) -> tuple[int, int]:
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
    return start, end


def all_extra_tokens() -> set[str]:
    extras = {"<mask>", "<unk>"}
    for source, replacements in AUGMENT_REPLACEMENTS.items():
        extras.add(source)
        extras.update(replacements)
    return extras


def build_word_vocab(train_rows: pd.DataFrame, max_vocab: int) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for text in train_rows["log_text"]:
        for line in str(text).split("\n"):
            counter.update(tokenize_model(line))
    vocab = {"<pad>": 0, "<unk>": 1, "<mask>": 2}
    for token in sorted(all_extra_tokens()):
        if token not in vocab:
            vocab[token] = len(vocab)
    for token, _ in counter.most_common(max_vocab - len(vocab)):
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def build_char_vocab(train_rows: pd.DataFrame, test_rows: pd.DataFrame | None = None) -> dict[str, int]:
    chars: set[str] = set("abcdefghijklmnopqrstuvwxyz0123456789_<>:/().,-=* ")
    frames = [train_rows]
    if test_rows is not None:
        frames.append(test_rows)
    for frame in frames:
        for text in frame["log_text"]:
            for line in str(text).split("\n"):
                chars.update(" ".join(tokenize_model(line)))
    for token in all_extra_tokens():
        chars.update(token)
    vocab = {"<pad>": 0, "<unk>": 1}
    for char in sorted(chars):
        if char not in vocab:
            vocab[char] = len(vocab)
    return vocab


def char_noise(token: str, rng: random.Random) -> str:
    if len(token) < 5:
        return token
    chars = list(token)
    operation = rng.choice(["swap", "delete", "duplicate"])
    idx = rng.randrange(len(chars))
    if operation == "swap" and idx < len(chars) - 1:
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
    elif operation == "delete" and len(chars) > 5:
        del chars[idx]
    elif operation == "duplicate":
        chars.insert(idx, chars[idx])
    return "".join(chars) or token


def augment_tokens(tokens: list[str], rng: random.Random, aggressive: bool) -> list[str]:
    replacement_prob = 0.36 if aggressive else 0.08
    noise_prob = 0.060 if aggressive else 0.015
    mask_prob = 0.030 if aggressive else 0.006
    augmented: list[str] = []
    for token in tokens:
        if token in PROTECTED_TOKENS or not ALPHA_PATTERN.match(token):
            augmented.append(token)
            continue
        if rng.random() < mask_prob:
            augmented.append("<mask>")
            continue
        if token in AUGMENT_REPLACEMENTS and rng.random() < replacement_prob:
            token = rng.choice(AUGMENT_REPLACEMENTS[token])
        if ALPHA_PATTERN.match(token) and rng.random() < noise_prob:
            token = char_noise(token, rng)
        augmented.append(token)
    return augmented


def encode_tokens(tokens: list[str], word_vocab: dict[str, int], max_tokens: int) -> list[int]:
    ids = [word_vocab.get(token, 1) for token in tokens[:max_tokens]]
    return ids or [1]


def encode_chars(tokens: list[str], char_vocab: dict[str, int], max_chars: int) -> list[int]:
    text = " ".join(tokens)[:max_chars]
    ids = [char_vocab.get(char, 1) for char in text]
    return ids or [1]


def make_item(
    row_id: int,
    line_tokens: list[list[str]],
    labels: list[int] | None,
    word_vocab: dict[str, int],
    char_vocab: dict[str, int],
    max_tokens: int,
    max_chars: int,
) -> dict[str, object]:
    item: dict[str, object] = {
        "id": row_id,
        "word_tokens": [encode_tokens(tokens, word_vocab, max_tokens) for tokens in line_tokens],
        "char_tokens": [encode_chars(tokens, char_vocab, max_chars) for tokens in line_tokens],
        "line_count": len(line_tokens),
    }
    if labels is not None:
        start, end = boundary_targets(labels)
        item["labels"] = labels
        item["start_idx"] = start
        item["end_idx"] = end
    return item


def encode_frame_hybrid(
    df: pd.DataFrame,
    word_vocab: dict[str, int],
    char_vocab: dict[str, int],
    max_tokens: int,
    max_chars: int,
    include_labels: bool,
    augment_copies: int,
    seed: int,
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    items: list[dict[str, object]] = []
    for row in df.itertuples(index=False):
        lines = str(row.log_text).split("\n")
        line_tokens = [tokenize_model(line) for line in lines]
        labels = parse_labels(row, len(lines)) if include_labels else None
        items.append(make_item(int(row.id), line_tokens, labels, word_vocab, char_vocab, max_tokens, max_chars))
        if not include_labels or int(row.has_anomaly) != 1:
            continue
        assert labels is not None
        for copy_idx in range(augment_copies):
            augmented_lines: list[list[str]] = []
            for line_idx, tokens in enumerate(line_tokens):
                aggressive = labels[line_idx] != 0 or rng.random() < 0.18
                augmented_lines.append(augment_tokens(tokens, rng, aggressive))
            aug_id = int(row.id) + (copy_idx + 1) * 10_000_000
            items.append(make_item(aug_id, augmented_lines, labels, word_vocab, char_vocab, max_tokens, max_chars))
    return items


class HybridBoundaryDataset(Dataset):
    def __init__(self, items: list[dict[str, object]]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, object]:
        return self.items[idx]


def collate_hybrid(batch: list[dict[str, object]]) -> dict[str, torch.Tensor | list[int]]:
    max_lines = max(int(item["line_count"]) for item in batch)
    max_words = max(max(len(line) for line in item["word_tokens"]) for item in batch)
    max_chars = max(max(len(line) for line in item["char_tokens"]) for item in batch)
    word_tensor = torch.zeros((len(batch), max_lines, max_words), dtype=torch.long)
    word_mask = torch.zeros((len(batch), max_lines, max_words), dtype=torch.float32)
    char_tensor = torch.zeros((len(batch), max_lines, max_chars), dtype=torch.long)
    char_mask = torch.zeros((len(batch), max_lines, max_chars), dtype=torch.float32)
    line_mask = torch.zeros((len(batch), max_lines), dtype=torch.bool)
    label_tensor = torch.full((len(batch), max_lines), -100, dtype=torch.long)
    start_targets = torch.full((len(batch),), -100, dtype=torch.long)
    end_targets = torch.full((len(batch),), -100, dtype=torch.long)
    ids: list[int] = []

    for bidx, item in enumerate(batch):
        ids.append(int(item["id"]))
        for lidx, token_ids in enumerate(item["word_tokens"]):
            word_tensor[bidx, lidx, : len(token_ids)] = torch.tensor(token_ids, dtype=torch.long)
            word_mask[bidx, lidx, : len(token_ids)] = 1.0
            line_mask[bidx, lidx] = True
        for lidx, char_ids in enumerate(item["char_tokens"]):
            char_tensor[bidx, lidx, : len(char_ids)] = torch.tensor(char_ids, dtype=torch.long)
            char_mask[bidx, lidx, : len(char_ids)] = 1.0
        if "labels" in item:
            labels = item["labels"]
            label_tensor[bidx, : len(labels)] = torch.tensor(labels, dtype=torch.long)
            if int(item["start_idx"]) >= 0:
                start_targets[bidx] = int(item["start_idx"])
                end_targets[bidx] = int(item["end_idx"])

    return {
        "ids": ids,
        "word_tokens": word_tensor,
        "word_mask": word_mask,
        "char_tokens": char_tensor,
        "char_mask": char_mask,
        "line_mask": line_mask,
        "labels": label_tensor,
        "start_targets": start_targets,
        "end_targets": end_targets,
    }


class HybridBoundaryTagger(nn.Module):
    def __init__(
        self,
        word_vocab_size: int,
        char_vocab_size: int,
        word_dim: int,
        char_dim: int,
        char_channels: int,
        line_dim: int,
        hidden_dim: int,
        label_count: int,
    ) -> None:
        super().__init__()
        self.word_embedding = nn.Embedding(word_vocab_size, word_dim, padding_idx=0)
        self.char_embedding = nn.Embedding(char_vocab_size, char_dim, padding_idx=0)
        self.char_convs = nn.ModuleList(
            [
                nn.Conv1d(char_dim, char_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
                for kernel_size in (3, 5, 7)
            ]
        )
        self.feature_projection = nn.Sequential(
            nn.Linear(word_dim + char_channels * len(self.char_convs), line_dim),
            nn.LayerNorm(line_dim),
            nn.GELU(),
            nn.Dropout(0.20),
        )
        self.gru = nn.GRU(line_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True, dropout=0.18)
        self.dropout = nn.Dropout(0.22)
        self.classifier = nn.Linear(hidden_dim * 2, label_count)
        self.start_head = nn.Linear(hidden_dim * 2, 1)
        self.end_head = nn.Linear(hidden_dim * 2, 1)

    def forward(
        self,
        word_tokens: torch.Tensor,
        word_mask: torch.Tensor,
        char_tokens: torch.Tensor,
        char_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        word_emb = self.word_embedding(word_tokens)
        word_denom = word_mask.sum(dim=2).clamp_min(1.0).unsqueeze(-1)
        word_features = (word_emb * word_mask.unsqueeze(-1)).sum(dim=2) / word_denom

        batch_size, line_count, char_count = char_tokens.shape
        flat_chars = char_tokens.reshape(batch_size * line_count, char_count)
        flat_char_mask = char_mask.reshape(batch_size * line_count, char_count)
        char_emb = self.char_embedding(flat_chars).transpose(1, 2)
        conv_features = []
        for conv in self.char_convs:
            values = torch.relu(conv(char_emb)).masked_fill(flat_char_mask.unsqueeze(1) == 0, -1e4)
            conv_features.append(values.max(dim=2).values)
        char_features = torch.cat(conv_features, dim=1).reshape(batch_size, line_count, -1)

        line_features = self.feature_projection(torch.cat([word_features, char_features], dim=2))
        seq_out, _ = self.gru(line_features)
        seq_out = self.dropout(seq_out)
        return self.classifier(seq_out), self.start_head(seq_out).squeeze(-1), self.end_head(seq_out).squeeze(-1)


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
        word_tokens = batch["word_tokens"].to(device)
        word_mask = batch["word_mask"].to(device)
        char_tokens = batch["char_tokens"].to(device)
        char_mask = batch["char_mask"].to(device)
        labels = batch["labels"].to(device)
        line_mask = batch["line_mask"].to(device)
        start_targets = batch["start_targets"].to(device)
        end_targets = batch["end_targets"].to(device)
        optimizer.zero_grad(set_to_none=True)
        line_logits, start_logits, end_logits = model(word_tokens, word_mask, char_tokens, char_mask)
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
def predict_outputs(
    model: nn.Module,
    items: list[dict[str, object]],
    batch_size: int,
    device: torch.device,
) -> dict[int, dict[str, np.ndarray]]:
    model.eval()
    loader = DataLoader(HybridBoundaryDataset(items), batch_size=batch_size, shuffle=False, collate_fn=collate_hybrid)
    outputs: dict[int, dict[str, np.ndarray]] = {}
    for batch in loader:
        line_logits, start_logits, end_logits = model(
            batch["word_tokens"].to(device),
            batch["word_mask"].to(device),
            batch["char_tokens"].to(device),
            batch["char_mask"].to(device),
        )
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


def build_model(args: argparse.Namespace, word_vocab: dict[str, int], char_vocab: dict[str, int], device: torch.device) -> HybridBoundaryTagger:
    return HybridBoundaryTagger(
        word_vocab_size=len(word_vocab),
        char_vocab_size=len(char_vocab),
        word_dim=args.word_dim,
        char_dim=args.char_dim,
        char_channels=args.char_channels,
        line_dim=args.line_dim,
        hidden_dim=args.hidden_dim,
        label_count=len(LABELS),
    ).to(device)


def save_artifact(
    args: argparse.Namespace,
    model: nn.Module,
    word_vocab: dict[str, int],
    char_vocab: dict[str, int],
    history: list[dict[str, object]],
    output_path: Path,
    best_score: dict[str, float] | None = None,
    threshold: float | None = None,
    decode_boundary_weight: float | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": "hybrid_char_word_boundary",
            "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "word_vocab": word_vocab,
            "char_vocab": char_vocab,
            "labels": LABELS,
            "max_tokens": args.max_tokens,
            "max_chars": args.max_chars,
            "word_dim": args.word_dim,
            "char_dim": args.char_dim,
            "char_channels": args.char_channels,
            "line_dim": args.line_dim,
            "hidden_dim": args.hidden_dim,
            "threshold": float(args.threshold if threshold is None else threshold),
            "decode_boundary_weight": float(args.decode_boundary_weight if decode_boundary_weight is None else decode_boundary_weight),
            "boundary_loss_weight": float(args.boundary_loss_weight),
            "augment_copies": int(args.augment_copies),
            "history": history,
            "best_score": best_score,
        },
        output_path,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path) if args.full and Path(args.test_path).exists() else None

    if args.full:
        fit_df = train_df.reset_index(drop=True)
        valid_df = None
    else:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=args.seed)
        train_idx, valid_idx = next(splitter.split(train_df, train_df["primary_anomaly_type"]))
        fit_df = train_df.iloc[train_idx].reset_index(drop=True)
        valid_df = train_df.iloc[valid_idx].reset_index(drop=True)

    word_vocab = build_word_vocab(fit_df, args.max_vocab)
    char_vocab = build_char_vocab(fit_df, test_df if args.full else None)
    fit_items = encode_frame_hybrid(
        fit_df,
        word_vocab,
        char_vocab,
        args.max_tokens,
        args.max_chars,
        include_labels=True,
        augment_copies=args.augment_copies,
        seed=args.seed + 1000,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args, word_vocab, char_vocab, device)
    line_criterion = nn.CrossEntropyLoss(weight=class_weights(fit_items, device), ignore_index=-100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.6e-3, weight_decay=1.2e-4)
    loader = DataLoader(
        HybridBoundaryDataset(fit_items),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_hybrid,
        num_workers=0,
    )

    history: list[dict[str, object]] = []
    best_payload: dict[str, object] | None = None
    valid_items: list[dict[str, object]] | None = None
    if valid_df is not None:
        valid_items = encode_frame_hybrid(
            valid_df,
            word_vocab,
            char_vocab,
            args.max_tokens,
            args.max_chars,
            include_labels=True,
            augment_copies=0,
            seed=args.seed + 2000,
        )

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, loader, optimizer, line_criterion, args.boundary_loss_weight, device)
        record: dict[str, object] = {"epoch": epoch, "loss": float(loss)}
        if valid_df is not None and valid_items is not None:
            outputs = predict_outputs(model, valid_items, args.batch_size, device)
            threshold, decode_boundary_weight, score = tune_decode(valid_df, outputs)
            record.update({"threshold": threshold, "decode_boundary_weight": decode_boundary_weight, **score})
            if best_payload is None or score["final_score"] > best_payload["score"]["final_score"]:
                best_payload = {
                    "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    "threshold": threshold,
                    "decode_boundary_weight": decode_boundary_weight,
                    "score": score,
                }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    artifact_path = Path(args.artifact_path)
    metrics_path = Path(args.metrics_path)
    if best_payload is not None:
        model.load_state_dict(best_payload["state_dict"])
        assert valid_df is not None and valid_items is not None
        outputs = predict_outputs(model, valid_items, args.batch_size, device)
        val_pred = decode_frame(valid_df, outputs, float(best_payload["threshold"]), float(best_payload["decode_boundary_weight"]))
        VAL_PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
        val_pred.to_csv(VAL_PRED_PATH, index=False, encoding="utf-8")
        save_artifact(
            args,
            model,
            word_vocab,
            char_vocab,
            history,
            artifact_path,
            best_score=best_payload["score"],
            threshold=float(best_payload["threshold"]),
            decode_boundary_weight=float(best_payload["decode_boundary_weight"]),
        )
        metrics = {"history": history, "best": best_payload["score"]}
    else:
        save_artifact(args, model, word_vocab, char_vocab, history, artifact_path)
        metrics = {
            "history": history,
            "threshold": args.threshold,
            "decode_boundary_weight": args.decode_boundary_weight,
            "mean_train_loss": float(np.mean([float(item["loss"]) for item in history])),
        }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Artifact saved to: {artifact_path}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
