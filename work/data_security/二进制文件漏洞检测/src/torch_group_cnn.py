from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from common import NO_VULN, PROCESSED, RAW, SEED, SUBMISSIONS, ensure_dirs, load_test
from enhanced_cwe_calibrator import DOC_CACHE, GROUPS
from predict import validate_submission
from train import build_split


REPORT_VALID = PROCESSED.parents[0] / "report" / "torch_group_cnn_valid_predictions.csv"


class ByteCNN(nn.Module):
    def __init__(self, num_classes: int, embed_dim: int = 32, channels: int = 96, dropout: float = 0.25):
        super().__init__()
        self.embedding = nn.Embedding(257, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(embed_dim, channels, kernel_size=kernel, padding=kernel // 2)
                for kernel in (3, 5, 7, 11)
            ]
        )
        self.norm = nn.LayerNorm(channels * len(self.convs))
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(channels * len(self.convs), 192),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(192, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x).transpose(1, 2)
        pooled = [torch.relu(conv(emb)).amax(dim=2) for conv in self.convs]
        feat = self.norm(torch.cat(pooled, dim=1))
        return self.head(feat)


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_doc_cache() -> dict[str, dict[str, object]]:
    if not DOC_CACHE.exists():
        raise FileNotFoundError(f"missing enhanced doc cache: {DOC_CACHE}")
    loaded = joblib.load(DOC_CACHE)
    return {
        key: (value if isinstance(value, dict) else value.__dict__)
        for key, value in loaded.items()
    }


def body_tensor(binary_ids: list[str], cache: dict[str, dict[str, object]], max_len: int) -> np.ndarray:
    arr = np.zeros((len(binary_ids), max_len), dtype=np.int64)
    for row_idx, binary_id in enumerate(binary_ids):
        hex_text = str(cache[binary_id].get("body_hex", ""))
        raw = bytes.fromhex(hex_text[: max_len * 2]) if hex_text else b""
        if raw:
            values = np.frombuffer(raw, dtype=np.uint8).astype(np.int64) + 1
            arr[row_idx, : len(values)] = values
    return arr


def train_one_group(
    x: np.ndarray,
    labels: np.ndarray,
    train_idx: list[int],
    group_name: str,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> tuple[ByteCNN, dict[str, int], dict[int, str]]:
    classes = list(GROUPS[group_name])
    class_to_idx = {label: idx for idx, label in enumerate(classes)}
    idx_to_class = {idx: label for label, idx in class_to_idx.items()}
    group_train = [idx for idx in train_idx if labels[idx] in class_to_idx]
    y = np.array([class_to_idx[labels[idx]] for idx in group_train], dtype=np.int64)
    x_group = x[group_train]

    counts = np.bincount(y, minlength=len(classes)).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()

    dataset = TensorDataset(torch.from_numpy(x_group), torch.from_numpy(y))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    model = ByteCNN(len(classes)).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(epochs, 1))

    model.train()
    for _epoch in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optim.step()
        sched.step()
    return model, class_to_idx, idx_to_class


@torch.no_grad()
def predict_group(
    model: ByteCNN,
    x: np.ndarray,
    positions: list[int],
    idx_to_class: dict[int, str],
    batch_size: int,
    device: torch.device,
) -> tuple[list[str], np.ndarray]:
    if not positions:
        return [], np.array([], dtype=np.float32)
    model.eval()
    preds: list[str] = []
    confs: list[float] = []
    data = torch.from_numpy(x[positions])
    loader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=False, num_workers=0)
    for (xb,) in loader:
        logits = model(xb.to(device, non_blocking=True))
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        preds.extend(idx_to_class[int(item)] for item in pred.cpu())
        confs.extend(float(item) for item in conf.cpu())
    return preds, np.array(confs, dtype=np.float32)


def apply_predictions(
    base: pd.DataFrame,
    valid_idx: pd.Index,
    labels: np.ndarray,
    group_outputs: dict[str, tuple[list[int], list[str], np.ndarray]],
    thresholds: dict[str, float],
) -> tuple[list[str], Counter]:
    pred = base["pred"].tolist()
    stats: Counter = Counter()
    for group_name, (positions, group_pred, confs) in group_outputs.items():
        threshold = thresholds.get(group_name, float("inf"))
        for pos, new_target, conf in zip(positions, group_pred, confs, strict=False):
            if float(conf) < threshold:
                continue
            old = pred[pos]
            if new_target != old:
                true = labels[valid_idx[pos]]
                pred[pos] = new_target
                stats[f"changed_{group_name}"] += 1
                stats["useful"] += int(new_target == true and old != true)
                stats["harmful"] += int(new_target != true and old == true)
    return pred, stats


def choose_thresholds(
    base: pd.DataFrame,
    valid_idx: pd.Index,
    labels: np.ndarray,
    group_outputs: dict[str, tuple[list[int], list[str], np.ndarray]],
) -> tuple[dict[str, float], list[str], float, Counter]:
    y_true = base["true"].to_numpy()
    current = base["pred"].tolist()
    best_score = f1_score(y_true, current, average="macro")
    chosen: dict[str, float] = {}
    grid = [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.99]
    for group_name in group_outputs:
        local_best = best_score
        local_threshold = float("inf")
        local_pred = current
        for threshold in grid:
            trial_df = base.copy()
            trial_df["pred"] = current
            trial_pred, _stats = apply_predictions(trial_df, valid_idx, labels, {group_name: group_outputs[group_name]}, {group_name: threshold})
            score = f1_score(y_true, trial_pred, average="macro")
            if score > local_best:
                local_best = score
                local_threshold = threshold
                local_pred = trial_pred
        chosen[group_name] = local_threshold
        current = local_pred
        best_score = local_best
    final_df = base.copy()
    final_df["pred"] = current
    _pred, stats = apply_predictions(base, valid_idx, labels, group_outputs, chosen)
    return chosen, current, best_score, stats


def evaluate(max_len: int, epochs: int, batch_size: int, lr: float) -> None:
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = pd.read_csv(RAW / "train.csv")
    train["target"] = train["cwe_id"].fillna(NO_VULN)
    train_idx, valid_idx = build_split(train, 0.2)
    labels = train["target"].to_numpy()
    cache = load_doc_cache()
    x = body_tensor(train["binary_id"].tolist(), cache, max_len)
    base = pd.read_csv(PROCESSED.parents[0] / "report" / "enhanced_calibrator_valid_predictions.csv")
    base_macro = f1_score(base["true"], base["pred"], average="macro")
    print(f"device={device}")
    print(f"base_macro={base_macro:.6f}")

    group_outputs: dict[str, tuple[list[int], list[str], np.ndarray]] = {}
    for group_name, classes in GROUPS.items():
        model, _class_to_idx, idx_to_class = train_one_group(
            x,
            labels,
            list(train_idx),
            group_name,
            epochs,
            batch_size,
            lr,
            device,
        )
        group = set(classes)
        positions = [pos for pos, value in enumerate(base["pred"]) if value in group]
        sample_idx = [valid_idx[pos] for pos in positions]
        group_pred, confs = predict_group(model, x, sample_idx, idx_to_class, batch_size, device)
        group_outputs[group_name] = (positions, group_pred, confs)
        print(f"group={group_name} candidates={len(positions)} mean_conf={float(confs.mean()) if len(confs) else 0:.4f}")

    thresholds, final_pred, score, stats = choose_thresholds(base, valid_idx, labels, group_outputs)
    print(f"thresholds={thresholds}")
    print(f"torch_group_macro={score:.6f}")
    print(f"stats={dict(stats)}")
    out = base.copy()
    out["pred"] = final_pred
    out.to_csv(REPORT_VALID, index=False, encoding="utf-8")
    errors = out[out["true"] != out["pred"]]
    print("error_pairs")
    print(errors.groupby(["true", "pred"]).size().sort_values(ascending=False).head(50).to_string())
    print(REPORT_VALID)


def parse_thresholds(value: str) -> dict[str, float]:
    if not value:
        return {}
    result = {}
    for item in value.split(","):
        key, raw = item.split("=", 1)
        parsed = float(raw)
        if math.isfinite(parsed):
            result[key.strip()] = parsed
    return result


def generate(max_len: int, epochs: int, batch_size: int, lr: float, thresholds: dict[str, float], input_name: str, output: str) -> None:
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = pd.read_csv(RAW / "train.csv")
    train["target"] = train["cwe_id"].fillna(NO_VULN)
    test = load_test()
    cache = load_doc_cache()
    train_x = body_tensor(train["binary_id"].tolist(), cache, max_len)
    test_x = body_tensor(test["binary_id"].tolist(), cache, max_len)
    labels = train["target"].to_numpy()
    train_idx = list(train.index[labels != NO_VULN])

    sub = pd.read_csv(SUBMISSIONS / input_name)
    if list(sub.columns) == ["name", "label", "cwe_id"]:
        sub = sub.rename(columns={"name": "binary_id"})
    final_targets = [NO_VULN if row.label == 0 else row.cwe_id for row in sub.itertuples(index=False)]
    stats: Counter = Counter()

    for group_name, classes in GROUPS.items():
        threshold = thresholds.get(group_name, float("inf"))
        if not math.isfinite(threshold):
            continue
        model, _class_to_idx, idx_to_class = train_one_group(
            train_x,
            labels,
            train_idx,
            group_name,
            epochs,
            batch_size,
            lr,
            device,
        )
        group = set(classes)
        positions = [pos for pos, value in enumerate(final_targets) if value in group]
        group_pred, confs = predict_group(model, test_x, positions, idx_to_class, batch_size, device)
        for pos, new_target, conf in zip(positions, group_pred, confs, strict=False):
            if float(conf) >= threshold and new_target != final_targets[pos]:
                final_targets[pos] = new_target
                stats[f"changed_{group_name}"] += 1

    out = pd.DataFrame(
        {
            "binary_id": test["binary_id"],
            "label": [0 if target == NO_VULN else 1 for target in final_targets],
            "cwe_id": ["" if target == NO_VULN else target for target in final_targets],
        }
    )
    validate_submission(out, test["binary_id"])
    output_path = SUBMISSIONS / output
    out.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(output_path)
    print(f"stats={dict(stats)}")
    print(f"label_dist={out['label'].value_counts().to_dict()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Byte-level PyTorch CNN calibrator for confused CWE groups.")
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--thresholds", default="")
    parser.add_argument("--input", default="submission_enhanced_calibrator_v5_utf8_sig.csv")
    parser.add_argument("--output", default="submission_torch_group_cnn_v6_utf8_sig.csv")
    args = parser.parse_args()
    ensure_dirs()
    if args.eval:
        evaluate(args.max_len, args.epochs, args.batch_size, args.lr)
    if args.generate:
        generate(
            args.max_len,
            args.epochs,
            args.batch_size,
            args.lr,
            parse_thresholds(args.thresholds),
            args.input,
            args.output,
        )


if __name__ == "__main__":
    main()
