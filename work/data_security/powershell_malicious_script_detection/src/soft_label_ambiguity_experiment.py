from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
REPORT_DIR = ROOT / "reports"
CLASSES = np.array([0, 1, 2], dtype=int)
HIDDEN_COUNTS = {0: 14000, 1: 2500, 2: 3500}
SEED = 20260503


def find_data_dir() -> Path:
    candidates = [p for p in DATA_ROOT.rglob("*") if (p / "data_train.csv").exists()]
    if not candidates:
        raise FileNotFoundError(f"data_train.csv not found under {DATA_ROOT}")
    return candidates[0]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_id(name: object) -> int:
    match = re.search(r"(\d+)", str(name))
    if not match:
        raise ValueError(f"Cannot extract numeric id from {name!r}")
    return int(match.group(1))


def encode_onehot(train: pd.DataFrame, valid: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    parts_train = []
    parts_valid = []
    for col in features:
        values = sorted(set(train[col].tolist()) | set(valid[col].tolist()))
        mapping = {value: i for i, value in enumerate(values)}
        eye = np.eye(len(values), dtype=np.float32)
        parts_train.append(eye[train[col].map(mapping).to_numpy(dtype=int)])
        parts_valid.append(eye[valid[col].map(mapping).to_numpy(dtype=int)])
    return np.concatenate(parts_train, axis=1), np.concatenate(parts_valid, axis=1)


def class_sorted_positions(frame: pd.DataFrame) -> dict[int, np.ndarray]:
    y = frame["label"].to_numpy(dtype=int)
    ids = frame["_id"].to_numpy(dtype=int)
    out: dict[int, np.ndarray] = {}
    for cls in CLASSES:
        idx = np.where(y == cls)[0]
        out[int(cls)] = idx[np.argsort(ids[idx])]
    return out


def make_split(train: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    positions = class_sorted_positions(train)
    fit_parts = []
    valid_parts = []
    for cls in CLASSES:
        pos = positions[int(cls)]
        n = len(pos)
        if mode == "ratio_prefix":
            hold = int(round(n * HIDDEN_COUNTS[int(cls)] / (n + HIDDEN_COUNTS[int(cls)])))
        elif mode == "actual_prefix":
            hold = min(HIDDEN_COUNTS[int(cls)], n // 2)
        elif mode == "small_prefix":
            hold = max(600, int(round(0.12 * n)))
        elif mode == "middle_ratio":
            hold = int(round(n * HIDDEN_COUNTS[int(cls)] / (n + HIDDEN_COUNTS[int(cls)])))
            start = (n - hold) // 2
            valid_parts.append(train.iloc[pos[start : start + hold]])
            fit_parts.append(pd.concat([train.iloc[pos[:start]], train.iloc[pos[start + hold :]]]))
            continue
        else:
            raise ValueError(mode)
        valid_parts.append(train.iloc[pos[:hold]])
        fit_parts.append(train.iloc[pos[hold:]])
    return pd.concat(fit_parts, ignore_index=True), pd.concat(valid_parts, ignore_index=True)


def combo_soft_targets(frame: pd.DataFrame, features: list[str], alpha: float) -> np.ndarray:
    y = frame["label"].to_numpy(dtype=int)
    global_prob = np.bincount(y, minlength=len(CLASSES)).astype(float)
    global_prob = global_prob / global_prob.sum()
    counts: dict[tuple[int, ...], np.ndarray] = {}
    for key, label in zip(map(tuple, frame[features].to_numpy(dtype=np.int16)), y):
        if key not in counts:
            counts[key] = np.zeros(len(CLASSES), dtype=float)
        counts[key][int(label)] += 1.0
    targets = np.zeros((len(frame), len(CLASSES)), dtype=np.float32)
    for i, key in enumerate(map(tuple, frame[features].to_numpy(dtype=np.int16))):
        prob = counts[key] + alpha * global_prob
        targets[i] = (prob / prob.sum()).astype(np.float32)
    return targets


class MLP(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 192),
            nn.BatchNorm1d(192),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(192, 96),
            nn.BatchNorm1d(96),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(96, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def adjust_to_quota(scores: np.ndarray, target_counts: np.ndarray) -> np.ndarray:
    target = np.asarray(target_counts, dtype=int)
    pred = scores.argmax(axis=1).astype(int)
    counts = np.bincount(pred, minlength=len(CLASSES))
    log_scores = np.log(np.clip(scores, 1e-15, None))
    for dst in np.where(target > counts)[0]:
        need = int(target[dst] - counts[dst])
        moves: list[tuple[float, int, int]] = []
        for src in np.where(counts > target)[0]:
            rows = np.where(pred == src)[0]
            losses = log_scores[rows, src] - log_scores[rows, dst]
            moves.extend((float(loss), int(row), int(src)) for row, loss in zip(rows, losses))
        moves.sort(key=lambda x: x[0])
        changed = 0
        for _, row, src in moves:
            if changed >= need:
                break
            if pred[row] != src or counts[src] <= target[src]:
                continue
            pred[row] = dst
            counts[src] -= 1
            counts[dst] += 1
            changed += 1
    return pred


def train_eval(
    fit: pd.DataFrame,
    valid: pd.DataFrame,
    features: list[str],
    alpha: float,
    soft_weight: float,
    source_decay: float,
    dropout: float,
    seed: int,
) -> dict[str, object]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_fit, x_valid = encode_onehot(fit, valid, features)
    y_fit = fit["label"].to_numpy(dtype=np.int64)
    y_valid = valid["label"].to_numpy(dtype=np.int64)
    soft = combo_soft_targets(fit, features, alpha=alpha)
    positions = class_sorted_positions(fit)
    sample_weight = np.zeros(len(fit), dtype=np.float32)
    for cls in CLASSES:
        pos = positions[int(cls)]
        if len(pos) == 1:
            local = np.ones(1, dtype=np.float32)
        else:
            local = np.exp(-source_decay * np.linspace(0.0, 1.0, len(pos))).astype(np.float32)
        sample_weight[pos] = local
    sample_weight = sample_weight / sample_weight.mean()

    train_ds = TensorDataset(
        torch.tensor(x_fit, dtype=torch.float32),
        torch.tensor(y_fit, dtype=torch.long),
        torch.tensor(soft, dtype=torch.float32),
        torch.tensor(sample_weight, dtype=torch.float32),
    )
    loader = DataLoader(train_ds, batch_size=4096, shuffle=True, num_workers=0)
    model = MLP(x_fit.shape[1], dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best: dict[str, object] | None = None
    best_score = -1.0
    patience = 8
    bad = 0
    x_valid_t = torch.tensor(x_valid, dtype=torch.float32, device=device)
    for epoch in range(1, 81):
        model.train()
        for xb, yb, sb, wb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            sb = sb.to(device)
            wb = wb.to(device)
            logits = model(xb)
            hard_loss = nn.functional.cross_entropy(logits, yb, reduction="none")
            logp = nn.functional.log_softmax(logits, dim=1)
            soft_loss = -(sb * logp).sum(dim=1)
            loss = ((1.0 - soft_weight) * hard_loss + soft_weight * soft_loss) * wb
            opt.zero_grad(set_to_none=True)
            loss.mean().backward()
            opt.step()
        if epoch % 4 != 0:
            continue
        model.eval()
        with torch.no_grad():
            proba = torch.softmax(model(x_valid_t), dim=1).detach().cpu().numpy()
        raw = proba.argmax(axis=1)
        quota = adjust_to_quota(proba, np.bincount(y_valid, minlength=len(CLASSES)))
        raw_f1 = f1_score(y_valid, raw, average="macro")
        quota_f1 = f1_score(y_valid, quota, average="macro")
        current = max(raw_f1, quota_f1)
        if current > best_score:
            best_score = float(current)
            best = {
                "best_epoch": epoch,
                "raw_macro_f1": float(raw_f1),
                "quota_macro_f1": float(quota_f1),
                "raw_counts": {str(cls): int((raw == cls).sum()) for cls in CLASSES},
                "quota_counts": {str(cls): int((quota == cls).sum()) for cls in CLASSES},
            }
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    assert best is not None
    return best


def main() -> None:
    start = time.time()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    data_dir = find_data_dir()
    train = pd.read_csv(data_dir / "data_train.csv")
    features = [c for c in train.columns if c not in ["name", "label"]]
    train["_id"] = train["name"].map(extract_id)
    configs = [
        {"alpha": 1.0, "soft_weight": 1.0, "source_decay": 0.0, "dropout": 0.05},
        {"alpha": 1.0, "soft_weight": 0.7, "source_decay": 3.0, "dropout": 0.08},
    ]
    records = []
    for split_mode in ["ratio_prefix", "actual_prefix"]:
        fit, valid = make_split(train, split_mode)
        for i, cfg in enumerate(configs):
            result = train_eval(fit, valid, features, seed=SEED + i, **cfg)
            rec = {"split": split_mode, "config_id": i, **cfg, **result}
            records.append(rec)
            print(rec)
    frame = pd.DataFrame(records)
    avg = (
        frame.groupby(["config_id", "alpha", "soft_weight", "source_decay", "dropout"])["quota_macro_f1"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        .sort_values(["mean", "min"], ascending=False)
    )
    csv_path = REPORT_DIR / "soft_label_ambiguity_results.csv"
    frame.to_csv(csv_path, index=False, encoding="utf-8")
    summary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "runtime_seconds": round(time.time() - start, 3),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "configs": configs,
        "top_by_quota_average": json.loads(avg.to_json(orient="records", force_ascii=False)),
        "records": records,
        "result_csv": str(csv_path.relative_to(ROOT)),
        "conclusion": "Soft combo-label training tests whether hard labels inside ambiguous feature combos are the main modeling error.",
    }
    out = REPORT_DIR / "soft_label_ambiguity_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary["top_by_quota_average"], indent=2, ensure_ascii=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
