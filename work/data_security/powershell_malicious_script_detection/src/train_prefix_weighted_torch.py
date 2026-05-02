from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
MODEL_DIR = ROOT / "models"
SUBMISSION_DIR = ROOT / "submissions"
REPORT_DIR = ROOT / "reports"
SEED = 20260501
CLASSES = np.array([0, 1, 2], dtype=int)


def find_data_dir() -> Path:
    candidates = [p for p in DATA_ROOT.rglob("*") if (p / "data_train.csv").exists()]
    if not candidates:
        raise FileNotFoundError(f"data_train.csv not found under {DATA_ROOT}")
    return candidates[0]


def validate_submission(test: pd.DataFrame, submission: pd.DataFrame) -> None:
    if list(submission.columns) != ["name", "label"]:
        raise ValueError("Submission must have columns: name,label")
    if len(submission) != len(test):
        raise ValueError("Submission row count does not match test row count")
    if not submission["name"].equals(test["name"]):
        raise ValueError("Submission name column does not match test order")
    if submission.isna().any().any():
        raise ValueError("Submission contains missing values")
    if not set(submission["label"].unique()).issubset(set(CLASSES)):
        raise ValueError("Submission contains invalid labels")


def extract_id(name: str) -> int:
    digits = "".join(ch for ch in str(name) if ch.isdigit())
    return int(digits)


def encode(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    train_cat = np.zeros((len(train), len(features)), dtype=np.int64)
    test_cat = np.zeros((len(test), len(features)), dtype=np.int64)
    cards: list[int] = []
    for j, col in enumerate(features):
        values = sorted(set(train[col].tolist()) | set(test[col].tolist()))
        mapping = {value: idx for idx, value in enumerate(values)}
        train_cat[:, j] = train[col].map(mapping).to_numpy(dtype=np.int64)
        test_cat[:, j] = test[col].map(mapping).to_numpy(dtype=np.int64)
        cards.append(len(values))
    return train_cat, test_cat, cards


def onehot(cat: np.ndarray, cards: list[int]) -> np.ndarray:
    parts = [np.eye(card, dtype=np.float32)[cat[:, j]] for j, card in enumerate(cards)]
    return np.concatenate(parts, axis=1).astype(np.float32)


class OneHotNet(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 192),
            nn.BatchNorm1d(192),
            nn.SiLU(),
            nn.Dropout(0.06),
            nn.Linear(192, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.06),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Linear(64, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def class_prefix_weights(train: pd.DataFrame, decay: float) -> np.ndarray:
    ids = train["name"].map(extract_id)
    weights = np.ones(len(train), dtype=np.float32)
    for label, idx in train.groupby("label").groups.items():
        order = ids.loc[list(idx)].sort_values().index.to_numpy()
        pos = np.linspace(0.0, 1.0, len(order), dtype=np.float32)
        weights[order] = np.exp(-decay * pos)
    return weights.astype(np.float32)


def train_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    sample_weights: np.ndarray,
    class_weights: list[float],
    seed: int,
    epochs: int,
    device: torch.device,
) -> np.ndarray:
    set_seed(seed)
    model = OneHotNet(train_x.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    dataset = TensorDataset(torch.from_numpy(train_x).float(), torch.from_numpy(train_y).long())
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_x), replacement=True)
    loader = DataLoader(dataset, batch_size=1024, sampler=sampler)
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb.to(device))
            loss = criterion(logits, yb.to(device))
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    out = []
    test_loader = DataLoader(TensorDataset(torch.from_numpy(test_x).float()), batch_size=4096, shuffle=False)
    with torch.no_grad():
        for (xb,) in test_loader:
            out.append(torch.softmax(model(xb.to(device)), dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)


def write_submission(test: pd.DataFrame, pred: np.ndarray, filename: str) -> dict[str, object]:
    submission = pd.DataFrame({"name": test["name"], "label": pred.astype(int)})
    validate_submission(test, submission)
    out_path = SUBMISSION_DIR / filename
    submission.to_csv(out_path, index=False, encoding="utf-8", lineterminator="\n")
    return {
        "path": str(out_path.relative_to(ROOT)),
        "pred_counts": {str(cls): int((pred == cls).sum()) for cls in CLASSES},
    }


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    data_dir = find_data_dir()
    train = pd.read_csv(data_dir / "data_train.csv")
    test = pd.read_csv(data_dir / "data_test.csv")
    features = [c for c in train.columns if c not in ["name", "label"]]
    train_cat, test_cat, cards = encode(train, test, features)
    train_x = onehot(train_cat, cards)
    test_x = onehot(test_cat, cards)
    train_y = train["label"].to_numpy(dtype=np.int64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sample_weights = class_prefix_weights(train, decay=1.5)
    class_weights = [1.15, 0.75, 0.95]
    probas = []
    seeds = [SEED + 301, SEED + 302, SEED + 303, SEED + 304, SEED + 305]
    epochs = 24
    for seed in seeds:
        proba = train_predict(
            train_x=train_x,
            train_y=train_y,
            test_x=test_x,
            sample_weights=sample_weights,
            class_weights=class_weights,
            seed=seed,
            epochs=epochs,
            device=device,
        )
        probas.append(proba)
        print(f"trained seed={seed}", flush=True)

    ensemble = np.mean(probas, axis=0)
    np.save(MODEL_DIR / "prefix_weighted_torch_proba.npy", ensemble)
    base_mlp = np.load(MODEL_DIR / "mlp_onehot_proba.npy")
    hybrid = 0.65 * base_mlp + 0.35 * ensemble
    np.save(MODEL_DIR / "prefix_weighted_torch_hybrid_proba.npy", hybrid)

    candidates: dict[str, tuple[np.ndarray, str]] = {
        "sub_prefix_weighted_raw.csv": (
            ensemble.argmax(axis=1).astype(int),
            "5-seed prefix-weighted torch one-hot MLP ensemble raw",
        ),
        "sub_prefix_weighted_w.csv": (
            (ensemble * np.array([1.0, 0.55, 1.0], dtype=float)).argmax(axis=1).astype(int),
            "5-seed prefix-weighted torch one-hot MLP ensemble, post weights [1,0.55,1]",
        ),
        "sub_prefix_hybrid_w.csv": (
            (hybrid * np.array([1.05, 0.75, 1.0], dtype=float)).argmax(axis=1).astype(int),
            "0.65*online-best MLP + 0.35*prefix-weighted torch, post weights [1.05,0.75,1]",
        ),
    }
    online_best_pred = pd.read_csv(SUBMISSION_DIR / "submission_mlp_onehot_v1.csv")["label"].to_numpy(dtype=int)
    summary: dict[str, object] = {
        "device": str(device),
        "epochs": epochs,
        "seeds": seeds,
        "sample_weight": "exp(-1.5 * normalized_position_within_label)",
        "class_weights": class_weights,
        "candidates": {},
    }
    for filename, (pred, recipe) in candidates.items():
        item = write_submission(test, pred, filename)
        item.update({"recipe": recipe, "diff_vs_mlp_onehot_v1": int((pred != online_best_pred).sum())})
        summary["candidates"][filename] = item

    (REPORT_DIR / "prefix_weighted_torch_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
