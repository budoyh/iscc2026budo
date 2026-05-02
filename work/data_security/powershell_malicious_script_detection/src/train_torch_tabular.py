from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def write_submission(test: pd.DataFrame, proba: np.ndarray, filename: str, weights: list[float] | None = None) -> dict[str, object]:
    if weights is None:
        weights = [1.0, 1.0, 1.0]
    pred = (proba * np.array(weights, dtype=float)).argmax(axis=1).astype(int)
    submission = pd.DataFrame({"name": test["name"], "label": pred})
    validate_submission(test, submission)
    out_path = SUBMISSION_DIR / filename
    submission.to_csv(out_path, index=False, encoding="utf-8", lineterminator="\n")
    return {
        "path": str(out_path.relative_to(ROOT)),
        "weights": weights,
        "pred_counts": {str(cls): int((pred == cls).sum()) for cls in CLASSES},
    }


def adjust_to_quota(proba: np.ndarray, target_counts: list[int]) -> np.ndarray:
    target = np.array(target_counts, dtype=int)
    pred = proba.argmax(axis=1).astype(int)
    counts = np.bincount(pred, minlength=3)
    log_scores = np.log(np.clip(proba, 1e-15, None))
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
        if changed != need:
            raise RuntimeError(f"Could not satisfy quota for class {dst}: {changed}/{need}")
    if not np.array_equal(np.bincount(pred, minlength=3), target):
        raise RuntimeError(f"Quota adjustment failed: {np.bincount(pred, minlength=3)} vs {target}")
    return pred


def write_pred_submission(test: pd.DataFrame, pred: np.ndarray, filename: str) -> dict[str, object]:
    submission = pd.DataFrame({"name": test["name"], "label": pred.astype(int)})
    validate_submission(test, submission)
    out_path = SUBMISSION_DIR / filename
    submission.to_csv(out_path, index=False, encoding="utf-8", lineterminator="\n")
    return {
        "path": str(out_path.relative_to(ROOT)),
        "pred_counts": {str(cls): int((pred == cls).sum()) for cls in CLASSES},
    }


def encode_data(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    train_x = np.zeros((len(train), len(features)), dtype=np.int64)
    test_x = np.zeros((len(test), len(features)), dtype=np.int64)
    cardinalities: list[int] = []
    for j, col in enumerate(features):
        values = sorted(set(train[col].tolist()) | set(test[col].tolist()))
        mapping = {value: idx for idx, value in enumerate(values)}
        train_x[:, j] = train[col].map(mapping).to_numpy(dtype=np.int64)
        test_x[:, j] = test[col].map(mapping).to_numpy(dtype=np.int64)
        cardinalities.append(len(values))
    return train_x, test_x, cardinalities


def make_onehot(x: np.ndarray, cardinalities: list[int]) -> np.ndarray:
    parts = []
    for j, card in enumerate(cardinalities):
        eye = np.eye(card, dtype=np.float32)
        parts.append(eye[x[:, j]])
    return np.concatenate(parts, axis=1)


class OneHotMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: tuple[int, ...], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for width in hidden:
            layers.extend([nn.Linear(prev, width), nn.BatchNorm1d(width), nn.SiLU(), nn.Dropout(dropout)])
            prev = width
        layers.append(nn.Linear(prev, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, onehot: torch.Tensor, cat: torch.Tensor) -> torch.Tensor:
        return self.net(onehot)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class EmbResMLP(nn.Module):
    def __init__(self, cardinalities: list[int], input_dim: int, emb_dim: int, hidden_dim: int, depth: int, dropout: float) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(card, emb_dim) for card in cardinalities])
        self.in_proj = nn.Linear(len(cardinalities) * emb_dim + input_dim, hidden_dim)
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim, dropout) for _ in range(depth)])
        self.head = nn.Sequential(nn.BatchNorm1d(hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 3))

    def forward(self, onehot: torch.Tensor, cat: torch.Tensor) -> torch.Tensor:
        embs = [emb(cat[:, j]) for j, emb in enumerate(self.embeddings)]
        x = torch.cat(embs + [onehot], dim=1)
        x = self.in_proj(x)
        x = self.blocks(x)
        return self.head(x)


class TabTransformer(nn.Module):
    def __init__(self, cardinalities: list[int], dim: int, depth: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.value_embeddings = nn.ModuleList([nn.Embedding(card, dim) for card in cardinalities])
        self.feature_embedding = nn.Parameter(torch.randn(len(cardinalities), dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.head = nn.Sequential(
            nn.LayerNorm(len(cardinalities) * dim),
            nn.Linear(len(cardinalities) * dim, 192),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(192, 3),
        )

    def forward(self, onehot: torch.Tensor, cat: torch.Tensor) -> torch.Tensor:
        tokens = torch.stack([emb(cat[:, j]) for j, emb in enumerate(self.value_embeddings)], dim=1)
        tokens = tokens + self.feature_embedding.unsqueeze(0)
        tokens = self.encoder(tokens)
        return self.head(tokens.flatten(start_dim=1))


@dataclass
class TrainConfig:
    name: str
    kind: str
    seed_offset: int
    max_epochs: int = 220
    patience: int = 28
    lr: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.0
    batch_size: int = 1024


def build_model(config: TrainConfig, input_dim: int, cardinalities: list[int]) -> nn.Module:
    if config.kind == "onehot_mlp":
        return OneHotMLP(input_dim=input_dim, hidden=(192, 128, 64), dropout=0.06)
    if config.kind == "wide_mlp":
        return OneHotMLP(input_dim=input_dim, hidden=(320, 160, 80), dropout=0.08)
    if config.kind == "emb_res":
        return EmbResMLP(cardinalities=cardinalities, input_dim=input_dim, emb_dim=8, hidden_dim=192, depth=3, dropout=0.08)
    if config.kind == "tab_transformer":
        return TabTransformer(cardinalities=cardinalities, dim=32, depth=2, heads=4, dropout=0.06)
    raise ValueError(f"Unknown model kind: {config.kind}")


def predict_proba(model: nn.Module, onehot: np.ndarray, cat: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    out = []
    dataset = TensorDataset(torch.from_numpy(onehot).float(), torch.from_numpy(cat).long())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for xb_onehot, xb_cat in loader:
            logits = model(xb_onehot.to(device), xb_cat.to(device))
            out.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)


def train_one_fold(
    config: TrainConfig,
    train_onehot: np.ndarray,
    train_cat: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    test_onehot: np.ndarray,
    test_cat: np.ndarray,
    cardinalities: list[int],
    device: torch.device,
    fold: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    set_seed(SEED + config.seed_offset + fold)
    model = build_model(config, input_dim=train_onehot.shape[1], cardinalities=cardinalities).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_epochs)

    train_dataset = TensorDataset(
        torch.from_numpy(train_onehot[train_idx]).float(),
        torch.from_numpy(train_cat[train_idx]).long(),
        torch.from_numpy(y[train_idx]).long(),
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=False)

    best_score = -math.inf
    best_state = None
    best_epoch = 0
    stale = 0
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        for xb_onehot, xb_cat, yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb_onehot.to(device), xb_cat.to(device))
            loss = criterion(logits, yb.to(device))
            loss.backward()
            optimizer.step()
        scheduler.step()

        if epoch < 20 or epoch % 5 != 0:
            continue
        valid_proba = predict_proba(
            model,
            train_onehot[valid_idx],
            train_cat[valid_idx],
            device=device,
            batch_size=config.batch_size,
        )
        valid_pred = valid_proba.argmax(axis=1)
        score = f1_score(y[valid_idx], valid_pred, average="macro")
        if score > best_score:
            best_score = float(score)
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 5
        if stale >= config.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    valid_proba = predict_proba(model, train_onehot[valid_idx], train_cat[valid_idx], device, config.batch_size)
    test_proba = predict_proba(model, test_onehot, test_cat, device, config.batch_size)
    meta = {
        "fold": fold,
        "best_epoch": best_epoch,
        "valid_macro_f1": float(f1_score(y[valid_idx], valid_proba.argmax(axis=1), average="macro")),
        "valid_counts": {str(cls): int((valid_proba.argmax(axis=1) == cls).sum()) for cls in CLASSES},
    }
    return valid_proba, test_proba, meta


def main() -> None:
    start = time.time()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    data_dir = find_data_dir()
    train = pd.read_csv(data_dir / "data_train.csv")
    test = pd.read_csv(data_dir / "data_test.csv")
    features = [c for c in train.columns if c not in ["name", "label"]]
    train_cat, test_cat, cardinalities = encode_data(train, test, features)
    train_onehot = make_onehot(train_cat, cardinalities)
    test_onehot = make_onehot(test_cat, cardinalities)
    y = train["label"].to_numpy(dtype=np.int64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    configs = [
        TrainConfig(name="torch_onehot_mlp_cv_v1", kind="onehot_mlp", seed_offset=11, label_smoothing=0.0),
        TrainConfig(name="torch_wide_mlp_cv_v1", kind="wide_mlp", seed_offset=23, label_smoothing=0.0),
        TrainConfig(name="torch_emb_res_cv_v1", kind="emb_res", seed_offset=37, label_smoothing=0.01),
        TrainConfig(name="torch_tabtransformer_cv_v1", kind="tab_transformer", seed_offset=51, label_smoothing=0.01, lr=8e-4),
    ]

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    summary: dict[str, object] = {
        "device": str(device),
        "torch_version": torch.__version__,
        "features": features,
        "cardinalities": cardinalities,
        "configs": {},
        "candidates": {},
    }
    config_test_probs: dict[str, np.ndarray] = {}

    for config in configs:
        print(f"training {config.name} on {device}", flush=True)
        oof = np.zeros((len(train), len(CLASSES)), dtype=np.float32)
        test_accum = np.zeros((len(test), len(CLASSES)), dtype=np.float32)
        fold_meta = []
        for fold, (train_idx, valid_idx) in enumerate(skf.split(train_cat, y), start=1):
            valid_proba, test_proba, meta = train_one_fold(
                config,
                train_onehot,
                train_cat,
                y,
                train_idx,
                valid_idx,
                test_onehot,
                test_cat,
                cardinalities,
                device,
                fold,
            )
            oof[valid_idx] = valid_proba
            test_accum += test_proba.astype(np.float32) / skf.n_splits
            fold_meta.append(meta)
            print(f"{config.name} fold {fold}: f1={meta['valid_macro_f1']:.6f} epoch={meta['best_epoch']}", flush=True)

        np.save(MODEL_DIR / f"{config.name}_oof.npy", oof)
        np.save(MODEL_DIR / f"{config.name}_test_proba.npy", test_accum)
        config_test_probs[config.name] = test_accum

        oof_pred = oof.argmax(axis=1)
        raw_item = write_submission(test, test_accum, f"submission_{config.name}.csv")
        raw_item["recipe"] = f"{config.kind} 3-fold torch CV ensemble"
        summary["candidates"][f"submission_{config.name}.csv"] = raw_item

        quota_pred = adjust_to_quota(test_accum, [13500, 3000, 3500])
        quota_item = write_pred_submission(test, quota_pred, f"submission_{config.name}_quota13500.csv")
        quota_item["recipe"] = f"{config.kind} 3-fold torch CV ensemble, least-loss quota 13500/3000/3500"
        summary["candidates"][f"submission_{config.name}_quota13500.csv"] = quota_item

        summary["configs"][config.name] = {
            "kind": config.kind,
            "oof_macro_f1": float(f1_score(y, oof_pred, average="macro")),
            "oof_counts": {str(cls): int((oof_pred == cls).sum()) for cls in CLASSES},
            "folds": fold_meta,
        }

    torch_ensemble = np.mean(list(config_test_probs.values()), axis=0)
    np.save(MODEL_DIR / "torch_nn_ensemble_test_proba.npy", torch_ensemble)
    item = write_submission(test, torch_ensemble, "submission_torch_nn_ensemble_v1.csv")
    item["recipe"] = "mean of torch onehot/wide/emb_res/tabtransformer CV ensembles"
    summary["candidates"]["submission_torch_nn_ensemble_v1.csv"] = item
    quota_pred = adjust_to_quota(torch_ensemble, [13500, 3000, 3500])
    item = write_pred_submission(test, quota_pred, "submission_torch_nn_ensemble_quota13500_v1.csv")
    item["recipe"] = "mean torch ensemble, least-loss quota 13500/3000/3500"
    summary["candidates"]["submission_torch_nn_ensemble_quota13500_v1.csv"] = item

    sklearn_mlp_path = MODEL_DIR / "mlp_onehot_proba.npy"
    if sklearn_mlp_path.exists():
        sklearn_mlp = np.load(sklearn_mlp_path)
        hybrid = 0.5 * torch_ensemble + 0.5 * sklearn_mlp
        np.save(MODEL_DIR / "torch_sklearn_nn_ensemble_test_proba.npy", hybrid)
        item = write_submission(test, hybrid, "submission_torch_sklearn_nn_ensemble_v1.csv")
        item["recipe"] = "0.5*torch_nn_ensemble + 0.5*sklearn_mlp_onehot"
        summary["candidates"]["submission_torch_sklearn_nn_ensemble_v1.csv"] = item
        quota_pred = adjust_to_quota(hybrid, [13500, 3000, 3500])
        item = write_pred_submission(test, quota_pred, "submission_torch_sklearn_nn_ensemble_quota13500_v1.csv")
        item["recipe"] = "0.5*torch_nn_ensemble + 0.5*sklearn_mlp_onehot, least-loss quota 13500/3000/3500"
        summary["candidates"]["submission_torch_sklearn_nn_ensemble_quota13500_v1.csv"] = item

    summary["seconds"] = round(time.time() - start, 3)
    (REPORT_DIR / "torch_tabular_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
