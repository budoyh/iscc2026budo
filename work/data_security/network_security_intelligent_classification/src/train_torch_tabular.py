from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw" / "data"
MODELS = ROOT / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PyTorch tabular models with optional domain adaptation.")
    parser.add_argument("--model", choices=["resnet", "fttransformer"], default="resnet")
    parser.add_argument("--features", choices=["raw", "raw_rank"], default="raw_rank")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--domain-weight", type=float, default=0.0)
    parser.add_argument("--class-weight", action="store_true")
    parser.add_argument("--run-id", type=str, default=None)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def load_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(RAW / "train_data.csv"), pd.read_csv(RAW / "test_data.csv")


def make_features(train: pd.DataFrame, test: pd.DataFrame, feature_mode: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    x_train = train.drop(columns=["label", "id"]).copy()
    x_test = test.drop(columns=["id"]).copy()
    feature_names = x_train.columns.tolist()

    scaler = StandardScaler()
    x_train_std = scaler.fit_transform(x_train).astype("float32")
    x_test_std = scaler.transform(x_test).astype("float32")

    if feature_mode == "raw":
        return x_train_std, x_test_std, feature_names

    combined = pd.concat([x_train, x_test], axis=0, ignore_index=True)
    ranks = combined.rank(method="average", pct=True).to_numpy(dtype="float32")
    train_rank = ranks[: len(x_train)]
    test_rank = ranks[len(x_train) :]
    rank_names = [f"{name}_rank" for name in feature_names]
    return (
        np.concatenate([x_train_std, train_rank], axis=1).astype("float32"),
        np.concatenate([x_test_std, test_rank], axis=1).astype("float32"),
        feature_names + rank_names,
    )


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return GradReverse.apply(x, lambd)


class ResBlock(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ResNetTabular(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, width: int = 256, depth: int = 5, dropout: float = 0.12) -> None:
        super().__init__()
        self.input = nn.Sequential(nn.Linear(in_dim, width), nn.GELU(), nn.Dropout(dropout))
        self.blocks = nn.Sequential(*[ResBlock(width, width * 2, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, num_classes)
        self.domain = nn.Sequential(
            nn.Linear(width, width // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width // 2, 2),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.blocks(self.input(x)))

    def forward(self, x: torch.Tensor, grl_lambda: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.head(z), self.domain(grad_reverse(z, grl_lambda))


class FTTransformer(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, d_token: int = 64, n_heads: int = 8, depth: int = 3) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_dim, d_token) * 0.02)
        self.bias = nn.Parameter(torch.zeros(in_dim, d_token))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_token * 4,
            dropout=0.12,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.head = nn.Sequential(nn.LayerNorm(d_token), nn.Linear(d_token, num_classes))
        self.domain = nn.Sequential(nn.LayerNorm(d_token), nn.Linear(d_token, d_token), nn.GELU(), nn.Linear(d_token, 2))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        cls = self.cls.expand(x.shape[0], -1, -1)
        out = self.encoder(torch.cat([cls, tokens], dim=1))
        return out[:, 0]

    def forward(self, x: torch.Tensor, grl_lambda: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.head(z), self.domain(grad_reverse(z, grl_lambda))


def make_model(name: str, in_dim: int, num_classes: int) -> nn.Module:
    if name == "resnet":
        return ResNetTabular(in_dim=in_dim, num_classes=num_classes)
    if name == "fttransformer":
        return FTTransformer(in_dim=in_dim, num_classes=num_classes)
    raise ValueError(name)


def cycle(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def predict_proba(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch_size * 2, shuffle=False)
    probs: list[np.ndarray] = []
    for (xb,) in loader:
        logits, _ = model(xb.to(device))
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probs, axis=0)


def train_fold(
    args: argparse.Namespace,
    fold: int,
    x: np.ndarray,
    y: np.ndarray,
    x_test: np.ndarray,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    class_weights: torch.Tensor | None,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    seed_everything(args.seed + fold - 1)
    model = make_model(args.model, x.shape[1], int(y.max() + 1)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    supervised_loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    domain_loss = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x[tr_idx]), torch.from_numpy(y[tr_idx]).long()),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    test_loader = DataLoader(TensorDataset(torch.from_numpy(x_test)), batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_iter = cycle(test_loader)

    best_score = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        p = epoch / max(args.epochs, 1)
        grl = 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits, domain_train = model(xb, grl_lambda=grl if args.domain_weight > 0 else 0.0)
            loss = supervised_loss(logits, yb)

            if args.domain_weight > 0:
                (xt,) = next(test_iter)
                xt = xt.to(device)
                _, domain_test = model(xt, grl_lambda=grl)
                d_logits = torch.cat([domain_train, domain_test], dim=0)
                d_labels = torch.cat(
                    [
                        torch.zeros(domain_train.shape[0], dtype=torch.long, device=device),
                        torch.ones(domain_test.shape[0], dtype=torch.long, device=device),
                    ],
                    dim=0,
                )
                loss = loss + args.domain_weight * domain_loss(d_logits, d_labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()

        valid_proba = predict_proba(model, x[va_idx], device, args.batch_size)
        score = f1_score(y[va_idx], valid_proba.argmax(axis=1), average="macro")
        if score > best_score:
            best_score = float(score)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        print(f"fold={fold} epoch={epoch} macro_f1={score:.6f} best={best_score:.6f}")
        if bad_epochs >= args.patience:
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    valid_proba = predict_proba(model, x[va_idx], device, args.batch_size)
    test_proba = predict_proba(model, x_test, device, args.batch_size)
    return valid_proba, test_proba, best_score


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    MODELS.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train, test = load_frames()
    x, x_test, features = make_features(train, test, args.features)
    encoder = LabelEncoder()
    y = encoder.fit_transform(train["label"]).astype("int64")
    num_classes = len(encoder.classes_)

    run_id = args.run_id or (
        f"torch_{args.model}_{args.features}_dw{str(args.domain_weight).replace('.', 'p')}"
        f"{'_cw' if args.class_weight else ''}_f{args.folds}_s{args.seed}"
    )
    run_dir = MODELS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    class_weights = None
    if args.class_weight:
        counts = np.bincount(y, minlength=num_classes)
        weights = len(y) / (num_classes * np.maximum(counts, 1))
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof = np.zeros((len(train), num_classes), dtype=np.float64)
    test_proba = np.zeros((len(test), num_classes), dtype=np.float64)
    fold_scores: list[float] = []

    for fold, (tr_idx, va_idx) in enumerate(splitter.split(x, y), start=1):
        valid_proba, fold_test, best_score = train_fold(
            args=args,
            fold=fold,
            x=x,
            y=y,
            x_test=x_test,
            tr_idx=tr_idx,
            va_idx=va_idx,
            class_weights=class_weights,
            device=device,
        )
        oof[va_idx] = valid_proba
        test_proba += fold_test / args.folds
        fold_scores.append(best_score)

    local_score = float(f1_score(y, oof.argmax(axis=1), average="macro"))
    np.save(run_dir / "oof_proba.npy", oof)
    np.save(run_dir / "test_proba.npy", test_proba)
    pd.DataFrame(
        {
            "id": train["id"],
            "true_label": train["label"],
            "pred_label": encoder.inverse_transform(oof.argmax(axis=1)),
        }
    ).to_csv(run_dir / "oof_predictions.csv", index=False, encoding="utf-8")

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "strategy": f"torch_{args.model}",
        "features_mode": args.features,
        "seed": args.seed,
        "folds": args.folds,
        "epochs": args.epochs,
        "patience": args.patience,
        "domain_weight": args.domain_weight,
        "class_weight": args.class_weight,
        "classes": encoder.classes_.tolist(),
        "features": features,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "feature_count": int(x.shape[1]),
        "fold_scores": fold_scores,
        "local_macro_f1": local_score,
        "device": str(device),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"run_id": run_id, "local_macro_f1": local_score, "fold_scores": fold_scores}, ensure_ascii=False))


if __name__ == "__main__":
    main()
