from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw" / "data"
MODELS = ROOT / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="kNN / label propagation style baseline.")
    parser.add_argument("--features", choices=["standard", "rank", "standard_rank"], default="standard_rank")
    parser.add_argument("--n-neighbors", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", type=str, default=None)
    return parser.parse_args()


def make_features(train: pd.DataFrame, test: pd.DataFrame, mode: str) -> tuple[np.ndarray, np.ndarray]:
    x_train = train.drop(columns=["label", "id"]).copy()
    x_test = test.drop(columns=["id"]).copy()
    scaler = StandardScaler()
    xs_train = scaler.fit_transform(x_train).astype("float32")
    xs_test = scaler.transform(x_test).astype("float32")

    combined = pd.concat([x_train, x_test], ignore_index=True)
    rank = combined.rank(method="average", pct=True).to_numpy(dtype="float32")
    xr_train = rank[: len(x_train)]
    xr_test = rank[len(x_train) :]

    if mode == "standard":
        return xs_train, xs_test
    if mode == "rank":
        return xr_train, xr_test
    return np.concatenate([xs_train, xr_train], axis=1), np.concatenate([xs_test, xr_test], axis=1)


def knn_proba(x_ref: np.ndarray, y_ref: np.ndarray, x_query: np.ndarray, num_classes: int, k: int, temp: float) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=-1)
    nn.fit(x_ref)
    dist, idx = nn.kneighbors(x_query, return_distance=True)
    scale = np.median(dist[:, max(1, k // 4)])
    scale = max(float(scale), 1e-6)
    weights = np.exp(-(dist / (scale * temp)) ** 2)
    proba = np.zeros((len(x_query), num_classes), dtype=np.float64)
    labels = y_ref[idx]
    for cls in range(num_classes):
        proba[:, cls] = (weights * (labels == cls)).sum(axis=1)
    proba += 1e-9
    proba /= proba.sum(axis=1, keepdims=True)
    return proba


def main() -> None:
    args = parse_args()
    MODELS.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(RAW / "train_data.csv")
    test = pd.read_csv(RAW / "test_data.csv")
    x, x_test = make_features(train, test, args.features)
    encoder = LabelEncoder()
    y = encoder.fit_transform(train["label"])
    num_classes = len(encoder.classes_)

    run_id = args.run_id or f"knn_{args.features}_k{args.n_neighbors}_t{str(args.temperature).replace('.', 'p')}_s{args.seed}"
    run_dir = MODELS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof = np.zeros((len(train), num_classes), dtype=np.float64)
    fold_scores: list[float] = []
    for fold, (tr_idx, va_idx) in enumerate(cv.split(x, y), start=1):
        p = knn_proba(x[tr_idx], y[tr_idx], x[va_idx], num_classes, args.n_neighbors, args.temperature)
        oof[va_idx] = p
        score = f1_score(y[va_idx], p.argmax(axis=1), average="macro")
        fold_scores.append(float(score))
        print(f"fold={fold} macro_f1={score:.6f}")

    test_proba = knn_proba(x, y, x_test, num_classes, args.n_neighbors, args.temperature)
    local_score = float(f1_score(y, oof.argmax(axis=1), average="macro"))

    np.save(run_dir / "oof_proba.npy", oof)
    np.save(run_dir / "test_proba.npy", test_proba)
    pd.DataFrame(
        {"id": train["id"], "true_label": train["label"], "pred_label": encoder.inverse_transform(oof.argmax(axis=1))}
    ).to_csv(run_dir / "oof_predictions.csv", index=False, encoding="utf-8")

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "strategy": "knn_label_propagation",
        "features_mode": args.features,
        "n_neighbors": args.n_neighbors,
        "temperature": args.temperature,
        "seed": args.seed,
        "folds": args.folds,
        "classes": encoder.classes_.tolist(),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "feature_count": int(x.shape[1]),
        "fold_scores": fold_scores,
        "local_macro_f1": local_score,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"run_id": run_id, "local_macro_f1": local_score, "fold_scores": fold_scores}))


if __name__ == "__main__":
    main()
