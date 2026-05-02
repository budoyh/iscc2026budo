from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw" / "data"
MODELS = ROOT / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train sklearn MLP CV model.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", type=str, default="mlp_standard_f5_s42")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    MODELS.mkdir(parents=True, exist_ok=True)
    run_dir = MODELS / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(RAW / "train_data.csv")
    test = pd.read_csv(RAW / "test_data.csv")
    x = train.drop(columns=["label", "id"])
    x_test = test.drop(columns=["id"])

    encoder = LabelEncoder()
    y = encoder.fit_transform(train["label"])
    num_classes = len(encoder.classes_)

    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof = np.zeros((len(train), num_classes), dtype=np.float64)
    test_proba = np.zeros((len(test), num_classes), dtype=np.float64)
    fold_scores = []

    for fold, (tr_idx, va_idx) in enumerate(splitter.split(x, y), start=1):
        model = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(256, 128),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                batch_size=512,
                learning_rate_init=1e-3,
                max_iter=180,
                early_stopping=True,
                validation_fraction=0.12,
                n_iter_no_change=15,
                random_state=args.seed + fold - 1,
                verbose=False,
            ),
        )
        model.fit(x.iloc[tr_idx], y[tr_idx])
        fold_oof = model.predict_proba(x.iloc[va_idx])
        oof[va_idx] = fold_oof
        test_proba += model.predict_proba(x_test) / args.folds
        score = f1_score(y[va_idx], fold_oof.argmax(axis=1), average="macro")
        fold_scores.append(float(score))
        print(f"fold={fold} macro_f1={score:.6f}")

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
        "run_id": args.run_id,
        "strategy": "sklearn_mlp",
        "seed": args.seed,
        "folds": args.folds,
        "classes": encoder.classes_.tolist(),
        "features": x.columns.tolist(),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "feature_count": int(x.shape[1]),
        "fold_scores": fold_scores,
        "local_macro_f1": local_score,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"run_id": args.run_id, "local_macro_f1": local_score, "fold_scores": fold_scores}))


if __name__ == "__main__":
    main()
