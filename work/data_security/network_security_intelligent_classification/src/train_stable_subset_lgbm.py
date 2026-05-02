from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw" / "data"
MODELS = ROOT / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LightGBM on a drift-filtered stable feature subset.")
    parser.add_argument("--drop-top-drift", type=int, default=10)
    parser.add_argument("--ranking", choices=["smd", "smd_std"], default="smd_std")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=900)
    parser.add_argument("--run-id", type=str, default=None)
    return parser.parse_args()


def rank_drift_features(train_x: pd.DataFrame, test_x: pd.DataFrame, ranking: str) -> pd.Series:
    train_std = train_x.std(axis=0).replace(0, np.nan)
    smd = ((test_x.mean(axis=0) - train_x.mean(axis=0)) / train_std).abs()
    if ranking == "smd":
        return smd.sort_values(ascending=False)
    std_shift = ((test_x.std(axis=0) / train_std) - 1.0).abs()
    score = smd + 0.5 * std_shift
    return score.sort_values(ascending=False)


def build_model(seed: int, num_classes: int, n_estimators: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="multiclass",
        num_class=num_classes,
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def main() -> None:
    args = parse_args()
    MODELS.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(RAW / "train_data.csv")
    test = pd.read_csv(RAW / "test_data.csv")
    all_features = [col for col in train.columns if col not in {"id", "label"}]
    drift_rank = rank_drift_features(train[all_features], test[all_features], args.ranking)
    dropped = drift_rank.index[: args.drop_top_drift].tolist()
    features = [col for col in all_features if col not in set(dropped)]

    encoder = LabelEncoder()
    y = encoder.fit_transform(train["label"]).astype(np.int64)
    classes = encoder.classes_.tolist()
    x = train[features]
    x_test = test[features]

    run_id = args.run_id or f"lgbm_stable_drop{args.drop_top_drift}_{args.ranking}_n{args.n_estimators}_f{args.folds}_s{args.seed}"
    run_dir = MODELS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof = np.zeros((len(train), len(classes)), dtype=np.float64)
    test_proba = np.zeros((len(test), len(classes)), dtype=np.float64)
    fold_scores: list[float] = []
    for fold, (tr_idx, va_idx) in enumerate(cv.split(x, y), start=1):
        model = build_model(args.seed + fold - 1, len(classes), args.n_estimators)
        model.fit(x.iloc[tr_idx], y[tr_idx])
        valid_proba = model.predict_proba(x.iloc[va_idx])
        oof[va_idx] = valid_proba
        test_proba += model.predict_proba(x_test) / args.folds
        score = float(f1_score(y[va_idx], valid_proba.argmax(axis=1), average="macro"))
        fold_scores.append(score)
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
        "run_id": run_id,
        "strategy": "stable_subset_lgbm",
        "ranking": args.ranking,
        "drop_top_drift": args.drop_top_drift,
        "dropped_features": dropped,
        "drift_scores": drift_rank.loc[dropped].to_dict(),
        "seed": args.seed,
        "folds": args.folds,
        "classes": classes,
        "features": features,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "feature_count": len(features),
        "fold_scores": fold_scores,
        "local_macro_f1": local_score,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"run_id": run_id, "local_macro_f1": local_score, "dropped_features": dropped}, ensure_ascii=False))


if __name__ == "__main__":
    main()
