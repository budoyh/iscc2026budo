from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw" / "data"
MODELS = ROOT / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CatBoost CV models for numeric traffic features.")
    parser.add_argument("--features", choices=["raw", "combined_rank", "raw_plus_combined_rank"], default="raw")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=0.035)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--l2-leaf-reg", type=float, default=4.0)
    parser.add_argument("--auto-class-weights", choices=["None", "Balanced", "SqrtBalanced"], default="Balanced")
    parser.add_argument("--early-stopping-rounds", type=int, default=180)
    parser.add_argument("--run-id", type=str, default=None)
    return parser.parse_args()


def make_features(train: pd.DataFrame, test: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_train = train.drop(columns=["id", "label"]).copy()
    x_test = test.drop(columns=["id"]).copy()
    if mode == "raw":
        return x_train, x_test

    combined = pd.concat([x_train, x_test], axis=0, ignore_index=True)
    rank = combined.rank(method="average", pct=True).astype("float32")
    train_rank = rank.iloc[: len(x_train)].reset_index(drop=True)
    test_rank = rank.iloc[len(x_train) :].reset_index(drop=True)
    train_rank.columns = [f"{col}_rank" for col in x_train.columns]
    test_rank.columns = train_rank.columns

    if mode == "combined_rank":
        train_rank.index = x_train.index
        test_rank.index = x_test.index
        return train_rank, test_rank

    train_aug = pd.concat([x_train.reset_index(drop=True), train_rank], axis=1)
    test_aug = pd.concat([x_test.reset_index(drop=True), test_rank], axis=1)
    train_aug.index = x_train.index
    test_aug.index = x_test.index
    return train_aug, test_aug


def build_model(args: argparse.Namespace, seed: int) -> CatBoostClassifier:
    params: dict[str, object] = {
        "loss_function": "MultiClass",
        "eval_metric": "TotalF1:average=Macro",
        "iterations": args.iterations,
        "learning_rate": args.learning_rate,
        "depth": args.depth,
        "l2_leaf_reg": args.l2_leaf_reg,
        "random_seed": seed,
        "bootstrap_type": "Bayesian",
        "bagging_temperature": 0.6,
        "random_strength": 0.8,
        "od_type": "Iter",
        "od_wait": args.early_stopping_rounds,
        "allow_writing_files": False,
        "thread_count": -1,
        "verbose": 100,
    }
    if args.auto_class_weights != "None":
        params["auto_class_weights"] = args.auto_class_weights
    return CatBoostClassifier(**params)


def main() -> None:
    args = parse_args()
    MODELS.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(RAW / "train_data.csv")
    test = pd.read_csv(RAW / "test_data.csv")
    x, x_test = make_features(train, test, args.features)

    encoder = LabelEncoder()
    y = encoder.fit_transform(train["label"])
    classes = encoder.classes_.tolist()

    run_id = args.run_id or (
        f"catboost_{args.features}_d{args.depth}_lr{str(args.learning_rate).replace('.', 'p')}"
        f"_{args.auto_class_weights.lower()}_f{args.folds}_s{args.seed}"
    )
    run_dir = MODELS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof = np.zeros((len(train), len(classes)), dtype=np.float64)
    test_proba = np.zeros((len(test), len(classes)), dtype=np.float64)
    fold_scores: list[float] = []
    best_iterations: list[int] = []

    for fold, (tr_idx, va_idx) in enumerate(splitter.split(x, y), start=1):
        model = build_model(args, args.seed + fold - 1)
        train_pool = Pool(x.iloc[tr_idx], y[tr_idx])
        valid_pool = Pool(x.iloc[va_idx], y[va_idx])
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

        valid_proba = model.predict_proba(valid_pool)
        fold_test = model.predict_proba(Pool(x_test))
        oof[va_idx] = valid_proba
        test_proba += fold_test / args.folds

        score = float(f1_score(y[va_idx], valid_proba.argmax(axis=1), average="macro"))
        fold_scores.append(score)
        best_iterations.append(int(model.get_best_iteration() or args.iterations))
        print(f"fold={fold} macro_f1={score:.6f} best_iteration={best_iterations[-1]}")

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
        "strategy": "catboost",
        "features_mode": args.features,
        "seed": args.seed,
        "folds": args.folds,
        "iterations": args.iterations,
        "learning_rate": args.learning_rate,
        "depth": args.depth,
        "l2_leaf_reg": args.l2_leaf_reg,
        "auto_class_weights": args.auto_class_weights,
        "classes": classes,
        "features": x.columns.tolist(),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "feature_count": int(x.shape[1]),
        "fold_scores": fold_scores,
        "best_iterations": best_iterations,
        "local_macro_f1": local_score,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"run_id": run_id, "local_macro_f1": local_score, "fold_scores": fold_scores}, ensure_ascii=False))


if __name__ == "__main__":
    main()
