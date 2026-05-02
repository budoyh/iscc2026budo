from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw" / "data"
MODELS = ROOT / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CV models for ISCC network security classification.")
    parser.add_argument(
        "--strategy",
        choices=["lgbm_base", "lgbm_adv_weighted", "xgb_base"],
        default="lgbm_adv_weighted",
        help="Training strategy.",
    )
    parser.add_argument("--folds", type=int, default=5, help="Number of stratified folds.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--n-estimators", type=int, default=1400, help="Number of boosting rounds.")
    parser.add_argument(
        "--transform",
        choices=["raw", "split_rank", "split_standard", "combined_rank", "raw_plus_combined_rank", "coral"],
        default="raw",
        help="Feature transform. split_* aligns train and test marginal distributions separately.",
    )
    parser.add_argument(
        "--class-weight",
        choices=["none", "balanced"],
        default="none",
        help="Optional class-balanced training weights for macro F1.",
    )
    parser.add_argument("--run-id", type=str, default=None, help="Output run id. Defaults to strategy_f{folds}_s{seed}.")
    parser.add_argument(
        "--save-fold-models",
        action="store_true",
        help="Persist each fold model with joblib.",
    )
    return parser.parse_args()


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(RAW / "train_data.csv")
    test = pd.read_csv(RAW / "test_data.csv")
    return train, test


def get_feature_frames(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_train = train.drop(columns=["label", "id"]).copy()
    x_test = test.drop(columns=["id"]).copy()
    return x_train, x_test


def apply_transform(x_train: pd.DataFrame, x_test: pd.DataFrame, transform: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if transform == "raw":
        return x_train, x_test

    if transform == "split_rank":
        train_rank = x_train.rank(method="average", pct=True)
        test_rank = x_test.rank(method="average", pct=True)
        return train_rank.astype("float32"), test_rank.astype("float32")

    if transform == "split_standard":
        train_std = x_train.std(axis=0).replace(0, 1.0)
        test_std = x_test.std(axis=0).replace(0, 1.0)
        train_z = (x_train - x_train.mean(axis=0)) / train_std
        test_z = (x_test - x_test.mean(axis=0)) / test_std
        return train_z.astype("float32"), test_z.astype("float32")

    if transform in {"combined_rank", "raw_plus_combined_rank"}:
        combined = pd.concat([x_train, x_test], axis=0, ignore_index=True)
        ranked = combined.rank(method="average", pct=True).astype("float32")
        train_rank = ranked.iloc[: len(x_train)].reset_index(drop=True)
        test_rank = ranked.iloc[len(x_train) :].reset_index(drop=True)
        train_rank.index = x_train.index
        test_rank.index = x_test.index
        if transform == "combined_rank":
            train_rank.columns = x_train.columns
            test_rank.columns = x_test.columns
            return train_rank, test_rank

        train_aug = pd.concat(
            [x_train.reset_index(drop=True), train_rank.add_suffix("_combined_rank").reset_index(drop=True)], axis=1
        )
        test_aug = pd.concat(
            [x_test.reset_index(drop=True), test_rank.add_suffix("_combined_rank").reset_index(drop=True)], axis=1
        )
        train_aug.index = x_train.index
        test_aug.index = x_test.index
        return train_aug.astype("float32"), test_aug.astype("float32")

    if transform == "coral":
        columns = x_train.columns
        train_values = x_train.to_numpy(dtype=np.float64)
        test_values = x_test.to_numpy(dtype=np.float64)
        train_mean = train_values.mean(axis=0, keepdims=True)
        test_mean = test_values.mean(axis=0, keepdims=True)
        train_centered = train_values - train_mean
        test_centered = test_values - test_mean

        eps = 1e-3
        train_cov = np.cov(train_centered, rowvar=False) + eps * np.eye(train_centered.shape[1])
        test_cov = np.cov(test_centered, rowvar=False) + eps * np.eye(test_centered.shape[1])

        train_vals, train_vecs = np.linalg.eigh(train_cov)
        test_vals, test_vecs = np.linalg.eigh(test_cov)
        train_inv_sqrt = train_vecs @ np.diag(1.0 / np.sqrt(np.maximum(train_vals, eps))) @ train_vecs.T
        test_sqrt = test_vecs @ np.diag(np.sqrt(np.maximum(test_vals, eps))) @ test_vecs.T

        aligned_train = train_centered @ train_inv_sqrt @ test_sqrt + test_mean
        train_frame = pd.DataFrame(aligned_train, columns=columns, index=x_train.index)
        return train_frame.astype("float32"), x_test.astype("float32")

    raise ValueError(f"Unsupported transform: {transform}")


def label_encode(train: pd.DataFrame) -> tuple[np.ndarray, LabelEncoder]:
    encoder = LabelEncoder()
    y = encoder.fit_transform(train["label"])
    return y, encoder


def build_domain_scores(x_train: pd.DataFrame, x_test: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray]:
    domain_x = pd.concat([x_train.assign(_is_test=0), x_test.assign(_is_test=1)], ignore_index=True)
    domain_y = domain_x.pop("_is_test").to_numpy()
    oof_scores = np.zeros(len(domain_x), dtype=np.float64)
    cv = KFold(n_splits=3, shuffle=True, random_state=seed)

    for train_idx, valid_idx in cv.split(domain_x):
        model = LGBMClassifier(
            objective="binary",
            n_estimators=800,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(domain_x.iloc[train_idx], domain_y[train_idx])
        oof_scores[valid_idx] = model.predict_proba(domain_x.iloc[valid_idx])[:, 1]

    return oof_scores[: len(x_train)], oof_scores[len(x_train) :]


def build_sample_weights(train_scores: np.ndarray) -> np.ndarray:
    odds = train_scores / np.maximum(1.0 - train_scores, 1e-6)
    weights = np.clip(odds, 0.05, 10.0)
    return weights / weights.mean()


def build_class_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y)
    class_weights = len(y) / (len(counts) * np.maximum(counts, 1))
    weights = class_weights[y]
    return weights / weights.mean()


def make_model(strategy: str, seed: int, num_classes: int, n_estimators: int):
    if strategy in {"lgbm_base", "lgbm_adv_weighted"}:
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

    if strategy == "xgb_base":
        return XGBClassifier(
            objective="multi:softprob",
            num_class=num_classes,
            n_estimators=n_estimators,
            learning_rate=0.05,
            max_depth=8,
            min_child_weight=1,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
            eval_metric="mlogloss",
        )

    raise ValueError(f"Unsupported strategy: {strategy}")


def fit_one_fold(
    strategy: str,
    seed: int,
    num_classes: int,
    n_estimators: int,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    sample_weight: np.ndarray | None,
):
    model = make_model(strategy=strategy, seed=seed, num_classes=num_classes, n_estimators=n_estimators)
    fit_kwargs: dict[str, object] = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    if strategy == "xgb_base":
        model.fit(x_train, y_train, verbose=False, **fit_kwargs)
    else:
        model.fit(x_train, y_train, **fit_kwargs)
    return model


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    MODELS.mkdir(parents=True, exist_ok=True)

    train_df, test_df = load_data()
    x_train, x_test = get_feature_frames(train_df, test_df)
    x_train, x_test = apply_transform(x_train, x_test, args.transform)
    y, encoder = label_encode(train_df)

    transform_suffix = "" if args.transform == "raw" else f"_{args.transform}"
    class_weight_suffix = "" if args.class_weight == "none" else f"_cw_{args.class_weight}"
    estimators_suffix = "" if args.n_estimators == 1400 else f"_n{args.n_estimators}"
    run_id = args.run_id or f"{args.strategy}{transform_suffix}{class_weight_suffix}{estimators_suffix}_f{args.folds}_s{args.seed}"
    run_dir = MODELS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    train_scores = None
    test_scores = None
    sample_weights = None
    if args.strategy == "lgbm_adv_weighted":
        train_scores, test_scores = build_domain_scores(x_train, x_test, seed=args.seed)
        sample_weights = build_sample_weights(train_scores)
    if args.class_weight == "balanced":
        class_weights = build_class_weights(y)
        sample_weights = class_weights if sample_weights is None else sample_weights * class_weights
        sample_weights = sample_weights / sample_weights.mean()

    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    num_classes = len(encoder.classes_)
    oof_proba = np.zeros((len(train_df), num_classes), dtype=np.float64)
    test_proba = np.zeros((len(test_df), num_classes), dtype=np.float64)
    fold_scores: list[float] = []

    for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(x_train, y), start=1):
        fold_weight = None
        if sample_weights is not None:
            fold_weight = sample_weights[train_idx]

        model = fit_one_fold(
            strategy=args.strategy,
            seed=args.seed + fold_idx - 1,
            num_classes=num_classes,
            n_estimators=args.n_estimators,
            x_train=x_train.iloc[train_idx],
            y_train=y[train_idx],
            sample_weight=fold_weight,
        )

        fold_valid_proba = model.predict_proba(x_train.iloc[valid_idx])
        fold_test_proba = model.predict_proba(x_test)

        oof_proba[valid_idx] = fold_valid_proba
        test_proba += fold_test_proba / args.folds

        valid_pred = fold_valid_proba.argmax(axis=1)
        fold_score = f1_score(y[valid_idx], valid_pred, average="macro")
        fold_scores.append(float(fold_score))
        print(f"fold={fold_idx} macro_f1={fold_score:.6f}")

        if args.save_fold_models:
            joblib.dump(model, run_dir / f"fold_{fold_idx}.joblib")

    local_score = float(f1_score(y, oof_proba.argmax(axis=1), average="macro"))

    oof_pred_labels = encoder.inverse_transform(oof_proba.argmax(axis=1))
    oof_frame = pd.DataFrame(
        {
            "id": train_df["id"],
            "true_label": train_df["label"],
            "pred_label": oof_pred_labels,
        }
    )
    if train_scores is not None:
        oof_frame["adv_score"] = train_scores
        oof_frame["sample_weight"] = sample_weights
    oof_frame.to_csv(run_dir / "oof_predictions.csv", index=False, encoding="utf-8")

    np.save(run_dir / "oof_proba.npy", oof_proba)
    np.save(run_dir / "test_proba.npy", test_proba)

    if train_scores is not None:
        np.save(run_dir / "train_adv_scores.npy", train_scores)
        np.save(run_dir / "test_adv_scores.npy", test_scores)

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "strategy": args.strategy,
        "seed": args.seed,
        "folds": args.folds,
        "n_estimators": args.n_estimators,
        "transform": args.transform,
        "class_weight": args.class_weight,
        "drop_id": True,
        "classes": encoder.classes_.tolist(),
        "features": x_train.columns.tolist(),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "feature_count": int(x_train.shape[1]),
        "fold_scores": fold_scores,
        "local_macro_f1": local_score,
    }
    save_json(run_dir / "metadata.json", metadata)

    print(json.dumps({"run_id": run_id, "local_macro_f1": local_score, "fold_scores": fold_scores}, ensure_ascii=False))


if __name__ == "__main__":
    main()
