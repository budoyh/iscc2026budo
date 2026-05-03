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
    parser = argparse.ArgumentParser(description="Train LightGBM with class-conditional weighted quantile target augmentation.")
    parser.add_argument("--source-runs", nargs="+", required=True)
    parser.add_argument("--source-weights", nargs="+", type=float, required=True)
    parser.add_argument("--soft-alpha", type=float, default=1.0)
    parser.add_argument("--target-power", type=float, default=1.0)
    parser.add_argument("--quantile-grid", type=int, default=201)
    parser.add_argument("--aug-weight", type=float, default=0.7)
    parser.add_argument("--pseudo-threshold", type=float, default=0.90)
    parser.add_argument("--pseudo-weight", type=float, default=0.55)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=900)
    parser.add_argument("--run-id", type=str, default=None)
    return parser.parse_args()


def load_source_blend(run_ids: list[str], weights: list[float]) -> tuple[list[str], np.ndarray, np.ndarray]:
    weights_arr = np.asarray(weights, dtype=np.float64)
    weights_arr /= weights_arr.sum()
    classes = None
    oof = None
    test = None
    for run_id, weight in zip(run_ids, weights_arr):
        run_dir = MODELS / run_id
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        current_oof = np.load(run_dir / "oof_proba.npy")
        current_test = np.load(run_dir / "test_proba.npy")
        if classes is None:
            classes = metadata["classes"]
            oof = np.zeros_like(current_oof, dtype=np.float64)
            test = np.zeros_like(current_test, dtype=np.float64)
        elif metadata["classes"] != classes:
            raise ValueError(f"class order mismatch for {run_id}")
        oof += weight * current_oof
        test += weight * current_test
    assert classes is not None and oof is not None and test is not None
    return classes, oof, test


def apply_uniform_prior(proba: np.ndarray, alpha: float) -> np.ndarray:
    target = np.full(proba.shape[1], 1.0 / proba.shape[1], dtype=np.float64)
    current = np.clip(proba.mean(axis=0), 1e-15, 1.0)
    adjusted = proba * ((target / current) ** alpha)
    adjusted /= adjusted.sum(axis=1, keepdims=True)
    return adjusted


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights) - 0.5 * sorted_weights
    cumulative /= np.clip(sorted_weights.sum(), 1e-12, None)
    return np.interp(quantiles, cumulative, sorted_values, left=sorted_values[0], right=sorted_values[-1])


def build_target_quantiles(
    x_test: np.ndarray,
    test_proba: np.ndarray,
    power: float,
    q_grid: np.ndarray,
) -> np.ndarray:
    class_count = test_proba.shape[1]
    target_q = np.empty((class_count, len(q_grid), x_test.shape[1]), dtype=np.float32)
    for cls in range(class_count):
        weights = np.clip(test_proba[:, cls], 1e-9, 1.0) ** power
        for feature_idx in range(x_test.shape[1]):
            target_q[cls, :, feature_idx] = weighted_quantile(x_test[:, feature_idx], weights, q_grid).astype(np.float32)
    return target_q


def augment_quantile(
    x_fold: np.ndarray,
    y_fold: np.ndarray,
    class_count: int,
    target_q: np.ndarray,
    q_grid: np.ndarray,
) -> np.ndarray:
    out = np.empty_like(x_fold, dtype=np.float32)
    for cls in range(class_count):
        class_positions = np.where(y_fold == cls)[0]
        if len(class_positions) == 0:
            continue
        cls_x = x_fold[class_positions]
        cls_out = np.empty_like(cls_x, dtype=np.float32)
        denom = max(len(cls_x) - 1, 1)
        for feature_idx in range(x_fold.shape[1]):
            source_sorted = np.sort(cls_x[:, feature_idx])
            ranks = np.searchsorted(source_sorted, cls_x[:, feature_idx], side="left").astype(np.float64) / denom
            ranks = np.clip(ranks, q_grid[0], q_grid[-1])
            cls_out[:, feature_idx] = np.interp(ranks, q_grid, target_q[cls, :, feature_idx]).astype(np.float32)
        out[class_positions] = cls_out
    return out


def build_model(seed: int, num_classes: int, n_estimators: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="multiclass",
        num_class=num_classes,
        n_estimators=n_estimators,
        learning_rate=0.045,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.88,
        colsample_bytree=0.88,
        reg_lambda=1.2,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def main() -> None:
    args = parse_args()
    MODELS.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(RAW / "train_data.csv")
    test = pd.read_csv(RAW / "test_data.csv")
    features = [col for col in train.columns if col not in {"id", "label"}]
    x = train[features].to_numpy(dtype=np.float32)
    x_test = test[features].to_numpy(dtype=np.float32)

    encoder = LabelEncoder()
    y = encoder.fit_transform(train["label"]).astype(np.int64)
    classes, _, source_test = load_source_blend(args.source_runs, args.source_weights)
    if classes != encoder.classes_.tolist():
        raise ValueError("source class order mismatch")
    source_test = apply_uniform_prior(source_test, args.soft_alpha)
    class_count = len(classes)
    q_grid = np.linspace(0.002, 0.998, args.quantile_grid, dtype=np.float64)
    target_q = build_target_quantiles(x_test, source_test, args.target_power, q_grid)

    pseudo_conf = source_test.max(axis=1)
    pseudo_mask = pseudo_conf >= args.pseudo_threshold
    pseudo_y = source_test[pseudo_mask].argmax(axis=1)
    pseudo_x = x_test[pseudo_mask]

    run_id = args.run_id or (
        f"quantile_target_aug_pow{str(args.target_power).replace('.', 'p')}"
        f"_q{args.quantile_grid}_aug{str(args.aug_weight).replace('.', 'p')}"
        f"_t{str(args.pseudo_threshold).replace('.', 'p')}_w{str(args.pseudo_weight).replace('.', 'p')}_s{args.seed}"
    )
    run_dir = MODELS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof = np.zeros((len(train), class_count), dtype=np.float64)
    test_proba = np.zeros((len(test), class_count), dtype=np.float64)
    fold_scores: list[float] = []

    for fold, (tr_idx, va_idx) in enumerate(cv.split(x, y), start=1):
        fold_x = x[tr_idx]
        fold_y = y[tr_idx]
        aug_x = augment_quantile(fold_x, fold_y, class_count, target_q, q_grid)
        train_x = np.concatenate([fold_x, aug_x, pseudo_x], axis=0)
        train_y = np.concatenate([fold_y, fold_y, pseudo_y], axis=0)
        weights = np.concatenate(
            [
                np.ones(len(fold_x), dtype=np.float64),
                np.full(len(aug_x), args.aug_weight, dtype=np.float64),
                args.pseudo_weight * np.clip(pseudo_conf[pseudo_mask], 0.5, 1.0),
            ]
        )
        model = build_model(args.seed + fold - 1, class_count, args.n_estimators)
        model.fit(train_x, train_y, sample_weight=weights)
        valid_proba = model.predict_proba(x[va_idx])
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
        "strategy": "quantile_target_aug_lgbm",
        "source_runs": args.source_runs,
        "source_weights": args.source_weights,
        "soft_alpha": args.soft_alpha,
        "target_power": args.target_power,
        "quantile_grid": args.quantile_grid,
        "aug_weight": args.aug_weight,
        "pseudo_threshold": args.pseudo_threshold,
        "pseudo_weight": args.pseudo_weight,
        "pseudo_count": int(pseudo_mask.sum()),
        "pseudo_confidence_mean": float(pseudo_conf[pseudo_mask].mean()) if np.any(pseudo_mask) else None,
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
    print(json.dumps({"run_id": run_id, "local_macro_f1": local_score, "pseudo_count": int(pseudo_mask.sum())}))


if __name__ == "__main__":
    main()
