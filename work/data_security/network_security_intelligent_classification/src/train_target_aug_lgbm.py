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
    parser = argparse.ArgumentParser(description="Train LightGBM with class-conditional target-domain augmentation.")
    parser.add_argument("--source-runs", nargs="+", required=True)
    parser.add_argument("--source-weights", nargs="+", type=float, required=True)
    parser.add_argument("--soft-alpha", type=float, default=1.0)
    parser.add_argument("--target-power", type=float, default=2.0)
    parser.add_argument("--aug-weight", type=float, default=0.45)
    parser.add_argument("--pseudo-threshold", type=float, default=0.97)
    parser.add_argument("--pseudo-weight", type=float, default=0.35)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=1400)
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
        meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        if classes is None:
            classes = meta["classes"]
            oof = np.zeros_like(np.load(run_dir / "oof_proba.npy"), dtype=np.float64)
            test = np.zeros_like(np.load(run_dir / "test_proba.npy"), dtype=np.float64)
        elif classes != meta["classes"]:
            raise ValueError(f"class order mismatch for {run_id}")
        oof += weight * np.load(run_dir / "oof_proba.npy")
        test += weight * np.load(run_dir / "test_proba.npy")
    assert classes is not None and oof is not None and test is not None
    return classes, oof, test


def apply_uniform_prior(proba: np.ndarray, alpha: float) -> np.ndarray:
    target = np.full(proba.shape[1], 1.0 / proba.shape[1], dtype=np.float64)
    current = np.clip(proba.mean(axis=0), 1e-15, 1.0)
    adjusted = proba * ((target / current) ** alpha)
    adjusted /= adjusted.sum(axis=1, keepdims=True)
    return adjusted


def weighted_target_stats(x_test: np.ndarray, test_proba: np.ndarray, power: float) -> tuple[np.ndarray, np.ndarray]:
    weights = np.clip(test_proba, 1e-9, 1.0) ** power
    weights /= np.clip(weights.sum(axis=0, keepdims=True), 1e-12, None)
    mean = weights.T @ x_test
    second = weights.T @ (x_test * x_test)
    var = np.maximum(second - mean * mean, 1e-8)
    return mean, np.sqrt(var)


def augment_to_target(
    x_fold: np.ndarray,
    y_fold: np.ndarray,
    num_classes: int,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> np.ndarray:
    out = np.empty_like(x_fold, dtype=np.float32)
    global_mean = x_fold.mean(axis=0)
    global_std = np.maximum(x_fold.std(axis=0), 1e-4)
    for cls in range(num_classes):
        mask = y_fold == cls
        if not np.any(mask):
            continue
        cls_x = x_fold[mask]
        cls_mean = cls_x.mean(axis=0)
        cls_std = np.maximum(cls_x.std(axis=0), 1e-4)
        # Shrink rare-class statistics toward the fold-global values for stability.
        shrink = min(1.0, mask.sum() / 800.0)
        src_mean = shrink * cls_mean + (1.0 - shrink) * global_mean
        src_std = shrink * cls_std + (1.0 - shrink) * global_std
        out[mask] = ((cls_x - src_mean) / src_std * target_std[cls] + target_mean[cls]).astype(np.float32)
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

    target_mean, target_std = weighted_target_stats(x_test, source_test, args.target_power)
    pseudo_conf = source_test.max(axis=1)
    pseudo_mask = pseudo_conf >= args.pseudo_threshold
    pseudo_y = source_test[pseudo_mask].argmax(axis=1)
    pseudo_x = x_test[pseudo_mask]

    run_id = args.run_id or (
        f"target_aug_lgbm_pow{str(args.target_power).replace('.', 'p')}_aug{str(args.aug_weight).replace('.', 'p')}"
        f"_t{str(args.pseudo_threshold).replace('.', 'p')}_w{str(args.pseudo_weight).replace('.', 'p')}_s{args.seed}"
    )
    run_dir = MODELS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof = np.zeros((len(train), len(classes)), dtype=np.float64)
    test_proba = np.zeros((len(test), len(classes)), dtype=np.float64)
    fold_scores: list[float] = []

    for fold, (tr_idx, va_idx) in enumerate(cv.split(x, y), start=1):
        fold_x = x[tr_idx]
        fold_y = y[tr_idx]
        aug_x = augment_to_target(fold_x, fold_y, len(classes), target_mean, target_std)

        train_x = np.concatenate([fold_x, aug_x, pseudo_x], axis=0)
        train_y = np.concatenate([fold_y, fold_y, pseudo_y], axis=0)
        weights = np.concatenate(
            [
                np.ones(len(fold_x), dtype=np.float64),
                np.full(len(aug_x), args.aug_weight, dtype=np.float64),
                args.pseudo_weight * np.clip(pseudo_conf[pseudo_mask], 0.5, 1.0),
            ]
        )

        model = build_model(args.seed + fold - 1, len(classes), args.n_estimators)
        model.fit(train_x, train_y, sample_weight=weights)
        valid_proba = model.predict_proba(x[va_idx])
        fold_test = model.predict_proba(x_test)
        oof[va_idx] = valid_proba
        test_proba += fold_test / args.folds
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
        "strategy": "target_aug_lgbm",
        "source_runs": args.source_runs,
        "source_weights": args.source_weights,
        "soft_alpha": args.soft_alpha,
        "target_power": args.target_power,
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
