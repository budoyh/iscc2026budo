from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw" / "data"
MODELS = ROOT / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LightGBM with local class-conditional target adaptation.")
    parser.add_argument("--source-runs", nargs="+", required=True)
    parser.add_argument("--source-weights", nargs="+", type=float, required=True)
    parser.add_argument("--soft-alpha", type=float, default=1.0)
    parser.add_argument("--target-power", type=float, default=1.2)
    parser.add_argument("--mode", choices=["cov_coral", "cluster_diag"], default="cov_coral")
    parser.add_argument("--cov-shrink", type=float, default=0.4)
    parser.add_argument("--clusters", type=int, default=3)
    parser.add_argument("--cluster-min-weight", type=float, default=40.0)
    parser.add_argument("--aug-weight", type=float, default=0.7)
    parser.add_argument("--pseudo-threshold", type=float, default=0.93)
    parser.add_argument("--pseudo-weight", type=float, default=0.55)
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


def weighted_mean_std(x: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / np.clip(weights.sum(), 1e-12, None)
    mean = weights @ x
    second = weights @ (x * x)
    std = np.sqrt(np.maximum(second - mean * mean, 1e-8))
    return mean.astype(np.float64), std.astype(np.float64)


def weighted_cov(x: np.ndarray, weights: np.ndarray, mean: np.ndarray, shrink: float) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / np.clip(weights.sum(), 1e-12, None)
    centered = x - mean
    cov = (centered * weights[:, None]).T @ centered
    diag = np.diag(np.diag(cov))
    cov = (1.0 - shrink) * cov + shrink * diag
    return cov + 1e-4 * np.eye(x.shape[1])


def matrix_sqrt_and_inv_sqrt(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eigh(cov)
    values = np.maximum(values, 1e-6)
    sqrt = vectors @ np.diag(np.sqrt(values)) @ vectors.T
    inv_sqrt = vectors @ np.diag(1.0 / np.sqrt(values)) @ vectors.T
    return sqrt, inv_sqrt


def build_target_coral_stats(
    x_test: np.ndarray,
    test_proba: np.ndarray,
    power: float,
    cov_shrink: float,
) -> list[dict[str, np.ndarray]]:
    stats = []
    for cls in range(test_proba.shape[1]):
        weights = np.clip(test_proba[:, cls], 1e-9, 1.0) ** power
        mean, _ = weighted_mean_std(x_test, weights)
        cov = weighted_cov(x_test, weights, mean, cov_shrink)
        target_sqrt, _ = matrix_sqrt_and_inv_sqrt(cov)
        stats.append({"mean": mean, "sqrt": target_sqrt})
    return stats


def augment_cov_coral(
    x_fold: np.ndarray,
    y_fold: np.ndarray,
    class_count: int,
    target_stats: list[dict[str, np.ndarray]],
    cov_shrink: float,
) -> np.ndarray:
    out = np.empty_like(x_fold, dtype=np.float32)
    global_mean = x_fold.mean(axis=0)
    global_cov = np.cov(x_fold - global_mean, rowvar=False) + 1e-4 * np.eye(x_fold.shape[1])
    for cls in range(class_count):
        mask = y_fold == cls
        if not np.any(mask):
            continue
        cls_x = x_fold[mask].astype(np.float64)
        cls_mean = cls_x.mean(axis=0)
        shrink = min(1.0, mask.sum() / 800.0)
        src_mean = shrink * cls_mean + (1.0 - shrink) * global_mean
        cls_cov = np.cov(cls_x - cls_mean, rowvar=False) + 1e-4 * np.eye(x_fold.shape[1])
        src_cov = shrink * cls_cov + (1.0 - shrink) * global_cov
        diag = np.diag(np.diag(src_cov))
        src_cov = (1.0 - cov_shrink) * src_cov + cov_shrink * diag
        _, src_inv_sqrt = matrix_sqrt_and_inv_sqrt(src_cov)
        out[mask] = ((cls_x - src_mean) @ src_inv_sqrt @ target_stats[cls]["sqrt"] + target_stats[cls]["mean"]).astype(
            np.float32
        )
    return out


def build_target_cluster_stats(
    x_test: np.ndarray,
    z_test: np.ndarray,
    test_proba: np.ndarray,
    class_count: int,
    clusters: int,
    power: float,
    min_weight: float,
    seed: int,
) -> list[dict[str, np.ndarray]]:
    stats = []
    for cls in range(class_count):
        weights = np.clip(test_proba[:, cls], 1e-9, 1.0) ** power
        kmeans = KMeans(n_clusters=clusters, n_init=5, random_state=seed + cls)
        kmeans.fit(z_test, sample_weight=weights)
        assignment = kmeans.predict(z_test)
        global_mean, global_std = weighted_mean_std(x_test, weights)
        means = np.zeros((clusters, x_test.shape[1]), dtype=np.float64)
        stds = np.zeros((clusters, x_test.shape[1]), dtype=np.float64)
        effective_weights = np.zeros(clusters, dtype=np.float64)
        for cluster_id in range(clusters):
            mask = assignment == cluster_id
            cluster_weights = weights[mask]
            effective_weights[cluster_id] = cluster_weights.sum()
            if effective_weights[cluster_id] < min_weight:
                means[cluster_id] = global_mean
                stds[cluster_id] = global_std
                continue
            mean, std = weighted_mean_std(x_test[mask], cluster_weights)
            shrink = min(1.0, effective_weights[cluster_id] / (min_weight * 4.0))
            means[cluster_id] = shrink * mean + (1.0 - shrink) * global_mean
            stds[cluster_id] = shrink * std + (1.0 - shrink) * global_std
        stats.append(
            {
                "centers": kmeans.cluster_centers_.astype(np.float64),
                "means": means,
                "stds": np.maximum(stds, 1e-4),
                "effective_weights": effective_weights,
            }
        )
    return stats


def augment_cluster_diag(
    x_fold: np.ndarray,
    z_fold: np.ndarray,
    y_fold: np.ndarray,
    class_count: int,
    target_stats: list[dict[str, np.ndarray]],
    clusters: int,
    seed: int,
) -> np.ndarray:
    out = np.empty_like(x_fold, dtype=np.float32)
    global_mean = x_fold.mean(axis=0)
    global_std = np.maximum(x_fold.std(axis=0), 1e-4)
    for cls in range(class_count):
        mask = y_fold == cls
        if not np.any(mask):
            continue
        cls_x = x_fold[mask]
        cls_z = z_fold[mask]
        local_k = min(clusters, max(1, len(cls_x) // 150))
        if local_k == 1:
            cls_mean = cls_x.mean(axis=0)
            cls_std = np.maximum(cls_x.std(axis=0), 1e-4)
            target_mean = target_stats[cls]["means"].mean(axis=0)
            target_std = target_stats[cls]["stds"].mean(axis=0)
            out[mask] = ((cls_x - cls_mean) / cls_std * target_std + target_mean).astype(np.float32)
            continue
        src_kmeans = KMeans(n_clusters=local_k, n_init=5, random_state=seed + cls)
        source_assignment = src_kmeans.fit_predict(cls_z)
        target_centers = target_stats[cls]["centers"]
        distance = ((src_kmeans.cluster_centers_[:, None, :] - target_centers[None, :, :]) ** 2).sum(axis=2)
        source_ids, target_ids = linear_sum_assignment(distance)
        mapping = {int(source_id): int(target_id) for source_id, target_id in zip(source_ids, target_ids)}
        for source_id in range(local_k):
            local_mask = source_assignment == source_id
            if not np.any(local_mask):
                continue
            src_x = cls_x[local_mask]
            shrink = min(1.0, len(src_x) / 250.0)
            src_mean = shrink * src_x.mean(axis=0) + (1.0 - shrink) * global_mean
            src_std = shrink * np.maximum(src_x.std(axis=0), 1e-4) + (1.0 - shrink) * global_std
            target_id = mapping.get(source_id, int(np.argmin(distance[source_id])))
            target_mean = target_stats[cls]["means"][target_id]
            target_std = target_stats[cls]["stds"][target_id]
            class_positions = np.where(mask)[0]
            out[class_positions[local_mask]] = ((src_x - src_mean) / src_std * target_std + target_mean).astype(
                np.float32
            )
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

    pseudo_conf = source_test.max(axis=1)
    pseudo_mask = pseudo_conf >= args.pseudo_threshold
    pseudo_y = source_test[pseudo_mask].argmax(axis=1)
    pseudo_x = x_test[pseudo_mask]

    scaler = StandardScaler().fit(np.vstack([x, x_test]))
    z = scaler.transform(x).astype(np.float32)
    z_test = scaler.transform(x_test).astype(np.float32)

    target_stats: list[dict[str, np.ndarray]]
    if args.mode == "cov_coral":
        target_stats = build_target_coral_stats(x_test, source_test, args.target_power, args.cov_shrink)
    else:
        target_stats = build_target_cluster_stats(
            x_test,
            z_test,
            source_test,
            class_count,
            args.clusters,
            args.target_power,
            args.cluster_min_weight,
            args.seed,
        )

    run_id = args.run_id or (
        f"local_target_aug_{args.mode}_pow{str(args.target_power).replace('.', 'p')}"
        f"_aug{str(args.aug_weight).replace('.', 'p')}_t{str(args.pseudo_threshold).replace('.', 'p')}"
        f"_w{str(args.pseudo_weight).replace('.', 'p')}_s{args.seed}"
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
        if args.mode == "cov_coral":
            aug_x = augment_cov_coral(fold_x, fold_y, class_count, target_stats, args.cov_shrink)
        else:
            aug_x = augment_cluster_diag(
                fold_x,
                z[tr_idx],
                fold_y,
                class_count,
                target_stats,
                args.clusters,
                args.seed + fold * 100,
            )

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
        "strategy": "local_target_aug_lgbm",
        "mode": args.mode,
        "source_runs": args.source_runs,
        "source_weights": args.source_weights,
        "soft_alpha": args.soft_alpha,
        "target_power": args.target_power,
        "cov_shrink": args.cov_shrink,
        "clusters": args.clusters,
        "cluster_min_weight": args.cluster_min_weight,
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
