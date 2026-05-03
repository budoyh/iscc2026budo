from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw" / "data"
MODELS = ROOT / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LightGBM on combined-domain PCA features.")
    parser.add_argument("--components", type=int, default=32)
    parser.add_argument("--drop-pattern", action="store_true")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=700)
    parser.add_argument("--run-id", type=str, default=None)
    return parser.parse_args()


PATTERN_FEATURES = {
    "pattern_transition_ratio",
    "pattern_diversity_ratio",
    "pattern_switch_frequency",
    "pattern_mix_density",
    "pattern_change_ratio",
    "pattern_concentration",
}


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
    if args.drop_pattern:
        features = [col for col in features if col not in PATTERN_FEATURES]
    x_raw = train[features].to_numpy(dtype=np.float32)
    test_raw = test[features].to_numpy(dtype=np.float32)
    scaler = StandardScaler().fit(np.vstack([x_raw, test_raw]))
    z_all = scaler.transform(np.vstack([x_raw, test_raw]))
    pca = PCA(n_components=args.components, whiten=True, random_state=args.seed)
    pca_all = pca.fit_transform(z_all).astype(np.float32)
    x = pca_all[: len(train)]
    x_test = pca_all[len(train) :]
    columns = [f"pca_{idx:02d}" for idx in range(args.components)]

    encoder = LabelEncoder()
    y = encoder.fit_transform(train["label"]).astype(np.int64)
    classes = encoder.classes_.tolist()
    suffix = "_drop_pattern" if args.drop_pattern else ""
    run_id = args.run_id or f"lgbm_pca{args.components}{suffix}_n{args.n_estimators}_f{args.folds}_s{args.seed}"
    run_dir = MODELS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof = np.zeros((len(train), len(classes)), dtype=np.float64)
    test_proba = np.zeros((len(test), len(classes)), dtype=np.float64)
    fold_scores: list[float] = []
    for fold, (tr_idx, va_idx) in enumerate(cv.split(x, y), start=1):
        model = build_model(args.seed + fold - 1, len(classes), args.n_estimators)
        model.fit(x[tr_idx], y[tr_idx])
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
        "strategy": "pca_lgbm",
        "components": args.components,
        "drop_pattern": args.drop_pattern,
        "explained_variance_sum": float(pca.explained_variance_ratio_.sum()),
        "seed": args.seed,
        "folds": args.folds,
        "classes": classes,
        "features": columns,
        "source_features": features,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "feature_count": len(columns),
        "fold_scores": fold_scores,
        "local_macro_f1": local_score,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"run_id": run_id, "local_macro_f1": local_score, "explained_variance_sum": metadata["explained_variance_sum"]}))


if __name__ == "__main__":
    main()
