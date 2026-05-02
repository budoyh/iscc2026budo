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
    parser = argparse.ArgumentParser(description="Train transductive LightGBM with high-confidence pseudo labels.")
    parser.add_argument("--source-runs", nargs="+", required=True)
    parser.add_argument("--source-weights", nargs="+", type=float, required=True)
    parser.add_argument("--threshold", type=float, default=0.98)
    parser.add_argument("--pseudo-weight", type=float, default=0.35)
    parser.add_argument("--soft-prior", choices=["none", "uniform", "train"], default="none")
    parser.add_argument("--soft-alpha", type=float, default=1.0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=1200)
    parser.add_argument("--run-id", type=str, default=None)
    return parser.parse_args()


def load_source_blend(run_ids: list[str], weights: list[float]) -> tuple[list[str], np.ndarray, np.ndarray]:
    weights_arr = np.asarray(weights, dtype=np.float64)
    weights_arr = weights_arr / weights_arr.sum()
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
            raise ValueError("class ordering mismatch")
        oof += weight * np.load(run_dir / "oof_proba.npy")
        test += weight * np.load(run_dir / "test_proba.npy")
    assert classes is not None and oof is not None and test is not None
    return classes, oof, test


def apply_soft_prior(proba: np.ndarray, target_prior: np.ndarray, alpha: float) -> np.ndarray:
    current_prior = np.clip(proba.mean(axis=0), 1e-15, 1.0)
    factor = (target_prior / current_prior) ** alpha
    adjusted = proba * factor
    adjusted /= adjusted.sum(axis=1, keepdims=True)
    return adjusted


def main() -> None:
    args = parse_args()
    MODELS.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(RAW / "train_data.csv")
    test = pd.read_csv(RAW / "test_data.csv")
    x = train.drop(columns=["label", "id"])
    x_test = test.drop(columns=["id"])

    encoder = LabelEncoder()
    y = encoder.fit_transform(train["label"])
    classes, _, source_test = load_source_blend(args.source_runs, args.source_weights)
    if classes != encoder.classes_.tolist():
        raise ValueError("source class order mismatch")
    if args.soft_prior != "none":
        if args.soft_prior == "uniform":
            target_prior = np.full(len(classes), 1.0 / len(classes))
        else:
            target_prior = train["label"].value_counts(normalize=True).reindex(classes).to_numpy()
        source_test = apply_soft_prior(source_test, target_prior=target_prior, alpha=args.soft_alpha)

    confidence = source_test.max(axis=1)
    pseudo_mask = confidence >= args.threshold
    pseudo_x = x_test.loc[pseudo_mask].reset_index(drop=True)
    pseudo_y = source_test[pseudo_mask].argmax(axis=1)
    pseudo_conf = confidence[pseudo_mask]

    run_id = args.run_id or (
        f"pseudo_lgbm_t{str(args.threshold).replace('.', 'p')}_w{str(args.pseudo_weight).replace('.', 'p')}_s{args.seed}"
    )
    run_dir = MODELS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof = np.zeros((len(train), len(classes)), dtype=np.float64)
    test_proba = np.zeros((len(test), len(classes)), dtype=np.float64)
    fold_scores: list[float] = []

    for fold, (tr_idx, va_idx) in enumerate(cv.split(x, y), start=1):
        fold_x = pd.concat([x.iloc[tr_idx].reset_index(drop=True), pseudo_x], axis=0, ignore_index=True)
        fold_y = np.concatenate([y[tr_idx], pseudo_y])
        pseudo_weights = args.pseudo_weight * np.clip(pseudo_conf, 0.5, 1.0)
        weights = np.concatenate([np.ones(len(tr_idx), dtype=np.float64), pseudo_weights])

        model = LGBMClassifier(
            objective="multiclass",
            num_class=len(classes),
            n_estimators=args.n_estimators,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=20,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            random_state=args.seed + fold - 1,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(fold_x, fold_y, sample_weight=weights)
        valid_proba = model.predict_proba(x.iloc[va_idx])
        oof[va_idx] = valid_proba
        test_proba += model.predict_proba(x_test) / args.folds
        score = f1_score(y[va_idx], valid_proba.argmax(axis=1), average="macro")
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
        "run_id": run_id,
        "strategy": "pseudo_lgbm",
        "source_runs": args.source_runs,
        "source_weights": args.source_weights,
        "threshold": args.threshold,
        "pseudo_weight": args.pseudo_weight,
        "soft_prior": args.soft_prior,
        "soft_alpha": args.soft_alpha,
        "pseudo_count": int(pseudo_mask.sum()),
        "pseudo_confidence_mean": float(pseudo_conf.mean()) if len(pseudo_conf) else None,
        "seed": args.seed,
        "folds": args.folds,
        "classes": classes,
        "features": x.columns.tolist(),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "feature_count": int(x.shape[1]),
        "fold_scores": fold_scores,
        "local_macro_f1": local_score,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"run_id": run_id, "local_macro_f1": local_score, "pseudo_count": int(pseudo_mask.sum())}))


if __name__ == "__main__":
    main()
