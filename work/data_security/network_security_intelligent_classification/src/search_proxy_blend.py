from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw" / "data"
MODELS = ROOT / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid-search probability blends using target-like proxy validation.")
    parser.add_argument("--domain-estimators", type=int, default=500)
    return parser.parse_args()


def apply_uniform_prior(proba: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    target = np.full(proba.shape[1], 1.0 / proba.shape[1], dtype=np.float64)
    current = np.clip(proba.mean(axis=0), 1e-15, 1.0)
    out = proba * ((target / current) ** alpha)
    out /= out.sum(axis=1, keepdims=True)
    return out


def load_run(run_id: str, classes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    metadata = json.loads((MODELS / run_id / "metadata.json").read_text(encoding="utf-8"))
    if metadata["classes"] != classes:
        raise ValueError(f"class order mismatch for {run_id}")
    return np.load(MODELS / run_id / "oof_proba.npy").astype(np.float64), np.load(MODELS / run_id / "test_proba.npy").astype(np.float64)


def load_report(path: Path, classes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    report = json.loads(path.read_text(encoding="utf-8"))
    weights = np.asarray(report["weights"], dtype=np.float64)
    weights /= weights.sum()
    oof = None
    test = None
    for run_id, weight in zip(report["run_ids"], weights):
        current_oof, current_test = load_run(run_id, classes)
        if oof is None:
            oof = np.zeros_like(current_oof, dtype=np.float64)
            test = np.zeros_like(current_test, dtype=np.float64)
        oof += weight * current_oof
        test += weight * current_test
    assert oof is not None and test is not None
    return oof, test


def build_domain_scores(x_train: pd.DataFrame, x_test: pd.DataFrame, n_estimators: int) -> tuple[np.ndarray, float]:
    x_all = pd.concat([x_train, x_test], ignore_index=True)
    y_domain = np.r_[np.zeros(len(x_train), dtype=np.int64), np.ones(len(x_test), dtype=np.int64)]
    scores = np.zeros(len(x_all), dtype=np.float64)
    cv = KFold(n_splits=3, shuffle=True, random_state=2026)
    for tr_idx, va_idx in cv.split(x_all):
        model = LGBMClassifier(
            objective="binary",
            n_estimators=n_estimators,
            learning_rate=0.06,
            num_leaves=63,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=2026,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(x_all.iloc[tr_idx], y_domain[tr_idx])
        scores[va_idx] = model.predict_proba(x_all.iloc[va_idx])[:, 1]
    return scores[: len(x_train)], float(roc_auc_score(y_domain, scores))


def main() -> None:
    args = parse_args()
    train = pd.read_csv(RAW / "train_data.csv")
    test = pd.read_csv(RAW / "test_data.csv")
    features = [col for col in train.columns if col not in {"id", "label"}]
    encoder = LabelEncoder()
    y = encoder.fit_transform(train["label"])
    classes = encoder.classes_.tolist()
    train_scores, domain_auc = build_domain_scores(train[features], test[features], args.domain_estimators)
    masks = {
        "full": np.ones(len(train), dtype=bool),
        "top10": train_scores >= np.quantile(train_scores, 0.9),
        "top20": train_scores >= np.quantile(train_scores, 0.8),
        "top30": train_scores >= np.quantile(train_scores, 0.7),
        "top50": train_scores >= np.quantile(train_scores, 0.5),
    }

    sources = {
        "c01": load_report(ROOT / "upload_ready_breakthrough2" / "c01_b01_iter_target_aug_high.json", classes),
        "c02": load_report(ROOT / "upload_ready_breakthrough2" / "c02_b01_iter_target_aug_safe.json", classes),
        "target_b01_pow1p2": load_run("target_aug_lgbm_b01_pow1p2_aug080_t093_w055_f5_s42", classes),
        "target_b01_pow0p8": load_run("target_aug_lgbm_b01_pow0p8_aug100_t090_w065_f5_s42", classes),
    }
    optional_runs = [
        "target_aug_domain_c01_pow1p2_aug080_t093_w055_g050_n700_f5_s42",
        "local_cluster_diag_c01_k3_pow1p2_aug075_t093_w055_n700_f5_s42",
    ]
    for run_id in optional_runs:
        if (MODELS / run_id / "metadata.json").exists():
            sources[run_id] = load_run(run_id, classes)

    rows = []
    weight_grid = [0.0, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    base_oof, base_test = sources["c01"]
    for name, (other_oof, other_test) in sources.items():
        if name == "c01":
            continue
        for other_weight in weight_grid:
            c01_weight = 1.0 - other_weight
            oof = c01_weight * base_oof + other_weight * other_oof
            test_proba = c01_weight * base_test + other_weight * other_test
            oof = apply_uniform_prior(oof, 1.0)
            test_proba = apply_uniform_prior(test_proba, 1.0)
            pred = oof.argmax(axis=1)
            row = {"members": f"c01+{name}", "weights": f"{c01_weight:.4f},{other_weight:.4f}"}
            for mask_name, mask in masks.items():
                row[mask_name] = f1_score(y[mask], pred[mask], average="macro")
            row["test_entropy"] = float(
                (-np.clip(test_proba, 1e-15, 1.0) * np.log(np.clip(test_proba, 1e-15, 1.0))).sum(axis=1).mean()
            )
            row["test_counts"] = np.bincount(test_proba.argmax(axis=1), minlength=len(classes)).astype(int).tolist()
            rows.append(row)

    frame = pd.DataFrame(rows)
    print("DOMAIN_AUC", round(domain_auc, 6))
    print("TOP_BY_TOP20")
    print(frame.sort_values(["top20", "top30", "full"], ascending=False).head(25).round(6).to_string(index=False))
    print("TOP_BY_TOP30")
    print(frame.sort_values(["top30", "top20", "full"], ascending=False).head(25).round(6).to_string(index=False))


if __name__ == "__main__":
    main()
