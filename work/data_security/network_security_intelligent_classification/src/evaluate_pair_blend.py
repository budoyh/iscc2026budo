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
    parser = argparse.ArgumentParser(description="Evaluate a base submission report blended with one model run.")
    parser.add_argument("--base-report", required=True)
    parser.add_argument("--other-run", required=True)
    parser.add_argument("--weights", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--domain-estimators", type=int, default=350)
    return parser.parse_args()


def apply_uniform_prior(proba: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    target = np.full(proba.shape[1], 1.0 / proba.shape[1], dtype=np.float64)
    current = np.clip(proba.mean(axis=0), 1e-15, 1.0)
    out = proba * ((target / current) ** alpha)
    out /= out.sum(axis=1, keepdims=True)
    return out


def load_run(run_id: str, classes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    run_dir = MODELS / run_id
    meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    if meta["classes"] != classes:
        raise ValueError(f"class order mismatch: {run_id}")
    return np.load(run_dir / "oof_proba.npy").astype(np.float64), np.load(run_dir / "test_proba.npy").astype(np.float64)


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
    if report.get("soft_prior") == "uniform":
        oof = apply_uniform_prior(oof, float(report.get("soft_alpha", 1.0)))
        test = apply_uniform_prior(test, float(report.get("soft_alpha", 1.0)))
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
    base_oof, base_test = load_report(ROOT / args.base_report, classes)
    other_oof, other_test = load_run(args.other_run, classes)
    domain_scores, domain_auc = build_domain_scores(train[features], test[features], args.domain_estimators)
    masks = {
        "full": np.ones(len(train), dtype=bool),
        "top10": domain_scores >= np.quantile(domain_scores, 0.9),
        "top20": domain_scores >= np.quantile(domain_scores, 0.8),
        "top30": domain_scores >= np.quantile(domain_scores, 0.7),
        "top50": domain_scores >= np.quantile(domain_scores, 0.5),
    }
    rows = []
    base_pred = base_oof.argmax(axis=1)
    base_test_pred = base_test.argmax(axis=1)
    for weight in args.weights:
        oof = (1.0 - weight) * base_oof + weight * other_oof
        test_proba = (1.0 - weight) * base_test + weight * other_test
        oof = apply_uniform_prior(oof, 1.0)
        test_proba = apply_uniform_prior(test_proba, 1.0)
        pred = oof.argmax(axis=1)
        test_pred = test_proba.argmax(axis=1)
        row = {
            "other_weight": weight,
            "changed_oof": int(np.sum(pred != base_pred)),
            "changed_test": int(np.sum(test_pred != base_test_pred)),
            "test_entropy": float((-np.clip(test_proba, 1e-15, 1.0) * np.log(np.clip(test_proba, 1e-15, 1.0))).sum(axis=1).mean()),
            "test_counts": np.bincount(test_pred, minlength=len(classes)).astype(int).tolist(),
        }
        for name, mask in masks.items():
            row[name] = f1_score(y[mask], pred[mask], average="macro")
        rows.append(row)
    frame = pd.DataFrame(rows)
    print(json.dumps({"domain_auc": domain_auc}, ensure_ascii=False))
    print(frame.sort_values(["top20", "top30", "full"], ascending=False).round(6).to_string(index=False))


if __name__ == "__main__":
    main()
