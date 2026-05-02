from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.optimize import nnls
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw" / "data"
MODELS = ROOT / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search transductive prior corrections for a blend report.")
    parser.add_argument("--blend-report", required=True)
    parser.add_argument("--domain-estimators", type=int, default=500)
    return parser.parse_args()


def load_blend(report_path: Path, classes: list[str]) -> tuple[np.ndarray, np.ndarray, dict]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    weights = np.asarray(report["weights"], dtype=np.float64)
    weights /= weights.sum()
    oof = None
    test = None
    for run_id, weight in zip(report["run_ids"], weights):
        metadata = json.loads((MODELS / run_id / "metadata.json").read_text(encoding="utf-8"))
        if metadata["classes"] != classes:
            raise ValueError(f"class order mismatch for {run_id}")
        current_oof = np.load(MODELS / run_id / "oof_proba.npy")
        current_test = np.load(MODELS / run_id / "test_proba.npy")
        if oof is None:
            oof = np.zeros_like(current_oof, dtype=np.float64)
            test = np.zeros_like(current_test, dtype=np.float64)
        oof += weight * current_oof
        test += weight * current_test
    assert oof is not None and test is not None
    return oof, test, report


def adjust(proba: np.ndarray, target_prior: np.ndarray, alpha: float) -> np.ndarray:
    current = np.clip(proba.mean(axis=0), 1e-15, 1.0)
    out = proba * ((target_prior / current) ** alpha)
    out /= out.sum(axis=1, keepdims=True)
    return out


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


def bbse_prior(oof: np.ndarray, test: np.ndarray, y: np.ndarray, class_count: int) -> np.ndarray:
    oof_pred = oof.argmax(axis=1)
    test_pred = test.argmax(axis=1)
    confusion = confusion_matrix(y, oof_pred, labels=np.arange(class_count)).astype(np.float64)
    confusion /= np.clip(confusion.sum(axis=1, keepdims=True), 1e-12, None)
    predicted_test = np.bincount(test_pred, minlength=class_count).astype(np.float64)
    predicted_test /= predicted_test.sum()
    solution, _ = nnls(confusion.T, predicted_test)
    if solution.sum() <= 0:
        return np.full(class_count, 1.0 / class_count)
    return solution / solution.sum()


def em_prior(test: np.ndarray, source_prior: np.ndarray, steps: int = 200) -> np.ndarray:
    prior = source_prior.copy()
    prior /= prior.sum()
    source_prior = np.clip(source_prior, 1e-12, 1.0)
    for _ in range(steps):
        weights = test * (prior / source_prior)
        weights /= np.clip(weights.sum(axis=1, keepdims=True), 1e-12, None)
        new_prior = weights.mean(axis=0)
        if np.max(np.abs(new_prior - prior)) < 1e-9:
            prior = new_prior
            break
        prior = new_prior
    return prior / prior.sum()


def main() -> None:
    args = parse_args()
    train = pd.read_csv(RAW / "train_data.csv")
    test_frame = pd.read_csv(RAW / "test_data.csv")
    features = [col for col in train.columns if col not in {"id", "label"}]
    encoder = LabelEncoder()
    y = encoder.fit_transform(train["label"])
    classes = encoder.classes_.tolist()
    class_count = len(classes)

    raw_oof, raw_test, report = load_blend(Path(args.blend_report), classes)
    train_scores, domain_auc = build_domain_scores(train[features], test_frame[features], args.domain_estimators)
    train_prior = train["label"].value_counts(normalize=True).reindex(classes).to_numpy(dtype=np.float64)
    uniform = np.full(class_count, 1.0 / class_count)
    target_like20 = (
        pd.Series(np.asarray(classes)[y[train_scores >= np.quantile(train_scores, 0.8)]])
        .value_counts(normalize=True)
        .reindex(classes)
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )
    target_like30 = (
        pd.Series(np.asarray(classes)[y[train_scores >= np.quantile(train_scores, 0.7)]])
        .value_counts(normalize=True)
        .reindex(classes)
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )
    priors = {
        "uniform": uniform,
        "train": train_prior,
        "target_like20": target_like20,
        "target_like30": target_like30,
        "bbse": bbse_prior(raw_oof, raw_test, y, class_count),
        "em_train": em_prior(raw_test, train_prior),
        "em_uniform": em_prior(raw_test, uniform),
    }

    print("DOMAIN_AUC", round(domain_auc, 6))
    print("RAW_TEST_ARGMAX_COUNTS", dict(zip(classes, np.bincount(raw_test.argmax(axis=1), minlength=class_count).astype(int).tolist())))
    rows = []
    for prior_name, prior in priors.items():
        for alpha in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]:
            oof = adjust(raw_oof, prior, alpha)
            pred = oof.argmax(axis=1)
            row = {
                "prior": prior_name,
                "alpha": alpha,
                "full": f1_score(y, pred, average="macro"),
            }
            for q in [0.1, 0.2, 0.3, 0.5]:
                mask = train_scores >= np.quantile(train_scores, 1.0 - q)
                row[f"top{int(q * 100)}"] = f1_score(y[mask], pred[mask], average="macro")
            test_adj = adjust(raw_test, prior, alpha)
            row["test_entropy"] = float((-np.clip(test_adj, 1e-15, 1) * np.log(np.clip(test_adj, 1e-15, 1))).sum(axis=1).mean())
            row["test_counts"] = np.bincount(test_adj.argmax(axis=1), minlength=class_count).astype(int).tolist()
            rows.append(row)
    frame = pd.DataFrame(rows)
    print("PRIORS")
    for name, prior in priors.items():
        print(name, dict(zip(classes, prior.round(5).tolist())))
    print("TOP_BY_TOP20")
    print(frame.sort_values(["top20", "top30", "full"], ascending=False).head(20).round(6).to_string(index=False))
    print("TOP_BY_TOP30")
    print(frame.sort_values(["top30", "top20", "full"], ascending=False).head(20).round(6).to_string(index=False))


if __name__ == "__main__":
    main()
