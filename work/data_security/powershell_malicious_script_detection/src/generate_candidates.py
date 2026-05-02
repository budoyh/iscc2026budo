from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
MODEL_DIR = ROOT / "models"
SUBMISSION_DIR = ROOT / "submissions"
REPORT_DIR = ROOT / "reports"
CLASSES = np.array([0, 1, 2], dtype=int)


def find_data_dir() -> Path:
    candidates = [p for p in DATA_ROOT.rglob("*") if (p / "data_train.csv").exists()]
    if not candidates:
        raise FileNotFoundError(f"data_train.csv not found under {DATA_ROOT}")
    return candidates[0]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    data_dir = find_data_dir()
    train = pd.read_csv(data_dir / "data_train.csv")
    test = pd.read_csv(data_dir / "data_test.csv")
    features = [c for c in train.columns if c not in ["name", "label"]]
    return train, test, features


def exact_probabilities(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> np.ndarray:
    y = train["label"].to_numpy(dtype=int)
    prior = np.bincount(y, minlength=len(CLASSES)).astype(float)
    prior /= prior.sum()

    counts = train.groupby(features)["label"].value_counts().unstack(fill_value=0)
    for cls in CLASSES:
        if cls not in counts.columns:
            counts[cls] = 0
    counts = counts[CLASSES]
    probs = counts + prior
    probs = probs.div(probs.sum(axis=1), axis=0)
    lookup = {k: row.to_numpy(dtype=float) for k, row in probs.iterrows()}
    return np.array([lookup.get(k, prior) for k in test[features].itertuples(index=False, name=None)])


def load_model_probabilities(test: pd.DataFrame) -> dict[str, np.ndarray]:
    x = test.astype("int16")
    out: dict[str, np.ndarray] = {}
    aliases = {
        "lgbm": "lgbm.joblib",
        "xgb": "xgb.joblib",
        "et": "extra_trees.joblib",
        "hist": "hist_gbdt.joblib",
        "catboost": "catboost.joblib",
    }
    for name, filename in aliases.items():
        model = joblib.load(MODEL_DIR / filename)
        if name == "hist":
            x_use = x.copy()
            for col in x_use.columns:
                x_use[col] = x_use[col].astype("category")
            out[name] = model.predict_proba(x_use)
        else:
            out[name] = model.predict_proba(x)
    return out


def validate_submission(test: pd.DataFrame, submission: pd.DataFrame) -> None:
    if list(submission.columns) != ["name", "label"]:
        raise ValueError("Submission must have columns: name,label")
    if len(submission) != len(test):
        raise ValueError("Submission row count does not match test row count")
    if not submission["name"].equals(test["name"]):
        raise ValueError("Submission name column does not match test order")
    if submission.isna().any().any():
        raise ValueError("Submission contains missing values")
    labels = set(submission["label"].unique())
    if not labels.issubset(set(CLASSES)):
        raise ValueError(f"Submission contains invalid labels: {labels}")


def main() -> None:
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    train, test, features = load_data()
    exact = exact_probabilities(train, test, features)
    probs = load_model_probabilities(test[features])
    core = np.mean([probs["lgbm"], probs["xgb"], probs["hist"], probs["et"]], axis=0)

    candidates = {
        "submission_blend_shift_v1.csv": {
            "base": 0.1 * exact + 0.9 * core,
            "weights": [1.0, 0.7, 1.15],
            "recipe": "0.1*exact + 0.9*mean(lgbm,xgb,hist,extra_trees), class weights [1.0,0.7,1.15]",
        },
        "submission_blend_shift_v2.csv": {
            "base": 0.1 * exact + 0.9 * core,
            "weights": [1.0, 0.7, 1.20],
            "recipe": "0.1*exact + 0.9*mean(lgbm,xgb,hist,extra_trees), class weights [1.0,0.7,1.20]",
        },
        "submission_blend_shift_v3.csv": {
            "base": 0.1 * exact + 0.9 * core,
            "weights": [1.0, 0.7, 1.30],
            "recipe": "0.1*exact + 0.9*mean(lgbm,xgb,hist,extra_trees), class weights [1.0,0.7,1.30]",
        },
        "submission_core_shift_v1.csv": {
            "base": core,
            "weights": [1.0, 0.7, 1.15],
            "recipe": "mean(lgbm,xgb,hist,extra_trees), class weights [1.0,0.7,1.15]",
        },
    }

    summary: dict[str, dict[str, object]] = {}
    for filename, cfg in candidates.items():
        pred = (cfg["base"] * np.array(cfg["weights"], dtype=float)).argmax(axis=1).astype(int)
        submission = pd.DataFrame({"name": test["name"], "label": pred})
        validate_submission(test, submission)
        submission.to_csv(SUBMISSION_DIR / filename, index=False, encoding="utf-8")
        summary[filename] = {
            "recipe": cfg["recipe"],
            "pred_counts": {str(cls): int((pred == cls).sum()) for cls in CLASSES},
        }

    (REPORT_DIR / "candidate_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
