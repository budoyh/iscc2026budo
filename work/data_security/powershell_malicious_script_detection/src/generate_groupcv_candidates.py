from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
MODEL_DIR = ROOT / "models"
SUBMISSION_DIR = ROOT / "submissions"
REPORT_DIR = ROOT / "reports"
SEED = 20260501
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


def load_existing_probs(test_x: pd.DataFrame) -> dict[str, np.ndarray]:
    x = test_x.astype("int16")
    out: dict[str, np.ndarray] = {}
    aliases = {
        "lgbm": "lgbm.joblib",
        "xgb": "xgb.joblib",
        "hist": "hist_gbdt.joblib",
        "et": "extra_trees.joblib",
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


def fit_rf(train_x: pd.DataFrame, train_y: np.ndarray):
    model = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        max_features=None,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(train_x, train_y)
    return model


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    train, test, features = load_data()
    train_x = train[features].astype("int16")
    test_x = test[features].astype("int16")
    train_y = train["label"].to_numpy(dtype=int)

    rf_model = fit_rf(train_x, train_y)
    joblib.dump(rf_model, MODEL_DIR / "rf_groupcv.joblib")

    probs = load_existing_probs(test_x)
    probs["rf"] = rf_model.predict_proba(test_x)

    new_prob = 0.4 * probs["rf"] + 0.1 * probs["lgbm"] + 0.5 * probs["xgb"]
    old_prob = 0.25 * (probs["lgbm"] + probs["xgb"] + probs["hist"] + probs["et"])

    candidates = {
        "submission_groupcv_blend_v1.csv": {
            "prob": new_prob,
            "weights": [1.0, 0.95, 1.05],
            "recipe": "0.4*rf + 0.1*lgbm + 0.5*xgb, class weights [1.0,0.95,1.05]",
        },
        "submission_groupcv_compromise_v1.csv": {
            "prob": 0.7 * new_prob + 0.3 * old_prob,
            "weights": [1.0, 0.95, 1.125],
            "recipe": "0.7*(0.4*rf + 0.1*lgbm + 0.5*xgb) + 0.3*mean(lgbm,xgb,hist,et), class weights [1.0,0.95,1.125]",
        },
        "submission_groupcv_equal_v1.csv": {
            "prob": (probs["rf"] + probs["lgbm"] + probs["xgb"]) / 3.0,
            "weights": [1.0, 0.85, 0.95],
            "recipe": "mean(rf,lgbm,xgb), class weights [1.0,0.85,0.95]",
        },
        "submission_groupcv_rf_v1.csv": {
            "prob": probs["rf"],
            "weights": [1.0, 1.0, 1.0],
            "recipe": "random forest only",
        },
    }

    summary: dict[str, dict[str, object]] = {}
    for filename, cfg in candidates.items():
        pred = (cfg["prob"] * np.array(cfg["weights"], dtype=float)).argmax(axis=1).astype(int)
        submission = pd.DataFrame({"name": test["name"], "label": pred})
        validate_submission(test, submission)
        submission.to_csv(SUBMISSION_DIR / filename, index=False, encoding="utf-8")
        summary[filename] = {
            "recipe": cfg["recipe"],
            "pred_counts": {str(cls): int((pred == cls).sum()) for cls in CLASSES},
        }

    (REPORT_DIR / "groupcv_candidate_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
