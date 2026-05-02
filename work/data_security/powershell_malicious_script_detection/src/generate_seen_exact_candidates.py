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


def align_proba(model, proba: np.ndarray) -> np.ndarray:
    classes = getattr(model, "classes_", CLASSES)
    aligned = np.zeros((len(proba), len(CLASSES)), dtype=float)
    for src_idx, cls in enumerate(classes):
        dst = np.where(CLASSES == int(cls))[0]
        if len(dst):
            aligned[:, dst[0]] = proba[:, src_idx]
    return aligned


def predict_model_proba(name: str, model, x: pd.DataFrame) -> np.ndarray:
    if name == "hist":
        x_use = x.copy()
        for col in x_use.columns:
            x_use[col] = x_use[col].astype("category")
    else:
        x_use = x
    return align_proba(model, model.predict_proba(x_use))


def load_existing_probs(test_x: pd.DataFrame) -> dict[str, np.ndarray]:
    x = test_x.astype("int16")
    aliases = {
        "lgbm": "lgbm.joblib",
        "xgb": "xgb.joblib",
        "hist": "hist_gbdt.joblib",
        "et": "extra_trees.joblib",
    }
    out: dict[str, np.ndarray] = {}
    for name, filename in aliases.items():
        model = joblib.load(MODEL_DIR / filename)
        out[name] = predict_model_proba(name, model, x)
    rf_path = MODEL_DIR / "rf_groupcv.joblib"
    if rf_path.exists():
        rf_model = joblib.load(rf_path)
        out["rf"] = predict_model_proba("rf", rf_model, x)
    return out


def fit_rf_if_needed(train_x: pd.DataFrame, train_y: np.ndarray, test_x: pd.DataFrame, probs: dict[str, np.ndarray]) -> None:
    if "rf" in probs:
        return
    model = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        max_features=None,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(train_x, train_y)
    joblib.dump(model, MODEL_DIR / "rf_groupcv.joblib")
    probs["rf"] = align_proba(model, model.predict_proba(test_x))


def exact_seen_proba(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    fallback_proba: np.ndarray,
    smoothing: float,
) -> tuple[np.ndarray, np.ndarray]:
    train_y = train["label"].to_numpy(dtype=int)
    global_counts = np.bincount(train_y, minlength=len(CLASSES)).astype(float)
    global_prob = global_counts / global_counts.sum()

    counts = train.groupby(features)["label"].value_counts().unstack(fill_value=0)
    for cls in CLASSES:
        if cls not in counts.columns:
            counts[cls] = 0
    counts = counts[CLASSES]
    probs = counts.astype(float) + smoothing * global_prob
    probs = probs.div(probs.sum(axis=1), axis=0)
    lookup = {tuple(k if isinstance(k, tuple) else (k,)): row.to_numpy(dtype=float) for k, row in probs.iterrows()}

    out = np.array(fallback_proba, copy=True)
    seen_mask = np.zeros(len(test), dtype=bool)
    for i, key in enumerate(test[features].itertuples(index=False, name=None)):
        prob = lookup.get(key)
        if prob is None:
            continue
        out[i] = prob
        seen_mask[i] = True
    return out, seen_mask


def make_submission(test: pd.DataFrame, proba: np.ndarray, weights: list[float], filename: str) -> dict[str, object]:
    weighted = proba * np.array(weights, dtype=float)
    pred = weighted.argmax(axis=1).astype(int)
    submission = pd.DataFrame({"name": test["name"], "label": pred})
    validate_submission(test, submission)
    out_path = SUBMISSION_DIR / filename
    submission.to_csv(out_path, index=False, encoding="utf-8")
    return {
        "path": str(out_path.relative_to(ROOT)),
        "weights": weights,
        "pred_counts": {str(cls): int((pred == cls).sum()) for cls in CLASSES},
    }


def main() -> None:
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train, test, features = load_data()
    train_x = train[features].astype("int16")
    test_x = test[features].astype("int16")
    train_y = train["label"].to_numpy(dtype=int)

    probs = load_existing_probs(test_x)
    fit_rf_if_needed(train_x, train_y, test_x, probs)

    fallback_probs = {
        "group": 0.4 * probs["rf"] + 0.1 * probs["lgbm"] + 0.5 * probs["xgb"],
        "old_core": 0.25 * (probs["lgbm"] + probs["xgb"] + probs["hist"] + probs["et"]),
    }

    candidate_specs = {
        "submission_seen_exact_mode_group_v1.csv": {
            "fallback": "group",
            "smoothing": 0.0,
            "weights": [1.0, 1.0, 1.0],
            "recipe": "seen exact combo posterior without smoothing; unseen fallback = group rf/lgbm/xgb blend",
        },
        "submission_seen_exact_s0_w120080090_group_v1.csv": {
            "fallback": "group",
            "smoothing": 0.0,
            "weights": [1.2, 0.8, 0.9],
            "recipe": "seen exact combo posterior without smoothing; class weights [1.2,0.8,0.9]; unseen fallback = group blend",
        },
        "submission_seen_exact_s1_w120080090_group_v1.csv": {
            "fallback": "group",
            "smoothing": 1.0,
            "weights": [1.2, 0.8, 0.9],
            "recipe": "seen exact combo posterior with smoothing=1; class weights [1.2,0.8,0.9]; unseen fallback = group blend",
        },
        "submission_seen_exact_s5_w115080090_group_v1.csv": {
            "fallback": "group",
            "smoothing": 5.0,
            "weights": [1.15, 0.8, 0.9],
            "recipe": "seen exact combo posterior with smoothing=5; class weights [1.15,0.8,0.9]; unseen fallback = group blend",
        },
        "submission_seen_exact_s5_w120080095_group_v1.csv": {
            "fallback": "group",
            "smoothing": 5.0,
            "weights": [1.2, 0.8, 0.95],
            "recipe": "seen exact combo posterior with smoothing=5; class weights [1.2,0.8,0.95]; unseen fallback = group blend",
        },
        "submission_seen_exact_s10_w120080105_group_v1.csv": {
            "fallback": "group",
            "smoothing": 10.0,
            "weights": [1.2, 0.8, 1.05],
            "recipe": "seen exact combo posterior with smoothing=10; class weights [1.2,0.8,1.05]; unseen fallback = group blend",
        },
        "submission_seen_exact_s5_w120080095_oldcore_v1.csv": {
            "fallback": "old_core",
            "smoothing": 5.0,
            "weights": [1.2, 0.8, 0.95],
            "recipe": "seen exact combo posterior with smoothing=5; class weights [1.2,0.8,0.95]; unseen fallback = mean(lgbm,xgb,hist,extra_trees)",
        },
        "submission_seen_exact_s10_w120080105_oldcore_v1.csv": {
            "fallback": "old_core",
            "smoothing": 10.0,
            "weights": [1.2, 0.8, 1.05],
            "recipe": "seen exact combo posterior with smoothing=10; class weights [1.2,0.8,1.05]; unseen fallback = mean(lgbm,xgb,hist,extra_trees)",
        },
    }

    summary: dict[str, object] = {}
    seen_counts: dict[float, int] = {}
    for filename, spec in candidate_specs.items():
        smoothing = float(spec["smoothing"])
        proba, seen_mask = exact_seen_proba(
            train=train,
            test=test,
            features=features,
            fallback_proba=fallback_probs[str(spec["fallback"])],
            smoothing=smoothing,
        )
        seen_counts[smoothing] = int(seen_mask.sum())
        item = make_submission(test, proba, list(spec["weights"]), filename)
        item.update(
            {
                "recipe": spec["recipe"],
                "fallback": spec["fallback"],
                "smoothing": smoothing,
                "seen_rows": int(seen_mask.sum()),
                "unseen_rows": int((~seen_mask).sum()),
            }
        )
        summary[filename] = item

    report = {
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "features": features,
        "seen_rows": int(next(iter(seen_counts.values()))),
        "unseen_rows": int(len(test) - next(iter(seen_counts.values()))),
        "seen_ratio": float(next(iter(seen_counts.values())) / len(test)),
        "candidates": summary,
    }
    (REPORT_DIR / "seen_exact_candidate_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
