from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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


def validate_submission(test: pd.DataFrame, submission: pd.DataFrame) -> None:
    if list(submission.columns) != ["name", "label"]:
        raise ValueError("Submission must have columns: name,label")
    if len(submission) != len(test):
        raise ValueError("Submission row count does not match test row count")
    if not submission["name"].equals(test["name"]):
        raise ValueError("Submission name column does not match test order")
    if submission.isna().any().any():
        raise ValueError("Submission contains missing values")
    if not set(submission["label"].unique()).issubset(set(CLASSES)):
        raise ValueError("Submission contains labels outside {0,1,2}")


def build_mlp(seed: int):
    return make_pipeline(
        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(128, 64),
            alpha=1e-4,
            learning_rate_init=0.001,
            max_iter=160,
            early_stopping=True,
            n_iter_no_change=12,
            random_state=seed,
            batch_size=512,
            verbose=False,
        ),
    )


def align_proba(model, proba: np.ndarray) -> np.ndarray:
    classes = getattr(model, "classes_", CLASSES)
    out = np.zeros((len(proba), len(CLASSES)), dtype=float)
    for src, cls in enumerate(classes):
        out[:, int(cls)] = proba[:, src]
    return out


def exact_combo_proba(train: pd.DataFrame, test: pd.DataFrame, features: list[str], smoothing: float = 1.0) -> np.ndarray:
    y = train["label"].to_numpy(dtype=int)
    global_counts = np.bincount(y, minlength=len(CLASSES)).astype(float)
    global_prob = global_counts / global_counts.sum()
    counts = train.groupby(features)["label"].value_counts().unstack(fill_value=0)
    for cls in CLASSES:
        if cls not in counts.columns:
            counts[cls] = 0
    counts = counts[CLASSES]
    probs = counts.astype(float) + smoothing * global_prob
    probs = probs.div(probs.sum(axis=1), axis=0)
    lookup = {tuple(k if isinstance(k, tuple) else (k,)): row.to_numpy(dtype=float) for k, row in probs.iterrows()}
    out = np.tile(global_prob, (len(test), 1))
    for i, key in enumerate(test[features].itertuples(index=False, name=None)):
        prob = lookup.get(key)
        if prob is not None:
            out[i] = prob
    return out


def blend_v1_proba(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> np.ndarray:
    x = test[features].astype("int16")
    probas = []
    for filename in ["lgbm.joblib", "catboost.joblib", "xgb.joblib", "extra_trees.joblib"]:
        model = joblib.load(MODEL_DIR / filename)
        probas.append(align_proba(model, model.predict_proba(x)))
    hist = joblib.load(MODEL_DIR / "hist_gbdt.joblib")
    x_hist = x.copy()
    for col in x_hist.columns:
        x_hist[col] = x_hist[col].astype("category")
    probas.append(align_proba(hist, hist.predict_proba(x_hist)))
    return 0.3 * exact_combo_proba(train, test, features, smoothing=1.0) + 0.7 * np.mean(probas, axis=0)


def make_submission(test: pd.DataFrame, proba: np.ndarray, weights: list[float], filename: str) -> dict[str, object]:
    pred = (proba * np.array(weights, dtype=float)).argmax(axis=1).astype(int)
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
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    data_dir = find_data_dir()
    train = pd.read_csv(data_dir / "data_train.csv")
    test = pd.read_csv(data_dir / "data_test.csv")
    features = [c for c in train.columns if c not in ["name", "label"]]
    x = train[features].astype(int)
    y = train["label"].astype(int)
    test_x = test[features].astype(int)

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    oof = np.zeros((len(train), len(CLASSES)), dtype=float)
    fold_scores = []
    for fold, (train_idx, valid_idx) in enumerate(skf.split(x, y), start=1):
        clf = build_mlp(SEED + fold)
        clf.fit(x.iloc[train_idx], y.iloc[train_idx])
        proba = clf.predict_proba(x.iloc[valid_idx])
        oof[valid_idx] = align_proba(clf.named_steps["mlpclassifier"], proba)
        pred = oof[valid_idx].argmax(axis=1)
        fold_scores.append(float(f1_score(y.iloc[valid_idx], pred, average="macro")))

    full_model = build_mlp(SEED)
    full_model.fit(x, y)
    joblib.dump(full_model, MODEL_DIR / "mlp_onehot.joblib")
    mlp_proba = align_proba(full_model.named_steps["mlpclassifier"], full_model.predict_proba(test_x))
    np.save(MODEL_DIR / "mlp_onehot_proba.npy", mlp_proba)

    base_proba = blend_v1_proba(train, test, features)
    base_pred = (base_proba * np.array([1.0, 1.15, 1.0])).argmax(axis=1).astype(int)

    candidates = {
        "submission_mlp_onehot_v1.csv": (mlp_proba, [1.0, 1.0, 1.0], "MLP one-hot only"),
        "submission_mlp_onehot_w100105_v1.csv": (mlp_proba, [1.0, 1.0, 1.05], "MLP one-hot, class 2 small boost"),
        "submission_blend_mlp_a010_w105_v1.csv": (
            0.9 * base_proba + 0.1 * mlp_proba,
            [1.0, 1.05, 1.0],
            "0.9*blend_v1_prob + 0.1*mlp, class weights [1,1.05,1]",
        ),
        "submission_blend_mlp_a020_w100105_v1.csv": (
            0.8 * base_proba + 0.2 * mlp_proba,
            [1.0, 1.0, 1.05],
            "0.8*blend_v1_prob + 0.2*mlp, class weights [1,1,1.05]",
        ),
        "submission_blend_mlp_a030_w115_v1.csv": (
            0.7 * base_proba + 0.3 * mlp_proba,
            [1.0, 1.15, 1.0],
            "0.7*blend_v1_prob + 0.3*mlp, original class weights [1,1.15,1]",
        ),
    }

    summary: dict[str, object] = {
        "oof_macro_f1": float(f1_score(y, oof.argmax(axis=1), average="macro")),
        "fold_macro_f1": fold_scores,
        "base_blend_counts": {str(cls): int((base_pred == cls).sum()) for cls in CLASSES},
        "candidates": {},
    }
    for filename, (proba, weights, recipe) in candidates.items():
        item = make_submission(test, proba, weights, filename)
        pred = pd.read_csv(SUBMISSION_DIR / filename)["label"].to_numpy(dtype=int)
        item.update(
            {
                "recipe": recipe,
                "diff_vs_submission_blend_v1": int((pred != base_pred).sum()),
            }
        )
        summary["candidates"][filename] = item

    (REPORT_DIR / "mlp_candidate_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
