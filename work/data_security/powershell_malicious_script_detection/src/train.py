from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


SEED = 20260501
CLASSES = np.array([0, 1, 2], dtype=int)
ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
REPORT_DIR = ROOT / "reports"
SUBMISSION_DIR = ROOT / "submissions"
MODEL_DIR = ROOT / "models"


def find_data_dir() -> Path:
    candidates = [p for p in DATA_ROOT.rglob("*") if (p / "data_train.csv").exists()]
    if not candidates:
        raise FileNotFoundError(f"data_train.csv not found under {DATA_ROOT}")
    return candidates[0]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    data_dir = find_data_dir()
    train = pd.read_csv(data_dir / "data_train.csv")
    test = pd.read_csv(data_dir / "data_test.csv")
    sample = pd.read_csv(data_dir / "submission.csv")
    features = [c for c in train.columns if c not in ["name", "label"]]
    return train, test, sample, features


def validate_data(train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame, features: list[str]) -> dict:
    expected_test_cols = ["name"] + features
    if list(test.columns) != expected_test_cols:
        raise ValueError(f"Unexpected test columns: {test.columns.tolist()}")
    if list(sample.columns) != ["name", "label"]:
        raise ValueError(f"Unexpected sample submission columns: {sample.columns.tolist()}")
    if train["name"].duplicated().any() or test["name"].duplicated().any():
        raise ValueError("Duplicate name values found.")
    if train[features + ["label"]].isna().any().any() or test[features].isna().any().any():
        raise ValueError("Missing values found.")
    if not set(train["label"].unique()).issubset(set(CLASSES)):
        raise ValueError(f"Unexpected labels: {sorted(train['label'].unique())}")

    eda = {
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "label_counts": {str(k): int(v) for k, v in train["label"].value_counts().sort_index().items()},
        "feature_nunique": {c: int(train[c].nunique()) for c in features},
        "test_seen_combo_ratio": seen_combo_ratio(train, test, features),
    }
    return eda


def seen_combo_ratio(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> float:
    train_keys = set(map(tuple, train[features].to_numpy()))
    test_keys = list(map(tuple, test[features].to_numpy()))
    return float(sum(k in train_keys for k in test_keys) / len(test_keys))


def exact_combo_predict_proba(
    train_part: pd.DataFrame,
    predict_part: pd.DataFrame,
    features: list[str],
    smoothing: float = 1.0,
) -> np.ndarray:
    y = train_part["label"].to_numpy(dtype=int)
    global_counts = np.bincount(y, minlength=len(CLASSES)).astype(float)
    global_prob = global_counts / global_counts.sum()

    counts = train_part.groupby(features)["label"].value_counts().unstack(fill_value=0)
    for cls in CLASSES:
        if cls not in counts.columns:
            counts[cls] = 0
    counts = counts[CLASSES]
    probs = counts + smoothing * global_prob
    probs = probs.div(probs.sum(axis=1), axis=0)
    lookup = {k: row.to_numpy(dtype=float) for k, row in probs.iterrows()}

    out = np.zeros((len(predict_part), len(CLASSES)), dtype=float)
    for i, key in enumerate(predict_part[features].itertuples(index=False, name=None)):
        out[i] = lookup.get(key, global_prob)
    return out


def model_factory(name: str):
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            max_features=None,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        )
    if name == "hist_gbdt":
        return HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            categorical_features="from_dtype",
            random_state=SEED,
        )
    if name == "lgbm":
        return LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=600,
            learning_rate=0.04,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
            verbosity=-1,
        )
    if name == "xgb":
        return XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=500,
            learning_rate=0.04,
            max_depth=5,
            min_child_weight=5,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=2.0,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=SEED,
            n_jobs=-1,
        )
    if name == "catboost":
        return CatBoostClassifier(
            loss_function="MultiClass",
            eval_metric="TotalF1:average=Macro",
            iterations=600,
            learning_rate=0.04,
            depth=6,
            l2_leaf_reg=4,
            auto_class_weights="Balanced",
            random_seed=SEED,
            verbose=False,
            allow_writing_files=False,
        )
    raise ValueError(f"Unknown model: {name}")


def prepare_x_for_model(x: pd.DataFrame, model_name: str) -> pd.DataFrame:
    out = x.copy()
    if model_name == "hist_gbdt":
        for col in out.columns:
            out[col] = out[col].astype("category")
    return out


def fit_predict_proba(model_name: str, x_train: pd.DataFrame, y_train: np.ndarray, x_pred: pd.DataFrame) -> np.ndarray:
    model = clone(model_factory(model_name))
    if model_name == "catboost":
        model.fit(x_train, y_train, cat_features=list(range(x_train.shape[1])))
        return model.predict_proba(x_pred)
    if model_name == "lgbm":
        model.fit(x_train, y_train, categorical_feature=list(x_train.columns))
        return model.predict_proba(x_pred)

    x_fit = prepare_x_for_model(x_train, model_name)
    x_out = prepare_x_for_model(x_pred, model_name)
    model.fit(x_fit, y_train)
    return model.predict_proba(x_out)


def fit_full_model(model_name: str, x_train: pd.DataFrame, y_train: np.ndarray):
    model = clone(model_factory(model_name))
    if model_name == "catboost":
        model.fit(x_train, y_train, cat_features=list(range(x_train.shape[1])))
    elif model_name == "lgbm":
        model.fit(x_train, y_train, categorical_feature=list(x_train.columns))
    else:
        model.fit(prepare_x_for_model(x_train, model_name), y_train)
    return model


def full_model_predict_proba(model_name: str, model, x_pred: pd.DataFrame) -> np.ndarray:
    if model_name in {"catboost", "lgbm"}:
        return model.predict_proba(x_pred)
    return model.predict_proba(prepare_x_for_model(x_pred, model_name))


def run_oof(
    train: pd.DataFrame,
    features: list[str],
    model_names: list[str],
    n_splits: int,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    x = train[features].astype("int16")
    y = train["label"].astype(int).to_numpy()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    oof_probs: dict[str, np.ndarray] = {"exact": np.zeros((len(train), len(CLASSES)), dtype=float)}
    rows = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(x, y), start=1):
        prob = exact_combo_predict_proba(train.iloc[tr_idx], train.iloc[va_idx], features)
        oof_probs["exact"][va_idx] = prob
    pred = oof_probs["exact"].argmax(axis=1)
    rows.append(score_row("exact", y, pred, seconds=0.0))

    for name in model_names:
        start = time.time()
        oof_probs[name] = np.zeros((len(train), len(CLASSES)), dtype=float)
        for fold, (tr_idx, va_idx) in enumerate(skf.split(x, y), start=1):
            prob = fit_predict_proba(name, x.iloc[tr_idx], y[tr_idx], x.iloc[va_idx])
            oof_probs[name][va_idx] = prob
        pred = oof_probs[name].argmax(axis=1)
        rows.append(score_row(name, y, pred, seconds=time.time() - start))

    return oof_probs, pd.DataFrame(rows)


def score_row(name: str, y_true: np.ndarray, y_pred: np.ndarray, seconds: float) -> dict:
    return {
        "name": name,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "pred_count_0": int((y_pred == 0).sum()),
        "pred_count_1": int((y_pred == 1).sum()),
        "pred_count_2": int((y_pred == 2).sum()),
        "seconds": round(float(seconds), 3),
    }


def optimize_ensemble(oof_probs: dict[str, np.ndarray], y: np.ndarray, model_names: list[str]) -> tuple[dict, pd.DataFrame]:
    groups = {
        "exact_only": [],
        "tree_core": [m for m in ["lgbm", "catboost", "xgb", "extra_trees"] if m in model_names],
        "tree_plus_hist": [m for m in ["lgbm", "catboost", "xgb", "extra_trees", "hist_gbdt"] if m in model_names],
    }
    candidate_rows = []
    best: dict | None = None
    alphas = np.round(np.linspace(0.0, 1.0, 11), 3)
    weight_values = np.round(np.linspace(0.75, 1.25, 11), 3)

    for group_name, members in groups.items():
        if group_name == "exact_only":
            model_prob = oof_probs["exact"]
            alpha_values = [1.0]
        else:
            if not members:
                continue
            model_prob = np.mean([oof_probs[m] for m in members], axis=0)
            alpha_values = alphas

        for alpha in alpha_values:
            base = alpha * oof_probs["exact"] + (1.0 - alpha) * model_prob
            for w1 in weight_values:
                for w2 in weight_values:
                    weights = np.array([1.0, w1, w2], dtype=float)
                    pred = (base * weights).argmax(axis=1)
                    macro = float(f1_score(y, pred, average="macro"))
                    row = {
                        "ensemble": group_name,
                        "members": ",".join(members),
                        "alpha_exact": float(alpha),
                        "weight_0": 1.0,
                        "weight_1": float(w1),
                        "weight_2": float(w2),
                        "macro_f1": macro,
                        "weighted_f1": float(f1_score(y, pred, average="weighted")),
                        "pred_count_0": int((pred == 0).sum()),
                        "pred_count_1": int((pred == 1).sum()),
                        "pred_count_2": int((pred == 2).sum()),
                    }
                    candidate_rows.append(row)
                    if best is None or macro > best["macro_f1"]:
                        best = row

    if best is None:
        raise RuntimeError("No ensemble candidate was evaluated.")
    return best, pd.DataFrame(candidate_rows).sort_values("macro_f1", ascending=False)


def predict_final(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    best: dict,
) -> tuple[np.ndarray, dict[str, object]]:
    x = train[features].astype("int16")
    y = train["label"].astype(int).to_numpy()
    xt = test[features].astype("int16")

    exact_prob = exact_combo_predict_proba(train, test, features)
    if best["ensemble"] == "exact_only":
        model_prob = exact_prob
        members: list[str] = []
    else:
        members = [m for m in str(best["members"]).split(",") if m]
        model_probs = []
        for name in members:
            model = fit_full_model(name, x, y)
            joblib.dump(model, MODEL_DIR / f"{name}.joblib")
            model_probs.append(full_model_predict_proba(name, model, xt))
        model_prob = np.mean(model_probs, axis=0)

    alpha = float(best["alpha_exact"])
    weights = np.array([float(best["weight_0"]), float(best["weight_1"]), float(best["weight_2"])])
    final_prob = alpha * exact_prob + (1.0 - alpha) * model_prob
    pred = (final_prob * weights).argmax(axis=1).astype(int)
    meta = {
        "members": members,
        "alpha_exact": alpha,
        "class_weights": weights.tolist(),
        "pred_counts": {str(i): int((pred == i).sum()) for i in CLASSES},
    }
    return pred, meta


def validate_submission(test: pd.DataFrame, submission: pd.DataFrame) -> None:
    if list(submission.columns) != ["name", "label"]:
        raise ValueError("Submission must have columns: name,label")
    if len(submission) != len(test):
        raise ValueError(f"Submission row count {len(submission)} != test row count {len(test)}")
    if not submission["name"].equals(test["name"]):
        raise ValueError("Submission name column does not match test order.")
    if submission.isna().any().any():
        raise ValueError("Submission contains missing values.")
    bad = set(submission["label"].unique()) - set(CLASSES)
    if bad:
        raise ValueError(f"Submission contains invalid labels: {bad}")


def leakage_report(train: pd.DataFrame) -> dict:
    extracted = train["name"].astype(str).str.extract(r"data_(\d+)\.ps1")[0]
    if extracted.isna().any():
        return {"name_numeric_pattern": False}
    nums = extracted.astype(int)
    runs = []
    sorted_train = train.assign(_num=nums).sort_values("_num")
    start = prev = None
    last_label = None
    for num_raw, label_raw in sorted_train[["_num", "label"]].itertuples(index=False, name=None):
        num = int(num_raw)
        label = int(label_raw)
        if last_label is None or label != last_label or num != prev + 1:
            if last_label is not None:
                runs.append({"start": start, "end": prev, "label": last_label, "length": prev - start + 1})
            start = num
            last_label = label
        prev = num
    runs.append({"start": start, "end": prev, "label": last_label, "length": prev - start + 1})

    full_range = set(range(int(nums.min()), int(nums.max()) + 1))
    missing = sorted(full_range - set(nums))
    intervals = []
    if missing:
        s = p = missing[0]
        for x in missing[1:]:
            if x == p + 1:
                p = x
            else:
                intervals.append({"start": s, "end": p, "length": p - s + 1})
                s = p = x
        intervals.append({"start": s, "end": p, "length": p - s + 1})

    global_missing = sorted(set(range(int(nums.max()) + 1)) - set(nums))
    return {
        "name_numeric_pattern": True,
        "min_num": int(nums.min()),
        "max_num": int(nums.max()),
        "train_label_runs_by_num": runs,
        "missing_intervals_inside_train_num_range": intervals,
        "global_missing_count_0_to_max": len(global_missing),
        "note": "Training names are perfectly ordered by label. Treat as leakage risk, not as the default modeling route.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["extra_trees", "hist_gbdt", "lgbm", "xgb", "catboost"],
        choices=["extra_trees", "hist_gbdt", "lgbm", "xgb", "catboost"],
    )
    parser.add_argument("--submission-name", default="submission_blend_v1.csv")
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train, test, sample, features = load_data()
    eda = validate_data(train, test, sample, features)
    leak = leakage_report(train)
    (REPORT_DIR / "eda.json").write_text(json.dumps(eda, indent=2, ensure_ascii=False), encoding="utf-8")
    (REPORT_DIR / "leakage_report.json").write_text(json.dumps(leak, indent=2, ensure_ascii=False), encoding="utf-8")

    oof_probs, cv_results = run_oof(train, features, args.models, args.n_splits)
    y = train["label"].astype(int).to_numpy()
    best, candidates = optimize_ensemble(oof_probs, y, args.models)
    best_pred = (
        (
            float(best["alpha_exact"]) * oof_probs["exact"]
            + (1.0 - float(best["alpha_exact"]))
            * (
                oof_probs["exact"]
                if best["ensemble"] == "exact_only"
                else np.mean([oof_probs[m] for m in str(best["members"]).split(",") if m], axis=0)
            )
        )
        * np.array([float(best["weight_0"]), float(best["weight_1"]), float(best["weight_2"])])
    ).argmax(axis=1)

    cv_results.to_csv(REPORT_DIR / "cv_results.csv", index=False, encoding="utf-8")
    candidates.head(200).to_csv(REPORT_DIR / "ensemble_candidates_top200.csv", index=False, encoding="utf-8")
    (REPORT_DIR / "best_oof_report.txt").write_text(
        classification_report(y, best_pred, digits=4),
        encoding="utf-8",
    )

    pred, meta = predict_final(train, test, features, best)
    submission = pd.DataFrame({"name": test["name"], "label": pred})
    validate_submission(test, submission)
    submission_path = SUBMISSION_DIR / args.submission_name
    submission.to_csv(submission_path, index=False, encoding="utf-8")

    run_summary = {
        "seed": SEED,
        "models": args.models,
        "best_oof": best,
        "final_prediction": meta,
        "submission_path": str(submission_path),
        "eda": eda,
    }
    (REPORT_DIR / "run_summary.json").write_text(json.dumps(run_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(run_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
