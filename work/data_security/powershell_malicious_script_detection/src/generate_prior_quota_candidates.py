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
    out = np.zeros((len(proba), len(CLASSES)), dtype=float)
    for src, cls in enumerate(classes):
        dst = np.where(CLASSES == int(cls))[0]
        if len(dst):
            out[:, dst[0]] = proba[:, src]
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


def model_probas(test_x: pd.DataFrame) -> dict[str, np.ndarray]:
    x = test_x.astype("int16")
    out: dict[str, np.ndarray] = {}
    for name, filename in {
        "lgbm": "lgbm.joblib",
        "xgb": "xgb.joblib",
        "extra_trees": "extra_trees.joblib",
        "catboost": "catboost.joblib",
    }.items():
        model = joblib.load(MODEL_DIR / filename)
        out[name] = align_proba(model, model.predict_proba(x))

    hist = joblib.load(MODEL_DIR / "hist_gbdt.joblib")
    x_hist = x.copy()
    for col in x_hist.columns:
        x_hist[col] = x_hist[col].astype("category")
    out["hist_gbdt"] = align_proba(hist, hist.predict_proba(x_hist))

    rf_path = MODEL_DIR / "rf_groupcv.joblib"
    if rf_path.exists():
        rf = joblib.load(rf_path)
        out["rf"] = align_proba(rf, rf.predict_proba(x))
    return out


def adjust_to_quota(scores: np.ndarray, target_counts: list[int]) -> np.ndarray:
    target = np.array(target_counts, dtype=int)
    if target.sum() != len(scores):
        raise ValueError(f"Target counts must sum to {len(scores)}: {target_counts}")

    pred = scores.argmax(axis=1).astype(int)
    counts = np.bincount(pred, minlength=len(CLASSES))
    if np.array_equal(counts, target):
        return pred

    eps = 1e-15
    log_scores = np.log(np.clip(scores, eps, None))

    # The intended quota candidates all need more class 0 and fewer class 1/2.
    # Greedy least-loss reassignment preserves the strongest baseline decisions.
    for dst in np.where(target > counts)[0]:
        need = int(target[dst] - counts[dst])
        candidate_rows: list[tuple[float, int, int]] = []
        for src in np.where(counts > target)[0]:
            rows = np.where(pred == src)[0]
            loss = log_scores[rows, src] - log_scores[rows, dst]
            for row, row_loss in zip(rows, loss):
                candidate_rows.append((float(row_loss), int(row), int(src)))
        candidate_rows.sort(key=lambda x: x[0])
        changed = 0
        for _, row, src in candidate_rows:
            if changed >= need:
                break
            if pred[row] != src:
                continue
            if counts[src] <= target[src]:
                continue
            pred[row] = dst
            counts[src] -= 1
            counts[dst] += 1
            changed += 1
        if changed != need:
            raise RuntimeError(f"Could not satisfy quota for class {dst}: changed={changed}, need={need}")

    if not np.array_equal(np.bincount(pred, minlength=len(CLASSES)), target):
        raise RuntimeError(f"Quota adjustment failed: got {np.bincount(pred, minlength=3)}, target {target}")
    return pred


def estimate_test_prior(train: pd.DataFrame, test: pd.DataFrame, features: list[str], alpha: float) -> np.ndarray:
    counts = train.groupby(features + ["label"]).size().unstack(fill_value=0)
    for cls in CLASSES:
        if cls not in counts.columns:
            counts[cls] = 0
    counts = counts[CLASSES]
    combo_index = pd.MultiIndex.from_frame(pd.concat([train[features], test[features]], ignore_index=True).drop_duplicates())
    counts = counts.reindex(combo_index, fill_value=0)
    test_counts = test.groupby(features).size().reindex(combo_index, fill_value=0).to_numpy(dtype=float)

    cond = counts.to_numpy(dtype=float) + alpha
    cond = cond / cond.sum(axis=0, keepdims=True)
    pi = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
    for _ in range(10000):
        mix = cond * pi
        resp = mix / mix.sum(axis=1, keepdims=True)
        new_pi = (test_counts[:, None] * resp).sum(axis=0) / test_counts.sum()
        if np.max(np.abs(new_pi - pi)) < 1e-12:
            break
        pi = new_pi
    return pi


def rounded_counts(pi: np.ndarray, total: int) -> list[int]:
    raw = pi * total
    base = np.floor(raw).astype(int)
    remain = int(total - base.sum())
    order = np.argsort(-(raw - base))
    for idx in order[:remain]:
        base[idx] += 1
    return base.tolist()


def main() -> None:
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    train, test, features = load_data()
    test_x = test[features].astype("int16")
    probs = model_probas(test_x)
    exact = exact_combo_proba(train, test, features, smoothing=1.0)

    tree_plus_hist = (probs["lgbm"] + probs["catboost"] + probs["xgb"] + probs["extra_trees"] + probs["hist_gbdt"]) / 5.0
    blend_v1_prob = 0.3 * exact + 0.7 * tree_plus_hist
    blend_v1_scores = blend_v1_prob * np.array([1.0, 1.15, 1.0])

    group_prob = None
    if "rf" in probs:
        group_prob = 0.4 * probs["rf"] + 0.1 * probs["lgbm"] + 0.5 * probs["xgb"]
        group_scores = group_prob * np.array([1.0, 0.95, 1.05])
    else:
        group_scores = None

    em001 = rounded_counts(estimate_test_prior(train, test, features, alpha=0.01), len(test))
    em01 = rounded_counts(estimate_test_prior(train, test, features, alpha=0.1), len(test))

    candidates: dict[str, tuple[np.ndarray, list[int], str]] = {
        "submission_prior_quota_em001_blend_v1.csv": (
            blend_v1_scores,
            em001,
            "blend_v1 scores adjusted by least-loss flips to EM(alpha=0.01) prior",
        ),
        "submission_prior_quota_em01_blend_v1.csv": (
            blend_v1_scores,
            em01,
            "blend_v1 scores adjusted by least-loss flips to EM(alpha=0.1) prior",
        ),
        "submission_prior_quota_13500_3000_3500_blend_v1.csv": (
            blend_v1_scores,
            [13500, 3000, 3500],
            "blend_v1 scores adjusted by least-loss flips to conservative prior 13500/3000/3500",
        ),
        "submission_prior_quota_missing_14000_2500_3500_blend_v1.csv": (
            blend_v1_scores,
            [14000, 2500, 3500],
            "blend_v1 scores adjusted by least-loss flips to name-gap prior 14000/2500/3500",
        ),
    }
    if group_scores is not None:
        candidates["submission_prior_quota_em001_group_v1.csv"] = (
            group_scores,
            em001,
            "groupcv_blend scores adjusted by least-loss flips to EM(alpha=0.01) prior",
        )
        candidates["submission_prior_quota_missing_14000_2500_3500_group_v1.csv"] = (
            group_scores,
            [14000, 2500, 3500],
            "groupcv_blend scores adjusted by least-loss flips to name-gap prior 14000/2500/3500",
        )

    base_blend_pred = blend_v1_scores.argmax(axis=1).astype(int)
    summary: dict[str, object] = {
        "em_alpha_0_01_counts": em001,
        "em_alpha_0_1_counts": em01,
        "base_blend_counts": {str(cls): int((base_blend_pred == cls).sum()) for cls in CLASSES},
        "candidates": {},
    }
    if group_scores is not None:
        group_pred = group_scores.argmax(axis=1).astype(int)
        summary["group_blend_counts"] = {str(cls): int((group_pred == cls).sum()) for cls in CLASSES}

    for filename, (scores, target, recipe) in candidates.items():
        pred = adjust_to_quota(scores, target)
        submission = pd.DataFrame({"name": test["name"], "label": pred})
        validate_submission(test, submission)
        out_path = SUBMISSION_DIR / filename
        submission.to_csv(out_path, index=False, encoding="utf-8")
        diff_vs_blend = int((pred != base_blend_pred).sum())
        summary["candidates"][filename] = {
            "path": str(out_path.relative_to(ROOT)),
            "recipe": recipe,
            "target_counts": {str(cls): int(target[cls]) for cls in CLASSES},
            "pred_counts": {str(cls): int((pred == cls).sum()) for cls in CLASSES},
            "diff_vs_submission_blend_v1": diff_vs_blend,
        }

    (REPORT_DIR / "prior_quota_candidate_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
