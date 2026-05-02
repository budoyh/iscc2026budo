from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
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
        raise ValueError("Submission contains invalid labels")


def build_model(hidden: tuple[int, ...], alpha: float, seed: int):
    return make_pipeline(
        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=hidden,
            alpha=alpha,
            learning_rate_init=0.001,
            max_iter=220,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=14,
            random_state=seed,
            batch_size=512,
            verbose=False,
        ),
    )


def adjust_to_quota(proba: np.ndarray, target_counts: list[int]) -> np.ndarray:
    target = np.array(target_counts, dtype=int)
    pred = proba.argmax(axis=1).astype(int)
    counts = np.bincount(pred, minlength=3)
    log_scores = np.log(np.clip(proba, 1e-15, None))
    for dst in np.where(target > counts)[0]:
        need = int(target[dst] - counts[dst])
        moves: list[tuple[float, int, int]] = []
        for src in np.where(counts > target)[0]:
            rows = np.where(pred == src)[0]
            losses = log_scores[rows, src] - log_scores[rows, dst]
            moves.extend((float(loss), int(row), int(src)) for row, loss in zip(rows, losses))
        moves.sort(key=lambda x: x[0])
        changed = 0
        for _, row, src in moves:
            if changed >= need:
                break
            if pred[row] != src or counts[src] <= target[src]:
                continue
            pred[row] = dst
            counts[src] -= 1
            counts[dst] += 1
            changed += 1
        if changed != need:
            raise RuntimeError(f"Could not satisfy quota for class {dst}")
    if not np.array_equal(np.bincount(pred, minlength=3), target):
        raise RuntimeError(f"Quota adjustment failed: {np.bincount(pred, minlength=3)} vs {target}")
    return pred


def write_submission(test: pd.DataFrame, pred: np.ndarray, filename: str) -> dict[str, object]:
    submission = pd.DataFrame({"name": test["name"], "label": pred.astype(int)})
    validate_submission(test, submission)
    out_path = SUBMISSION_DIR / filename
    submission.to_csv(out_path, index=False, encoding="utf-8", lineterminator="\n")
    return {
        "path": str(out_path.relative_to(ROOT)),
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

    model_specs = [
        ((128, 64), 1e-4, SEED + 101),
        ((128, 64), 3e-4, SEED + 102),
        ((160, 80), 1e-4, SEED + 103),
        ((192, 96), 1e-4, SEED + 104),
        ((192, 96), 3e-4, SEED + 105),
        ((256, 128, 64), 1e-4, SEED + 106),
        ((256, 128, 64), 3e-4, SEED + 107),
        ((320, 160, 80), 1e-4, SEED + 108),
    ]

    probas = []
    model_summaries = []
    for idx, (hidden, alpha, seed) in enumerate(model_specs, start=1):
        model = build_model(hidden, alpha, seed)
        model.fit(x, y)
        proba = model.predict_proba(test_x)
        probas.append(proba)
        model_path = MODEL_DIR / f"sklearn_mlp_ensemble_{idx}.joblib"
        joblib.dump(model, model_path)
        pred = proba.argmax(axis=1)
        model_summaries.append(
            {
                "model": str(model_path.relative_to(ROOT)),
                "hidden": hidden,
                "alpha": alpha,
                "seed": seed,
                "n_iter": int(model.named_steps["mlpclassifier"].n_iter_),
                "test_counts": {str(cls): int((pred == cls).sum()) for cls in CLASSES},
            }
        )
        print(f"trained {idx}/{len(model_specs)} hidden={hidden} alpha={alpha} seed={seed}", flush=True)

    ensemble = np.mean(probas, axis=0)
    np.save(MODEL_DIR / "sklearn_mlp_ensemble_proba.npy", ensemble)
    single = np.load(MODEL_DIR / "mlp_onehot_proba.npy")
    hybrid = 0.5 * ensemble + 0.5 * single
    np.save(MODEL_DIR / "sklearn_mlp_single_multi_hybrid_proba.npy", hybrid)

    candidates: dict[str, tuple[np.ndarray, str]] = {
        "submission_sklearn_mlp_ensemble_v1.csv": (
            ensemble.argmax(axis=1).astype(int),
            "mean of 8 sklearn MLP full-data models",
        ),
        "submission_sklearn_mlp_ensemble_quota13500_v1.csv": (
            adjust_to_quota(ensemble, [13500, 3000, 3500]),
            "mean of 8 sklearn MLPs, least-loss quota 13500/3000/3500",
        ),
        "submission_sklearn_mlp_hybrid_v1.csv": (
            hybrid.argmax(axis=1).astype(int),
            "0.5*single best sklearn MLP + 0.5*8-model MLP ensemble",
        ),
        "submission_sklearn_mlp_hybrid_quota13500_v1.csv": (
            adjust_to_quota(hybrid, [13500, 3000, 3500]),
            "0.5*single best sklearn MLP + 0.5*8-model MLP ensemble, quota 13500/3000/3500",
        ),
    }

    summary: dict[str, object] = {
        "online_reference": {"submission_mlp_onehot_v1.csv": 0.70201},
        "models": model_summaries,
        "candidates": {},
    }
    base = single.argmax(axis=1).astype(int)
    for filename, (pred, recipe) in candidates.items():
        item = write_submission(test, pred, filename)
        item.update({"recipe": recipe, "diff_vs_mlp_onehot_v1": int((pred != base).sum())})
        summary["candidates"][filename] = item

    (REPORT_DIR / "sklearn_mlp_ensemble_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
