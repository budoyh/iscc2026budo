from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
SUBMISSION_DIR = ROOT / "submissions"
REPORT_DIR = ROOT / "reports"
DATA_ROOT = ROOT / "data"
CLASSES = np.array([0, 1, 2], dtype=int)


def find_data_dir() -> Path:
    candidates = [p for p in DATA_ROOT.rglob("*") if (p / "data_test.csv").exists()]
    if not candidates:
        raise FileNotFoundError(f"data_test.csv not found under {DATA_ROOT}")
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


def adjust_to_quota(scores: np.ndarray, target_counts: list[int]) -> np.ndarray:
    target = np.array(target_counts, dtype=int)
    if target.sum() != len(scores):
        raise ValueError(f"Target counts must sum to {len(scores)}")
    pred = scores.argmax(axis=1).astype(int)
    counts = np.bincount(pred, minlength=len(CLASSES))
    log_scores = np.log(np.clip(scores, 1e-15, None))
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
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    data_dir = find_data_dir()
    test = pd.read_csv(data_dir / "data_test.csv")
    proba = np.load(MODEL_DIR / "mlp_onehot_proba.npy")
    base_pred = proba.argmax(axis=1).astype(int)

    candidates: dict[str, tuple[np.ndarray, str]] = {}
    weight_specs = {
        "submission_mlp_w102000100_v1.csv": [1.02, 1.0, 1.0],
        "submission_mlp_w108092100_v1.csv": [1.08, 0.92, 1.0],
        "submission_mlp_w110090103_v1.csv": [1.10, 0.90, 1.03],
        "submission_mlp_w115085105_v1.csv": [1.15, 0.85, 1.05],
    }
    for filename, weights in weight_specs.items():
        pred = (proba * np.array(weights, dtype=float)).argmax(axis=1).astype(int)
        candidates[filename] = (pred, f"sklearn MLP proba class weights {weights}")

    quota_specs = {
        "submission_mlp_quota_13500_3000_3500_v1.csv": [13500, 3000, 3500],
        "submission_mlp_quota_gap_14000_2500_3500_v1.csv": [14000, 2500, 3500],
        "submission_mlp_quota_em001_13411_3125_3464_v1.csv": [13411, 3125, 3464],
    }
    for filename, target in quota_specs.items():
        pred = adjust_to_quota(proba, target)
        candidates[filename] = (pred, f"sklearn MLP least-loss quota adjustment to {target}")

    summary: dict[str, object] = {
        "online_reference": {
            "submission_mlp_onehot_v1.csv": 0.70201,
            "submission_blend_mlp_a020_w100105_v1.csv": 0.69821,
            "submission.csv_prior_quota_13500_3000_3500": 0.69772,
        },
        "base_mlp_counts": {str(cls): int((base_pred == cls).sum()) for cls in CLASSES},
        "candidates": {},
    }
    for filename, (pred, recipe) in candidates.items():
        item = write_submission(test, pred, filename)
        item.update(
            {
                "recipe": recipe,
                "diff_vs_mlp_onehot_v1": int((pred != base_pred).sum()),
            }
        )
        summary["candidates"][filename] = item

    (REPORT_DIR / "mlp_refine_candidate_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
