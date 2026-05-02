from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"
PROCESSED = ROOT / "processed"
MODELS = ROOT / "models"
SUBMISSIONS = ROOT / "submissions"
LOGS = ROOT / "logs"
REPORT = ROOT / "report"

TRAIN_CSV = RAW / "train.csv"
TEST_CSV = RAW / "test.csv"
BINARY_DIR = RAW / "binaries"

FEATURE_CACHE = PROCESSED / "features.joblib"
MODEL_PATH = MODELS / "linear_svc_hash.joblib"
VALID_PRED_PATH = REPORT / "valid_predictions.csv"

NO_VULN = "NO_VULN"
SEED = 20260501


def ensure_dirs() -> None:
    for path in [PROCESSED, MODELS, SUBMISSIONS, LOGS, REPORT]:
        path.mkdir(parents=True, exist_ok=True)


def load_train() -> pd.DataFrame:
    frame = pd.read_csv(TRAIN_CSV)
    frame["target"] = frame["cwe_id"].fillna(NO_VULN)
    return frame


def load_test() -> pd.DataFrame:
    return pd.read_csv(TEST_CSV)


def save_joblib(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path, compress=3)


def load_joblib(path: Path) -> object:
    return joblib.load(path)


def entropy_from_counts(counts: np.ndarray) -> float:
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())


def dataframe_to_sparse(frame: pd.DataFrame, columns: list[str]) -> sparse.csr_matrix:
    array = frame[columns].to_numpy(dtype=np.float32, copy=False)
    return sparse.csr_matrix(array)
