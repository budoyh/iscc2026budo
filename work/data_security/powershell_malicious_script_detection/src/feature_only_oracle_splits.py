from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
REPORT_DIR = ROOT / "reports"
CLASSES = np.array([0, 1, 2], dtype=int)
HIDDEN_COUNTS = {0: 14000, 1: 2500, 2: 3500}


def find_data_dir() -> Path:
    candidates = [p for p in DATA_ROOT.rglob("*") if (p / "data_train.csv").exists()]
    if not candidates:
        raise FileNotFoundError(f"data_train.csv not found under {DATA_ROOT}")
    return candidates[0]


def extract_id(name: object) -> int:
    match = re.search(r"(\d+)", str(name))
    if not match:
        raise ValueError(f"Cannot extract numeric id from {name!r}")
    return int(match.group(1))


def class_positions(frame: pd.DataFrame) -> dict[int, np.ndarray]:
    labels = frame["label"].to_numpy(dtype=int)
    ids = frame["_id"].to_numpy(dtype=int)
    out: dict[int, np.ndarray] = {}
    for cls in CLASSES:
        idx = np.where(labels == cls)[0]
        out[int(cls)] = idx[np.argsort(ids[idx])]
    return out


def make_split(train: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    positions = class_positions(train)
    valid_parts = []
    fit_parts = []
    for cls in CLASSES:
        pos = positions[int(cls)]
        n = len(pos)
        if mode == "ratio_prefix":
            hold = int(round(n * HIDDEN_COUNTS[int(cls)] / (n + HIDDEN_COUNTS[int(cls)])))
            valid_idx = pos[:hold]
            fit_idx = pos[hold:]
        elif mode == "actual_prefix":
            hold = min(HIDDEN_COUNTS[int(cls)], n // 2)
            valid_idx = pos[:hold]
            fit_idx = pos[hold:]
        elif mode == "middle_ratio":
            hold = int(round(n * HIDDEN_COUNTS[int(cls)] / (n + HIDDEN_COUNTS[int(cls)])))
            start = (n - hold) // 2
            valid_idx = pos[start : start + hold]
            fit_idx = np.concatenate([pos[:start], pos[start + hold :]])
        elif mode == "suffix_ratio":
            hold = int(round(n * HIDDEN_COUNTS[int(cls)] / (n + HIDDEN_COUNTS[int(cls)])))
            valid_idx = pos[-hold:]
            fit_idx = pos[:-hold]
        else:
            raise ValueError(mode)
        valid_parts.append(train.iloc[valid_idx])
        fit_parts.append(train.iloc[fit_idx])
    return pd.concat(fit_parts, ignore_index=True), pd.concat(valid_parts, ignore_index=True)


def combo_keys(frame: pd.DataFrame, features: list[str]) -> list[tuple[int, ...]]:
    return list(map(tuple, frame[features].to_numpy(dtype=np.int16)))


def hard_feature_oracle(valid: pd.DataFrame, features: list[str]) -> np.ndarray:
    keys = combo_keys(valid, features)
    y = valid["label"].to_numpy(dtype=int)
    counts: dict[tuple[int, ...], np.ndarray] = defaultdict(lambda: np.zeros(len(CLASSES), dtype=int))
    for key, label in zip(keys, y):
        counts[key][int(label)] += 1
    return np.array([counts[key].argmax() for key in keys], dtype=int)


def train_combo_majority(fit: pd.DataFrame, valid: pd.DataFrame, features: list[str]) -> np.ndarray:
    fit_keys = combo_keys(fit, features)
    valid_keys = combo_keys(valid, features)
    y = fit["label"].to_numpy(dtype=int)
    global_counts = np.bincount(y, minlength=len(CLASSES))
    counts: dict[tuple[int, ...], np.ndarray] = defaultdict(lambda: np.zeros(len(CLASSES), dtype=int))
    for key, label in zip(fit_keys, y):
        counts[key][int(label)] += 1
    return np.array([(counts.get(key, global_counts)).argmax() for key in valid_keys], dtype=int)


def summarize(y_true: np.ndarray, pred: np.ndarray) -> dict[str, object]:
    return {
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
        "per_class_f1": {
            str(cls): float(v)
            for cls, v in zip(CLASSES, f1_score(y_true, pred, labels=CLASSES, average=None))
        },
        "pred_counts": {str(cls): int((pred == cls).sum()) for cls in CLASSES},
    }


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    data_dir = find_data_dir()
    train = pd.read_csv(data_dir / "data_train.csv")
    features = [c for c in train.columns if c not in ["name", "label"]]
    train["_id"] = train["name"].map(extract_id)
    out: dict[str, object] = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "features": features,
        "splits": {},
        "conclusion": "Hard feature oracle assigns one label per exact 15-feature combo using validation truth. Scores far below 0.9 indicate feature-only deterministic classification cannot reach the requested level on prefix-like proxies.",
    }
    for mode in ["ratio_prefix", "actual_prefix", "middle_ratio", "suffix_ratio"]:
        fit, valid = make_split(train, mode)
        y_true = valid["label"].to_numpy(dtype=int)
        oracle = hard_feature_oracle(valid, features)
        majority = train_combo_majority(fit, valid, features)
        out["splits"][mode] = {
            "valid_counts": {str(cls): int((y_true == cls).sum()) for cls in CLASSES},
            "unique_combos": int(pd.Series(combo_keys(valid, features)).nunique()),
            "hard_feature_oracle": summarize(y_true, oracle),
            "train_combo_majority": summarize(y_true, majority),
        }
    path = REPORT_DIR / "feature_only_oracle_splits.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
