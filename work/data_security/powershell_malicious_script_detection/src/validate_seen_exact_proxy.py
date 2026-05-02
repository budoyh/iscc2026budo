from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
REPORT_DIR = ROOT / "reports"
SEED = 20260501
CLASSES = np.array([0, 1, 2], dtype=int)


def find_data_dir() -> Path:
    candidates = [p for p in DATA_ROOT.rglob("*") if (p / "data_train.csv").exists()]
    if not candidates:
        raise FileNotFoundError(f"data_train.csv not found under {DATA_ROOT}")
    return candidates[0]


def exact_proba(
    train_part: pd.DataFrame,
    valid_part: pd.DataFrame,
    features: list[str],
    smoothing: float,
) -> tuple[np.ndarray, np.ndarray]:
    y = train_part["label"].to_numpy(dtype=int)
    global_counts = np.bincount(y, minlength=len(CLASSES)).astype(float)
    global_prob = global_counts / global_counts.sum()

    counts = train_part.groupby(features)["label"].value_counts().unstack(fill_value=0)
    for cls in CLASSES:
        if cls not in counts.columns:
            counts[cls] = 0
    counts = counts[CLASSES]
    probs = counts.astype(float) + smoothing * global_prob
    probs = probs.div(probs.sum(axis=1), axis=0)
    lookup = {tuple(k if isinstance(k, tuple) else (k,)): row.to_numpy(dtype=float) for k, row in probs.iterrows()}

    out = np.tile(global_prob, (len(valid_part), 1))
    seen = np.zeros(len(valid_part), dtype=bool)
    for i, key in enumerate(valid_part[features].itertuples(index=False, name=None)):
        prob = lookup.get(key)
        if prob is None:
            continue
        out[i] = prob
        seen[i] = True
    return out, seen


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    data_dir = find_data_dir()
    train = pd.read_csv(data_dir / "data_train.csv")
    features = [c for c in train.columns if c not in ["name", "label"]]

    specs = {
        "mode_w111": (0.0, [1.0, 1.0, 1.0]),
        "s0_w120080090": (0.0, [1.2, 0.8, 0.9]),
        "s1_w120080090": (1.0, [1.2, 0.8, 0.9]),
        "s5_w115080090": (5.0, [1.15, 0.8, 0.9]),
        "s5_w120080095": (5.0, [1.2, 0.8, 0.95]),
        "s10_w120080105": (10.0, [1.2, 0.8, 1.05]),
    }

    splitter = StratifiedShuffleSplit(n_splits=10, test_size=20000, random_state=SEED)
    rows: list[dict[str, object]] = []
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(train[features], train["label"]), start=1):
        train_part = train.iloc[train_idx]
        valid_part = train.iloc[valid_idx]
        y_true = valid_part["label"].to_numpy(dtype=int)
        for spec_name, (smoothing, weights) in specs.items():
            proba, seen = exact_proba(train_part, valid_part, features, smoothing)
            pred = (proba * np.array(weights, dtype=float)).argmax(axis=1).astype(int)
            rows.append(
                {
                    "fold": fold,
                    "spec": spec_name,
                    "macro_f1": f1_score(y_true, pred, average="macro"),
                    "seen_ratio": float(seen.mean()),
                    "pred_0": int((pred == 0).sum()),
                    "pred_1": int((pred == 1).sum()),
                    "pred_2": int((pred == 2).sum()),
                }
            )

    detailed = pd.DataFrame(rows)
    summary = (
        detailed.groupby("spec")
        .agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            seen_ratio_mean=("seen_ratio", "mean"),
            pred_0_mean=("pred_0", "mean"),
            pred_1_mean=("pred_1", "mean"),
            pred_2_mean=("pred_2", "mean"),
        )
        .sort_values("macro_f1_mean", ascending=False)
    )
    detailed.to_csv(REPORT_DIR / "seen_exact_proxy_validation_detailed.csv", index=False, encoding="utf-8")
    summary.to_csv(REPORT_DIR / "seen_exact_proxy_validation.csv", encoding="utf-8")
    print(summary.to_string())


if __name__ == "__main__":
    main()
