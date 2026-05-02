from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
REPORT_DIR = ROOT / "reports"
CLASSES = np.array([0, 1, 2], dtype=int)


def find_data_dir() -> Path:
    candidates = [p for p in DATA_ROOT.rglob("*") if (p / "data_train.csv").exists()]
    if not candidates:
        raise FileNotFoundError(f"data_train.csv not found under {DATA_ROOT}")
    return candidates[0]


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    data_dir = find_data_dir()
    train = pd.read_csv(data_dir / "data_train.csv")
    test = pd.read_csv(data_dir / "data_test.csv")
    features = [col for col in train.columns if col not in ["name", "label"]]

    counts = train.groupby(features)["label"].value_counts().unstack(fill_value=0)
    for cls in CLASSES:
        if cls not in counts.columns:
            counts[cls] = 0
    counts = counts[CLASSES]
    combo_total = counts.sum(axis=1)
    label_nunique = (counts > 0).sum(axis=1)
    majority_label = counts.idxmax(axis=1)
    majority_frac = counts.max(axis=1) / combo_total

    lookup = majority_label.to_dict()
    train_pred = np.array(
        [lookup[tuple(row)] for row in train[features].itertuples(index=False, name=None)],
        dtype=int,
    )
    train_y = train["label"].to_numpy(dtype=int)

    test_keys = list(test[features].itertuples(index=False, name=None))
    train_key_set = set(counts.index)
    seen_mask = np.array([key in train_key_set for key in test_keys], dtype=bool)
    ambiguous_key_set = set(counts.index[label_nunique > 1])
    ambiguous_seen_mask = np.array([key in ambiguous_key_set for key in test_keys], dtype=bool)

    seen_majority_fracs = [float(majority_frac.loc[key]) for key in test_keys if key in train_key_set]
    seen_combo_train_counts = [int(combo_total.loc[key]) for key in test_keys if key in train_key_set]
    weighted_seen_posterior = np.zeros(len(CLASSES), dtype=float)
    for key in test_keys:
        if key not in train_key_set:
            continue
        row = counts.loc[key].to_numpy(dtype=float)
        weighted_seen_posterior += row / row.sum()

    summary = {
        "features": features,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_label_counts": {str(cls): int((train_y == cls).sum()) for cls in CLASSES},
        "train_unique_combos": int(len(counts)),
        "train_combo_label_nunique": {str(int(k)): int(v) for k, v in label_nunique.value_counts().sort_index().items()},
        "train_rows_in_ambiguous_combos": int(
            sum(tuple(row) in ambiguous_key_set for row in train[features].itertuples(index=False, name=None))
        ),
        "train_majority_combo_accuracy": float((train_pred == train_y).mean()),
        "train_majority_combo_macro_f1": float(f1_score(train_y, train_pred, average="macro")),
        "test_seen_combo_rows": int(seen_mask.sum()),
        "test_seen_combo_ratio": float(seen_mask.mean()),
        "test_ambiguous_seen_combo_rows": int(ambiguous_seen_mask.sum()),
        "test_ambiguous_seen_combo_ratio": float(ambiguous_seen_mask.mean()),
        "test_unseen_combo_rows": int((~seen_mask).sum()),
        "test_seen_combo_majority_frac_quantiles": {
            str(q): float(np.quantile(seen_majority_fracs, q)) for q in [0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]
        },
        "test_seen_combo_train_count_quantiles": {
            str(q): float(np.quantile(seen_combo_train_counts, q)) for q in [0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]
        },
        "test_seen_rows_train_posterior_sum": {
            str(cls): float(weighted_seen_posterior[cls]) for cls in CLASSES
        },
        "interpretation": [
            "The 15 official features are highly non-injective: most train rows share an exact feature combo with other labels.",
            "Feature-only modeling has a low empirical ceiling unless the hidden test conditional distribution is much cleaner than train.",
            "A score near 0.9 would require reliable hidden structure such as source domain/order, not ordinary IID classification on these columns.",
        ],
    }
    path = REPORT_DIR / "feature_ambiguity_ceiling_summary.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
