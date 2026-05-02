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
SEED = 20260503


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
    y = frame["label"].to_numpy(dtype=int)
    ids = frame["_id"].to_numpy(dtype=int)
    out: dict[int, np.ndarray] = {}
    for cls in CLASSES:
        idx = np.where(y == cls)[0]
        out[int(cls)] = idx[np.argsort(ids[idx])]
    return out


def make_split(train: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    positions = class_positions(train)
    fit_parts = []
    valid_parts = []
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
        else:
            raise ValueError(mode)
        fit_parts.append(train.iloc[fit_idx])
        valid_parts.append(train.iloc[valid_idx])
    return pd.concat(fit_parts, ignore_index=True), pd.concat(valid_parts, ignore_index=True)


def combo_keys(frame: pd.DataFrame, features: list[str]) -> list[tuple[int, ...]]:
    return list(map(tuple, frame[features].to_numpy(dtype=np.int16)))


def combo_counts(frame: pd.DataFrame, features: list[str]) -> dict[tuple[int, ...], np.ndarray]:
    counts: dict[tuple[int, ...], np.ndarray] = defaultdict(lambda: np.zeros(len(CLASSES), dtype=int))
    for key, label in zip(combo_keys(frame, features), frame["label"].to_numpy(dtype=int)):
        counts[key][int(label)] += 1
    return counts


def f1_from_confusion(conf: np.ndarray) -> tuple[float, list[float]]:
    scores = []
    for cls in CLASSES:
        tp = conf[int(cls), int(cls)]
        fp = conf[:, int(cls)].sum() - tp
        fn = conf[int(cls), :].sum() - tp
        denom = 2 * tp + fp + fn
        scores.append(float(0.0 if denom <= 0 else 2 * tp / denom))
    return float(np.mean(scores)), scores


def expected_confusion(
    true_counts: dict[tuple[int, ...], np.ndarray],
    pred_counts: dict[tuple[int, ...], np.ndarray],
) -> np.ndarray:
    conf = np.zeros((len(CLASSES), len(CLASSES)), dtype=float)
    for key, true_c in true_counts.items():
        pred_c = pred_counts.get(key)
        if pred_c is None:
            pred_c = np.zeros(len(CLASSES), dtype=int)
            pred_c[int(np.argmax(true_c))] = int(true_c.sum())
        n = float(true_c.sum())
        if n <= 0:
            continue
        conf += np.outer(true_c.astype(float), pred_c.astype(float)) / n
    return conf


def actual_pred_from_counts(valid: pd.DataFrame, features: list[str], pred_counts: dict[tuple[int, ...], np.ndarray], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros(len(valid), dtype=int)
    groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for i, key in enumerate(combo_keys(valid, features)):
        groups[key].append(i)
    for key, rows in groups.items():
        counts = pred_counts[key].astype(int)
        labels = np.repeat(CLASSES, counts)
        if len(labels) != len(rows):
            raise ValueError((key, len(labels), len(rows), counts.tolist()))
        rng.shuffle(labels)
        out[np.asarray(rows, dtype=int)] = labels
    return out


def allocation_from_prob(
    target_group_sizes: dict[tuple[int, ...], int],
    probs: dict[tuple[int, ...], np.ndarray],
    quota: np.ndarray | None,
) -> dict[tuple[int, ...], np.ndarray]:
    pred_counts: dict[tuple[int, ...], np.ndarray] = {}
    residuals: list[tuple[float, tuple[int, ...], int]] = []
    totals = np.zeros(len(CLASSES), dtype=int)
    for key, n in target_group_sizes.items():
        p = probs[key]
        raw = p * n
        base = np.floor(raw).astype(int)
        remain = int(n - base.sum())
        for cls in np.argsort(-(raw - base))[:remain]:
            base[int(cls)] += 1
        pred_counts[key] = base
        totals += base
        for cls in CLASSES:
            residuals.append((float(raw[int(cls)] - np.floor(raw[int(cls)])), key, int(cls)))
    if quota is None:
        return pred_counts

    # Greedy least-loss movement to exact global quota.
    def score_loss(key: tuple[int, ...], src: int, dst: int) -> float:
        p = np.clip(probs[key], 1e-12, None)
        return float(np.log(p[src]) - np.log(p[dst]))

    while not np.array_equal(totals, quota):
        deficits = np.where(totals < quota)[0]
        surplus = np.where(totals > quota)[0]
        if len(deficits) == 0 or len(surplus) == 0:
            break
        best: tuple[float, tuple[int, ...], int, int] | None = None
        for dst in deficits:
            for src in surplus:
                for key, counts in pred_counts.items():
                    if counts[int(src)] <= 0:
                        continue
                    loss = score_loss(key, int(src), int(dst))
                    if best is None or loss < best[0]:
                        best = (loss, key, int(src), int(dst))
        if best is None:
            break
        _, key, src, dst = best
        pred_counts[key][src] -= 1
        pred_counts[key][dst] += 1
        totals[src] -= 1
        totals[dst] += 1
    return pred_counts


def estimate_probs_from_fit(
    fit: pd.DataFrame,
    valid: pd.DataFrame,
    features: list[str],
    mode: str,
    alpha: float,
) -> dict[tuple[int, ...], np.ndarray]:
    fit_counts = combo_counts(fit, features)
    y = fit["label"].to_numpy(dtype=int)
    global_prob = np.bincount(y, minlength=len(CLASSES)).astype(float)
    global_prob /= global_prob.sum()
    valid_sizes = {key: int(count.sum()) for key, count in combo_counts(valid, features).items()}
    if mode == "fit_all":
        source_counts = fit_counts
    elif mode == "fit_front_half":
        parts = []
        for cls in CLASSES:
            sub = fit[fit["label"] == int(cls)].sort_values("_id")
            parts.append(sub.iloc[: max(1, len(sub) // 2)])
        source_counts = combo_counts(pd.concat(parts, ignore_index=True), features)
    elif mode == "valid_oracle":
        source_counts = combo_counts(valid, features)
    else:
        raise ValueError(mode)
    probs = {}
    for key in valid_sizes:
        c = source_counts.get(key, np.zeros(len(CLASSES), dtype=int)).astype(float)
        p = c + alpha * global_prob
        probs[key] = p / p.sum()
    return probs


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    data_dir = find_data_dir()
    train = pd.read_csv(data_dir / "data_train.csv")
    features = [c for c in train.columns if c not in ["name", "label"]]
    train["_id"] = train["name"].map(extract_id)
    records: list[dict[str, object]] = []
    for split_mode in ["ratio_prefix", "actual_prefix", "middle_ratio"]:
        fit, valid = make_split(train, split_mode)
        true_counts = combo_counts(valid, features)
        target_sizes = {key: int(c.sum()) for key, c in true_counts.items()}
        y_true = valid["label"].to_numpy(dtype=int)
        quota = np.bincount(y_true, minlength=len(CLASSES))
        for prob_mode in ["valid_oracle", "fit_all", "fit_front_half"]:
            for alpha in [0.01, 0.2, 1.0, 5.0]:
                probs = estimate_probs_from_fit(fit, valid, features, prob_mode, alpha)
                # The diagnostic question is whether mixed labels inside exact
                # duplicate combos help at all when row identity is unavailable.
                # Global quota optimization is intentionally skipped here; a
                # one-row greedy quota loop was too slow and did not affect this
                # first-order upper-bound question.
                for quota_mode in ["none"]:
                    pred_counts = allocation_from_prob(target_sizes, probs, None)
                    conf = expected_confusion(true_counts, pred_counts)
                    macro, per_class = f1_from_confusion(conf)
                    actual_scores = []
                    if prob_mode == "valid_oracle":
                        # Estimate realized score variance when row identity inside a combo is unavailable.
                        for rep in range(10):
                            pred = actual_pred_from_counts(valid, features, pred_counts, seed=SEED + rep)
                            actual_scores.append(float(f1_score(y_true, pred, average="macro")))
                    records.append(
                        {
                            "split": split_mode,
                            "prob_mode": prob_mode,
                            "alpha": alpha,
                            "quota_mode": quota_mode,
                            "expected_macro_f1": macro,
                            "expected_per_class_f1": {str(cls): per_class[int(cls)] for cls in CLASSES},
                            "actual_macro_f1_mean": float(np.mean(actual_scores)) if actual_scores else None,
                            "actual_macro_f1_std": float(np.std(actual_scores)) if actual_scores else None,
                            "quota": {str(cls): int(quota[int(cls)]) for cls in CLASSES},
                        }
                    )
    frame = pd.DataFrame(records)
    csv_path = REPORT_DIR / "combo_allocation_oracle_results.csv"
    frame.to_csv(csv_path, index=False, encoding="utf-8")
    summary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "features": features,
        "top_expected": json.loads(
            frame.sort_values("expected_macro_f1", ascending=False).head(30).to_json(orient="records", force_ascii=False)
        ),
        "by_train_estimator": json.loads(
            frame[frame["prob_mode"] != "valid_oracle"]
            .sort_values("expected_macro_f1", ascending=False)
            .head(30)
            .to_json(orient="records", force_ascii=False)
        ),
        "result_csv": str(csv_path.relative_to(ROOT)),
        "conclusion": "Tests whether assigning mixed labels inside duplicate exact-feature combos can beat single-label-per-combo models when row identity is unavailable.",
    }
    out = REPORT_DIR / "combo_allocation_oracle_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary["top_expected"][:12], indent=2, ensure_ascii=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
