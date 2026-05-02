from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
REPORT_DIR = ROOT / "reports"
SUBMISSION_DIR = ROOT / "submissions"
CLASSES = np.array([0, 1, 2], dtype=int)
HIDDEN_COUNTS = {0: 14000, 1: 2500, 2: 3500}
EPS = 1e-12


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


def keys(frame: pd.DataFrame, features: list[str]) -> list[tuple[int, ...]]:
    return list(map(tuple, frame[features].to_numpy(dtype=np.int16)))


def make_vocab(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> tuple[dict[tuple[int, ...], int], list[tuple[int, ...]]]:
    vocab_keys = sorted(set(keys(train, features)) | set(keys(test, features)))
    return {key: i for i, key in enumerate(vocab_keys)}, vocab_keys


def class_sorted_positions(frame: pd.DataFrame) -> dict[int, np.ndarray]:
    y = frame["label"].to_numpy(dtype=int)
    ids = frame["_id"].to_numpy(dtype=int)
    out: dict[int, np.ndarray] = {}
    for cls in CLASSES:
        idx = np.where(y == cls)[0]
        out[int(cls)] = idx[np.argsort(ids[idx])]
    return out


def make_prefix_split(train: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, int]]:
    positions = class_sorted_positions(train)
    fit_parts = []
    valid_parts = []
    heldout: dict[int, int] = {}
    for cls in CLASSES:
        pos = positions[int(cls)]
        n = len(pos)
        if mode == "ratio_prefix":
            hold = int(round(n * HIDDEN_COUNTS[int(cls)] / (n + HIDDEN_COUNTS[int(cls)])))
        elif mode == "actual_prefix":
            hold = min(HIDDEN_COUNTS[int(cls)], n // 2)
        elif mode == "small_prefix":
            hold = max(600, int(round(0.12 * n)))
        else:
            raise ValueError(mode)
        heldout[int(cls)] = hold
        valid_parts.append(train.iloc[pos[:hold]])
        fit_parts.append(train.iloc[pos[hold:]])
    return pd.concat(fit_parts, ignore_index=True), pd.concat(valid_parts, ignore_index=True), heldout


def fit_trend_distribution(
    fit: pd.DataFrame,
    features: list[str],
    vocab: dict[tuple[int, ...], int],
    hidden_counts: dict[int, int],
    n_bins: int,
    degree: int,
    alpha: float,
    trend_weight: float,
    front_weight: float,
) -> np.ndarray:
    n_vocab = len(vocab)
    p = np.zeros((len(CLASSES), n_vocab), dtype=float)
    global_counts = np.zeros(n_vocab, dtype=float)
    fit_keys = keys(fit, features)
    for key in fit_keys:
        global_counts[vocab[key]] += 1.0
    global_prob = (global_counts + alpha) / (global_counts.sum() + alpha * n_vocab)

    positions = class_sorted_positions(fit)
    key_arr = np.array([vocab[key] for key in fit_keys], dtype=int)
    for cls in CLASSES:
        pos = positions[int(cls)]
        m = len(pos)
        h = int(hidden_counts[int(cls)])
        total = h + m
        if m == 0:
            p[int(cls)] = global_prob
            continue
        bins = [part for part in np.array_split(pos, min(n_bins, m)) if len(part)]
        freq = np.zeros((len(bins), n_vocab), dtype=float)
        x = np.zeros(len(bins), dtype=float)
        for b, part in enumerate(bins):
            counts = np.bincount(key_arr[part], minlength=n_vocab).astype(float)
            freq[b] = (counts + alpha * global_prob) / (counts.sum() + alpha)
            local_rank_center = (part.min() + part.max()) / 2.0
            # fit indices start after the hidden prefix in the simulated original block
            x[b] = (h + local_rank_center + 0.5) / total
        hidden_centers = (np.arange(max(h, 1), dtype=float) + 0.5) / total
        if degree == 0 or len(bins) <= degree:
            trend = np.tile(freq[0], (len(hidden_centers), 1)).mean(axis=0)
        else:
            design = np.vstack([x**d for d in range(degree + 1)]).T
            coef, *_ = np.linalg.lstsq(design, freq, rcond=None)
            hidden_design = np.vstack([hidden_centers**d for d in range(degree + 1)]).T
            trend = (hidden_design @ coef).mean(axis=0)
            trend = np.clip(trend, 0.0, None)
            if trend.sum() <= 0:
                trend = freq[0].copy()
            else:
                trend = trend / trend.sum()
        front = freq[0]
        class_counts = np.bincount(key_arr[pos], minlength=n_vocab).astype(float)
        class_all = (class_counts + alpha * global_prob) / (class_counts.sum() + alpha)
        combo = trend_weight * trend + front_weight * front + max(0.0, 1.0 - trend_weight - front_weight) * class_all
        combo = np.clip(combo, EPS, None)
        p[int(cls)] = combo / combo.sum()
    return p


def predict_from_distribution(
    target: pd.DataFrame,
    features: list[str],
    vocab: dict[tuple[int, ...], int],
    dist: np.ndarray,
    prior: np.ndarray,
) -> np.ndarray:
    target_keys = keys(target, features)
    log_scores = np.zeros((len(target), len(CLASSES)), dtype=float)
    log_prior = np.log(np.clip(prior, EPS, None))
    fallback = np.log(np.clip(dist.mean(axis=0), EPS, None))
    for i, key in enumerate(target_keys):
        j = vocab.get(key)
        if j is None:
            log_scores[i] = log_prior + fallback.mean()
        else:
            log_scores[i] = log_prior + np.log(np.clip(dist[:, j], EPS, None))
    z = log_scores - log_scores.max(axis=1, keepdims=True)
    proba = np.exp(np.clip(z, -80, 80))
    return proba / proba.sum(axis=1, keepdims=True)


def adjust_to_quota(scores: np.ndarray, target_counts: np.ndarray) -> np.ndarray:
    target = np.asarray(target_counts, dtype=int)
    pred = scores.argmax(axis=1).astype(int)
    counts = np.bincount(pred, minlength=len(CLASSES))
    log_scores = np.log(np.clip(scores, EPS, None))
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
    return pred


def score(y_true: np.ndarray, pred: np.ndarray) -> dict[str, object]:
    return {
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
        "per_class_f1": {
            str(cls): float(v)
            for cls, v in zip(CLASSES, f1_score(y_true, pred, labels=CLASSES, average=None))
        },
        "counts": {str(cls): int((pred == cls).sum()) for cls in CLASSES},
    }


def validate_submission(test: pd.DataFrame, submission: pd.DataFrame) -> None:
    if list(submission.columns) != ["name", "label"]:
        raise ValueError("Submission must have columns name,label")
    if len(submission) != len(test):
        raise ValueError("Submission row count mismatch")
    if not submission["name"].equals(test["name"]):
        raise ValueError("Submission name order mismatch")


def main() -> None:
    start = time.time()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    data_dir = find_data_dir()
    train = pd.read_csv(data_dir / "data_train.csv")
    test = pd.read_csv(data_dir / "data_test.csv")
    features = [c for c in train.columns if c not in ["name", "label"]]
    train["_id"] = train["name"].map(extract_id)
    vocab, _ = make_vocab(train, test, features)

    grid = []
    for n_bins in [4, 8, 12]:
        for degree in [0, 1, 2]:
            for alpha in [0.5, 3.0, 15.0]:
                for trend_weight, front_weight in [(0.0, 1.0), (0.25, 0.75), (0.5, 0.5), (0.75, 0.25), (1.0, 0.0)]:
                    grid.append((n_bins, degree, alpha, trend_weight, front_weight))

    records: list[dict[str, object]] = []
    top_by_split: dict[str, list[dict[str, object]]] = {}
    for split_mode in ["small_prefix", "ratio_prefix", "actual_prefix"]:
        fit, valid, heldout = make_prefix_split(train, split_mode)
        y_true = valid["label"].to_numpy(dtype=int)
        target_counts = np.bincount(y_true, minlength=len(CLASSES))
        prior = target_counts / target_counts.sum()
        split_records = []
        for n_bins, degree, alpha, trend_weight, front_weight in grid:
            dist = fit_trend_distribution(
                fit,
                features,
                vocab,
                hidden_counts=heldout,
                n_bins=n_bins,
                degree=degree,
                alpha=alpha,
                trend_weight=trend_weight,
                front_weight=front_weight,
            )
            proba = predict_from_distribution(valid, features, vocab, dist, prior)
            for variant, pred in [
                ("raw", proba.argmax(axis=1).astype(int)),
                ("quota", adjust_to_quota(proba, target_counts)),
            ]:
                metrics = score(y_true, pred)
                rec = {
                    "split": split_mode,
                    "variant": variant,
                    "n_bins": n_bins,
                    "degree": degree,
                    "alpha": alpha,
                    "trend_weight": trend_weight,
                    "front_weight": front_weight,
                    **metrics,
                }
                records.append(rec)
                split_records.append(rec)
        top_by_split[split_mode] = sorted(split_records, key=lambda r: float(r["macro_f1"]), reverse=True)[:15]
        print(split_mode, top_by_split[split_mode][0])

    results = pd.DataFrame(records)
    key_cols = ["variant", "n_bins", "degree", "alpha", "trend_weight", "front_weight"]
    avg = (
        results.groupby(key_cols)["macro_f1"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        .sort_values(["mean", "min"], ascending=False)
    )
    best = avg.iloc[0].to_dict()
    full_hidden = HIDDEN_COUNTS
    full_prior = np.array([full_hidden[int(cls)] for cls in CLASSES], dtype=float)
    full_prior = full_prior / full_prior.sum()
    dist = fit_trend_distribution(
        train,
        features,
        vocab,
        hidden_counts=full_hidden,
        n_bins=int(best["n_bins"]),
        degree=int(best["degree"]),
        alpha=float(best["alpha"]),
        trend_weight=float(best["trend_weight"]),
        front_weight=float(best["front_weight"]),
    )
    test_proba = predict_from_distribution(test, features, vocab, dist, full_prior)
    pred = adjust_to_quota(test_proba, np.array([14000, 2500, 3500], dtype=int))
    submission = pd.DataFrame({"name": test["name"], "label": pred})
    validate_submission(test, submission)
    sub_path = SUBMISSION_DIR / "submission_combo_trend_gap_prior_v1.csv"
    submission.to_csv(sub_path, index=False, encoding="utf-8", lineterminator="\n")

    csv_path = REPORT_DIR / "combo_trend_extrapolation_results.csv"
    results.to_csv(csv_path, index=False, encoding="utf-8")
    summary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "runtime_seconds": round(time.time() - start, 3),
        "features": features,
        "grid_size": len(grid),
        "top_by_split": top_by_split,
        "top_by_average": json.loads(avg.head(30).to_json(orient="records", force_ascii=False)),
        "generated_submission": {
            "path": str(sub_path.relative_to(ROOT)),
            "counts": {str(cls): int((pred == cls).sum()) for cls in CLASSES},
            "diff_vs_labelshift_q4": None,
        },
        "result_csv": str(csv_path.relative_to(ROOT)),
        "conclusion": "Trend extrapolation is useful only if prefix splits improve materially over front-window exact posterior and MLP quota proxies.",
    }
    q4_path = SUBMISSION_DIR / "submission_mlp_labelshift_hard_q4_v1.csv"
    if q4_path.exists():
        q4 = pd.read_csv(q4_path)["label"].to_numpy(dtype=int)
        summary["generated_submission"]["diff_vs_labelshift_q4"] = int((pred != q4).sum())
    out_path = REPORT_DIR / "combo_trend_extrapolation_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary["top_by_average"][:10], indent=2, ensure_ascii=False))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
