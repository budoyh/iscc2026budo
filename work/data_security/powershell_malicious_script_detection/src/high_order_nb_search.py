from __future__ import annotations

import itertools
import json
import math
import re
import time
from collections import Counter
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


def class_positions(frame: pd.DataFrame) -> dict[int, np.ndarray]:
    y = frame["label"].to_numpy(dtype=int)
    ids = frame["_id"].to_numpy(dtype=int)
    out = {}
    for cls in CLASSES:
        idx = np.where(y == cls)[0]
        out[int(cls)] = idx[np.argsort(ids[idx])]
    return out


def make_split(train: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    pos = class_positions(train)
    fit_parts, valid_parts = [], []
    for cls in CLASSES:
        p = pos[int(cls)]
        n = len(p)
        if mode == "ratio_prefix":
            hold = int(round(n * HIDDEN_COUNTS[int(cls)] / (n + HIDDEN_COUNTS[int(cls)])))
            valid_idx, fit_idx = p[:hold], p[hold:]
        elif mode == "actual_prefix":
            hold = min(HIDDEN_COUNTS[int(cls)], n // 2)
            valid_idx, fit_idx = p[:hold], p[hold:]
        elif mode == "middle_ratio":
            hold = int(round(n * HIDDEN_COUNTS[int(cls)] / (n + HIDDEN_COUNTS[int(cls)])))
            start = (n - hold) // 2
            valid_idx = p[start : start + hold]
            fit_idx = np.concatenate([p[:start], p[start + hold :]])
        else:
            raise ValueError(mode)
        fit_parts.append(train.iloc[fit_idx])
        valid_parts.append(train.iloc[valid_idx])
    return pd.concat(fit_parts, ignore_index=True), pd.concat(valid_parts, ignore_index=True)


def source_select(fit: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "all":
        return fit
    parts = []
    pos = class_positions(fit)
    for cls in CLASSES:
        p = pos[int(cls)]
        if mode == "front025":
            take = max(1, int(round(0.25 * len(p))))
        elif mode == "front05":
            take = max(1, int(round(0.5 * len(p))))
        elif mode == "front075":
            take = max(1, int(round(0.75 * len(p))))
        else:
            raise ValueError(mode)
        parts.append(fit.iloc[p[:take]])
    return pd.concat(parts, ignore_index=True)


def subset_keys(arr: np.ndarray, subset: tuple[int, ...]) -> list[tuple[int, ...]]:
    return list(map(tuple, arr[:, subset]))


def fit_counts(source: pd.DataFrame, features: list[str], subsets: list[tuple[int, ...]]):
    x = source[features].to_numpy(dtype=np.int16)
    y = source["label"].to_numpy(dtype=int)
    class_mass = np.bincount(y, minlength=len(CLASSES)).astype(float)
    counts = []
    global_counts = []
    for subset in subsets:
        per_class = [Counter() for _ in CLASSES]
        glob = Counter()
        for key, label in zip(subset_keys(x, subset), y):
            per_class[int(label)][key] += 1
            glob[key] += 1
        counts.append(per_class)
        global_counts.append(glob)
    return counts, global_counts, class_mass, float(len(source))


def score_nb(
    source: pd.DataFrame,
    target: pd.DataFrame,
    features: list[str],
    subset_groups: list[tuple[str, list[tuple[int, ...]], float]],
    alpha: float,
    prior: np.ndarray,
) -> np.ndarray:
    all_subsets = [subset for _, subsets, _ in subset_groups for subset in subsets]
    counts, global_counts, class_mass, total = fit_counts(source, features, all_subsets)
    target_x = target[features].to_numpy(dtype=np.int16)
    log_scores = np.tile(np.log(np.clip(prior, EPS, None)), (len(target), 1))
    offset = 0
    for _, subsets, group_weight in subset_groups:
        if group_weight <= 0:
            offset += len(subsets)
            continue
        per_weight = group_weight / max(1, len(subsets))
        for subset in subsets:
            per_class = counts[offset]
            glob = global_counts[offset]
            keys = subset_keys(target_x, subset)
            for i, key in enumerate(keys):
                gprob = (glob.get(key, 0.0) + 1.0) / (total + max(1, len(glob)))
                for cls in CLASSES:
                    prob = (per_class[int(cls)].get(key, 0.0) + alpha * gprob) / max(EPS, class_mass[int(cls)] + alpha)
                    log_scores[i, int(cls)] += per_weight * math.log(max(prob, EPS))
            offset += 1
    z = log_scores - log_scores.max(axis=1, keepdims=True)
    p = np.exp(np.clip(z, -80, 80))
    return p / p.sum(axis=1, keepdims=True)


def adjust_to_quota(scores: np.ndarray, target_counts: np.ndarray) -> np.ndarray:
    target = np.asarray(target_counts, dtype=int)
    pred = scores.argmax(axis=1).astype(int)
    counts = np.bincount(pred, minlength=len(CLASSES))
    log_scores = np.log(np.clip(scores, EPS, None))
    for dst in np.where(target > counts)[0]:
        need = int(target[dst] - counts[dst])
        moves = []
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


def metric(y: np.ndarray, pred: np.ndarray) -> dict[str, object]:
    return {
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "per_class_f1": {str(cls): float(v) for cls, v in zip(CLASSES, f1_score(y, pred, average=None, labels=CLASSES))},
        "counts": {str(cls): int((pred == cls).sum()) for cls in CLASSES},
    }


def validate_submission(test: pd.DataFrame, submission: pd.DataFrame) -> None:
    if list(submission.columns) != ["name", "label"]:
        raise ValueError("Submission columns must be name,label")
    if len(submission) != len(test):
        raise ValueError("Submission length mismatch")
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
    singles = [(i,) for i in range(len(features))]
    pairs = list(itertools.combinations(range(len(features)), 2))
    configs = [
        ("single", [(singles, 1.0)]),
        ("pair", [(pairs, 1.0)]),
        ("single_pair", [(singles, 0.5), (pairs, 1.0)]),
    ]
    records = []
    for split in ["ratio_prefix", "actual_prefix", "middle_ratio"]:
        fit, valid = make_split(train, split)
        y = valid["label"].to_numpy(dtype=int)
        target_counts = np.bincount(y, minlength=len(CLASSES))
        prior = target_counts / target_counts.sum()
        split_records = []
        for source_mode in ["all", "front025", "front05"]:
            source = source_select(fit, source_mode)
            for cfg_name, groups in configs:
                subset_groups = [(f"g{i}", subsets, weight) for i, (subsets, weight) in enumerate(groups)]
                for alpha in [0.5, 5.0]:
                    scores = score_nb(source, valid, features, subset_groups, alpha=alpha, prior=prior)
                    for variant, pred in [
                        ("raw", scores.argmax(axis=1).astype(int)),
                        ("quota", adjust_to_quota(scores, target_counts)),
                    ]:
                        rec = {
                            "split": split,
                            "source_mode": source_mode,
                            "config": cfg_name,
                            "alpha": alpha,
                            "variant": variant,
                            **metric(y, pred),
                        }
                        records.append(rec)
                        split_records.append(rec)
        print(split, sorted(split_records, key=lambda r: float(r["macro_f1"]), reverse=True)[0])
    frame = pd.DataFrame(records)
    avg = (
        frame.groupby(["source_mode", "config", "alpha", "variant"])["macro_f1"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        .sort_values(["mean", "min"], ascending=False)
    )
    best = avg.iloc[0].to_dict()
    full_prior = np.array([14000, 2500, 3500], dtype=float)
    full_prior /= full_prior.sum()
    groups = dict(configs)[best["config"]]
    subset_groups = [(f"g{i}", subsets, weight) for i, (subsets, weight) in enumerate(groups)]
    source = source_select(train, best["source_mode"])
    scores = score_nb(source, test, features, subset_groups, alpha=float(best["alpha"]), prior=full_prior)
    if best["variant"] == "quota":
        pred = adjust_to_quota(scores, np.array([14000, 2500, 3500], dtype=int))
    else:
        pred = scores.argmax(axis=1).astype(int)
    sub = pd.DataFrame({"name": test["name"], "label": pred})
    validate_submission(test, sub)
    sub_path = SUBMISSION_DIR / "submission_high_order_nb_v1.csv"
    sub.to_csv(sub_path, index=False, encoding="utf-8", lineterminator="\n")
    q4_path = SUBMISSION_DIR / "submission_mlp_labelshift_hard_q4_v1.csv"
    diff_q4 = None
    if q4_path.exists():
        q4 = pd.read_csv(q4_path)["label"].to_numpy(dtype=int)
        diff_q4 = int((pred != q4).sum())
    csv_path = REPORT_DIR / "high_order_nb_search_results.csv"
    frame.to_csv(csv_path, index=False, encoding="utf-8")
    summary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "runtime_seconds": round(time.time() - start, 3),
        "top_by_average": json.loads(avg.head(30).to_json(orient="records", force_ascii=False)),
        "generated_submission": {
            "path": str(sub_path.relative_to(ROOT)),
            "counts": {str(cls): int((pred == cls).sum()) for cls in CLASSES},
            "diff_vs_labelshift_q4": diff_q4,
            "best_params": best,
        },
        "result_csv": str(csv_path.relative_to(ROOT)),
    }
    out = REPORT_DIR / "high_order_nb_search_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
