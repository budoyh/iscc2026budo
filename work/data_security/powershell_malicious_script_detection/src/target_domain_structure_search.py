from __future__ import annotations

import itertools
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

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


def softmax_log_scores(log_scores: np.ndarray) -> np.ndarray:
    z = log_scores - log_scores.max(axis=1, keepdims=True)
    exp = np.exp(np.clip(z, -80, 80))
    return exp / exp.sum(axis=1, keepdims=True)


def adjust_to_quota(scores: np.ndarray, target_counts: np.ndarray) -> np.ndarray:
    target = np.asarray(target_counts, dtype=int)
    if target.sum() != len(scores):
        raise ValueError(f"target_counts={target.tolist()} len={len(scores)}")
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
            raise RuntimeError(f"Could not satisfy quota for class {dst}: {changed}/{need}")
    final = np.bincount(pred, minlength=len(CLASSES))
    if not np.array_equal(final, target):
        raise RuntimeError(f"Quota failed: {final.tolist()} vs {target.tolist()}")
    return pred


def key_rows(frame: pd.DataFrame, features: list[str], subset: tuple[int, ...] | None) -> list[tuple[int, ...]]:
    cols = features if subset is None else [features[i] for i in subset]
    return list(map(tuple, frame[cols].to_numpy(dtype=np.int16)))


def class_positions(frame: pd.DataFrame) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for label in CLASSES:
        idx = np.where(frame["label"].to_numpy(dtype=int) == label)[0]
        ids = frame["_id"].to_numpy(dtype=int)[idx]
        out[int(label)] = idx[np.argsort(ids)]
    return out


def make_split(train: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    positions = class_positions(train)
    valid_parts = []
    fit_parts = []
    rng = np.random.default_rng(SEED)
    for label in CLASSES:
        pos = positions[int(label)]
        n = len(pos)
        if mode == "ratio_prefix":
            total = n + HIDDEN_COUNTS[int(label)]
            hold = int(round(n * HIDDEN_COUNTS[int(label)] / total))
            valid_idx = pos[:hold]
            fit_idx = pos[hold:]
        elif mode == "actual_prefix":
            hold = min(HIDDEN_COUNTS[int(label)], n // 2)
            valid_idx = pos[:hold]
            fit_idx = pos[hold:]
        elif mode == "small_prefix":
            hold = max(600, int(round(0.12 * n)))
            valid_idx = pos[:hold]
            fit_idx = pos[hold:]
        elif mode == "middle_ratio":
            total = n + HIDDEN_COUNTS[int(label)]
            hold = int(round(n * HIDDEN_COUNTS[int(label)] / total))
            start = max(0, (n - hold) // 2)
            valid_idx = pos[start : start + hold]
            fit_idx = np.concatenate([pos[:start], pos[start + hold :]])
        elif mode == "suffix_ratio":
            total = n + HIDDEN_COUNTS[int(label)]
            hold = int(round(n * HIDDEN_COUNTS[int(label)] / total))
            valid_idx = pos[-hold:]
            fit_idx = pos[:-hold]
        elif mode == "random_ratio":
            total = n + HIDDEN_COUNTS[int(label)]
            hold = int(round(n * HIDDEN_COUNTS[int(label)] / total))
            shuffled = pos.copy()
            rng.shuffle(shuffled)
            valid_idx = shuffled[:hold]
            fit_idx = shuffled[hold:]
        else:
            raise ValueError(mode)
        valid_parts.append(train.iloc[valid_idx])
        fit_parts.append(train.iloc[fit_idx])
    fit = pd.concat(fit_parts, ignore_index=True)
    valid = pd.concat(valid_parts, ignore_index=True)
    return fit, valid


def source_weights(fit: pd.DataFrame, mode: str) -> np.ndarray:
    weights = np.zeros(len(fit), dtype=float)
    positions = class_positions(fit)
    for label in CLASSES:
        pos = positions[int(label)]
        n = len(pos)
        if mode == "all":
            local = np.ones(n, dtype=float)
        elif mode.startswith("front"):
            size = int(mode.replace("front", ""))
            local = np.zeros(n, dtype=float)
            local[: min(size, n)] = 1.0
        elif mode.startswith("frac"):
            frac = float(mode.replace("frac", ""))
            size = max(1, int(round(frac * n)))
            local = np.zeros(n, dtype=float)
            local[:size] = 1.0
        elif mode.startswith("decay"):
            tau = float(mode.replace("decay", ""))
            x = np.linspace(0.0, 1.0, n)
            local = np.exp(-tau * x)
        else:
            raise ValueError(mode)
        weights[pos] = local
    if weights.sum() <= 0:
        raise ValueError(f"empty weights for {mode}")
    return weights


def exact_posterior(
    fit: pd.DataFrame,
    target: pd.DataFrame,
    features: list[str],
    weights: np.ndarray,
    alpha: float,
    prior: np.ndarray,
) -> np.ndarray:
    train_keys = key_rows(fit, features, None)
    target_keys = key_rows(target, features, None)
    labels = fit["label"].to_numpy(dtype=int)
    global_counts = np.bincount(labels, weights=weights, minlength=len(CLASSES)).astype(float)
    global_prob = global_counts / global_counts.sum()
    counts: dict[tuple[int, ...], np.ndarray] = defaultdict(lambda: np.zeros(len(CLASSES), dtype=float))
    for key, label, weight in zip(train_keys, labels, weights):
        counts[key][int(label)] += float(weight)
    out = np.zeros((len(target), len(CLASSES)), dtype=float)
    for i, key in enumerate(target_keys):
        c = counts.get(key)
        if c is None:
            prob = global_prob.copy()
        else:
            prob = c + alpha * prior
            prob = prob / prob.sum()
        out[i] = prob
    return out


def build_subset_counts(
    fit: pd.DataFrame,
    features: list[str],
    subsets: Iterable[tuple[int, ...] | None],
    weights: np.ndarray,
) -> tuple[dict[tuple[int, ...] | None, list[Counter]], dict[tuple[int, ...] | None, Counter]]:
    labels = fit["label"].to_numpy(dtype=int)
    by_subset: dict[tuple[int, ...] | None, list[Counter]] = {}
    global_by_subset: dict[tuple[int, ...] | None, Counter] = {}
    for subset in subsets:
        class_counts = [Counter() for _ in CLASSES]
        global_counts: Counter = Counter()
        for key, label, weight in zip(key_rows(fit, features, subset), labels, weights):
            w = float(weight)
            if w <= 0:
                continue
            class_counts[int(label)][key] += w
            global_counts[key] += w
        by_subset[subset] = class_counts
        global_by_subset[subset] = global_counts
    return by_subset, global_by_subset


def subset_nb(
    fit: pd.DataFrame,
    target: pd.DataFrame,
    features: list[str],
    weights: np.ndarray,
    alpha: float,
    prior: np.ndarray,
    use_full: float,
    use_single: float,
    use_pair: float,
) -> np.ndarray:
    single_subsets = [(i,) for i in range(len(features))]
    pair_subsets = list(itertools.combinations(range(len(features)), 2))
    subsets: list[tuple[int, ...] | None] = []
    if use_full > 0:
        subsets.append(None)
    if use_single > 0:
        subsets.extend(single_subsets)
    if use_pair > 0:
        subsets.extend(pair_subsets)
    by_subset, global_by_subset = build_subset_counts(fit, features, subsets, weights)
    labels = fit["label"].to_numpy(dtype=int)
    class_mass = np.bincount(labels, weights=weights, minlength=len(CLASSES)).astype(float)
    total_mass = float(class_mass.sum())

    log_scores = np.tile(np.log(np.clip(prior, 1e-12, None)), (len(target), 1))

    def add_subset_group(group: list[tuple[int, ...] | None], group_weight: float) -> None:
        if group_weight <= 0:
            return
        per_subset_weight = group_weight / max(1, len(group))
        for subset in group:
            global_counts = global_by_subset[subset]
            global_total = max(total_mass, 1e-12)
            target_keys = key_rows(target, features, subset)
            class_counts = by_subset[subset]
            for i, key in enumerate(target_keys):
                gprob = float(global_counts.get(key, 0.0)) / global_total
                if gprob <= 0:
                    gprob = 1e-12
                for cls in CLASSES:
                    cnt = float(class_counts[int(cls)].get(key, 0.0))
                    denom = class_mass[int(cls)] + alpha
                    prob = (cnt + alpha * gprob) / max(denom, 1e-12)
                    log_scores[i, int(cls)] += per_subset_weight * math.log(max(prob, 1e-15))

    if use_full > 0:
        add_subset_group([None], use_full)
    if use_single > 0:
        add_subset_group(single_subsets, use_single)
    if use_pair > 0:
        add_subset_group(pair_subsets, use_pair)
    return softmax_log_scores(log_scores)


def summarize_prediction(y_true: np.ndarray, pred: np.ndarray) -> dict[str, object]:
    return {
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
        "per_class_f1": {
            str(cls): float(score)
            for cls, score in zip(CLASSES, f1_score(y_true, pred, average=None, labels=CLASSES))
        },
        "counts": {str(cls): int((pred == cls).sum()) for cls in CLASSES},
    }


def evaluate_scores(scores: np.ndarray, y_true: np.ndarray, target_counts: np.ndarray) -> dict[str, object]:
    raw = scores.argmax(axis=1).astype(int)
    quota = adjust_to_quota(scores, target_counts)
    return {
        "raw": summarize_prediction(y_true, raw),
        "quota": summarize_prediction(y_true, quota),
    }


def main() -> None:
    start = time.time()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    data_dir = find_data_dir()
    train = pd.read_csv(data_dir / "data_train.csv")
    features = [c for c in train.columns if c not in ["name", "label"]]
    train["_id"] = train["name"].map(extract_id)

    # First pass deliberately stays compact. The full pairwise grid was too slow
    # for iteration; expand only if this fast pass shows a real proxy jump.
    split_modes = ["ratio_prefix", "actual_prefix", "middle_ratio", "suffix_ratio"]
    weight_modes = ["all", "front2500", "front5000", "front9000", "frac0.5", "decay3.0"]
    exact_alphas = [0.05, 0.5, 2.0, 10.0]
    nb_configs = [
        ("nb_single", 0.0, 1.0, 0.0),
        ("nb_full", 1.0, 0.0, 0.0),
        ("nb_full_single", 1.0, 0.5, 0.0),
    ]
    nb_alphas = [0.5, 5.0]

    records: list[dict[str, object]] = []
    top_by_split: dict[str, list[dict[str, object]]] = {}
    for split_mode in split_modes:
        fit, valid = make_split(train, split_mode)
        y_true = valid["label"].to_numpy(dtype=int)
        target_counts = np.bincount(y_true, minlength=len(CLASSES))
        prior = target_counts / target_counts.sum()
        split_records: list[dict[str, object]] = []
        for weight_mode in weight_modes:
            weights = source_weights(fit, weight_mode)
            active = weights > 0
            if active.sum() < 100:
                continue
            for alpha in exact_alphas:
                scores = exact_posterior(fit, valid, features, weights, alpha=alpha, prior=prior)
                metrics = evaluate_scores(scores, y_true, target_counts)
                for variant in ["raw", "quota"]:
                    rec = {
                        "split": split_mode,
                        "model": "exact_posterior",
                        "variant": variant,
                        "weight_mode": weight_mode,
                        "alpha": alpha,
                        "active_rows": int(active.sum()),
                        "macro_f1": metrics[variant]["macro_f1"],
                        "per_class_f1": metrics[variant]["per_class_f1"],
                        "pred_counts": metrics[variant]["counts"],
                    }
                    records.append(rec)
                    split_records.append(rec)
            for model_name, use_full, use_single, use_pair in nb_configs:
                for alpha in nb_alphas:
                    scores = subset_nb(
                        fit,
                        valid,
                        features,
                        weights,
                        alpha=alpha,
                        prior=prior,
                        use_full=use_full,
                        use_single=use_single,
                        use_pair=use_pair,
                    )
                    metrics = evaluate_scores(scores, y_true, target_counts)
                    for variant in ["raw", "quota"]:
                        rec = {
                            "split": split_mode,
                            "model": model_name,
                            "variant": variant,
                            "weight_mode": weight_mode,
                            "alpha": alpha,
                            "active_rows": int(active.sum()),
                            "macro_f1": metrics[variant]["macro_f1"],
                            "per_class_f1": metrics[variant]["per_class_f1"],
                            "pred_counts": metrics[variant]["counts"],
                        }
                        records.append(rec)
                        split_records.append(rec)
        top_by_split[split_mode] = sorted(split_records, key=lambda x: float(x["macro_f1"]), reverse=True)[:15]
        print(split_mode, top_by_split[split_mode][0])

    frame = pd.DataFrame(records)
    key_cols = ["model", "variant", "weight_mode", "alpha"]
    agg = (
        frame.groupby(key_cols)["macro_f1"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        .sort_values(["mean", "min"], ascending=False)
    )
    out_csv = REPORT_DIR / "target_domain_structure_search_results.csv"
    frame.to_csv(out_csv, index=False, encoding="utf-8")
    summary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "runtime_seconds": round(time.time() - start, 3),
        "features": features,
        "split_modes": split_modes,
        "weight_modes": weight_modes,
        "total_records": len(records),
        "top_by_split": top_by_split,
        "top_by_average": json.loads(agg.head(30).to_json(orient="records", force_ascii=False)),
        "result_csv": str(out_csv.relative_to(ROOT)),
        "conclusion": "Large gains on prefix-like splits would support target-domain source modeling; flat or unstable results falsify this generative/backoff family.",
    }
    out_json = REPORT_DIR / "target_domain_structure_search_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary["top_by_average"][:10], indent=2, ensure_ascii=False))
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
