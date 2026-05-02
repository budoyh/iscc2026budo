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


def class_sorted_positions(frame: pd.DataFrame) -> dict[int, np.ndarray]:
    y = frame["label"].to_numpy(dtype=int)
    ids = frame["_id"].to_numpy(dtype=int)
    out: dict[int, np.ndarray] = {}
    for cls in CLASSES:
        idx = np.where(y == cls)[0]
        out[int(cls)] = idx[np.argsort(ids[idx])]
    return out


def make_split(train: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    positions = class_sorted_positions(train)
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


def keys_array(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    return frame[features].to_numpy(dtype=np.int16)


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


def graph_proba(
    fit: pd.DataFrame,
    target: pd.DataFrame,
    features: list[str],
    alpha_seed: float,
    gamma: float,
    knn: int,
    prop_alpha: float,
    max_iter: int = 120,
) -> np.ndarray:
    fit_x = keys_array(fit, features)
    target_x = keys_array(target, features)
    all_x = np.vstack([fit_x, target_x])
    uniq, inv = np.unique(all_x, axis=0, return_inverse=True)
    n_fit = len(fit_x)
    fit_nodes = inv[:n_fit]
    target_nodes = inv[n_fit:]
    n_nodes = len(uniq)

    y = fit["label"].to_numpy(dtype=int)
    class_counts = np.zeros((n_nodes, len(CLASSES)), dtype=float)
    for node, label in zip(fit_nodes, y):
        class_counts[node, int(label)] += 1.0
    global_prob = np.bincount(y, minlength=len(CLASSES)).astype(float)
    global_prob /= global_prob.sum()
    seed = class_counts + alpha_seed * global_prob
    seed_mass = class_counts.sum(axis=1, keepdims=True)
    seed = np.where(seed_mass > 0, seed / seed.sum(axis=1, keepdims=True), 0.0)
    labeled = (seed_mass[:, 0] > 0).astype(float)[:, None]

    diff = uniq[:, None, :] != uniq[None, :, :]
    dist = diff.sum(axis=2).astype(float)
    sim = np.exp(-gamma * dist)
    np.fill_diagonal(sim, 0.0)
    if knn < n_nodes - 1:
        keep = np.argpartition(sim, -knn, axis=1)[:, -knn:]
        mask = np.zeros_like(sim, dtype=bool)
        rows = np.arange(n_nodes)[:, None]
        mask[rows, keep] = True
        sim = np.where(mask, sim, 0.0)
    sim = np.maximum(sim, sim.T)
    row_sum = sim.sum(axis=1, keepdims=True)
    trans = np.divide(sim, np.maximum(row_sum, EPS))

    f = np.where(labeled > 0, seed, global_prob)
    clamp = seed.copy()
    for _ in range(max_iter):
        new_f = prop_alpha * (trans @ f) + (1.0 - prop_alpha) * global_prob
        new_f = np.where(labeled > 0, 0.75 * clamp + 0.25 * new_f, new_f)
        new_f = np.clip(new_f, EPS, None)
        new_f = new_f / new_f.sum(axis=1, keepdims=True)
        if np.abs(new_f - f).max() < 1e-7:
            f = new_f
            break
        f = new_f
    return f[target_nodes]


def metric(y_true: np.ndarray, pred: np.ndarray) -> dict[str, object]:
    return {
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
        "per_class_f1": {
            str(cls): float(v)
            for cls, v in zip(CLASSES, f1_score(y_true, pred, labels=CLASSES, average=None))
        },
        "counts": {str(cls): int((pred == cls).sum()) for cls in CLASSES},
    }


def main() -> None:
    start = time.time()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    data_dir = find_data_dir()
    train = pd.read_csv(data_dir / "data_train.csv")
    features = [c for c in train.columns if c not in ["name", "label"]]
    train["_id"] = train["name"].map(extract_id)
    grid = []
    for alpha_seed in [0.1, 1.0, 5.0]:
        for gamma in [0.7, 1.2, 2.0]:
            for knn in [20, 50, 120]:
                for prop_alpha in [0.5, 0.75, 0.9]:
                    grid.append((alpha_seed, gamma, knn, prop_alpha))
    records = []
    for split_mode in ["ratio_prefix", "actual_prefix", "middle_ratio"]:
        fit, valid = make_split(train, split_mode)
        y_true = valid["label"].to_numpy(dtype=int)
        target_counts = np.bincount(y_true, minlength=len(CLASSES))
        split_records = []
        for alpha_seed, gamma, knn, prop_alpha in grid:
            proba = graph_proba(fit, valid, features, alpha_seed, gamma, knn, prop_alpha)
            for variant, pred in [
                ("raw", proba.argmax(axis=1).astype(int)),
                ("quota", adjust_to_quota(proba, target_counts)),
            ]:
                rec = {
                    "split": split_mode,
                    "variant": variant,
                    "alpha_seed": alpha_seed,
                    "gamma": gamma,
                    "knn": knn,
                    "prop_alpha": prop_alpha,
                    **metric(y_true, pred),
                }
                records.append(rec)
                split_records.append(rec)
        print(split_mode, sorted(split_records, key=lambda r: float(r["macro_f1"]), reverse=True)[0])
    frame = pd.DataFrame(records)
    key_cols = ["variant", "alpha_seed", "gamma", "knn", "prop_alpha"]
    avg = (
        frame.groupby(key_cols)["macro_f1"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        .sort_values(["mean", "min"], ascending=False)
    )
    csv_path = REPORT_DIR / "combo_graph_label_propagation_results.csv"
    frame.to_csv(csv_path, index=False, encoding="utf-8")
    summary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "runtime_seconds": round(time.time() - start, 3),
        "features": features,
        "grid_size": len(grid),
        "top_by_average": json.loads(avg.head(30).to_json(orient="records", force_ascii=False)),
        "result_csv": str(csv_path.relative_to(ROOT)),
        "conclusion": "Graph propagation tests whether unlabeled target combo geometry can resolve ambiguous low-cardinality features.",
    }
    out = REPORT_DIR / "combo_graph_label_propagation_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary["top_by_average"][:10], indent=2, ensure_ascii=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
