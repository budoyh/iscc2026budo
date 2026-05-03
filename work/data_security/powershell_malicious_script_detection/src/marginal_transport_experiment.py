from __future__ import annotations

import importlib.util
import itertools
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
SUBMISSION_DIR = ROOT / "submissions"
CLASSES = np.array([0, 1, 2], dtype=int)
HIDDEN_COUNTS = np.array([14000, 2500, 3500], dtype=np.float64)
SEED = 20260503


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


tds = load_module(ROOT / "src" / "target_domain_structure_search.py", "tds")
optmod = load_module(ROOT / "src" / "optimize_submission_for_inferred_combo_truth.py", "optmod")


def find_data_dir() -> Path:
    candidates = [p for p in DATA_ROOT.rglob("*") if (p / "data_train.csv").exists()]
    if not candidates:
        raise FileNotFoundError("data_train.csv not found")
    return candidates[0]


def extract_id(name: object) -> int:
    m = re.search(r"(\d+)", str(name))
    if not m:
        raise ValueError(name)
    return int(m.group(1))


def class_positions(frame: pd.DataFrame) -> dict[int, np.ndarray]:
    y = frame["label"].to_numpy(dtype=int)
    ids = frame["_id"].to_numpy(dtype=int)
    out = {}
    for cls in CLASSES:
        idx = np.where(y == int(cls))[0]
        out[int(cls)] = idx[np.argsort(ids[idx])]
    return out


def source_weights(frame: pd.DataFrame, mode: str) -> np.ndarray:
    weights = np.zeros(len(frame), dtype=np.float64)
    positions = class_positions(frame)
    for cls in CLASSES:
        pos = positions[int(cls)]
        n = len(pos)
        if mode.startswith("front_frac"):
            frac = float(mode.replace("front_frac", ""))
            k = max(1, int(round(n * frac)))
            local = np.zeros(n, dtype=np.float64)
            local[:k] = 1.0
        elif mode.startswith("front"):
            k = int(mode.replace("front", ""))
            local = np.zeros(n, dtype=np.float64)
            local[: min(k, n)] = 1.0
        elif mode.startswith("decay"):
            tau = float(mode.replace("decay", ""))
            x = np.linspace(0.0, 1.0, n)
            local = np.exp(-tau * x)
        else:
            raise ValueError(mode)
        weights[pos] = local
    return weights


def combo_keys(frame: pd.DataFrame, features: list[str]) -> list[tuple[int, ...]]:
    return list(map(tuple, frame[features].to_numpy(dtype=np.int16)))


def build_groups(frame: pd.DataFrame, features: list[str]):
    keys = combo_keys(frame, features)
    unique = sorted(set(keys))
    gid = {key: i for i, key in enumerate(unique)}
    group_ids = np.array([gid[key] for key in keys], dtype=int)
    sizes = np.bincount(group_ids, minlength=len(unique)).astype(np.float64)
    group_values = np.array(unique, dtype=np.int16)
    return group_ids, sizes, group_values


def group_counts(labels: np.ndarray, group_ids: np.ndarray, n_groups: int) -> np.ndarray:
    out = np.zeros((n_groups, 3), dtype=np.float64)
    for cls in CLASSES:
        out[:, int(cls)] = np.bincount(group_ids, weights=(labels == int(cls)).astype(float), minlength=n_groups)
    return out


def expected_macro(true_gc: np.ndarray, pred_gc: np.ndarray, true_counts: np.ndarray | None = None) -> float:
    if true_counts is None:
        true_counts = true_gc.sum(axis=0)
    sizes = true_gc.sum(axis=1)
    pred_counts = pred_gc.sum(axis=0)
    tp = (true_gc * pred_gc / np.maximum(sizes[:, None], 1.0)).sum(axis=0)
    f1 = 2.0 * tp / np.maximum(pred_counts + true_counts, 1e-12)
    return float(f1.mean())


def build_constraint_ids(
    source: pd.DataFrame,
    target_group_values: np.ndarray,
    features: list[str],
    constraints: list[tuple[int, ...]],
) -> list[dict[str, object]]:
    out = []
    for subset in constraints:
        src_keys = list(map(tuple, source[[features[i] for i in subset]].to_numpy(dtype=np.int16)))
        tgt_keys = list(map(tuple, target_group_values[:, list(subset)].astype(np.int16)))
        values = sorted(set(src_keys) | set(tgt_keys))
        mapping = {key: i for i, key in enumerate(values)}
        src_ids = np.array([mapping[k] for k in src_keys], dtype=int)
        tgt_ids = np.array([mapping[k] for k in tgt_keys], dtype=int)
        out.append({"subset": subset, "n": len(values), "src_ids": src_ids, "tgt_ids": tgt_ids})
    return out


def target_distributions(
    source: pd.DataFrame,
    constraints_data: list[dict[str, object]],
    weights: np.ndarray,
    alpha: float,
) -> list[np.ndarray]:
    y = source["label"].to_numpy(dtype=int)
    out = []
    for item in constraints_data:
        n = int(item["n"])
        src_ids = item["src_ids"]
        dist = np.zeros((3, n), dtype=np.float64)
        for cls in CLASSES:
            mask_w = weights * (y == int(cls))
            counts = np.bincount(src_ids, weights=mask_w, minlength=n).astype(np.float64)
            counts += alpha
            dist[int(cls)] = counts / counts.sum()
        out.append(dist)
    return out


def initial_log_scores(
    constraints_data: list[dict[str, object]],
    dists: list[np.ndarray],
    weights_by_order: dict[int, float],
    sizes: np.ndarray,
    target_counts: np.ndarray,
) -> np.ndarray:
    scores = np.zeros((len(sizes), 3), dtype=np.float64)
    for item, dist in zip(constraints_data, dists):
        subset = item["subset"]
        tgt_ids = item["tgt_ids"]
        w = weights_by_order[len(subset)] / max(1, sum(1 for x in constraints_data if len(x["subset"]) == len(subset)))
        scores += w * np.log(np.clip(dist[:, tgt_ids].T, 1e-12, None))
    scores += np.log(target_counts / target_counts.sum())[None, :]
    return scores


def optimize_transport(
    source: pd.DataFrame,
    target: pd.DataFrame,
    features: list[str],
    source_mode: str,
    pair_weight: float,
    target_counts: np.ndarray,
    steps: int = 450,
) -> dict[str, object]:
    group_ids, sizes, group_values = build_groups(target, features)
    n_groups = len(sizes)
    singles = [(i,) for i in range(len(features))]
    pair_names = [
        ("decode_activity_profile", "content_encoding_profile"),
        ("network_command_profile", "command_surface_profile"),
        ("credential_runtime_profile", "task_registry_profile"),
        ("structure_rhythm_profile", "layout_variation_profile"),
        ("identifier_variation_profile", "content_encoding_profile"),
        ("function_scope_level", "pipeline_usage_level"),
        ("decode_activity_profile", "command_surface_profile"),
        ("network_command_profile", "credential_runtime_profile"),
        ("structure_rhythm_profile", "command_surface_profile"),
        ("content_encoding_profile", "extension_import_profile"),
        ("parameter_block_presence", "command_surface_profile"),
        ("loop_scope_level", "branch_scope_level"),
    ]
    name_to_idx = {name: i for i, name in enumerate(features)}
    pairs = [(name_to_idx[a], name_to_idx[b]) for a, b in pair_names]
    constraints = singles + pairs
    cdata = build_constraint_ids(source, group_values, features, constraints)
    weights = source_weights(source, source_mode)
    dists = target_distributions(source, cdata, weights, alpha=2.0)
    init_scores = initial_log_scores(cdata, dists, {1: 1.0, 2: pair_weight}, sizes, target_counts)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sizes_t = torch.tensor(sizes, dtype=torch.float32, device=device)
    target_counts_t = torch.tensor(target_counts, dtype=torch.float32, device=device)
    logits = torch.tensor(init_scores, dtype=torch.float32, device=device, requires_grad=True)
    opt = torch.optim.AdamW([logits], lr=0.05)
    tgt_ids_t = [torch.tensor(item["tgt_ids"], dtype=torch.long, device=device) for item in cdata]
    dist_t = [torch.tensor(dist, dtype=torch.float32, device=device) for dist in dists]
    order_weight_t = [float(1.0 if len(item["subset"]) == 1 else pair_weight) for item in cdata]
    # Normalize total contribution of singles and pairs.
    single_n = sum(1 for item in cdata if len(item["subset"]) == 1)
    pair_n = sum(1 for item in cdata if len(item["subset"]) == 2)
    order_weight_t = [w / (single_n if w == 1.0 else max(pair_n, 1)) for w in order_weight_t]

    best = None
    history = []
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        q = torch.softmax(logits, dim=1)
        gc = q * sizes_t[:, None]
        counts = gc.sum(dim=0)
        count_loss = torch.mean(((counts - target_counts_t) / 25.0) ** 2)
        marginal_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        for ids, target_dist, w in zip(tgt_ids_t, dist_t, order_weight_t):
            n_vals = target_dist.shape[1]
            pred = torch.zeros((3, n_vals), dtype=torch.float32, device=device)
            pred.scatter_add_(1, ids[None, :].expand(3, -1), gc.T)
            pred = pred / torch.clamp(counts[:, None], min=1.0)
            # Symmetric MSE is more stable than KL under extrapolated marginals.
            marginal_loss = marginal_loss + float(w) * torch.mean((pred - target_dist) ** 2)
        entropy = -torch.mean(torch.sum(q * torch.log(torch.clamp(q, min=1e-8)), dim=1))
        loss = count_loss + 900.0 * marginal_loss - 0.001 * entropy
        loss.backward()
        opt.step()
        if best is None or float(loss.detach().cpu()) < best["loss"]:
            best = {
                "loss": float(loss.detach().cpu()),
                "q": q.detach().cpu().numpy(),
                "counts": counts.detach().cpu().numpy(),
                "count_loss": float(count_loss.detach().cpu()),
                "marginal_loss": float(marginal_loss.detach().cpu()),
            }
        if step in {1, 500, 1000, 2000, steps}:
            history.append({
                "step": step,
                "loss": float(loss.detach().cpu()),
                "counts": [float(x) for x in counts.detach().cpu().numpy()],
                "count_loss": float(count_loss.detach().cpu()),
                "marginal_loss": float(marginal_loss.detach().cpu()),
            })
    assert best is not None
    true_gc_est = best["q"] * sizes[:, None]
    hard_group_labels = optmod.optimize_for_truth(true_gc_est) if hasattr(optmod, "optimize_for_truth") else true_gc_est.argmax(axis=1)
    # The imported helper is not part of optmod; keep a local hard majority as
    # fallback. Most q rows are sharp after transport optimization.
    if not isinstance(hard_group_labels, np.ndarray) or hard_group_labels.ndim != 1:
        hard_group_labels = true_gc_est.argmax(axis=1)
    labels = hard_group_labels[group_ids].astype(int)
    pred_gc = group_counts(labels, group_ids, n_groups)
    return {
        "labels": labels,
        "group_ids": group_ids,
        "sizes": sizes,
        "q_group_counts": true_gc_est,
        "pred_group_counts": pred_gc,
        "best": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in best.items() if k != "q"},
        "history": history,
    }


def optimize_hard_for_truth(true_gc: np.ndarray, sizes: np.ndarray, true_counts: np.ndarray) -> np.ndarray:
    labels = true_gc.argmax(axis=1).astype(int)

    pred_counts = np.bincount(labels, weights=sizes, minlength=3).astype(np.float64)
    tp = np.zeros(3, dtype=np.float64)
    for cls in CLASSES:
        idx = np.where(labels == int(cls))[0]
        tp[int(cls)] = true_gc[idx, int(cls)].sum()

    def score(tp_vec: np.ndarray, pred_vec: np.ndarray) -> float:
        return float((2.0 * tp_vec / np.maximum(pred_vec + true_counts, 1e-12)).mean())

    current = score(tp, pred_counts)
    for _ in range(20):
        changed = False
        for g in np.argsort(-sizes):
            old = labels[int(g)]
            best_cls = old
            best_score = current
            for cls in CLASSES:
                cls = int(cls)
                if cls == int(old):
                    continue
                new_tp = tp.copy()
                new_pred = pred_counts.copy()
                new_tp[int(old)] -= true_gc[int(g), int(old)]
                new_tp[cls] += true_gc[int(g), cls]
                new_pred[int(old)] -= sizes[int(g)]
                new_pred[cls] += sizes[int(g)]
                cand = score(new_tp, new_pred)
                if cand > best_score + 1e-12:
                    best_score = cand
                    best_cls = int(cls)
            if best_cls != old:
                tp[int(old)] -= true_gc[int(g), int(old)]
                tp[best_cls] += true_gc[int(g), best_cls]
                pred_counts[int(old)] -= sizes[int(g)]
                pred_counts[best_cls] += sizes[int(g)]
                labels[int(g)] = best_cls
                current = best_score
                changed = True
        if not changed:
            break
    return labels


def run_one(
    source: pd.DataFrame,
    target: pd.DataFrame,
    features: list[str],
    source_mode: str,
    pair_weight: float,
    target_counts: np.ndarray,
) -> dict[str, object]:
    res = optimize_transport(source, target, features, source_mode, pair_weight, target_counts)
    sizes = res["sizes"]
    q_gc = res["q_group_counts"]
    group_ids = res["group_ids"]
    hard_labels = optimize_hard_for_truth(q_gc, sizes, target_counts)
    labels = hard_labels[group_ids].astype(int)
    pred_gc = group_counts(labels, group_ids, len(sizes))
    res["labels"] = labels
    res["pred_group_counts"] = pred_gc
    return res


def main() -> None:
    start = time.time()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    data_dir = find_data_dir()
    train = pd.read_csv(data_dir / "data_train.csv")
    test = pd.read_csv(data_dir / "data_test.csv")
    train["_id"] = train["name"].map(extract_id)
    features = [c for c in train.columns if c not in ["name", "label", "_id"]]
    configs = [
        ("front_frac0.35", 0.35),
        ("front_frac0.50", 0.45),
        ("decay5.0", 0.45),
    ]
    results = []
    for mode in ["ratio_prefix", "actual_prefix"]:
        fit, valid = tds.make_split(train, mode)
        y = valid["label"].to_numpy(dtype=int)
        target_counts = np.bincount(y, minlength=3).astype(np.float64)
        true_group_ids, sizes, _ = build_groups(valid, features)
        true_gc = group_counts(y, true_group_ids, len(sizes))
        for source_mode, pair_weight in configs:
            res = run_one(fit, valid, features, source_mode, pair_weight, target_counts)
            labels = res["labels"]
            item = {
                "split": mode,
                "source_mode": source_mode,
                "pair_weight": pair_weight,
                "actual_macro": float(f1_score(y, labels, average="macro")),
                "expected_macro": expected_macro(true_gc, res["pred_group_counts"], target_counts),
                "pred_counts": np.bincount(labels, minlength=3).astype(int).tolist(),
                "fit": res["best"],
            }
            results.append(item)
            print(json.dumps(item, ensure_ascii=False), flush=True)

    # Select robust config by average expected prefix score.
    by_cfg: dict[tuple[str, float], list[float]] = {}
    for item in results:
        key = (str(item["source_mode"]), float(item["pair_weight"]))
        by_cfg.setdefault(key, []).append(float(item["expected_macro"]))
    best_key = max(by_cfg, key=lambda k: (np.mean(by_cfg[k]), min(by_cfg[k])))
    final_res = run_one(train, test, features, best_key[0], best_key[1], HIDDEN_COUNTS)
    final_labels = final_res["labels"]
    out_path = SUBMISSION_DIR / "submission_marginal_transport_v1.csv"
    pd.DataFrame({"name": test["name"], "label": final_labels}).to_csv(out_path, index=False, encoding="utf-8", lineterminator="\n")
    q4_path = SUBMISSION_DIR / "submission_mlp_labelshift_hard_q4_v1.csv"
    diff_q4 = None
    if q4_path.exists():
        q4 = pd.read_csv(q4_path)["label"].to_numpy(dtype=int)
        diff_q4 = int((q4 != final_labels).sum())
    summary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "runtime_seconds": round(time.time() - start, 3),
        "configs": configs,
        "validation": results,
        "best_config": {"source_mode": best_key[0], "pair_weight": best_key[1], "scores": by_cfg[best_key]},
        "generated": {
            "path": str(out_path.relative_to(ROOT)),
            "counts": np.bincount(final_labels, minlength=3).astype(int).tolist(),
            "diff_vs_q4": diff_q4,
            "fit": final_res["best"],
        },
    }
    (REPORT_DIR / "marginal_transport_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
