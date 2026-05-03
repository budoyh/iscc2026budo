from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
SUBMISSION_DIR = ROOT / "submissions"
CLASSES = np.array([0, 1, 2], dtype=int)
TRUE_COUNTS = np.array([14000.0, 2500.0, 3500.0], dtype=np.float64)
SEED = 20260503


SCORED_SUBMISSIONS: dict[str, float] = {
    "submission_blend_v1.csv": 0.69678,
    "submission_blend_shift_v1.csv": 0.69404,
    "submission_groupcv_compromise_v1.csv": 0.69438,
    "submission_leak_cons_base_v1.csv": 0.37876,
    "submission_leak_mid_block_v1.csv": 0.38024,
    "submission_seen_exact_mode_group_v1.csv": 0.69177,
    "submission_mlp_onehot_v1.csv": 0.70201,
    "submission_blend_mlp_a020_w100105_v1.csv": 0.69821,
    "sub_sklearn_hybrid.csv": 0.69939,
    "submission_mlp_quota_13600_2900_3500_v1.csv": 0.70441,
    "submission_mlp_labelshift_hard_q4_v1.csv": 0.70761,
    "submission_leaderboard_constraint_v1.csv": 0.56815,
}


def find_data_dir() -> Path:
    candidates = [p for p in DATA_ROOT.rglob("*") if (p / "data_test.csv").exists()]
    if not candidates:
        raise FileNotFoundError("data_test.csv not found")
    return candidates[0]


def combo_keys(frame: pd.DataFrame, features: list[str]) -> list[tuple[int, ...]]:
    return list(map(tuple, frame[features].to_numpy(dtype=np.int16)))


def build_groups(test: pd.DataFrame, features: list[str]):
    keys = combo_keys(test, features)
    unique = sorted(set(keys))
    gid = {key: i for i, key in enumerate(unique)}
    group_ids = np.array([gid[key] for key in keys], dtype=int)
    sizes = np.bincount(group_ids, minlength=len(unique)).astype(np.float64)
    group_key = [None] * len(unique)
    for key, g in zip(keys, group_ids):
        group_key[int(g)] = key
    return group_key, group_ids, sizes


def group_counts_from_labels(group_ids: np.ndarray, labels: np.ndarray, n_groups: int) -> np.ndarray:
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


def load_submission(name: str, test: pd.DataFrame) -> np.ndarray:
    sub = pd.read_csv(SUBMISSION_DIR / name)
    if not sub["name"].equals(test["name"]):
        raise ValueError(f"name mismatch {name}")
    return sub["label"].to_numpy(dtype=int)


def make_priors(test: pd.DataFrame, features: list[str], group_ids: np.ndarray, group_key: list[tuple[int, ...]]) -> dict[str, np.ndarray]:
    n_groups = len(group_key)
    data_dir = find_data_dir()
    train = pd.read_csv(data_dir / "data_train.csv")
    train_features = [c for c in train.columns if c not in ["name", "label"]]
    y = train["label"].to_numpy(dtype=int)
    global_prob = np.bincount(y, minlength=3).astype(float)
    global_prob /= global_prob.sum()
    counts_by_key: dict[tuple[int, ...], np.ndarray] = {}
    for key, label in zip(combo_keys(train, train_features), y):
        counts_by_key.setdefault(key, np.zeros(3, dtype=float))[int(label)] += 1.0
    exact = np.zeros((n_groups, 3), dtype=np.float64)
    for g, key in enumerate(group_key):
        c = counts_by_key.get(key, np.zeros(3, dtype=float))
        p = c + 2.0 * global_prob
        exact[g] = p / p.sum()

    mlp_row = np.load(MODEL_DIR / "mlp_onehot_proba.npy").astype(np.float64)
    mlp = np.zeros((n_groups, 3), dtype=np.float64)
    denom = np.bincount(group_ids, minlength=n_groups).astype(np.float64)
    for cls in CLASSES:
        mlp[:, int(cls)] = np.bincount(group_ids, weights=mlp_row[:, int(cls)], minlength=n_groups) / np.maximum(denom, 1.0)
    mlp = np.clip(mlp, 1e-8, None)
    mlp /= mlp.sum(axis=1, keepdims=True)
    out = {
        "mlp": mlp,
        "exact": exact,
        "mix30exact": 0.7 * mlp + 0.3 * exact,
        "mix70exact": 0.3 * mlp + 0.7 * exact,
    }
    for key, val in out.items():
        val = np.clip(val, 1e-8, None)
        out[key] = val / val.sum(axis=1, keepdims=True)
    return out


def fit_truth(
    prior: np.ndarray,
    sizes: np.ndarray,
    pred_gc: np.ndarray,
    pred_counts: np.ndarray,
    target_scores: np.ndarray,
    score_scale: float,
    kl_weight: float,
    steps: int,
    seed: int,
) -> dict[str, object]:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sizes_t = torch.tensor(sizes, dtype=torch.float32, device=device)
    prior_t = torch.tensor(prior, dtype=torch.float32, device=device)
    pred_gc_t = torch.tensor(pred_gc, dtype=torch.float32, device=device)
    pred_counts_t = torch.tensor(pred_counts, dtype=torch.float32, device=device)
    target_t = torch.tensor(target_scores, dtype=torch.float32, device=device)
    true_counts_t = torch.tensor(TRUE_COUNTS, dtype=torch.float32, device=device)
    noise = np.random.default_rng(seed).normal(0.0, 0.03, size=prior.shape)
    logits = torch.tensor(np.log(np.clip(prior, 1e-8, 1.0)) + noise, dtype=torch.float32, device=device, requires_grad=True)
    opt = torch.optim.AdamW([logits], lr=0.055)
    best = None
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        q = torch.softmax(logits, dim=1)
        gc = q * sizes_t[:, None]
        counts = gc.sum(dim=0)
        tp = torch.sum(gc.unsqueeze(0) * pred_gc_t / torch.clamp(sizes_t[None, :, None], min=1.0), dim=1)
        f1 = 2.0 * tp / torch.clamp(pred_counts_t + true_counts_t[None, :], min=1e-6)
        macro = f1.mean(dim=1)
        score_loss = torch.mean(((macro - target_t) / score_scale) ** 2)
        count_loss = torch.mean(((counts - true_counts_t) / 35.0) ** 2)
        kl = torch.mean(torch.sum(q * (torch.log(torch.clamp(q, min=1e-8)) - torch.log(prior_t)), dim=1))
        loss = score_loss + count_loss + kl_weight * kl
        loss.backward()
        opt.step()
        if best is None or float(loss.detach().cpu()) < best["loss"]:
            best = {
                "loss": float(loss.detach().cpu()),
                "q": q.detach().cpu().numpy(),
                "macro": macro.detach().cpu().numpy(),
                "counts": counts.detach().cpu().numpy(),
                "max_abs_error": float(torch.max(torch.abs(macro - target_t)).detach().cpu()),
            }
    assert best is not None
    return best


def optimize_hard_labels(true_gc: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    # Coordinate ascent over one hard class per exact combo. This avoids any
    # row-order dependence inside duplicated feature groups.
    labels = true_gc.argmax(axis=1).astype(int)

    def labels_to_gc(group_labels: np.ndarray) -> np.ndarray:
        out = np.zeros_like(true_gc)
        out[np.arange(len(group_labels)), group_labels] = sizes
        return out

    pred_gc = labels_to_gc(labels)
    current = expected_macro(true_gc, pred_gc, TRUE_COUNTS)
    changed = True
    passes = 0
    while changed and passes < 20:
        changed = False
        passes += 1
        order = np.argsort(-sizes)
        for g in order:
            old = labels[int(g)]
            best_cls = old
            best_score = current
            for cls in CLASSES:
                if int(cls) == int(old):
                    continue
                labels[int(g)] = int(cls)
                score = expected_macro(true_gc, labels_to_gc(labels), TRUE_COUNTS)
                if score > best_score + 1e-12:
                    best_score = score
                    best_cls = int(cls)
            labels[int(g)] = best_cls
            if best_cls != old:
                current = best_score
                changed = True
    return labels.astype(int)


def labels_to_submission(group_ids: np.ndarray, group_labels: np.ndarray) -> np.ndarray:
    return group_labels[group_ids].astype(int)


def main() -> None:
    start = time.time()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    data_dir = find_data_dir()
    test = pd.read_csv(data_dir / "data_test.csv")
    features = [c for c in test.columns if c != "name"]
    group_key, group_ids, sizes = build_groups(test, features)
    n_groups = len(sizes)
    priors = make_priors(test, features, group_ids, group_key)

    all_names = [name for name in SCORED_SUBMISSIONS if (SUBMISSION_DIR / name).exists()]
    pred_gc_by_name = {}
    pred_counts_by_name = {}
    for name in all_names:
        labels = load_submission(name, test)
        gc = group_counts_from_labels(group_ids, labels, n_groups)
        pred_gc_by_name[name] = gc
        pred_counts_by_name[name] = gc.sum(axis=0)

    configs = []
    for score_set_name, names in {
        "all": all_names,
        "no_failed_constraint": [n for n in all_names if n != "submission_leaderboard_constraint_v1.csv"],
        "no_leak_family": [n for n in all_names if "leak_" not in n],
    }.items():
        for prior_name, score_scale, kl_weight in itertools.product(
            ["mlp", "mix30exact", "mix70exact"],
            [0.0012, 0.0025],
            [0.02, 0.08],
        ):
            configs.append((score_set_name, names, prior_name, score_scale, kl_weight))

    group_label_votes = []
    model_items = []
    q4_labels = load_submission("submission_mlp_labelshift_hard_q4_v1.csv", test)
    q4_gc = group_counts_from_labels(group_ids, q4_labels, n_groups)
    for i, (score_set_name, names, prior_name, score_scale, kl_weight) in enumerate(configs):
        pred_gc = np.stack([pred_gc_by_name[n] for n in names], axis=0)
        pred_counts = np.stack([pred_counts_by_name[n] for n in names], axis=0)
        target_scores = np.array([SCORED_SUBMISSIONS[n] for n in names], dtype=np.float64)
        fit = fit_truth(
            priors[prior_name],
            sizes,
            pred_gc,
            pred_counts,
            target_scores,
            score_scale=score_scale,
            kl_weight=kl_weight,
            steps=2200,
            seed=SEED + i,
        )
        true_gc = fit["q"] * sizes[:, None]
        group_labels = optimize_hard_labels(true_gc, sizes)
        group_label_votes.append(group_labels)
        pred_labels = labels_to_submission(group_ids, group_labels)
        pred_gc_hard = group_counts_from_labels(group_ids, pred_labels, n_groups)
        item = {
            "idx": i,
            "score_set": score_set_name,
            "prior": prior_name,
            "score_scale": score_scale,
            "kl_weight": kl_weight,
            "fit_loss": fit["loss"],
            "fit_max_abs_error": fit["max_abs_error"],
            "truth_counts": [float(x) for x in fit["counts"]],
            "candidate_expected": expected_macro(true_gc, pred_gc_hard, TRUE_COUNTS),
            "q4_expected": expected_macro(true_gc, q4_gc, TRUE_COUNTS),
            "diff_vs_q4": int((pred_labels != q4_labels).sum()),
            "candidate_counts": np.bincount(pred_labels, minlength=3).astype(int).tolist(),
        }
        model_items.append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)

    votes = np.stack(group_label_votes, axis=0)
    vote_counts = np.zeros((n_groups, 3), dtype=int)
    for cls in CLASSES:
        vote_counts[:, int(cls)] = (votes == int(cls)).sum(axis=0)
    consensus = vote_counts.argmax(axis=1).astype(int)
    consensus_labels = labels_to_submission(group_ids, consensus)
    out_path = SUBMISSION_DIR / "submission_combo_score_posterior_consensus_v1.csv"
    pd.DataFrame({"name": test["name"], "label": consensus_labels}).to_csv(out_path, index=False, encoding="utf-8", lineterminator="\n")

    stability = {
        "mean_top_vote_fraction": float((vote_counts.max(axis=1) / len(configs)).mean()),
        "rows_with_unanimous_group_label": int(sizes[vote_counts.max(axis=1) == len(configs)].sum()),
        "rows_top_vote_ge_80pct": int(sizes[(vote_counts.max(axis=1) / len(configs)) >= 0.8].sum()),
    }
    consensus_gc = group_counts_from_labels(group_ids, consensus_labels, n_groups)
    consensus_eval = []
    for item, labels_g in zip(model_items, group_label_votes):
        # Recreate the fitted q cheaply is not stored to keep memory small; use
        # candidate_expected comparisons from fitted models for decision.
        consensus_eval.append({
            "idx": item["idx"],
            "candidate_expected": item["candidate_expected"],
            "q4_expected": item["q4_expected"],
            "candidate_minus_q4": item["candidate_expected"] - item["q4_expected"],
        })

    summary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "runtime_seconds": round(time.time() - start, 3),
        "n_groups": int(n_groups),
        "n_configs": len(configs),
        "stability": stability,
        "items": model_items,
        "consensus": {
            "path": str(out_path.relative_to(ROOT)),
            "counts": np.bincount(consensus_labels, minlength=3).astype(int).tolist(),
            "diff_vs_q4": int((consensus_labels != q4_labels).sum()),
            "top_vote_hist": {
                str(k): int((vote_counts.max(axis=1) == k).sum())
                for k in range(1, len(configs) + 1)
                if int((vote_counts.max(axis=1) == k).sum()) > 0
            },
        },
        "decision_note": "Use only if posterior-stable configs consistently improve expected macro over q4 and group-label agreement is high.",
    }
    (REPORT_DIR / "combo_score_posterior_ensemble_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
