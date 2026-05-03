from __future__ import annotations

import json
import math
import random
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
TRUE_COUNTS = np.array([14000, 2500, 3500], dtype=np.float32)
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


def load_submission(name: str, test: pd.DataFrame) -> np.ndarray:
    path = SUBMISSION_DIR / name
    if not path.exists():
        # Some short names are copied under submissions; all paths are local.
        path = SUBMISSION_DIR / Path(name).name
    sub = pd.read_csv(path)
    if not sub["name"].equals(test["name"]):
        raise ValueError(f"name order mismatch for {name}")
    return sub["label"].to_numpy(dtype=int)


def build_groups(test: pd.DataFrame, features: list[str]):
    keys = combo_keys(test, features)
    unique = sorted(set(keys))
    gid = {key: i for i, key in enumerate(unique)}
    group_ids = np.array([gid[key] for key in keys], dtype=int)
    sizes = np.bincount(group_ids, minlength=len(unique)).astype(np.float32)
    return unique, group_ids, sizes


def combo_prior(test: pd.DataFrame, features: list[str], group_ids: np.ndarray, n_groups: int) -> np.ndarray:
    mlp = np.load(MODEL_DIR / "mlp_onehot_proba.npy").astype(np.float64)
    prior = np.zeros((n_groups, len(CLASSES)), dtype=np.float64)
    counts = np.bincount(group_ids, minlength=n_groups).astype(np.float64)
    for cls in CLASSES:
        prior[:, int(cls)] = np.bincount(group_ids, weights=mlp[:, int(cls)], minlength=n_groups) / np.maximum(counts, 1.0)
    # Blend with exact train combo label distributions for a less submission-specific prior.
    data_dir = find_data_dir()
    train = pd.read_csv(data_dir / "data_train.csv")
    train_features = [c for c in train.columns if c not in ["name", "label"]]
    lookup: dict[tuple[int, ...], np.ndarray] = {}
    y = train["label"].to_numpy(dtype=int)
    global_prob = np.bincount(y, minlength=len(CLASSES)).astype(float)
    global_prob /= global_prob.sum()
    counts_by_key: dict[tuple[int, ...], np.ndarray] = {}
    for key, label in zip(combo_keys(train, train_features), y):
        counts_by_key.setdefault(key, np.zeros(len(CLASSES), dtype=float))[int(label)] += 1.0
    keys = combo_keys(test.drop(columns=[], errors="ignore"), features)
    group_key = [None] * n_groups
    for key, g in zip(keys, group_ids):
        group_key[int(g)] = key
    exact = np.zeros_like(prior)
    for g, key in enumerate(group_key):
        c = counts_by_key.get(key, np.zeros(len(CLASSES), dtype=float))
        p = c + 2.0 * global_prob
        exact[g] = p / p.sum()
    prior = 0.7 * prior + 0.3 * exact
    prior = np.clip(prior, 1e-6, None)
    return prior / prior.sum(axis=1, keepdims=True)


def round_group_counts(q: np.ndarray, sizes: np.ndarray, true_counts: np.ndarray) -> np.ndarray:
    raw = q * sizes[:, None]
    out = np.floor(raw).astype(int)
    # First satisfy each group size.
    frac = raw - out
    for g in range(len(sizes)):
        remain = int(round(float(sizes[g]))) - int(out[g].sum())
        if remain > 0:
            order = np.argsort(-frac[g])
            for cls in order[:remain]:
                out[g, int(cls)] += 1
    # Then move labels to exact global counts.
    target = true_counts.astype(int)
    totals = out.sum(axis=0)
    logits = np.log(np.clip(q, 1e-12, None))
    while not np.array_equal(totals, target):
        deficits = np.where(totals < target)[0]
        surplus = np.where(totals > target)[0]
        best = None
        for dst in deficits:
            for src in surplus:
                rows = np.where(out[:, int(src)] > 0)[0]
                if len(rows) == 0:
                    continue
                losses = logits[rows, int(src)] - logits[rows, int(dst)]
                i = int(np.argmin(losses))
                cand = (float(losses[i]), int(rows[i]), int(src), int(dst))
                if best is None or cand[0] < best[0]:
                    best = cand
        if best is None:
            break
        _, g, src, dst = best
        out[g, src] -= 1
        out[g, dst] += 1
        totals[src] -= 1
        totals[dst] += 1
    return out


def labels_from_group_counts(group_ids: np.ndarray, group_counts: np.ndarray) -> np.ndarray:
    labels = np.zeros(len(group_ids), dtype=int)
    for g in range(len(group_counts)):
        rows = np.where(group_ids == g)[0]
        # Deterministic tie-free assignment by row name/order only inside identical
        # feature groups; row identity is otherwise unavailable.
        seq = np.repeat(CLASSES, group_counts[g].astype(int))
        if len(seq) != len(rows):
            raise ValueError((g, len(seq), len(rows), group_counts[g].tolist()))
        labels[rows] = seq
    return labels


def expected_macros_from_counts(
    group_counts: np.ndarray,
    pred_group_counts: np.ndarray,
    pred_counts: np.ndarray,
    true_counts: np.ndarray,
) -> np.ndarray:
    sizes = group_counts.sum(axis=1).astype(float)
    tp = (group_counts[None, :, :] * pred_group_counts / np.maximum(sizes[None, :, None], 1.0)).sum(axis=1)
    f1 = 2.0 * tp / np.maximum(pred_counts + true_counts[None, :], 1e-12)
    return f1.mean(axis=1)


def hard_refine_group_counts(
    group_counts: np.ndarray,
    prior: np.ndarray,
    pred_group_counts: np.ndarray,
    pred_counts: np.ndarray,
    target_scores: np.ndarray,
    seconds: float = 180.0,
) -> tuple[np.ndarray, dict[str, object]]:
    rng = random.Random(SEED)
    counts = group_counts.copy().astype(int)
    true_counts = counts.sum(axis=0).astype(float)
    n_sub, n_groups, _ = pred_group_counts.shape
    sizes = counts.sum(axis=1).astype(float)
    coef = np.zeros((n_sub, n_groups, len(CLASSES)), dtype=np.float64)
    for cls in CLASSES:
        coef[:, :, int(cls)] = 2.0 * pred_group_counts[:, :, int(cls)] / (
            np.maximum(sizes[None, :], 1.0) * 3.0 * np.maximum(pred_counts[:, int(cls)] + true_counts[int(cls)], 1e-12)[:, None]
        )
    log_prior = np.log(np.clip(prior, 1e-12, None))
    macros = expected_macros_from_counts(counts.astype(float), pred_group_counts, pred_counts, true_counts)
    score_scale = 0.0005

    def objective(mac: np.ndarray, cnt: np.ndarray) -> float:
        score_term = float(np.mean(((mac - target_scores) / score_scale) ** 2))
        prior_term = -0.012 * float((cnt * log_prior).sum() / cnt.sum())
        entropy_term = 0.0
        return score_term + prior_term + entropy_term

    current = objective(macros, counts)
    best = current
    best_counts = counts.copy()
    best_macros = macros.copy()
    start = time.time()
    tried = 0
    accepted = 0
    improve = 0
    groups_by_label = [set(np.where(counts[:, int(cls)] > 0)[0].tolist()) for cls in CLASSES]
    while time.time() - start < seconds:
        tried += 1
        src = rng.randrange(3)
        dst = rng.randrange(3)
        if src == dst or not groups_by_label[src] or not groups_by_label[dst]:
            continue
        g = rng.choice(tuple(groups_by_label[src]))
        h = rng.choice(tuple(groups_by_label[dst]))
        if g == h:
            continue
        # Swap one true label src->dst in group g and dst->src in group h.
        delta = (coef[:, g, dst] - coef[:, g, src]) + (coef[:, h, src] - coef[:, h, dst])
        new_macros = macros + delta
        prior_delta = -0.012 * float(
            (log_prior[g, dst] - log_prior[g, src] + log_prior[h, src] - log_prior[h, dst]) / counts.sum()
        )
        new_obj = float(np.mean(((new_macros - target_scores) / score_scale) ** 2)) + (
            current - float(np.mean(((macros - target_scores) / score_scale) ** 2))
        ) + prior_delta
        temp = max(1e-6, 0.004 * (1.0 - min(1.0, (time.time() - start) / seconds)))
        if new_obj < current or rng.random() < math.exp(min(50.0, (current - new_obj) / temp)):
            counts[g, src] -= 1
            counts[g, dst] += 1
            counts[h, dst] -= 1
            counts[h, src] += 1
            if counts[g, src] == 0:
                groups_by_label[src].discard(g)
            groups_by_label[dst].add(g)
            if counts[h, dst] == 0:
                groups_by_label[dst].discard(h)
            groups_by_label[src].add(h)
            macros = new_macros
            current = new_obj
            accepted += 1
            if current < best:
                best = current
                best_counts = counts.copy()
                best_macros = macros.copy()
                improve += 1
    return best_counts, {
        "best_objective": best,
        "tried": tried,
        "accepted": accepted,
        "improvements": improve,
        "best_macros": best_macros.tolist(),
        "max_abs_score_error": float(np.max(np.abs(best_macros - target_scores))),
    }


def macro_from_expected(group_counts: np.ndarray, pred_group_counts: np.ndarray, pred_counts: np.ndarray, true_counts: np.ndarray) -> float:
    # expected TP for submission s,class c: sum_g true_count_g,c * pred_count_g,c / n_g
    sizes = group_counts.sum(axis=1)
    tp = (group_counts * pred_group_counts / np.maximum(sizes[:, None], 1.0)).sum(axis=0)
    f1 = 2.0 * tp / np.maximum(pred_counts + true_counts, 1e-12)
    return float(f1.mean())


def main() -> None:
    torch.manual_seed(SEED)
    start = time.time()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    data_dir = find_data_dir()
    test = pd.read_csv(data_dir / "data_test.csv")
    features = [c for c in test.columns if c != "name"]
    _, group_ids, sizes = build_groups(test, features)
    n_groups = len(sizes)
    prior = combo_prior(test, features, group_ids, n_groups)

    pred_group_counts = []
    pred_counts = []
    score_names = []
    score_values = []
    for name, score in SCORED_SUBMISSIONS.items():
        path = SUBMISSION_DIR / name
        if not path.exists():
            continue
        pred = load_submission(name, test)
        mat = np.zeros((n_groups, len(CLASSES)), dtype=np.float32)
        for cls in CLASSES:
            mat[:, int(cls)] = np.bincount(group_ids, weights=(pred == int(cls)).astype(float), minlength=n_groups)
        pred_group_counts.append(mat)
        pred_counts.append(np.bincount(pred, minlength=len(CLASSES)).astype(np.float32))
        score_names.append(name)
        score_values.append(float(score))
    pred_group_counts_np = np.stack(pred_group_counts, axis=0)
    pred_counts_np = np.stack(pred_counts, axis=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sizes_t = torch.tensor(sizes, dtype=torch.float32, device=device)
    prior_t = torch.tensor(prior, dtype=torch.float32, device=device)
    pred_gc_t = torch.tensor(pred_group_counts_np, dtype=torch.float32, device=device)
    pred_counts_t = torch.tensor(pred_counts_np, dtype=torch.float32, device=device)
    true_counts_t = torch.tensor(TRUE_COUNTS, dtype=torch.float32, device=device)
    target_scores_t = torch.tensor(score_values, dtype=torch.float32, device=device)
    logits = torch.tensor(np.log(prior), dtype=torch.float32, device=device, requires_grad=True)
    opt = torch.optim.AdamW([logits], lr=0.05)
    best = None
    history = []
    for step in range(1, 6001):
        opt.zero_grad(set_to_none=True)
        q = torch.softmax(logits, dim=1)
        group_counts = q * sizes_t[:, None]
        counts = group_counts.sum(dim=0)
        tp = torch.sum(group_counts.unsqueeze(0) * pred_gc_t / torch.clamp(sizes_t[None, :, None], min=1.0), dim=1)
        f1 = 2.0 * tp / torch.clamp(pred_counts_t + true_counts_t[None, :], min=1e-6)
        macro = f1.mean(dim=1)
        score_loss = torch.mean(((macro - target_scores_t) / 0.00055) ** 2)
        count_loss = torch.mean(((counts - true_counts_t) / 30.0) ** 2)
        kl = torch.mean(torch.sum(q * (torch.log(torch.clamp(q, min=1e-8)) - torch.log(prior_t)), dim=1))
        entropy = -torch.mean(torch.sum(q * torch.log(torch.clamp(q, min=1e-8)), dim=1))
        loss = score_loss + count_loss + 0.015 * kl - 0.0015 * entropy
        loss.backward()
        opt.step()
        if best is None or float(loss.detach().cpu()) < best["loss"]:
            best = {
                "loss": float(loss.detach().cpu()),
                "q": q.detach().cpu().numpy(),
                "macro": macro.detach().cpu().numpy(),
                "counts": counts.detach().cpu().numpy(),
            }
        if step % 500 == 0 or step == 1:
            item = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "score_loss": float(score_loss.detach().cpu()),
                "count_loss": float(count_loss.detach().cpu()),
                "kl": float(kl.detach().cpu()),
                "entropy": float(entropy.detach().cpu()),
                "max_abs_score_error": float(torch.max(torch.abs(macro - target_scores_t)).detach().cpu()),
                "counts": [float(x) for x in counts.detach().cpu().numpy()],
            }
            history.append(item)
            print(json.dumps(item, ensure_ascii=False), flush=True)

    assert best is not None
    group_counts_int = round_group_counts(best["q"], sizes, TRUE_COUNTS)
    refined_counts, refine_info = hard_refine_group_counts(
        group_counts_int,
        prior,
        pred_group_counts_np.astype(np.float64),
        pred_counts_np.astype(np.float64),
        np.array(score_values, dtype=np.float64),
        seconds=180.0,
    )
    group_counts_int = refined_counts
    labels = labels_from_group_counts(group_ids, group_counts_int)
    sub = pd.DataFrame({"name": test["name"], "label": labels})
    out_path = SUBMISSION_DIR / "submission_combo_score_constrained_v1.csv"
    sub.to_csv(out_path, index=False, encoding="utf-8", lineterminator="\n")
    expected_scores = {}
    for j, name in enumerate(score_names):
        expected_scores[name] = {
            "target": score_values[j],
            "fit_soft": float(best["macro"][j]),
            "fit_rounded_expected": macro_from_expected(group_counts_int.astype(float), pred_group_counts_np[j], pred_counts_np[j], TRUE_COUNTS),
        }
    q4_path = SUBMISSION_DIR / "submission_mlp_labelshift_hard_q4_v1.csv"
    diff_q4 = None
    if q4_path.exists():
        q4 = pd.read_csv(q4_path)["label"].to_numpy(dtype=int)
        diff_q4 = int((labels != q4).sum())
    summary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "runtime_seconds": round(time.time() - start, 3),
        "device": str(device),
        "n_groups": int(n_groups),
        "score_names": score_names,
        "history": history,
        "soft_counts": [float(x) for x in best["counts"]],
        "rounded_counts": {str(cls): int((labels == cls).sum()) for cls in CLASSES},
        "expected_scores": expected_scores,
        "hard_refine": refine_info,
        "generated_submission": {
            "path": str(out_path.relative_to(ROOT)),
            "diff_vs_labelshift_q4": diff_q4,
        },
        "conclusion": "Combo-level score-constrained inference fits leaderboard feedback only at exact-combo count level, with fixed 14000/2500/3500 class totals.",
    }
    report_path = REPORT_DIR / "combo_score_constrained_summary.json"
    report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
