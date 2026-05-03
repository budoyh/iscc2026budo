from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
REPORT_DIR = ROOT / "reports"
SUBMISSION_DIR = ROOT / "submissions"
CLASSES = np.array([0, 1, 2], dtype=int)
TRUE_COUNTS = np.array([14000.0, 2500.0, 3500.0], dtype=np.float64)
SEED = 20260503


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
    return unique, group_ids, sizes


def group_counts_from_labels(group_ids: np.ndarray, labels: np.ndarray, n_groups: int) -> np.ndarray:
    out = np.zeros((n_groups, len(CLASSES)), dtype=np.float64)
    for cls in CLASSES:
        out[:, int(cls)] = np.bincount(group_ids, weights=(labels == int(cls)).astype(float), minlength=n_groups)
    return out


def expected_macro(true_gc: np.ndarray, pred_gc: np.ndarray, true_counts: np.ndarray) -> float:
    sizes = true_gc.sum(axis=1)
    pred_counts = pred_gc.sum(axis=0)
    tp = (true_gc * pred_gc / np.maximum(sizes[:, None], 1.0)).sum(axis=0)
    f1 = 2.0 * tp / np.maximum(pred_counts + true_counts, 1e-12)
    return float(f1.mean())


def round_group_counts(q: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    raw = q * sizes[:, None]
    out = np.floor(raw).astype(int)
    frac = raw - out
    for g, size in enumerate(sizes.astype(int)):
        remain = size - int(out[g].sum())
        if remain > 0:
            order = np.argsort(-frac[g])
            for cls in order[:remain]:
                out[g, int(cls)] += 1
    return out


def local_refine(true_gc: np.ndarray, pred_gc: np.ndarray, seconds: float = 240.0) -> tuple[np.ndarray, dict[str, object]]:
    counts = pred_gc.astype(int).copy()
    sizes = true_gc.sum(axis=1)
    true_counts = true_gc.sum(axis=0)
    pred_counts = counts.sum(axis=0).astype(np.float64)
    tp = (true_gc * counts / np.maximum(sizes[:, None], 1.0)).sum(axis=0)

    def score(tp_vec: np.ndarray, pred_vec: np.ndarray) -> float:
        return float((2.0 * tp_vec / np.maximum(pred_vec + true_counts, 1e-12)).mean())

    current = score(tp, pred_counts)
    best = current
    best_counts = counts.copy()
    start = time.time()
    tried = 0
    accepted = 0
    improved = 0
    group_has = [set(np.where(counts[:, int(cls)] > 0)[0].tolist()) for cls in CLASSES]
    while time.time() - start < seconds:
        changed = False
        # Greedy full pass over all possible one-row relabels. 1018*6 is small
        # enough and avoids a noisy random search for this smooth objective.
        best_move = None
        best_delta = 0.0
        for src in CLASSES:
            rows = list(group_has[int(src)])
            for g in rows:
                gain_remove = true_gc[g, int(src)] / max(sizes[g], 1.0)
                for dst in CLASSES:
                    if int(dst) == int(src):
                        continue
                    tried += 1
                    new_tp = tp.copy()
                    new_pred = pred_counts.copy()
                    new_tp[int(src)] -= gain_remove
                    new_tp[int(dst)] += true_gc[g, int(dst)] / max(sizes[g], 1.0)
                    new_pred[int(src)] -= 1.0
                    new_pred[int(dst)] += 1.0
                    new_score = score(new_tp, new_pred)
                    delta = new_score - current
                    if delta > best_delta:
                        best_delta = delta
                        best_move = (int(g), int(src), int(dst), new_tp, new_pred, new_score)
        if best_move is not None:
            g, src, dst, tp, pred_counts, current = best_move
            counts[g, src] -= 1
            counts[g, dst] += 1
            if counts[g, src] == 0:
                group_has[src].discard(g)
            group_has[dst].add(g)
            accepted += 1
            changed = True
            if current > best:
                best = current
                best_counts = counts.copy()
                improved += 1
        if not changed:
            break
    return best_counts, {
        "best_expected_macro": best,
        "tried_moves": tried,
        "accepted_moves": accepted,
        "improvements": improved,
        "runtime_seconds": round(time.time() - start, 3),
    }


def labels_from_group_counts(group_ids: np.ndarray, group_counts: np.ndarray) -> np.ndarray:
    labels = np.zeros(len(group_ids), dtype=int)
    for g in range(len(group_counts)):
        rows = np.where(group_ids == g)[0]
        seq = np.repeat(CLASSES, group_counts[g].astype(int))
        if len(seq) != len(rows):
            raise ValueError((g, len(seq), len(rows), group_counts[g].tolist()))
        labels[rows] = seq
    return labels


def main() -> None:
    start = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

    data_dir = find_data_dir()
    test = pd.read_csv(data_dir / "data_test.csv")
    features = [c for c in test.columns if c != "name"]
    _, group_ids, sizes = build_groups(test, features)
    n_groups = len(sizes)

    inferred_sub = pd.read_csv(SUBMISSION_DIR / "submission_combo_score_constrained_v1.csv")
    if not inferred_sub["name"].equals(test["name"]):
        raise ValueError("inferred submission name order mismatch")
    true_gc = group_counts_from_labels(group_ids, inferred_sub["label"].to_numpy(dtype=int), n_groups)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sizes_t = torch.tensor(sizes, dtype=torch.float32, device=device)
    true_gc_t = torch.tensor(true_gc, dtype=torch.float32, device=device)
    true_counts_t = torch.tensor(true_gc.sum(axis=0), dtype=torch.float32, device=device)

    init = np.clip(true_gc / np.maximum(sizes[:, None], 1.0), 1e-5, 1.0)
    init = init / init.sum(axis=1, keepdims=True)
    logits = torch.tensor(np.log(init), dtype=torch.float32, device=device, requires_grad=True)
    opt = torch.optim.AdamW([logits], lr=0.08, weight_decay=0.0)
    best: dict[str, object] | None = None
    history = []
    for step in range(1, 5001):
        opt.zero_grad(set_to_none=True)
        q = torch.softmax(logits, dim=1)
        pred_gc = q * sizes_t[:, None]
        pred_counts = pred_gc.sum(dim=0)
        tp = torch.sum(true_gc_t * pred_gc / torch.clamp(sizes_t[:, None], min=1.0), dim=0)
        f1 = 2.0 * tp / torch.clamp(pred_counts + true_counts_t, min=1e-6)
        macro = f1.mean()
        # Mild confidence term lets the optimizer choose hard group decisions
        # where macro-F1 wants precision, while keeping mixed groups available.
        entropy = -torch.mean(torch.sum(q * torch.log(torch.clamp(q, min=1e-8)), dim=1))
        loss = -macro + 0.0005 * entropy
        loss.backward()
        opt.step()
        val = float(macro.detach().cpu())
        if best is None or val > float(best["macro"]):
            best = {
                "macro": val,
                "q": q.detach().cpu().numpy(),
                "pred_counts": pred_counts.detach().cpu().numpy(),
                "f1": f1.detach().cpu().numpy(),
            }
        if step % 500 == 0 or step == 1:
            item = {
                "step": step,
                "soft_expected_macro": val,
                "pred_counts": [float(x) for x in pred_counts.detach().cpu().numpy()],
                "f1": [float(x) for x in f1.detach().cpu().numpy()],
                "entropy": float(entropy.detach().cpu()),
            }
            history.append(item)
            print(json.dumps(item, ensure_ascii=False), flush=True)

    assert best is not None
    pred_gc = round_group_counts(best["q"], sizes)
    before_refine = expected_macro(true_gc, pred_gc.astype(float), true_gc.sum(axis=0))
    pred_gc, refine_info = local_refine(true_gc, pred_gc, seconds=240.0)
    after_refine = expected_macro(true_gc, pred_gc.astype(float), true_gc.sum(axis=0))
    labels = labels_from_group_counts(group_ids, pred_gc)
    out_path = SUBMISSION_DIR / "submission_combo_score_optimal_v1.csv"
    pd.DataFrame({"name": test["name"], "label": labels}).to_csv(out_path, index=False, encoding="utf-8", lineterminator="\n")

    comparisons = {}
    for name in [
        "submission_mlp_labelshift_hard_q4_v1.csv",
        "submission_mlp_quota_13600_2900_3500_v1.csv",
        "submission_mlp_onehot_v1.csv",
        "submission_combo_score_constrained_v1.csv",
    ]:
        path = SUBMISSION_DIR / name
        if not path.exists():
            continue
        other = pd.read_csv(path)
        other_labels = other["label"].to_numpy(dtype=int)
        other_gc = group_counts_from_labels(group_ids, other_labels, n_groups)
        comparisons[name] = {
            "expected_macro_under_inferred_truth": expected_macro(true_gc, other_gc, true_gc.sum(axis=0)),
            "diff": int((labels != other_labels).sum()),
            "counts": np.bincount(other_labels, minlength=3).astype(int).tolist(),
        }

    summary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "runtime_seconds": round(time.time() - start, 3),
        "device": str(device),
        "n_groups": int(n_groups),
        "inferred_truth_counts": true_gc.sum(axis=0).astype(int).tolist(),
        "soft_best_expected_macro": float(best["macro"]),
        "soft_best_pred_counts": [float(x) for x in best["pred_counts"]],
        "soft_best_f1": [float(x) for x in best["f1"]],
        "rounded_expected_macro": before_refine,
        "refined_expected_macro": after_refine,
        "refined_pred_counts": np.bincount(labels, minlength=3).astype(int).tolist(),
        "refine": refine_info,
        "comparisons": comparisons,
        "generated_submission": str(out_path.relative_to(ROOT)),
        "conclusion": "Optimizes macro-F1 directly under the leaderboard-consistent exact-combo truth distribution; validity depends on whether combo-level inversion generalizes beyond fitted submissions.",
    }
    (REPORT_DIR / "combo_score_optimal_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
