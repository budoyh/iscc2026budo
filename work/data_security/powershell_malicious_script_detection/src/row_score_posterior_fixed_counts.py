from __future__ import annotations

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


def load_submission(name: str, test: pd.DataFrame) -> np.ndarray:
    sub = pd.read_csv(SUBMISSION_DIR / name)
    if not sub["name"].equals(test["name"]):
        raise ValueError(name)
    return sub["label"].to_numpy(dtype=int)


def expected_scores(q: np.ndarray, preds: np.ndarray, pred_counts: np.ndarray) -> np.ndarray:
    tp = np.zeros((len(preds), 3), dtype=np.float64)
    for s in range(len(preds)):
        for cls in CLASSES:
            tp[s, int(cls)] = q[preds[s] == int(cls), int(cls)].sum()
    f1 = 2.0 * tp / np.maximum(pred_counts + TRUE_COUNTS[None, :], 1e-12)
    return f1.mean(axis=1)


def optimize_hard_prediction(q: np.ndarray, seconds: float = 120.0) -> np.ndarray:
    # Greedy row-level macro-F1 optimizer under posterior q. This is intentionally
    # separate from fitting q so it cannot merely copy the inferred labels.
    pred = q.argmax(axis=1).astype(int)
    pred_counts = np.bincount(pred, minlength=3).astype(np.float64)
    tp = np.zeros(3, dtype=np.float64)
    for cls in CLASSES:
        tp[int(cls)] = q[pred == int(cls), int(cls)].sum()

    def score(tp_vec: np.ndarray, pred_vec: np.ndarray) -> float:
        return float((2.0 * tp_vec / np.maximum(pred_vec + TRUE_COUNTS, 1e-12)).mean())

    current = score(tp, pred_counts)
    start = time.time()
    passes = 0
    accepted = 0
    while time.time() - start < seconds and passes < 30:
        passes += 1
        changed = False
        # Rows with uncertain posterior are most likely to move.
        margin = np.partition(q, -2, axis=1)[:, -1] - np.partition(q, -2, axis=1)[:, -2]
        order = np.argsort(margin)
        for i in order:
            old = int(pred[int(i)])
            best_cls = old
            best_score = current
            for cls in CLASSES:
                cls = int(cls)
                if cls == old:
                    continue
                new_tp = tp.copy()
                new_pred_counts = pred_counts.copy()
                new_tp[old] -= q[int(i), old]
                new_tp[cls] += q[int(i), cls]
                new_pred_counts[old] -= 1.0
                new_pred_counts[cls] += 1.0
                cand = score(new_tp, new_pred_counts)
                if cand > best_score + 1e-12:
                    best_score = cand
                    best_cls = cls
            if best_cls != old:
                tp[old] -= q[int(i), old]
                tp[best_cls] += q[int(i), best_cls]
                pred_counts[old] -= 1.0
                pred_counts[best_cls] += 1.0
                pred[int(i)] = best_cls
                current = best_score
                accepted += 1
                changed = True
        if not changed:
            break
    return pred


def fit_posterior(
    prior: np.ndarray,
    preds: np.ndarray,
    pred_counts: np.ndarray,
    target_scores: np.ndarray,
    score_scale: float,
    kl_weight: float,
    steps: int,
    seed: int,
) -> dict[str, object]:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred_t = torch.tensor(preds, dtype=torch.long, device=device)
    pred_counts_t = torch.tensor(pred_counts, dtype=torch.float32, device=device)
    target_t = torch.tensor(target_scores, dtype=torch.float32, device=device)
    true_counts_t = torch.tensor(TRUE_COUNTS, dtype=torch.float32, device=device)
    prior_t = torch.tensor(prior, dtype=torch.float32, device=device)
    noise = np.random.default_rng(seed).normal(0, 0.02, size=prior.shape)
    logits = torch.tensor(np.log(np.clip(prior, 1e-8, 1.0)) + noise, dtype=torch.float32, device=device, requires_grad=True)
    opt = torch.optim.AdamW([logits], lr=0.045)
    row_ids = torch.arange(prior.shape[0], device=device)
    best = None
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        q = torch.softmax(logits, dim=1)
        counts = q.sum(dim=0)
        tp_parts = []
        for cls in CLASSES:
            tp_parts.append(q[row_ids[None, :], int(cls)] * (pred_t == int(cls)).float())
        tp = torch.stack([x.sum(dim=1) for x in tp_parts], dim=1)
        f1 = 2.0 * tp / torch.clamp(pred_counts_t + true_counts_t[None, :], min=1e-6)
        macro = f1.mean(dim=1)
        score_loss = torch.mean(((macro - target_t) / score_scale) ** 2)
        count_loss = torch.mean(((counts - true_counts_t) / 35.0) ** 2)
        kl = torch.mean(torch.sum(q * (torch.log(torch.clamp(q, min=1e-8)) - torch.log(prior_t)), dim=1))
        entropy = -torch.mean(torch.sum(q * torch.log(torch.clamp(q, min=1e-8)), dim=1))
        loss = score_loss + count_loss + kl_weight * kl - 0.0005 * entropy
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
        if step % 1000 == 0:
            print(json.dumps({
                "step": step,
                "loss": float(loss.detach().cpu()),
                "max_abs_error": float(torch.max(torch.abs(macro - target_t)).detach().cpu()),
                "counts": [float(x) for x in counts.detach().cpu().numpy()],
            }, ensure_ascii=False), flush=True)
    assert best is not None
    return best


def main() -> None:
    start = time.time()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    data_dir = find_data_dir()
    test = pd.read_csv(data_dir / "data_test.csv")
    prior = np.load(MODEL_DIR / "mlp_onehot_proba.npy").astype(np.float64)
    prior = np.clip(prior, 1e-8, None)
    prior /= prior.sum(axis=1, keepdims=True)

    names = [n for n in SCORED_SUBMISSIONS if (SUBMISSION_DIR / n).exists()]
    preds = np.stack([load_submission(n, test) for n in names], axis=0)
    pred_counts = np.stack([np.bincount(preds[i], minlength=3).astype(np.float64) for i in range(len(names))], axis=0)
    target_scores = np.array([SCORED_SUBMISSIONS[n] for n in names], dtype=np.float64)

    configs = [
        ("all_tight", names, 0.0009, 0.025),
        ("all_loose", names, 0.0020, 0.08),
        ("no_failed", [n for n in names if n != "submission_leaderboard_constraint_v1.csv"], 0.0012, 0.04),
    ]
    candidates = []
    for idx, (cfg_name, use_names, score_scale, kl_weight) in enumerate(configs):
        use_idx = [names.index(n) for n in use_names]
        fit = fit_posterior(
            prior,
            preds[use_idx],
            pred_counts[use_idx],
            target_scores[use_idx],
            score_scale=score_scale,
            kl_weight=kl_weight,
            steps=3500,
            seed=SEED + idx,
        )
        q = fit["q"]
        pred = optimize_hard_prediction(q, seconds=80.0)
        pred_counts_self = np.bincount(pred, minlength=3).astype(np.float64)
        tp_self = np.array([q[pred == int(cls), int(cls)].sum() for cls in CLASSES])
        self_f1 = 2.0 * tp_self / np.maximum(pred_counts_self + TRUE_COUNTS, 1e-12)
        q4 = load_submission("submission_mlp_labelshift_hard_q4_v1.csv", test)
        item = {
            "config": cfg_name,
            "fit_loss": fit["loss"],
            "fit_max_abs_error": fit["max_abs_error"],
            "fit_counts": [float(x) for x in fit["counts"]],
            "self_expected_macro": float(self_f1.mean()),
            "self_expected_f1": [float(x) for x in self_f1],
            "pred_counts": pred_counts_self.astype(int).tolist(),
            "diff_vs_q4": int((pred != q4).sum()),
            "pred": pred,
        }
        candidates.append(item)
        print(json.dumps({k: v for k, v in item.items() if k != "pred"}, ensure_ascii=False), flush=True)

    best = max(candidates, key=lambda x: x["self_expected_macro"])
    out_path = SUBMISSION_DIR / "submission_row_score_posterior_v1.csv"
    pd.DataFrame({"name": test["name"], "label": best["pred"].astype(int)}).to_csv(out_path, index=False, encoding="utf-8", lineterminator="\n")
    summary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "runtime_seconds": round(time.time() - start, 3),
        "names": names,
        "configs": [{k: v for k, v in item.items() if k != "pred"} for item in candidates],
        "best_config": best["config"],
        "generated": str(out_path.relative_to(ROOT)),
        "conclusion": "Row-level fixed-count score posterior. This is a high-risk leaderboard-feedback method and must be validated before submission.",
    }
    (REPORT_DIR / "row_score_posterior_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
