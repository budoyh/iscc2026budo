from __future__ import annotations

import json
import math
import re
import time
import zlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
SUBMISSION_DIR = ROOT / "submissions"
SEED = 20260501
CLASSES = np.array([0, 1, 2], dtype=int)


def find_data_dir() -> Path:
    candidates = [p for p in DATA_ROOT.rglob("*") if (p / "data_train.csv").exists()]
    if not candidates:
        raise FileNotFoundError(f"data_train.csv not found under {DATA_ROOT}")
    return candidates[0]


def extract_id(name: str) -> int:
    match = re.search(r"(\d+)", str(name))
    if not match:
        raise ValueError(f"Cannot extract numeric id from {name!r}")
    return int(match.group(1))


def align_proba(model: MLPClassifier, proba: np.ndarray) -> np.ndarray:
    out = np.zeros((len(proba), len(CLASSES)), dtype=float)
    for src, cls in enumerate(model.classes_):
        out[:, int(cls)] = proba[:, src]
    return out


def build_mlp(seed: int) -> object:
    return make_pipeline(
        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(128, 64),
            alpha=1e-4,
            learning_rate_init=0.001,
            max_iter=180,
            early_stopping=True,
            n_iter_no_change=14,
            random_state=seed,
            batch_size=512,
            verbose=False,
        ),
    )


def round_counts(prior: np.ndarray, total: int) -> np.ndarray:
    raw = np.asarray(prior, dtype=float) * total
    counts = np.floor(raw).astype(int)
    remain = int(total - counts.sum())
    if remain > 0:
        order = np.argsort(-(raw - counts))
        counts[order[:remain]] += 1
    return counts


def adjust_to_quota(scores: np.ndarray, target_counts: np.ndarray) -> np.ndarray:
    target = np.asarray(target_counts, dtype=int)
    if target.sum() != len(scores):
        raise ValueError(f"Target counts {target.tolist()} do not sum to {len(scores)}")
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
    if not np.array_equal(np.bincount(pred, minlength=len(CLASSES)), target):
        raise RuntimeError(f"Quota failed: {np.bincount(pred, minlength=3)} vs {target}")
    return pred


def validate_submission(test: pd.DataFrame, submission: pd.DataFrame) -> None:
    if list(submission.columns) != ["name", "label"]:
        raise ValueError("Submission must have columns: name,label")
    if len(submission) != len(test):
        raise ValueError("Submission row count does not match test row count")
    if not submission["name"].equals(test["name"]):
        raise ValueError("Submission name order does not match test")
    if submission.isna().any().any():
        raise ValueError("Submission contains missing values")
    if not set(submission["label"].unique()).issubset(set(CLASSES)):
        raise ValueError("Submission contains invalid labels")


def write_submission(test: pd.DataFrame, pred: np.ndarray, filename: str) -> dict[str, object]:
    submission = pd.DataFrame({"name": test["name"], "label": pred.astype(int)})
    validate_submission(test, submission)
    path = SUBMISSION_DIR / filename
    submission.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    return {
        "path": str(path.relative_to(ROOT)),
        "pred_counts": {str(cls): int((pred == cls).sum()) for cls in CLASSES},
    }


def make_vectorizer(
    train_part: pd.DataFrame,
    target_part: pd.DataFrame,
    features: list[str],
    top_k: int,
    combo_weight: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    enc.fit(pd.concat([train_part[features], target_part[features]], ignore_index=True))
    train_marg = enc.transform(train_part[features]).astype(np.float32)
    target_marg = enc.transform(target_part[features]).astype(np.float32)

    train_keys = pd.Series(list(map(tuple, train_part[features].to_numpy())))
    target_keys = pd.Series(list(map(tuple, target_part[features].to_numpy())))
    top = list(pd.concat([train_keys, target_keys], ignore_index=True).value_counts().head(top_k).index)
    lookup = {key: i for i, key in enumerate(top)}
    train_combo = np.zeros((len(train_part), len(top)), dtype=np.float32)
    target_combo = np.zeros((len(target_part), len(top)), dtype=np.float32)
    for i, key in enumerate(train_keys):
        j = lookup.get(key)
        if j is not None:
            train_combo[i, j] = combo_weight
    for i, key in enumerate(target_keys):
        j = lookup.get(key)
        if j is not None:
            target_combo[i, j] = combo_weight
    return np.concatenate([train_marg, train_combo], axis=1), np.concatenate([target_marg, target_combo], axis=1)


def source_domains(train_part: pd.DataFrame, ids: np.ndarray, n_bins: int) -> list[dict[str, object]]:
    domains: list[dict[str, object]] = []
    labels = train_part["label"].to_numpy(dtype=int)
    local_ids = ids[train_part.index.to_numpy()]
    for label in CLASSES:
        local_pos = np.where(labels == label)[0]
        local_pos = local_pos[np.argsort(local_ids[local_pos])]
        for bin_id, part in enumerate(np.array_split(local_pos, n_bins)):
            if len(part) == 0:
                continue
            original_idx = train_part.index.to_numpy()[part]
            domains.append(
                {
                    "name": f"L{label}D{bin_id}",
                    "label": int(label),
                    "original_idx": original_idx,
                    "local_pos": part,
                    "size": int(len(part)),
                    "start": int(ids[original_idx].min()),
                    "end": int(ids[original_idx].max()),
                }
            )
    return domains


def estimate_prior(
    train_part: pd.DataFrame,
    target_part: pd.DataFrame,
    ids: np.ndarray,
    features: list[str],
    n_bins: int,
    top_k: int,
) -> dict[str, object]:
    train_vec, target_vec = make_vectorizer(train_part, target_part, features, top_k=top_k)
    domains = source_domains(train_part, ids, n_bins=n_bins)
    matrix = np.stack([train_vec[d["local_pos"]].mean(axis=0) for d in domains], axis=1)
    target = target_vec.mean(axis=0)
    reg = LinearRegression(positive=True, fit_intercept=False)
    reg.fit(matrix, target)
    weights = np.maximum(reg.coef_, 0)
    if weights.sum() <= 0:
        weights[:] = 1.0 / len(weights)
    else:
        weights = weights / weights.sum()
    prior = np.zeros(len(CLASSES), dtype=float)
    for i, domain in enumerate(domains):
        prior[int(domain["label"])] += weights[i]
    recon = matrix @ weights
    top_domains = sorted(
        [
            {
                "weight": float(weight),
                "domain": str(domain["name"]),
                "label": int(domain["label"]),
                "start": int(domain["start"]),
                "end": int(domain["end"]),
            }
            for weight, domain in zip(weights, domains)
            if weight > 1e-3
        ],
        key=lambda item: item["weight"],
        reverse=True,
    )[:10]
    return {
        "prior": prior,
        "counts": round_counts(prior, len(target_part)),
        "l1_error": float(np.mean(np.abs(recon - target))),
        "top_domains": top_domains,
    }


def contiguous_pick(sorted_idx: np.ndarray, start_frac: float, size: int) -> np.ndarray:
    if size > len(sorted_idx):
        raise ValueError("requested size exceeds class size")
    start = int(round((len(sorted_idx) - size) * start_frac))
    return sorted_idx[start : start + size]


def scenario_indices(train: pd.DataFrame, ids: np.ndarray) -> dict[str, np.ndarray]:
    by_label: dict[int, np.ndarray] = {}
    labels = train["label"].to_numpy(dtype=int)
    for label in CLASSES:
        idx = np.where(labels == label)[0]
        by_label[int(label)] = idx[np.argsort(ids[idx])]

    gap_sizes = {0: 8802, 1: 2088, 2: 2693}
    scenarios = {
        "prefix_gap_sizes": np.concatenate([contiguous_pick(by_label[label], 0.0, gap_sizes[label]) for label in CLASSES]),
        "middle_gap_sizes": np.concatenate([contiguous_pick(by_label[label], 0.5, gap_sizes[label]) for label in CLASSES]),
        "suffix_gap_sizes": np.concatenate([contiguous_pick(by_label[label], 1.0, gap_sizes[label]) for label in CLASSES]),
    }

    # Test-like proportions from robust domain-mixture estimates: about 70/14/16.
    mix_sizes = {0: 8400, 1: 1680, 2: 1920}
    scenarios["test_like_70_14_16_early_mid"] = np.concatenate(
        [
            contiguous_pick(by_label[0], 0.05, mix_sizes[0]),
            contiguous_pick(by_label[1], 0.40, mix_sizes[1]),
            contiguous_pick(by_label[2], 0.45, mix_sizes[2]),
        ]
    )
    scenarios["test_like_70_14_16_late"] = np.concatenate(
        [
            contiguous_pick(by_label[0], 0.55, mix_sizes[0]),
            contiguous_pick(by_label[1], 0.55, mix_sizes[1]),
            contiguous_pick(by_label[2], 0.60, mix_sizes[2]),
        ]
    )
    return scenarios


def evaluate_scenario(
    name: str,
    valid_idx: np.ndarray,
    train: pd.DataFrame,
    ids: np.ndarray,
    features: list[str],
    n_bins: int,
    top_k: int,
) -> dict[str, object]:
    valid_idx = np.asarray(sorted(set(map(int, valid_idx))), dtype=int)
    train_idx = np.setdiff1d(np.arange(len(train)), valid_idx)
    train_part = train.iloc[train_idx].copy()
    valid_part = train.iloc[valid_idx].copy()
    x_train = train_part[features].astype(int)
    y_train = train_part["label"].astype(int)
    x_valid = valid_part[features].astype(int)
    y_valid = valid_part["label"].to_numpy(dtype=int)

    model = build_mlp(SEED + zlib.crc32(name.encode("utf-8")) % 10000)
    model.fit(x_train, y_train)
    mlp = model.named_steps["mlpclassifier"]
    proba = align_proba(mlp, model.predict_proba(x_valid))
    raw_pred = proba.argmax(axis=1).astype(int)

    mix = estimate_prior(
        train_part=train_part,
        target_part=valid_part,
        ids=ids,
        features=features,
        n_bins=n_bins,
        top_k=top_k,
    )
    mix_pred = adjust_to_quota(proba, mix["counts"])
    true_counts = np.bincount(y_valid, minlength=len(CLASSES))
    oracle_pred = adjust_to_quota(proba, true_counts)

    fixed_prior = np.array([0.704, 0.138, 0.158], dtype=float)
    fixed_pred = adjust_to_quota(proba, round_counts(fixed_prior, len(valid_part)))
    handoff_prior = np.array([13600, 2900, 3500], dtype=float) / 20000.0
    handoff_pred = adjust_to_quota(proba, round_counts(handoff_prior, len(valid_part)))

    def item(pred: np.ndarray) -> dict[str, object]:
        return {
            "macro_f1": float(f1_score(y_valid, pred, average="macro")),
            "counts": {str(cls): int((pred == cls).sum()) for cls in CLASSES},
        }

    return {
        "name": name,
        "valid_size": int(len(valid_part)),
        "true_counts": {str(cls): int(true_counts[cls]) for cls in CLASSES},
        "true_prior": {str(cls): float(true_counts[cls] / len(valid_part)) for cls in CLASSES},
        "domain_mix": {
            "estimated_prior": {str(cls): float(mix["prior"][cls]) for cls in CLASSES},
            "estimated_counts": {str(cls): int(mix["counts"][cls]) for cls in CLASSES},
            "l1_error": mix["l1_error"],
            "top_domains": mix["top_domains"],
        },
        "raw_mlp": item(raw_pred),
        "domain_mix_quota": item(mix_pred),
        "fixed_704_138_158_quota": item(fixed_pred),
        "handoff_680_145_175_quota": item(handoff_pred),
        "oracle_true_quota": item(oracle_pred),
    }


def final_test_prior(
    train: pd.DataFrame,
    test: pd.DataFrame,
    ids: np.ndarray,
    features: list[str],
) -> dict[str, object]:
    estimates = []
    for n_bins in [10, 20, 40]:
        for top_k in [300, 800, 1500]:
            estimates.append(
                {
                    "n_bins": n_bins,
                    "top_k": top_k,
                    **estimate_prior(train, test, ids, features, n_bins=n_bins, top_k=top_k),
                }
            )
    priors = np.stack([item["prior"] for item in estimates], axis=0)
    # The 40-bin estimates are intentionally high variance; median keeps the robust 10/20-bin signal.
    robust_prior = np.median(priors, axis=0)
    robust_prior = robust_prior / robust_prior.sum()
    return {
        "estimates": [
            {
                "n_bins": int(item["n_bins"]),
                "top_k": int(item["top_k"]),
                "prior": {str(cls): float(item["prior"][cls]) for cls in CLASSES},
                "counts": {str(cls): int(item["counts"][cls]) for cls in CLASSES},
                "l1_error": float(item["l1_error"]),
                "top_domains": item["top_domains"],
            }
            for item in estimates
        ],
        "median_prior": {str(cls): float(robust_prior[cls]) for cls in CLASSES},
        "median_counts": {str(cls): int(round_counts(robust_prior, len(test))[cls]) for cls in CLASSES},
    }


def main() -> None:
    start = time.time()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

    data_dir = find_data_dir()
    train = pd.read_csv(data_dir / "data_train.csv")
    test = pd.read_csv(data_dir / "data_test.csv")
    features = [col for col in train.columns if col not in ["name", "label"]]
    ids = train["name"].map(extract_id).to_numpy(dtype=int)

    scenarios = scenario_indices(train, ids)
    validations = []
    for scenario_name, valid_idx in scenarios.items():
        print(f"evaluating {scenario_name}", flush=True)
        validations.append(
            evaluate_scenario(
                scenario_name,
                valid_idx,
                train=train,
                ids=ids,
                features=features,
                n_bins=20,
                top_k=800,
            )
        )

    test_prior = final_test_prior(train, test, ids, features)
    mlp_proba_path = MODEL_DIR / "mlp_onehot_proba.npy"
    if not mlp_proba_path.exists():
        raise FileNotFoundError(f"Missing {mlp_proba_path}; run generate_mlp_candidates.py first")
    mlp_proba = np.load(mlp_proba_path)
    target_counts = np.array([test_prior["median_counts"][str(cls)] for cls in CLASSES], dtype=int)
    pred = adjust_to_quota(mlp_proba, target_counts)
    candidate = write_submission(test, pred, "submission_mlp_domain_mix_prior_v1.csv")
    online_best_path = SUBMISSION_DIR / "submission_mlp_onehot_v1.csv"
    if online_best_path.exists():
        online_best = pd.read_csv(online_best_path)["label"].to_numpy(dtype=int)
        candidate["diff_vs_mlp_onehot_v1"] = int((pred != online_best).sum())
    candidate["recipe"] = "online-best sklearn MLP probabilities, least-loss quota to domain-mixture median prior"

    summary = {
        "seed": SEED,
        "hypothesis": "test is a non-IID mixture of hidden source domains; estimate target label prior from unlabeled test distribution, then quota-adjust the strongest MLP probabilities",
        "features": features,
        "validation": validations,
        "test_prior": test_prior,
        "candidate": candidate,
        "seconds": round(time.time() - start, 3),
    }
    (REPORT_DIR / "domain_mix_prior_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
