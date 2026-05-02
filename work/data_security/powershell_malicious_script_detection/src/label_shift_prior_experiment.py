from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit

from domain_mix_prior_experiment import (
    CLASSES,
    DATA_ROOT,
    MODEL_DIR,
    REPORT_DIR,
    ROOT,
    SEED,
    SUBMISSION_DIR,
    adjust_to_quota,
    align_proba,
    build_mlp,
    contiguous_pick,
    extract_id,
    find_data_dir,
    round_counts,
    validate_submission,
)


def hard_bbse_prior(cal_y: np.ndarray, cal_pred: np.ndarray, target_pred: np.ndarray) -> np.ndarray:
    confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=float)
    for y, pred in zip(cal_y, cal_pred):
        confusion[int(pred), int(y)] += 1.0
    confusion = confusion / np.maximum(confusion.sum(axis=0, keepdims=True), 1.0)
    target_pred_dist = np.bincount(target_pred, minlength=len(CLASSES)).astype(float) / len(target_pred)
    reg = LinearRegression(positive=True, fit_intercept=False)
    reg.fit(confusion, target_pred_dist)
    prior = np.maximum(reg.coef_, 0.0)
    if prior.sum() <= 0:
        prior[:] = 1.0 / len(prior)
    else:
        prior = prior / prior.sum()
    return prior


def soft_bbse_prior(cal_y: np.ndarray, cal_proba: np.ndarray, target_proba: np.ndarray) -> np.ndarray:
    confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=float)
    for cls in CLASSES:
        mask = cal_y == cls
        confusion[:, int(cls)] = cal_proba[mask].mean(axis=0)
    target_prob_dist = target_proba.mean(axis=0)
    reg = LinearRegression(positive=True, fit_intercept=False)
    reg.fit(confusion, target_prob_dist)
    prior = np.maximum(reg.coef_, 0.0)
    if prior.sum() <= 0:
        prior[:] = 1.0 / len(prior)
    else:
        prior = prior / prior.sum()
    return prior


def scenario_segments(train: pd.DataFrame, ids: np.ndarray, name: str) -> list[np.ndarray]:
    labels = train["label"].to_numpy(dtype=int)
    by_label: dict[int, np.ndarray] = {}
    for label in CLASSES:
        idx = np.where(labels == label)[0]
        by_label[int(label)] = idx[np.argsort(ids[idx])]

    gap_sizes = {0: 8802, 1: 2088, 2: 2693}
    if name in {"prefix", "middle", "suffix"}:
        frac = {"prefix": 0.0, "middle": 0.5, "suffix": 1.0}[name]
        return [np.concatenate([contiguous_pick(by_label[label], frac, gap_sizes[label]) for label in CLASSES])]

    if name == "ordered_two_block":
        first = {0: (0.04, 4680), 1: (0.15, 900), 2: (0.05, 420)}
        second = {0: (0.58, 4020), 1: (0.55, 780), 2: (0.50, 1200)}
        return [
            np.concatenate([contiguous_pick(by_label[label], start, size) for label, (start, size) in spec.items()])
            for spec in [first, second]
        ]

    raise ValueError(f"Unknown scenario: {name}")


def fit_calibrated_model(
    train_part: pd.DataFrame,
    features: list[str],
    seed: int,
) -> tuple[object, pd.DataFrame, np.ndarray]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    fit_local, cal_local = next(splitter.split(train_part[features], train_part["label"]))
    fit_part = train_part.iloc[fit_local]
    cal_part = train_part.iloc[cal_local]
    model = build_mlp(seed + 10000)
    model.fit(fit_part[features].astype(int), fit_part["label"].astype(int))
    mlp = model.named_steps["mlpclassifier"]
    cal_proba = align_proba(mlp, model.predict_proba(cal_part[features].astype(int)))
    return model, cal_part, cal_proba


def evaluate_scenario(
    train: pd.DataFrame,
    ids: np.ndarray,
    features: list[str],
    scenario_name: str,
    seeds: list[int],
) -> dict[str, object]:
    segments = scenario_segments(train, ids, scenario_name)
    ordered_valid_idx = np.concatenate(segments)
    unique_valid_idx = np.asarray(sorted(set(map(int, ordered_valid_idx))), dtype=int)
    train_idx = np.setdiff1d(np.arange(len(train)), unique_valid_idx)
    train_part = train.iloc[train_idx].copy()
    valid = train.iloc[ordered_valid_idx].copy()
    y_true = valid["label"].to_numpy(dtype=int)

    runs = []
    for seed in seeds:
        model, cal_part, cal_proba = fit_calibrated_model(train_part, features, seed)
        mlp = model.named_steps["mlpclassifier"]
        target_proba = align_proba(mlp, model.predict_proba(valid[features].astype(int)))
        raw_pred = target_proba.argmax(axis=1)
        cal_y = cal_part["label"].to_numpy(dtype=int)
        cal_pred = cal_proba.argmax(axis=1)

        hard_prior = hard_bbse_prior(cal_y, cal_pred, raw_pred)
        hard_counts = round_counts(hard_prior, len(valid))
        hard_pred = adjust_to_quota(target_proba, hard_counts)

        soft_prior = soft_bbse_prior(cal_y, cal_proba, target_proba)
        soft_counts = round_counts(soft_prior, len(valid))
        soft_pred = adjust_to_quota(target_proba, soft_counts)

        segment_pred = np.empty(len(valid), dtype=int)
        segment_priors = []
        start = 0
        for segment in segments:
            end = start + len(segment)
            seg_raw = raw_pred[start:end]
            seg_proba = target_proba[start:end]
            seg_prior = hard_bbse_prior(cal_y, cal_pred, seg_raw)
            seg_counts = round_counts(seg_prior, len(segment))
            segment_pred[start:end] = adjust_to_quota(seg_proba, seg_counts)
            segment_priors.append(
                {
                    "size": int(len(segment)),
                    "prior": {str(cls): float(seg_prior[cls]) for cls in CLASSES},
                    "counts": {str(cls): int(seg_counts[cls]) for cls in CLASSES},
                }
            )
            start = end

        runs.append(
            {
                "seed": int(seed),
                "raw_mlp": {
                    "macro_f1": float(f1_score(y_true, raw_pred, average="macro")),
                    "counts": {str(cls): int((raw_pred == cls).sum()) for cls in CLASSES},
                },
                "hard_global": {
                    "macro_f1": float(f1_score(y_true, hard_pred, average="macro")),
                    "prior": {str(cls): float(hard_prior[cls]) for cls in CLASSES},
                    "counts": {str(cls): int(hard_counts[cls]) for cls in CLASSES},
                },
                "soft_global": {
                    "macro_f1": float(f1_score(y_true, soft_pred, average="macro")),
                    "prior": {str(cls): float(soft_prior[cls]) for cls in CLASSES},
                    "counts": {str(cls): int(soft_counts[cls]) for cls in CLASSES},
                },
                "hard_segment": {
                    "macro_f1": float(f1_score(y_true, segment_pred, average="macro")),
                    "segments": segment_priors,
                    "counts": {str(cls): int((segment_pred == cls).sum()) for cls in CLASSES},
                },
            }
        )

    def mean_score(key: str) -> float:
        return float(np.mean([run[key]["macro_f1"] for run in runs]))

    return {
        "name": scenario_name,
        "valid_size": int(len(valid)),
        "true_counts": {str(cls): int((y_true == cls).sum()) for cls in CLASSES},
        "segment_true_counts": [
            {str(cls): int((train.iloc[segment]["label"].to_numpy(dtype=int) == cls).sum()) for cls in CLASSES}
            for segment in segments
        ],
        "mean_scores": {
            "raw_mlp": mean_score("raw_mlp"),
            "hard_global": mean_score("hard_global"),
            "soft_global": mean_score("soft_global"),
            "hard_segment": mean_score("hard_segment"),
        },
        "runs": runs,
    }


def estimate_test_priors(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    seeds: list[int],
) -> dict[str, object]:
    chunks = {
        "all": np.arange(len(test)),
        "q0": np.arange(0, 5000),
        "q1": np.arange(5000, 10000),
        "q2": np.arange(10000, 15000),
        "q3": np.arange(15000, 20000),
    }
    priors: dict[str, list[np.ndarray]] = {name: [] for name in chunks}
    raw_dists: dict[str, list[np.ndarray]] = {name: [] for name in chunks}

    for seed in seeds:
        model, cal_part, cal_proba = fit_calibrated_model(train, features, seed)
        mlp = model.named_steps["mlpclassifier"]
        test_proba = align_proba(mlp, model.predict_proba(test[features].astype(int)))
        test_pred = test_proba.argmax(axis=1)
        cal_y = cal_part["label"].to_numpy(dtype=int)
        cal_pred = cal_proba.argmax(axis=1)
        for name, idx in chunks.items():
            priors[name].append(hard_bbse_prior(cal_y, cal_pred, test_pred[idx]))
            raw_dists[name].append(np.bincount(test_pred[idx], minlength=len(CLASSES)).astype(float) / len(idx))

    output = {}
    for name, idx in chunks.items():
        mean_prior = np.mean(priors[name], axis=0)
        mean_prior = mean_prior / mean_prior.sum()
        mean_raw = np.mean(raw_dists[name], axis=0)
        output[name] = {
            "size": int(len(idx)),
            "hard_prior_mean": {str(cls): float(mean_prior[cls]) for cls in CLASSES},
            "hard_counts": {str(cls): int(round_counts(mean_prior, len(idx))[cls]) for cls in CLASSES},
            "raw_pred_dist_mean": {str(cls): float(mean_raw[cls]) for cls in CLASSES},
            "raw_pred_counts": {str(cls): int(round_counts(mean_raw, len(idx))[cls]) for cls in CLASSES},
        }
    return output


def write_submission(test: pd.DataFrame, pred: np.ndarray, filename: str) -> dict[str, object]:
    submission = pd.DataFrame({"name": test["name"], "label": pred.astype(int)})
    validate_submission(test, submission)
    path = SUBMISSION_DIR / filename
    submission.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    return {
        "path": str(path.relative_to(ROOT)),
        "pred_counts": {str(cls): int((pred == cls).sum()) for cls in CLASSES},
    }


def make_final_candidates(test: pd.DataFrame, test_prior: dict[str, object]) -> dict[str, object]:
    mlp_proba = np.load(MODEL_DIR / "mlp_onehot_proba.npy")
    online_best_path = SUBMISSION_DIR / "submission_mlp_onehot_v1.csv"
    online_best = None
    if online_best_path.exists():
        online_best = pd.read_csv(online_best_path)["label"].to_numpy(dtype=int)

    candidates = {}
    all_counts = np.array([test_prior["all"]["hard_counts"][str(cls)] for cls in CLASSES], dtype=int)
    pred = adjust_to_quota(mlp_proba, all_counts)
    item = write_submission(test, pred, "submission_mlp_labelshift_hard_global_v1.csv")
    item["recipe"] = "online-best sklearn MLP probabilities, hard BBSE global quota"
    if online_best is not None:
        item["diff_vs_mlp_onehot_v1"] = int((pred != online_best).sum())
    candidates["submission_mlp_labelshift_hard_global_v1.csv"] = item

    segment_pred = np.empty(len(test), dtype=int)
    start = 0
    for chunk_name in ["q0", "q1", "q2", "q3"]:
        end = start + int(test_prior[chunk_name]["size"])
        counts = np.array([test_prior[chunk_name]["hard_counts"][str(cls)] for cls in CLASSES], dtype=int)
        segment_pred[start:end] = adjust_to_quota(mlp_proba[start:end], counts)
        start = end
    item = write_submission(test, segment_pred, "submission_mlp_labelshift_hard_q4_v1.csv")
    item["recipe"] = "online-best sklearn MLP probabilities, hard BBSE quota independently per 5000-row test segment"
    if online_best is not None:
        item["diff_vs_mlp_onehot_v1"] = int((segment_pred != online_best).sum())
    candidates["submission_mlp_labelshift_hard_q4_v1.csv"] = item
    return candidates


def main() -> None:
    start = time.time()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

    data_dir = find_data_dir()
    train = pd.read_csv(data_dir / "data_train.csv")
    test = pd.read_csv(data_dir / "data_test.csv")
    features = [col for col in train.columns if col not in ["name", "label"]]
    ids = train["name"].map(extract_id).to_numpy(dtype=int)

    validation_seeds = [SEED + 3000 + i for i in range(5)]
    test_prior_seeds = [SEED + 5000 + i for i in range(8)]
    validation = [
        evaluate_scenario(train, ids, features, scenario, validation_seeds)
        for scenario in ["prefix", "middle", "suffix", "ordered_two_block"]
    ]
    test_prior = estimate_test_priors(train, test, features, test_prior_seeds)
    candidates = make_final_candidates(test, test_prior)

    summary = {
        "seed": SEED,
        "hypothesis": "label shift exists between training rows and hidden test; estimate target prior from a calibrated MLP confusion matrix and unlabeled test predictions",
        "validation_seeds": validation_seeds,
        "test_prior_seeds": test_prior_seeds,
        "validation": validation,
        "test_prior": test_prior,
        "candidates": candidates,
        "seconds": round(time.time() - start, 3),
    }
    (REPORT_DIR / "label_shift_prior_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
