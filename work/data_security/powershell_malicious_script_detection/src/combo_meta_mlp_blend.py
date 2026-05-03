from __future__ import annotations

import json
import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from combo_meta_prefix_model import (
    CLASSES,
    ROOT,
    DATA_ROOT,
    HIDDEN_COUNTS,
    REPORT_DIR,
    SUBMISSION_DIR,
    combo_keys,
    extract_id,
    make_examples,
    make_outer_split,
    train_meta_model,
    validate_submission,
)


SEED = 20260503


def find_data_dir() -> Path:
    candidates = [p for p in DATA_ROOT.rglob("*") if (p / "data_train.csv").exists()]
    if not candidates:
        raise FileNotFoundError(f"data_train.csv not found under {DATA_ROOT}")
    return candidates[0]


def build_mlp(seed: int):
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


def align_proba(model, proba: np.ndarray) -> np.ndarray:
    out = np.zeros((len(proba), len(CLASSES)), dtype=float)
    for src, cls in enumerate(model.classes_):
        out[:, int(cls)] = proba[:, src]
    return out


def meta_row_proba(model, source: pd.DataFrame, target: pd.DataFrame, features: list[str]) -> np.ndarray:
    x, _, _, keys = make_examples(source, target, features, require_labels=False)
    proba = model.predict_proba(x)
    aligned = np.zeros((len(keys), len(CLASSES)), dtype=float)
    for src, cls in enumerate(model.classes_):
        aligned[:, int(cls)] = proba[:, src]
    lookup = {key: aligned[i] for i, key in enumerate(keys)}
    return np.vstack([lookup[key] for key in combo_keys(target, features)])


def adjust_to_quota(scores: np.ndarray, target_counts: np.ndarray) -> np.ndarray:
    target = np.asarray(target_counts, dtype=int)
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
    return pred


def metric(y: np.ndarray, pred: np.ndarray) -> dict[str, object]:
    return {
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "per_class_f1": {str(cls): float(v) for cls, v in zip(CLASSES, f1_score(y, pred, average=None, labels=CLASSES))},
        "counts": {str(cls): int((pred == cls).sum()) for cls in CLASSES},
    }


def evaluate_split(train: pd.DataFrame, features: list[str], split: str) -> list[dict[str, object]]:
    fit, valid = make_outer_split(train, split)
    y = valid["label"].to_numpy(dtype=int)
    x_fit = fit[features].astype(int)
    x_valid = valid[features].astype(int)
    mlp = build_mlp(SEED)
    mlp.fit(x_fit, fit["label"].astype(int))
    mlp_p = align_proba(mlp.named_steps["mlpclassifier"], mlp.predict_proba(x_valid))
    meta, info = train_meta_model(fit, features, "target_like")
    meta_p = meta_row_proba(meta, fit, valid, features)
    target_counts = np.bincount(y, minlength=len(CLASSES))
    records = []
    for a in np.linspace(0.0, 1.0, 11):
        for w2 in [0.9, 1.0, 1.08, 1.16, 1.25]:
            scores = ((1.0 - a) * mlp_p + a * meta_p) * np.array([1.0, 1.0, w2])
            for variant, pred in [
                ("raw", scores.argmax(axis=1).astype(int)),
                ("quota", adjust_to_quota(scores, target_counts)),
            ]:
                records.append({"split": split, "alpha_meta": float(a), "w2": float(w2), "variant": variant, **metric(y, pred)})
    return sorted(records, key=lambda r: float(r["macro_f1"]), reverse=True)


def main() -> None:
    start = time.time()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    data_dir = find_data_dir()
    train = pd.read_csv(data_dir / "data_train.csv")
    test = pd.read_csv(data_dir / "data_test.csv")
    features = [c for c in train.columns if c not in ["name", "label"]]
    train["_id"] = train["name"].map(extract_id)
    all_records = []
    top_by_split = {}
    for split in ["ratio_prefix", "actual_prefix", "middle_ratio"]:
        recs = evaluate_split(train, features, split)
        top_by_split[split] = recs[:10]
        all_records.extend(recs)
        print(split, recs[0])
    frame = pd.DataFrame(all_records)
    avg = (
        frame.groupby(["alpha_meta", "w2", "variant"])["macro_f1"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        .sort_values(["mean", "min"], ascending=False)
    )
    best = avg.iloc[0].to_dict()

    mlp_p = np.load(ROOT / "models" / "mlp_onehot_proba.npy")
    meta, final_info = train_meta_model(train, features, "target_like")
    meta_p = meta_row_proba(meta, train, test, features)
    scores = ((1.0 - float(best["alpha_meta"])) * mlp_p + float(best["alpha_meta"]) * meta_p) * np.array(
        [1.0, 1.0, float(best["w2"])]
    )
    if best["variant"] == "quota":
        pred = adjust_to_quota(scores, np.array([14000, 2500, 3500], dtype=int))
    else:
        pred = scores.argmax(axis=1).astype(int)
    sub = pd.DataFrame({"name": test["name"], "label": pred})
    validate_submission(test, sub)
    sub_path = SUBMISSION_DIR / "submission_combo_meta_mlp_blend_v1.csv"
    sub.to_csv(sub_path, index=False, encoding="utf-8", lineterminator="\n")
    q4_path = SUBMISSION_DIR / "submission_mlp_labelshift_hard_q4_v1.csv"
    diff_q4 = None
    if q4_path.exists():
        q4 = pd.read_csv(q4_path)["label"].to_numpy(dtype=int)
        diff_q4 = int((pred != q4).sum())
    csv_path = REPORT_DIR / "combo_meta_mlp_blend_results.csv"
    frame.to_csv(csv_path, index=False, encoding="utf-8")
    summary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "runtime_seconds": round(time.time() - start, 3),
        "top_by_split": top_by_split,
        "top_by_average": json.loads(avg.head(25).to_json(orient="records", force_ascii=False)),
        "generated_submission": {
            "path": str(sub_path.relative_to(ROOT)),
            "counts": {str(cls): int((pred == cls).sum()) for cls in CLASSES},
            "diff_vs_labelshift_q4": diff_q4,
            "best_params": best,
            "final_meta_info": final_info,
        },
        "result_csv": str(csv_path.relative_to(ROOT)),
    }
    out = REPORT_DIR / "combo_meta_mlp_blend_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
