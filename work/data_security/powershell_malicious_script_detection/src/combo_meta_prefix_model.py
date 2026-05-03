from __future__ import annotations

import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
REPORT_DIR = ROOT / "reports"
SUBMISSION_DIR = ROOT / "submissions"
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


def combo_keys(frame: pd.DataFrame, features: list[str]) -> list[tuple[int, ...]]:
    return list(map(tuple, frame[features].to_numpy(dtype=np.int16)))


def class_positions(frame: pd.DataFrame) -> dict[int, np.ndarray]:
    labels = frame["label"].to_numpy(dtype=int)
    ids = frame["_id"].to_numpy(dtype=int)
    out: dict[int, np.ndarray] = {}
    for cls in CLASSES:
        idx = np.where(labels == cls)[0]
        out[int(cls)] = idx[np.argsort(ids[idx])]
    return out


def make_outer_split(train: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    positions = class_positions(train)
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


def make_inner_task(source_all: pd.DataFrame, hold_fracs: dict[int, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    positions = class_positions(source_all)
    source_parts = []
    target_parts = []
    for cls in CLASSES:
        pos = positions[int(cls)]
        n = len(pos)
        hold = max(20, int(round(n * hold_fracs[int(cls)])))
        hold = min(hold, max(1, n - 20))
        target_parts.append(source_all.iloc[pos[:hold]])
        source_parts.append(source_all.iloc[pos[hold:]])
    return pd.concat(source_parts, ignore_index=True), pd.concat(target_parts, ignore_index=True)


def source_stats(source: pd.DataFrame, features: list[str]) -> dict[str, object]:
    y = source["label"].to_numpy(dtype=int)
    keys = combo_keys(source, features)
    combo_counts: dict[tuple[int, ...], np.ndarray] = defaultdict(lambda: np.zeros(len(CLASSES), dtype=float))
    for key, label in zip(keys, y):
        combo_counts[key][int(label)] += 1.0

    positions = class_positions(source)
    front_counts: dict[str, dict[tuple[int, ...], np.ndarray]] = {}
    for frac in [0.25, 0.5, 1.0]:
        name = f"front{frac}"
        front_counts[name] = defaultdict(lambda: np.zeros(len(CLASSES), dtype=float))
        for cls in CLASSES:
            pos = positions[int(cls)]
            take = max(1, int(round(len(pos) * frac)))
            sub = source.iloc[pos[:take]]
            for key in combo_keys(sub, features):
                front_counts[name][key][int(cls)] += 1.0

    feature_value_counts: dict[str, list[dict[int, float]]] = {}
    feature_global_counts: dict[str, dict[int, float]] = {}
    class_mass = np.bincount(y, minlength=len(CLASSES)).astype(float)
    for col in features:
        feature_value_counts[col] = []
        feature_global_counts[col] = source[col].value_counts().to_dict()
        for cls in CLASSES:
            feature_value_counts[col].append(source.loc[source["label"] == int(cls), col].value_counts().to_dict())
    return {
        "combo_counts": combo_counts,
        "front_counts": front_counts,
        "class_mass": class_mass,
        "global_prob": class_mass / class_mass.sum(),
        "feature_value_counts": feature_value_counts,
        "feature_global_counts": feature_global_counts,
        "n_source": len(source),
    }


def target_counts(target: pd.DataFrame, features: list[str]) -> dict[tuple[int, ...], np.ndarray]:
    counts: dict[tuple[int, ...], np.ndarray] = defaultdict(lambda: np.zeros(len(CLASSES), dtype=float))
    for key, label in zip(combo_keys(target, features), target["label"].to_numpy(dtype=int)):
        counts[key][int(label)] += 1.0
    return counts


def unlabeled_sizes(target: pd.DataFrame, features: list[str]) -> dict[tuple[int, ...], int]:
    sizes: dict[tuple[int, ...], int] = defaultdict(int)
    for key in combo_keys(target, features):
        sizes[key] += 1
    return sizes


def build_feature_row(
    key: tuple[int, ...],
    target_n: int,
    stats: dict[str, object],
    features: list[str],
) -> list[float]:
    global_prob = stats["global_prob"]
    combo_counts = stats["combo_counts"].get(key, np.zeros(len(CLASSES), dtype=float))
    total_combo = float(combo_counts.sum())
    row: list[float] = [float(v) for v in key]
    row.extend([math.log1p(target_n), target_n / max(1.0, float(stats["n_source"]))])
    row.extend([math.log1p(total_combo), total_combo / max(1.0, float(stats["n_source"]))])
    smooth = combo_counts + 1.0 * global_prob
    smooth = smooth / smooth.sum()
    row.extend(combo_counts.tolist())
    row.extend(smooth.tolist())
    for name in ["front0.25", "front0.5", "front1.0"]:
        fc = stats["front_counts"][name].get(key, np.zeros(len(CLASSES), dtype=float))
        fs = fc + 1.0 * global_prob
        fs = fs / fs.sum()
        row.extend(fc.tolist())
        row.extend(fs.tolist())

    nb_log = np.log(np.clip(global_prob, 1e-12, None))
    for col, value in zip(features, key):
        global_values = stats["feature_global_counts"][col]
        vocab_size = max(1, len(global_values))
        g = float(global_values.get(int(value), 0.0))
        for cls in CLASSES:
            cls_counts = stats["feature_value_counts"][col][int(cls)]
            cnt = float(cls_counts.get(int(value), 0.0))
            denom = float(stats["class_mass"][int(cls)] + vocab_size)
            nb_log[int(cls)] += math.log((cnt + 1.0) / max(1e-12, denom))
        row.append(g / max(1.0, float(stats["n_source"])))
    nb = np.exp(nb_log - nb_log.max())
    nb = nb / nb.sum()
    row.extend(nb.tolist())
    row.extend((nb / np.clip(smooth, 1e-9, None)).tolist())
    return row


def make_examples(
    source: pd.DataFrame,
    target: pd.DataFrame,
    features: list[str],
    require_labels: bool,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, list[tuple[int, ...]]]:
    stats = source_stats(source, features)
    sizes = unlabeled_sizes(target, features)
    labels_by_key = target_counts(target, features) if require_labels else {}
    x_rows = []
    y_rows = []
    weights = []
    keys = []
    for key, n in sizes.items():
        x_rows.append(build_feature_row(key, n, stats, features))
        keys.append(key)
        weights.append(float(n))
        if require_labels:
            y_rows.append(int(np.argmax(labels_by_key[key])))
    x = np.asarray(x_rows, dtype=np.float32)
    y = np.asarray(y_rows, dtype=int) if require_labels else None
    w = np.asarray(weights, dtype=np.float32)
    return x, y, w, keys


def train_meta_model(source_all: pd.DataFrame, features: list[str], variant: str):
    task_specs = []
    if variant == "broad":
        for f0 in [0.08, 0.15, 0.25, 0.37, 0.5]:
            task_specs.append({0: f0, 1: min(0.55, f0 * 0.8 + 0.08), 2: min(0.55, f0 * 0.9 + 0.06)})
    elif variant == "target_like":
        sizes = {int(cls): int((source_all["label"] == int(cls)).sum()) for cls in CLASSES}
        task_specs.append({cls: min(0.55, HIDDEN_COUNTS[cls] / max(1, sizes[cls] + HIDDEN_COUNTS[cls])) for cls in HIDDEN_COUNTS})
        task_specs.append({0: 0.35, 1: 0.18, 2: 0.25})
        task_specs.append({0: 0.45, 1: 0.25, 2: 0.35})
    else:
        raise ValueError(variant)
    xs, ys, ws = [], [], []
    for spec in task_specs:
        source, target = make_inner_task(source_all, spec)
        x, y, w, _ = make_examples(source, target, features, require_labels=True)
        xs.append(x)
        ys.append(y)
        ws.append(w)
    x_train = np.vstack(xs)
    y_train = np.concatenate(ys)
    w_train = np.concatenate(ws)
    if variant == "broad":
        model = ExtraTreesClassifier(
            n_estimators=700,
            max_features=0.65,
            min_samples_leaf=2,
            random_state=SEED,
            n_jobs=-1,
            class_weight="balanced",
        )
    else:
        model = RandomForestClassifier(
            n_estimators=600,
            max_features=0.7,
            min_samples_leaf=2,
            random_state=SEED + 17,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
    model.fit(x_train, y_train, sample_weight=np.sqrt(w_train))
    return model, {"n_examples": int(len(x_train)), "task_specs": task_specs}


def predict_rows_from_combo_model(
    model,
    source: pd.DataFrame,
    target: pd.DataFrame,
    features: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    x, _, _, keys = make_examples(source, target, features, require_labels=False)
    proba = model.predict_proba(x)
    aligned = np.zeros((len(keys), len(CLASSES)), dtype=float)
    for src, cls in enumerate(model.classes_):
        aligned[:, int(cls)] = proba[:, src]
    combo_pred = aligned.argmax(axis=1).astype(int)
    lookup = {key: int(pred) for key, pred in zip(keys, combo_pred)}
    row_pred = np.array([lookup[key] for key in combo_keys(target, features)], dtype=int)
    counts = {str(cls): int((row_pred == cls).sum()) for cls in CLASSES}
    return row_pred, aligned, counts


def evaluate_split(train: pd.DataFrame, features: list[str], split_mode: str, variant: str) -> dict[str, object]:
    fit, valid = make_outer_split(train, split_mode)
    model, info = train_meta_model(fit, features, variant)
    pred, _, counts = predict_rows_from_combo_model(model, fit, valid, features)
    y = valid["label"].to_numpy(dtype=int)
    return {
        "split": split_mode,
        "variant": variant,
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "per_class_f1": {str(cls): float(v) for cls, v in zip(CLASSES, f1_score(y, pred, average=None, labels=CLASSES))},
        "pred_counts": counts,
        "train_info": info,
    }


def validate_submission(test: pd.DataFrame, submission: pd.DataFrame) -> None:
    if list(submission.columns) != ["name", "label"]:
        raise ValueError("Submission must have name,label")
    if len(submission) != len(test):
        raise ValueError("Submission length mismatch")
    if not submission["name"].equals(test["name"]):
        raise ValueError("Submission name order mismatch")


def main() -> None:
    start = time.time()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    data_dir = find_data_dir()
    train = pd.read_csv(data_dir / "data_train.csv")
    test = pd.read_csv(data_dir / "data_test.csv")
    features = [c for c in train.columns if c not in ["name", "label"]]
    train["_id"] = train["name"].map(extract_id)
    records = []
    for variant in ["broad", "target_like"]:
        for split in ["ratio_prefix", "actual_prefix", "middle_ratio"]:
            rec = evaluate_split(train, features, split, variant)
            records.append(rec)
            print(rec)
    best_variant = max(
        ["broad", "target_like"],
        key=lambda v: np.mean([r["macro_f1"] for r in records if r["variant"] == v and r["split"] in ["ratio_prefix", "actual_prefix"]]),
    )
    final_model, final_info = train_meta_model(train, features, best_variant)
    pred, _, counts = predict_rows_from_combo_model(final_model, train, test, features)
    sub_path = SUBMISSION_DIR / "submission_combo_meta_prefix_v1.csv"
    submission = pd.DataFrame({"name": test["name"], "label": pred})
    validate_submission(test, submission)
    submission.to_csv(sub_path, index=False, encoding="utf-8", lineterminator="\n")

    q4_path = SUBMISSION_DIR / "submission_mlp_labelshift_hard_q4_v1.csv"
    diff_q4 = None
    if q4_path.exists():
        q4 = pd.read_csv(q4_path)["label"].to_numpy(dtype=int)
        diff_q4 = int((pred != q4).sum())
    summary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "runtime_seconds": round(time.time() - start, 3),
        "features": features,
        "records": records,
        "best_variant": best_variant,
        "generated_submission": {
            "path": str(sub_path.relative_to(ROOT)),
            "counts": counts,
            "diff_vs_labelshift_q4": diff_q4,
            "final_train_info": final_info,
        },
        "conclusion": "Combo-level meta-learning tries to predict hidden-prefix majority label per exact feature combo from simulated prefix tasks inside train.",
    }
    out = REPORT_DIR / "combo_meta_prefix_model_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
