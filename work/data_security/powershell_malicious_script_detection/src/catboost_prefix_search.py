from __future__ import annotations

import importlib.util
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
SUBMISSION_DIR = ROOT / "submissions"
CLASSES = np.array([0, 1, 2], dtype=int)
SEED = 20260503


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


tds = load_module(ROOT / "src" / "target_domain_structure_search.py", "tds")


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


def add_interactions(df: pd.DataFrame, features: list[str], pairs: list[tuple[str, str]]) -> pd.DataFrame:
    out = df.copy()
    for a, b in pairs:
        out[f"{a}__{b}"] = out[a].astype(str) + "_" + out[b].astype(str)
    return out


def adjust_to_quota(proba: np.ndarray, target_counts: list[int]) -> np.ndarray:
    target = np.array(target_counts, dtype=int)
    pred = proba.argmax(axis=1).astype(int)
    counts = np.bincount(pred, minlength=3)
    logp = np.log(np.clip(proba, 1e-15, None))
    for dst in np.where(counts < target)[0]:
        need = int(target[dst] - counts[dst])
        moves = []
        for src in np.where(counts > target)[0]:
            rows = np.where(pred == src)[0]
            loss = logp[rows, src] - logp[rows, dst]
            moves.extend((float(v), int(r), int(src)) for r, v in zip(rows, loss))
        moves.sort(key=lambda x: x[0])
        moved = 0
        for _, row, src in moves:
            if moved >= need:
                break
            if pred[row] != src or counts[src] <= target[src]:
                continue
            pred[row] = int(dst)
            counts[src] -= 1
            counts[dst] += 1
            moved += 1
    return pred


def eval_pred(y: np.ndarray, pred: np.ndarray) -> dict[str, object]:
    return {
        "macro": float(f1_score(y, pred, average="macro")),
        "per_class": [float(x) for x in f1_score(y, pred, labels=CLASSES, average=None)],
        "counts": np.bincount(pred, minlength=3).astype(int).tolist(),
    }


def train_predict(
    fit: pd.DataFrame,
    valid: pd.DataFrame,
    features: list[str],
    cat_features: list[int],
    cfg: dict[str, object],
) -> tuple[np.ndarray, int]:
    model = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="TotalF1:average=Macro",
        random_seed=int(cfg["seed"]),
        iterations=int(cfg["iterations"]),
        learning_rate=float(cfg["learning_rate"]),
        depth=int(cfg["depth"]),
        l2_leaf_reg=float(cfg["l2_leaf_reg"]),
        random_strength=float(cfg["random_strength"]),
        bagging_temperature=float(cfg["bagging_temperature"]),
        bootstrap_type="Bayesian",
        od_type="Iter",
        od_wait=120,
        allow_writing_files=False,
        verbose=False,
        thread_count=-1,
    )
    train_pool = Pool(fit[features], fit["label"].astype(int), cat_features=cat_features)
    valid_pool = Pool(valid[features], valid["label"].astype(int), cat_features=cat_features)
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    return model.predict_proba(valid_pool), int(model.get_best_iteration() or int(cfg["iterations"]))


def main() -> None:
    start = time.time()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    data_dir = find_data_dir()
    train = pd.read_csv(data_dir / "data_train.csv")
    test = pd.read_csv(data_dir / "data_test.csv")
    train["_id"] = train["name"].map(extract_id)
    base_features = [c for c in train.columns if c not in ["name", "label", "_id"]]
    # Pairs chosen from the fields most associated with malicious behavior and
    # obfuscation; CatBoost handles them as categorical tokens.
    pair_names = [
        ("decode_activity_profile", "content_encoding_profile"),
        ("network_command_profile", "command_surface_profile"),
        ("credential_runtime_profile", "task_registry_profile"),
        ("structure_rhythm_profile", "layout_variation_profile"),
        ("identifier_variation_profile", "content_encoding_profile"),
        ("function_scope_level", "pipeline_usage_level"),
    ]
    train_i = add_interactions(train, base_features, pair_names)
    test_i = add_interactions(test, base_features, pair_names)
    features = base_features + [f"{a}__{b}" for a, b in pair_names]
    cat_features = list(range(len(features)))

    cfgs = [
        {"name": "d5_lr035_l2", "depth": 5, "learning_rate": 0.035, "l2_leaf_reg": 8.0, "random_strength": 0.8, "bagging_temperature": 0.4, "iterations": 3500},
        {"name": "d7_lr025_l2", "depth": 7, "learning_rate": 0.025, "l2_leaf_reg": 10.0, "random_strength": 1.2, "bagging_temperature": 0.8, "iterations": 4200},
        {"name": "d8_lr018_l2", "depth": 8, "learning_rate": 0.018, "l2_leaf_reg": 16.0, "random_strength": 1.6, "bagging_temperature": 1.0, "iterations": 5000},
        {"name": "d6_lr03_l2hi", "depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 25.0, "random_strength": 1.5, "bagging_temperature": 0.7, "iterations": 4200},
    ]
    results = []
    for mode in ["ratio_prefix", "actual_prefix", "middle_ratio"]:
        fit, valid = tds.make_split(train_i, mode)
        y = valid["label"].to_numpy(dtype=int)
        target_counts = np.bincount(y, minlength=3).astype(int).tolist()
        for j, cfg in enumerate(cfgs):
            run_cfg = dict(cfg)
            run_cfg["seed"] = SEED + j
            proba, best_iter = train_predict(fit, valid, features, cat_features, run_cfg)
            raw = proba.argmax(axis=1).astype(int)
            item = {
                "mode": mode,
                "config": cfg["name"],
                "best_iter": best_iter,
                "raw": eval_pred(y, raw),
                "quota": eval_pred(y, adjust_to_quota(proba, target_counts)),
            }
            results.append(item)
            print(json.dumps(item, ensure_ascii=False), flush=True)

    # Pick by robust average of prefix-like validation, then train full model
    # with the median best iteration observed for that config.
    by_cfg: dict[str, list[float]] = {}
    by_iter: dict[str, list[int]] = {}
    for item in results:
        by_cfg.setdefault(str(item["config"]), []).append(float(item["quota"]["macro"]))
        by_iter.setdefault(str(item["config"]), []).append(int(item["best_iter"]))
    best_cfg_name = max(by_cfg, key=lambda k: (np.mean(by_cfg[k]), min(by_cfg[k])))
    cfg = next(c for c in cfgs if c["name"] == best_cfg_name)
    final_cfg = dict(cfg)
    final_cfg["seed"] = SEED + 100
    final_cfg["iterations"] = int(np.median(by_iter[best_cfg_name]) * 1.12)
    final_cfg["iterations"] = max(final_cfg["iterations"], 600)
    model = CatBoostClassifier(
        loss_function="MultiClass",
        random_seed=int(final_cfg["seed"]),
        iterations=int(final_cfg["iterations"]),
        learning_rate=float(final_cfg["learning_rate"]),
        depth=int(final_cfg["depth"]),
        l2_leaf_reg=float(final_cfg["l2_leaf_reg"]),
        random_strength=float(final_cfg["random_strength"]),
        bagging_temperature=float(final_cfg["bagging_temperature"]),
        bootstrap_type="Bayesian",
        allow_writing_files=False,
        verbose=False,
        thread_count=-1,
    )
    pool = Pool(train_i[features], train_i["label"].astype(int), cat_features=cat_features)
    test_pool = Pool(test_i[features], cat_features=cat_features)
    model.fit(pool)
    test_proba = model.predict_proba(test_pool)
    np.save(MODEL_DIR / "catboost_prefix_search_test_proba.npy", test_proba)
    raw_pred = test_proba.argmax(axis=1).astype(int)
    raw_path = SUBMISSION_DIR / "submission_catboost_prefix_raw_v1.csv"
    pd.DataFrame({"name": test["name"], "label": raw_pred}).to_csv(raw_path, index=False, encoding="utf-8", lineterminator="\n")
    quota_pred = adjust_to_quota(test_proba, [13600, 2800, 3600])
    quota_path = SUBMISSION_DIR / "submission_catboost_prefix_quota_v1.csv"
    pd.DataFrame({"name": test["name"], "label": quota_pred}).to_csv(quota_path, index=False, encoding="utf-8", lineterminator="\n")
    q4 = pd.read_csv(SUBMISSION_DIR / "submission_mlp_labelshift_hard_q4_v1.csv")["label"].to_numpy(dtype=int)
    summary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "runtime_seconds": round(time.time() - start, 3),
        "features": features,
        "results": results,
        "best_config": best_cfg_name,
        "final_iterations": int(final_cfg["iterations"]),
        "generated": {
            "raw": {
                "path": str(raw_path.relative_to(ROOT)),
                "counts": np.bincount(raw_pred, minlength=3).astype(int).tolist(),
                "diff_vs_q4": int((raw_pred != q4).sum()),
            },
            "quota": {
                "path": str(quota_path.relative_to(ROOT)),
                "counts": np.bincount(quota_pred, minlength=3).astype(int).tolist(),
                "diff_vs_q4": int((quota_pred != q4).sum()),
            },
        },
    }
    (REPORT_DIR / "catboost_prefix_search_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
