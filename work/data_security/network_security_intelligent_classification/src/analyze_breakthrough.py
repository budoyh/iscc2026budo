from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw" / "data"
MODELS = ROOT / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Breakthrough diagnostics for target shift and hidden structure.")
    parser.add_argument("--domain-estimators", type=int, default=500)
    parser.add_argument("--cluster-seed", type=int, default=2026)
    parser.add_argument("--extra-runs", nargs="*", default=[])
    return parser.parse_args()


def apply_uniform_prior(proba: np.ndarray, alpha: float) -> np.ndarray:
    target = np.full(proba.shape[1], 1.0 / proba.shape[1], dtype=np.float64)
    current = np.clip(proba.mean(axis=0), 1e-15, 1.0)
    adjusted = proba * ((target / current) ** alpha)
    adjusted /= adjusted.sum(axis=1, keepdims=True)
    return adjusted


def load_run(run_id: str, classes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    run_dir = MODELS / run_id
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata["classes"] != classes:
        raise ValueError(f"class order mismatch for {run_id}")
    return np.load(run_dir / "oof_proba.npy").astype(np.float64), np.load(run_dir / "test_proba.npy").astype(np.float64)


def blend_runs(run_ids: list[str], weights: list[float], classes: list[str], soft_alpha: float | None) -> tuple[np.ndarray, np.ndarray]:
    weights_arr = np.asarray(weights, dtype=np.float64)
    weights_arr /= weights_arr.sum()
    oof = None
    test = None
    for run_id, weight in zip(run_ids, weights_arr):
        current_oof, current_test = load_run(run_id, classes)
        if oof is None:
            oof = np.zeros_like(current_oof, dtype=np.float64)
            test = np.zeros_like(current_test, dtype=np.float64)
        oof += weight * current_oof
        test += weight * current_test
    assert oof is not None and test is not None
    if soft_alpha is not None:
        oof = apply_uniform_prior(oof, soft_alpha)
        test = apply_uniform_prior(test, soft_alpha)
    return oof, test


def read_blend_report(path: Path) -> tuple[list[str], list[float], float | None]:
    report = json.loads(path.read_text(encoding="utf-8"))
    soft_alpha = report.get("soft_alpha") if report.get("soft_prior") == "uniform" else None
    return report["run_ids"], report["weights"], soft_alpha


def build_domain_scores(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_domain: np.ndarray,
    n_estimators: int,
) -> tuple[np.ndarray, float]:
    x_all = pd.concat([x_train, x_test], ignore_index=True)
    scores = np.zeros(len(x_all), dtype=np.float64)
    cv = KFold(n_splits=3, shuffle=True, random_state=2026)
    for tr_idx, va_idx in cv.split(x_all):
        model = LGBMClassifier(
            objective="binary",
            n_estimators=n_estimators,
            learning_rate=0.06,
            num_leaves=63,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=2026,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(x_all.iloc[tr_idx], y_domain[tr_idx])
        scores[va_idx] = model.predict_proba(x_all.iloc[va_idx])[:, 1]
    return scores, float(roc_auc_score(y_domain, scores))


def print_target_like_priors(train_scores: np.ndarray, y: np.ndarray, classes: list[str], class_array: np.ndarray) -> None:
    for q in [0.1, 0.2, 0.3, 0.5]:
        mask = train_scores >= np.quantile(train_scores, 1.0 - q)
        counts = pd.Series(class_array[y[mask]]).value_counts(normalize=True).reindex(classes).fillna(0.0)
        print("TARGET_LIKE_LABEL_PRIOR", q, counts.round(4).to_dict())


def score_existing_configs(
    train_scores: np.ndarray,
    y: np.ndarray,
    classes: list[str],
    extra_runs: list[str],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    configs = {
        "lgbm": (["lgbm_base_f5_s42"], [1.0], None),
        "xgb": (["xgb_base_f5_s42"], [1.0], None),
        "iter2": (["pseudo_lgbm_iter2_twoft_uniform100_t097_w045_f5_s42"], [1.0], None),
        "b01": read_blend_report(ROOT / "upload_ready_breakthrough" / "b01_target_aug_high.json"),
        "c01": read_blend_report(ROOT / "upload_ready_breakthrough2" / "c01_b01_iter_target_aug_high.json"),
        "c02": read_blend_report(ROOT / "upload_ready_breakthrough2" / "c02_b01_iter_target_aug_safe.json"),
        "target_b01_pow1p2": (["target_aug_lgbm_b01_pow1p2_aug080_t093_w055_f5_s42"], [1.0], None),
        "target_b01_pow0p8": (["target_aug_lgbm_b01_pow0p8_aug100_t090_w065_f5_s42"], [1.0], None),
    }
    for run_id in extra_runs:
        configs[run_id] = ([run_id], [1.0], None)
    probas: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    rows = []
    for name, (run_ids, weights, soft_alpha) in configs.items():
        oof, test = blend_runs(run_ids, weights, classes, soft_alpha)
        probas[name] = (oof, test)
        pred = oof.argmax(axis=1)
        row = {"model": name, "full": f1_score(y, pred, average="macro")}
        for q in [0.1, 0.2, 0.3, 0.5]:
            mask = train_scores >= np.quantile(train_scores, 1.0 - q)
            row[f"top{int(q * 100)}"] = f1_score(y[mask], pred[mask], average="macro")
        bottom = train_scores <= np.quantile(train_scores, 0.2)
        row["bottom20"] = f1_score(y[bottom], pred[bottom], average="macro")
        row["test_conf"] = float(test.max(axis=1).mean())
        entropy = -np.clip(test, 1e-15, 1.0) * np.log(np.clip(test, 1e-15, 1.0))
        row["test_entropy"] = float(entropy.sum(axis=1).mean())
        rows.append(row)

    print("PROXY_F1")
    print(pd.DataFrame(rows).round(6).to_string(index=False))
    return probas


def print_confusions(
    probas: dict[str, tuple[np.ndarray, np.ndarray]],
    train_scores: np.ndarray,
    y: np.ndarray,
    classes: list[str],
) -> None:
    mask = train_scores >= np.quantile(train_scores, 0.8)
    for name in ["lgbm", "b01", "c01", "target_b01_pow1p2"]:
        pred = probas[name][0].argmax(axis=1)
        cm = confusion_matrix(y[mask], pred[mask], labels=np.arange(len(classes)))
        pairs = []
        for i in range(len(classes)):
            for j in range(len(classes)):
                if i != j and cm[i, j] > 0:
                    pairs.append((int(cm[i, j]), classes[i], classes[j]))
        print("TOP_CONFUSIONS", name, sorted(pairs, reverse=True)[:12])


def inspect_clusters(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y: np.ndarray,
    classes: list[str],
    probas: dict[str, tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> None:
    class_count = len(classes)
    scaler = StandardScaler().fit(np.vstack([x_train, x_test]))
    z_all = scaler.transform(np.vstack([x_train, x_test])).astype(np.float32)
    for k in [12, 24, 48, 96]:
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=4096, n_init=5, max_iter=200)
        cluster = kmeans.fit_predict(z_all)
        train_cluster = cluster[: len(x_train)]
        test_cluster = cluster[len(x_train) :]
        majority = np.full(k, -1, dtype=np.int64)
        purity = []
        purity_by_cluster = np.zeros(k, dtype=np.float64)
        train_count = []
        test_count = []
        for cluster_id in range(k):
            train_idx = np.where(train_cluster == cluster_id)[0]
            train_count.append(int(len(train_idx)))
            test_count.append(int((test_cluster == cluster_id).sum()))
            if len(train_idx):
                values = np.bincount(y[train_idx], minlength=class_count)
                majority[cluster_id] = values.argmax()
                purity_by_cluster[cluster_id] = float(values.max() / values.sum())
                purity.append(float(purity_by_cluster[cluster_id]))
        train_pred = np.asarray([majority[c] if majority[c] >= 0 else 0 for c in train_cluster])
        valid = majority[train_cluster] >= 0
        cluster_f1 = f1_score(y[valid], train_pred[valid], average="macro")
        c01_test = probas["c01"][1].argmax(axis=1)
        test_pred = np.asarray([majority[c] if majority[c] >= 0 else 0 for c in test_cluster])
        agree = float((test_pred == c01_test).mean())

        high_test_share = []
        for cluster_id in range(k):
            total = train_count[cluster_id] + test_count[cluster_id]
            if total >= 100:
                cls = classes[majority[cluster_id]] if majority[cluster_id] >= 0 else "NA"
                high_test_share.append(
                    (
                        test_count[cluster_id] / total,
                        train_count[cluster_id],
                        test_count[cluster_id],
                        cls,
                        float(purity_by_cluster[cluster_id]),
                    )
                )
        high_test_share = sorted(high_test_share, reverse=True)[:8]
        print(
            "KMEANS",
            {
                "k": k,
                "train_cluster_majority_f1": round(float(cluster_f1), 4),
                "mean_purity": round(float(np.mean(purity)), 4),
                "test_majority_agree_c01": round(agree, 4),
                "high_test_share_clusters": [
                    (round(a, 3), b, c, d, round(e, 3)) for a, b, c, d, e in high_test_share
                ],
            },
        )


def print_public_counts(classes: list[str]) -> None:
    public_scores = {
        "r02_blend_raw_xgb.csv": 0.69083,
        "r03_blend_raw_xgb_soft_uniform.csv": 0.69291,
        "d02_raw_xgb_ft_pseudo_uniform_a1p0.csv": 0.69616,
        "m01_proxy_iter2_uniform_a1.csv": 0.69848,
        "b01_target_aug_high.csv": 0.70584,
        "c01_b01_iter_target_aug_high.csv": 0.71035,
    }
    paths = []
    for dirname in [
        "upload_ready_recommended",
        "upload_ready_deep",
        "upload_ready_main",
        "upload_ready_breakthrough",
        "upload_ready_breakthrough2",
    ]:
        for path in (ROOT / dirname).glob("*.csv"):
            if path.name in public_scores:
                paths.append(path)

    base_counts = None
    for path in paths:
        counts = pd.read_csv(path)["label"].value_counts().reindex(classes).fillna(0).astype(int)
        if path.name == "r03_blend_raw_xgb_soft_uniform.csv":
            base_counts = counts

    print("PUBLIC_COUNTS")
    for path in sorted(paths, key=lambda p: public_scores[p.name]):
        counts = pd.read_csv(path)["label"].value_counts().reindex(classes).fillna(0).astype(int)
        delta = (counts - base_counts).to_dict() if base_counts is not None else {}
        print(path.name, public_scores[path.name], counts.to_dict(), "delta_vs_r03", delta)


def main() -> None:
    args = parse_args()
    train = pd.read_csv(RAW / "train_data.csv")
    test = pd.read_csv(RAW / "test_data.csv")
    features = [col for col in train.columns if col not in {"id", "label"}]
    x_train = train[features]
    x_test = test[features]
    encoder = LabelEncoder()
    y = encoder.fit_transform(train["label"])
    classes = encoder.classes_.tolist()
    class_array = np.asarray(classes)

    print("DATA", {"train": train.shape, "test": test.shape, "classes": classes})
    print("TRAIN_COUNTS", train["label"].value_counts().sort_index().to_dict())
    unique_desc = train[features].nunique().describe().to_dict()
    integer_like = sum(
        np.allclose(train[col].to_numpy(), np.round(train[col].to_numpy()), atol=1e-8)
        and np.allclose(test[col].to_numpy(), np.round(test[col].to_numpy()), atol=1e-8)
        for col in features
    )
    mean_diff = (test[features].mean() - train[features].mean()) / train[features].std().replace(0, np.nan)
    std_ratio = test[features].std() / train[features].std().replace(0, np.nan)
    print("FEATURE_UNIQUE_DESC", {k: float(v) for k, v in unique_desc.items()}, "integer_like_features", int(integer_like))
    print("TOP_MEAN_SHIFT", mean_diff.abs().sort_values(ascending=False).head(10).round(4).to_dict())
    print("TOP_STD_RATIO", (std_ratio - 1.0).abs().sort_values(ascending=False).head(10).round(4).to_dict())

    train_hash = pd.util.hash_pandas_object(train[features], index=False)
    test_hash = pd.util.hash_pandas_object(test[features], index=False)
    print(
        "DUPLICATES",
        {
            "train_exact_dup": int(train_hash.duplicated().sum()),
            "test_exact_dup": int(test_hash.duplicated().sum()),
            "cross_exact_overlap": int(np.intersect1d(train_hash.values, test_hash.values).size),
        },
    )

    y_domain = np.r_[np.zeros(len(train), dtype=np.int64), np.ones(len(test), dtype=np.int64)]
    domain_scores, domain_auc = build_domain_scores(x_train, x_test, y_domain, args.domain_estimators)
    train_scores = domain_scores[: len(train)]
    test_scores = domain_scores[len(train) :]
    print(
        "DOMAIN",
        {
            "auc": round(domain_auc, 6),
            "train_q": np.quantile(train_scores, [0, 0.1, 0.2, 0.5, 0.8, 0.9, 1]).round(4).tolist(),
            "test_q": np.quantile(test_scores, [0, 0.1, 0.2, 0.5, 0.8, 0.9, 1]).round(4).tolist(),
        },
    )
    print_target_like_priors(train_scores, y, classes, class_array)
    probas = score_existing_configs(train_scores, y, classes, args.extra_runs)
    print_confusions(probas, train_scores, y, classes)
    inspect_clusters(x_train.to_numpy(np.float32), x_test.to_numpy(np.float32), y, classes, probas, args.cluster_seed)
    print_public_counts(classes)


if __name__ == "__main__":
    main()
