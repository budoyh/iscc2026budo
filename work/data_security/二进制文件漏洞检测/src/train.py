from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC

from common import FEATURE_CACHE, MODEL_PATH, NO_VULN, REPORT, SEED, VALID_PRED_PATH, dataframe_to_sparse, ensure_dirs, load_train, save_joblib


def build_split(frame: pd.DataFrame, valid_size: float) -> tuple[pd.Index, pd.Index]:
    counts = frame["target"].value_counts()
    rare_classes = set(counts[counts < 2].index)
    rare_mask = frame["target"].isin(rare_classes)

    regular = frame.loc[~rare_mask]
    rare = frame.loc[rare_mask]
    train_idx, valid_idx = train_test_split(
        regular.index,
        test_size=valid_size,
        random_state=SEED,
        stratify=regular["target"],
    )
    train_index = pd.Index(train_idx).append(rare.index)
    valid_index = pd.Index(valid_idx)
    return train_index, valid_index


def prepare_matrices(
    frame: pd.DataFrame,
    text_dim: int,
    norm: str,
) -> tuple[sparse.csr_matrix, list[str], HashingVectorizer, MaxAbsScaler]:
    drop_cols = {"binary_id", "path", "doc", "split", "target", "label", "cwe_id"}
    numeric_cols = [col for col in frame.columns if col not in drop_cols]
    vectorizer = HashingVectorizer(
        n_features=text_dim,
        alternate_sign=False,
        norm=norm,
        ngram_range=(1, 2),
        lowercase=False,
        token_pattern=r"(?u)\S+",
    )
    text_matrix = vectorizer.transform(frame["doc"].fillna(""))
    numeric_matrix = dataframe_to_sparse(frame, numeric_cols)
    scaler = MaxAbsScaler()
    numeric_matrix = scaler.fit_transform(numeric_matrix)
    matrix = sparse.hstack([text_matrix, numeric_matrix], format="csr")
    return matrix, numeric_cols, vectorizer, scaler


def evaluate_predictions(labels_true: np.ndarray, pred_target: np.ndarray) -> tuple[float, float]:
    macro_f1 = f1_score(labels_true, pred_target, average="macro")
    binary_true = np.where(labels_true == NO_VULN, 0, 1)
    binary_pred = np.where(pred_target == NO_VULN, 0, 1)
    binary_f1 = f1_score(binary_true, binary_pred, average="macro")
    return float(macro_f1), float(binary_f1)


def fit_flat(
    matrix: sparse.csr_matrix,
    labels: np.ndarray,
    train_idx: pd.Index,
    valid_idx: pd.Index,
    c: float,
    max_iter: int,
) -> tuple[dict[str, object], np.ndarray]:
    clf = LinearSVC(C=c, class_weight="balanced", random_state=SEED, max_iter=max_iter)
    clf.fit(matrix[train_idx], labels[train_idx])
    pred_target = clf.predict(matrix[valid_idx])
    artifact = {
        "strategy": "flat",
        "model": clf,
        "c": c,
    }
    return artifact, pred_target


def fit_hierarchical(
    matrix: sparse.csr_matrix,
    labels: np.ndarray,
    train_idx: pd.Index,
    valid_idx: pd.Index,
    binary_c: float,
    cwe_c: float,
    max_iter: int,
) -> tuple[dict[str, object], np.ndarray]:
    binary_y = np.where(labels == NO_VULN, 0, 1)
    binary_clf = LinearSVC(C=binary_c, random_state=SEED, max_iter=max_iter)
    binary_clf.fit(matrix[train_idx], binary_y[train_idx])

    pos_train_idx = train_idx[labels[train_idx] != NO_VULN]
    cwe_clf = LinearSVC(C=cwe_c, class_weight="balanced", random_state=SEED, max_iter=max_iter)
    cwe_clf.fit(matrix[pos_train_idx], labels[pos_train_idx])

    pred_bin = binary_clf.predict(matrix[valid_idx])
    pred_target = np.full(len(valid_idx), NO_VULN, dtype=object)
    pos_mask = pred_bin == 1
    if np.any(pos_mask):
        pred_target[pos_mask] = cwe_clf.predict(matrix[valid_idx][pos_mask])

    artifact = {
        "strategy": "hierarchical",
        "binary_model": binary_clf,
        "cwe_model": cwe_clf,
        "binary_c": binary_c,
        "cwe_c": cwe_c,
    }
    return artifact, pred_target


def fit_full_flat(matrix: sparse.csr_matrix, labels: np.ndarray, c: float, max_iter: int) -> LinearSVC:
    clf = LinearSVC(C=c, class_weight="balanced", random_state=SEED, max_iter=max_iter)
    clf.fit(matrix, labels)
    return clf


def fit_full_hierarchical(
    matrix: sparse.csr_matrix,
    labels: np.ndarray,
    binary_c: float,
    cwe_c: float,
    max_iter: int,
) -> tuple[LinearSVC, LinearSVC]:
    binary_y = np.where(labels == NO_VULN, 0, 1)
    binary_clf = LinearSVC(C=binary_c, random_state=SEED, max_iter=max_iter)
    binary_clf.fit(matrix, binary_y)
    pos_mask = labels != NO_VULN
    cwe_clf = LinearSVC(C=cwe_c, class_weight="balanced", random_state=SEED, max_iter=max_iter)
    cwe_clf.fit(matrix[pos_mask], labels[pos_mask])
    return binary_clf, cwe_clf


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the baseline model.")
    parser.add_argument("--text-dim", type=int, default=2**19)
    parser.add_argument("--valid-size", type=float, default=0.2)
    parser.add_argument("--c", type=float, default=1.2)
    parser.add_argument("--binary-c", type=float, default=1.0)
    parser.add_argument("--cwe-c", type=float, default=1.1)
    parser.add_argument("--norm", type=str, default="l2")
    parser.add_argument("--max-iter", type=int, default=12000)
    parser.add_argument("--strategy", choices=["flat", "hierarchical", "both"], default="both")
    parser.add_argument("--full", action="store_true", help="Fit on the full training set after validation.")
    args = parser.parse_args()

    ensure_dirs()
    if not FEATURE_CACHE.exists():
        raise FileNotFoundError(f"feature cache not found: {FEATURE_CACHE}")

    all_features = pd.read_pickle(FEATURE_CACHE)
    train = load_train()[["binary_id", "target", "label", "cwe_id"]]
    frame = train.merge(all_features, on=["binary_id", "target"], how="left", validate="one_to_one")

    matrix, numeric_cols, vectorizer, scaler = prepare_matrices(frame, args.text_dim, args.norm)
    labels = frame["target"].to_numpy()
    train_idx, valid_idx = build_split(frame, args.valid_size)

    candidates: list[tuple[dict[str, object], np.ndarray]] = []
    if args.strategy in {"flat", "both"}:
        candidates.append(fit_flat(matrix, labels, train_idx, valid_idx, args.c, args.max_iter))
    if args.strategy in {"hierarchical", "both"}:
        candidates.append(
            fit_hierarchical(
                matrix,
                labels,
                train_idx,
                valid_idx,
                args.binary_c,
                args.cwe_c,
                args.max_iter,
            )
        )

    scored: list[tuple[float, float, dict[str, object], np.ndarray]] = []
    for artifact, pred_target in candidates:
        macro_f1, binary_f1 = evaluate_predictions(labels[valid_idx], pred_target)
        scored.append((macro_f1, binary_f1, artifact, pred_target))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    macro_f1, binary_f1, best_artifact, valid_pred = scored[0]

    report = classification_report(labels[valid_idx], valid_pred, zero_division=0, output_dict=True)
    report_path = REPORT / "valid_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    valid_frame = frame.loc[valid_idx, ["binary_id", "target"]].copy()
    valid_frame["pred_target"] = valid_pred
    valid_frame["label"] = np.where(valid_frame["target"] == NO_VULN, 0, 1)
    valid_frame["pred_label"] = np.where(valid_frame["pred_target"] == NO_VULN, 0, 1)
    valid_frame["cwe_id"] = valid_frame["target"].replace({NO_VULN: ""})
    valid_frame["pred_cwe_id"] = valid_frame["pred_target"].replace({NO_VULN: ""})
    valid_frame.to_csv(VALID_PRED_PATH, index=False, encoding="utf-8")

    artifact = {
        **best_artifact,
        "vectorizer": vectorizer,
        "scaler": scaler,
        "numeric_cols": numeric_cols,
        "text_dim": args.text_dim,
        "norm": args.norm,
        "c": args.c,
        "macro_f1": float(macro_f1),
        "binary_f1": float(binary_f1),
    }

    if args.full:
        if artifact["strategy"] == "flat":
            artifact["model"] = fit_full_flat(matrix, labels, args.c, args.max_iter)
        else:
            binary_model, cwe_model = fit_full_hierarchical(
                matrix,
                labels,
                args.binary_c,
                args.cwe_c,
                args.max_iter,
            )
            artifact["binary_model"] = binary_model
            artifact["cwe_model"] = cwe_model

    save_joblib(artifact, MODEL_PATH)
    print(f"valid_macro_f1={macro_f1:.6f}")
    print(f"valid_binary_macro_f1={binary_f1:.6f}")
    print(f"strategy={artifact['strategy']}")
    print(f"artifact={MODEL_PATH}")
    print(f"report={report_path}")
    print(f"valid_predictions={VALID_PRED_PATH}")


if __name__ == "__main__":
    main()
