from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from collections import defaultdict
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC

from common import FEATURE_CACHE, NO_VULN, SEED, dataframe_to_sparse, load_train
from postprocess_submission import top_cwe
from probe_leakage import file_bytes, text_hash
from symbol_rule_predict import predict_one
from train import build_split, prepare_matrices


def macro(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, average="macro")


def main() -> None:
    features = pd.read_pickle(FEATURE_CACHE)
    train = load_train()[["binary_id", "target", "label", "cwe_id"]]
    frame = train.merge(features, on=["binary_id", "target"], how="left", validate="one_to_one")
    matrix, _, _, _ = prepare_matrices(frame, 2**19, "l2")
    labels = frame["target"].to_numpy()
    train_idx, valid_idx = build_split(frame, 0.2)

    pos_train_idx = train_idx[labels[train_idx] != NO_VULN]
    cwe_model = LinearSVC(C=1.1, class_weight="balanced", random_state=SEED, max_iter=12000)
    cwe_model.fit(matrix[pos_train_idx], labels[pos_train_idx])
    cwe_pred_valid = cwe_model.predict(matrix[valid_idx])

    hash_to_targets: dict[str, list[str]] = defaultdict(list)
    train_binary_ids = frame.loc[train_idx, "binary_id"].to_numpy()
    for binary_id, target in zip(train_binary_ids, labels[train_idx], strict=False):
        hash_to_targets[text_hash(file_bytes(binary_id))].append(target)
    pure_hash_target = {
        sig: values[0]
        for sig, values in hash_to_targets.items()
        if len(values) >= 1 and len(set(values)) == 1
    }

    symbol_labels = []
    hints = []
    unresolved = 0
    for binary_id in frame.loc[valid_idx, "binary_id"]:
        label, _, _ = predict_one(binary_id)
        if label is None:
            unresolved += 1
        symbol_labels.append(label)
        hints.append(top_cwe(binary_id))

    base_symbol_pred = []
    hint_symbol_pred = []
    hash_symbol_pred = []
    hash_hint_symbol_pred = []
    fallback_symbol_pred = []
    valid_binary_ids = frame.loc[valid_idx, "binary_id"].to_numpy()
    for binary_id, idx, symbol_label, cwe_pred, hint in zip(
        valid_binary_ids, valid_idx, symbol_labels, cwe_pred_valid, hints, strict=False
    ):
        if symbol_label == 0:
            base_target = NO_VULN
        elif symbol_label == 1:
            base_target = cwe_pred
        else:
            base_target = cwe_pred
        base_symbol_pred.append(base_target)

        hinted = base_target
        if hinted != NO_VULN and hint:
            hinted = hint
        hint_symbol_pred.append(hinted)

        mapped_target = pure_hash_target.get(text_hash(file_bytes(binary_id)))
        if symbol_label == 1 and mapped_target and mapped_target != NO_VULN:
            hash_target = mapped_target
        elif symbol_label is None and mapped_target:
            hash_target = mapped_target
        else:
            hash_target = base_target
        hash_symbol_pred.append(hash_target)
        hash_hint_target = hash_target
        if hash_hint_target != NO_VULN and hint:
            hash_hint_target = hint
        hash_hint_symbol_pred.append(hash_hint_target)

        fallback_symbol_pred.append(base_target)

    y_true = labels[valid_idx]
    print("symbol_cwe_model_macro", macro(y_true, base_symbol_pred))
    print("symbol_cwe_model_hint_macro", macro(y_true, hint_symbol_pred))
    print("symbol_cwe_model_hash_macro", macro(y_true, hash_symbol_pred))
    print("symbol_cwe_model_hash_hint_macro", macro(y_true, hash_hint_symbol_pred))
    print("unresolved", unresolved)

    positive_true = y_true != NO_VULN
    print("cwe_only_accuracy_on_true_positive", float(np.mean(np.array(base_symbol_pred, dtype=object)[positive_true] == y_true[positive_true])))
    print("hint_cwe_only_accuracy_on_true_positive", float(np.mean(np.array(hint_symbol_pred, dtype=object)[positive_true] == y_true[positive_true])))


if __name__ == "__main__":
    main()
