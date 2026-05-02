from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC

from common import FEATURE_CACHE, NO_VULN, REPORT, SEED, load_train
from postprocess_submission import top_cwe
from probe_leakage import file_bytes, text_hash
from symbol_rule_predict import predict_one
from train import build_split, prepare_matrices


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

    groups: dict[str, list[str]] = defaultdict(list)
    for binary_id, target in zip(frame.loc[train_idx, "binary_id"], labels[train_idx], strict=False):
        groups[text_hash(file_bytes(binary_id))].append(target)
    pure_hash = {sig: values[0] for sig, values in groups.items() if len(values) >= 1 and len(set(values)) == 1}

    rows = []
    for binary_id, true_target, cwe_pred in zip(
        frame.loc[valid_idx, "binary_id"], labels[valid_idx], cwe_pred_valid, strict=False
    ):
        symbol_label, _, calls = predict_one(binary_id)
        if symbol_label == 0:
            pred = NO_VULN
        elif symbol_label == 1:
            pred = cwe_pred
        else:
            pred = cwe_pred
        mapped = pure_hash.get(text_hash(file_bytes(binary_id)))
        if symbol_label == 1 and mapped and mapped != NO_VULN:
            pred = mapped
        elif symbol_label is None and mapped:
            pred = mapped
        hint = top_cwe(binary_id)
        if pred != NO_VULN and hint:
            pred = hint
        rows.append(
            {
                "binary_id": binary_id,
                "true": true_target,
                "pred": pred,
                "cwe_model_pred": cwe_pred,
                "hint": hint,
                "symbol_label": symbol_label,
                "calls": "|".join(calls),
            }
        )
    out = pd.DataFrame(rows)
    out_path = REPORT / "symbol_hybrid_valid_errors.csv"
    out.to_csv(out_path, index=False, encoding="utf-8")
    errors = out[out["true"] != out["pred"]]
    print("rows", len(out), "errors", len(errors), "error_rate", len(errors) / len(out))
    print("error_pairs")
    print(errors.groupby(["true", "pred"]).size().sort_values(ascending=False).head(40).to_string())
    print("true_error_counts")
    print(errors["true"].value_counts().head(40).to_string())
    report = classification_report(out["true"], out["pred"], zero_division=0, output_dict=True)
    low = []
    for label, metrics in report.items():
        if isinstance(metrics, dict) and label not in {"accuracy", "macro avg", "weighted avg"}:
            low.append((label, metrics.get("f1-score", 0.0), metrics.get("support", 0.0)))
    print("low_f1")
    print(sorted(low, key=lambda item: item[1])[:30])
    print(out_path)


if __name__ == "__main__":
    main()
