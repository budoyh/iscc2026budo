from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import sparse

from common import FEATURE_CACHE, MODEL_PATH, NO_VULN, RAW, SUBMISSIONS, dataframe_to_sparse, ensure_dirs, load_joblib, load_test
from postprocess_submission import top_cwe
from probe_leakage import file_bytes, text_hash
from predict import validate_submission
from symbol_rule_predict import predict_one


def model_targets(artifact: dict, matrix: sparse.csr_matrix) -> np.ndarray:
    if artifact["strategy"] == "flat":
        return artifact["model"].predict(matrix)
    pred_bin = artifact["binary_model"].predict(matrix)
    targets = np.full(matrix.shape[0], NO_VULN, dtype=object)
    pos_mask = pred_bin == 1
    if np.any(pos_mask):
        targets[pos_mask] = artifact["cwe_model"].predict(matrix[pos_mask])
    return targets


def build_pure_text_hash_map() -> dict[str, str]:
    train = pd.read_csv(RAW / "train.csv")
    train["target"] = train["cwe_id"].fillna(NO_VULN)
    groups: dict[str, list[str]] = defaultdict(list)
    for row in train.itertuples(index=False):
        groups[text_hash(file_bytes(row.binary_id))].append(row.target)
    return {sig: values[0] for sig, values in groups.items() if values and len(set(values)) == 1}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate submission with symbol-derived label and model CWE.")
    parser.add_argument("--output", default="submission_symbol_hybrid_utf8_sig.csv")
    parser.add_argument("--no-hint", action="store_true")
    parser.add_argument("--use-text-hash", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    artifact = load_joblib(MODEL_PATH)
    features = pd.read_pickle(FEATURE_CACHE)
    test = load_test()
    test_frame = test.merge(features, on="binary_id", how="left", validate="one_to_one")
    text_matrix = artifact["vectorizer"].transform(test_frame["doc"].fillna(""))
    numeric_matrix = dataframe_to_sparse(test_frame, artifact["numeric_cols"])
    numeric_matrix = artifact["scaler"].transform(numeric_matrix)
    matrix = sparse.hstack([text_matrix, numeric_matrix], format="csr")

    fallback_targets = model_targets(artifact, matrix)
    cwe_targets = artifact["cwe_model"].predict(matrix) if artifact["strategy"] == "hierarchical" else fallback_targets
    pure_hash_map = build_pure_text_hash_map() if args.use_text_hash else {}

    final_targets = []
    unresolved = []
    changed_from_fallback = 0
    hinted = 0
    hash_overrides = 0
    for idx, binary_id in enumerate(test["binary_id"]):
        symbol_label, _, calls = predict_one(binary_id)
        if symbol_label == 0:
            target = NO_VULN
        elif symbol_label == 1:
            target = cwe_targets[idx]
        else:
            unresolved.append((binary_id, calls))
            target = fallback_targets[idx]

        mapped = pure_hash_map.get(text_hash(file_bytes(binary_id))) if pure_hash_map else None
        can_use_hash = (symbol_label == 1 and mapped and mapped != NO_VULN) or (symbol_label is None and mapped)
        if can_use_hash and mapped != target:
            target = mapped
            hash_overrides += 1

        hint = "" if args.no_hint else top_cwe(binary_id)
        if target != NO_VULN and hint and hint != target:
            target = hint
            hinted += 1
        if target != fallback_targets[idx]:
            changed_from_fallback += 1
        final_targets.append(target)

    out = pd.DataFrame(
        {
            "binary_id": test["binary_id"],
            "label": [0 if target == NO_VULN else 1 for target in final_targets],
            "cwe_id": ["" if target == NO_VULN else target for target in final_targets],
        }
    )
    validate_submission(out, test["binary_id"])
    output_path = SUBMISSIONS / args.output
    out.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(output_path)
    print(f"unresolved={len(unresolved)}")
    print(f"changed_from_fallback={changed_from_fallback}")
    print(f"hash_overrides={hash_overrides}")
    print(f"hinted={hinted}")
    print(f"label_dist={out['label'].value_counts().to_dict()}")
    print(f"top_cwe={out.loc[out['label'] == 1, 'cwe_id'].value_counts().head(15).to_dict()}")


if __name__ == "__main__":
    main()
