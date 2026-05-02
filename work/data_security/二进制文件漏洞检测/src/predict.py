from __future__ import annotations

import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import sparse

from common import FEATURE_CACHE, MODEL_PATH, NO_VULN, SUBMISSIONS, dataframe_to_sparse, ensure_dirs, load_joblib, load_test


def validate_submission(frame: pd.DataFrame, expected_ids: pd.Series) -> None:
    if list(frame.columns) not in (["binary_id", "label", "cwe_id"], ["name", "label", "cwe_id"]):
        raise ValueError(f"bad columns: {frame.columns.tolist()}")
    id_col = frame.columns[0]
    if len(frame) != len(expected_ids):
        raise ValueError(f"bad row count: {len(frame)} != {len(expected_ids)}")
    if frame[id_col].duplicated().any():
        raise ValueError(f"duplicated {id_col} found")
    if frame[id_col].tolist() != expected_ids.tolist():
        raise ValueError(f"{id_col} order does not match test.csv")
    if frame["label"].isna().any():
        raise ValueError("label contains null")
    if not frame.loc[frame["label"] == 0, "cwe_id"].fillna("").eq("").all():
        raise ValueError("label=0 rows must have empty cwe_id")
    if not frame.loc[frame["label"] == 1, "cwe_id"].astype(str).str.startswith("CWE-").all():
        raise ValueError("label=1 rows must have CWE-* cwe_id")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate submission from the trained baseline.")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--id-column", choices=["binary_id", "name"], default="binary_id")
    parser.add_argument("--encoding", choices=["utf-8", "utf-8-sig"], default="utf-8")
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
    if artifact["strategy"] == "flat":
        pred_target = artifact["model"].predict(matrix)
    else:
        pred_bin = artifact["binary_model"].predict(matrix)
        pred_target = np.full(len(test_frame), NO_VULN, dtype=object)
        pos_mask = pred_bin == 1
        if np.any(pos_mask):
            pred_target[pos_mask] = artifact["cwe_model"].predict(matrix[pos_mask])

    submission = pd.DataFrame(
        {
            args.id_column: test["binary_id"],
            "label": np.where(pred_target == NO_VULN, 0, 1).astype(int),
            "cwe_id": np.where(pred_target == NO_VULN, "", pred_target),
        }
    )
    validate_submission(submission, test["binary_id"])

    if args.output:
        output_path = SUBMISSIONS / args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = SUBMISSIONS / f"submission_linear_svc_{timestamp}.csv"
    submission.to_csv(output_path, index=False, encoding=args.encoding)
    print(output_path)


if __name__ == "__main__":
    main()
