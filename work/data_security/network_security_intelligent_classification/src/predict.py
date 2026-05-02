from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw" / "data"
MODELS = ROOT / "models"
SUBMISSIONS = ROOT / "submissions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create validated submission files from saved CV probabilities.")
    parser.add_argument(
        "--run-ids",
        nargs="+",
        required=True,
        help="One or more model run ids under models/.",
    )
    parser.add_argument(
        "--weights",
        nargs="*",
        type=float,
        default=None,
        help="Optional blend weights aligned with run ids.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Submission file name. Defaults to submission_<run ids>.csv.",
    )
    return parser.parse_args()


def load_run(run_id: str) -> tuple[dict, np.ndarray, np.ndarray]:
    run_dir = MODELS / run_id
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    oof = np.load(run_dir / "oof_proba.npy")
    test = np.load(run_dir / "test_proba.npy")
    return metadata, oof, test


def normalize_weights(run_ids: list[str], weights: list[float] | None) -> np.ndarray:
    if weights is None or len(weights) == 0:
        return np.full(len(run_ids), 1.0 / len(run_ids), dtype=np.float64)
    if len(weights) != len(run_ids):
        raise ValueError("weights count must match run ids count")
    weights_arr = np.asarray(weights, dtype=np.float64)
    if np.any(weights_arr < 0):
        raise ValueError("weights must be non-negative")
    if weights_arr.sum() <= 0:
        raise ValueError("weights sum must be positive")
    return weights_arr / weights_arr.sum()


def validate_submission(submission: pd.DataFrame, sample: pd.DataFrame, classes: set[str]) -> dict[str, object]:
    return {
        "row_count": int(len(submission)),
        "sample_row_count": int(len(sample)),
        "columns_ok": submission.columns.tolist() == ["id", "label"],
        "id_order_ok": submission["id"].equals(sample["id"]),
        "null_count": int(submission.isna().sum().sum()),
        "duplicate_id_count": int(submission["id"].duplicated().sum()),
        "labels_ok": bool(set(submission["label"].unique()).issubset(classes)),
    }


def main() -> None:
    args = parse_args()
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)

    weights = normalize_weights(args.run_ids, args.weights)
    test_df = pd.read_csv(RAW / "test_data.csv")
    sample_df = pd.read_csv(RAW / "sample_submission.csv")

    blend_oof = None
    blend_test = None
    classes = None
    true_labels = None
    metadata_list = []

    for weight, run_id in zip(weights, args.run_ids):
        metadata, oof, test = load_run(run_id)
        metadata_list.append(metadata)

        current_classes = metadata["classes"]
        if classes is None:
            classes = current_classes
            oof_frame = pd.read_csv(MODELS / run_id / "oof_predictions.csv")
            true_labels = oof_frame["true_label"].to_numpy()
            blend_oof = np.zeros_like(oof, dtype=np.float64)
            blend_test = np.zeros_like(test, dtype=np.float64)
        elif current_classes != classes:
            raise ValueError("all runs must share the same class ordering")

        blend_oof += weight * oof
        blend_test += weight * test

    assert blend_oof is not None
    assert blend_test is not None
    assert classes is not None
    assert true_labels is not None

    label_array = np.asarray(classes)
    oof_pred = label_array[blend_oof.argmax(axis=1)]
    test_pred = label_array[blend_test.argmax(axis=1)]
    local_score = float(f1_score(true_labels, oof_pred, average="macro"))

    submission = pd.DataFrame({"id": test_df["id"], "label": test_pred})
    output_name = args.output_name or f"submission_{'_'.join(args.run_ids)}.csv"
    output_path = SUBMISSIONS / output_name
    submission.to_csv(output_path, index=False, encoding="utf-8")

    validation = validate_submission(submission, sample_df, set(classes))
    validation["local_macro_f1"] = local_score
    validation["weights"] = weights.tolist()
    validation["run_ids"] = args.run_ids
    validation["output_path"] = str(output_path)
    validation["model_strategies"] = [item["strategy"] for item in metadata_list]

    report_path = output_path.with_suffix(".json")
    report_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(validation, ensure_ascii=False))


if __name__ == "__main__":
    main()
