from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import DEFAULT_PARAMS, MODEL_PATH, TEST_PATH, empty_prediction, extract_groups, load_model, validate_submission


STRATEGIES = {"abstain_many", "abstain_late_multi", "abstain_close", "abstain_multi"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ambiguity-abstained submissions.")
    parser.add_argument("--test-path", type=str, default=str(TEST_PATH))
    parser.add_argument("--model-path", type=str, default=str(MODEL_PATH))
    parser.add_argument("--strategy", choices=sorted(STRATEGIES), default="abstain_many")
    parser.add_argument("--output-path", type=str, default="")
    return parser.parse_args()


def should_abstain(strategy: str, candidates: list[dict[str, object]], chosen: dict[str, object]) -> bool:
    if strategy == "abstain_multi":
        return len(candidates) >= 2
    if strategy == "abstain_close":
        return len(candidates) >= 2 and float(candidates[1]["score"]) >= 0.45 * float(chosen["score"])
    if strategy == "abstain_late_multi":
        return len(candidates) >= 2 and int(chosen["start"]) > 20
    if strategy == "abstain_many":
        return len(candidates) >= 3
    return False


def predict_row_with_abstain(
    row_id: int,
    log_text: str,
    templates: dict[str, dict[str, object]],
    params: dict[str, float],
    strategy: str,
) -> dict[str, object]:
    groups = extract_groups(log_text, templates, params)
    candidates = [
        group
        for group in groups
        if int(group["match_count"]) >= params["group_min_matches"] and float(group["score"]) >= params["group_min_score"]
    ]
    if not candidates:
        return empty_prediction(row_id)

    candidates = sorted(
        candidates,
        key=lambda group: (int(group["start"]), -float(group["score"]), -int(group.get("max_support", 0))),
    )
    chosen = candidates[0]
    if should_abstain(strategy, candidates, chosen):
        return empty_prediction(row_id)

    lines = str(log_text).split("\n")
    start = int(chosen["start"])
    end = int(chosen["end"])
    if end - start + 1 < params["min_span_len"]:
        end = min(len(lines) - 1, start + params["min_span_len"] - 1)
    anomaly_type = str(chosen["type"])
    return {
        "id": row_id,
        "has_anomaly": 1,
        "primary_start_idx": start,
        "primary_end_idx": end,
        "primary_anomaly_type": anomaly_type,
        "all_spans": f"{start}|{end}|{anomaly_type}",
    }


def main() -> None:
    args = parse_args()
    test_df = pd.read_csv(args.test_path)
    model = load_model(Path(args.model_path))
    params = model.get("params", DEFAULT_PARAMS)

    records = [
        predict_row_with_abstain(int(row.id), row.log_text, model["templates"], params, args.strategy)
        for row in test_df.itertuples(index=False)
    ]
    submission = pd.DataFrame(records)
    errors = validate_submission(submission, expected_rows=len(test_df))
    if errors:
        raise SystemExit("submission validation failed: " + " | ".join(errors))

    output_path = Path(args.output_path) if args.output_path else Path(args.model_path).parents[1] / "submissions" / f"tail_clause_{args.strategy}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Submission saved to: {output_path}")
    print(f"Strategy: {args.strategy}")
    print(f"Anomaly count: {int(submission['has_anomaly'].sum())} / {len(submission)}")
    print("Validation: OK")


if __name__ == "__main__":
    main()
