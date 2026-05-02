from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import DEFAULT_PARAMS, MODEL_PATH, SUBMISSION_PATH, TEST_PATH, load_model, predict_frame, validate_submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a competition submission with the template-based detector.")
    parser.add_argument("--test-path", type=str, default=str(TEST_PATH))
    parser.add_argument("--model-path", type=str, default=str(MODEL_PATH))
    parser.add_argument("--output-path", type=str, default=str(SUBMISSION_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test_df = pd.read_csv(args.test_path)
    model = load_model(Path(args.model_path))

    submission = predict_frame(test_df, model["templates"], model.get("params", DEFAULT_PARAMS))
    errors = validate_submission(submission, expected_rows=len(test_df))
    if errors:
        raise SystemExit("submission validation failed: " + " | ".join(errors))

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Submission saved to: {output_path}")
    print("Validation: OK")


if __name__ == "__main__":
    main()
