from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from common import CV_METRICS_PATH, DEFAULT_PARAMS, MODEL_PATH, OOF_PATH, TRAIN_PATH, build_template_counts, run_cross_validation, save_model, summarize_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the template-based log anomaly detector.")
    parser.add_argument("--train-path", type=str, default=str(TRAIN_PATH))
    parser.add_argument("--model-path", type=str, default=str(MODEL_PATH))
    parser.add_argument("--oof-path", type=str, default=str(OOF_PATH))
    parser.add_argument("--metrics-path", type=str, default=str(CV_METRICS_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_df = pd.read_csv(args.train_path)

    metrics, oof_df = run_cross_validation(train_df, DEFAULT_PARAMS)
    summary = summarize_metrics(metrics)

    oof_path = Path(args.oof_path)
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    oof_df.to_csv(oof_path, index=False, encoding="utf-8")

    metrics_path = Path(args.metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    full_template_counts = build_template_counts(train_df)
    save_model(full_template_counts, metrics, path=Path(args.model_path), params=DEFAULT_PARAMS)

    print("CV mean metrics")
    print(json.dumps(summary["mean"], ensure_ascii=False, indent=2))
    print(f"OOF saved to: {oof_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Model saved to: {args.model_path}")


if __name__ == "__main__":
    main()
