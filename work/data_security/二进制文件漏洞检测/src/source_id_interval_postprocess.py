from __future__ import annotations

import argparse

import pandas as pd
from sklearn.metrics import f1_score

from common import NO_VULN, RAW, REPORT, SUBMISSIONS, ensure_dirs, load_test
from predict import validate_submission
from source_id_chunk_postprocess import add_source_id, source_id_number
from train import build_split


def build_intervals(train: pd.DataFrame, indices: list[int], min_count: int, margin: int) -> list[tuple[int, int, str]]:
    part = train.iloc[indices]
    part = part[part["label"] == 1]
    intervals: list[tuple[int, int, str]] = []
    for cwe, group in part.groupby("cwe_id"):
        if len(group) < min_count:
            continue
        lo = int(group["source_id_num"].min()) + margin
        hi = int(group["source_id_num"].max()) - margin
        if lo <= hi:
            intervals.append((lo, hi, str(cwe)))
    return sorted(intervals)


def interval_prediction(source_id: int, intervals: list[tuple[int, int, str]]) -> str | None:
    hits = [cwe for lo, hi, cwe in intervals if lo <= source_id <= hi]
    if len(hits) == 1:
        return hits[0]
    return None


def apply_intervals(
    targets: list[str],
    binary_ids: list[str],
    intervals: list[tuple[int, int, str]],
) -> tuple[list[str], list[tuple[str, int, str, str]]]:
    out = list(targets)
    changed: list[tuple[str, int, str, str]] = []
    for pos, (binary_id, target) in enumerate(zip(binary_ids, targets, strict=False)):
        if target == NO_VULN:
            continue
        source_id = source_id_number(binary_id)
        mapped = interval_prediction(source_id, intervals)
        if mapped and mapped != target:
            out[pos] = mapped
            changed.append((binary_id, source_id, target, mapped))
    return out, changed


def evaluate(input_name: str, min_count: int, margin: int) -> None:
    train = pd.read_csv(RAW / "train.csv")
    train["target"] = train["cwe_id"].fillna(NO_VULN)
    train = add_source_id(train)
    train_idx, _valid_idx = build_split(train, 0.2)
    intervals = build_intervals(train, list(train_idx), min_count=min_count, margin=margin)

    frame = pd.read_csv(REPORT / input_name)
    targets, changed = apply_intervals(frame["pred"].tolist(), frame["binary_id"].tolist(), intervals)
    useful = harmful = neutral = 0
    truth = dict(zip(frame["binary_id"], frame["true"], strict=False))
    for binary_id, _source_id, old, new in changed:
        true = str(truth[binary_id])
        useful += int(new == true and old != true)
        harmful += int(new != true and old == true)
        neutral += int(new != true and old != true)

    print(f"base_macro={f1_score(frame['true'], frame['pred'], average='macro'):.12f}")
    print(f"source_interval_macro={f1_score(frame['true'], targets, average='macro'):.12f}")
    print(f"intervals={len(intervals)} changed={len(changed)} useful={useful} harmful={harmful} neutral={neutral}")
    print(f"changed_examples={changed[:30]}")

    out = frame.copy()
    out["pred"] = targets
    out_path = REPORT / "source_id_interval_valid_predictions.csv"
    out.to_csv(out_path, index=False, encoding="utf-8")
    print(out_path)


def generate(input_name: str, output: str, min_count: int, margin: int) -> None:
    train = pd.read_csv(RAW / "train.csv")
    train["target"] = train["cwe_id"].fillna(NO_VULN)
    train = add_source_id(train)
    intervals = build_intervals(train, list(train.index), min_count=min_count, margin=margin)

    test = load_test()
    sub = pd.read_csv(SUBMISSIONS / input_name)
    if list(sub.columns) == ["name", "label", "cwe_id"]:
        sub = sub.rename(columns={"name": "binary_id"})
    targets = [NO_VULN if row.label == 0 else row.cwe_id for row in sub.itertuples(index=False)]
    targets, changed = apply_intervals(targets, sub["binary_id"].tolist(), intervals)

    out = pd.DataFrame(
        {
            "binary_id": test["binary_id"],
            "label": [0 if target == NO_VULN else 1 for target in targets],
            "cwe_id": ["" if target == NO_VULN else target for target in targets],
        }
    )
    validate_submission(out, test["binary_id"])
    output_path = SUBMISSIONS / output
    out.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(output_path)
    print(f"intervals={len(intervals)} changed={len(changed)}")
    print(f"changed_examples={changed[:40]}")
    print(f"label_dist={out['label'].value_counts().to_dict()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Postprocess CWE predictions using unique source-id CWE intervals.")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--input", default="uninit457_valid_predictions.csv")
    parser.add_argument("--output", default="submission_source_interval_v13_utf8_sig.csv")
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--margin", type=int, default=0)
    args = parser.parse_args()

    ensure_dirs()
    if args.eval:
        evaluate(args.input, args.min_count, args.margin)
    if args.generate:
        generate(args.input, args.output, args.min_count, args.margin)


if __name__ == "__main__":
    main()
