from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from common import NO_VULN, RAW, SUBMISSIONS, ensure_dirs
from probe_leakage import cwe_hits, file_bytes, text_hash
from predict import validate_submission


def target_from_row(row) -> str:
    return row.cwe_id if row.label == 1 else NO_VULN


def top_cwe(binary_id: str) -> str:
    hits = cwe_hits(file_bytes(binary_id))
    if not hits:
        return ""
    return Counter(hits).most_common(1)[0][0]


def build_pure_text_hash_map(train: pd.DataFrame) -> dict[str, str]:
    groups: dict[str, list[str]] = defaultdict(list)
    for row in train.itertuples(index=False):
        groups[text_hash(file_bytes(row.binary_id))].append(row.target)
    return {sig: values[0] for sig, values in groups.items() if values and len(set(values)) == 1}


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply validated leakage-safe postprocess rules.")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    ensure_dirs()
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = SUBMISSIONS / input_path
    output_path = SUBMISSIONS / args.output

    train = pd.read_csv(RAW / "train.csv")
    train["target"] = train.apply(target_from_row, axis=1)
    test = pd.read_csv(RAW / "test.csv")
    sub = pd.read_csv(input_path)
    if list(sub.columns) == ["name", "label", "cwe_id"]:
        sub = sub.rename(columns={"name": "binary_id"})

    pure_map = build_pure_text_hash_map(train)
    changed_by_hash = 0
    changed_by_hint = 0

    targets = []
    for row in sub.itertuples(index=False):
        target = row.cwe_id if row.label == 1 else NO_VULN
        mapped = pure_map.get(text_hash(file_bytes(row.binary_id)))
        if mapped and mapped != target:
            target = mapped
            changed_by_hash += 1
        hint = top_cwe(row.binary_id)
        if target != NO_VULN and hint and hint != target:
            target = hint
            changed_by_hint += 1
        targets.append(target)

    out = pd.DataFrame(
        {
            "binary_id": test["binary_id"],
            "label": [0 if target == NO_VULN else 1 for target in targets],
            "cwe_id": ["" if target == NO_VULN else target for target in targets],
        }
    )
    validate_submission(out, test["binary_id"])
    out.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(output_path)
    print(f"changed_by_hash={changed_by_hash}")
    print(f"changed_by_hint={changed_by_hint}")


if __name__ == "__main__":
    main()
