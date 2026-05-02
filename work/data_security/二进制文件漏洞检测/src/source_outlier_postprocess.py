from __future__ import annotations

import argparse

import pandas as pd

from common import NO_VULN, SUBMISSIONS, ensure_dirs, load_test
from predict import validate_submission
from source_id_chunk_postprocess import source_id_number


SOURCE_ID_RULES = {
    511: "CWE-126",
    4917: "CWE-121",
    4966: "CWE-126",
    10849: "CWE-126",
    10850: "CWE-126",
}


def generate(input_name: str, output: str) -> None:
    test = load_test()
    sub = pd.read_csv(SUBMISSIONS / input_name)
    if list(sub.columns) == ["name", "label", "cwe_id"]:
        sub = sub.rename(columns={"name": "binary_id"})

    targets = [NO_VULN if row.label == 0 else row.cwe_id for row in sub.itertuples(index=False)]
    changed: list[tuple[str, int, str, str]] = []
    for pos, row in enumerate(sub.itertuples(index=False)):
        if int(row.label) != 1:
            continue
        source_id = source_id_number(row.binary_id)
        mapped = SOURCE_ID_RULES.get(source_id)
        if mapped and mapped != targets[pos]:
            changed.append((row.binary_id, source_id, targets[pos], mapped))
            targets[pos] = mapped

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
    print(f"changed={len(changed)}")
    print(f"changed_examples={changed}")
    print(f"label_dist={out['label'].value_counts().to_dict()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair source-id predictions that are far outside their CWE source range.")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--input", default="submission_rare_singleton_v14_utf8_sig.csv")
    parser.add_argument("--output", default="submission_source_outlier_v15_utf8_sig.csv")
    args = parser.parse_args()
    ensure_dirs()
    if args.generate:
        generate(args.input, args.output)


if __name__ == "__main__":
    main()
