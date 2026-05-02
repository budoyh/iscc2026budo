from __future__ import annotations

import argparse

import pandas as pd
from sklearn.metrics import f1_score

from common import NO_VULN, RAW, REPORT, SUBMISSIONS, ensure_dirs, load_test
from postprocess_submission import top_cwe
from predict import validate_submission


OVERRIDABLE = {"CWE-124", "CWE-126", "CWE-127"}
BUFFER_DIRECT_HINTS = {"CWE-121", "CWE-124", "CWE-127"}


def corrected_target(binary_id: str, target: str) -> str:
    if target == NO_VULN or not target:
        return target
    hint = top_cwe(binary_id)
    if target in OVERRIDABLE and hint in BUFFER_DIRECT_HINTS and hint != target:
        return hint
    return target


def evaluate(input_name: str) -> None:
    path = REPORT / input_name
    frame = pd.read_csv(path)
    old = frame["pred"].tolist()
    new = [corrected_target(row.binary_id, row.pred) for row in frame.itertuples(index=False)]
    changed = useful = harmful = 0
    for true, before, after in zip(frame["true"], old, new, strict=False):
        if before != after:
            changed += 1
            useful += int(after == true and before != true)
            harmful += int(after != true and before == true)
    print(f"base_macro={f1_score(frame['true'], old, average='macro'):.6f}")
    print(f"hint_post_macro={f1_score(frame['true'], new, average='macro'):.6f}")
    print(f"changed={changed} useful={useful} harmful={harmful}")
    out = frame.copy()
    out["pred"] = new
    out_path = REPORT / "targeted_hint_valid_predictions.csv"
    out.to_csv(out_path, index=False, encoding="utf-8")
    errors = out[out["true"] != out["pred"]]
    print("error_pairs")
    print(errors.groupby(["true", "pred"]).size().sort_values(ascending=False).head(50).to_string())
    print(out_path)


def generate(input_name: str, output: str) -> None:
    test = load_test()
    sub = pd.read_csv(SUBMISSIONS / input_name)
    if list(sub.columns) == ["name", "label", "cwe_id"]:
        sub = sub.rename(columns={"name": "binary_id"})
    targets = []
    changed = 0
    for row in sub.itertuples(index=False):
        target = NO_VULN if row.label == 0 else row.cwe_id
        new_target = corrected_target(row.binary_id, target)
        changed += int(new_target != target)
        targets.append(new_target)
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
    print(f"changed={changed}")
    print(f"label_dist={out['label'].value_counts().to_dict()}")
    print(f"top_cwe={out.loc[out['label'] == 1, 'cwe_id'].value_counts().head(20).to_dict()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Targeted CWE hint repair for buffer-family confusions.")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--input", default="torch_group_cnn_valid_predictions.csv")
    parser.add_argument("--output", default="submission_torch_hint_v7_utf8_sig.csv")
    args = parser.parse_args()
    ensure_dirs()
    if args.eval:
        evaluate(args.input)
    if args.generate:
        generate(args.input, args.output)


if __name__ == "__main__":
    main()
