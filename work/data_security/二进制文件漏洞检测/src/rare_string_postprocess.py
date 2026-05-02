from __future__ import annotations

import argparse

import pandas as pd
from sklearn.metrics import f1_score

from common import BINARY_DIR, NO_VULN, REPORT, SUBMISSIONS, ensure_dirs, load_test
from predict import validate_submission


def binary_text(binary_id: str) -> str:
    return (BINARY_DIR / f"{binary_id}.exe").read_bytes().decode("latin1", "ignore").lower()


def corrected_target(binary_id: str, target: str) -> str:
    if target == NO_VULN or not target:
        return target
    text = binary_text(binary_id)
    if target == "CWE-685" and "intfive" in text:
        return "CWE-688"
    if target == "CWE-398" and "does not use the parameter variable" in text:
        return "CWE-563"
    return target


def evaluate(input_name: str) -> None:
    frame = pd.read_csv(REPORT / input_name)
    old = frame["pred"].tolist()
    new = [corrected_target(row.binary_id, row.pred) for row in frame.itertuples(index=False)]
    changed = useful = harmful = 0
    for true, before, after in zip(frame["true"], old, new, strict=False):
        if before != after:
            changed += 1
            useful += int(after == true and before != true)
            harmful += int(after != true and before == true)
    print(f"base_macro={f1_score(frame['true'], old, average='macro'):.6f}")
    print(f"rare_string_macro={f1_score(frame['true'], new, average='macro'):.6f}")
    print(f"changed={changed} useful={useful} harmful={harmful}")
    out = frame.copy()
    out["pred"] = new
    out_path = REPORT / "rare_string_valid_predictions.csv"
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
    targets = [NO_VULN if row.label == 0 else row.cwe_id for row in sub.itertuples(index=False)]
    changed = 0
    for idx, row in enumerate(sub.itertuples(index=False)):
        new_target = corrected_target(row.binary_id, targets[idx])
        changed += int(new_target != targets[idx])
        targets[idx] = new_target
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
    parser = argparse.ArgumentParser(description="Rare high-purity string fixes.")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--input", default="loop835_valid_predictions.csv")
    parser.add_argument("--output", default="submission_rare_string_v10_utf8_sig.csv")
    args = parser.parse_args()
    ensure_dirs()
    if args.eval:
        evaluate(args.input)
    if args.generate:
        generate(args.input, args.output)


if __name__ == "__main__":
    main()
