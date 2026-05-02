from __future__ import annotations

import argparse

import pandas as pd
import pefile
from capstone import CS_ARCH_X86, CS_MODE_32, CS_MODE_64, Cs
from sklearn.metrics import f1_score

from common import BINARY_DIR, NO_VULN, REPORT, SUBMISSIONS, ensure_dirs, load_test
from enhanced_cwe_calibrator import bad_function_payload
from predict import validate_submission


def has_unconditional_backward_jmp(binary_id: str) -> bool:
    raw = (BINARY_DIR / f"{binary_id}.exe").read_bytes()
    pe = pefile.PE(data=raw, fast_load=True)
    body, start_va, _addr_to_names = bad_function_payload(binary_id, raw, pe)
    mode = CS_MODE_64 if pe.FILE_HEADER.Machine == 0x8664 else CS_MODE_32
    md = Cs(CS_ARCH_X86, mode)
    for insn in md.disasm(body, start_va):
        if insn.mnemonic.lower() != "jmp":
            continue
        try:
            target = int(insn.op_str, 16)
        except ValueError:
            continue
        if target < insn.address:
            return True
    return False


def corrected_target(binary_id: str, target: str) -> str:
    if target == "CWE-546" and has_unconditional_backward_jmp(binary_id):
        return "CWE-835"
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
    print(f"loop835_macro={f1_score(frame['true'], new, average='macro'):.6f}")
    print(f"changed={changed} useful={useful} harmful={harmful}")
    out = frame.copy()
    out["pred"] = new
    out_path = REPORT / "loop835_valid_predictions.csv"
    out.to_csv(out_path, index=False, encoding="utf-8")
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
    parser = argparse.ArgumentParser(description="Repair CWE-835 infinite-loop samples.")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--input", default="retval_semantic_valid_predictions.csv")
    parser.add_argument("--output", default="submission_loop835_v9_utf8_sig.csv")
    args = parser.parse_args()
    ensure_dirs()
    if args.eval:
        evaluate(args.input)
    if args.generate:
        generate(args.input, args.output)


if __name__ == "__main__":
    main()
