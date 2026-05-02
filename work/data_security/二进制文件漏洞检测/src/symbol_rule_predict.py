from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd
import pefile
from capstone import CS_ARCH_X86, CS_MODE_32, CS_MODE_64, Cs
from capstone.x86 import X86_OP_IMM

from common import BINARY_DIR, NO_VULN, RAW, SUBMISSIONS, ensure_dirs
from inspect_symbols import symbols
from postprocess_submission import top_cwe
from predict import validate_submission


IGNORE_CALL_EXACT = {
    "__main",
    "_pei386_runtime_relocator",
    "_onexit",
    "atexit",
    "printf",
    "puts",
    "time",
    "srand",
    "srand64",
}
IGNORE_CALL_PREFIXES = (
    "__main",
    "__mingw",
    "__security",
    "__tmain",
    "maincrtstartup",
    "winmaincrtstartup",
)


def section_for_symbol(pe: pefile.PE, section_number: int):
    if section_number <= 0 or section_number > len(pe.sections):
        return None
    return pe.sections[section_number - 1]


def symbol_va(pe: pefile.PE, section_number: int, value: int) -> int:
    section = section_for_symbol(pe, section_number)
    if section is None:
        return pe.OPTIONAL_HEADER.ImageBase + value
    return pe.OPTIONAL_HEADER.ImageBase + section.VirtualAddress + value


def is_ignored_call(name: str) -> bool:
    lowered = name.lower()
    if lowered in {".text", ""}:
        return True
    if lowered in IGNORE_CALL_EXACT:
        return True
    return any(lowered.startswith(prefix) for prefix in IGNORE_CALL_PREFIXES)


def called_symbols_from_main(binary_id: str) -> list[str]:
    path = BINARY_DIR / f"{binary_id}.exe"
    raw = path.read_bytes()
    pe = pefile.PE(data=raw, fast_load=True)
    rows = symbols(binary_id)
    code_symbols = [(name, value, sec) for name, value, sec, storage in rows if sec > 0]
    addr_to_names: dict[int, list[str]] = {}
    for name, value, sec in code_symbols:
        addr_to_names.setdefault(symbol_va(pe, sec, value), []).append(name)

    main_candidates = [(name, value, sec) for name, value, sec, storage in rows if name == "main" and sec > 0]
    if not main_candidates:
        return []
    _, main_value, main_sec = main_candidates[0]
    section = section_for_symbol(pe, main_sec)
    if section is None:
        return []

    same_section_values = sorted(value for _, value, sec in code_symbols if sec == main_sec and value > main_value)
    end_value = same_section_values[0] if same_section_values else main_value + 256
    code = section.get_data()[main_value:end_value]
    mode = CS_MODE_64 if pe.FILE_HEADER.Machine == 0x8664 else CS_MODE_32
    md = Cs(CS_ARCH_X86, mode)
    md.detail = True

    calls: list[str] = []
    start_va = pe.OPTIONAL_HEADER.ImageBase + section.VirtualAddress + main_value
    for insn in md.disasm(code, start_va):
        if not insn.mnemonic.startswith("call"):
            continue
        if not insn.operands or insn.operands[0].type != X86_OP_IMM:
            continue
        target = insn.operands[0].imm
        for name in addr_to_names.get(target, []):
            if not is_ignored_call(name):
                calls.append(name)
    return calls


def label_from_calls(calls: list[str]) -> int | None:
    text = " ".join(calls).lower()
    if "entry_bad" in text:
        return 1
    if "entry_good" in text:
        return 0
    for name in calls:
        lowered = name.lower()
        if "bad" in lowered and "good" not in lowered:
            return 1
        if "good" in lowered and "bad" not in lowered:
            return 0
    return None


def predict_one(binary_id: str) -> tuple[int | None, str, list[str]]:
    calls = called_symbols_from_main(binary_id)
    label = label_from_calls(calls)
    cwe = top_cwe(binary_id) if label == 1 else ""
    return label, cwe, calls


def evaluate_train() -> None:
    train = pd.read_csv(RAW / "train.csv")
    covered = 0
    label_correct = 0
    cwe_correct = 0
    failures = []
    call_counter = Counter()
    for row in train.itertuples(index=False):
        label, cwe, calls = predict_one(row.binary_id)
        for call in calls:
            call_counter[call] += 1
        if label is None:
            failures.append((row.binary_id, row.label, row.cwe_id, calls))
            continue
        covered += 1
        label_correct += int(label == row.label)
        if row.label == 1:
            cwe_correct += int(cwe == row.cwe_id)
            if cwe != row.cwe_id and len(failures) < 12:
                failures.append((row.binary_id, row.label, row.cwe_id, cwe, calls))
    positives = int(train["label"].sum())
    print(f"covered={covered}/{len(train)}")
    print(f"label_accuracy_on_covered={label_correct}/{covered}={label_correct / max(covered, 1):.6f}")
    print(f"positive_cwe_accuracy={cwe_correct}/{positives}={cwe_correct / max(positives, 1):.6f}")
    print(f"common_calls={call_counter.most_common(20)}")
    print(f"failures={failures[:12]}")


def generate(output: str) -> None:
    test = pd.read_csv(RAW / "test.csv")
    rows = []
    unresolved = []
    for binary_id in test["binary_id"]:
        label, cwe, calls = predict_one(binary_id)
        if label is None:
            unresolved.append((binary_id, calls))
            label = 0
            cwe = ""
        rows.append((binary_id, label, cwe if label == 1 else ""))
    out = pd.DataFrame(rows, columns=["binary_id", "label", "cwe_id"])
    validate_submission(out, test["binary_id"])
    output_path = SUBMISSIONS / output
    out.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(output_path)
    print(f"unresolved={len(unresolved)}")
    print(f"unresolved_examples={unresolved[:10]}")
    print(f"label_dist={out['label'].value_counts().to_dict()}")
    print(f"top_cwe={out.loc[out['label'] == 1, 'cwe_id'].value_counts().head(15).to_dict()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict labels from COFF symbols and main call target.")
    parser.add_argument("--evaluate-train", action="store_true")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    ensure_dirs()
    if args.evaluate_train:
        evaluate_train()
    if args.output:
        generate(args.output)


if __name__ == "__main__":
    main()
