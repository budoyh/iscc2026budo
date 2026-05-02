from __future__ import annotations

import argparse
from functools import lru_cache

import numpy as np
import pandas as pd
import pefile
from capstone import CS_ARCH_X86, CS_MODE_32, CS_MODE_64, Cs
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import BINARY_DIR, NO_VULN, RAW, REPORT, SUBMISSIONS, ensure_dirs, load_test
from enhanced_cwe_calibrator import GROUPS, bad_function_payload
from predict import validate_submission
from train import build_split


RETVAL_CLASSES = set(GROUPS["retval"])
STR_FEATURES = [
    "fread",
    "fwrite",
    "remove",
    "fopen",
    "recv",
    "socket",
    "file",
    "globalreturnstrueorfalse",
    "globalreturnsfalse",
    "globalreturnstrue",
    "failed",
    "removeme",
]
OPS = ["test", "cmp", "je", "jne", "js", "jle", "jmp", "call", "mov", "lea"]


@lru_cache(maxsize=None)
def semantic_features(binary_id: str) -> np.ndarray:
    raw = (BINARY_DIR / f"{binary_id}.exe").read_bytes()
    pe = pefile.PE(data=raw, fast_load=True)
    body, start_va, _addr_to_names = bad_function_payload(binary_id, raw, pe)
    mode = CS_MODE_64 if pe.FILE_HEADER.Machine == 0x8664 else CS_MODE_32
    md = Cs(CS_ARCH_X86, mode)
    insns = list(md.disasm(body, start_va))
    mnemonics = [insn.mnemonic.lower() for insn in insns]
    operands = [insn.op_str.lower() for insn in insns]

    values: list[float] = [
        float(len(body)),
        float(len(insns)),
        float(sum(mnemonic.startswith("call") for mnemonic in mnemonics)),
        float(sum(mnemonic.startswith("j") for mnemonic in mnemonics)),
    ]
    checked = cmped = tested = branch_after = ignored = ret_stored = zero_cmp = neg_cmp = big_cmp = 0
    for idx, mnemonic in enumerate(mnemonics):
        if not mnemonic.startswith("call"):
            continue
        saw_check = False
        for next_mnemonic, operand in zip(mnemonics[idx + 1 : idx + 7], operands[idx + 1 : idx + 7], strict=False):
            mentions_ret = "rax" in operand or "eax" in operand or "al" in operand
            if mentions_ret and next_mnemonic in {"test", "cmp"}:
                saw_check = True
                checked += 1
                tested += int(next_mnemonic == "test")
                cmped += int(next_mnemonic == "cmp")
                zero_cmp += int(", 0" in operand or ",0" in operand)
                neg_cmp += int("0xffffffff" in operand or "-1" in operand)
                big_cmp += int("0x63" in operand or "0x64" in operand or "99" in operand or "100" in operand)
            if saw_check and next_mnemonic.startswith("j"):
                branch_after += 1
                break
            if next_mnemonic == "mov" and mentions_ret and "[" in operand:
                ret_stored += 1
        if not saw_check:
            ignored += 1
    values.extend([checked, cmped, tested, branch_after, ignored, ret_stored, zero_cmp, neg_cmp, big_cmp])

    text = raw.decode("latin1", "ignore").lower()
    values.extend(float(item in text) for item in STR_FEATURES)
    values.extend(float(sum(mnemonic == op for mnemonic in mnemonics)) for op in OPS)
    return np.array(values, dtype=np.float32)


def fit_model(train: pd.DataFrame, indices: list[int], c: float) -> object:
    labels = train["target"].to_numpy()
    train_idx = [idx for idx in indices if labels[idx] in RETVAL_CLASSES]
    x_train = np.vstack([semantic_features(train.at[idx, "binary_id"]) for idx in train_idx])
    y_train = labels[train_idx]
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced", C=c),
    )
    model.fit(x_train, y_train)
    return model


def evaluate(input_name: str, c: float, threshold: float) -> None:
    train = pd.read_csv(RAW / "train.csv")
    train["target"] = train["cwe_id"].fillna(NO_VULN)
    train_idx, valid_idx = build_split(train, 0.2)
    model = fit_model(train, list(train_idx), c)
    frame = pd.read_csv(REPORT / input_name)
    positions = [pos for pos, pred in enumerate(frame["pred"]) if pred in RETVAL_CLASSES]
    sample_idx = [valid_idx[pos] for pos in positions]
    x_valid = np.vstack([semantic_features(train.at[idx, "binary_id"]) for idx in sample_idx])
    probs = model.predict_proba(x_valid)
    pred = model.classes_[np.argmax(probs, axis=1)]
    conf = np.max(probs, axis=1)
    targets = frame["pred"].tolist()
    changed = useful = harmful = 0
    for pos, new_target, score in zip(positions, pred, conf, strict=False):
        if float(score) < threshold or new_target == targets[pos]:
            continue
        old = targets[pos]
        true = frame.at[pos, "true"]
        targets[pos] = str(new_target)
        changed += 1
        useful += int(new_target == true and old != true)
        harmful += int(new_target != true and old == true)
    print(f"base_macro={f1_score(frame['true'], frame['pred'], average='macro'):.6f}")
    print(f"retval_semantic_macro={f1_score(frame['true'], targets, average='macro'):.6f}")
    print(f"changed={changed} useful={useful} harmful={harmful}")
    out = frame.copy()
    out["pred"] = targets
    out_path = REPORT / "retval_semantic_valid_predictions.csv"
    out.to_csv(out_path, index=False, encoding="utf-8")
    print(out_path)


def generate(input_name: str, output: str, c: float, threshold: float) -> None:
    train = pd.read_csv(RAW / "train.csv")
    train["target"] = train["cwe_id"].fillna(NO_VULN)
    test = load_test()
    model = fit_model(train, list(train.index), c)
    sub = pd.read_csv(SUBMISSIONS / input_name)
    if list(sub.columns) == ["name", "label", "cwe_id"]:
        sub = sub.rename(columns={"name": "binary_id"})
    targets = [NO_VULN if row.label == 0 else row.cwe_id for row in sub.itertuples(index=False)]
    positions = [pos for pos, target in enumerate(targets) if target in RETVAL_CLASSES]
    x_test = np.vstack([semantic_features(sub.at[pos, "binary_id"]) for pos in positions])
    probs = model.predict_proba(x_test)
    pred = model.classes_[np.argmax(probs, axis=1)]
    conf = np.max(probs, axis=1)
    changed = 0
    for pos, new_target, score in zip(positions, pred, conf, strict=False):
        if float(score) >= threshold and new_target != targets[pos]:
            targets[pos] = str(new_target)
            changed += 1
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic retval CWE postprocess.")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--input", default="targeted_hint_valid_predictions.csv")
    parser.add_argument("--output", default="submission_retval_semantic_v8_utf8_sig.csv")
    parser.add_argument("--c", type=float, default=3.0)
    parser.add_argument("--threshold", type=float, default=0.99)
    args = parser.parse_args()
    ensure_dirs()
    if args.eval:
        evaluate(args.input, args.c, args.threshold)
    if args.generate:
        generate(args.input, args.output, args.c, args.threshold)


if __name__ == "__main__":
    main()
