from __future__ import annotations

import argparse
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

from common import BINARY_DIR, NO_VULN, RAW, REPORT, SUBMISSIONS
from predict import validate_submission
from symbol_rule_predict import predict_one
from train import build_split


ASCII_RE = re.compile(rb"[ -~]{4,180}")
SKIP_PARTS = (
    "gnu c",
    "x86_tune",
    "argument domain error",
    "overflow range error",
    "unknown error",
    "runtime error",
    "api-ms-win",
    "long long",
    "unsigned",
    "short int",
    "wchar_t",
    "printf",
    "wprintf",
    "printline",
    "printintline",
    "entry_bad",
    "entry_good",
    "globalreturn",
    "decodehex",
    "mingw",
    "free",
    "malloc",
    "calloc",
    "getlasterror",
    "freelibrary",
    "configthreadlocale",
)


def strings_for(binary_id: str) -> set[str]:
    raw = (BINARY_DIR / f"{binary_id}.exe").read_bytes()
    result = set()
    for blob in ASCII_RE.findall(raw):
        text = blob.decode("latin1", "ignore").strip().lower()
        if len(text) < 4:
            continue
        if any(part in text for part in SKIP_PARTS):
            continue
        if text.replace("_", "").replace("-", "").isalnum() and len(text) <= 3:
            continue
        result.add(text)
    return result


def build_rules(train: pd.DataFrame, min_support: int, min_purity: float) -> dict[str, tuple[str, float, int]]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for row in train.itertuples(index=False):
        if row.target == NO_VULN:
            continue
        for text in strings_for(row.binary_id):
            counts[text][row.target] += 1
    rules: dict[str, tuple[str, float, int]] = {}
    for text, counter in counts.items():
        total = sum(counter.values())
        if total < min_support:
            continue
        cwe, count = counter.most_common(1)[0]
        purity = count / total
        if purity >= min_purity:
            rules[text] = (cwe, purity, total)
    return rules


def predict_cwe(binary_id: str, rules: dict[str, tuple[str, float, int]]) -> str:
    votes: Counter = Counter()
    for text in strings_for(binary_id):
        if text not in rules:
            continue
        cwe, purity, support = rules[text]
        votes[cwe] += math.log1p(support) * purity
    if not votes:
        return ""
    winner, score = votes.most_common(1)[0]
    if len(votes) > 1 and score < votes.most_common(2)[1][1] * 1.15:
        return ""
    return winner


def evaluate(min_support: int, min_purity: float) -> None:
    train = pd.read_csv(RAW / "train.csv")
    train["target"] = train["cwe_id"].fillna(NO_VULN)
    train_idx, valid_idx = build_split(train, 0.2)
    rules = build_rules(train.loc[train_idx].copy(), min_support, min_purity)
    pred_frame = pd.read_csv(REPORT / "symbol_hybrid_valid_errors.csv")
    pred_map = pred_frame.set_index("binary_id")["pred"].to_dict()
    y_true = []
    y_pred = []
    changed = useful = harmful = covered = 0
    for row in train.loc[valid_idx].itertuples(index=False):
        old = pred_map[row.binary_id]
        new = old
        symbol_label, _, _ = predict_one(row.binary_id)
        if symbol_label == 1:
            rule_cwe = predict_cwe(row.binary_id, rules)
            if rule_cwe:
                covered += 1
                new = rule_cwe
        if new != old:
            changed += 1
            useful += int(new == row.target and old != row.target)
            harmful += int(new != row.target and old == row.target)
        y_true.append(row.target)
        y_pred.append(new)
    print(f"rules={len(rules)} covered={covered} changed={changed} useful={useful} harmful={harmful}")
    print(f"string_rule_macro={f1_score(y_true, y_pred, average='macro'):.6f}")
    out = pd.DataFrame({"binary_id": train.loc[valid_idx, "binary_id"].tolist(), "true": y_true, "pred": y_pred})
    out_path = REPORT / "string_rule_valid_predictions.csv"
    out.to_csv(out_path, index=False, encoding="utf-8")
    errors = out[out["true"] != out["pred"]]
    print("error_pairs")
    print(errors.groupby(["true", "pred"]).size().sort_values(ascending=False).head(40).to_string())
    print("true_error_counts")
    print(errors["true"].value_counts().head(40).to_string())
    print(out_path)


def generate(input_name: str, output: str, min_support: int, min_purity: float) -> None:
    train = pd.read_csv(RAW / "train.csv")
    train["target"] = train["cwe_id"].fillna(NO_VULN)
    test = pd.read_csv(RAW / "test.csv")
    rules = build_rules(train, min_support, min_purity)
    input_path = SUBMISSIONS / input_name
    sub = pd.read_csv(input_path)
    if list(sub.columns) == ["name", "label", "cwe_id"]:
        sub = sub.rename(columns={"name": "binary_id"})
    rows = []
    changed = 0
    for row in sub.itertuples(index=False):
        cwe = "" if row.label == 0 else row.cwe_id
        symbol_label, _, _ = predict_one(row.binary_id)
        if symbol_label == 1:
            rule_cwe = predict_cwe(row.binary_id, rules)
            if rule_cwe and rule_cwe != cwe:
                cwe = rule_cwe
                changed += 1
        label = 0 if symbol_label == 0 else 1
        rows.append((row.binary_id, label, "" if label == 0 else cwe))
    out = pd.DataFrame(rows, columns=["binary_id", "label", "cwe_id"])
    validate_submission(out, test["binary_id"])
    output_path = SUBMISSIONS / output
    out.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(output_path)
    print(f"rules={len(rules)} changed={changed}")
    print(f"label_dist={out['label'].value_counts().to_dict()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="High-purity printable string CWE rules.")
    parser.add_argument("--min-support", type=int, default=3)
    parser.add_argument("--min-purity", type=float, default=0.98)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--input", default="submission_symbol_hybrid_hash_hint_utf8_sig.csv")
    parser.add_argument("--output", default="submission_string_rule_utf8_sig.csv")
    args = parser.parse_args()
    if args.eval:
        evaluate(args.min_support, args.min_purity)
    if args.generate:
        generate(args.input, args.output, args.min_support, args.min_purity)


if __name__ == "__main__":
    main()
