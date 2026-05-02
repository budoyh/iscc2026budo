from __future__ import annotations

import argparse
import re
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from tqdm import tqdm

from common import RAW, REPORT, SUBMISSIONS, ensure_dirs, load_test
from entry_body_cwe import choose_bad_symbol
from predict import validate_submission


CWE_RE = re.compile(r"CWE[_-]?0*([0-9]{2,4})", re.IGNORECASE)


def normalize_cwe(value: str) -> str:
    match = CWE_RE.search(value or "")
    if not match:
        return ""
    return f"CWE-{int(match.group(1)):03d}"


def one_entry_symbol(binary_id: str) -> tuple[str, str, str]:
    symbol = choose_bad_symbol(binary_id) or ""
    return binary_id, normalize_cwe(symbol), symbol


def collect_entry_cwes(binary_ids: list[str], workers: int) -> pd.DataFrame:
    if workers <= 1:
        rows = [one_entry_symbol(binary_id) for binary_id in tqdm(binary_ids, desc="entry-cwe", ncols=100)]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(
                tqdm(
                    executor.map(one_entry_symbol, binary_ids, chunksize=64),
                    total=len(binary_ids),
                    desc="entry-cwe",
                    ncols=100,
                )
            )
    return pd.DataFrame(rows, columns=["binary_id", "entry_cwe", "entry_symbol"])


def check_train(workers: int) -> None:
    train = pd.read_csv(RAW / "train.csv")
    pos = train[train["label"] == 1][["binary_id", "cwe_id"]].reset_index(drop=True)
    entries = collect_entry_cwes(pos["binary_id"].tolist(), workers)
    out = pos.merge(entries, on="binary_id", how="left", validate="one_to_one")
    with_entry = out[out["entry_cwe"] != ""]
    mismatch = with_entry[with_entry["entry_cwe"] != with_entry["cwe_id"]]
    REPORT.mkdir(parents=True, exist_ok=True)
    out.to_csv(REPORT / "train_entry_symbol_cwe_check.csv", index=False, encoding="utf-8")
    print(f"train_pos={len(out)} with_entry={len(with_entry)} mismatch={len(mismatch)}")
    if len(with_entry):
        print(f"entry_acc={float((with_entry['entry_cwe'] == with_entry['cwe_id']).mean()):.12f}")
    if len(mismatch):
        print(mismatch.head(80).to_string(index=False))


def check_submission(input_name: str, workers: int) -> pd.DataFrame:
    sub = pd.read_csv(SUBMISSIONS / input_name, keep_default_na=False)
    if list(sub.columns) == ["name", "label", "cwe_id"]:
        sub = sub.rename(columns={"name": "binary_id"})
    pos = sub[sub["label"] == 1][["binary_id", "cwe_id"]].reset_index(drop=True)
    entries = collect_entry_cwes(pos["binary_id"].tolist(), workers)
    out = pos.merge(entries, on="binary_id", how="left", validate="one_to_one")
    diff = out[(out["entry_cwe"] != "") & (out["entry_cwe"] != out["cwe_id"])]
    REPORT.mkdir(parents=True, exist_ok=True)
    out.to_csv(REPORT / "test_entry_symbol_cwe_check.csv", index=False, encoding="utf-8")
    print(f"test_pos={len(out)} with_entry={(out['entry_cwe'] != '').sum()} diff={len(diff)}")
    if len(diff):
        print(diff.to_string(index=False))
    return diff


def generate(input_name: str, output: str, workers: int) -> None:
    test = load_test()
    sub = pd.read_csv(SUBMISSIONS / input_name, keep_default_na=False)
    if list(sub.columns) == ["name", "label", "cwe_id"]:
        sub = sub.rename(columns={"name": "binary_id"})
    entries = collect_entry_cwes(sub.loc[sub["label"] == 1, "binary_id"].tolist(), workers)
    entry_map = dict(zip(entries["binary_id"], entries["entry_cwe"], strict=False))
    out = sub.copy()
    changed = []
    for idx, row in out.iterrows():
        if int(row["label"]) != 1:
            continue
        mapped = entry_map.get(row["binary_id"], "")
        if mapped and mapped != row["cwe_id"]:
            changed.append((row["binary_id"], row["cwe_id"], mapped))
            out.at[idx, "cwe_id"] = mapped
    out = out[["binary_id", "label", "cwe_id"]]
    validate_submission(out, test["binary_id"])
    output_path = SUBMISSIONS / output
    out.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(output_path)
    print(f"changed={len(changed)}")
    if changed:
        print(f"changed_examples={changed[:80]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Use chosen entry_bad symbol CWE when it is explicit.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--check-train", action="store_true")
    parser.add_argument("--check-sub", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--input", default="s15.csv")
    parser.add_argument("--output", default="s16.csv")
    args = parser.parse_args()
    ensure_dirs()
    if args.check_train:
        check_train(args.workers)
    if args.check_sub:
        check_submission(args.input, args.workers)
    if args.generate:
        generate(args.input, args.output, args.workers)


if __name__ == "__main__":
    main()
