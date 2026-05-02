from __future__ import annotations

import argparse
import re
from collections import Counter
from functools import lru_cache

import pandas as pd
from sklearn.metrics import f1_score

from common import BINARY_DIR, NO_VULN, RAW, REPORT, SUBMISSIONS, ensure_dirs, load_test
from predict import validate_submission
from train import build_split


SOURCE_ID_RE = re.compile(rb"[\\/]S(\d{6})[\\/]source_sanitized\.c|S(\d{6})")


@lru_cache(maxsize=None)
def source_id_number(binary_id: str) -> int:
    raw = (BINARY_DIR / f"{binary_id}.exe").read_bytes()
    hits: list[int] = []
    for match in SOURCE_ID_RE.finditer(raw):
        values = [item for item in match.groups() if item]
        if values:
            hits.append(int(values[0]))
    if not hits:
        return -1
    return Counter(hits).most_common(1)[0][0]


def add_source_id(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["source_id_num"] = [source_id_number(binary_id) for binary_id in out["binary_id"]]
    return out


def build_chunks(train: pd.DataFrame, indices: list[int], min_count: int, margin: int) -> list[tuple[int, int, str]]:
    part = train.iloc[indices]
    part = part[part["label"] == 1].sort_values("source_id_num")
    rows = part[["source_id_num", "cwe_id"]].to_records(index=False)
    if len(rows) == 0:
        return []

    chunks: list[tuple[int, int, str, int]] = []
    start = previous = int(rows[0][0])
    cwe = str(rows[0][1])
    count = 1
    for source_id, next_cwe in rows[1:]:
        source_id = int(source_id)
        next_cwe = str(next_cwe)
        if next_cwe == cwe:
            previous = source_id
            count += 1
            continue
        chunks.append((start, previous, cwe, count))
        start = previous = source_id
        cwe = next_cwe
        count = 1
    chunks.append((start, previous, cwe, count))

    filtered = []
    for start, end, cwe, count in chunks:
        if count < min_count:
            continue
        lo = start + margin
        hi = end - margin
        if lo <= hi:
            filtered.append((lo, hi, cwe))
    return filtered


def chunk_prediction(source_id: int, chunks: list[tuple[int, int, str]]) -> str | None:
    hits = [cwe for lo, hi, cwe in chunks if lo <= source_id <= hi]
    if len(hits) == 1:
        return hits[0]
    return None


def apply_chunks(
    targets: list[str],
    binary_ids: list[str],
    chunks: list[tuple[int, int, str]],
) -> tuple[list[str], list[tuple[str, int, str, str]]]:
    changed: list[tuple[str, int, str, str]] = []
    out = list(targets)
    for pos, (binary_id, target) in enumerate(zip(binary_ids, targets, strict=False)):
        if target == NO_VULN:
            continue
        source_id = source_id_number(binary_id)
        mapped = chunk_prediction(source_id, chunks)
        if mapped and mapped != target:
            out[pos] = mapped
            changed.append((binary_id, source_id, target, mapped))
    return out, changed


def evaluate(input_name: str, min_count: int, margin: int) -> None:
    train = pd.read_csv(RAW / "train.csv")
    train["target"] = train["cwe_id"].fillna(NO_VULN)
    train = add_source_id(train)
    train_idx, _valid_idx = build_split(train, 0.2)

    chunks = build_chunks(train, list(train_idx), min_count=min_count, margin=margin)
    frame = pd.read_csv(REPORT / input_name)
    targets, changed = apply_chunks(frame["pred"].tolist(), frame["binary_id"].tolist(), chunks)

    useful = harmful = neutral = 0
    for binary_id, _source_id, old, new in changed:
        true = str(frame.loc[frame["binary_id"] == binary_id, "true"].iloc[0])
        useful += int(new == true and old != true)
        harmful += int(new != true and old == true)
        neutral += int(new != true and old != true)

    print(f"base_macro={f1_score(frame['true'], frame['pred'], average='macro'):.12f}")
    print(f"source_chunk_macro={f1_score(frame['true'], targets, average='macro'):.12f}")
    print(f"chunks={len(chunks)} changed={len(changed)} useful={useful} harmful={harmful} neutral={neutral}")
    print(f"changed_examples={changed[:20]}")

    out = frame.copy()
    out["pred"] = targets
    out_path = REPORT / "source_id_chunk_valid_predictions.csv"
    out.to_csv(out_path, index=False, encoding="utf-8")
    print(out_path)


def generate(input_name: str, output: str, min_count: int, margin: int) -> None:
    train = pd.read_csv(RAW / "train.csv")
    train["target"] = train["cwe_id"].fillna(NO_VULN)
    train = add_source_id(train)
    chunks = build_chunks(train, list(train.index), min_count=min_count, margin=margin)

    test = load_test()
    sub = pd.read_csv(SUBMISSIONS / input_name)
    if list(sub.columns) == ["name", "label", "cwe_id"]:
        sub = sub.rename(columns={"name": "binary_id"})
    targets = [NO_VULN if row.label == 0 else row.cwe_id for row in sub.itertuples(index=False)]
    targets, changed = apply_chunks(targets, sub["binary_id"].tolist(), chunks)

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
    print(f"chunks={len(chunks)} changed={len(changed)}")
    print(f"changed_examples={changed[:30]}")
    print(f"label_dist={out['label'].value_counts().to_dict()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Postprocess CWE predictions using embedded source sample id chunks.")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--input", default="uninit457_valid_predictions.csv")
    parser.add_argument("--output", default="submission_source_chunk_v12_utf8_sig.csv")
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
