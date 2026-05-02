from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import pefile

from common import BINARY_DIR, RAW


ASCII_RE = re.compile(rb"[ -~]{4,300}")
CWE_RE = re.compile(r"CWE[-_ ]?(\d{2,4})", re.IGNORECASE)


def file_bytes(binary_id: str) -> bytes:
    return (BINARY_DIR / f"{binary_id}.exe").read_bytes()


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def normalize_pe(raw: bytes) -> bytes:
    data = bytearray(raw)
    try:
        pe = pefile.PE(data=raw, fast_load=True)
        pe_offset = pe.DOS_HEADER.e_lfanew
        timestamp_offset = pe_offset + 8
        data[timestamp_offset : timestamp_offset + 4] = b"\x00" * 4
        checksum_offset = pe.OPTIONAL_HEADER.get_file_offset() + 64
        data[checksum_offset : checksum_offset + 4] = b"\x00" * 4
    except Exception:
        pass
    return bytes(data)


def text_hash(raw: bytes) -> str:
    try:
        pe = pefile.PE(data=raw, fast_load=True)
        chunks = []
        for section in pe.sections:
            name = section.Name.rstrip(b"\x00").decode("latin1", "ignore").lower()
            if name in {".text", "text"} or section.Characteristics & 0x20000000:
                chunks.append(section.get_data())
        return sha256(b"".join(chunks))
    except Exception:
        return ""


def imphash(raw: bytes) -> str:
    try:
        return pefile.PE(data=raw, fast_load=False).get_imphash()
    except Exception:
        return ""


def cwe_hits(raw: bytes) -> list[str]:
    hits = []
    for item in ASCII_RE.findall(raw):
        text = item.decode("latin1", "ignore")
        for match in CWE_RE.finditer(text):
            hits.append(f"CWE-{int(match.group(1)):03d}")
    return hits


def compare_signature(name: str, train_ids: pd.Series, test_ids: pd.Series, func) -> None:
    train_map: dict[str, list[str]] = defaultdict(list)
    for binary_id in train_ids:
        train_map[func(file_bytes(binary_id))].append(binary_id)
    matches = []
    ambiguous = 0
    for binary_id in test_ids:
        sig = func(file_bytes(binary_id))
        if sig in train_map:
            ids = train_map[sig]
            matches.append((binary_id, ids[:5], len(ids)))
            if len(ids) > 1:
                ambiguous += 1
    print(f"{name}_matches={len(matches)} ambiguous={ambiguous}")
    print(f"{name}_examples={matches[:8]}")


def main() -> None:
    train = pd.read_csv(RAW / "train.csv")
    test = pd.read_csv(RAW / "test.csv")
    train_ids = train["binary_id"]
    test_ids = test["binary_id"]

    compare_signature("sha256", train_ids, test_ids, sha256)
    compare_signature("normalized_sha256", train_ids, test_ids, lambda raw: sha256(normalize_pe(raw)))
    compare_signature("text_hash", train_ids, test_ids, text_hash)
    compare_signature("imphash", train_ids, test_ids, imphash)

    train_hit_counter = Counter()
    train_hit_correct = Counter()
    train_hit_total = 0
    for row in train.itertuples(index=False):
        hits = cwe_hits(file_bytes(row.binary_id))
        if hits:
            train_hit_total += 1
            top = Counter(hits).most_common(1)[0][0]
            train_hit_counter[top] += 1
            if row.label == 1 and top == row.cwe_id:
                train_hit_correct[top] += 1
            if row.label == 0:
                train_hit_correct["false_positive_no_vuln"] += 1

    test_hit_counter = Counter()
    test_hit_total = 0
    for binary_id in test_ids:
        hits = cwe_hits(file_bytes(binary_id))
        if hits:
            test_hit_total += 1
            test_hit_counter[Counter(hits).most_common(1)[0][0]] += 1

    print(f"train_cwe_string_hits={train_hit_total}")
    print(f"train_cwe_string_top={train_hit_counter.most_common(20)}")
    print(f"train_cwe_string_correct_top={train_hit_correct.most_common(20)}")
    print(f"test_cwe_string_hits={test_hit_total}")
    print(f"test_cwe_string_top={test_hit_counter.most_common(20)}")


if __name__ == "__main__":
    main()
