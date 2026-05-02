from __future__ import annotations

import struct
import sys

import pefile

from common import BINARY_DIR


def read_symbol_name(raw: bytes, offset: int, string_table: bytes) -> str:
    first = raw[offset : offset + 8]
    zeroes, str_offset = struct.unpack_from("<II", first)
    if zeroes == 0 and str_offset:
        start = str_offset
        end = string_table.find(b"\x00", start)
        if end == -1:
            end = len(string_table)
        return string_table[start:end].decode("latin1", "ignore")
    return first.split(b"\x00", 1)[0].decode("latin1", "ignore")


def symbols(binary_id: str) -> list[tuple[str, int, int, int]]:
    path = BINARY_DIR / f"{binary_id}.exe"
    raw = path.read_bytes()
    pe = pefile.PE(data=raw, fast_load=True)
    ptr = pe.FILE_HEADER.PointerToSymbolTable
    count = pe.FILE_HEADER.NumberOfSymbols
    string_start = ptr + count * 18
    string_table = raw[string_start:]
    result = []
    idx = 0
    while idx < count:
        offset = ptr + idx * 18
        name = read_symbol_name(raw, offset, string_table)
        value, section_number, typ, storage, aux_count = struct.unpack_from("<IhHBB", raw, offset + 8)
        result.append((name, value, section_number, storage))
        idx += 1 + aux_count
    return result


def main() -> None:
    ids = sys.argv[1:] or ["BIN_1000013", "BIN_1000508", "BIN_1000102"]
    for binary_id in ids:
        print(f"\n== {binary_id} ==")
        rows = symbols(binary_id)
        print("symbols", len(rows))
        hits = [
            row
            for row in rows
            if any(part.lower() in row[0].lower() for part in ["cwe", "bad", "good", "main", "juliet"])
        ]
        for row in hits[:120]:
            print(row)


if __name__ == "__main__":
    main()
