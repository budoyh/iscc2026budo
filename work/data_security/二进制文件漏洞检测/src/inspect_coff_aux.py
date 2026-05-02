from __future__ import annotations

import re
import struct
import sys

import pefile

from common import BINARY_DIR
from inspect_symbols import read_symbol_name


ASCII_RE = re.compile(rb"[ -~]{4,260}")


def dump_aux(binary_id: str, limit: int = 80) -> None:
    raw = (BINARY_DIR / f"{binary_id}.exe").read_bytes()
    pe = pefile.PE(data=raw, fast_load=True)
    ptr = pe.FILE_HEADER.PointerToSymbolTable
    count = pe.FILE_HEADER.NumberOfSymbols
    string_start = ptr + count * 18
    string_table = raw[string_start:]
    idx = 0
    printed = 0
    while idx < count and printed < limit:
        offset = ptr + idx * 18
        name = read_symbol_name(raw, offset, string_table)
        value, section_number, typ, storage, aux_count = struct.unpack_from("<IhHBB", raw, offset + 8)
        aux = raw[offset + 18 : offset + 18 + aux_count * 18]
        text_hits = [item.decode("latin1", "ignore") for item in ASCII_RE.findall(aux)]
        if aux_count and (storage == 103 or name == ".file" or any("CWE" in item for item in text_hits)):
            print(
                {
                    "idx": idx,
                    "name": name,
                    "value": value,
                    "section": section_number,
                    "storage": storage,
                    "aux_count": aux_count,
                    "aux_ascii": text_hits[:8],
                    "aux_hex_head": aux[:72].hex(),
                }
            )
            printed += 1
        idx += 1 + aux_count


def main() -> None:
    for binary_id in sys.argv[1:] or ["BIN_1000205", "BIN_1000642", "BIN_1000013"]:
        print(f"\n== {binary_id} ==")
        dump_aux(binary_id)


if __name__ == "__main__":
    main()
