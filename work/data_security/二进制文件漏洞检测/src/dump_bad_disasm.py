from __future__ import annotations

import sys

import pefile
from capstone import CS_ARCH_X86, CS_MODE_32, CS_MODE_64, Cs

from common import BINARY_DIR
from enhanced_cwe_calibrator import bad_function_bytes
from entry_body_cwe import choose_bad_symbol


def main() -> None:
    for binary_id in sys.argv[1:]:
        raw = (BINARY_DIR / f"{binary_id}.exe").read_bytes()
        pe = pefile.PE(data=raw, fast_load=True)
        body = bad_function_bytes(binary_id, raw, pe, max_bytes=512)
        mode = CS_MODE_64 if pe.FILE_HEADER.Machine == 0x8664 else CS_MODE_32
        md = Cs(CS_ARCH_X86, mode)
        print(f"\n== {binary_id} symbol={choose_bad_symbol(binary_id)} len={len(body)} ==")
        for insn in md.disasm(body, 0):
            print(f"{insn.address:04x}: {insn.mnemonic:8s} {insn.op_str}")


if __name__ == "__main__":
    main()
