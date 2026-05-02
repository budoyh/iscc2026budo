from __future__ import annotations

import sys

import pefile
from capstone import CS_ARCH_X86, CS_MODE_32, CS_MODE_64, Cs
from capstone.x86 import X86_OP_IMM

from common import BINARY_DIR
from inspect_symbols import symbols


def section_for_symbol(pe: pefile.PE, section_number: int):
    if section_number <= 0 or section_number > len(pe.sections):
        return None
    return pe.sections[section_number - 1]


def symbol_rva(pe: pefile.PE, section_number: int, value: int) -> int:
    section = section_for_symbol(pe, section_number)
    if section is None:
        return value
    return section.VirtualAddress + value


def main() -> None:
    ids = sys.argv[1:] or ["BIN_1000013", "BIN_1000508"]
    for binary_id in ids:
        path = BINARY_DIR / f"{binary_id}.exe"
        raw = path.read_bytes()
        pe = pefile.PE(data=raw, fast_load=True)
        machine = pe.FILE_HEADER.Machine
        mode = CS_MODE_64 if machine == 0x8664 else CS_MODE_32
        md = Cs(CS_ARCH_X86, mode)
        md.detail = True
        rows = symbols(binary_id)
        code_symbols = [(name, value, sec) for name, value, sec, storage in rows if sec > 0]
        addr_to_names = {}
        for name, value, sec in code_symbols:
            rva = symbol_rva(pe, sec, value)
            addr_to_names.setdefault(pe.OPTIONAL_HEADER.ImageBase + rva, []).append(name)
        main_rows = [(name, value, sec) for name, value, sec, storage in rows if name == "main"]
        if not main_rows:
            print(binary_id, "no main")
            continue
        _, main_value, main_sec = main_rows[0]
        section = section_for_symbol(pe, main_sec)
        start_rva = section.VirtualAddress + main_value
        same_sec_values = sorted(value for name, value, sec in code_symbols if sec == main_sec and value > main_value)
        end_value = same_sec_values[0] if same_sec_values else main_value + 256
        code = section.get_data()[main_value:end_value]
        print(f"\n== {binary_id} main_value={main_value} len={len(code)} ==")
        for insn in md.disasm(code, pe.OPTIONAL_HEADER.ImageBase + start_rva):
            target_names = []
            if insn.mnemonic.startswith("call") and insn.operands and insn.operands[0].type == X86_OP_IMM:
                target = insn.operands[0].imm
                target_names = addr_to_names.get(target, [])
            if insn.mnemonic.startswith("call") or target_names:
                print(hex(insn.address), insn.mnemonic, insn.op_str, target_names)


if __name__ == "__main__":
    main()
