from __future__ import annotations

import argparse
import os
import re
import struct
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pefile
from capstone import CS_ARCH_X86, CS_MODE_32, CS_MODE_64, Cs
from tqdm import tqdm

from common import BINARY_DIR, FEATURE_CACHE, NO_VULN, ensure_dirs, entropy_from_counts


BYTE_BINS = [f"byte_{idx:02x}" for idx in range(256)]
COMMON_SECTION_NAMES = [
    ".text",
    ".data",
    ".rdata",
    ".idata",
    ".edata",
    ".rsrc",
    ".reloc",
    ".pdata",
    ".bss",
    ".tls",
    ".crt",
]
STRING_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_./:%+\\-]{2,31}")
SYMBOL_PART_RE = re.compile(r"[A-Za-z0-9]{2,48}")
PRINTABLE_RE = re.compile(rb"[ -~]{4,96}")
SUSPICIOUS_TERMS = {
    "alloc",
    "argv",
    "atoi",
    "copy",
    "exec",
    "file",
    "format",
    "free",
    "gets",
    "heap",
    "http",
    "input",
    "malloc",
    "mem",
    "null",
    "open",
    "overflow",
    "path",
    "printf",
    "read",
    "scanf",
    "shell",
    "size",
    "sprintf",
    "stack",
    "strcat",
    "strcpy",
    "strlen",
    "system",
    "tmp",
    "unlink",
    "user",
    "write",
}
EXECUTABLE_SCN_MASK = 0x20000000
READABLE_SCN_MASK = 0x40000000
WRITABLE_SCN_MASK = 0x80000000
TRACKED_MNEMONICS = [
    "mov",
    "lea",
    "call",
    "cmp",
    "test",
    "jmp",
    "je",
    "jne",
    "push",
    "pop",
    "add",
    "sub",
    "xor",
    "and",
    "or",
    "imul",
    "idiv",
    "shl",
    "shr",
    "ret",
    "nop",
]
SYMBOL_KEYWORDS = ("cwe", "entry", "bad", "good", "sink", "source", "data", "alloc", "free", "copy", "overflow")


def safe_div(value: float, denom: float) -> float:
    if not denom:
        return 0.0
    return float(value / denom)


def decode_text(value: bytes) -> str:
    return value.decode("latin1", "ignore").strip().lower()


def decode_section_name(value: bytes) -> str:
    return value.rstrip(b"\x00").decode("latin1", "ignore").strip().lower()


def read_symbol_name(raw: bytes, offset: int, string_table: bytes) -> str:
    name_bytes = raw[offset : offset + 8]
    zeroes, str_offset = struct.unpack_from("<II", name_bytes)
    if zeroes == 0 and str_offset:
        start = str_offset
        end = string_table.find(b"\x00", start)
        if end == -1:
            end = len(string_table)
        return string_table[start:end].decode("latin1", "ignore")
    return name_bytes.split(b"\x00", 1)[0].decode("latin1", "ignore")


def build_symbol_tokens(raw: bytes, pe: pefile.PE, limit: int = 512) -> tuple[list[str], dict[str, float]]:
    tokens: list[str] = []
    pointer = int(getattr(pe.FILE_HEADER, "PointerToSymbolTable", 0))
    count = int(getattr(pe.FILE_HEADER, "NumberOfSymbols", 0))
    if not pointer or not count:
        return tokens, {
            "coff_symbol_count": 0.0,
            "coff_relevant_symbol_count": 0.0,
            "coff_entry_bad_count": 0.0,
            "coff_entry_good_count": 0.0,
        }

    string_table = raw[pointer + count * 18 :]
    relevant_count = 0
    entry_bad_count = 0
    entry_good_count = 0
    idx = 0
    while idx < count and len(tokens) < limit:
        offset = pointer + idx * 18
        if offset + 18 > len(raw):
            break
        name = read_symbol_name(raw, offset, string_table)
        _, section_number, _, _, aux_count = struct.unpack_from("<IhHBB", raw, offset + 8)
        idx += 1 + aux_count
        if section_number <= 0:
            continue
        lowered = name.lower()
        if "entry_bad" in lowered:
            entry_bad_count += 1
        if "entry_good" in lowered:
            entry_good_count += 1
        if not any(keyword in lowered for keyword in SYMBOL_KEYWORDS):
            continue
        relevant_count += 1
        compact = re.sub(r"\s+", "_", lowered)[:180]
        tokens.append(f"sym:{compact}")
        for part in SYMBOL_PART_RE.findall(lowered):
            tokens.append(f"sympart:{part.lower()}")
            if len(tokens) >= limit:
                break

    return tokens, {
        "coff_symbol_count": float(count),
        "coff_relevant_symbol_count": float(relevant_count),
        "coff_entry_bad_count": float(entry_bad_count),
        "coff_entry_good_count": float(entry_good_count),
    }


def entropy_bytes(raw: bytes) -> float:
    if not raw:
        return 0.0
    counts = np.bincount(np.frombuffer(raw, dtype=np.uint8), minlength=256)
    return entropy_from_counts(counts)


def build_byte_histogram(raw: bytes) -> dict[str, float]:
    if not raw:
        return {name: 0.0 for name in BYTE_BINS}
    counts = np.bincount(np.frombuffer(raw, dtype=np.uint8), minlength=256).astype(np.float32)
    counts /= max(len(raw), 1)
    return {name: float(value) for name, value in zip(BYTE_BINS, counts, strict=False)}


def select_string_tokens(raw: bytes, limit: int = 256) -> tuple[list[str], dict[str, float]]:
    tokens: list[str] = []
    suspicious_hits = 0
    strings = PRINTABLE_RE.findall(raw)
    longest = 0
    total_len = 0
    for blob in strings:
        text = decode_text(blob)
        if not text:
            continue
        longest = max(longest, len(text))
        total_len += len(text)
        for term in SUSPICIOUS_TERMS:
            if term in text:
                suspicious_hits += 1
        for token in STRING_TOKEN_RE.findall(text):
            lowered = token.lower()
            if len(lowered) > 31:
                continue
            if lowered.count(".") > 2:
                continue
            tokens.append(f"str:{lowered}")
            if len(tokens) >= limit:
                break
        if len(tokens) >= limit:
            break
    stats = {
        "string_count": float(len(strings)),
        "string_longest": float(longest),
        "string_mean_len": float(total_len / len(strings)) if strings else 0.0,
        "string_suspicious_hits": float(suspicious_hits),
    }
    return tokens, stats


def build_disasm_stats(pe: pefile.PE, max_insns: int = 4096) -> dict[str, float]:
    machine = pe.FILE_HEADER.Machine
    if machine == 0x14C:
        mode = CS_MODE_32
    elif machine == 0x8664:
        mode = CS_MODE_64
    else:
        base = {"insn_count": 0.0, "call_count": 0.0, "branch_count": 0.0}
        for mnemonic in TRACKED_MNEMONICS:
            base[f"op_count::{mnemonic}"] = 0.0
            base[f"op_ratio::{mnemonic}"] = 0.0
        return base

    md = Cs(CS_ARCH_X86, mode)
    md.detail = False
    insn_count = 0
    call_count = 0
    branch_count = 0
    opcode_counts = {mnemonic: 0 for mnemonic in TRACKED_MNEMONICS}
    for section in pe.sections:
        if not section.Characteristics & EXECUTABLE_SCN_MASK:
            continue
        code = section.get_data()
        base = pe.OPTIONAL_HEADER.ImageBase + section.VirtualAddress
        for insn in md.disasm(code, base):
            mnemonic = insn.mnemonic.lower()
            insn_count += 1
            if mnemonic.startswith("call"):
                call_count += 1
            if mnemonic.startswith("j"):
                branch_count += 1
            if mnemonic in opcode_counts:
                opcode_counts[mnemonic] += 1
            if insn_count >= max_insns:
                break
        if insn_count >= max_insns:
            break
    stats: dict[str, float] = {
        "insn_count": float(insn_count),
        "call_count": float(call_count),
        "branch_count": float(branch_count),
    }
    for mnemonic, value in opcode_counts.items():
        stats[f"op_count::{mnemonic}"] = float(value)
        stats[f"op_ratio::{mnemonic}"] = safe_div(value, insn_count)
    return stats


def extract_one(binary_id: str, include_symbols: bool = False) -> dict[str, object]:
    path = BINARY_DIR / f"{binary_id}.exe"
    raw = path.read_bytes()
    features: dict[str, object] = {
        "binary_id": binary_id,
        "path": str(path),
        "file_size": float(len(raw)),
        "file_entropy": float(entropy_bytes(raw)),
    }
    features.update(build_byte_histogram(raw))

    quarter = max(len(raw) // 4, 1)
    for idx in range(4):
        start = idx * quarter
        end = len(raw) if idx == 3 else min(len(raw), (idx + 1) * quarter)
        features[f"chunk_{idx}_entropy"] = float(entropy_bytes(raw[start:end]))

    string_tokens, string_stats = select_string_tokens(raw)
    features.update(string_stats)
    doc_tokens = list(string_tokens)

    try:
        pe = pefile.PE(data=raw, fast_load=True)
        pe.parse_data_directories(
            directories=[
                pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"],
                pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_EXPORT"],
            ]
        )
        file_header = pe.FILE_HEADER
        opt = pe.OPTIONAL_HEADER
        features.update(
            {
                "machine": float(file_header.Machine),
                "num_sections": float(file_header.NumberOfSections),
                "timestamp": float(file_header.TimeDateStamp),
                "characteristics": float(file_header.Characteristics),
                "opt_magic": float(getattr(opt, "Magic", 0)),
                "major_linker": float(getattr(opt, "MajorLinkerVersion", 0)),
                "minor_linker": float(getattr(opt, "MinorLinkerVersion", 0)),
                "size_of_code": float(getattr(opt, "SizeOfCode", 0)),
                "size_of_initialized_data": float(getattr(opt, "SizeOfInitializedData", 0)),
                "size_of_uninitialized_data": float(getattr(opt, "SizeOfUninitializedData", 0)),
                "address_of_entry_point": float(getattr(opt, "AddressOfEntryPoint", 0)),
                "base_of_code": float(getattr(opt, "BaseOfCode", 0)),
                "image_base": float(getattr(opt, "ImageBase", 0)),
                "section_alignment": float(getattr(opt, "SectionAlignment", 0)),
                "file_alignment": float(getattr(opt, "FileAlignment", 0)),
                "major_os_version": float(getattr(opt, "MajorOperatingSystemVersion", 0)),
                "minor_os_version": float(getattr(opt, "MinorOperatingSystemVersion", 0)),
                "major_image_version": float(getattr(opt, "MajorImageVersion", 0)),
                "minor_image_version": float(getattr(opt, "MinorImageVersion", 0)),
                "major_subsystem_version": float(getattr(opt, "MajorSubsystemVersion", 0)),
                "minor_subsystem_version": float(getattr(opt, "MinorSubsystemVersion", 0)),
                "size_of_image": float(getattr(opt, "SizeOfImage", 0)),
                "size_of_headers": float(getattr(opt, "SizeOfHeaders", 0)),
                "checksum": float(getattr(opt, "CheckSum", 0)),
                "subsystem": float(getattr(opt, "Subsystem", 0)),
                "dll_characteristics": float(getattr(opt, "DllCharacteristics", 0)),
                "stack_reserve": float(getattr(opt, "SizeOfStackReserve", 0)),
                "stack_commit": float(getattr(opt, "SizeOfStackCommit", 0)),
                "heap_reserve": float(getattr(opt, "SizeOfHeapReserve", 0)),
                "heap_commit": float(getattr(opt, "SizeOfHeapCommit", 0)),
                "loader_flags": float(getattr(opt, "LoaderFlags", 0)),
                "number_rva_and_sizes": float(getattr(opt, "NumberOfRvaAndSizes", 0)),
            }
        )

        exec_sections = 0
        writable_sections = 0
        readable_sections = 0
        section_entropies: list[float] = []
        section_raw_sizes: list[int] = []
        section_virtual_sizes: list[int] = []
        for name in COMMON_SECTION_NAMES:
            features[f"sec_present::{name}"] = 0.0
            features[f"sec_raw_ratio::{name}"] = 0.0
            features[f"sec_entropy::{name}"] = 0.0

        for section in pe.sections:
            name = decode_section_name(section.Name)
            raw_size = int(section.SizeOfRawData)
            virtual_size = int(section.Misc_VirtualSize)
            entropy = float(section.get_entropy())
            section_entropies.append(entropy)
            section_raw_sizes.append(raw_size)
            section_virtual_sizes.append(virtual_size)
            doc_tokens.append(f"sec:{name}")
            if section.Characteristics & EXECUTABLE_SCN_MASK:
                exec_sections += 1
            if section.Characteristics & READABLE_SCN_MASK:
                readable_sections += 1
            if section.Characteristics & WRITABLE_SCN_MASK:
                writable_sections += 1
            if name in COMMON_SECTION_NAMES:
                features[f"sec_present::{name}"] = 1.0
                features[f"sec_raw_ratio::{name}"] = safe_div(raw_size, len(raw))
                features[f"sec_entropy::{name}"] = entropy

        features.update(
            {
                "exec_sections": float(exec_sections),
                "readable_sections": float(readable_sections),
                "writable_sections": float(writable_sections),
                "section_entropy_mean": float(np.mean(section_entropies)) if section_entropies else 0.0,
                "section_entropy_std": float(np.std(section_entropies)) if section_entropies else 0.0,
                "section_entropy_max": float(np.max(section_entropies)) if section_entropies else 0.0,
                "section_entropy_min": float(np.min(section_entropies)) if section_entropies else 0.0,
                "section_raw_mean": float(np.mean(section_raw_sizes)) if section_raw_sizes else 0.0,
                "section_raw_std": float(np.std(section_raw_sizes)) if section_raw_sizes else 0.0,
                "section_virtual_mean": float(np.mean(section_virtual_sizes)) if section_virtual_sizes else 0.0,
                "section_virtual_std": float(np.std(section_virtual_sizes)) if section_virtual_sizes else 0.0,
            }
        )

        import_dll_count = 0
        import_func_count = 0
        for entry in getattr(pe, "DIRECTORY_ENTRY_IMPORT", []):
            dll = decode_text(entry.dll)
            if dll:
                import_dll_count += 1
                doc_tokens.append(f"dll:{dll}")
            for item in entry.imports:
                if item.name:
                    import_func_count += 1
                    doc_tokens.append(f"imp:{decode_text(item.name)}")
                elif item.ordinal is not None:
                    doc_tokens.append(f"ord:{dll}:{item.ordinal}")
        export_count = 0
        export_dir = getattr(pe, "DIRECTORY_ENTRY_EXPORT", None)
        if export_dir is not None:
            for symbol in getattr(export_dir, "symbols", []) or []:
                if symbol.name:
                    export_count += 1
                    doc_tokens.append(f"exp:{decode_text(symbol.name)}")
        features["import_dll_count"] = float(import_dll_count)
        features["import_func_count"] = float(import_func_count)
        features["export_count"] = float(export_count)

        if include_symbols:
            symbol_tokens, symbol_stats = build_symbol_tokens(raw, pe)
            doc_tokens.extend(symbol_tokens)
            features.update(symbol_stats)
        features.update(build_disasm_stats(pe))
    except Exception:
        features["pe_parse_failed"] = 1.0
    else:
        features["pe_parse_failed"] = 0.0

    features["doc"] = " ".join(doc_tokens)
    return features


def _extract_one_from_args(args: tuple[str, bool]) -> dict[str, object]:
    return extract_one(*args)


def build_feature_frame(binary_ids: list[str], workers: int, include_symbols: bool) -> pd.DataFrame:
    if workers <= 1:
        rows = [
            extract_one(binary_id, include_symbols=include_symbols)
            for binary_id in tqdm(binary_ids, desc="extract", ncols=100)
        ]
        return pd.DataFrame(rows)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        iterator = executor.map(_extract_one_from_args, [(binary_id, include_symbols) for binary_id in binary_ids], chunksize=32)
        rows = list(tqdm(iterator, total=len(binary_ids), desc="extract", ncols=100))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract PE/static features for the challenge binaries.")
    parser.add_argument("--workers", type=int, default=max(os.cpu_count() // 2, 1))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--include-symbols", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    if FEATURE_CACHE.exists() and not args.force:
        print(f"feature cache exists: {FEATURE_CACHE}")
        return

    train = pd.read_csv(BINARY_DIR.parents[0] / "train.csv")
    test = pd.read_csv(BINARY_DIR.parents[0] / "test.csv")
    binary_ids = pd.concat([train["binary_id"], test["binary_id"]], ignore_index=True).tolist()
    frame = build_feature_frame(binary_ids, workers=max(args.workers, 1), include_symbols=args.include_symbols)
    frame["doc_len"] = frame["doc"].str.len().fillna(0).astype(np.float32)
    frame["token_count"] = frame["doc"].str.count(" ").fillna(0).astype(np.float32) + 1
    frame["split"] = np.where(frame["binary_id"].isin(set(train["binary_id"])), "train", "test")
    frame["target"] = NO_VULN
    target_map = train.set_index("binary_id")["cwe_id"].fillna(NO_VULN).to_dict()
    frame["target"] = frame["binary_id"].map(target_map).fillna(NO_VULN)
    frame = frame.sort_values("binary_id").reset_index(drop=True)
    FEATURE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    frame.to_pickle(FEATURE_CACHE)
    print(f"saved {len(frame)} rows to {FEATURE_CACHE}")


if __name__ == "__main__":
    main()
