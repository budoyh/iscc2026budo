from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pefile
from capstone import CS_ARCH_X86, CS_MODE_32, CS_MODE_64, Cs
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier

from common import BINARY_DIR, PROCESSED, RAW, REPORT, ensure_dirs
from enhanced_cwe_calibrator import bad_function_bytes
from entry_body_cwe import choose_bad_symbol
from inspect_symbols import symbols


CLASSES = ["CWE-121", "CWE-122", "CWE-126"]


def entry_value(binary_id: str) -> int:
    symbol_name = choose_bad_symbol(binary_id)
    for name, value, section, _storage in symbols(binary_id):
        if name == symbol_name and section == 1:
            return int(value)
    return 0


def text_symbols(binary_id: str) -> list[tuple[int, str]]:
    return [
        (int(value), name)
        for name, value, section, _storage in symbols(binary_id)
        if section == 1 and name and not name.startswith(".")
    ]


def nearest_symbol_name(rows: list[tuple[int, str]], value: int) -> str:
    best_name = ""
    best_dist = 1 << 30
    for symbol_value, name in rows:
        dist = abs(symbol_value - value)
        if dist < best_dist:
            best_name = name
            best_dist = dist
    return best_name if best_dist <= 8 else ""


def one_features(binary_id: str) -> dict[str, object]:
    raw = (BINARY_DIR / f"{binary_id}.exe").read_bytes()
    pe = pefile.PE(data=raw, fast_load=True)
    body = bad_function_bytes(binary_id, raw, pe, max_bytes=512)
    mode = CS_MODE_64 if pe.FILE_HEADER.Machine == 0x8664 else CS_MODE_32
    md = Cs(CS_ARCH_X86, mode)
    value_base = entry_value(binary_id)
    text_syms = text_symbols(binary_id)

    feats: dict[str, object] = {
        "binary_id": binary_id,
        "body_len": len(body),
        "n_insn": 0,
        "calls": 0,
        "call_malloc": 0,
        "call_calloc": 0,
        "call_free": 0,
        "call_memcpy": 0,
        "call_printline": 0,
        "call_printwline": 0,
        "call_printlong": 0,
        "call_exit": 0,
        "stack_store": 0,
        "heap_store": 0,
        "stack_load": 0,
        "heap_load": 0,
        "stack_lea": 0,
        "byte_store_stack": 0,
        "word_store_stack": 0,
        "byte_store_heap": 0,
        "word_store_heap": 0,
        "malloc_20": 0,
        "malloc_30": 0,
        "memcpy_20": 0,
        "memcpy_30": 0,
        "heap_off_0f": 0,
        "heap_off_1e": 0,
        "heap_off_10": 0,
        "heap_off_20": 0,
        "stack_off_11": 0,
        "stack_off_12": 0,
        "movabs": 0,
    }
    call_names: list[str] = []
    last_ecx: int | None = None
    last_r8d: int | None = None

    for insn in md.disasm(body, 0):
        feats["n_insn"] = int(feats["n_insn"]) + 1
        mnemonic = insn.mnemonic
        op = insn.op_str.lower()
        if mnemonic == "movabs":
            feats["movabs"] = int(feats["movabs"]) + 1
        if mnemonic.startswith("lea") and "[rbp -" in op:
            feats["stack_lea"] = int(feats["stack_lea"]) + 1
        if mnemonic.startswith("mov") and "[rbp -" in op:
            if op.strip().startswith(("byte ptr [rbp", "word ptr [rbp", "dword ptr [rbp", "qword ptr [rbp")):
                feats["stack_store"] = int(feats["stack_store"]) + 1
            else:
                feats["stack_load"] = int(feats["stack_load"]) + 1
        if mnemonic.startswith("mov") and "[rax +" in op:
            if op.strip().startswith(("byte ptr [rax", "word ptr [rax", "dword ptr [rax", "qword ptr [rax")):
                feats["heap_store"] = int(feats["heap_store"]) + 1
            else:
                feats["heap_load"] = int(feats["heap_load"]) + 1
        for prefix, key in [
            ("byte ptr [rbp", "byte_store_stack"),
            ("word ptr [rbp", "word_store_stack"),
            ("byte ptr [rax", "byte_store_heap"),
            ("word ptr [rax", "word_store_heap"),
        ]:
            if op.strip().startswith(prefix):
                feats[key] = int(feats[key]) + 1
        for needle, key in [
            ("- 0x11", "stack_off_11"),
            ("- 0x12", "stack_off_12"),
            ("+ 0xf", "heap_off_0f"),
            ("+ 0x1e", "heap_off_1e"),
            ("+ 0x10", "heap_off_10"),
            ("+ 0x20", "heap_off_20"),
        ]:
            if needle in op:
                feats[key] = int(feats[key]) + 1
        if mnemonic == "mov" and op.startswith("ecx,"):
            try:
                last_ecx = int(op.split(",", 1)[1].strip(), 0)
            except ValueError:
                pass
        if mnemonic == "mov" and op.startswith("r8d,"):
            try:
                last_r8d = int(op.split(",", 1)[1].strip(), 0)
            except ValueError:
                pass
        if mnemonic != "call":
            continue
        feats["calls"] = int(feats["calls"]) + 1
        try:
            rel_target = int(op, 0)
        except ValueError:
            rel_target = 0
        name = nearest_symbol_name(text_syms, value_base + rel_target)
        call_names.append(name)
        lowered = name.lower()
        if lowered == "malloc":
            feats["call_malloc"] = int(feats["call_malloc"]) + 1
            if last_ecx == 0x20:
                feats["malloc_20"] = int(feats["malloc_20"]) + 1
            if last_ecx == 0x30:
                feats["malloc_30"] = int(feats["malloc_30"]) + 1
        if lowered == "calloc":
            feats["call_calloc"] = int(feats["call_calloc"]) + 1
        if lowered == "free":
            feats["call_free"] = int(feats["call_free"]) + 1
        if lowered == "memcpy":
            feats["call_memcpy"] = int(feats["call_memcpy"]) + 1
            if last_r8d == 0x20:
                feats["memcpy_20"] = int(feats["memcpy_20"]) + 1
            if last_r8d == 0x30:
                feats["memcpy_30"] = int(feats["memcpy_30"]) + 1
        if lowered in {"exit", "abort"}:
            feats["call_exit"] = int(feats["call_exit"]) + 1
        if lowered == "printline":
            feats["call_printline"] = int(feats["call_printline"]) + 1
        if lowered == "printwline":
            feats["call_printwline"] = int(feats["call_printwline"]) + 1
        if lowered.startswith("printlong"):
            feats["call_printlong"] = int(feats["call_printlong"]) + 1
    feats["call_sig"] = " ".join(call_names)
    return feats


def build_cache(binary_ids: list[str], force: bool) -> pd.DataFrame:
    cache_path = PROCESSED / "struct_121_122_126_features.csv"
    if cache_path.exists() and not force:
        cached = pd.read_csv(cache_path)
    else:
        cached = pd.DataFrame()
    have = set(cached["binary_id"]) if len(cached) else set()
    rows = []
    failures = []
    for pos, binary_id in enumerate(binary_ids, start=1):
        if binary_id in have:
            continue
        try:
            rows.append(one_features(binary_id))
        except Exception as exc:  # noqa: BLE001 - keep feature extraction robust.
            failures.append((binary_id, type(exc).__name__, str(exc)))
        if pos % 500 == 0:
            print(f"processed={pos} new={len(rows)} failures={len(failures)}")
    if rows:
        cached = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True).drop_duplicates("binary_id")
        cached.to_csv(cache_path, index=False, encoding="utf-8")
    if failures:
        pd.DataFrame(failures, columns=["binary_id", "error", "message"]).to_csv(
            REPORT / "struct_121_122_126_failures.csv", index=False, encoding="utf-8"
        )
    return cached


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--diff", default="s17b.csv.diff.csv")
    args = parser.parse_args()
    ensure_dirs()

    train = pd.read_csv(RAW / "train.csv")
    diff = pd.read_csv(REPORT / args.diff)
    ids = (
        pd.concat([train.loc[train["cwe_id"].isin(CLASSES), "binary_id"], diff["binary_id"]], ignore_index=True)
        .drop_duplicates()
        .tolist()
    )
    cached = build_cache(ids, force=args.force)

    part = train.loc[train["cwe_id"].isin(CLASSES), ["binary_id", "cwe_id"]].merge(
        cached, on="binary_id", how="left", validate="one_to_one"
    )
    feature_cols = [col for col in cached.columns if col not in {"binary_id", "call_sig"}]
    x = part[feature_cols].fillna(0).to_numpy()
    y = part["cwe_id"].to_numpy()
    cv = StratifiedKFold(5, shuffle=True, random_state=20260501)
    models = [
        DecisionTreeClassifier(max_depth=6, random_state=1, class_weight="balanced"),
        RandomForestClassifier(n_estimators=400, max_depth=10, random_state=2, class_weight="balanced_subsample"),
        ExtraTreesClassifier(n_estimators=500, random_state=3, class_weight="balanced"),
    ]
    for model in models:
        pred = cross_val_predict(model, x, y, cv=cv)
        print(f"{type(model).__name__} acc={accuracy_score(y, pred):.12f} macro={f1_score(y, pred, average='macro'):.12f}")
        print(pd.crosstab(pd.Series(y, name="true"), pd.Series(pred, name="pred")).to_string())

    model = ExtraTreesClassifier(n_estimators=500, random_state=3, class_weight="balanced")
    model.fit(x, y)
    cand = diff.merge(cached, on="binary_id", how="left", validate="one_to_one")
    xt = cand[feature_cols].fillna(0).to_numpy()
    prob = model.predict_proba(xt)
    labels = model.classes_.tolist()
    cand["struct_pred"] = model.predict(xt)
    for label in labels:
        cand[f"struct_prob_{label}"] = prob[:, labels.index(label)]
    cand["struct_old_prob"] = [cand.iloc[i][f"struct_prob_{cand.iloc[i]['old_cwe']}"] for i in range(len(cand))]
    cand["struct_new_prob"] = [cand.iloc[i][f"struct_prob_{cand.iloc[i]['new_cwe']}"] for i in range(len(cand))]
    out_cols = [
        "binary_id",
        "sid",
        "old_cwe",
        "new_cwe",
        "dist",
        "struct_pred",
        "struct_old_prob",
        "struct_new_prob",
        "call_malloc",
        "call_free",
        "call_memcpy",
        "malloc_20",
        "malloc_30",
        "memcpy_20",
        "memcpy_30",
        "stack_store",
        "heap_store",
        "byte_store_stack",
        "word_store_stack",
        "byte_store_heap",
        "word_store_heap",
        "call_sig",
    ]
    out_path = REPORT / "s17b_struct_semantics.csv"
    cand[out_cols].to_csv(out_path, index=False, encoding="utf-8")
    print(cand[out_cols].to_string(index=False))
    print(out_path)


if __name__ == "__main__":
    main()
