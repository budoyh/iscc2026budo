from __future__ import annotations

import argparse
import math
import re
import struct
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
import pefile
from capstone import CS_ARCH_X86, CS_MODE_32, CS_MODE_64, Cs
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import f1_score
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC
from tqdm import tqdm

from common import BINARY_DIR, FEATURE_CACHE, NO_VULN, PROCESSED, RAW, SEED, SUBMISSIONS, dataframe_to_sparse, ensure_dirs, load_test
from entry_body_cwe import choose_bad_symbol
from inspect_symbols import read_symbol_name, symbols
from predict import validate_submission
from string_rule_cwe import build_rules, predict_cwe
from symbol_rule_predict import predict_one, section_for_symbol
from train import build_split, prepare_matrices


DOC_CACHE = PROCESSED / "enhanced_cwe_docs.joblib"

ASCII_RE = re.compile(rb"[ -~]{4,180}")
IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,64}")
HEX_RE = re.compile(r"0x[0-9a-fA-F]+|\b\d{2,}\b")
SKIP_STRING_PARTS = (
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
    "mingw",
    "configthreadlocale",
)

GROUPS: dict[str, tuple[str, ...]] = {
    "retval": ("CWE-252", "CWE-253", "CWE-390", "CWE-391", "CWE-468", "CWE-571"),
    "buffer": (
        "CWE-121",
        "CWE-122",
        "CWE-124",
        "CWE-126",
        "CWE-127",
        "CWE-135",
        "CWE-194",
        "CWE-369",
        "CWE-398",
        "CWE-457",
        "CWE-467",
        "CWE-476",
        "CWE-590",
        "CWE-665",
        "CWE-690",
    ),
    "integer": ("CWE-190", "CWE-191", "CWE-195", "CWE-196", "CWE-197", "CWE-680", "CWE-681"),
}


@dataclass(frozen=True)
class DocRow:
    binary_id: str
    doc: str
    body_hex: str
    body_len: float
    insn_count: float
    call_count: float
    branch_count: float


def read_symbol_table(raw: bytes, pe: pefile.PE) -> list[tuple[str, int, int, int, int]]:
    ptr = int(getattr(pe.FILE_HEADER, "PointerToSymbolTable", 0))
    count = int(getattr(pe.FILE_HEADER, "NumberOfSymbols", 0))
    if not ptr or not count:
        return []
    string_table = raw[ptr + count * 18 :]
    rows: list[tuple[str, int, int, int, int]] = []
    idx = 0
    while idx < count:
        offset = ptr + idx * 18
        if offset + 18 > len(raw):
            break
        name = read_symbol_name(raw, offset, string_table)
        value, section_number, _typ, storage, aux_count = struct.unpack_from("<IhHBB", raw, offset + 8)
        rows.append((name, value, section_number, storage, aux_count))
        idx += 1 + aux_count
    return rows


def symbol_va(pe: pefile.PE, section_number: int, value: int) -> int:
    section = section_for_symbol(pe, section_number)
    if section is None:
        return int(getattr(pe.OPTIONAL_HEADER, "ImageBase", 0)) + value
    return int(pe.OPTIONAL_HEADER.ImageBase + section.VirtualAddress + value)


def bad_function_payload(binary_id: str, raw: bytes, pe: pefile.PE, max_bytes: int = 4096) -> tuple[bytes, int, dict[int, list[str]]]:
    target_name = choose_bad_symbol(binary_id)
    if not target_name:
        return b"", 0, {}
    rows = symbols(binary_id)
    code_symbols: list[tuple[str, int, int]] = []
    target: tuple[int, int] | None = None
    for name, value, sec, _storage in rows:
        if sec > 0:
            code_symbols.append((name, value, sec))
        if name == target_name and sec > 0:
            target = (value, sec)
    if target is None:
        return b"", 0, {}
    addr_to_names: dict[int, list[str]] = {}
    for name, value, sec in code_symbols:
        addr_to_names.setdefault(symbol_va(pe, sec, value), []).append(name)
    value, sec = target
    section = section_for_symbol(pe, sec)
    if section is None:
        return b"", 0, addr_to_names
    next_values = sorted(v for _name, v, s in code_symbols if s == sec and v > value)
    end = next_values[0] if next_values else value + max_bytes
    start_va = int(pe.OPTIONAL_HEADER.ImageBase + section.VirtualAddress + value)
    return section.get_data()[value:end][:max_bytes], start_va, addr_to_names


def bad_function_bytes(binary_id: str, raw: bytes, pe: pefile.PE, max_bytes: int = 4096) -> bytes:
    body, _start_va, _addr_to_names = bad_function_payload(binary_id, raw, pe, max_bytes)
    return body


def norm_operand_text(text: str) -> str:
    text = HEX_RE.sub("N", text.lower())
    text = text.replace("ptr ", "ptr_")
    text = re.sub(r"\s+", "", text)
    return text[:80]


def disasm_tokens(
    raw: bytes,
    pe: pefile.PE,
    body: bytes,
    start_va: int = 0,
    addr_to_names: dict[int, list[str]] | None = None,
    max_insns: int = 1024,
) -> tuple[list[str], int, int, int]:
    if not body:
        return [], 0, 0, 0
    mode = CS_MODE_64 if pe.FILE_HEADER.Machine == 0x8664 else CS_MODE_32
    md = Cs(CS_ARCH_X86, mode)
    md.detail = True
    tokens: list[str] = []
    call_names: list[str] = []
    insn_count = call_count = branch_count = 0
    for insn in md.disasm(body, start_va):
        mnemonic = insn.mnemonic.lower()
        op_text = norm_operand_text(insn.op_str)
        insn_count += 1
        if mnemonic.startswith("call"):
            call_count += 1
            target_names: list[str] = []
            if addr_to_names and insn.operands and insn.operands[0].type == 2:
                target_names = addr_to_names.get(int(insn.operands[0].imm), [])
            for name in target_names[:4]:
                lowered = name.lower()
                tokens.append(f"callname:{lowered[:120]}")
                for ident in IDENT_RE.findall(lowered):
                    tokens.append(f"callid:{ident}")
                call_names.append(lowered)
        if mnemonic.startswith("j"):
            branch_count += 1
        tokens.append(f"op:{mnemonic}")
        tokens.append(f"ins:{mnemonic}:{op_text}")
        if op_text:
            for part in re.split(r"[,\\[\\]\+\*:\-]", op_text):
                if part and part != "n":
                    tokens.append(f"opnd:{part}")
        if insn_count >= max_insns:
            break
    for first, second in zip(call_names, call_names[1:], strict=False):
        tokens.append(f"callseq:{first[:48]}>{second[:48]}")
    return tokens, insn_count, call_count, branch_count


def string_tokens(raw: bytes, limit: int = 320) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    for blob in ASCII_RE.findall(raw):
        text = blob.decode("latin1", "ignore").strip().lower()
        if len(text) < 4 or any(part in text for part in SKIP_STRING_PARTS):
            continue
        compact = re.sub(r"\s+", "_", text)[:120]
        if compact not in seen:
            seen.add(compact)
            tokens.append(f"str:{compact}")
        for ident in IDENT_RE.findall(text):
            lowered = ident.lower()
            if lowered in {"char", "int", "long", "short", "entry_bad", "entry_good"}:
                continue
            tokens.append(f"id:{lowered}")
        if len(tokens) >= limit:
            break
    return tokens[:limit]


def symbol_tokens(raw: bytes, pe: pefile.PE, limit: int = 260) -> list[str]:
    tokens: list[str] = []
    for name, value, section_number, storage, aux_count in read_symbol_table(raw, pe):
        lowered = name.lower()
        if not lowered or lowered in {".text", ".data", ".rdata", ".bss"}:
            continue
        if section_number > 0:
            tokens.append(f"sym:{lowered[:120]}")
            tokens.append(f"symsec:{section_number}:{storage}")
            tokens.append(f"symbucket:{int(math.log2(max(value, 1)))}")
            for ident in IDENT_RE.findall(lowered):
                if ident not in {"entry_bad", "entry_good"}:
                    tokens.append(f"symid:{ident}")
        elif lowered == ".file" or storage == 103:
            tokens.append(f"aux:{lowered[:80]}")
        if aux_count:
            tokens.append(f"auxcnt:{min(aux_count, 8)}")
        if len(tokens) >= limit:
            break
    return tokens[:limit]


def extract_one(binary_id: str) -> DocRow:
    raw = (BINARY_DIR / f"{binary_id}.exe").read_bytes()
    tokens: list[str] = []
    body = b""
    insn_count = call_count = branch_count = 0
    try:
        pe = pefile.PE(data=raw, fast_load=True)
        body, start_va, addr_to_names = bad_function_payload(binary_id, raw, pe)
        dis_tokens, insn_count, call_count, branch_count = disasm_tokens(raw, pe, body, start_va, addr_to_names)
        tokens.extend(dis_tokens)
        tokens.extend(symbol_tokens(raw, pe))
    except Exception as exc:
        tokens.append(f"pe_error:{type(exc).__name__}")
    tokens.extend(string_tokens(raw))
    tokens.append(f"body_len_bucket:{int(math.log2(max(len(body), 1)))}")
    body_hex = body[:4096].hex()
    return DocRow(
        binary_id=binary_id,
        doc=" ".join(tokens),
        body_hex=body_hex,
        body_len=float(len(body)),
        insn_count=float(insn_count),
        call_count=float(call_count),
        branch_count=float(branch_count),
    )


def build_docs(binary_ids: list[str], workers: int, force: bool = False) -> pd.DataFrame:
    cached: dict[str, dict[str, object]] = {}
    converted_cache = False
    if DOC_CACHE.exists() and not force:
        loaded = joblib.load(DOC_CACHE)
        converted_cache = any(not isinstance(value, dict) for value in loaded.values())
        cached = {
            key: (value if isinstance(value, dict) else value.__dict__)
            for key, value in loaded.items()
        }
    missing = [binary_id for binary_id in binary_ids if binary_id not in cached]
    if missing:
        if workers <= 1:
            rows = [extract_one(binary_id) for binary_id in tqdm(missing, desc="enhanced", ncols=100)]
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                rows = list(tqdm(executor.map(extract_one, missing, chunksize=32), total=len(missing), desc="enhanced", ncols=100))
        for row in rows:
            cached[row.binary_id] = row.__dict__
        DOC_CACHE.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(cached, DOC_CACHE, compress=3)
    if missing or converted_cache:
        joblib.dump(cached, DOC_CACHE, compress=3)
    return pd.DataFrame([cached[binary_id] for binary_id in binary_ids])


def enhanced_matrix(docs: pd.DataFrame) -> tuple[sparse.csr_matrix, dict[str, object]]:
    word_vec = HashingVectorizer(
        n_features=2**20,
        alternate_sign=False,
        norm="l2",
        ngram_range=(1, 3),
        lowercase=False,
        token_pattern=r"(?u)\S+",
        dtype=np.float32,
    )
    hex_vec = HashingVectorizer(
        analyzer="char",
        n_features=2**18,
        alternate_sign=False,
        norm="l2",
        ngram_range=(4, 8),
        lowercase=False,
        dtype=np.float32,
    )
    word_matrix = word_vec.transform(docs["doc"].fillna(""))
    hex_matrix = hex_vec.transform(docs["body_hex"].fillna(""))
    numeric_cols = ["body_len", "insn_count", "call_count", "branch_count"]
    scaler = MaxAbsScaler()
    numeric = scaler.fit_transform(sparse.csr_matrix(docs[numeric_cols].to_numpy(dtype=np.float32)))
    matrix = sparse.hstack([word_matrix, hex_matrix, numeric], format="csr")
    return matrix, {"word_vec": word_vec, "hex_vec": hex_vec, "scaler": scaler, "numeric_cols": numeric_cols}


def transform_enhanced(docs: pd.DataFrame, artifact: dict[str, object]) -> sparse.csr_matrix:
    word_matrix = artifact["word_vec"].transform(docs["doc"].fillna(""))
    hex_matrix = artifact["hex_vec"].transform(docs["body_hex"].fillna(""))
    numeric = artifact["scaler"].transform(sparse.csr_matrix(docs[artifact["numeric_cols"]].to_numpy(dtype=np.float32)))
    return sparse.hstack([word_matrix, hex_matrix, numeric], format="csr")


def append_static_matrix(matrix: sparse.csr_matrix, frame: pd.DataFrame) -> sparse.csr_matrix:
    features = pd.read_pickle(FEATURE_CACHE)
    merged = frame[["binary_id", "target"]].merge(features, on=["binary_id", "target"], how="left", validate="one_to_one")
    static_matrix, _, _, _ = prepare_matrices(merged, 2**19, "l2")
    return sparse.hstack([matrix, static_matrix], format="csr")


def top2_margin(scores: np.ndarray) -> float:
    if scores.ndim == 0:
        return abs(float(scores))
    if scores.shape[0] == 1:
        return abs(float(scores[0]))
    top = np.partition(scores, -2)[-2:]
    return float(top[-1] - top[-2])


def classifier_scores(clf: LinearSVC, matrix: sparse.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    pred = clf.predict(matrix)
    raw_scores = clf.decision_function(matrix)
    if raw_scores.ndim == 1:
        margins = np.abs(raw_scores)
    else:
        margins = np.array([top2_margin(row) for row in raw_scores], dtype=np.float32)
    return pred, margins


def train_group_models(matrix: sparse.csr_matrix, labels: np.ndarray, train_idx: pd.Index, c: float) -> dict[str, LinearSVC]:
    models: dict[str, LinearSVC] = {}
    for name, classes in GROUPS.items():
        cls_set = set(classes)
        idx = [i for i in train_idx if labels[i] in cls_set]
        present = sorted(set(labels[idx]))
        if len(present) < 2:
            continue
        clf = LinearSVC(C=c, class_weight="balanced", random_state=SEED, max_iter=12000)
        clf.fit(matrix[idx], labels[idx])
        models[name] = clf
    return models


def apply_group_models(
    base_pred: list[str],
    matrix: sparse.csr_matrix,
    valid_idx: pd.Index,
    models: dict[str, LinearSVC],
    thresholds: dict[str, float],
) -> tuple[list[str], dict[str, int]]:
    pred = list(base_pred)
    stats = Counter()
    local_pos = {idx: pos for pos, idx in enumerate(valid_idx)}
    for name, clf in models.items():
        group = set(GROUPS[name])
        candidate_positions = [pos for pos, value in enumerate(pred) if value in group]
        if not candidate_positions:
            continue
        candidate_idx = [valid_idx[pos] for pos in candidate_positions]
        group_pred, margins = classifier_scores(clf, matrix[candidate_idx])
        threshold = thresholds.get(name, float("inf"))
        for out_pos, new_target, margin in zip(candidate_positions, group_pred, margins, strict=False):
            if float(margin) < threshold:
                continue
            if new_target != pred[out_pos]:
                pred[out_pos] = str(new_target)
                stats[f"changed_{name}"] += 1
    return pred, dict(stats)


def build_base_valid_predictions(min_support: int, min_purity: float) -> pd.DataFrame:
    train = pd.read_csv(RAW / "train.csv")
    train["target"] = train["cwe_id"].fillna(NO_VULN)
    train_idx, valid_idx = build_split(train, 0.2)
    rules = build_rules(train.loc[train_idx].copy(), min_support, min_purity)
    pred_frame = pd.read_csv(PROCESSED.parents[0] / "report" / "symbol_hybrid_valid_errors.csv")
    pred_map = pred_frame.set_index("binary_id")["pred"].to_dict()
    rows = []
    for row in train.loc[valid_idx].itertuples(index=False):
        old = pred_map[row.binary_id]
        new = old
        symbol_label, _, _ = predict_one(row.binary_id)
        if symbol_label == 1:
            rule_cwe = predict_cwe(row.binary_id, rules)
            if rule_cwe:
                new = rule_cwe
        rows.append((row.binary_id, row.target, new))
    return pd.DataFrame(rows, columns=["binary_id", "true", "pred"])


def evaluate(workers: int, force_docs: bool, c: float, include_static: bool, min_support: int, min_purity: float) -> None:
    train = pd.read_csv(RAW / "train.csv")
    train["target"] = train["cwe_id"].fillna(NO_VULN)
    train_idx, valid_idx = build_split(train, 0.2)
    labels = train["target"].to_numpy()

    docs = build_docs(train["binary_id"].tolist(), workers, force=force_docs)
    matrix, artifact = enhanced_matrix(docs)
    if include_static:
        matrix = append_static_matrix(matrix, train)

    base = build_base_valid_predictions(min_support, min_purity)
    base_pred = base["pred"].tolist()
    y_true = base["true"].to_numpy()
    print(f"base_macro={f1_score(y_true, base_pred, average='macro'):.6f}")

    models = train_group_models(matrix, labels, train_idx, c)
    print(f"models={list(models)}")
    raw_group_preds: dict[str, tuple[np.ndarray, np.ndarray, list[int]]] = {}
    for name, clf in models.items():
        group = set(GROUPS[name])
        candidate_positions = [pos for pos, value in enumerate(base_pred) if value in group]
        candidate_idx = [valid_idx[pos] for pos in candidate_positions]
        if not candidate_idx:
            continue
        raw_group_preds[name] = (*classifier_scores(clf, matrix[candidate_idx]), candidate_positions)

    thresholds_grid = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
    chosen: dict[str, float] = {}
    current_pred = list(base_pred)
    current_score = f1_score(y_true, current_pred, average="macro")
    for name in models:
        best_score = current_score
        best_threshold = float("inf")
        best_pred = list(current_pred)
        for threshold in thresholds_grid:
            trial = list(current_pred)
            group_pred, margins, positions = raw_group_preds.get(name, (np.array([]), np.array([]), []))
            for pos, new_target, margin in zip(positions, group_pred, margins, strict=False):
                if float(margin) >= threshold:
                    trial[pos] = str(new_target)
            score = f1_score(y_true, trial, average="macro")
            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_pred = trial
        chosen[name] = best_threshold
        current_pred = best_pred
        current_score = best_score
        print(f"group={name} threshold={best_threshold} macro={current_score:.6f}")

    stats = Counter()
    for old, new, true in zip(base_pred, current_pred, y_true, strict=False):
        if old != new:
            stats["changed"] += 1
            stats["useful"] += int(new == true and old != true)
            stats["harmful"] += int(new != true and old == true)
    print(f"final_macro={current_score:.6f}")
    print(f"thresholds={chosen}")
    print(f"stats={dict(stats)}")
    out = base.copy()
    out["pred"] = current_pred
    out_path = PROCESSED.parents[0] / "report" / "enhanced_calibrator_valid_predictions.csv"
    out.to_csv(out_path, index=False, encoding="utf-8")
    errors = out[out["true"] != out["pred"]]
    print("error_pairs")
    print(errors.groupby(["true", "pred"]).size().sort_values(ascending=False).head(50).to_string())
    print(out_path)


def generate(
    workers: int,
    force_docs: bool,
    c: float,
    include_static: bool,
    min_support: int,
    min_purity: float,
    thresholds: dict[str, float],
    input_name: str,
    output: str,
) -> None:
    train = pd.read_csv(RAW / "train.csv")
    train["target"] = train["cwe_id"].fillna(NO_VULN)
    test = load_test()
    all_ids = pd.concat([train["binary_id"], test["binary_id"]], ignore_index=True).tolist()
    docs = build_docs(all_ids, workers, force=force_docs)
    train_docs = docs.iloc[: len(train)].reset_index(drop=True)
    test_docs = docs.iloc[len(train) :].reset_index(drop=True)
    train_matrix, artifact = enhanced_matrix(train_docs)
    test_matrix = transform_enhanced(test_docs, artifact)
    if include_static:
        features = pd.read_pickle(FEATURE_CACHE)
        train_static_frame = train[["binary_id", "target"]].merge(features, on=["binary_id", "target"], how="left", validate="one_to_one")
        test_static_frame = test.merge(features, on="binary_id", how="left", validate="one_to_one")
        static_train, numeric_cols, vectorizer, scaler = prepare_matrices(train_static_frame, 2**19, "l2")
        test_text = vectorizer.transform(test_static_frame["doc"].fillna(""))
        test_numeric = scaler.transform(dataframe_to_sparse(test_static_frame, numeric_cols))
        static_test = sparse.hstack([test_text, test_numeric], format="csr")
        train_matrix = sparse.hstack([train_matrix, static_train], format="csr")
        test_matrix = sparse.hstack([test_matrix, static_test], format="csr")

    labels = train["target"].to_numpy()
    full_idx = pd.Index(train.index[labels != NO_VULN])
    models = train_group_models(train_matrix, labels, full_idx, c)
    sub = pd.read_csv(SUBMISSIONS / input_name)
    if list(sub.columns) == ["name", "label", "cwe_id"]:
        sub = sub.rename(columns={"name": "binary_id"})
    base_targets = [NO_VULN if row.label == 0 else row.cwe_id for row in sub.itertuples(index=False)]
    final_targets = list(base_targets)
    stats = Counter()
    for name, clf in models.items():
        group = set(GROUPS[name])
        positions = [pos for pos, value in enumerate(final_targets) if value in group]
        if not positions:
            continue
        group_pred, margins = classifier_scores(clf, test_matrix[positions])
        threshold = thresholds.get(name, float("inf"))
        for pos, new_target, margin in zip(positions, group_pred, margins, strict=False):
            if float(margin) >= threshold and new_target != final_targets[pos]:
                final_targets[pos] = str(new_target)
                stats[f"changed_{name}"] += 1

    out = pd.DataFrame(
        {
            "binary_id": test["binary_id"],
            "label": [0 if target == NO_VULN else 1 for target in final_targets],
            "cwe_id": ["" if target == NO_VULN else target for target in final_targets],
        }
    )
    validate_submission(out, test["binary_id"])
    output_path = SUBMISSIONS / output
    out.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(output_path)
    print(f"stats={dict(stats)}")
    print(f"label_dist={out['label'].value_counts().to_dict()}")
    print(f"top_cwe={out.loc[out['label'] == 1, 'cwe_id'].value_counts().head(20).to_dict()}")


def parse_thresholds(value: str) -> dict[str, float]:
    if not value:
        return {}
    result: dict[str, float] = {}
    for item in value.split(","):
        key, raw_val = item.split("=", 1)
        parsed = float(raw_val)
        if math.isfinite(parsed):
            result[key.strip()] = parsed
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Targeted CWE calibrator for high-confusion groups.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force-docs", action="store_true")
    parser.add_argument("--c", type=float, default=0.8)
    parser.add_argument("--include-static", action="store_true")
    parser.add_argument("--min-support", type=int, default=5)
    parser.add_argument("--min-purity", type=float, default=1.0)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--thresholds", default="")
    parser.add_argument("--input", default="submission_symbol_hybrid_string_rule_v4_utf8_sig.csv")
    parser.add_argument("--output", default="submission_enhanced_calibrator_v5_utf8_sig.csv")
    args = parser.parse_args()
    ensure_dirs()
    if args.eval:
        evaluate(args.workers, args.force_docs, args.c, args.include_static, args.min_support, args.min_purity)
    if args.generate:
        generate(
            args.workers,
            args.force_docs,
            args.c,
            args.include_static,
            args.min_support,
            args.min_purity,
            parse_thresholds(args.thresholds),
            args.input,
            args.output,
        )


if __name__ == "__main__":
    main()
