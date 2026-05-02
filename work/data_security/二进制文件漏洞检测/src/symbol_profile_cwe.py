from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd
import pefile
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import f1_score
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC
from tqdm import tqdm

from common import BINARY_DIR, MODELS, NO_VULN, RAW, SEED, SUBMISSIONS, dataframe_to_sparse, ensure_dirs
from inspect_symbols import symbols
from postprocess_submission import top_cwe
from predict import validate_submission
from probe_leakage import file_bytes, text_hash
from symbol_rule_predict import predict_one
from train import build_split


SYMBOL_PROFILE_MODEL_PATH = MODELS / "symbol_profile_cwe.joblib"
RUNTIME_PARTS = (
    "crt",
    "mingw",
    "ucrt",
    "libgcc",
    "dll",
    "tls",
    "__imp_",
    "__native",
    "__getmainargs",
    "__main",
)


def should_keep_symbol(name: str) -> bool:
    lowered = name.lower()
    if lowered in {".text", ".data", ".bss", ".rdata", ".pdata", ".xdata", ".idata", ".reloc"}:
        return False
    if any(part in lowered for part in RUNTIME_PARTS):
        return False
    return any(
        part in lowered
        for part in (
            "entry",
            "bad",
            "good",
            "source",
            "sink",
            "data",
            "cwe",
            "alloc",
            "free",
            "copy",
            "mem",
            "str",
            "socket",
            "file",
            "print",
            "scan",
            "char",
            "int",
        )
    )


def next_symbol_value(code_symbols: list[tuple[str, int, int]], value: int, sec: int) -> int:
    values = sorted(v for _, v, s in code_symbols if s == sec and v > value)
    return values[0] if values else value


def extract_profile(binary_id: str) -> dict[str, object]:
    rows = symbols(binary_id)
    code_symbols = [(name, value, sec) for name, value, sec, _ in rows if sec > 0]
    features: dict[str, object] = {
        "binary_id": binary_id,
        "coff_symbol_count": float(len(rows)),
        "code_symbol_count": float(len(code_symbols)),
    }
    doc_tokens: list[str] = []
    app_symbols: list[tuple[str, int, int, int]] = []
    for name, value, sec in code_symbols:
        if should_keep_symbol(name):
            end = next_symbol_value(code_symbols, value, sec)
            length = max(end - value, 0)
            app_symbols.append((name, value, sec, length))
            clean = name.lower()
            doc_tokens.append(f"sym:{clean}")
            for sep in ["__", "_", "2", "3", "4", "5", "9", "10"]:
                clean = clean.replace(sep, " ")
            for part in clean.replace("(", " ").replace(")", " ").split():
                if len(part) >= 2:
                    doc_tokens.append(f"part:{part[:48]}")

    features["app_symbol_count"] = float(len(app_symbols))
    lengths = [item[3] for item in app_symbols]
    features["app_len_mean"] = float(np.mean(lengths)) if lengths else 0.0
    features["app_len_std"] = float(np.std(lengths)) if lengths else 0.0
    features["app_len_max"] = float(np.max(lengths)) if lengths else 0.0
    features["app_len_min"] = float(np.min(lengths)) if lengths else 0.0

    for key in [
        "entry_bad",
        "entry_good",
        "badsource",
        "bad_source",
        "badsink",
        "bad_sink",
        "goodsource",
        "good_sink",
        "goodg2b",
        "goodb2g",
        "good1",
        "good2",
    ]:
        features[f"sym_has::{key}"] = 0.0
        features[f"sym_count::{key}"] = 0.0
        features[f"sym_len::{key}"] = 0.0

    for name, value, sec, length in app_symbols:
        lowered = name.lower()
        for key in [
            "entry_bad",
            "entry_good",
            "badsource",
            "bad_source",
            "badsink",
            "bad_sink",
            "goodsource",
            "good_sink",
            "goodg2b",
            "goodb2g",
            "good1",
            "good2",
        ]:
            if key in lowered:
                features[f"sym_has::{key}"] = 1.0
                features[f"sym_count::{key}"] += 1.0
                features[f"sym_len::{key}"] += float(length)

    features["symbol_doc"] = " ".join(doc_tokens)
    return features


def build_profiles(binary_ids: list[str], workers: int) -> pd.DataFrame:
    if workers <= 1:
        rows = [extract_profile(binary_id) for binary_id in tqdm(binary_ids, desc="symprof", ncols=100)]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(
                tqdm(executor.map(extract_profile, binary_ids, chunksize=64), total=len(binary_ids), desc="symprof", ncols=100)
            )
    return pd.DataFrame(rows)


def make_matrix(frame: pd.DataFrame, text_dim: int):
    numeric_cols = [col for col in frame.columns if col not in {"binary_id", "symbol_doc", "target"}]
    vectorizer = HashingVectorizer(
        n_features=text_dim,
        alternate_sign=False,
        norm="l2",
        ngram_range=(1, 2),
        lowercase=False,
        token_pattern=r"(?u)\S+",
        dtype=np.float32,
    )
    text_matrix = vectorizer.transform(frame["symbol_doc"].fillna(""))
    numeric_matrix = dataframe_to_sparse(frame, numeric_cols)
    scaler = MaxAbsScaler()
    numeric_matrix = scaler.fit_transform(numeric_matrix)
    return sparse.hstack([text_matrix, numeric_matrix], format="csr"), vectorizer, scaler, numeric_cols


def build_pure_text_hash_map(train: pd.DataFrame) -> dict[str, str]:
    groups: dict[str, list[str]] = defaultdict(list)
    for row in train.itertuples(index=False):
        groups[text_hash(file_bytes(row.binary_id))].append(row.target)
    return {sig: values[0] for sig, values in groups.items() if values and len(set(values)) == 1}


def final_targets(binary_ids: list[str], cwe_pred: np.ndarray, pure_hash: dict[str, str], use_hint: bool = True) -> list[str]:
    targets: list[str] = []
    for binary_id, pred in zip(binary_ids, cwe_pred, strict=False):
        symbol_label, _, _ = predict_one(binary_id)
        if symbol_label == 0:
            targets.append(NO_VULN)
            continue
        target = str(pred)
        if symbol_label == 1:
            mapped = pure_hash.get(text_hash(file_bytes(binary_id)))
            if mapped and mapped != NO_VULN:
                target = mapped
            hint = top_cwe(binary_id) if use_hint else ""
            if hint:
                target = hint
        targets.append(target)
    return targets


def evaluate(workers: int, text_dim: int, c: float) -> None:
    train = pd.read_csv(RAW / "train.csv")
    train["target"] = train["cwe_id"].fillna(NO_VULN)
    profiles = build_profiles(train["binary_id"].tolist(), workers).merge(
        train[["binary_id", "target"]], on="binary_id", how="left", validate="one_to_one"
    )
    matrix, _, _, _ = make_matrix(profiles, text_dim)
    train_idx, valid_idx = build_split(train, 0.2)
    labels = train["target"].to_numpy()
    pos_train_idx = train_idx[labels[train_idx] != NO_VULN]
    clf = LinearSVC(C=c, class_weight="balanced", random_state=SEED, max_iter=12000)
    clf.fit(matrix[pos_train_idx], labels[pos_train_idx])
    cwe_pred = clf.predict(matrix[valid_idx])
    pure_hash = build_pure_text_hash_map(train.loc[train_idx].copy())
    valid_ids = train.loc[valid_idx, "binary_id"].tolist()
    pred = final_targets(valid_ids, cwe_pred, pure_hash, use_hint=True)
    y_true = labels[valid_idx]
    print(f"symbol_profile_hint_macro={f1_score(y_true, pred, average='macro'):.6f}")
    pos_mask = y_true != NO_VULN
    print(f"symbol_profile_positive_acc={float(np.mean(np.array(pred, dtype=object)[pos_mask] == y_true[pos_mask])):.6f}")


def train_full_and_predict(workers: int, text_dim: int, c: float, output: str) -> None:
    train = pd.read_csv(RAW / "train.csv")
    test = pd.read_csv(RAW / "test.csv")
    train["target"] = train["cwe_id"].fillna(NO_VULN)
    all_ids = pd.concat([train["binary_id"], test["binary_id"]], ignore_index=True).tolist()
    profiles = build_profiles(all_ids, workers)
    train_profiles = profiles.iloc[: len(train)].copy()
    test_profiles = profiles.iloc[len(train) :].copy()
    train_profiles = train_profiles.merge(train[["binary_id", "target"]], on="binary_id", how="left", validate="one_to_one")
    matrix, vectorizer, scaler, numeric_cols = make_matrix(train_profiles, text_dim)
    labels = train_profiles["target"].to_numpy()
    pos_mask = labels != NO_VULN
    clf = LinearSVC(C=c, class_weight="balanced", random_state=SEED, max_iter=12000)
    clf.fit(matrix[pos_mask], labels[pos_mask])
    test_text = vectorizer.transform(test_profiles["symbol_doc"].fillna(""))
    test_numeric = dataframe_to_sparse(test_profiles, numeric_cols)
    test_numeric = scaler.transform(test_numeric)
    test_matrix = sparse.hstack([test_text, test_numeric], format="csr")
    cwe_pred = clf.predict(test_matrix)
    pure_hash = build_pure_text_hash_map(train)
    targets = final_targets(test["binary_id"].tolist(), cwe_pred, pure_hash, use_hint=True)
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
    joblib.dump(
        {"model": clf, "vectorizer": vectorizer, "scaler": scaler, "numeric_cols": numeric_cols, "text_dim": text_dim, "c": c},
        SYMBOL_PROFILE_MODEL_PATH,
        compress=3,
    )
    print(output_path)
    print(f"label_dist={out['label'].value_counts().to_dict()}")
    print(f"top_cwe={out.loc[out['label'] == 1, 'cwe_id'].value_counts().head(15).to_dict()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CWE classifier from COFF symbol profile.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--text-dim", type=int, default=2**18)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--output", default="submission_symbol_profile_utf8_sig.csv")
    args = parser.parse_args()
    ensure_dirs()
    if args.eval:
        evaluate(args.workers, args.text_dim, args.c)
    if args.predict:
        train_full_and_predict(args.workers, args.text_dim, args.c, args.output)


if __name__ == "__main__":
    main()
