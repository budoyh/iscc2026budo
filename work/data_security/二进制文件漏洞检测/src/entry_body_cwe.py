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
from sklearn.svm import LinearSVC
from tqdm import tqdm

from common import BINARY_DIR, MODELS, NO_VULN, RAW, SEED, SUBMISSIONS, ensure_dirs
from inspect_symbols import symbols
from postprocess_submission import top_cwe
from predict import validate_submission
from probe_leakage import file_bytes, text_hash
from symbol_rule_predict import called_symbols_from_main, predict_one, section_for_symbol
from train import build_split


ENTRY_BODY_MODEL_PATH = MODELS / "entry_body_cwe.joblib"


def choose_bad_symbol(binary_id: str) -> str:
    calls = called_symbols_from_main(binary_id)
    for name in calls:
        lowered = name.lower()
        if "entry_bad" in lowered or ("bad" in lowered and "good" not in lowered):
            return name
    for name, _, sec, _ in symbols(binary_id):
        lowered = name.lower()
        if sec > 0 and ("entry_bad" in lowered or lowered == "bad"):
            return name
    return ""


def function_body(binary_id: str, symbol_name: str, max_bytes: int = 4096) -> str:
    raw = (BINARY_DIR / f"{binary_id}.exe").read_bytes()
    pe = pefile.PE(data=raw, fast_load=True)
    rows = symbols(binary_id)
    target = None
    code_symbols = []
    for name, value, sec, storage in rows:
        if sec > 0:
            code_symbols.append((name, value, sec))
        if name == symbol_name and sec > 0:
            target = (value, sec)
    if target is None:
        return ""
    value, sec = target
    section = section_for_symbol(pe, sec)
    if section is None:
        return ""
    next_values = sorted(v for _, v, s in code_symbols if s == sec and v > value)
    end = next_values[0] if next_values else value + max_bytes
    body = section.get_data()[value:end][:max_bytes]
    return body.decode("latin1", "ignore")


def entry_body_text(binary_id: str) -> str:
    symbol_name = choose_bad_symbol(binary_id)
    return function_body(binary_id, symbol_name) if symbol_name else ""


def build_texts(binary_ids: list[str], workers: int) -> list[str]:
    if workers <= 1:
        return [entry_body_text(binary_id) for binary_id in tqdm(binary_ids, desc="entry", ncols=100)]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(tqdm(executor.map(entry_body_text, binary_ids, chunksize=64), total=len(binary_ids), desc="entry", ncols=100))


def build_pure_text_hash_map(train: pd.DataFrame) -> dict[str, str]:
    groups: dict[str, list[str]] = defaultdict(list)
    for row in train.itertuples(index=False):
        groups[text_hash(file_bytes(row.binary_id))].append(row.target)
    return {sig: values[0] for sig, values in groups.items() if values and len(set(values)) == 1}


def final_targets(binary_ids: list[str], cwe_pred: np.ndarray, pure_hash: dict[str, str], use_hint: bool) -> list[str]:
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


def evaluate(workers: int, n_features: int, ngram_min: int, ngram_max: int, c: float) -> None:
    train = pd.read_csv(RAW / "train.csv")
    train["target"] = train["cwe_id"].fillna(NO_VULN)
    train_idx, valid_idx = build_split(train, 0.2)
    pos_train_idx = train_idx[train.loc[train_idx, "target"] != NO_VULN]
    texts = build_texts(train["binary_id"].tolist(), workers)
    vectorizer = HashingVectorizer(
        analyzer="char",
        ngram_range=(ngram_min, ngram_max),
        n_features=n_features,
        alternate_sign=False,
        norm="l2",
        lowercase=False,
        dtype=np.float32,
    )
    matrix = vectorizer.transform(texts)
    labels = train["target"].to_numpy()
    clf = LinearSVC(C=c, class_weight="balanced", random_state=SEED, max_iter=12000)
    clf.fit(matrix[pos_train_idx], labels[pos_train_idx])
    cwe_pred = clf.predict(matrix[valid_idx])
    pure_hash = build_pure_text_hash_map(train.loc[train_idx].copy())
    valid_ids = train.loc[valid_idx, "binary_id"].tolist()
    pred_nohint = final_targets(valid_ids, cwe_pred, pure_hash, use_hint=False)
    pred_hint = final_targets(valid_ids, cwe_pred, pure_hash, use_hint=True)
    y_true = labels[valid_idx]
    print(f"entry_body_macro={f1_score(y_true, pred_nohint, average='macro'):.6f}")
    print(f"entry_body_hint_macro={f1_score(y_true, pred_hint, average='macro'):.6f}")
    pos_mask = y_true != NO_VULN
    print(f"entry_body_positive_acc={float(np.mean(np.array(pred_hint, dtype=object)[pos_mask] == y_true[pos_mask])):.6f}")


def train_full_and_predict(workers: int, n_features: int, ngram_min: int, ngram_max: int, c: float, output: str) -> None:
    train = pd.read_csv(RAW / "train.csv")
    test = pd.read_csv(RAW / "test.csv")
    train["target"] = train["cwe_id"].fillna(NO_VULN)
    all_ids = pd.concat([train["binary_id"], test["binary_id"]], ignore_index=True).tolist()
    texts = build_texts(all_ids, workers)
    train_texts = texts[: len(train)]
    test_texts = texts[len(train) :]
    vectorizer = HashingVectorizer(
        analyzer="char",
        ngram_range=(ngram_min, ngram_max),
        n_features=n_features,
        alternate_sign=False,
        norm="l2",
        lowercase=False,
        dtype=np.float32,
    )
    train_matrix = vectorizer.transform(train_texts)
    test_matrix = vectorizer.transform(test_texts)
    pos_mask = train["target"] != NO_VULN
    clf = LinearSVC(C=c, class_weight="balanced", random_state=SEED, max_iter=12000)
    clf.fit(train_matrix[pos_mask.to_numpy()], train.loc[pos_mask, "target"].to_numpy())
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
    joblib.dump({"model": clf, "n_features": n_features, "ngram_range": (ngram_min, ngram_max), "c": c}, ENTRY_BODY_MODEL_PATH, compress=3)
    print(output_path)
    print(f"label_dist={out['label'].value_counts().to_dict()}")
    print(f"top_cwe={out.loc[out['label'] == 1, 'cwe_id'].value_counts().head(15).to_dict()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CWE classifier from entry_bad function body bytes.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--n-features", type=int, default=2**18)
    parser.add_argument("--ngram-min", type=int, default=2)
    parser.add_argument("--ngram-max", type=int, default=5)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--output", default="submission_entry_body_symbol_utf8_sig.csv")
    args = parser.parse_args()
    ensure_dirs()
    if args.eval:
        evaluate(args.workers, args.n_features, args.ngram_min, args.ngram_max, args.c)
    if args.predict:
        train_full_and_predict(args.workers, args.n_features, args.ngram_min, args.ngram_max, args.c, args.output)


if __name__ == "__main__":
    main()
