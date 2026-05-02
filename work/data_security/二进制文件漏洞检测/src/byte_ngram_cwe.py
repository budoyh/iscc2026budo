from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

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
from postprocess_submission import top_cwe
from predict import validate_submission
from probe_leakage import file_bytes, text_hash
from symbol_rule_predict import predict_one
from train import build_split


BYTE_MODEL_PATH = MODELS / "byte_ngram_cwe.joblib"
EXECUTABLE_SCN_MASK = 0x20000000


def executable_bytes(binary_id: str, max_bytes: int = 32768) -> str:
    raw = (BINARY_DIR / f"{binary_id}.exe").read_bytes()
    chunks: list[bytes] = []
    try:
        pe = pefile.PE(data=raw, fast_load=True)
        for section in pe.sections:
            if section.Characteristics & EXECUTABLE_SCN_MASK:
                chunks.append(section.get_data())
    except Exception:
        chunks = [raw]
    data = b"".join(chunks)[:max_bytes]
    return data.decode("latin1", "ignore")


def build_texts(binary_ids: list[str], workers: int) -> list[str]:
    if workers <= 1:
        return [executable_bytes(binary_id) for binary_id in tqdm(binary_ids, desc="bytes", ncols=100)]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(tqdm(executor.map(executable_bytes, binary_ids, chunksize=64), total=len(binary_ids), desc="bytes", ncols=100))


def build_pure_text_hash_map(train: pd.DataFrame) -> dict[str, str]:
    groups: dict[str, list[str]] = {}
    for row in train.itertuples(index=False):
        groups.setdefault(text_hash(file_bytes(row.binary_id)), []).append(row.target)
    return {sig: values[0] for sig, values in groups.items() if values and len(set(values)) == 1}


def symbol_hybrid_predict(binary_ids: list[str], cwe_pred: np.ndarray, pure_hash: dict[str, str], use_hint: bool = True) -> list[str]:
    final: list[str] = []
    for binary_id, model_cwe in zip(binary_ids, cwe_pred, strict=False):
        symbol_label, _, _ = predict_one(binary_id)
        if symbol_label == 0:
            target = NO_VULN
        elif symbol_label == 1:
            target = str(model_cwe)
            mapped = pure_hash.get(text_hash(file_bytes(binary_id)))
            if mapped and mapped != NO_VULN:
                target = mapped
            hint = top_cwe(binary_id) if use_hint else ""
            if hint:
                target = hint
        else:
            target = str(model_cwe)
        final.append(target)
    return final


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
    binary_ids = train.loc[valid_idx, "binary_id"].tolist()
    pred_plain = symbol_hybrid_predict(binary_ids, cwe_pred, pure_hash, use_hint=False)
    pred_hint = symbol_hybrid_predict(binary_ids, cwe_pred, pure_hash, use_hint=True)
    y_true = labels[valid_idx]
    print(f"byte_symbol_macro={f1_score(y_true, pred_plain, average='macro'):.6f}")
    print(f"byte_symbol_hint_macro={f1_score(y_true, pred_hint, average='macro'):.6f}")
    pos_mask = y_true != NO_VULN
    print(f"byte_cwe_positive_acc={float(np.mean(np.array(pred_hint, dtype=object)[pos_mask] == y_true[pos_mask])):.6f}")


def train_full_and_predict(workers: int, n_features: int, ngram_min: int, ngram_max: int, c: float, output: str) -> None:
    train = pd.read_csv(RAW / "train.csv")
    test = pd.read_csv(RAW / "test.csv")
    train["target"] = train["cwe_id"].fillna(NO_VULN)
    pos_mask = train["target"] != NO_VULN
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
    clf = LinearSVC(C=c, class_weight="balanced", random_state=SEED, max_iter=12000)
    clf.fit(train_matrix[pos_mask.to_numpy()], train.loc[pos_mask, "target"].to_numpy())
    cwe_pred = clf.predict(test_matrix)
    pure_hash = build_pure_text_hash_map(train)
    final_targets = symbol_hybrid_predict(test["binary_id"].tolist(), cwe_pred, pure_hash, use_hint=True)
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
    joblib.dump({"model": clf, "n_features": n_features, "ngram_range": (ngram_min, ngram_max), "c": c}, BYTE_MODEL_PATH, compress=3)
    print(output_path)
    print(f"label_dist={out['label'].value_counts().to_dict()}")
    print(f"top_cwe={out.loc[out['label'] == 1, 'cwe_id'].value_counts().head(15).to_dict()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Byte/char n-gram CWE classifier for executable sections.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--n-features", type=int, default=2**20)
    parser.add_argument("--ngram-min", type=int, default=4)
    parser.add_argument("--ngram-max", type=int, default=7)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--output", default="submission_byte_ngram_symbol_utf8_sig.csv")
    args = parser.parse_args()
    ensure_dirs()
    if args.eval:
        evaluate(args.workers, args.n_features, args.ngram_min, args.ngram_max, args.c)
    if args.predict:
        train_full_and_predict(args.workers, args.n_features, args.ngram_min, args.ngram_max, args.c, args.output)


if __name__ == "__main__":
    main()
