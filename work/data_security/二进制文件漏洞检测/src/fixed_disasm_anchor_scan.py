from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import pandas as pd
from tqdm import tqdm

from common import PROCESSED, RAW, SUBMISSIONS
from enhanced_cwe_calibrator import extract_one


KEY_COLS = (
    "key_body_hex",
    "key_ins_full",
    "key_op_full",
    "key_ins_tail36",
    "key_ins_tail30",
    "key_ins_tail24",
    "key_ins_tail18",
    "key_marker_tail",
    "key_callseq",
    "key_len_ins_calls",
)


def digest(parts: list[str]) -> str:
    return hashlib.sha1(" ".join(parts).encode("utf-8", "ignore")).hexdigest()[:16]


def extract_fixed_doc(binary_id: str) -> dict[str, object]:
    row = extract_one(binary_id).__dict__
    tokens = str(row["doc"]).split()
    ins = [token[4:] for token in tokens if token.startswith("ins:")]
    ops = [token[3:] for token in tokens if token.startswith("op:")]
    calls = [token[9:] for token in tokens if token.startswith("callname:") and token != "callname:.text"]
    markers: list[str] = []
    for insn in ins:
        if any(name in insn for name in ("memcpy", "strncpy", "wmemcpy", "wcscpy", "strcpy", "malloc", "free")):
            markers.append(insn)
        elif any(mem in insn for mem in ("[rbp-", "[rax+", "[rsp+", "[rcx+", "[rdx+")):
            markers.append(insn)
    return {
        "binary_id": binary_id,
        "body_len": int(row["body_len"]),
        "insn_count": int(row["insn_count"]),
        "call_count": int(row["call_count"]),
        "branch_count": int(row["branch_count"]),
        "key_body_hex": hashlib.sha1(str(row["body_hex"]).encode("utf-8", "ignore")).hexdigest()[:16],
        "key_ins_full": digest(ins),
        "key_op_full": digest(ops),
        "key_ins_tail36": digest(ins[-36:]),
        "key_ins_tail30": digest(ins[-30:]),
        "key_ins_tail24": digest(ins[-24:]),
        "key_ins_tail18": digest(ins[-18:]),
        "key_marker_tail": digest(markers[-48:]),
        "key_callseq": "|".join(calls),
        "key_len_ins_calls": f"{int(row['body_len'])}|{int(row['insn_count'])}|{int(row['call_count'])}|{int(row['branch_count'])}",
        "ins_preview": " ".join(ins[:90])[:1400],
    }


def build_docs(binary_ids: list[str], cache_path: Path, workers: int, force: bool) -> pd.DataFrame:
    cached: dict[str, dict[str, object]] = {}
    if cache_path.exists() and not force:
        cached = joblib.load(cache_path)
    missing = [binary_id for binary_id in binary_ids if binary_id not in cached]
    if missing:
        if workers <= 1:
            rows = [extract_fixed_doc(binary_id) for binary_id in tqdm(missing, desc="fixed-disasm", ncols=100)]
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                rows = list(
                    tqdm(
                        executor.map(extract_fixed_doc, missing, chunksize=24),
                        total=len(missing),
                        desc="fixed-disasm",
                        ncols=100,
                    )
                )
        for row in rows:
            cached[str(row["binary_id"])] = row
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(cached, cache_path, compress=3)
    return pd.DataFrame([cached[binary_id] for binary_id in binary_ids])


def leave_one_out_stats(train_docs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key_col in KEY_COLS:
        groups = train_docs.groupby(key_col)["cwe_id"].apply(list)
        covered = correct = wrong = 0
        by_support: dict[int, list[int]] = defaultdict(lambda: [0, 0])
        for _, row in train_docs.iterrows():
            values = list(groups[row[key_col]])
            values.remove(row["cwe_id"])
            counts = Counter(values)
            support = sum(counts.values())
            if support >= 2 and len(counts) == 1:
                pred = next(iter(counts))
                ok = pred == row["cwe_id"]
                covered += 1
                correct += int(ok)
                wrong += int(not ok)
                by_support[support][0] += 1
                by_support[support][1] += int(ok)
        rows.append(
            {
                "key_type": key_col,
                "covered": covered,
                "correct": correct,
                "wrong": wrong,
                "acc": correct / covered if covered else 0.0,
                "support_breakdown": json.dumps(dict(sorted(by_support.items())), ensure_ascii=False),
            }
        )
    return pd.DataFrame(rows).sort_values(["acc", "covered"], ascending=[False, False])


def scan_candidates(train_docs: pd.DataFrame, test_docs: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    stats = leave_one_out_stats(train_docs).set_index("key_type")
    for key_col in KEY_COLS:
        grouped = train_docs.groupby(key_col)["cwe_id"].agg(lambda values: dict(Counter(values))).reset_index(name="counts")
        grouped["support"] = grouped["counts"].apply(lambda counts: sum(counts.values()))
        grouped["pred"] = grouped["counts"].apply(lambda counts: max(counts, key=counts.get))
        grouped["purity"] = grouped["counts"].apply(lambda counts: max(counts.values()) / sum(counts.values()))
        pure = grouped[(grouped["support"] >= 2) & (grouped["purity"] == 1.0)].set_index(key_col)
        frame = test_docs[["binary_id", "source_id_num", "label", "cwe_id", key_col, "ins_preview"]].copy()
        frame["key_type"] = key_col
        frame["pred"] = frame[key_col].map(pure["pred"])
        frame["support"] = frame[key_col].map(pure["support"])
        frame["purity"] = frame[key_col].map(pure["purity"])
        frame["counts"] = frame[key_col].map(pure["counts"])
        frame = frame[frame["pred"].notna() & (frame["pred"] != frame["cwe_id"])].copy()
        if not frame.empty:
            frame["key_loo_acc"] = float(stats.loc[key_col, "acc"])
            frame["key_loo_wrong"] = int(stats.loc[key_col, "wrong"])
            pieces.append(frame)
    if not pieces:
        return pd.DataFrame()
    out = pd.concat(pieces, ignore_index=True)
    out["support"] = out["support"].astype(int)
    return out.sort_values(["binary_id", "key_loo_acc", "support"], ascending=[True, False, False])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="s17b.csv")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    train = pd.read_csv(RAW / "train.csv")
    sub = pd.read_csv(SUBMISSIONS / args.input)
    source_ids = pd.read_csv(PROCESSED / "source_ids.csv")
    train_pos = train[train["label"] == 1].merge(source_ids[source_ids["split"] == "train"], on="binary_id", how="left")
    test_pos = sub[sub["label"] == 1].merge(source_ids[source_ids["split"] == "test"], on="binary_id", how="left")
    binary_ids = pd.concat([train_pos["binary_id"], test_pos["binary_id"]], ignore_index=True).drop_duplicates().tolist()
    docs = build_docs(binary_ids, PROCESSED / "fixed_entry_docs_pos.joblib", args.workers, args.force)
    train_docs = train_pos[["binary_id", "source_id_num", "label", "cwe_id"]].merge(docs, on="binary_id", how="left")
    test_docs = test_pos[["binary_id", "source_id_num", "label", "cwe_id"]].merge(docs, on="binary_id", how="left")

    stats = leave_one_out_stats(train_docs)
    stats.to_csv(Path("report") / "fixed_disasm_anchor_loo_stats.csv", index=False, encoding="utf-8-sig")
    candidates = scan_candidates(train_docs, test_docs)
    candidates.to_csv(Path("report") / f"fixed_disasm_anchor_candidates_{Path(args.input).stem}.csv", index=False, encoding="utf-8-sig")
    print(stats.to_string(index=False))
    print()
    if candidates.empty:
        print("no candidates")
    else:
        cols = ["binary_id", "source_id_num", "cwe_id", "pred", "key_type", "support", "key_loo_acc", "key_loo_wrong"]
        print(candidates[cols].to_string(index=False))


if __name__ == "__main__":
    main()
