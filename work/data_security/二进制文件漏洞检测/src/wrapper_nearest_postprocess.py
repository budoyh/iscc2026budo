from __future__ import annotations

import argparse
import re
from collections import Counter
from functools import lru_cache

import pandas as pd

from common import BINARY_DIR, NO_VULN, PROCESSED, RAW, REPORT, SUBMISSIONS, ensure_dirs, load_test
from predict import validate_submission
from source_id_chunk_postprocess import source_id_number


KIND_PATTERNS = {
    "wrapper_bad": 1,
    "wrapper_good": 0,
    "injected_bad": 1,
    "injected_good": 0,
}

KIND_RE = re.compile(rb"(wrapper|injected)_(bad|good)\.(?:c|cpp)")


@lru_cache(maxsize=None)
def binary_kind_label(binary_id: str) -> tuple[str, int | None]:
    raw = (BINARY_DIR / f"{binary_id}.exe").read_bytes()
    hits: Counter[str] = Counter()
    for match in KIND_RE.finditer(raw):
        key = f"{match.group(1).decode()}_{match.group(2).decode()}"
        hits[key] += 1
    if not hits:
        return "", None
    key, _count = hits.most_common(1)[0]
    return key.rsplit("_", 1)[0], KIND_PATTERNS[key]


def source_id_frame(split: str) -> pd.DataFrame | None:
    path = PROCESSED / "source_ids.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    frame = frame[frame["split"] == split][["binary_id", "source_id_num"]].copy()
    return frame.rename(columns={"source_id_num": "sid"})


def build_train_sources(with_kind: bool = False) -> pd.DataFrame:
    train = pd.read_csv(RAW / "train.csv")
    source_ids = source_id_frame("train")
    if source_ids is None:
        source_ids = pd.DataFrame(
            [(row.binary_id, source_id_number(row.binary_id)) for row in train.itertuples(index=False)],
            columns=["binary_id", "sid"],
        )
    out = train.merge(source_ids, on="binary_id", how="left", validate="one_to_one")
    if with_kind:
        kinds = [binary_kind_label(binary_id) for binary_id in out["binary_id"]]
        out["kind"] = [kind for kind, _label in kinds]
        out["embedded_label"] = [label for _kind, label in kinds]
    else:
        out["kind"] = ""
        out["embedded_label"] = pd.NA
    return out


def test_source_ids(test: pd.DataFrame) -> pd.DataFrame:
    source_ids = source_id_frame("test")
    if source_ids is not None:
        return source_ids
    return pd.DataFrame(
        [(row.binary_id, source_id_number(row.binary_id)) for row in test.itertuples(index=False)],
        columns=["binary_id", "sid"],
    )


def nearest_cwe(sid: int, train_pos: pd.DataFrame, kind: str | None = None) -> tuple[str, int, str, int]:
    pool = train_pos
    if kind:
        same = train_pos[train_pos["kind"] == kind]
        if len(same):
            pool = same
    dist = (pool["sid"] - sid).abs()
    idx = dist.idxmin()
    hit = pool.loc[idx]
    return str(hit["cwe_id"]), int(hit["sid"]), str(hit["binary_id"]), int(dist.loc[idx])


def one_error_drop_estimate(class_counts: pd.Series, old_cwe: str, new_cwe: str) -> float:
    class_count = len(class_counts)
    old_count = int(class_counts.get(old_cwe, 0))
    new_count = int(class_counts.get(new_cwe, 0))
    if class_count == 0 or old_count <= 0 or new_count <= 0:
        return 1.0
    old_f1 = 2 * (old_count - 1) / (2 * old_count - 1)
    new_f1 = 2 * new_count / (2 * new_count + 1)
    return float((2 - old_f1 - new_f1) / class_count)


def check_labels(input_name: str) -> None:
    train = build_train_sources(with_kind=True)
    mismatches = train[train["embedded_label"].notna() & (train["embedded_label"].astype(int) != train["label"])]
    test = load_test()
    sub = pd.read_csv(SUBMISSIONS / input_name, keep_default_na=False)
    if list(sub.columns) == ["name", "label", "cwe_id"]:
        sub = sub.rename(columns={"name": "binary_id"})
    rows = []
    for row in sub.itertuples(index=False):
        kind, embedded_label = binary_kind_label(row.binary_id)
        rows.append((row.binary_id, kind, embedded_label, int(row.label)))
    test_check = pd.DataFrame(rows, columns=["binary_id", "kind", "embedded_label", "submission_label"])
    test_check["diff"] = test_check["embedded_label"].astype(int) != test_check["submission_label"].astype(int)
    out_path = REPORT / "wrapper_nearest_label_check.csv"
    test_check.to_csv(out_path, index=False, encoding="utf-8")
    print(f"train={len(train)} embedded_covered={train['embedded_label'].notna().sum()} mismatches={len(mismatches)}")
    print(f"test={len(test)} label_diff_vs_{input_name}={int(test_check['diff'].sum())}")
    print(out_path)
    if len(mismatches):
        print(mismatches.head(20).to_string(index=False))


def generate(
    input_name: str,
    output: str,
    same_kind: bool,
    rewrite_label: bool,
    max_one_error_drop: float | None,
    allowed_cwes: set[str] | None,
) -> None:
    ensure_dirs()
    test = load_test()
    train = build_train_sources(with_kind=same_kind)
    train_pos = train[train["label"] == 1].copy()
    sub = pd.read_csv(SUBMISSIONS / input_name, keep_default_na=False)
    if list(sub.columns) == ["name", "label", "cwe_id"]:
        sub = sub.rename(columns={"name": "binary_id"})
    sub = sub.merge(test_source_ids(test), on="binary_id", how="left", validate="one_to_one")
    targets = sub["cwe_id"].where(sub["label"].astype(int).eq(1), NO_VULN)
    class_counts = targets.value_counts()

    changed = []
    skipped = []
    out = sub.copy()
    for idx, row in out.iterrows():
        kind, embedded_label = binary_kind_label(row["binary_id"]) if (rewrite_label or same_kind) else ("", None)
        label = int(embedded_label) if rewrite_label and embedded_label is not None else int(row["label"])
        old_label = int(row["label"])
        old_cwe = str(row["cwe_id"])
        out.at[idx, "label"] = label
        if label == 0:
            out.at[idx, "cwe_id"] = ""
            if old_label != 0 or old_cwe:
                changed.append((row["binary_id"], int(row["sid"]), old_label, old_cwe, label, "", kind, pd.NA, "", pd.NA, pd.NA))
            continue
        sid = int(row["sid"])
        mapped, nearest_sid, nearest_bid, dist = nearest_cwe(sid, train_pos, kind if same_kind else None)
        drop_est = one_error_drop_estimate(class_counts, old_cwe, mapped)
        if old_cwe != mapped:
            if max_one_error_drop is not None and drop_est > max_one_error_drop:
                skipped.append((row["binary_id"], sid, old_cwe, mapped, drop_est, "drop"))
                continue
            if allowed_cwes is not None and (old_cwe not in allowed_cwes or mapped not in allowed_cwes):
                skipped.append((row["binary_id"], sid, old_cwe, mapped, drop_est, "cwe"))
                continue
        out.at[idx, "cwe_id"] = mapped
        if old_label != label or old_cwe != mapped:
            changed.append(
                (
                    row["binary_id"],
                    sid,
                    old_label,
                    old_cwe,
                    label,
                    mapped,
                    kind,
                    nearest_sid,
                    nearest_bid,
                    dist,
                    drop_est,
                )
            )

    out = out[["binary_id", "label", "cwe_id"]]
    validate_submission(out, test["binary_id"])
    output_path = SUBMISSIONS / output
    out.to_csv(output_path, index=False, encoding="utf-8-sig")
    report_path = REPORT / f"{output}.diff.csv"
    pd.DataFrame(
        changed,
        columns=[
            "binary_id",
            "sid",
            "old_label",
            "old_cwe",
            "new_label",
            "new_cwe",
            "kind",
            "nearest_sid",
            "nearest_bid",
            "dist",
            "one_error_drop_est",
        ],
    ).to_csv(report_path, index=False, encoding="utf-8")
    print(output_path)
    print(f"same_kind={same_kind} rewrite_label={rewrite_label} changed={len(changed)}")
    print(f"skipped={len(skipped)}")
    print(report_path)
    if changed:
        print(pd.DataFrame(changed).head(80).to_string(index=False, header=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce wrapper/injected label leakage plus source-id nearest CWE mapping.")
    parser.add_argument("--input", default="s15.csv")
    parser.add_argument("--output", default="s17n.csv")
    parser.add_argument("--same-kind", action="store_true")
    parser.add_argument("--no-rewrite-label", action="store_true")
    parser.add_argument("--max-one-error-drop", type=float, default=None)
    parser.add_argument("--allowed-cwes", default="")
    parser.add_argument("--check-labels", action="store_true")
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()
    ensure_dirs()
    if args.check_labels:
        check_labels(args.input)
    if args.generate:
        allowed_cwes = {item.strip() for item in args.allowed_cwes.split(",") if item.strip()} or None
        generate(
            args.input,
            args.output,
            same_kind=args.same_kind,
            rewrite_label=not args.no_rewrite_label,
            max_one_error_drop=args.max_one_error_drop,
            allowed_cwes=allowed_cwes,
        )


if __name__ == "__main__":
    main()
