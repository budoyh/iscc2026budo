from __future__ import annotations

from collections import Counter, defaultdict

import pandas as pd

from common import PROCESSED, RAW, REPORT, SUBMISSIONS
from dwarf_line_3class import TARGET_CWES, extract_dwarf_line_features


KEY_COLUMNS = [
    "entry_line",
    "high_pc",
    "var_lines",
    "ptr_sig",
    "struct_size_sig",
    "member_sig",
    "char_first_type",
    "line_seq_compact",
]


def make_key(row: pd.Series, columns: list[str]) -> str:
    return "||".join(str(row.get(col, "")) for col in columns)


def build_features(binary_ids: list[str], cache_name: str) -> pd.DataFrame:
    cache_path = PROCESSED / cache_name
    if cache_path.exists():
        cached = pd.read_csv(cache_path, keep_default_na=False)
        have = set(cached["binary_id"].astype(str))
        missing = [binary_id for binary_id in binary_ids if binary_id not in have]
        if not missing:
            return cached[cached["binary_id"].isin(binary_ids)].copy()
        rows = [extract_dwarf_line_features(binary_id) for binary_id in missing]
        cached = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True)
        cached.to_csv(cache_path, index=False, encoding="utf-8")
        return cached[cached["binary_id"].isin(binary_ids)].copy()
    rows = [extract_dwarf_line_features(binary_id) for binary_id in binary_ids]
    frame = pd.DataFrame(rows)
    frame.to_csv(cache_path, index=False, encoding="utf-8")
    return frame


def loo_stats(train_features: pd.DataFrame, columns: list[str]) -> dict[str, object]:
    labels = train_features["cwe_id"].astype(str).tolist()
    keys = [make_key(row, columns) for _, row in train_features.iterrows()]
    counters: dict[str, Counter[str]] = defaultdict(Counter)
    for key, label in zip(keys, labels):
        counters[key][label] += 1

    covered = 0
    correct = 0
    wrong = 0
    abstain = 0
    for key, label in zip(keys, labels):
        counter = counters[key].copy()
        counter[label] -= 1
        if counter[label] <= 0:
            del counter[label]
        if not counter:
            abstain += 1
            continue
        pred, _count = counter.most_common(1)[0]
        covered += 1
        if pred == label:
            correct += 1
        else:
            wrong += 1
    return {
        "columns": "+".join(columns),
        "keys": len(counters),
        "covered": covered,
        "abstain": abstain,
        "correct": correct,
        "wrong": wrong,
        "accuracy": correct / covered if covered else 0.0,
    }


def main() -> None:
    train = pd.read_csv(RAW / "train.csv", keep_default_na=False)
    s17b = pd.read_csv(SUBMISSIONS / "s17b.csv", keep_default_na=False)
    train_tri = train[(train["label"].astype(int) == 1) & train["cwe_id"].isin(TARGET_CWES)].copy()
    test_tri = s17b[(s17b["label"].astype(int) == 1) & s17b["cwe_id"].isin(TARGET_CWES)].copy()

    train_features = build_features(train_tri["binary_id"].astype(str).tolist(), "dwarf_key_train_tri.csv")
    train_features = train_features.merge(train_tri[["binary_id", "cwe_id"]], on="binary_id", how="left", validate="one_to_one")
    test_features = build_features(test_tri["binary_id"].astype(str).tolist(), "dwarf_key_test_s17b_tri.csv")
    test_features = test_features.merge(test_tri[["binary_id", "cwe_id"]], on="binary_id", how="left", validate="one_to_one")

    key_sets = [
        KEY_COLUMNS,
        [col for col in KEY_COLUMNS if col != "high_pc"],
        ["entry_line", "var_lines", "ptr_sig", "struct_size_sig", "member_sig", "char_first_type", "line_seq_compact"],
        ["entry_line", "var_lines", "ptr_sig", "struct_size_sig", "line_seq_compact"],
        ["ptr_sig", "struct_size_sig", "member_sig", "char_first_type", "line_seq_compact"],
    ]
    stats = pd.DataFrame([loo_stats(train_features, cols) for cols in key_sets])
    stats.to_csv(REPORT / "dwarf_key_loo_stats.csv", index=False, encoding="utf-8")
    print(stats.to_string(index=False))

    all_candidates = []
    for columns in key_sets:
        counters: dict[str, Counter[str]] = defaultdict(Counter)
        examples: dict[str, list[str]] = defaultdict(list)
        for _, row in train_features.iterrows():
            key = make_key(row, columns)
            label = str(row["cwe_id"])
            counters[key][label] += 1
            if len(examples[key]) < 5:
                examples[key].append(str(row["binary_id"]))
        for _, row in test_features.iterrows():
            key = make_key(row, columns)
            counter = counters.get(key, Counter())
            if not counter:
                continue
            total = sum(counter.values())
            pred, count = counter.most_common(1)[0]
            purity = count / total
            current = str(row["cwe_id"])
            if pred != current and purity >= 1.0:
                all_candidates.append(
                    {
                        "binary_id": row["binary_id"],
                        "current": current,
                        "key_pred": pred,
                        "support": total,
                        "purity": purity,
                        "columns": "+".join(columns),
                        "examples": "|".join(examples[key]),
                        "entry_line": row.get("entry_line", ""),
                        "high_pc": row.get("high_pc", ""),
                        "var_lines": row.get("var_lines", ""),
                        "line_seq_compact": row.get("line_seq_compact", ""),
                    }
                )

    candidates = pd.DataFrame(all_candidates)
    if len(candidates):
        candidates = candidates.sort_values(["support", "binary_id"], ascending=[False, True])
    candidates.to_csv(REPORT / "dwarf_key_candidates.csv", index=False, encoding="utf-8")
    print("candidates", len(candidates))
    if len(candidates):
        print(candidates.head(80).to_string(index=False))


if __name__ == "__main__":
    main()
