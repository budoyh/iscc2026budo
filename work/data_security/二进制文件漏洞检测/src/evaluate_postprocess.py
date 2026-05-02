from __future__ import annotations

import re
from collections import Counter, defaultdict

import pandas as pd
from sklearn.metrics import f1_score

from common import NO_VULN, RAW, VALID_PRED_PATH
from probe_leakage import cwe_hits, file_bytes, text_hash
from train import build_split


def top_cwe(binary_id: str) -> str:
    hits = cwe_hits(file_bytes(binary_id))
    if not hits:
        return ""
    return Counter(hits).most_common(1)[0][0]


def target_from_row(row) -> str:
    return row.cwe_id if row.label == 1 else NO_VULN


def macro(true_values, pred_values) -> float:
    return f1_score(true_values, pred_values, average="macro")


def binary_macro(true_values, pred_values) -> float:
    true_bin = [0 if value == NO_VULN else 1 for value in true_values]
    pred_bin = [0 if value == NO_VULN else 1 for value in pred_values]
    return f1_score(true_bin, pred_bin, average="macro")


def main() -> None:
    train = pd.read_csv(RAW / "train.csv")
    train["target"] = train.apply(target_from_row, axis=1)
    train_idx, valid_idx = build_split(train, 0.2)
    train_fold = train.loc[train_idx].copy()
    valid_fold = train.loc[valid_idx].copy()

    valid_pred = pd.read_csv(VALID_PRED_PATH)
    pred_map = valid_pred.set_index("binary_id")["pred_target"].to_dict()
    base_pred = valid_fold["binary_id"].map(pred_map).tolist()
    y_true = valid_fold["target"].tolist()
    print("base_macro", macro(y_true, base_pred))
    print("base_binary_macro", binary_macro(y_true, base_pred))

    valid_hints = {binary_id: top_cwe(binary_id) for binary_id in valid_fold["binary_id"]}
    hint_pred = []
    changed = 0
    useful = 0
    harmful = 0
    for binary_id, true_target, pred_target in zip(valid_fold["binary_id"], y_true, base_pred, strict=False):
        hint = valid_hints[binary_id]
        new_target = pred_target
        if pred_target != NO_VULN and hint:
            new_target = hint
        if new_target != pred_target:
            changed += 1
            useful += int(new_target == true_target and pred_target != true_target)
            harmful += int(new_target != true_target and pred_target == true_target)
        hint_pred.append(new_target)
    print("cwe_hint_macro", macro(y_true, hint_pred))
    print("cwe_hint_changed", changed, "useful", useful, "harmful", harmful)

    hash_to_targets: dict[str, list[str]] = defaultdict(list)
    for row in train_fold.itertuples(index=False):
        hash_to_targets[text_hash(file_bytes(row.binary_id))].append(row.target)
    pure_hash_target = {
        sig: values[0]
        for sig, values in hash_to_targets.items()
        if len(values) >= 1 and len(set(values)) == 1
    }
    hash_pred = []
    hash_cover = 0
    hash_changed = 0
    hash_useful = 0
    hash_harmful = 0
    for binary_id, true_target, pred_target in zip(valid_fold["binary_id"], y_true, base_pred, strict=False):
        sig = text_hash(file_bytes(binary_id))
        new_target = pure_hash_target.get(sig, pred_target)
        if sig in pure_hash_target:
            hash_cover += 1
        if new_target != pred_target:
            hash_changed += 1
            hash_useful += int(new_target == true_target and pred_target != true_target)
            hash_harmful += int(new_target != true_target and pred_target == true_target)
        hash_pred.append(new_target)
    print("text_hash_macro", macro(y_true, hash_pred))
    print(
        "text_hash_cover",
        hash_cover,
        "changed",
        hash_changed,
        "useful",
        hash_useful,
        "harmful",
        hash_harmful,
    )

    combo_pred = []
    combo_changed = 0
    combo_useful = 0
    combo_harmful = 0
    for binary_id, true_target, pred_target in zip(valid_fold["binary_id"], y_true, base_pred, strict=False):
        sig = text_hash(file_bytes(binary_id))
        new_target = pure_hash_target.get(sig, pred_target)
        hint = valid_hints[binary_id]
        if new_target != NO_VULN and hint:
            new_target = hint
        if new_target != pred_target:
            combo_changed += 1
            combo_useful += int(new_target == true_target and pred_target != true_target)
            combo_harmful += int(new_target != true_target and pred_target == true_target)
        combo_pred.append(new_target)
    print("combo_macro", macro(y_true, combo_pred))
    print("combo_binary_macro", binary_macro(y_true, combo_pred))
    print("combo_changed", combo_changed, "useful", combo_useful, "harmful", combo_harmful)

    word_stats = []
    for word in ["bad", "good", "badsource", "goodsource", "badSink", "goodSink"]:
        pattern = re.compile(word, re.IGNORECASE)
        counts = Counter()
        for row in train.itertuples(index=False):
            raw = file_bytes(row.binary_id)
            if pattern.search(raw.decode("latin1", "ignore")):
                counts[row.target] += 1
        word_stats.append((word, counts.most_common(8), sum(counts.values())))
    print("word_stats", word_stats)


if __name__ == "__main__":
    main()
