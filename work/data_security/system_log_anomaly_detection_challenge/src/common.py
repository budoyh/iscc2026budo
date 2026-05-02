from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "raw" / "data"
TRAIN_PATH = RAW_DIR / "train.csv"
TEST_PATH = RAW_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH = RAW_DIR / "sample_submission.csv"
MODEL_PATH = ROOT / "models" / "posterior_tail_clause_model.json"
OOF_PATH = ROOT / "processed" / "oof_predictions_tail_clause.csv"
CV_METRICS_PATH = ROOT / "logs" / "cv_metrics_tail_clause.json"
SUBMISSION_PATH = ROOT / "submissions" / "submission_tail_clause_rule.csv"

ANOMALY_TYPES = [
    "timeout_retry",
    "resource_exhaustion",
    "slow_burn_warning",
    "state_conflict",
    "parameter_drift",
    "out_of_order",
    "missing_step",
    "duplicate_event",
    "cross_component_mismatch",
    "partial_recovery_loop",
]

DEFAULT_PARAMS = {
    "variant_min_anomaly_ratio": 0.35,
    "tail_min_anomaly_ratio": 0.28,
    "variant_min_anomaly_total": 1,
    "variant_min_top_anomaly_count": 1,
    "line_min_score": 0.25,
    "line_min_score_gap": 0.02,
    "group_max_gap": 2,
    "group_min_matches": 1,
    "group_min_score": 1.80,
    "min_span_len": 3,
}

PREFIX_TOKEN_LENGTHS = (8, 12)
TAIL_PREFIX_TOKEN_LENGTHS = (4, 6, 8, 10)
CLAUSE_SPLIT_PATTERN = re.compile(r" (?=(?:INFO|WARN|ERROR) [a-z_]+:)")
NORMAL_TAIL_PATTERNS = [
    re.compile(r"^info storage_worker: accepting segment seg src: <addr> dest: <addr>\s+(.+)$"),
    re.compile(r"^info storage_worker: accepted segment seg of size num from <addr>\s+(.+)$"),
    re.compile(r"^info storage_worker: accepted segment seg src: <addr> dest: <addr> of size num\s+(.+)$"),
    re.compile(r"^info storage_worker: id num for segment seg terminating\s+(.+)$"),
    re.compile(r"^info namespace_core: segment\* namesystem\.addstoredblock: blockmap updated: <addr> is added to seg size num\s+(.+)$"),
    re.compile(r"^info namespace_core: segment\* namesystem\.retire: seg is added to invalidset of <addr>\s+(.+)$"),
    re.compile(r"^info namespace_core: segment\* namesystem\.allocateblock: <path> seg\s+(.+)$"),
    re.compile(r"^info namespace_core: segment\* ask <addr> to replicate seg to storage_worker\(s\) <addr>\s+(.+)$"),
    re.compile(r"^info dfs_fsdataset: deleting segment seg file <path>\s+(.+)$"),
    re.compile(r"^info dfs_datablockscanner: verify succeeded for seg\s+(.+)$"),
    re.compile(r"^info storage_worker: <addr> served segment seg to <addr>\s+(.+)$"),
    re.compile(r"^info storage_worker: <addr>:transmitted segment seg to <addr>\s+(.+)$"),
    re.compile(r"^info storage_worker: <addr> starting thread to transfer segment seg to <addr>\s+(.+)$"),
    re.compile(r"^warn storage_worker: <addr>:got exception while serving seg to <addr>:\s+(.+)$"),
    re.compile(r"^warn replica_watch: pendingreplicationmonitor timed out segment seg\s+(.+)$"),
]

TYPO_MAP = {
    "acceptign": "accepting",
    "acceptnig": "accepting",
    "acecpting": "accepting",
    "accepitng": "accepting",
    "accpeting": "accepting",
    "accetped": "accepted",
    "accpeted": "accepted",
    "accepetd": "accepted",
    "termniating": "terminating",
    "terminatign": "terminating",
    "deletign": "deleting",
    "delteing": "deleting",
    "deletnig": "deleting",
    "segmnet": "segment",
    "semgent": "segment",
    "segmetn": "segment",
    "segemnt": "segment",
    "sgement": "segment",
    "tmied": "timed",
    "tiemd": "timed",
    "blocmkap": "blockmap",
    "blokcmap": "blockmap",
    "invaildset": "invalidset",
    "actino": "action",
    "detceted": "detected",
    "escalatoin": "escalation",
    "markre": "marker",
    "notde": "noted",
    "pendingreplicationmontior": "pendingreplicationmonitor",
    "recoevry": "recovery",
    "remianed": "remained",
    "stalbe": "stable",
    "tryign": "trying",
}
TYPO_PATTERN = re.compile("|".join(sorted(map(re.escape, TYPO_MAP), key=len, reverse=True)))

SEMANTIC_REPLACEMENTS = [
    (re.compile(r"\bapproved\b"), "accepted"),
    (re.compile(r"\bvalidated\b"), "verify"),
    (re.compile(r"\bvalidation\b"), "verify"),
    (re.compile(r"\bverification\b"), "verify"),
    (re.compile(r"\bre[- ]attempt\b"), "retry"),
    (re.compile(r"\bredundant\b"), "duplicate"),
]

SEMANTIC_RULES = {
    "timeout_retry": [
        (re.compile(r"\b(response latency|deadline|timeout|retry budget|fallback response|policy window)\b"), 1.4),
        (re.compile(r"\b(retry|re-attempt).*\b(deadline|response|delay|attempt)\b"), 1.2),
        (re.compile(r"\bservice delay widened\b"), 2.0),
        (re.compile(r"\brequest deadline breached\b"), 2.0),
    ],
    "resource_exhaustion": [
        (re.compile(r"\b(memory pressure|resident usage pressure|queue depth|quota_guard|backpressure)\b"), 1.7),
        (re.compile(r"\b(reserve_state=thin|reserve failure|allocation reason|dropped request)\b"), 1.6),
        (re.compile(r"\b(write path dropped|local volume rejected)\b"), 1.8),
    ],
    "slow_burn_warning": [
        (re.compile(r"\b(service time trend|p95_ms|sustained (variance|jitter|spread)|queue age)\b"), 1.6),
        (re.compile(r"\b(non-decisive warning marker|resource pressure onset|unstable phase)\b"), 1.8),
        (re.compile(r"\b(latency policy exceeded|service delay policy exceeded).*\b(sustained|persistent)\b"), 1.7),
        (re.compile(r"\brequests still accepted despite slower\b"), 1.5),
    ],
    "state_conflict": [
        (re.compile(r"\b(transition|state).*\b(disagreement|unresolved|without stable verify|approved without stable verify|accepted without stable verify)\b"), 1.8),
        (re.compile(r"\b(invalid transition|unexpected phase change|rollback attempted|reverse apply attempted)\b"), 1.7),
        (re.compile(r"\bdownstream state remained sealed\b"), 1.4),
        (re.compile(r"\brevision non-alignment\b"), 1.2),
    ],
    "parameter_drift": [
        (re.compile(r"\b(runtime profile drifting|placement drift|parameter moved|control boundary variance)\b"), 1.8),
        (re.compile(r"\b(placement factor|profile|margin).*\b(outside|drift|narrowed)\b"), 1.4),
    ],
    "out_of_order": [
        (re.compile(r"\b(ordering|sequence).*\b(moved backward|non-ideal order|relative position|non-alignment)\b"), 1.9),
        (re.compile(r"\bexpected stabilization marker.*\b(not immediately|late)\b"), 1.5),
        (re.compile(r"\bbefore digest verify\b"), 1.2),
    ],
    "missing_step": [
        (re.compile(r"\b(phase gap|transition gap).*\b(expected=.*current=commit|expected=checkpoint|expected=stabilization)\b"), 1.9),
        (re.compile(r"\bcommit observed without stable (verify|validation).*(marker|stage)\b"), 2.0),
        (re.compile(r"\bprepare stage acknowledged\b"), 1.0),
    ],
    "duplicate_event": [
        (re.compile(r"\b(duplicate|repeated|re-issued).*\b(delivery|suspicion|occurrence)\b"), 1.8),
        (re.compile(r"\bevent observed twice\b"), 2.0),
        (re.compile(r"\breplay-like dispatch\b"), 1.8),
        (re.compile(r"\bdelayed completion handling\b"), 1.2),
    ],
    "cross_component_mismatch": [
        (re.compile(r"\b(upstream metadata agreed|storage revision differed|lane_a advertised|lane_b remained)\b"), 1.8),
        (re.compile(r"\bcross-component (mismatch|non-alignment)\b"), 2.0),
        (re.compile(r"\b(reconcile skipped|older schema|older handoff contract|peer exposed older)\b"), 1.6),
        (re.compile(r"\bobserver alignment remained partially unresolved\b"), 1.2),
    ],
    "partial_recovery_loop": [
        (re.compile(r"\b(recovery loop|repeated stabilization cycle|repeat_fault)\b"), 1.9),
        (re.compile(r"\b(service restored partially|fallback regression|recovery.*exhausted)\b"), 1.7),
        (re.compile(r"\bpartial rollback appeared successful\b"), 1.2),
    ],
}


@dataclass(frozen=True)
class FoldMetric:
    fold: int
    detect_f1: float
    loc_iou: float
    type_f1: float
    final_score: float


def normalize_line(line: str) -> str:
    line = str(line).lower()
    line = TYPO_PATTERN.sub(lambda match: TYPO_MAP[match.group(0)], line)
    for pattern, replacement in SEMANTIC_REPLACEMENTS:
        line = pattern.sub(replacement, line)
    line = re.sub(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} ", "", line)
    line = re.sub(r"seg_[0-9a-f]+", "seg", line)
    line = re.sub(r"id_[0-9a-f]+", "id", line)
    line = re.sub(r"\b\d+\b", "num", line)
    line = re.sub(r"\s+", " ", line)
    return line.strip()


def split_lines(log_text: str) -> list[str]:
    return str(log_text).split("\n")


def split_embedded_clauses(line: str) -> list[str]:
    return [part.strip() for part in CLAUSE_SPLIT_PATTERN.split(str(line)) if part.strip()]


def extract_tail_annotation(normalized_line: str) -> str | None:
    for pattern in NORMAL_TAIL_PATTERNS:
        match = pattern.match(normalized_line)
        if not match:
            continue
        tail = match.group(1).strip()
        if tail and not re.match(r"^(info|warn|error) ", tail):
            return tail
        return None
    return None


def extract_line_variants(line: str) -> set[str]:
    variants: set[str] = set()
    normalized_line = normalize_line(line)
    if normalized_line:
        variants.add(normalized_line)

    for clause in split_embedded_clauses(line):
        normalized_clause = normalize_line(clause)
        if not normalized_clause:
            continue
        variants.add(normalized_clause)
        tokens = normalized_clause.split()
        for prefix_len in PREFIX_TOKEN_LENGTHS:
            if len(tokens) >= prefix_len:
                variants.add(" ".join(tokens[:prefix_len]))

    tail = extract_tail_annotation(normalized_line)
    if tail:
        variants.add("tail:" + tail)
        tail_tokens = tail.split()
        for prefix_len in TAIL_PREFIX_TOKEN_LENGTHS:
            if len(tail_tokens) >= prefix_len:
                variants.add("tail:" + " ".join(tail_tokens[:prefix_len]))
    return variants


def semantic_type_scores(line: str) -> dict[str, float]:
    normalized_line = normalize_line(line)
    tail = extract_tail_annotation(normalized_line)
    text = f"{normalized_line} {tail or ''}"
    scores: dict[str, float] = {}
    for anomaly_type, rules in SEMANTIC_RULES.items():
        score = 0.0
        for pattern, weight in rules:
            if pattern.search(text):
                score += weight
        if score > 0:
            scores[anomaly_type] = min(score, 3.0)
    return scores


def parse_spans(value: object) -> list[tuple[int, int, str]]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    spans: list[tuple[int, int, str]] = []
    for chunk in text.split(";"):
        start, end, anomaly_type = chunk.split("|")
        spans.append((int(start), int(end), anomaly_type))
    return spans


def empty_prediction(row_id: int) -> dict[str, object]:
    return {
        "id": row_id,
        "has_anomaly": 0,
        "primary_start_idx": -1,
        "primary_end_idx": -1,
        "primary_anomaly_type": "none",
        "all_spans": "",
    }


def build_template_counts(train_df: pd.DataFrame) -> dict[str, Counter]:
    template_counts: dict[str, Counter] = defaultdict(Counter)
    for row in train_df.itertuples(index=False):
        lines = split_lines(row.log_text)
        labels = ["none"] * len(lines)
        for start, end, anomaly_type in parse_spans(row.all_spans):
            for idx in range(start, end + 1):
                labels[idx] = anomaly_type
        for line, label in zip(lines, labels):
            for variant in extract_line_variants(line):
                template_counts[variant][label] += 1
    return template_counts


def compress_template_counts(template_counts: dict[str, Counter]) -> dict[str, dict[str, object]]:
    compressed: dict[str, dict[str, object]] = {}
    for key, counts in template_counts.items():
        total = int(sum(counts.values()))
        none_count = int(counts.get("none", 0))
        anomaly_counts = {name: int(value) for name, value in counts.items() if name != "none"}
        if anomaly_counts:
            top_anomaly_type, top_anomaly_count = max(anomaly_counts.items(), key=lambda item: item[1])
        else:
            top_anomaly_type, top_anomaly_count = "none", 0
        compressed[key] = {
            "total": total,
            "none_count": none_count,
            "anomaly_total": total - none_count,
            "top_anomaly_type": top_anomaly_type,
            "top_anomaly_count": int(top_anomaly_count),
            "counts": {name: int(value) for name, value in counts.items()},
        }
    return compressed


def load_model(path: Path = MODEL_PATH) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_model(
    template_counts: dict[str, Counter],
    metrics: list[FoldMetric],
    path: Path = MODEL_PATH,
    params: dict[str, float] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "params": params or DEFAULT_PARAMS,
        "template_count": len(template_counts),
        "variant_prefix_lengths": list(PREFIX_TOKEN_LENGTHS),
        "cv_metrics": [metric.__dict__ for metric in metrics],
        "templates": compress_template_counts(template_counts),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def line_match(
    line: str,
    templates: dict[str, dict[str, object]],
    params: dict[str, float],
) -> dict[str, object] | None:
    by_type_score: dict[str, float] = defaultdict(float)
    by_type_support: dict[str, int] = defaultdict(int)

    for variant in extract_line_variants(line):
        entry = templates.get(variant)
        if not entry:
            continue

        anomaly_total = int(entry["anomaly_total"])
        top_anomaly_count = int(entry["top_anomaly_count"])
        total = int(entry["total"])
        if (
            total <= 0
            or anomaly_total < params["variant_min_anomaly_total"]
            or top_anomaly_count < params["variant_min_top_anomaly_count"]
        ):
            continue

        anomaly_ratio = anomaly_total / total
        min_anomaly_ratio = params["tail_min_anomaly_ratio"] if variant.startswith("tail:") else params["variant_min_anomaly_ratio"]
        if anomaly_ratio < min_anomaly_ratio:
            continue

        anomaly_type = str(entry["top_anomaly_type"])
        type_ratio = top_anomaly_count / anomaly_total if anomaly_total else 0.0
        score = math.log1p(top_anomaly_count) * anomaly_ratio * type_ratio
        if variant.startswith("tail:"):
            score *= 1.25
        by_type_score[anomaly_type] = max(by_type_score[anomaly_type], score)
        by_type_support[anomaly_type] = max(by_type_support[anomaly_type], top_anomaly_count)

    if not by_type_score:
        return None

    ranked = sorted(by_type_score.items(), key=lambda item: item[1], reverse=True)
    best_type, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    if best_score < params["line_min_score"] or best_score - second_score < params["line_min_score_gap"]:
        return None

    return {
        "type": best_type,
        "support": by_type_support[best_type],
        "line_score": best_score,
    }


def extract_groups(
    log_text: str,
    templates: dict[str, dict[str, object]],
    params: dict[str, float] | None = None,
) -> list[dict[str, object]]:
    params = params or DEFAULT_PARAMS
    matches: list[dict[str, object]] = []
    for idx, line in enumerate(split_lines(log_text)):
        matched = line_match(line, templates, params)
        if matched is None:
            continue
        matched["idx"] = idx
        matches.append(matched)

    if not matches:
        return []

    grouped: list[list[dict[str, object]]] = []
    current_group = [matches[0]]
    for matched in matches[1:]:
        if (
            matched["type"] == current_group[-1]["type"]
            and int(matched["idx"]) - int(current_group[-1]["idx"]) <= params["group_max_gap"]
        ):
            current_group.append(matched)
        else:
            grouped.append(current_group)
            current_group = [matched]
    grouped.append(current_group)

    groups: list[dict[str, object]] = []
    for rank, group in enumerate(grouped):
        groups.append(
            {
                "rank": rank,
                "type": str(group[0]["type"]),
                "start": int(group[0]["idx"]),
                "end": int(group[-1]["idx"]),
                "match_count": len(group),
                "score": float(sum(float(item["line_score"]) for item in group)),
                "max_support": int(max(int(item["support"]) for item in group)),
            }
        )
    return groups


def select_primary_group(
    groups: Iterable[dict[str, object]],
    params: dict[str, float] | None = None,
) -> dict[str, object] | None:
    params = params or DEFAULT_PARAMS
    candidates = [
        group
        for group in groups
        if int(group["match_count"]) >= params["group_min_matches"]
        and float(group["score"]) >= params["group_min_score"]
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda item: (int(item["start"]), -float(item["score"]), -int(item["max_support"])))


def predict_row(
    row_id: int,
    log_text: str,
    templates: dict[str, dict[str, object]],
    params: dict[str, float] | None = None,
) -> dict[str, object]:
    params = params or DEFAULT_PARAMS
    lines = split_lines(log_text)
    primary_group = select_primary_group(extract_groups(log_text, templates, params), params)
    if primary_group is None:
        return empty_prediction(row_id)

    start = int(primary_group["start"])
    end = int(primary_group["end"])
    if end - start + 1 < params["min_span_len"]:
        end = min(len(lines) - 1, start + params["min_span_len"] - 1)
    anomaly_type = str(primary_group["type"])
    return {
        "id": row_id,
        "has_anomaly": 1,
        "primary_start_idx": start,
        "primary_end_idx": end,
        "primary_anomaly_type": anomaly_type,
        "all_spans": f"{start}|{end}|{anomaly_type}",
    }


def predict_frame(
    df: pd.DataFrame,
    templates: dict[str, dict[str, object]],
    params: dict[str, float] | None = None,
) -> pd.DataFrame:
    records = [predict_row(int(row.id), row.log_text, templates, params) for row in df.itertuples(index=False)]
    return pd.DataFrame(records)


def compute_score(true_df: pd.DataFrame, pred_df: pd.DataFrame) -> dict[str, float]:
    detect_f1 = f1_score(true_df["has_anomaly"], pred_df["has_anomaly"], average="macro")
    mask = (true_df["has_anomaly"] == 1) & (pred_df["has_anomaly"] == 1)

    if mask.any():
        ious: list[float] = []
        for true_row, pred_row in zip(true_df[mask].itertuples(index=False), pred_df[mask].itertuples(index=False)):
            true_start = int(true_row.primary_start_idx)
            true_end = int(true_row.primary_end_idx)
            pred_start = int(pred_row.primary_start_idx)
            pred_end = int(pred_row.primary_end_idx)
            intersection = max(0, min(true_end, pred_end) - max(true_start, pred_start) + 1)
            union = max(true_end, pred_end) - min(true_start, pred_start) + 1
            ious.append(intersection / union if union else 0.0)
        loc_iou = float(np.mean(ious))
        type_f1 = f1_score(
            true_df.loc[mask, "primary_anomaly_type"],
            pred_df.loc[mask, "primary_anomaly_type"],
            average="macro",
        )
    else:
        loc_iou = 0.0
        type_f1 = 0.0

    final_score = 0.15 * detect_f1 + 0.50 * loc_iou + 0.35 * type_f1
    return {
        "detect_f1": float(detect_f1),
        "loc_iou": float(loc_iou),
        "type_f1": float(type_f1),
        "final_score": float(final_score),
    }


def validate_submission(df: pd.DataFrame, expected_rows: int | None = None) -> list[str]:
    errors: list[str] = []
    expected_cols = [
        "id",
        "has_anomaly",
        "primary_start_idx",
        "primary_end_idx",
        "primary_anomaly_type",
        "all_spans",
    ]
    if list(df.columns) != expected_cols:
        errors.append(f"unexpected columns: {list(df.columns)}")
    if expected_rows is not None and len(df) != expected_rows:
        errors.append(f"unexpected row count: got={len(df)} expected={expected_rows}")
    if df["id"].duplicated().any():
        errors.append("duplicate id values found")
    if not df["id"].is_monotonic_increasing:
        errors.append("id column is not sorted ascending")
    if df.isna().any().any():
        errors.append("submission contains NaN")

    invalid_has_anomaly = ~df["has_anomaly"].isin([0, 1])
    if invalid_has_anomaly.any():
        errors.append("has_anomaly contains invalid values")

    normal_mask = df["has_anomaly"] == 0
    if not (df.loc[normal_mask, "primary_start_idx"] == -1).all():
        errors.append("normal rows must set primary_start_idx to -1")
    if not (df.loc[normal_mask, "primary_end_idx"] == -1).all():
        errors.append("normal rows must set primary_end_idx to -1")
    if not (df.loc[normal_mask, "primary_anomaly_type"] == "none").all():
        errors.append("normal rows must set primary_anomaly_type to none")

    anomaly_mask = df["has_anomaly"] == 1
    if not (df.loc[anomaly_mask, "primary_start_idx"] >= 0).all():
        errors.append("anomalous rows must have non-negative primary_start_idx")
    if not (df.loc[anomaly_mask, "primary_end_idx"] >= df.loc[anomaly_mask, "primary_start_idx"]).all():
        errors.append("anomalous rows must have primary_end_idx >= primary_start_idx")
    if not df.loc[anomaly_mask, "primary_anomaly_type"].isin(ANOMALY_TYPES).all():
        errors.append("anomalous rows contain invalid primary_anomaly_type")

    return errors


def run_cross_validation(train_df: pd.DataFrame, params: dict[str, float] | None = None) -> tuple[list[FoldMetric], pd.DataFrame]:
    params = params or DEFAULT_PARAMS
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y = train_df["primary_anomaly_type"].values

    metrics: list[FoldMetric] = []
    oof_parts: list[pd.DataFrame] = []

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(train_df, y), start=1):
        fold_train = train_df.iloc[train_idx].reset_index(drop=True)
        fold_valid = train_df.iloc[valid_idx].reset_index(drop=True)
        fold_templates = compress_template_counts(build_template_counts(fold_train))
        fold_pred = predict_frame(fold_valid[["id", "log_text"]], fold_templates, params)
        fold_score = compute_score(fold_valid, fold_pred)
        metrics.append(
            FoldMetric(
                fold=fold,
                detect_f1=fold_score["detect_f1"],
                loc_iou=fold_score["loc_iou"],
                type_f1=fold_score["type_f1"],
                final_score=fold_score["final_score"],
            )
        )

        merged = fold_valid[
            [
                "id",
                "has_anomaly",
                "primary_start_idx",
                "primary_end_idx",
                "primary_anomaly_type",
                "all_spans",
            ]
        ].copy()
        merged = merged.rename(
            columns={
                "has_anomaly": "true_has_anomaly",
                "primary_start_idx": "true_primary_start_idx",
                "primary_end_idx": "true_primary_end_idx",
                "primary_anomaly_type": "true_primary_anomaly_type",
                "all_spans": "true_all_spans",
            }
        )
        merged["pred_has_anomaly"] = fold_pred["has_anomaly"]
        merged["pred_primary_start_idx"] = fold_pred["primary_start_idx"]
        merged["pred_primary_end_idx"] = fold_pred["primary_end_idx"]
        merged["pred_primary_anomaly_type"] = fold_pred["primary_anomaly_type"]
        merged["pred_all_spans"] = fold_pred["all_spans"]
        merged["fold"] = fold
        oof_parts.append(merged)

    oof_df = pd.concat(oof_parts, ignore_index=True).sort_values("id").reset_index(drop=True)
    return metrics, oof_df


def summarize_metrics(metrics: list[FoldMetric]) -> dict[str, object]:
    return {
        "folds": [metric.__dict__ for metric in metrics],
        "mean": {
            "detect_f1": float(np.mean([metric.detect_f1 for metric in metrics])),
            "loc_iou": float(np.mean([metric.loc_iou for metric in metrics])),
            "type_f1": float(np.mean([metric.type_f1 for metric in metrics])),
            "final_score": float(np.mean([metric.final_score for metric in metrics])),
        },
    }
