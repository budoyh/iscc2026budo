from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from common import ROOT, TEST_PATH, TRAIN_PATH, normalize_line, semantic_type_scores, validate_submission


SUBMISSION_COLUMNS = [
    "has_anomaly",
    "primary_start_idx",
    "primary_end_idx",
    "primary_anomaly_type",
    "all_spans",
]

KNOWN_BAD_IDS = {591, 1229, 2276, 2611}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build small gated boundary candidates on top of push.csv.")
    parser.add_argument("--base", type=str, default=str(ROOT / "submissions" / "push.csv"))
    parser.add_argument("--train-path", type=str, default=str(TRAIN_PATH))
    parser.add_argument("--test-path", type=str, default=str(TEST_PATH))
    parser.add_argument("--output-prefix", type=str, default=str(ROOT / "submissions" / "meta"))
    parser.add_argument("--summary-path", type=str, default=str(ROOT / "logs" / "meta_boundary_summary.json"))
    parser.add_argument("--rows-path", type=str, default=str(ROOT / "processed" / "meta_boundary_candidates.csv"))
    parser.add_argument("--max-rows", action="append", type=int, default=[4, 6, 8, 10, 12])
    parser.add_argument("--min-score", type=float, default=4.0)
    parser.add_argument("--min-delta", type=float, default=-0.005)
    parser.add_argument("--min-evidence", type=float, default=0.0)
    parser.add_argument("--include-known-bad", action="store_true")
    return parser.parse_args()


def read_submission(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False).set_index("id", drop=False)


def read_confidence(name: str) -> pd.DataFrame | None:
    path = ROOT / "processed" / f"ranker_confidence_{name}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, keep_default_na=False).set_index("id", drop=False)


def load_sources() -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    submissions: dict[str, pd.DataFrame] = {}
    for name in ["next", "b4h3", "b5h3", "txt", "p2", "trim", "best", "cons"]:
        path = ROOT / "submissions" / f"{name}.csv"
        if path.exists():
            submissions[name] = read_submission(path)
    confidences = {name: frame for name in ["next", "b4h3", "b5h3", "txt"] if (frame := read_confidence(name)) is not None}
    return submissions, confidences


def build_line_stats(train_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for row in train_df.itertuples(index=False):
        lines = str(row.log_text).split("\n")
        has_anomaly = int(row.has_anomaly) == 1
        start = int(row.primary_start_idx)
        end = int(row.primary_end_idx)
        anomaly_type = str(row.primary_anomaly_type)
        for idx, line in enumerate(lines):
            key = normalize_line(line)
            item = stats.setdefault(
                key,
                {
                    "total": 0,
                    "inside": 0,
                    "outside": 0,
                    "start": 0,
                    "end": 0,
                    "types": defaultdict(lambda: {"inside": 0, "start": 0, "end": 0}),
                },
            )
            item["total"] += 1
            inside = has_anomaly and start <= idx <= end
            if inside:
                item["inside"] += 1
                type_item = item["types"][anomaly_type]
                type_item["inside"] += 1
                if idx == start:
                    item["start"] += 1
                    type_item["start"] += 1
                if idx == end:
                    item["end"] += 1
                    type_item["end"] += 1
            else:
                item["outside"] += 1
    # Convert defaultdict to normal dict before JSON serialization.
    for item in stats.values():
        item["types"] = dict(item["types"])
    return stats


def rows_equal(left: pd.Series, right: pd.Series) -> bool:
    return bool(left[SUBMISSION_COLUMNS].equals(right[SUBMISSION_COLUMNS]))


def row_confidence(
    row_id: int,
    row: pd.Series,
    submissions: dict[str, pd.DataFrame],
    confidences: dict[str, pd.DataFrame],
    preferred: str | None = None,
) -> tuple[float, str]:
    if preferred and preferred in confidences:
        return float(confidences[preferred].loc[row_id, "ranker_confidence"]), preferred
    for name in ["txt", "b5h3", "b4h3", "next"]:
        if name in submissions and name in confidences and rows_equal(row, submissions[name].loc[row_id]):
            return float(confidences[name].loc[row_id, "ranker_confidence"]), name
    values: list[tuple[float, str]] = []
    for name, confidence in confidences.items():
        if row_id in confidence.index:
            values.append((float(confidence.loc[row_id, "ranker_confidence"]), name))
    if not values:
        return 0.0, "none"
    return max(values)


def boundary_action_score(
    line: str,
    anomaly_type: str,
    action: str,
    side: str,
    stats: dict[str, dict[str, Any]],
) -> tuple[float, dict[str, Any]]:
    key = normalize_line(line)
    item = stats.get(key, {"total": 0, "inside": 0, "outside": 0, "start": 0, "end": 0, "types": {}})
    type_item = item.get("types", {}).get(anomaly_type, {"inside": 0, "start": 0, "end": 0})
    total = int(item.get("total", 0))
    inside = int(item.get("inside", 0))
    outside = int(item.get("outside", 0))
    boundary_count = int(type_item.get(side, 0))
    type_inside = int(type_item.get("inside", 0))
    semantic = semantic_type_scores(line)
    semantic_target = float(semantic.get(anomaly_type, 0.0))
    semantic_other = max([score for label, score in semantic.items() if label != anomaly_type] or [0.0])

    score = 0.0
    if action == "add":
        if semantic_target > 0:
            score += 2.5 + semantic_target
        elif semantic_other > 0:
            score -= 1.2 * semantic_other
        if boundary_count > 0:
            score += min(3.0, math.log1p(boundary_count))
        if type_inside >= 5 and total < 1000:
            score += min(2.0, math.log1p(type_inside) / 2)
        if total >= 1000 and inside / max(total, 1) < 0.01 and semantic_target == 0:
            score -= 2.0
        if total == 0 and semantic_target == 0:
            score -= 1.0
    else:
        if semantic_target > 0:
            score -= 3.0 + semantic_target
        elif semantic_other > 0:
            score -= 0.8 * semantic_other
        if boundary_count >= 3:
            score -= min(3.0, math.log1p(boundary_count))
        if total >= 1000 and inside / max(total, 1) < 0.01 and semantic_target == 0:
            score += 2.0
        if total > 0 and outside / max(total, 1) > 0.95 and semantic_target == 0 and semantic_other == 0:
            score += 1.0

    detail = {
        "normalized": key,
        "action": action,
        "side": side,
        "total": total,
        "inside": inside,
        "outside": outside,
        "type_inside": type_inside,
        "type_boundary": boundary_count,
        "semantic_target": semantic_target,
        "semantic_other": semantic_other,
        "boundary_score": score,
    }
    return score, detail


def candidate_score(
    row_id: int,
    base_row: pd.Series,
    alt_row: pd.Series,
    alt_name: str,
    test_text: str,
    submissions: dict[str, pd.DataFrame],
    confidences: dict[str, pd.DataFrame],
    stats: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    if rows_equal(base_row, alt_row):
        return None
    if not (int(base_row.has_anomaly) == 1 and int(alt_row.has_anomaly) == 1):
        return None
    if str(base_row.primary_anomaly_type) != str(alt_row.primary_anomaly_type):
        return None

    base_conf, base_source = row_confidence(row_id, base_row, submissions, confidences)
    alt_conf, alt_source = row_confidence(row_id, alt_row, submissions, confidences, preferred=alt_name if alt_name in confidences else None)
    conf_delta = alt_conf - base_conf
    agreement = sum(
        1
        for name in ["txt", "b5h3", "b4h3", "next"]
        if name in submissions and rows_equal(alt_row, submissions[name].loc[row_id])
    )

    base_start = int(base_row.primary_start_idx)
    base_end = int(base_row.primary_end_idx)
    alt_start = int(alt_row.primary_start_idx)
    alt_end = int(alt_row.primary_end_idx)
    anomaly_type = str(base_row.primary_anomaly_type)
    lines = str(test_text).split("\n")
    details: list[dict[str, Any]] = []
    evidence = 0.0
    for idx in sorted(set(range(min(base_start, alt_start), max(base_end, alt_end) + 1))):
        in_base = base_start <= idx <= base_end
        in_alt = alt_start <= idx <= alt_end
        if in_base == in_alt:
            continue
        action = "add" if in_alt else "remove"
        side = "start" if idx <= min(base_start, alt_start) else "end"
        line_score, detail = boundary_action_score(lines[idx], anomaly_type, action, side, stats)
        detail.update({"line_idx": idx, "line_text": lines[idx]})
        evidence += line_score
        details.append(detail)

    # Confidence is useful but noisy; evidence and independent agreement dominate.
    total_score = evidence + 70.0 * conf_delta + 0.8 * agreement
    if row_id in KNOWN_BAD_IDS:
        total_score -= 8.0
    return {
        "id": int(row_id),
        "alt": alt_name,
        "type": anomaly_type,
        "base_start": base_start,
        "base_end": base_end,
        "alt_start": alt_start,
        "alt_end": alt_end,
        "base_conf": base_conf,
        "base_source": base_source,
        "alt_conf": alt_conf,
        "alt_source": alt_source,
        "conf_delta": conf_delta,
        "agreement": agreement,
        "evidence": evidence,
        "score": total_score,
        "details": details,
    }


def materialize(base: pd.DataFrame, selected: list[dict[str, Any]], submissions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    output = base.copy()
    for item in selected:
        row = submissions[str(item["alt"])].loc[int(item["id"])]
        output.loc[int(item["id"]), SUBMISSION_COLUMNS] = row[SUBMISSION_COLUMNS]
    output = output.sort_index().reset_index(drop=True)
    return output


def main() -> None:
    args = parse_args()
    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path).set_index("id")
    base = read_submission(Path(args.base))
    submissions, confidences = load_sources()
    submissions["push"] = base
    stats = build_line_stats(train_df)

    by_id: dict[int, dict[str, Any]] = {}
    for alt_name, alt_submission in submissions.items():
        if alt_name == "push":
            continue
        for row_id in base.index:
            if not args.include_known_bad and int(row_id) in KNOWN_BAD_IDS:
                continue
            item = candidate_score(
                int(row_id),
                base.loc[row_id],
                alt_submission.loc[row_id],
                alt_name,
                str(test_df.loc[row_id, "log_text"]),
                submissions,
                confidences,
                stats,
            )
            if item is None:
                continue
            if item["conf_delta"] < args.min_delta or item["score"] < args.min_score or item["evidence"] < args.min_evidence:
                continue
            current = by_id.get(int(row_id))
            if current is None or item["score"] > current["score"]:
                by_id[int(row_id)] = item

    ranked = sorted(by_id.values(), key=lambda item: (item["score"], item["conf_delta"]), reverse=True)
    rows_path = Path(args.rows_path)
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    rows_df = pd.DataFrame([{key: value for key, value in item.items() if key != "details"} for item in ranked])
    rows_df.to_csv(rows_path, index=False, encoding="utf-8")

    outputs: dict[str, Any] = {}
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for max_rows in args.max_rows:
        selected = ranked[:max_rows]
        output = materialize(base, selected, submissions)
        errors = validate_submission(output, expected_rows=len(test_df))
        if errors:
            raise SystemExit(f"candidate {max_rows} validation failed: " + " | ".join(errors))
        out_path = prefix.with_name(f"{prefix.name}{max_rows}.csv")
        output.to_csv(out_path, index=False, encoding="utf-8")
        outputs[str(max_rows)] = {
            "path": str(out_path),
            "ids": [int(item["id"]) for item in selected],
            "anomaly_count": int(output["has_anomaly"].sum()),
        }

    summary = {
        "base": str(Path(args.base)),
        "known_bad_ids_excluded": not args.include_known_bad,
        "min_score": args.min_score,
        "min_delta": args.min_delta,
        "min_evidence": args.min_evidence,
        "candidate_count": len(ranked),
        "top_candidates": ranked[:20],
        "outputs": outputs,
    }
    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Rows saved to: {rows_path}")
    print(f"Summary saved to: {summary_path}")
    for key, value in outputs.items():
        print(f"meta{key}: {value['path']} ids={value['ids']} count={value['anomaly_count']}")


if __name__ == "__main__":
    main()
