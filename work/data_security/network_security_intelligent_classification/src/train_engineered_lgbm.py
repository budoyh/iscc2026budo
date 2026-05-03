from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw" / "data"
MODELS = ROOT / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LightGBM with domain-robust engineered traffic features.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=900)
    parser.add_argument("--run-id", type=str, default=None)
    return parser.parse_args()


def add_triplet_features(frame: pd.DataFrame, prefix: str, cols: list[str]) -> None:
    values = frame[cols]
    frame[f"{prefix}_em_diff"] = frame[cols[0]] - frame[cols[1]]
    frame[f"{prefix}_ml_diff"] = frame[cols[1]] - frame[cols[2]]
    frame[f"{prefix}_el_diff"] = frame[cols[0]] - frame[cols[2]]
    frame[f"{prefix}_mean3"] = values.mean(axis=1)
    frame[f"{prefix}_std3"] = values.std(axis=1)
    frame[f"{prefix}_range3"] = values.max(axis=1) - values.min(axis=1)
    frame[f"{prefix}_curvature"] = frame[cols[0]] - 2.0 * frame[cols[1]] + frame[cols[2]]
    frame[f"{prefix}_argmax3"] = values.to_numpy().argmax(axis=1).astype(np.float32)
    frame[f"{prefix}_argmin3"] = values.to_numpy().argmin(axis=1).astype(np.float32)


def make_features(df: pd.DataFrame, has_label: bool) -> pd.DataFrame:
    drop_cols = ["id"] + (["label"] if has_label else [])
    out = df.drop(columns=drop_cols).copy()
    add_triplet_features(out, "rate_phase", ["early_rate_mean", "mid_rate_mean", "late_rate_mean"])
    add_triplet_features(out, "volume_phase", ["early_volume_mean", "mid_volume_mean", "late_volume_mean"])
    add_triplet_features(out, "payload_phase", ["early_payload_mean", "mid_payload_mean", "late_payload_mean"])

    # Cross-channel differences are less sensitive to global target-domain level shifts.
    for phase in ["early", "mid", "late"]:
        out[f"{phase}_volume_minus_rate"] = out[f"{phase}_volume_mean"] - out[f"{phase}_rate_mean"]
        out[f"{phase}_payload_minus_volume"] = out[f"{phase}_payload_mean"] - out[f"{phase}_volume_mean"]
        out[f"{phase}_payload_minus_rate"] = out[f"{phase}_payload_mean"] - out[f"{phase}_rate_mean"]

    base_pairs = [
        ("volume_rate_mean", "traffic_rate_mean"),
        ("payload_unit_mean", "volume_rate_mean"),
        ("payload_unit_mean", "traffic_rate_mean"),
        ("volume_rate_dispersion", "traffic_rate_dispersion"),
        ("payload_unit_dispersion", "volume_rate_dispersion"),
        ("direction_rate_gap", "direction_gap_dispersion"),
        ("pattern_transition_ratio", "pattern_diversity_ratio"),
        ("pattern_mix_density", "pattern_concentration"),
        ("rate_shift_score", "traffic_trend_score"),
        ("volume_shift_score", "volume_trend_score"),
        ("payload_shift_score", "payload_trend_score"),
    ]
    for left, right in base_pairs:
        out[f"{left}_minus_{right}"] = out[left] - out[right]
        out[f"{left}_times_{right}"] = out[left] * out[right]

    out["template_time_minus_peak"] = out["behavior_template_time"] - 0.5 * (
        out["traffic_peak_position"] + out["volume_peak_position"]
    )
    out["template_volume_minus_flow"] = out["behavior_template_volume"] - 0.5 * (
        out["volume_rate_mean"] + out["volume_phase_mean3"]
    )
    out["template_control_minus_signal"] = out["behavior_template_control"] - out["control_signal_intensity"]
    out["compactness_minus_irregularity"] = out["behavior_compactness_score"] - (
        out["traffic_irregularity"] + out["volume_irregularity"] + out["payload_irregularity"]
    ) / 3.0
    out["spike_ratio_gap"] = out["traffic_rate_spike_ratio"] - out["volume_rate_spike_ratio"]
    out["high_ratio_gap"] = out["high_rate_ratio"] - out["high_volume_ratio"]
    out["jump_intensity"] = out["traffic_jump_mean"] * out["traffic_jump_max"]
    out["pattern_switch_x_change"] = out["pattern_switch_frequency"] * out["pattern_change_ratio"]
    return out.astype(np.float32)


def build_model(seed: int, num_classes: int, n_estimators: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="multiclass",
        num_class=num_classes,
        n_estimators=n_estimators,
        learning_rate=0.045,
        num_leaves=79,
        min_child_samples=20,
        subsample=0.88,
        colsample_bytree=0.82,
        reg_lambda=1.4,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def main() -> None:
    args = parse_args()
    MODELS.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(RAW / "train_data.csv")
    test = pd.read_csv(RAW / "test_data.csv")
    x = make_features(train, has_label=True)
    x_test = make_features(test, has_label=False)
    encoder = LabelEncoder()
    y = encoder.fit_transform(train["label"]).astype(np.int64)
    classes = encoder.classes_.tolist()

    run_id = args.run_id or f"lgbm_engineered_n{args.n_estimators}_f{args.folds}_s{args.seed}"
    run_dir = MODELS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof = np.zeros((len(train), len(classes)), dtype=np.float64)
    test_proba = np.zeros((len(test), len(classes)), dtype=np.float64)
    fold_scores: list[float] = []
    for fold, (tr_idx, va_idx) in enumerate(cv.split(x, y), start=1):
        model = build_model(args.seed + fold - 1, len(classes), args.n_estimators)
        model.fit(x.iloc[tr_idx], y[tr_idx])
        valid_proba = model.predict_proba(x.iloc[va_idx])
        oof[va_idx] = valid_proba
        test_proba += model.predict_proba(x_test) / args.folds
        score = float(f1_score(y[va_idx], valid_proba.argmax(axis=1), average="macro"))
        fold_scores.append(score)
        print(f"fold={fold} macro_f1={score:.6f}")

    local_score = float(f1_score(y, oof.argmax(axis=1), average="macro"))
    np.save(run_dir / "oof_proba.npy", oof)
    np.save(run_dir / "test_proba.npy", test_proba)
    pd.DataFrame(
        {
            "id": train["id"],
            "true_label": train["label"],
            "pred_label": encoder.inverse_transform(oof.argmax(axis=1)),
        }
    ).to_csv(run_dir / "oof_predictions.csv", index=False, encoding="utf-8")
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "strategy": "engineered_lgbm",
        "seed": args.seed,
        "folds": args.folds,
        "classes": classes,
        "features": x.columns.tolist(),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "feature_count": int(x.shape[1]),
        "fold_scores": fold_scores,
        "local_macro_f1": local_score,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"run_id": run_id, "local_macro_f1": local_score, "feature_count": int(x.shape[1])}))


if __name__ == "__main__":
    main()
