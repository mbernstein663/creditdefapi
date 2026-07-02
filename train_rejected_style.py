from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import pandas as pd

from src.artifacts import ModelBundle, file_fingerprint, package_versions, save_model_bundle
from src.calibration import ProbabilityCalibrator, calibration_summary, save_reliability_plot
from src.config import (
    ACCEPTED_CSV,
    ACCEPTED_TO_REJECTED_FEATURE_MAP,
    ARTIFACT_DIR,
    DEFAULT_REJECTED_STYLE_BUNDLE,
    FORBIDDEN_FEATURE_COLUMNS,
    REJECTED_STYLE_RISK_FEATURES,
    REPORT_DIR,
    TARGET,
)
from src.models import fit_model, predict_raw_default
from src.preprocessing import (
    map_accepted_to_rejected_style,
    prepare_accepted_loans,
    split_chronological,
    split_count_report,
    split_manifest,
    split_row_count_report,
)


def _read_accepted_for_rejected_style(path, sample=None):
    needed = set(ACCEPTED_TO_REJECTED_FEATURE_MAP) | {"id", "loan_status", "issue_d"}
    return pd.read_csv(path, usecols=lambda col: col in needed, low_memory=False, nrows=sample)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _paths(output_path, sample):
    report_dir = REPORT_DIR / "smoke" if sample else REPORT_DIR
    output = (
        DEFAULT_REJECTED_STYLE_BUNDLE.with_name("rejected_style_model_smoke.joblib")
        if sample
        else output_path
    )
    return output, report_dir


def select_rejected_style_candidate(train_df, calibration_df, validation_df):
    candidates = []
    for name, class_weight in [("logistic_balanced", "balanced"), ("logistic", None)]:
        model = fit_model(
            train_df,
            REJECTED_STYLE_RISK_FEATURES,
            kind="rejected_style",
            class_weight=class_weight,
        )
        raw_calibration = predict_raw_default(model, calibration_df, REJECTED_STYLE_RISK_FEATURES)
        calibrator = ProbabilityCalibrator("isotonic").fit(raw_calibration, calibration_df[TARGET])
        raw_validation = predict_raw_default(model, validation_df, REJECTED_STYLE_RISK_FEATURES)
        p_validation = calibrator.predict(raw_validation)
        candidates.append(
            {
                "name": name,
                "class_weight": class_weight,
                "model": model,
                "calibrator": calibrator,
                "probability": calibration_summary(validation_df[TARGET], p_validation),
            }
        )
    return min(candidates, key=lambda c: c["probability"]["brier_score"]), candidates


def train_rejected_style_model(
    csv_path=ACCEPTED_CSV,
    output_path=DEFAULT_REJECTED_STYLE_BUNDLE,
    sample=None,
):
    output_path, report_dir = _paths(output_path, sample)
    report_dir.mkdir(parents=True, exist_ok=True)
    accepted = prepare_accepted_loans(_read_accepted_for_rejected_style(csv_path, sample=sample))
    splits = split_chronological(accepted)
    manifest = split_manifest(splits)
    split_row_count_report(splits).to_csv(
        report_dir / "rejected_style_supervised_row_counts_by_split_year.csv", index=False
    )
    split_count_report({k: v for k, v in splits.items() if k != "test"}).to_csv(
        report_dir / "rejected_style_non_test_label_counts_by_split_year.csv", index=False
    )

    mapped = {name: map_accepted_to_rejected_style(frame) for name, frame in splits.items()}
    train_df = mapped["train"].dropna(subset=[TARGET])
    calibration_df = mapped["calibration"].dropna(subset=[TARGET])
    validation_df = mapped["validation"].dropna(subset=[TARGET])

    selected, candidates = select_rejected_style_candidate(train_df, calibration_df, validation_df)

    bundle = ModelBundle(
        model=selected["model"],
        calibrator=selected["calibrator"],
        feature_columns=list(REJECTED_STYLE_RISK_FEATURES),
        model_type="rejected_style",
        policy={
            "lgd": 1.0,
            "required_return": None,
            "approval_rule": "review unless profit inputs are supplied",
        },
        required_input_schema={
            "risk_features": list(REJECTED_STYLE_RISK_FEATURES),
            "optional_profit_inputs": ["funded_amnt", "term_months", "installment"],
        },
        metadata={
            "target": TARGET,
            "is_smoke_sample": bool(sample),
            "sample_rows_requested": sample,
            "source_fingerprint": file_fingerprint(csv_path),
            "resolved_row_count": int(len(accepted)),
            "split_manifest": manifest,
            "selected_candidate": selected["name"],
            "selection_rule": "lowest validation brier_score",
            "accepted_to_rejected_feature_map": ACCEPTED_TO_REJECTED_FEATURE_MAP,
            "excluded_rejected_decision_fields": ["policy_code"],
            "forbidden_feature_columns": sorted(FORBIDDEN_FEATURE_COLUMNS),
            "package_versions": package_versions(),
            "rejected_data_handling": "rejected applications are unlabeled and excluded",
            "output_limits": "risk/review only unless profit inputs are supplied",
            "training_timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
    saved = save_model_bundle(bundle, output_path)
    _write_json(
        report_dir / "rejected_style_validation_metrics.json",
        {
            "is_smoke_sample": bool(sample),
            "sample_rows_requested": sample,
            "source_fingerprint": file_fingerprint(csv_path),
            "resolved_row_count": int(len(accepted)),
            "split_manifest": {k: v for k, v in manifest.items() if k != "test_ids"},
            "selected_candidate": selected["name"],
            "selection_rule": bundle.metadata["selection_rule"],
            "candidates": [
                {
                    "name": c["name"],
                    "class_weight": c["class_weight"],
                    "probability": c["probability"],
                }
                for c in candidates
            ],
        },
    )
    raw_selected = predict_raw_default(bundle.model, validation_df, REJECTED_STYLE_RISK_FEATURES)
    p_validation = bundle.calibrator.predict(raw_selected)
    save_reliability_plot(
        validation_df[TARGET],
        p_validation,
        report_dir / "rejected_style_reliability.png",
    )
    return saved


def main():
    parser = argparse.ArgumentParser(description="Train rejected-style risk/review model.")
    parser.add_argument("--csv", default=ACCEPTED_CSV)
    parser.add_argument("--output", default=DEFAULT_REJECTED_STYLE_BUNDLE)
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    print(train_rejected_style_model(args.csv, args.output, sample=args.sample))


if __name__ == "__main__":
    main()
