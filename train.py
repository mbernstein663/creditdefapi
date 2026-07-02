from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import pandas as pd

from src.artifacts import ModelBundle, file_fingerprint, package_versions, save_model_bundle
from src.calibration import ProbabilityCalibrator, calibration_summary, save_reliability_plot
from src.config import (
    ACCEPTED_CSV,
    ACCEPTED_RISK_FEATURES,
    ARTIFACT_DIR,
    DEFAULT_ACCEPTED_BUNDLE,
    DEFAULT_LGD,
    FORBIDDEN_FEATURE_COLUMNS,
    PROFIT_INPUT_COLUMNS,
    REPORT_DIR,
    TARGET,
)
from src.evaluation import evaluate_profit_policy
from src.models import fit_model, predict_raw_default
from src.profit import policy_metrics
from src.preprocessing import (
    prepare_accepted_loans,
    split_chronological,
    split_count_report,
    split_manifest,
    split_row_count_report,
)


def _read_accepted(path, sample=None):
    needed = set(ACCEPTED_RISK_FEATURES + PROFIT_INPUT_COLUMNS)
    needed |= {"id", "term", "loan_status", "issue_d", "total_pymnt"}
    return pd.read_csv(path, usecols=lambda col: col in needed, low_memory=False, nrows=sample)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _paths(output_path, sample):
    report_dir = REPORT_DIR / "smoke" if sample else REPORT_DIR
    output = DEFAULT_ACCEPTED_BUNDLE.with_name("accepted_model_smoke.joblib") if sample else output_path
    return output, report_dir


def select_accepted_candidate(train_df, calibration_df, validation_df):
    candidates = []
    for name, class_weight in [("logistic_balanced", "balanced"), ("logistic", None)]:
        model = fit_model(train_df, ACCEPTED_RISK_FEATURES, kind="accepted", class_weight=class_weight)
        raw_calibration = predict_raw_default(model, calibration_df, ACCEPTED_RISK_FEATURES)
        calibrator = ProbabilityCalibrator("isotonic").fit(raw_calibration, calibration_df[TARGET])
        raw_validation = predict_raw_default(model, validation_df, ACCEPTED_RISK_FEATURES)
        p_validation = calibrator.predict(raw_validation)
        probability = calibration_summary(validation_df[TARGET], p_validation)
        profit = policy_metrics(validation_df, p_validation, lgd=DEFAULT_LGD)
        candidates.append(
            {
                "name": name,
                "class_weight": class_weight,
                "model": model,
                "calibrator": calibrator,
                "probability": probability,
                "profit_policy": profit,
            }
        )
    return max(
        candidates,
        key=lambda c: (c["profit_policy"]["expected_profit"], -c["probability"]["brier_score"]),
    ), candidates


def train_accepted_model(csv_path=ACCEPTED_CSV, output_path=DEFAULT_ACCEPTED_BUNDLE, sample=None):
    output_path, report_dir = _paths(output_path, sample)
    report_dir.mkdir(parents=True, exist_ok=True)
    accepted = prepare_accepted_loans(_read_accepted(csv_path, sample=sample))
    splits = split_chronological(accepted)
    manifest = split_manifest(splits)
    split_row_count_report(splits).to_csv(report_dir / "supervised_row_counts_by_split_year.csv", index=False)
    split_count_report({k: v for k, v in splits.items() if k != "test"}).to_csv(
        report_dir / "non_test_label_counts_by_split_year.csv", index=False
    )

    train_df = splits["train"].dropna(subset=[TARGET])
    calibration_df = splits["calibration"].dropna(subset=[TARGET])
    validation_df = splits["validation"].dropna(subset=[TARGET])

    selected, candidates = select_accepted_candidate(train_df, calibration_df, validation_df)
    bundle = ModelBundle(
        model=selected["model"],
        calibrator=selected["calibrator"],
        feature_columns=list(ACCEPTED_RISK_FEATURES),
        model_type="accepted",
        policy={
            "lgd": DEFAULT_LGD,
            "required_return": None,
            "approval_rule": "expected_profit > 0",
        },
        required_input_schema={
            "risk_features": list(ACCEPTED_RISK_FEATURES),
            "profit_inputs": list(PROFIT_INPUT_COLUMNS),
        },
        metadata={
            "target": TARGET,
            "is_smoke_sample": bool(sample),
            "sample_rows_requested": sample,
            "source_fingerprint": file_fingerprint(csv_path),
            "resolved_row_count": int(len(accepted)),
            "split_manifest": manifest,
            "selected_candidate": selected["name"],
            "selection_rule": "max validation expected_profit; tie-breaker lower validation brier_score",
            "target_good_statuses": "see src.config.GOOD_STATUSES",
            "target_bad_statuses": "see src.config.BAD_STATUSES",
            "forbidden_feature_columns": sorted(FORBIDDEN_FEATURE_COLUMNS),
            "package_versions": package_versions(),
            "split_rule": "chronological 60/15/15/10 by issue_d after dropping unresolved loans",
            "rejected_data_handling": "rejected applications are unlabeled and excluded",
            "training_timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
    saved = save_model_bundle(bundle, output_path)
    _write_json(
        report_dir / "accepted_validation_metrics.json",
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
                    "profit_policy": c["profit_policy"],
                }
                for c in candidates
            ],
        },
    )
    raw_selected = predict_raw_default(bundle.model, validation_df, ACCEPTED_RISK_FEATURES)
    p_selected = bundle.calibrator.predict(raw_selected)
    save_reliability_plot(validation_df[TARGET], p_selected, report_dir / "accepted_reliability.png")
    return saved


def main():
    parser = argparse.ArgumentParser(description="Train accepted-loan default/profit model.")
    parser.add_argument("--csv", default=ACCEPTED_CSV)
    parser.add_argument("--output", default=DEFAULT_ACCEPTED_BUNDLE)
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    print(train_accepted_model(args.csv, args.output, sample=args.sample))


if __name__ == "__main__":
    main()
