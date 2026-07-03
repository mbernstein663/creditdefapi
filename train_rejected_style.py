from __future__ import annotations

"""Train an accepted-loan-outcome model projected onto rejected-application-style inputs."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.artifacts import ModelBundle, file_fingerprint, package_versions, save_model_bundle
from src.calibration import (
    ProbabilityCalibrator,
    calibration_summary,
    save_reliability_plot,
    subgroup_calibration_summary,
)
from src.config import (
    ACCEPTED_CSV,
    ACCEPTED_TO_REJECTED_FEATURE_MAP,
    ARTIFACT_DIR,
    DEFAULT_REJECTED_STYLE_BUNDLE,
    FORBIDDEN_FEATURE_COLUMNS,
    REJECTED_STYLE_RISK_FEATURES,
    REPORT_DIR,
    ROOT,
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
    split_summary,
)


def _read_accepted_for_rejected_style(path, sample=None):
    needed = set(ACCEPTED_TO_REJECTED_FEATURE_MAP) | {"id", "loan_status", "issue_d"}
    return pd.read_csv(path, usecols=lambda col: col in needed, low_memory=False, nrows=sample)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _repo_path(path):
    p = Path(path).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def _repo_fingerprint(fingerprint):
    out = dict(fingerprint)
    if out.get("path"):
        out["path"] = _repo_path(out["path"])
    return out


def _paths(output_path, sample):
    report_dir = REPORT_DIR / "validation" / "smoke" if sample else REPORT_DIR / "validation"
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
    source_fingerprint = file_fingerprint(csv_path)
    training_timestamp = datetime.now(timezone.utc).isoformat()
    versions = package_versions()
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
    raw_selected = predict_raw_default(selected["model"], validation_df, REJECTED_STYLE_RISK_FEATURES)
    p_selected = selected["calibrator"].predict(raw_selected)

    bundle = ModelBundle(
        model=selected["model"],
        calibrator=selected["calibrator"],
        feature_columns=list(REJECTED_STYLE_RISK_FEATURES),
        model_type="rejected_style",
        policy={
            "lgd": 1.0,
            "required_return": None,
            "approval_rule": "review only; expected profit is scenario math when profit inputs are supplied",
            "use_npv": False,
            "annual_discount_rate": 0.08,
            "servicing_cost_rate": 0.0,
            "recovery_rate": 0.25,
        },
        required_input_schema={
            "risk_features": list(REJECTED_STYLE_RISK_FEATURES),
            "optional_profit_inputs": ["funded_amnt", "term_months", "installment"],
        },
        metadata={
            "target": TARGET,
            "target_mode": "resolved_default",
            "target_definition": "resolved funded accepted-loan default target projected onto limited fields",
            "target_limitation": "not true rejected-applicant default risk; rejected applications have no repayment outcomes",
            "is_smoke_sample": bool(sample),
            "sample_rows_requested": sample,
            "source_fingerprint": _repo_fingerprint(source_fingerprint),
            "resolved_row_count": int(len(accepted)),
            "split_manifest": manifest,
            "selected_candidate": selected["name"],
            "selection_rule": "lowest validation brier_score",
            "accepted_to_rejected_feature_map": ACCEPTED_TO_REJECTED_FEATURE_MAP,
            "excluded_rejected_decision_fields": ["policy_code"],
            "forbidden_feature_columns": sorted(FORBIDDEN_FEATURE_COLUMNS),
            "product_mode": "pre_underwriting_applicant",
            "random_state": 42,
            "package_versions": versions,
            "rejected_data_handling": "rejected applications are unlabeled and excluded",
            "output_limits": "limited-field risk estimate; review only for rejected-application-style inputs",
            "calibration_method": "isotonic",
            "training_timestamp": training_timestamp,
        },
    )
    saved = save_model_bundle(bundle, output_path)
    validation_report = {
        "is_smoke_sample": bool(sample),
        "sample_rows_requested": sample,
        "artifact_path": str(saved),
        "training_timestamp": training_timestamp,
        "package_versions": versions,
        "source_fingerprint": source_fingerprint,
        "resolved_row_count": int(len(accepted)),
        "split_manifest": {k: v for k, v in manifest.items() if k != "test_ids"},
        "split_summary": split_summary(splits),
        "feature_columns": list(REJECTED_STYLE_RISK_FEATURES),
        "selected_candidate": selected["name"],
        "selection_rule": bundle.metadata["selection_rule"],
        "subgroup_calibration": subgroup_calibration_summary(validation_df, TARGET, p_selected),
        "candidates": [
            {
                "name": c["name"],
                "class_weight": c["class_weight"],
                "probability": c["probability"],
            }
            for c in candidates
        ],
    }
    _write_json(report_dir / "rejected_style_validation_metrics.json", validation_report)
    _write_json(
        ROOT / "docs" / "limited_field_model_card.json",
        {
            "model_type": bundle.model_type,
            "target": bundle.metadata["target_definition"],
            "target_limitation": bundle.metadata["target_limitation"],
            "artifact_path": _repo_path(saved),
            "feature_columns": bundle.feature_columns,
            "policy": bundle.policy,
            "selected_candidate": selected["name"],
            "selection_rule": bundle.metadata["selection_rule"],
            "split_summary": validation_report["split_summary"],
            "source_fingerprint": _repo_fingerprint(source_fingerprint),
            "training_timestamp": training_timestamp,
            "package_versions": versions,
        },
    )
    save_reliability_plot(
        validation_df[TARGET],
        p_selected,
        report_dir / "rejected_style_reliability.png",
    )
    return saved


def main():
    parser = argparse.ArgumentParser(description="Train limited-field risk/review model.")
    parser.add_argument("--csv", default=ACCEPTED_CSV)
    parser.add_argument("--output", default=DEFAULT_REJECTED_STYLE_BUNDLE)
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    print(train_rejected_style_model(args.csv, args.output, sample=args.sample))


if __name__ == "__main__":
    main()
