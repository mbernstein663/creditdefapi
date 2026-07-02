from __future__ import annotations

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
    ACCEPTED_RISK_FEATURES,
    ARTIFACT_DIR,
    DEFAULT_ACCEPTED_BUNDLE,
    DEFAULT_LGD,
    DEFAULT_REQUIRED_RETURN,
    FORBIDDEN_FEATURE_COLUMNS,
    POST_PRICING_FIELDS,
    PROFIT_INPUT_COLUMNS,
    REPORT_DIR,
    ROOT,
    TARGET,
)
from src.models import fit_model, predict_raw_default
from src.profit import policy_metrics, policy_sensitivity
from src.preprocessing import (
    prepare_accepted_loans,
    split_chronological,
    split_count_report,
    split_manifest,
    split_row_count_report,
    split_summary,
)


def _read_accepted(path, sample=None):
    needed = set(ACCEPTED_RISK_FEATURES + PROFIT_INPUT_COLUMNS)
    needed |= {"id", "term", "loan_status", "issue_d", "total_pymnt"}
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
        profit = policy_metrics(
            validation_df,
            p_validation,
            lgd=DEFAULT_LGD,
            required_return=DEFAULT_REQUIRED_RETURN,
        )
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
    source_fingerprint = file_fingerprint(csv_path)
    training_timestamp = datetime.now(timezone.utc).isoformat()
    versions = package_versions()
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
    raw_selected = predict_raw_default(selected["model"], validation_df, ACCEPTED_RISK_FEATURES)
    p_selected = selected["calibrator"].predict(raw_selected)
    bundle = ModelBundle(
        model=selected["model"],
        calibrator=selected["calibrator"],
        feature_columns=list(ACCEPTED_RISK_FEATURES),
        model_type="accepted",
        policy={
            "lgd": DEFAULT_LGD,
            "required_return": DEFAULT_REQUIRED_RETURN,
            "approval_rule": "expected_return >= required_return",
        },
        required_input_schema={
            "risk_features": list(ACCEPTED_RISK_FEATURES),
            "profit_inputs": list(PROFIT_INPUT_COLUMNS),
        },
        metadata={
            "target": TARGET,
            "target_definition": "resolved funded accepted-loan default target",
            "target_limitation": "predicts eventual resolved default among accepted/funded loans, not all applicant risk",
            "fixed_horizon_extension": "not implemented; add issue-date plus performance-window logic before labeling current/unresolved loans",
            "is_smoke_sample": bool(sample),
            "sample_rows_requested": sample,
            "source_fingerprint": _repo_fingerprint(source_fingerprint),
            "resolved_row_count": int(len(accepted)),
            "split_manifest": manifest,
            "selected_candidate": selected["name"],
            "selection_rule": "max validation expected_profit at required_return=0.0; tie-breaker lower validation brier_score",
            "target_good_statuses": "see src.config.GOOD_STATUSES",
            "target_bad_statuses": "see src.config.BAD_STATUSES",
            "forbidden_feature_columns": sorted(FORBIDDEN_FEATURE_COLUMNS),
            "post_pricing_fields": list(POST_PRICING_FIELDS),
            "scoring_moment": "post-pricing/post-underwriting only",
            "random_state": 42,
            "package_versions": versions,
            "split_rule": "chronological 60/15/15/10 by issue_d after dropping unresolved loans",
            "rejected_data_handling": "rejected applications are unlabeled and excluded",
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
        "feature_columns": list(ACCEPTED_RISK_FEATURES),
        "post_pricing_fields": list(POST_PRICING_FIELDS),
        "selected_candidate": selected["name"],
        "selection_rule": bundle.metadata["selection_rule"],
        "selected_policy": bundle.policy,
        "selected_subgroup_calibration": subgroup_calibration_summary(validation_df, TARGET, p_selected),
        "selected_policy_sensitivity": policy_sensitivity(validation_df, p_selected),
        "candidates": [
            {
                "name": c["name"],
                "class_weight": c["class_weight"],
                "probability": c["probability"],
                "profit_policy": c["profit_policy"],
            }
            for c in candidates
        ],
    }
    _write_json(report_dir / "accepted_validation_metrics.json", validation_report)
    _write_json(
        ROOT / "docs" / "accepted_model_card.json",
        {
            "model_type": bundle.model_type,
            "target": bundle.metadata["target_definition"],
            "target_limitation": bundle.metadata["target_limitation"],
            "scoring_moment": bundle.metadata["scoring_moment"],
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
