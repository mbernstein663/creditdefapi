from __future__ import annotations

import argparse
import json

import pandas as pd

from src.artifacts import file_fingerprint, load_model_bundle
from src.config import (
    ACCEPTED_CSV,
    DEFAULT_ACCEPTED_BUNDLE,
    PROFIT_INPUT_COLUMNS,
    REPORT_DIR,
    TARGET,
)
from src.calibration import subgroup_calibration_summary
from src.evaluation import evaluate_probability, evaluate_profit_policy
from src.models import predict_raw_default
from src.preprocessing import prepare_accepted_loans


def _read_accepted(path, feature_columns, sample=None):
    needed = set(feature_columns + PROFIT_INPUT_COLUMNS)
    needed |= {"id", "term", "loan_status", "issue_d", "total_pymnt"}
    return pd.read_csv(path, usecols=lambda col: col in needed, low_memory=False, nrows=sample)


def evaluate_locked_model(bundle_path=DEFAULT_ACCEPTED_BUNDLE, csv_path=ACCEPTED_CSV, sample=None):
    bundle = load_model_bundle(bundle_path)
    expected_source = bundle.metadata.get("source_fingerprint")
    actual_source = file_fingerprint(csv_path)
    if expected_source and (
        expected_source.get("sha256") != actual_source.get("sha256")
        or expected_source.get("size_bytes") != actual_source.get("size_bytes")
    ):
        raise ValueError("source CSV fingerprint does not match the locked bundle")

    test_ids = set(bundle.metadata.get("split_manifest", {}).get("test_ids", []))
    if not test_ids:
        raise ValueError("locked bundle is missing saved test split IDs")

    accepted = prepare_accepted_loans(_read_accepted(csv_path, bundle.feature_columns, sample=sample))
    test_df = accepted.loc[accepted["id"].astype(str).isin(test_ids)].copy()
    if len(test_df) != len(test_ids):
        raise ValueError("source CSV does not contain every saved test split ID")
    if bundle.calibrator is None:
        raise ValueError("locked evaluation requires calibrated probabilities")

    raw = predict_raw_default(bundle.model, test_df, bundle.feature_columns)
    p_default = bundle.calibrator.predict(raw)
    result = {
        "probability": evaluate_probability(bundle, test_df),
        "subgroup_calibration": subgroup_calibration_summary(test_df, TARGET, p_default),
        "profit_policy": evaluate_profit_policy(bundle, test_df),
        "locked_policy": bundle.policy,
        "artifact_path": str(bundle_path),
        "selected_candidate": bundle.metadata.get("selected_candidate"),
        "training_timestamp": bundle.metadata.get("training_timestamp"),
        "package_versions": bundle.metadata.get("package_versions"),
        "split_manifest": {k: v for k, v in bundle.metadata.get("split_manifest", {}).items() if k != "test_ids"},
        "is_smoke_sample": bool(sample or bundle.metadata.get("is_smoke_sample")),
        "source_fingerprint": actual_source,
        "test_row_count": int(len(test_df)),
        "test_default_rate": float(test_df[TARGET].mean()) if len(test_df) else None,
        "note": "test set is evaluated after loading the locked bundle; no selection happens here",
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / ("locked_test_metrics_smoke.json" if result["is_smoke_sample"] else "locked_test_metrics.json")
    path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description="Evaluate one locked accepted-loan model on test data.")
    parser.add_argument("--bundle", default=DEFAULT_ACCEPTED_BUNDLE)
    parser.add_argument("--csv", default=ACCEPTED_CSV)
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()
    print(evaluate_locked_model(args.bundle, args.csv, sample=args.sample))


if __name__ == "__main__":
    main()
