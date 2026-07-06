from __future__ import annotations

import argparse

import pandas as pd

from src.artifacts import file_fingerprint, load_model_bundle, save_model_bundle
from src.config import ACCEPTED_CSV, DEFAULT_ACCEPTED_BUNDLE, REPORT_DIR
from src.evaluation import generate_evaluation_reports
from src.models import predict_raw_default
from src.preprocessing import prepare_accepted_loans


def _read_accepted(path, feature_columns, sample=None):
    needed = set(feature_columns) | {"id", "loan_status", "issue_d", "term"}
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

    raw = predict_raw_default(bundle.model, test_df, bundle.feature_columns)
    p_default = bundle.calibrator.predict(raw)
    report_dir = REPORT_DIR / "test" / "smoke" if sample else REPORT_DIR / "test"
    outputs = generate_evaluation_reports(bundle, test_df, p_default, report_dir, "test")
    save_model_bundle(bundle, bundle_path)
    return outputs["metrics_summary"]


def main():
    parser = argparse.ArgumentParser(description="Evaluate the locked accepted-loan bundle on the saved test split.")
    parser.add_argument("--bundle", default=DEFAULT_ACCEPTED_BUNDLE)
    parser.add_argument("--csv", default=ACCEPTED_CSV)
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()
    print(evaluate_locked_model(args.bundle, args.csv, sample=args.sample))


if __name__ == "__main__":
    main()
