from __future__ import annotations

import argparse
import json

from src.artifacts import file_fingerprint, load_model_bundle, save_model_bundle
from src.config import ACCEPTED_CSV, DEFAULT_ACCEPTED_BUNDLE, DEFAULT_PREPROCESSED_ACCEPTED_BUNDLE, REPORT_DIR
from src.evaluation import generate_evaluation_reports
from src.models import predict_raw_default
from src.preprocessing import load_preprocessed_accepted_loans, preprocess_accepted_loans, save_preprocessed_accepted_loans


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

    try:
        preprocessing = load_preprocessed_accepted_loans(DEFAULT_PREPROCESSED_ACCEPTED_BUNDLE)
        if (
            preprocessing.source_fingerprint != actual_source
            or getattr(preprocessing, "sample_rows_requested", None) != sample
        ):
            preprocessing = preprocess_accepted_loans(csv_path, sample=sample, selected_features=bundle.feature_columns)
            save_preprocessed_accepted_loans(preprocessing, DEFAULT_PREPROCESSED_ACCEPTED_BUNDLE)
    except Exception:
        preprocessing = preprocess_accepted_loans(csv_path, sample=sample, selected_features=bundle.feature_columns)
        save_preprocessed_accepted_loans(preprocessing, DEFAULT_PREPROCESSED_ACCEPTED_BUNDLE)
    test_df = preprocessing.splits["test"].copy()
    if len(test_df) != len(test_ids):
        raise ValueError("source CSV does not contain every saved test split ID")

    raw = predict_raw_default(bundle.model, test_df, bundle.feature_columns)
    p_default = bundle.calibrator.predict(raw)
    report_dir = REPORT_DIR / "smoke" / "test" if sample else REPORT_DIR / "test"
    outputs = generate_evaluation_reports(bundle, test_df, p_default, report_dir, "test")
    bundle.metadata["locked_test_metrics_summary"] = json.loads(outputs["metrics_summary"].read_text(encoding="utf-8"))
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
