from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.artifacts import file_fingerprint, load_model_bundle
from src.config import ACCEPTED_CSV, DEFAULT_ACCEPTED_BUNDLE, DEFAULT_PREPROCESSED_ACCEPTED_BUNDLE, REPORT_DIR
from src.evaluation import baseline_comparison_metrics, generate_evaluation_reports
from src.models import predict_raw_default
from src.preprocessing import load_preprocessed_accepted_loans, preprocess_accepted_loans, save_preprocessed_accepted_loans


def _verify_validation_report(bundle, report_dir: Path) -> None:
    path = report_dir / "validation" / "metrics_summary.json"
    try:
        metrics = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"validation report is missing: {path}") from exc
    metadata = bundle.metadata or {}
    expected = {
        "selected_model": metadata.get("selected_model_name"),
        "training_timestamp": metadata.get("training_timestamp"),
    }
    for field, value in expected.items():
        if metrics.get(field) != value:
            raise ValueError(f"validation report does not match locked bundle: {field}")


def evaluate_locked_model(bundle_path=DEFAULT_ACCEPTED_BUNDLE, csv_path=ACCEPTED_CSV, sample=None):
    bundle_fingerprint = file_fingerprint(bundle_path)
    bundle = load_model_bundle(bundle_path)
    validation_root = REPORT_DIR / "smoke" if bundle.metadata.get("sample_rows_requested") else REPORT_DIR
    _verify_validation_report(bundle, validation_root)
    expected_source = bundle.metadata.get("source_fingerprint")
    actual_source = file_fingerprint(csv_path)
    if expected_source and (
        expected_source.get("sha256") != actual_source.get("sha256")
        or expected_source.get("size_bytes") != actual_source.get("size_bytes")
    ):
        raise ValueError("source CSV fingerprint does not match the locked bundle")

    test_ids = {str(value) for value in bundle.metadata.get("split_manifest", {}).get("test_ids", [])}
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
    actual_test_ids = set(test_df["id"].astype(str))
    if actual_test_ids != test_ids:
        missing = sorted(test_ids - actual_test_ids)
        unexpected = sorted(actual_test_ids - test_ids)
        raise ValueError(
            "source CSV test split IDs do not match the locked bundle "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )

    raw = predict_raw_default(bundle.model, test_df, bundle.feature_columns)
    p_default = bundle.calibrator.predict(raw)
    baselines = baseline_comparison_metrics(bundle, preprocessing.splits, p_default)
    report_dir = REPORT_DIR / "smoke" / "test" if sample else REPORT_DIR / "test"
    outputs = generate_evaluation_reports(bundle, test_df, p_default, report_dir, "test", baseline_comparison=baselines)
    manifest = {
        "model_bundle_sha256": bundle_fingerprint["sha256"],
        "model_version": bundle.metadata.get("model_version"),
        "selected_model": bundle.metadata.get("selected_model_name"),
        "locked_test_metrics": json.loads(outputs["metrics_summary"].read_text(encoding="utf-8")),
        "baseline_comparison": baselines,
    }
    (report_dir / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
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
