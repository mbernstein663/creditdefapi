from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier

from src.artifacts import ModelBundle, save_model_bundle
from src.calibration import ProbabilityCalibrator
from src.config import ACCEPTED_RISK_FEATURES, FORBIDDEN_FEATURE_COLUMNS

FRONTEND_FEATURES = ["loan_amnt", "int_rate", "annual_inc", "dti", "fico_range_low"]


def _bundle(feature_columns: list[str], model_version: str) -> ModelBundle:
    frame = pd.DataFrame([[0] * len(feature_columns), [1] * len(feature_columns)], columns=feature_columns)
    model = DummyClassifier(strategy="prior").fit(frame, [0, 1])
    metadata = {
        "bundle_schema_version": 1,
        "model_version": model_version,
        "source_fingerprint": {"sha256": "synthetic-test-fixture", "size_bytes": 0},
        "split_manifest": {"fixture": True},
        "split_summary": [{"split": "synthetic", "rows": 2, "default_rate": 0.5}],
        "target_definition": "synthetic fixture for CI/demo inference only; not model evidence",
        "forbidden_feature_columns": sorted(FORBIDDEN_FEATURE_COLUMNS),
        "training_timestamp": "2000-01-01T00:00:00+00:00",
        "package_versions": {"fixture": "deterministic"},
        "selected_model_name": "dummy_prior",
        "selected_model_type": "synthetic_fixture",
        "calibration_method": "identity",
        "artifact_data_context": "synthetic_test_fixture",
        "limitations": ["CI/demo inference only; never use as real model evidence"],
        "risk_band_thresholds": {"low_max": 0.1, "medium_max": 0.2},
    }
    return ModelBundle(
        model=model,
        calibrator=ProbabilityCalibrator("identity").fit([0, 1], [0, 1]),
        feature_columns=feature_columns,
        model_type="synthetic_fixture",
        metadata=metadata,
        required_input_schema={"schema_version": 1, "required_fields": feature_columns},
    )


def generate(output_dir: str | Path) -> tuple[Path, Path]:
    output = Path(output_dir)
    return (
        save_model_bundle(
            _bundle(list(ACCEPTED_RISK_FEATURES), "synthetic-ci-accepted-v1"), output / "accepted_model.joblib"
        ),
        save_model_bundle(
            _bundle(FRONTEND_FEATURES, "synthetic-ci-frontend-v1"), output / "frontend_model.joblib"
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic model bundles for CI/demo inference only.")
    parser.add_argument("output_dir", nargs="?", default="artifacts")
    generate(parser.parse_args().output_dir)
