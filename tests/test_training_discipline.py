import json

import pytest
import pandas as pd

import evaluate_locked
import train
from src.artifacts import ModelBundle, file_fingerprint, save_model_bundle
from tests.test_artifacts_batch_scoring import DummyModel, IdentityCalibrator


def test_sample_training_uses_smoke_artifact_and_report_paths():
    output, report_dir = train._paths(train.DEFAULT_ACCEPTED_BUNDLE, sample=100)

    assert output.name == "accepted_model_smoke.joblib"
    assert report_dir.name == "smoke"


def test_locked_evaluation_uses_saved_test_ids(tmp_path):
    csv_path = tmp_path / "accepted.csv"
    pd.DataFrame(
        {
            "id": ["1", "2", "3"],
            "loan_amnt": [1000, 1000, 1000],
            "funded_amnt": [1000, 1000, 1000],
            "term": [" 36 months", " 36 months", " 36 months"],
            "installment": [40, 40, 40],
            "loan_status": ["Fully Paid", "Charged Off", "Fully Paid"],
            "issue_d": ["Jan-2018", "Feb-2018", "Mar-2018"],
            "total_pymnt": [1100, 200, 1200],
        }
    ).to_csv(csv_path, index=False)
    bundle = ModelBundle(
        DummyModel(),
        IdentityCalibrator(),
        ["loan_amnt"],
        "accepted",
        policy={"lgd": 1.0, "required_return": None, "approval_rule": "expected_profit > 0"},
        metadata={
            "source_fingerprint": file_fingerprint(csv_path),
            "split_manifest": {"test_ids": ["2"]},
        },
    )
    bundle_path = tmp_path / "bundle.joblib"
    save_model_bundle(bundle, bundle_path)

    output = evaluate_locked.evaluate_locked_model(bundle_path, csv_path)
    result = json.loads(output.read_text(encoding="utf-8"))

    assert output.name == "locked_test_metrics.json"
    assert result["test_row_count"] == 1
    assert result["test_default_rate"] == 1.0
    assert result["profit_policy"]["actual_default_rate"] == 1.0
    assert result["profit_policy"]["total_realized_profit"] == -800.0


def test_locked_evaluation_fails_without_saved_test_ids(tmp_path):
    csv_path = tmp_path / "accepted.csv"
    pd.DataFrame(
        {
            "id": ["1"],
            "loan_amnt": [1000],
            "funded_amnt": [1000],
            "term": [" 36 months"],
            "installment": [40],
            "loan_status": ["Fully Paid"],
            "issue_d": ["Jan-2018"],
            "total_pymnt": [1100],
        }
    ).to_csv(csv_path, index=False)
    bundle_path = tmp_path / "bundle.joblib"
    save_model_bundle(
        ModelBundle(
            DummyModel(),
            IdentityCalibrator(),
            ["loan_amnt"],
            "accepted",
            metadata={"source_fingerprint": file_fingerprint(csv_path)},
        ),
        bundle_path,
    )

    with pytest.raises(ValueError, match="missing saved test split IDs"):
        evaluate_locked.evaluate_locked_model(bundle_path, csv_path)


def test_file_fingerprint_changes_when_same_size_content_changes(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("abc\n", encoding="utf-8")
    first = file_fingerprint(path)
    path.write_text("abd\n", encoding="utf-8")
    second = file_fingerprint(path)

    assert first["size_bytes"] == second["size_bytes"]
    assert first["sha256"] != second["sha256"]
