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


def _policy_selection_frame(realized):
    return pd.DataFrame(
        {
            "funded_amnt": [1000, 1000, 1000],
            "term_months": [12, 12, 12],
            "installment": [100, 100, 100],
            "total_pymnt": [1000 + value for value in realized],
            "default": [0, 1, 0],
        }
    )


def test_required_return_selector_chooses_profitable_validation_threshold():
    frame = _policy_selection_frame([100, -700, 500])

    policy, rows = train.select_required_return_policy(frame, [0.10, 0.10, 0.01])

    assert policy["required_return"] >= 0
    assert policy["required_return"] > 0.08
    assert policy["validation_expected_profit"] >= 0
    assert policy["good_profit_haircut"] in train.GOOD_PROFIT_HAIRCUT_CANDIDATES
    assert policy["validation_approval_count"] == 1
    assert policy["validation_realized_profit"] == 500
    assert policy["required_return"] in [row["required_return"] for row in rows]


def test_required_return_selector_rejects_all_when_validation_lending_loses():
    frame = _policy_selection_frame([-100, -100, -100])

    policy, rows = train.select_required_return_policy(frame, [0.10, 0.10, 0.01])

    assert policy["required_return"] >= 0
    assert policy["validation_approval_count"] == 0
    assert policy["good_profit_haircut"] in train.GOOD_PROFIT_HAIRCUT_CANDIDATES
    assert policy["validation_realized_profit"] == 0
    assert "non-negative scenario EV" in policy["profit_warning"]
    assert any(row["approval_count"] > 0 for row in rows)


def test_training_considers_planned_default_model_candidates():
    assert train.GOOD_PROFIT_HAIRCUT_CANDIDATES == [0.25, 0.35, 0.50, 0.65, 0.80, 1.00]
    assert train.DEFAULT_MODEL_CANDIDATES == [
        "logistic_balanced",
        "logistic",
        "random_forest",
        "hist_gradient_boosting",
    ]


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
