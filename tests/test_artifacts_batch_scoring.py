import numpy as np
import pandas as pd
import pytest

from batch import score_csv
from src.artifacts import ModelBundle, load_model_bundle, save_model_bundle
from src.config import ACCEPTED_RISK_FEATURES
from src.scorer import score_records


class DummyModel:
    def predict_proba(self, frame):
        p = np.full(len(frame), 0.10)
        return np.column_stack([1 - p, p])


class IdentityCalibrator:
    def predict(self, raw):
        return raw


def accepted_bundle(metadata=None, feature_columns=None):
    feature_columns = list(feature_columns or ACCEPTED_RISK_FEATURES)
    return ModelBundle(
        DummyModel(),
        IdentityCalibrator(),
        feature_columns,
        "calibrated_logistic",
        metadata=metadata
        or {
            "bundle_schema_version": 1,
            "model_version": "accepted-default-v1",
            "selected_model_type": "calibrated_logistic",
            "selected_model_name": "logistic",
            "calibration_method": "isotonic",
            "risk_band_thresholds": {"low_max": 0.08, "medium_max": 0.18},
            "source_fingerprint": {"sha256": "abc", "size_bytes": 1},
            "split_manifest": {"test_ids": ["1"]},
            "split_summary": [],
            "target_definition": "resolved accepted/funded loan default target",
            "forbidden_feature_columns": ["loan_status"],
            "frontend_fields": ["loan_amnt", "int_rate", "annual_inc", "dti", "fico_range_low"],
            "feature_importance": [
                {"feature": "loan_amnt", "importance": 0.03},
                {"feature": "int_rate", "importance": 0.02},
                {"feature": "annual_inc", "importance": 0.01},
                {"feature": "dti", "importance": 0.009},
                {"feature": "fico_range_low", "importance": 0.008},
            ],
            "training_timestamp": "2026-07-06T00:00:00+00:00",
            "package_versions": {"python": "3.12"},
        },
        required_input_schema={"schema_version": 1, "required_fields": feature_columns},
    )


def accepted_row(**overrides):
    row = {
        "id": "1",
        "loan_amnt": 1000,
        "int_rate": 10,
        "annual_inc": 50000,
        "dti": 12,
        "fico_range_low": 700,
        "fico_range_high": 704,
        "delinq_2yrs": 0,
        "inq_last_6mths": 0,
        "open_acc": 8,
        "pub_rec": 0,
        "revol_bal": 1000,
        "revol_util": 20,
        "total_acc": 12,
        "mort_acc": 0,
        "acc_open_past_24mths": 1,
        "pub_rec_bankruptcies": 0,
        "grade": "A",
        "sub_grade": "A1",
        "emp_length": "4 years",
        "home_ownership": "RENT",
        "verification_status": "Not Verified",
        "purpose": "debt_consolidation",
        "addr_state": "NY",
        "application_type": "Individual",
        "initial_list_status": "w",
    }
    row.update(overrides)
    return row


def test_saved_model_bundle_loads_successfully(tmp_path):
    path = tmp_path / "bundle.joblib"
    bundle = accepted_bundle()

    save_model_bundle(bundle, path)
    loaded = load_model_bundle(path)

    assert loaded.feature_columns == list(ACCEPTED_RISK_FEATURES)
    assert loaded.required_input_schema["schema_version"] == 1
    assert loaded.metadata["model_version"] == "accepted-default-v1"


def test_batch_scoring_posts_csv_to_api_and_writes_response(monkeypatch, tmp_path):
    input_csv = tmp_path / "input.csv"
    output_csv = tmp_path / "output.csv"
    pd.DataFrame([accepted_row(), accepted_row(id="2", loan_amnt=2000)]).to_csv(input_csv, index=False)

    class DummyResponse:
        def __init__(self, content):
            self.content = content

        def raise_for_status(self):
            return None

    class DummyClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, files):
            assert url.endswith("/score-batch")
            return DummyResponse(b"id,p_default\n1,0.1\n2,0.1\n")

    monkeypatch.setattr("batch.httpx.Client", DummyClient)
    score_csv(input_csv, output_csv, api_url="http://localhost:8000")
    out = pd.read_csv(output_csv)

    assert len(out) == 2
    assert ["id", "p_default"] == list(out.columns)


def test_score_records_return_risk_only_schema():
    scored = score_records([accepted_row()], accepted_bundle())[0]

    assert scored["p_default"] == 0.10
    assert scored["p_non_default"] == 0.90
    assert scored["confidence"] == 0.8
    assert scored["risk_band"] == "medium"
    assert "scoring_note" in scored
    assert "decision" not in scored


def test_frontend_subset_bundle_can_score():
    subset = ["loan_amnt", "int_rate", "annual_inc", "dti", "fico_range_low"]
    scored = score_records([{key: accepted_row()[key] for key in subset}], accepted_bundle(feature_columns=subset))[0]

    assert scored["risk_band"] == "medium"


def test_missing_fields_fail_clearly():
    with pytest.raises(ValueError, match="missing required scoring fields"):
        score_records([{"loan_amnt": 1000}], accepted_bundle())
