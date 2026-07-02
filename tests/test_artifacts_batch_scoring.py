import numpy as np
import pandas as pd

from batch import score_csv
from src.artifacts import ModelBundle, load_model_bundle, save_model_bundle
from src.config import REJECTED_STYLE_RISK_FEATURES
from src.scorer import score_records


class DummyModel:
    def predict_proba(self, frame):
        p = np.full(len(frame), 0.10)
        return np.column_stack([1 - p, p])


class IdentityCalibrator:
    def predict(self, raw):
        return raw


def test_saved_model_bundle_loads_successfully(tmp_path):
    path = tmp_path / "bundle.joblib"
    bundle = ModelBundle(DummyModel(), IdentityCalibrator(), ["loan_amnt"], "accepted")

    save_model_bundle(bundle, path)
    loaded = load_model_bundle(path)

    assert loaded.feature_columns == ["loan_amnt"]
    assert loaded.model_type == "accepted"


def test_batch_scoring_uses_saved_artifact_without_refitting(tmp_path):
    bundle_path = tmp_path / "bundle.joblib"
    input_csv = tmp_path / "input.csv"
    output_csv = tmp_path / "output.csv"
    bundle = ModelBundle(DummyModel(), IdentityCalibrator(), ["loan_amnt"], "accepted")
    save_model_bundle(bundle, bundle_path)
    pd.DataFrame(
        {
            "loan_amnt": [1000, 2000],
            "funded_amnt": [1000, 2000],
            "term_months": [12, 12],
            "installment": [100, 190],
        }
    ).to_csv(input_csv, index=False)

    score_csv(input_csv, output_csv, bundle_path=bundle_path, chunksize=1)
    out = pd.read_csv(output_csv)

    assert len(out) == 2
    assert {"p_default", "expected_profit", "expected_return", "decision", "lgd", "approval_rule"} <= set(
        out.columns
    )


def test_rejected_style_inputs_score_only_when_required_fields_present():
    bundle = ModelBundle(
        DummyModel(),
        IdentityCalibrator(),
        list(REJECTED_STYLE_RISK_FEATURES),
        "rejected_style",
    )
    row = {
        "Amount Requested": 1000,
        "Risk_Score": 700,
        "Debt-To-Income Ratio": "10%",
        "Zip Code": "123xx",
        "State": "NY",
        "Employment Length": "4 years",
    }

    scored = score_records([row], bundle)[0]

    assert scored["p_default"] == 0.10
    assert scored["decision"] == "review"
    assert "profit decision unavailable" in scored["reason"]


def test_scoring_uses_bundle_lgd_by_default():
    bundle = ModelBundle(
        DummyModel(),
        IdentityCalibrator(),
        ["loan_amnt"],
        "accepted",
        policy={"lgd": 0.5},
    )
    scored = score_records(
        [
            {
                "loan_amnt": 1000,
                "funded_amnt": 1000,
                "term_months": 12,
                "installment": 100,
            }
        ],
        bundle,
    )[0]

    assert scored["expected_profit"] == 130
    assert scored["lgd"] == 0.5
    assert scored["approval_rule"] == "expected_profit > 0"


def test_invalid_profit_inputs_return_review_per_row():
    bundle = ModelBundle(DummyModel(), IdentityCalibrator(), ["loan_amnt"], "accepted")
    scored = score_records(
        [
            {
                "loan_amnt": 1000,
                "funded_amnt": 0,
                "term_months": 12,
                "installment": 100,
            },
            {
                "loan_amnt": 1000,
                "funded_amnt": 1000,
                "term_months": 12,
                "installment": 100,
            },
        ],
        bundle,
    )

    assert scored[0]["decision"] == "review"
    assert "invalid profit inputs" in scored[0]["reason"]
    assert scored[1]["decision"] == "approve"
    assert scored[1]["expected_profit"] == 80
