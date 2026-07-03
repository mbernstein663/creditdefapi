import numpy as np
import pandas as pd

from batch import score_csv
from src.artifacts import ModelBundle, load_model_bundle, save_model_bundle
from src.config import ACCEPTED_RISK_FEATURES, REJECTED_STYLE_RISK_FEATURES
from src.scorer import score_records


class DummyModel:
    def predict_proba(self, frame):
        p = np.full(len(frame), 0.10)
        return np.column_stack([1 - p, p])


class IdentityCalibrator:
    def predict(self, raw):
        return raw


def accepted_bundle(policy=None):
    return ModelBundle(
        DummyModel(),
        IdentityCalibrator(),
        list(ACCEPTED_RISK_FEATURES),
        "accepted",
        policy=policy or {"lgd": 1.0, "required_return": None},
        required_input_schema={"risk_features": list(ACCEPTED_RISK_FEATURES)},
    )


def accepted_row(**overrides):
    row = {
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
        "funded_amnt": 1000,
        "term_months": 12,
        "installment": 100,
    }
    row.update(overrides)
    return row


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
    bundle = accepted_bundle()
    save_model_bundle(bundle, bundle_path)
    pd.DataFrame([accepted_row(), accepted_row(loan_amnt=2000, funded_amnt=2000, installment=190)]).to_csv(
        input_csv, index=False
    )

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
    bundle = accepted_bundle(policy={"lgd": 0.5, "required_return": None})
    scored = score_records(
        [accepted_row()],
        bundle,
    )[0]

    assert scored["expected_profit"] == 130
    assert scored["lgd"] == 0.5
    assert scored["approval_rule"] == "expected_profit > 0"


def test_scoring_uses_bundle_good_profit_haircut():
    bundle = accepted_bundle(policy={"lgd": 1.0, "required_return": None, "good_profit_haircut": 0.5})
    scored = score_records([accepted_row()], bundle)[0]

    assert scored["expected_profit"] == -10
    assert scored["good_profit_haircut"] == 0.5
    assert scored["decision"] == "deny"


def test_invalid_profit_inputs_return_review_per_row():
    bundle = accepted_bundle()
    scored = score_records(
        [
            accepted_row(funded_amnt=0),
            accepted_row(),
        ],
        bundle,
    )

    assert scored[0]["decision"] == "review"
    assert "invalid profit inputs" in scored[0]["reason"]
    assert scored[1]["decision"] == "approve"
    assert scored[1]["expected_profit"] == 80


def test_required_return_policy_uses_expected_return_rule():
    bundle = accepted_bundle(policy={"lgd": 1.0, "required_return": 0.08})

    scored = score_records([accepted_row()], bundle)[0]

    assert scored["decision"] == "deny"
    assert scored["approval_rule"] == "expected_return > required_return"


def test_rejected_style_profit_inputs_remain_review_only():
    bundle = ModelBundle(
        DummyModel(),
        IdentityCalibrator(),
        list(REJECTED_STYLE_RISK_FEATURES),
        "rejected_style",
        policy={"lgd": 1.0, "required_return": None},
        required_input_schema={"risk_features": list(REJECTED_STYLE_RISK_FEATURES)},
    )

    scored = score_records(
        [
            {
                "amount_requested": 1000,
                "risk_score": 700,
                "dti": 10,
                "zip_code": "123xx",
                "state": "NY",
                "employment_length": "4 years",
                "funded_amnt": 1000,
                "term_months": 12,
                "installment": 100,
            }
        ],
        bundle,
    )[0]

    assert scored["expected_profit"] == 80
    assert scored["decision"] == "review"
    assert "scenario math" in scored["reason"]
