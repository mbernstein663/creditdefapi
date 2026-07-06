import json

import pandas as pd

from src.evaluation import generate_evaluation_reports
from tests.test_artifacts_batch_scoring import accepted_bundle


def _frame():
    n = 20
    issue_dates = pd.date_range("2018-01-01", periods=n, freq="MS")
    return pd.DataFrame(
        {
            "id": [str(i + 1) for i in range(n)],
            "loan_amnt": [1000 + i for i in range(n)],
            "int_rate": [10 + 0.1 * i for i in range(n)],
            "annual_inc": [50000 + 100 * i for i in range(n)],
            "dti": [12 + 0.1 * i for i in range(n)],
            "fico_range_low": [700 + (i % 4) for i in range(n)],
            "fico_range_high": [704 + (i % 4) for i in range(n)],
            "delinq_2yrs": [0] * n,
            "inq_last_6mths": [0] * n,
            "open_acc": [8] * n,
            "pub_rec": [0] * n,
            "revol_bal": [1000 + 5 * i for i in range(n)],
            "revol_util": [20 + 0.5 * i for i in range(n)],
            "total_acc": [12] * n,
            "mort_acc": [0] * n,
            "acc_open_past_24mths": [1] * n,
            "pub_rec_bankruptcies": [0] * n,
            "grade": ["A" if i % 2 == 0 else "B" for i in range(n)],
            "sub_grade": ["A1" if i % 2 == 0 else "B1" for i in range(n)],
            "emp_length": ["4 years"] * n,
            "home_ownership": ["RENT"] * n,
            "verification_status": ["Verified"] * n,
            "purpose": ["debt_consolidation"] * n,
            "addr_state": ["NY"] * n,
            "application_type": ["Individual"] * n,
            "initial_list_status": ["w"] * n,
            "term": [" 36 months"] * n,
            "issue_year": issue_dates.year.tolist(),
            "default": [int(i % 3 == 0) for i in range(n)],
        }
    )


def test_generate_evaluation_reports_creates_required_outputs(tmp_path):
    bundle = accepted_bundle()
    frame = _frame()
    p_default = pd.Series([0.05 + 0.02 * (i % 5) for i in range(len(frame))])

    outputs = generate_evaluation_reports(bundle, frame, p_default, tmp_path / "reports", "validation")

    for key in [
        "metrics_summary",
        "calibration_deciles",
        "risk_decile_lift",
        "calibration_by_issue_year",
        "roc_curve",
        "pr_curve",
        "reliability_plot",
        "roc_curve_plot",
        "pr_curve_plot",
        "model_card",
        "calibration_by_grade",
    ]:
        assert outputs[key].exists()

    summary = json.loads(outputs["metrics_summary"].read_text(encoding="utf-8"))
    assert summary["selected_model"] == "logistic"
    assert summary["calibration_method"] == "isotonic"

    deciles = pd.read_csv(outputs["calibration_deciles"])
    assert {
        "decile",
        "count",
        "mean_predicted_default",
        "observed_default_rate",
        "absolute_calibration_gap",
    } <= set(deciles.columns)

    lift = pd.read_csv(outputs["risk_decile_lift"])
    assert {
        "predicted_risk_decile",
        "count",
        "observed_default_rate",
        "lift_versus_portfolio_default_rate",
        "cumulative_share_of_defaults_captured",
    } <= set(lift.columns)
