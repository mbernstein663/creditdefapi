import json

import pandas as pd

from src.artifacts import ModelBundle
from src.config import ACCEPTED_RISK_FEATURES, TARGET
from src.reporting import bootstrap_intervals, cohort_backtest, generate_report_suite
from tests.test_artifacts_batch_scoring import DummyModel, IdentityCalibrator


def _report_bundle():
    return ModelBundle(
        DummyModel(),
        IdentityCalibrator(),
        list(ACCEPTED_RISK_FEATURES),
        "accepted",
        policy={
            "lgd": 1.0,
            "required_return": None,
            "approval_rule": "expected_profit > 0",
            "use_npv": False,
            "annual_discount_rate": 0.08,
            "servicing_cost_rate": 0.0,
            "recovery_rate": 0.25,
        },
        metadata={
            "product_mode": "post_pricing_investment",
            "target_mode": "resolved_default",
            "calibration_method": "isotonic",
            "source_fingerprint": {"sha256": "abc", "size_bytes": 1},
            "target_summary": {"included_rows": 4, "excluded_rows": 0, "total_rows": 4},
            "split_summary": [{"split": "validation", "rows": 4, "default_rate": 0.5, "date_min": None, "date_max": None}],
            "forbidden_feature_columns": ["loan_status"],
        },
        required_input_schema={"risk_features": list(ACCEPTED_RISK_FEATURES)},
    )


def _report_frame(single_class=False):
    statuses = ["Fully Paid", "Charged Off", "Fully Paid", "Charged Off"]
    if single_class:
        statuses = ["Fully Paid"] * 4
    return pd.DataFrame(
        {
            TARGET: [0, 1, 0, 1] if not single_class else [0, 0, 0, 0],
            "loan_status": statuses,
            "issue_dt": pd.to_datetime(["2018-01-01", "2018-04-01", "2018-07-01", "2018-10-01"]),
            "issue_year": [2018, 2018, 2018, 2018],
            "issue_quarter": ["2018Q1", "2018Q2", "2018Q3", "2018Q4"],
            "issue_month": ["2018-01", "2018-04", "2018-07", "2018-10"],
            "funded_amnt": [1000, 1000, 1000, 1000],
            "term_months": [12, 12, 12, 12],
            "installment": [100, 100, 100, 100],
            "total_pymnt": [1200, 300, 1150, 250],
            "addr_state": ["NY", "CA", "NY", "TX"],
            "home_ownership": ["RENT", "OWN", "RENT", "MORTGAGE"],
            "emp_length": ["1 year", "2 years", "3 years", "4 years"],
            "verification_status": ["Verified", "Not Verified", "Verified", "Not Verified"],
            "purpose": ["debt_consolidation"] * 4,
        }
    )


def test_generate_report_suite_writes_required_files_and_fields(tmp_path):
    bundle = _report_bundle()
    frame = _report_frame()
    paths = generate_report_suite(bundle, frame, [0.1, 0.8, 0.2, 0.7], output_dir=tmp_path, stage_summary={"row_count": 4, "split_summary": bundle.metadata["split_summary"]})

    for key in [
        "model_card",
        "evaluation_summary",
        "calibration_table",
        "policy_summary",
        "sensitivity_summary",
        "cohort_backtest",
        "bootstrap_intervals",
        "proxy_risk_diagnostics",
        "fairness_caveat",
    ]:
        assert paths[key].exists()

    summary = json.loads(paths["evaluation_summary"].read_text(encoding="utf-8"))
    assert summary["target_mode"] == "resolved_default"
    assert summary["product_mode"] == "post_pricing_investment"
    assert summary["policy"]["lgd"] == 1.0

    calibration = pd.read_csv(paths["calibration_table"])
    assert {"decile", "count", "mean_predicted_default", "observed_default_rate"} <= set(calibration.columns)

    bootstrap = pd.read_csv(paths["bootstrap_intervals"])
    assert {"metric", "estimate", "lower", "upper"} <= set(bootstrap.columns)


def test_bootstrap_intervals_has_expected_shape_and_columns():
    bundle = _report_bundle()
    frame = _report_frame()
    out = bootstrap_intervals(frame, [0.1, 0.8, 0.2, 0.7], bundle.policy, n_bootstrap=10, random_state=7)

    assert len(out) == 9
    assert {"metric", "estimate", "lower", "upper", "bootstrap_samples", "random_state"} <= set(out.columns)


def test_cohort_backtest_handles_single_class_cohorts():
    bundle = _report_bundle()
    frame = _report_frame(single_class=True)
    out = cohort_backtest(frame, [0.1, 0.1, 0.1, 0.1], bundle.policy, include_month=False)

    assert len(out) == 5
    assert out["auc"].isna().all()
    assert {"cohort_type", "cohort_value", "rows", "approval_rate", "expected_profit"} <= set(out.columns)
