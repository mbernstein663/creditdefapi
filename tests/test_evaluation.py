from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.artifacts import ModelBundle, save_model_bundle
from src.config import ACCEPTED_RISK_FEATURES
from src.evaluation_report import run_evaluation_suite
from tests.test_artifacts_batch_scoring import DummyModel, IdentityCalibrator


class ConstantProfitModel:
    def predict(self, frame):
        return np.full(len(frame), 125.0)


def _accepted_frame(single_class: bool = False, drop_realized_profit: bool = False) -> pd.DataFrame:
    n = 20
    defaults = [0 if single_class else int(i % 3 == 0) for i in range(n)]
    issue_dates = pd.date_range("2018-01-01", periods=n, freq="MS")
    frame = pd.DataFrame(
        {
            "id": [str(i + 1) for i in range(n)],
            "loan_amnt": [1000 + 10 * i for i in range(n)],
            "int_rate": [10 + 0.1 * i for i in range(n)],
            "annual_inc": [50000 + 100 * i for i in range(n)],
            "dti": [12 + 0.1 * i for i in range(n)],
            "fico_range_low": [700 + (i % 3) for i in range(n)],
            "fico_range_high": [704 + (i % 3) for i in range(n)],
            "delinq_2yrs": [0 for _ in range(n)],
            "inq_last_6mths": [0 for _ in range(n)],
            "open_acc": [8 for _ in range(n)],
            "pub_rec": [0 for _ in range(n)],
            "revol_bal": [1000 + 5 * i for i in range(n)],
            "revol_util": [20 + 0.5 * i for i in range(n)],
            "total_acc": [12 for _ in range(n)],
            "mort_acc": [0 for _ in range(n)],
            "acc_open_past_24mths": [1 for _ in range(n)],
            "pub_rec_bankruptcies": [0 for _ in range(n)],
            "grade": ["A" if i % 2 == 0 else "B" for i in range(n)],
            "sub_grade": ["A1" if i % 2 == 0 else "B1" for i in range(n)],
            "emp_length": [f"{(i % 5) + 1} year" for i in range(n)],
            "home_ownership": ["RENT", "OWN", "MORTGAGE", "RENT", "OWN"] * 4,
            "verification_status": ["Verified", "Not Verified"] * 10,
            "purpose": ["debt_consolidation"] * n,
            "addr_state": ["NY", "CA", "NY", "TX", "NJ"] * 4,
            "application_type": ["Individual"] * n,
            "initial_list_status": ["w"] * n,
            "title": ["Debt consolidation"] * n,
            "term": [" 36 months"] * n,
            "funded_amnt": [1000 + 10 * i for i in range(n)],
            "installment": [100 + i for i in range(n)],
            "loan_status": ["Fully Paid" if defaults[i] == 0 else "Charged Off" for i in range(n)],
            "issue_d": [d.strftime("%b-%Y") for d in issue_dates],
            "last_pymnt_d": [(d + pd.offsets.MonthEnd(6)).strftime("%b-%Y") for d in issue_dates],
            "total_pymnt": [1200 + 5 * i if defaults[i] == 0 else 300 + 5 * i for i in range(n)],
            "default": defaults,
            "issue_dt": issue_dates,
            "issue_year": issue_dates.year.tolist(),
            "issue_quarter": issue_dates.to_period("Q").astype(str).tolist(),
            "issue_month": issue_dates.to_period("M").astype(str).tolist(),
        }
    )
    if drop_realized_profit:
        frame = frame.drop(columns=["total_pymnt"])
    return frame


def _save_bundle(tmp_path: Path, name: str, bundle: ModelBundle) -> Path:
    path = tmp_path / name
    save_model_bundle(bundle, path)
    return path


def _accepted_bundle() -> ModelBundle:
    return ModelBundle(
        DummyModel(),
        IdentityCalibrator(),
        list(ACCEPTED_RISK_FEATURES),
        "accepted",
        policy={"lgd": 1.0, "required_return": None, "approval_rule": "expected_profit > 0"},
        metadata={"product_mode": "post_pricing_investment", "calibration_method": "isotonic"},
        required_input_schema={"risk_features": list(ACCEPTED_RISK_FEATURES)},
    )


def _rejected_bundle() -> ModelBundle:
    return ModelBundle(
        DummyModel(),
        IdentityCalibrator(),
        ["amount_requested", "risk_score", "dti", "zip_code", "state", "employment_length"],
        "rejected_style",
        policy={"approval_rule": "review only"},
        metadata={"product_mode": "pre_underwriting_applicant", "calibration_method": "isotonic"},
        required_input_schema={"risk_features": ["amount_requested", "risk_score", "dti", "zip_code", "state", "employment_length"]},
    )


def _profit_bundle() -> ModelBundle:
    return ModelBundle(
        ConstantProfitModel(),
        None,
        list(ACCEPTED_RISK_FEATURES),
        "direct_profit",
        policy={"type": "threshold", "threshold": 100.0},
        metadata={"selection_rule": "threshold"},
        required_input_schema={"risk_features": list(ACCEPTED_RISK_FEATURES)},
    )


def test_run_evaluation_suite_creates_csv_json_and_plots(tmp_path):
    csv_path = tmp_path / "accepted.csv"
    _accepted_frame().to_csv(csv_path, index=False)
    accepted_path = _save_bundle(tmp_path, "accepted.joblib", _accepted_bundle())
    rejected_path = _save_bundle(tmp_path, "rejected.joblib", _rejected_bundle())
    profit_path = _save_bundle(tmp_path, "profit.joblib", _profit_bundle())

    outputs = run_evaluation_suite(
        csv_path=csv_path,
        output_dir=tmp_path / "reports",
        bundle_paths={"accepted": accepted_path, "rejected_style": rejected_path, "direct_profit": profit_path},
        n_bootstrap=8,
        random_state=7,
    )

    expected = [
        "model_comparison",
        "calibration_deciles",
        "policy_threshold_curve",
        "top_percentile_curve",
        "decile_lift",
        "cohort_backtest",
        "feature_importance",
        "evaluation_summary",
    ]
    for key in expected:
        assert outputs[key].exists()

    figures_dir = tmp_path / "reports" / "figures"
    for name in [
        "calibration_curve.png",
        "profit_threshold_curve.png",
        "cumulative_profit_curve.png",
        "profit_by_expected_return_decile.png",
        "default_rate_by_risk_decile.png",
    ]:
        assert (figures_dir / name).exists()

    summary = json.loads(outputs["evaluation_summary"].read_text(encoding="utf-8"))
    assert summary["best_probability_model"] == "accepted"
    assert summary["generated_files"]["model_comparison"].endswith("model_comparison.csv")
    assert summary["warnings"]

    comparison = pd.read_csv(outputs["model_comparison"])
    assert {"model_name", "calibration_method", "expected_profit", "realized_profit", "selected_threshold"} <= set(
        comparison.columns
    )

    deciles = pd.read_csv(outputs["calibration_deciles"])
    assert {"decile", "count", "mean_predicted_default", "observed_default_rate", "default_count"} <= set(deciles.columns)


def test_missing_realized_profit_does_not_crash(tmp_path):
    csv_path = tmp_path / "accepted.csv"
    _accepted_frame(drop_realized_profit=True).to_csv(csv_path, index=False)
    accepted_path = _save_bundle(tmp_path, "accepted.joblib", _accepted_bundle())

    outputs = run_evaluation_suite(
        csv_path=csv_path,
        output_dir=tmp_path / "reports",
        bundle_paths={"accepted": accepted_path},
        n_bootstrap=4,
        random_state=3,
    )

    threshold = pd.read_csv(outputs["policy_threshold_curve"])
    assert "realized_profit" in threshold.columns

    summary = json.loads(outputs["evaluation_summary"].read_text(encoding="utf-8"))
    assert any("realized profit is unavailable" in warning for warning in summary["warnings"])


def test_single_class_cohort_auc_is_handled(tmp_path):
    csv_path = tmp_path / "accepted.csv"
    _accepted_frame(single_class=True).to_csv(csv_path, index=False)
    accepted_path = _save_bundle(tmp_path, "accepted.joblib", _accepted_bundle())

    outputs = run_evaluation_suite(
        csv_path=csv_path,
        output_dir=tmp_path / "reports",
        bundle_paths={"accepted": accepted_path},
        n_bootstrap=4,
        random_state=5,
    )

    cohort = pd.read_csv(outputs["cohort_backtest"])
    assert "roc_auc" in cohort.columns
    assert cohort["roc_auc"].isna().all()
