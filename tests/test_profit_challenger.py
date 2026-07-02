import pandas as pd
import pytest

from src.config import ACCEPTED_RISK_FEATURES, PROFIT_TARGET
from src.profit_challenger import (
    prepare_profit_frame,
    profit_policy_metrics,
    search_profit_policy,
    validate_profit_features,
)


def test_direct_profit_features_reject_leakage_columns():
    validate_profit_features(ACCEPTED_RISK_FEATURES)

    with pytest.raises(ValueError, match="forbidden model features"):
        validate_profit_features(["loan_amnt", "total_pymnt"])


def test_rejected_rows_are_not_given_realized_profit_labels():
    rejected_style = pd.DataFrame({"Amount Requested": [1000], "Risk_Score": [700]})

    with pytest.raises(ValueError, match="accepted loan data must include loan_status"):
        prepare_profit_frame(rejected_style)


def test_prepare_profit_frame_creates_target_only_for_resolved_accepted_loans():
    df = pd.DataFrame(
        {
            "loan_status": ["Fully Paid", "Charged Off", "Current"],
            "issue_d": ["Jan-2018", "Feb-2018", "Mar-2018"],
            "total_pymnt": [1200, 300, 100],
            "funded_amnt": [1000, 1000, 1000],
        }
    )

    out = prepare_profit_frame(df)

    assert out[PROFIT_TARGET].tolist() == [200, -700]


def test_profit_policy_metrics_and_search_use_realized_profit():
    df = pd.DataFrame(
        {
            PROFIT_TARGET: [100, -50, 300, -200],
            "funded_amnt": [1000, 1000, 1000, 1000],
            "default": [0, 1, 0, 1],
        }
    )
    predicted = [90, 80, 10, -20]

    metrics = profit_policy_metrics(df, predicted, {"type": "threshold", "threshold": 0})
    policy, candidates = search_profit_policy(df, predicted)

    assert metrics["approval_count"] == 3
    assert metrics["total_realized_profit"] == 350
    assert metrics["profit_per_dollar_funded"] == 350 / 3000
    assert metrics["realized_default_rate"] == 1 / 3
    assert "decile_lift" in metrics
    assert policy in [candidate["policy"] for candidate in candidates]
