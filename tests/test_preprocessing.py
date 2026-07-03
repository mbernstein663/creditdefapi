import pandas as pd
import pytest

from src.config import (
    ACCEPTED_RISK_FEATURES,
    DEFAULT_TARGET_HORIZON_MONTHS,
    PROFIT_INPUT_COLUMNS,
    PRODUCT_MODE_PRE_UNDERWRITING,
    REJECTED_STYLE_RISK_FEATURES,
    TARGET,
)
from src.preprocessing import (
    construct_target,
    ensure_no_forbidden_features,
    map_accepted_to_rejected_style,
    normalize_rejected_input,
    prepare_accepted_loans,
    split_chronological,
    split_count_report,
    split_manifest,
    feature_columns_for_product_mode,
)


def test_target_construction_maps_resolved_and_drops_unresolved():
    df = pd.DataFrame(
        {
            "loan_status": [
                "Fully Paid",
                "Does not meet the credit policy. Status:Fully Paid",
                "Charged Off",
                "Default",
                "Does not meet the credit policy. Status:Charged Off",
                "Current",
                "Late (31-120 days)",
                "",
            ]
        }
    )

    out = construct_target(df)

    assert out[TARGET].tolist() == [0, 0, 1, 1, 1]


def test_fixed_horizon_target_labels_only_conservative_observed_rows():
    df = pd.DataFrame(
        {
            "loan_status": [
                "Fully Paid",
                "Charged Off",
                "Fully Paid",
                "Charged Off",
                "Current",
            ],
            "issue_d": ["Jan-2018"] * 5,
            "last_pymnt_d": ["Feb-2021", "Mar-2019", "Dec-2018", "Jan-2022", "Jan-2019"],
        }
    )

    out, summary = construct_target(
        df,
        mode="default_within_horizon",
        target_config={"horizon_months": DEFAULT_TARGET_HORIZON_MONTHS},
        return_summary=True,
    )

    assert out[TARGET].tolist() == [0, 1]
    assert summary["included_rows"] == 2
    assert summary["excluded_rows"] == 3
    assert summary["mode"] == "default_within_horizon"


def test_unknown_status_is_not_silently_dropped():
    df = pd.DataFrame({"loan_status": ["Fully Paid", "Mystery Status"]})

    with pytest.raises(ValueError, match="unknown loan_status"):
        construct_target(df)


def test_rejected_input_is_not_labeled_by_assumption():
    rejected = pd.DataFrame({"Amount Requested": [1000], "Risk_Score": [700]})

    out = normalize_rejected_input(rejected)

    assert TARGET not in out.columns


def test_leakage_and_profit_inputs_are_not_risk_features():
    assert not set(PROFIT_INPUT_COLUMNS) & set(ACCEPTED_RISK_FEATURES)
    assert "policy_code" not in ACCEPTED_RISK_FEATURES
    assert "policy_code" not in REJECTED_STYLE_RISK_FEATURES
    ensure_no_forbidden_features(ACCEPTED_RISK_FEATURES)
    ensure_no_forbidden_features(REJECTED_STYLE_RISK_FEATURES)

    with pytest.raises(ValueError):
        ensure_no_forbidden_features(["loan_status"])


def test_pre_underwriting_mode_excludes_pricing_fields():
    features = feature_columns_for_product_mode(PRODUCT_MODE_PRE_UNDERWRITING)

    assert "grade" not in features
    assert "sub_grade" not in features
    assert "int_rate" not in features
    assert "initial_list_status" not in features
    assert "loan_amnt" in features


def test_accepted_to_rejected_feature_map_is_explicit():
    accepted = pd.DataFrame(
        {
            "loan_amnt": [1200],
            "issue_d": ["Jan-2018"],
            "title": ["Debt consolidation"],
            "fico_range_low": [700],
            "fico_range_high": [704],
            "dti": [10.5],
            "zip_code": ["123xx"],
            "addr_state": ["NY"],
            "emp_length": ["4 years"],
            "policy_code": [1],
            TARGET: [0],
        }
    )

    out = map_accepted_to_rejected_style(accepted)

    assert out.loc[0, "amount_requested"] == 1200
    assert out.loc[0, "risk_score"] == 702
    assert out.loc[0, "state"] == "NY"
    assert "policy_code" in out.columns
    assert "policy_code" not in REJECTED_STYLE_RISK_FEATURES


def test_splits_are_disjoint_and_report_counts_by_year():
    df = pd.DataFrame(
        {
            "loan_status": ["Fully Paid", "Charged Off"] * 10,
            "issue_d": pd.date_range("2018-01-01", periods=20, freq="MS").strftime("%b-%Y"),
        },
        index=range(100, 120),
    )
    prepared = prepare_accepted_loans(df)

    splits = split_chronological(prepared)
    seen = set()
    for frame in splits.values():
        assert seen.isdisjoint(frame.index)
        seen |= set(frame.index)

    report = split_count_report(splits)
    assert report["rows"].sum() == len(prepared)
    assert {"split", "issue_year", "rows", "defaults", "non_defaults"} <= set(report.columns)


def test_split_manifest_saves_test_ids_and_months_do_not_straddle_splits():
    df = pd.DataFrame(
        {
            "id": [str(i) for i in range(12)],
            "loan_status": ["Fully Paid", "Charged Off"] * 6,
            "issue_d": ["Jan-2018"] * 3 + ["Feb-2018"] * 3 + ["Mar-2018"] * 3 + ["Apr-2018"] * 3,
        }
    )
    splits = split_chronological(prepare_accepted_loans(df))
    month_to_split = {}
    for split, frame in splits.items():
        for month in frame["issue_d"].unique():
            assert month not in month_to_split
            month_to_split[month] = split

    manifest = split_manifest(splits)
    assert "test_ids" in manifest
    assert set(manifest["test_ids"]) == set(splits["test"]["id"].astype(str))


def test_split_chronological_fails_on_missing_split_dates():
    df = pd.DataFrame({"loan_status": ["Fully Paid"], "issue_d": ["not-a-date"]})
    prepared = prepare_accepted_loans(df)

    with pytest.raises(ValueError, match="issue_dt has 1 missing values"):
        split_chronological(prepared)
