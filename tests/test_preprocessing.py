import numpy as np
import pandas as pd
import pytest

from src.config import ACCEPTED_RISK_FEATURES, BAD_STATUSES, FORBIDDEN_FEATURE_COLUMNS, GOOD_STATUSES, TARGET
from src.preprocessing import (
    construct_target,
    ensure_no_forbidden_features,
    feature_columns,
    normalize_rejected_input,
    prepare_accepted_loans,
    split_chronological,
    split_manifest,
)
from src.preprocessing import preprocess_accepted_loans


def test_target_construction_maps_resolved_and_drops_unresolved():
    df = pd.DataFrame(
        {
            "loan_status": [
                sorted(GOOD_STATUSES)[1],
                sorted(GOOD_STATUSES)[0],
                sorted(BAD_STATUSES)[0],
                sorted(BAD_STATUSES)[1],
                sorted(BAD_STATUSES)[2],
                "Current",
                "Late (31-120 days)",
                "",
            ]
        }
    )

    out = construct_target(df)

    assert out[TARGET].tolist() == [0, 0, 1, 1, 1]


def test_unknown_status_is_not_silently_dropped():
    with pytest.raises(ValueError, match="unknown loan_status"):
        construct_target(pd.DataFrame({"loan_status": ["Fully Paid", "Mystery Status"]}))


def test_rejected_input_is_not_labeled_by_assumption():
    out = normalize_rejected_input(pd.DataFrame({"Amount Requested": [1000], "Risk_Score": [700]}))

    assert TARGET not in out.columns


def test_forbidden_columns_are_excluded_from_risk_features():
    assert feature_columns() == list(ACCEPTED_RISK_FEATURES)
    assert not set(feature_columns()) & set(FORBIDDEN_FEATURE_COLUMNS)
    ensure_no_forbidden_features(feature_columns())

    with pytest.raises(ValueError, match="forbidden model features"):
        ensure_no_forbidden_features(["loan_status"])


def test_prepare_accepted_loans_applies_safe_numeric_transforms():
    df = pd.DataFrame(
        {
            "loan_status": ["Fully Paid"],
            "loan_amnt": [9999],
            "annual_inc": [120000],
            "dti": [150],
            "revol_bal": [5000],
            "total_acc": [24],
            "int_rate": ["10%"],
            "revol_util": ["55%"],
            "fico_range_low": [700],
            "fico_range_high": [704],
            "delinq_2yrs": [0],
            "inq_last_6mths": [0],
            "open_acc": [8],
            "pub_rec": [0],
            "mort_acc": [0],
            "acc_open_past_24mths": [1],
            "pub_rec_bankruptcies": [0],
            "grade": ["A"],
            "sub_grade": ["A1"],
            "emp_length": ["4 years"],
            "home_ownership": ["RENT"],
            "verification_status": ["Verified"],
            "purpose": ["debt_consolidation"],
            "addr_state": ["NY"],
            "application_type": ["Individual"],
            "initial_list_status": ["w"],
            "issue_d": ["Jan-2018"],
        }
    )

    prepared = prepare_accepted_loans(df)

    assert prepared.loc[0, "loan_amnt"] == pytest.approx(np.log1p(9999))
    assert prepared.loc[0, "annual_inc"] == pytest.approx(np.log1p(120000))
    assert prepared.loc[0, "revol_bal"] == pytest.approx(np.log1p(5000))
    assert prepared.loc[0, "total_acc"] == pytest.approx(np.log1p(24))
    assert prepared.loc[0, "dti"] == 100


def test_splits_are_disjoint_and_chronological():
    df = pd.DataFrame(
        {
            "id": [str(i) for i in range(20)],
            "loan_status": ["Fully Paid", "Charged Off"] * 10,
            "issue_d": pd.date_range("2018-01-01", periods=20, freq="MS").strftime("%b-%Y"),
        }
    )
    prepared = prepare_accepted_loans(df)
    splits = split_chronological(prepared)

    seen = set()
    last_max = None
    for frame in splits.values():
        assert seen.isdisjoint(frame.index)
        seen |= set(frame.index)
        current_min = frame["issue_dt"].min()
        if last_max is not None:
            assert current_min >= last_max
        last_max = frame["issue_dt"].max()

    manifest = split_manifest(splits)
    assert set(manifest["test_ids"]) == set(splits["test"]["id"].astype(str))


def test_split_requires_at_least_four_distinct_issue_dates():
    df = pd.DataFrame(
        {
            "id": [str(i) for i in range(6)],
            "loan_status": ["Fully Paid", "Charged Off"] * 3,
            "issue_d": ["Jan-2018", "Jan-2018", "Feb-2018", "Feb-2018", "Mar-2018", "Mar-2018"],
        }
    )

    with pytest.raises(ValueError, match="at least 4 distinct issue dates"):
        split_chronological(prepare_accepted_loans(df))


def test_preprocessing_stage_returns_splits_and_manifest(tmp_path):
    csv_path = tmp_path / "accepted.csv"
    df = pd.DataFrame(
        {
            "id": ["1", "2", "3", "4"],
            "loan_status": ["Fully Paid", "Charged Off", "Fully Paid", "Charged Off"],
            "issue_d": ["Jan-2018", "Feb-2018", "Mar-2018", "Apr-2018"],
            "loan_amnt": [1000, 1000, 1000, 1000],
            "int_rate": [10, 10, 10, 10],
            "annual_inc": [50000, 50000, 50000, 50000],
            "dti": [10, 10, 10, 10],
            "fico_range_low": [700, 700, 700, 700],
            "fico_range_high": [704, 704, 704, 704],
            "delinq_2yrs": [0, 0, 0, 0],
            "inq_last_6mths": [0, 0, 0, 0],
            "open_acc": [8, 8, 8, 8],
            "pub_rec": [0, 0, 0, 0],
            "revol_bal": [1000, 1000, 1000, 1000],
            "revol_util": [20, 20, 20, 20],
            "total_acc": [12, 12, 12, 12],
            "mort_acc": [0, 0, 0, 0],
            "acc_open_past_24mths": [1, 1, 1, 1],
            "pub_rec_bankruptcies": [0, 0, 0, 0],
            "grade": ["A", "A", "A", "A"],
            "sub_grade": ["A1", "A1", "A1", "A1"],
            "emp_length": ["4 years", "4 years", "4 years", "4 years"],
            "home_ownership": ["RENT", "RENT", "RENT", "RENT"],
            "verification_status": ["Verified", "Verified", "Verified", "Verified"],
            "purpose": ["debt_consolidation", "debt_consolidation", "debt_consolidation", "debt_consolidation"],
            "addr_state": ["NY", "NY", "NY", "NY"],
            "application_type": ["Individual", "Individual", "Individual", "Individual"],
            "initial_list_status": ["w", "w", "w", "w"],
            "term": ["36 months", "36 months", "36 months", "36 months"],
        }
    )
    df.to_csv(csv_path, index=False)

    result = preprocess_accepted_loans(csv_path)

    assert set(result.splits) == {"train", "calibration", "validation", "test"}
    assert result.manifest["row_counts"]["test"] > 0
    assert result.target_summary["included_rows"] == 4


def test_sampled_preprocessing_reads_enough_months_for_chronological_split(tmp_path):
    csv_path = tmp_path / "accepted.csv"
    rows = []
    for month, count in [("Jan-2018", 3), ("Feb-2018", 3), ("Mar-2018", 3), ("Apr-2018", 3)]:
        for i in range(count):
            rows.append(
                {
                    "id": f"{month}-{i}",
                    "loan_status": "Fully Paid" if i % 2 == 0 else "Charged Off",
                    "issue_d": month,
                    "loan_amnt": 1000,
                    "int_rate": 10,
                    "annual_inc": 50000,
                    "dti": 10,
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
                    "verification_status": "Verified",
                    "purpose": "debt_consolidation",
                    "addr_state": "NY",
                    "application_type": "Individual",
                    "initial_list_status": "w",
                    "term": "36 months",
                }
            )
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    result = preprocess_accepted_loans(csv_path, sample=2)

    assert result.manifest["row_counts"]["train"] > 0
    assert result.accepted["issue_dt"].nunique() >= 4
