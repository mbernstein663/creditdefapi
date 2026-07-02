import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACCEPTED_CSV = ROOT / "accepted_2007_to_2018Q4.csv"
REJECTED_CSV = ROOT / "rejected_2007_to_2018Q4.csv"
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports"


def _path_from_env(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value) if value else default


DEFAULT_ACCEPTED_BUNDLE = _path_from_env("ACCEPTED_MODEL_BUNDLE", ARTIFACT_DIR / "accepted_model.joblib")
DEFAULT_REJECTED_STYLE_BUNDLE = _path_from_env(
    "REJECTED_STYLE_MODEL_BUNDLE",
    ARTIFACT_DIR / "rejected_style_model.joblib",
)

TARGET = "default"
DEFAULT_LGD = 1.0

BAD_STATUSES = {
    "Charged Off",
    "Default",
    "Does not meet the credit policy. Status:Charged Off",
}
GOOD_STATUSES = {
    "Fully Paid",
    "Does not meet the credit policy. Status:Fully Paid",
}
UNRESOLVED_STATUSES = {
    "Current",
    "In Grace Period",
    "Late (16-30 days)",
    "Late (31-120 days)",
    "Issued",
    "",
}

PROFIT_INPUT_COLUMNS = ["funded_amnt", "term_months", "installment"]

ACCEPTED_NUMERIC_RISK_FEATURES = [
    "loan_amnt",
    "int_rate",
    "annual_inc",
    "dti",
    "fico_range_low",
    "fico_range_high",
    "delinq_2yrs",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "mort_acc",
    "acc_open_past_24mths",
    "pub_rec_bankruptcies",
]
ACCEPTED_CATEGORICAL_RISK_FEATURES = [
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "verification_status",
    "purpose",
    "addr_state",
    "application_type",
    "initial_list_status",
]
ACCEPTED_RISK_FEATURES = ACCEPTED_NUMERIC_RISK_FEATURES + ACCEPTED_CATEGORICAL_RISK_FEATURES

REJECTED_STYLE_NUMERIC_RISK_FEATURES = ["amount_requested", "risk_score", "dti"]
REJECTED_STYLE_CATEGORICAL_RISK_FEATURES = ["zip_code", "state", "employment_length"]
REJECTED_STYLE_RISK_FEATURES = (
    REJECTED_STYLE_NUMERIC_RISK_FEATURES + REJECTED_STYLE_CATEGORICAL_RISK_FEATURES
)

# Explicit map from accepted-loan columns to rejected-application-style fields.
# policy_code is mapped for schema/reporting, but is intentionally not a model feature.
ACCEPTED_TO_REJECTED_FEATURE_MAP = {
    "loan_amnt": "amount_requested",
    "issue_d": "application_date",
    "title": "loan_title",
    "fico_range_low": "risk_score_low",
    "fico_range_high": "risk_score_high",
    "dti": "dti",
    "zip_code": "zip_code",
    "addr_state": "state",
    "emp_length": "employment_length",
    "policy_code": "policy_code",
}

REJECTED_RAW_ALIASES = {
    "Amount Requested": "amount_requested",
    "Application Date": "application_date",
    "Loan Title": "loan_title",
    "Risk_Score": "risk_score",
    "Debt-To-Income Ratio": "dti",
    "Zip Code": "zip_code",
    "State": "state",
    "Employment Length": "employment_length",
    "Policy Code": "policy_code",
}

FORBIDDEN_FEATURE_COLUMNS = {
    "loan_status",
    "total_pymnt",
    "total_pymnt_inv",
    "recoveries",
    "collection_recovery_fee",
    "last_pymnt_d",
    "last_pymnt_amnt",
    "next_pymnt_d",
    "last_credit_pull_d",
    "out_prncp",
    "out_prncp_inv",
    "total_rec_prncp",
    "total_rec_int",
    "total_rec_late_fee",
    "settlement_status",
    "hardship_flag",
    "hardship_type",
    "hardship_reason",
    "hardship_status",
    "hardship_loan_status",
    "debt_settlement_flag",
    "debt_settlement_flag_date",
    "settlement_date",
    "settlement_amount",
    "settlement_percentage",
    "settlement_term",
    "policy_code",
    *PROFIT_INPUT_COLUMNS,
}
