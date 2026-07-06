import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACCEPTED_CSV = ROOT / "accepted_2007_to_2018Q4.csv"
REJECTED_CSV = ROOT / "rejected_2007_to_2018Q4.csv"
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports"
MODEL_VERSION = "accepted-default-v1"


def _path_from_env(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value) if value else default


DEFAULT_ACCEPTED_BUNDLE = _path_from_env("ACCEPTED_MODEL_BUNDLE", ARTIFACT_DIR / "accepted_model.joblib")
DEFAULT_FRONTEND_BUNDLE = _path_from_env("FRONTEND_MODEL_BUNDLE", ARTIFACT_DIR / "frontend_model.joblib")
TARGET = "default"
EVALUATION_BOOTSTRAP_SAMPLES = 200
EVALUATION_BOOTSTRAP_RANDOM_STATE = 42
FRONTEND_TOP_FEATURE_COUNT = 5

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

RISK_NUMERIC_LOG_TRANSFORMS = {
    "loan_amnt",
    "annual_inc",
    "revol_bal",
    "total_acc",
}
RISK_NUMERIC_CLIP_RANGES = {
    "dti": (0.0, 100.0),
}

FORBIDDEN_FEATURE_COLUMNS = {
    "default",
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
}
