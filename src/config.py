import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_YAML = ROOT / "config.yaml"
ACCEPTED_CSV = ROOT / "accepted_2007_to_2018Q4.csv"
REJECTED_CSV = ROOT / "rejected_2007_to_2018Q4.csv"
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports"
MODEL_VERSION = "accepted-default-v1"
SUPPORTED_MODEL_CANDIDATES = [
    "logistic_balanced",
    "logistic",
    "random_forest",
    "hist_gradient_boosting",
]
MODEL_NAME_ALIASES = {
    "logistic": "logistic",
    "logistic_regression": "logistic",
    "logistic_balanced": "logistic_balanced",
    "random_forest": "random_forest",
    "histogram_gradient_boosting": "hist_gradient_boosting",
    "hist_gradient_boosting": "hist_gradient_boosting",
}
DEFAULT_TRAINING_MODELS = list(SUPPORTED_MODEL_CANDIDATES)
DEFAULT_CALIBRATION_METHODS = ["isotonic", "sigmoid"]
DEFAULT_CROSS_VALIDATION = True


"""
Defines config functions:

1. get environment path
2. standardizing config.yaml input & loading into files properly
"""


def _path_from_env(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value) if value else default


def _parse_scalar(value: str):
    value = value.strip()
    if not value:
        return ""
    if value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    lowered = value.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"null", "none", "~"}:
        return None
    return value


def _normalize_key(value: str) -> str:
    return re.sub(r"[\s\-]+", "_", value.strip().lower())


def _resolve_model_name(value: str) -> str:
    normalized = _normalize_key(value)
    return MODEL_NAME_ALIASES.get(normalized, normalized)


def _read_training_yaml(path: Path) -> dict:
    data: dict = {}
    if not path.exists():
        return data

    current = data
    current_list_key: str | None = None
    current_section: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if indent == 0 and stripped.endswith(":"):
            current_section = stripped[:-1].strip()
            current = data.setdefault(current_section, {})
            current_list_key = None
            continue
        if indent == 0 and ":" in stripped:
            key, value = stripped.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value:
                data[key] = _parse_scalar(value)
                current = data
                current_list_key = None
            else:
                current_section = key
                current = data.setdefault(current_section, {})
                current_list_key = None
            continue
        if indent > 0 and stripped.startswith("- "):
            if current_list_key is None:
                raise ValueError(f"unexpected list item in {path}: {raw_line}")
            if current.get(current_list_key) is None:
                current[current_list_key] = []
            if not isinstance(current.get(current_list_key), list):
                raise ValueError(f"unexpected list item in {path}: {raw_line}")
            current[current_list_key].append(_parse_scalar(stripped[2:]))
            continue
        if indent > 0 and ":" in stripped:
            key, value = stripped.split(":", 1)
            key = key.strip()
            value = value.strip()
            target = current
            if current_list_key is not None and indent > 2:
                if current.get(current_list_key) is None:
                    current[current_list_key] = {}
                if isinstance(current.get(current_list_key), list):
                    raise ValueError(f"unexpected mapping item in {path}: {raw_line}")
                target = current[current_list_key]
            if value:
                target[key] = _parse_scalar(value)
                if target is current:
                    current_list_key = None
            else:
                target[key] = None
                if target is current:
                    current_list_key = key
            continue
        raise ValueError(f"cannot parse config line in {path}: {raw_line}")

    return data


def load_training_config(path: str | Path | None = None) -> dict:
    config_path = Path(path) if path is not None else CONFIG_YAML
    raw = _read_training_yaml(config_path)
    config = raw.get("training", raw) if isinstance(raw, dict) else {}
    models = config.get("models", DEFAULT_TRAINING_MODELS)
    selected_model = config.get("selected_model")
    calibration_methods = config.get("calibration_methods", DEFAULT_CALIBRATION_METHODS)
    cross_validation = config.get("cross_validation", DEFAULT_CROSS_VALIDATION)
    if isinstance(models, dict):
        enabled_models = []
        for key, enabled in models.items():
            resolved = _resolve_model_name(str(key))
            if resolved not in SUPPORTED_MODEL_CANDIDATES:
                if bool(enabled):
                    raise ValueError(f"unknown model candidates in {config_path}: {key}")
                continue
            if bool(enabled):
                enabled_models.append(resolved)
        if not enabled_models:
            raise ValueError(f"no enabled model candidates in {config_path}")
        models = enabled_models
    elif isinstance(models, str):
        models = [_resolve_model_name(models)]
    else:
        models = [_resolve_model_name(str(model)) for model in models]

    if isinstance(calibration_methods, dict):
        calibration_methods = [str(method) for method, enabled in calibration_methods.items() if bool(enabled)]
        if not calibration_methods:
            raise ValueError(f"no enabled calibration methods in {config_path}")
    elif isinstance(calibration_methods, str):
        calibration_methods = [calibration_methods]
    else:
        calibration_methods = [str(method) for method in calibration_methods]
    unknown_models = [model for model in models if model not in SUPPORTED_MODEL_CANDIDATES]
    if unknown_models:
        raise ValueError(f"unknown model candidates in {config_path}: {', '.join(unknown_models)}")
    if selected_model is not None:
        selected_model = _resolve_model_name(str(selected_model))
        if selected_model not in models:
            raise ValueError(f"selected_model must be one of the enabled model candidates in {config_path}")
    unknown_methods = [method for method in calibration_methods if method not in DEFAULT_CALIBRATION_METHODS]
    if unknown_methods:
        raise ValueError(f"unknown calibration methods in {config_path}: {', '.join(unknown_methods)}")
    return {
        "path": str(config_path),
        "models": models,
        "selected_model": selected_model,
        "calibration_methods": calibration_methods,
        "cross_validation": bool(cross_validation),
    }

"""
Leakage flags and important config settings
"""


DEFAULT_ACCEPTED_BUNDLE = _path_from_env("ACCEPTED_MODEL_BUNDLE", ARTIFACT_DIR / "accepted_model.joblib")
DEFAULT_FRONTEND_BUNDLE = _path_from_env("FRONTEND_MODEL_BUNDLE", ARTIFACT_DIR / "frontend_model.joblib")
DEFAULT_PREPROCESSED_ACCEPTED_BUNDLE = _path_from_env(
    "PREPROCESSED_ACCEPTED_BUNDLE",
    ARTIFACT_DIR / "accepted_preprocessed.joblib",
)
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
