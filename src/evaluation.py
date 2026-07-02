from __future__ import annotations

from .calibration import calibration_summary
from .config import DEFAULT_LGD, TARGET
from .models import predict_raw_default
from .profit import policy_metrics


def evaluate_probability(bundle, frame):
    if bundle.calibrator is None:
        raise ValueError("locked evaluation requires calibrated probabilities")
    raw = predict_raw_default(bundle.model, frame, bundle.feature_columns)
    p_default = bundle.calibrator.predict(raw)
    return calibration_summary(frame[TARGET], p_default)


def evaluate_profit_policy(bundle, frame):
    if bundle.calibrator is None:
        raise ValueError("profit evaluation requires calibrated probabilities")
    if bundle.model_type != "accepted":
        raise ValueError("locked profit evaluation is only valid for accepted-loan bundles")
    required = {TARGET, "total_pymnt", "funded_amnt", "term_months", "installment"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing locked profit evaluation columns: {', '.join(missing)}")
    locked_lgd = bundle.policy.get("lgd", DEFAULT_LGD)
    raw = predict_raw_default(bundle.model, frame, bundle.feature_columns)
    p_default = bundle.calibrator.predict(raw)
    return policy_metrics(frame, p_default, lgd=locked_lgd, required_return=bundle.policy.get("required_return"))
