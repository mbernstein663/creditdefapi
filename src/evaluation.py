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


def evaluate_profit_policy(bundle, frame, lgd: float | None = None):
    if bundle.calibrator is None:
        raise ValueError("profit evaluation requires calibrated probabilities")
    locked_lgd = bundle.policy.get("lgd", DEFAULT_LGD) if lgd is None else lgd
    raw = predict_raw_default(bundle.model, frame, bundle.feature_columns)
    p_default = bundle.calibrator.predict(raw)
    return policy_metrics(frame, p_default, lgd=locked_lgd, required_return=bundle.policy.get("required_return"))
