from __future__ import annotations

import pandas as pd
import numpy as np

from .artifacts import ModelBundle
from .config import ACCEPTED_RISK_FEATURES, DEFAULT_LGD, PROFIT_INPUT_COLUMNS, REJECTED_STYLE_RISK_FEATURES
from .preprocessing import ensure_no_forbidden_features, normalize_rejected_input, parse_term_months
from .profit import approve, expected_profit, expected_return


def _prepare_frame(frame: pd.DataFrame, bundle: ModelBundle) -> pd.DataFrame:
    out = normalize_rejected_input(frame) if bundle.model_type == "rejected_style" else frame.copy()
    if "term_months" not in out.columns and "term" in out.columns:
        out["term_months"] = out["term"].map(parse_term_months)
    return out.where(pd.notna(out), np.nan)


def _validate_bundle_features(bundle: ModelBundle) -> None:
    expected = {
        "accepted": ACCEPTED_RISK_FEATURES,
        "rejected_style": REJECTED_STYLE_RISK_FEATURES,
    }.get(bundle.model_type)
    if expected is None:
        raise ValueError(f"unknown model type: {bundle.model_type}")
    if list(bundle.feature_columns) != list(expected):
        raise ValueError(f"{bundle.model_type} bundle feature columns do not match the configured allowlist")


def _approval_rule(required_return) -> str:
    return "expected_profit > 0" if required_return is None else "expected_return >= required_return"


def _missing(columns, frame: pd.DataFrame) -> list[str]:
    return [c for c in columns if c not in frame.columns]


def predict_default(bundle: ModelBundle, frame: pd.DataFrame):
    _validate_bundle_features(bundle)
    ensure_no_forbidden_features(bundle.feature_columns)
    missing = _missing(bundle.feature_columns, frame)
    if missing:
        raise ValueError(f"missing required scoring fields: {', '.join(missing)}")
    if bundle.calibrator is None:
        raise ValueError("scoring requires a calibrated model bundle")
    raw = bundle.model.predict_proba(frame[bundle.feature_columns])[:, 1]
    return bundle.calibrator.predict(raw)


def _profit_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[PROFIT_INPUT_COLUMNS].apply(pd.to_numeric, errors="coerce")


def score_frame(frame: pd.DataFrame, bundle: ModelBundle, lgd: float | None = None) -> pd.DataFrame:
    data = _prepare_frame(frame, bundle)
    p_default = predict_default(bundle, data)
    scored = data.copy()
    scored["p_default"] = p_default
    locked_lgd = bundle.policy.get("lgd", DEFAULT_LGD) if lgd is None else lgd
    required_return = bundle.policy.get("required_return")
    scored["lgd"] = locked_lgd
    scored["required_return"] = required_return
    scored["approval_rule"] = _approval_rule(required_return)

    missing_profit = _missing(PROFIT_INPUT_COLUMNS, scored)
    if missing_profit:
        scored["decision"] = "review"
        scored["reason"] = (
            "risk/review only; profit decision unavailable; missing " + ", ".join(missing_profit)
        )
        return scored

    profit_values = _profit_inputs(scored)
    valid_profit = profit_values.notna().all(axis=1) & (profit_values > 0).all(axis=1)
    scored["expected_profit"] = pd.NA
    scored["expected_return"] = pd.NA
    scored["decision"] = "review"
    scored["reason"] = "risk/review only; invalid profit inputs"
    if valid_profit.any():
        valid = scored.loc[valid_profit]
        ep = expected_profit(
            valid["p_default"],
            profit_values.loc[valid_profit, "funded_amnt"],
            profit_values.loc[valid_profit, "term_months"],
            profit_values.loc[valid_profit, "installment"],
            lgd=locked_lgd,
        )
        er = expected_return(ep, profit_values.loc[valid_profit, "funded_amnt"])
        decisions = approve(ep, er, required_return)
        scored.loc[valid_profit, "expected_profit"] = ep
        scored.loc[valid_profit, "expected_return"] = er
        if bundle.model_type != "rejected_style":
            scored.loc[valid_profit, "decision"] = pd.Series(decisions, index=valid.index).map(
                {True: "approve", False: "deny"}
            )
    if bundle.model_type == "rejected_style":
        scored.loc[valid_profit, "reason"] = (
            "limited-field risk estimate; profit is scenario math only, not a rejected-applicant approval"
        )
    else:
        scored.loc[valid_profit, "reason"] = "expected profit policy"
    return scored


def score_records(records, bundle: ModelBundle, lgd: float | None = None) -> list[dict]:
    frame = pd.DataFrame(records)
    scored = score_frame(frame, bundle, lgd=lgd)
    columns = [
        "p_default",
        "expected_profit",
        "expected_return",
        "decision",
        "reason",
        "lgd",
        "required_return",
        "approval_rule",
    ]
    result = scored[[c for c in columns if c in scored.columns]].astype(object)
    result = result.where(pd.notna(result), None)
    return result.to_dict(orient="records")
