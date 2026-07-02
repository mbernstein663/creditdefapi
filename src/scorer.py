from __future__ import annotations

import pandas as pd

from .artifacts import ModelBundle
from .config import DEFAULT_LGD, PROFIT_INPUT_COLUMNS
from .preprocessing import ensure_no_forbidden_features, normalize_rejected_input, parse_term_months
from .profit import approve, expected_profit, expected_return


def _prepare_frame(frame: pd.DataFrame, bundle: ModelBundle) -> pd.DataFrame:
    out = normalize_rejected_input(frame) if bundle.model_type == "rejected_style" else frame.copy()
    if "term_months" not in out.columns and "term" in out.columns:
        out["term_months"] = out["term"].map(parse_term_months)
    return out


def _missing(columns, frame: pd.DataFrame) -> list[str]:
    return [c for c in columns if c not in frame.columns]


def predict_default(bundle: ModelBundle, frame: pd.DataFrame):
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
    scored["lgd"] = locked_lgd
    scored["required_return"] = bundle.policy.get("required_return")
    scored["approval_rule"] = bundle.policy.get("approval_rule", "expected_profit > 0")

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
        decisions = approve(ep, er, bundle.policy.get("required_return"))
        scored.loc[valid_profit, "expected_profit"] = ep
        scored.loc[valid_profit, "expected_return"] = er
        scored.loc[valid_profit, "decision"] = pd.Series(decisions, index=valid.index).map(
            {True: "approve", False: "deny"}
        )
    if bundle.model_type == "rejected_style":
        scored.loc[valid_profit, "reason"] = (
            "accepted-loan-trained risk with supplied profit inputs; no rejected-loan outcome claim"
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
    return scored[[c for c in columns if c in scored.columns]].to_dict(orient="records")
