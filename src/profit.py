from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DEFAULT_LGD, PROFIT_INPUT_COLUMNS


def expected_profit(p_default, funded_amnt, term_months, installment, lgd: float = DEFAULT_LGD):
    p = np.asarray(p_default, dtype=float)
    funded = np.asarray(funded_amnt, dtype=float)
    term = np.asarray(term_months, dtype=float)
    payment = np.asarray(installment, dtype=float)
    good_profit = payment * term - funded
    bad_loss = -(lgd * funded)
    return (1 - p) * good_profit + p * bad_loss


def expected_return(expected_profit_value, funded_amnt):
    return np.asarray(expected_profit_value, dtype=float) / np.asarray(funded_amnt, dtype=float)


def approve(expected_profit_value, expected_return_value=None, required_return=None):
    if required_return is None:
        return np.asarray(expected_profit_value, dtype=float) > 0
    if expected_return_value is None:
        raise ValueError("expected_return_value is required when required_return is set")
    return np.asarray(expected_return_value, dtype=float) >= required_return


def realized_profit(total_pymnt, funded_amnt):
    return np.asarray(total_pymnt, dtype=float) - np.asarray(funded_amnt, dtype=float)


def require_valid_profit_inputs(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in PROFIT_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing profit input columns: {', '.join(missing)}")
    values = df[PROFIT_INPUT_COLUMNS].apply(pd.to_numeric, errors="coerce")
    invalid = values.isna().any(axis=1) | (values <= 0).any(axis=1)
    if invalid.any():
        raise ValueError(f"invalid profit inputs for {int(invalid.sum())} rows")
    return values


def policy_metrics(df: pd.DataFrame, p_default, lgd: float = DEFAULT_LGD, required_return=None) -> dict:
    profit_inputs = require_valid_profit_inputs(df)
    ep = expected_profit(
        p_default,
        profit_inputs["funded_amnt"],
        profit_inputs["term_months"],
        profit_inputs["installment"],
        lgd,
    )
    er = expected_return(ep, profit_inputs["funded_amnt"])
    decisions = approve(ep, er, required_return)
    approved = df.loc[decisions]
    approved_funded = np.asarray(approved["funded_amnt"], dtype=float) if decisions.any() else np.array([])
    metrics = {
        "expected_profit": float(np.sum(ep[decisions])),
        "expected_return": float(np.sum(ep[decisions]) / np.sum(approved_funded))
        if decisions.any() and np.sum(approved_funded)
        else 0.0,
        "approval_count": int(decisions.sum()),
        "selection_rate": float(decisions.mean()) if len(decisions) else 0.0,
        "mean_expected_profit": float(np.mean(ep[decisions])) if decisions.any() else 0.0,
        "mean_expected_return": float(np.mean(er[decisions])) if decisions.any() else 0.0,
        "lgd": lgd,
        "required_return": required_return,
    }
    if "default" in approved.columns:
        metrics["actual_default_rate"] = float(approved["default"].mean()) if len(approved) else 0.0
    if {"total_pymnt", "funded_amnt"}.issubset(approved.columns):
        rp = realized_profit(approved["total_pymnt"], approved["funded_amnt"])
        metrics["total_realized_profit"] = float(np.sum(rp)) if len(rp) else 0.0
        metrics["mean_realized_profit"] = float(np.mean(rp)) if len(rp) else 0.0
    return metrics


def policy_sensitivity(
    df: pd.DataFrame,
    p_default,
    lgds=(0.60, 0.75, 1.00),
    required_returns=(0.00, 0.05, 0.10),
) -> list[dict]:
    rows = []
    for lgd in lgds:
        for required_return in required_returns:
            metrics = policy_metrics(df, p_default, lgd=lgd, required_return=required_return)
            rows.append(
                {
                    "lgd": lgd,
                    "required_return": required_return,
                    "approval_count": metrics["approval_count"],
                    "selection_rate": metrics["selection_rate"],
                    "expected_profit": metrics["expected_profit"],
                    "expected_return": metrics["expected_return"],
                    "total_realized_profit": metrics.get("total_realized_profit"),
                }
            )
    return rows
