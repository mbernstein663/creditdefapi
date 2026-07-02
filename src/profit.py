from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DEFAULT_LGD


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
    return np.asarray(expected_return_value, dtype=float) > required_return


def realized_profit(total_pymnt, funded_amnt):
    return np.asarray(total_pymnt, dtype=float) - np.asarray(funded_amnt, dtype=float)


def policy_metrics(df: pd.DataFrame, p_default, lgd: float = DEFAULT_LGD, required_return=None) -> dict:
    ep = expected_profit(p_default, df["funded_amnt"], df["term_months"], df["installment"], lgd)
    er = expected_return(ep, df["funded_amnt"])
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
