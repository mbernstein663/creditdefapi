from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DEFAULT_LGD, PROFIT_INPUT_COLUMNS


def expected_profit(
    p_default,
    funded_amnt,
    term_months,
    installment,
    lgd: float = DEFAULT_LGD,
    good_profit_haircut: float = 1.0,
):
    p = np.asarray(p_default, dtype=float)
    funded = np.asarray(funded_amnt, dtype=float)
    term = np.asarray(term_months, dtype=float)
    payment = np.asarray(installment, dtype=float)
    good_profit = (payment * term - funded) * float(good_profit_haircut)
    bad_loss = -(lgd * funded)
    return (1 - p) * good_profit + p * bad_loss


def expected_return(expected_profit_value, funded_amnt):
    return np.asarray(expected_profit_value, dtype=float) / np.asarray(funded_amnt, dtype=float)


def expected_npv_profit(
    p_default,
    funded_amnt,
    term_months,
    installment,
    lgd: float = DEFAULT_LGD,
    annual_discount_rate: float = 0.08,
    servicing_cost_rate: float = 0.0,
    good_profit_haircut: float = 1.0,
):
    p = np.asarray(p_default, dtype=float)
    funded = np.asarray(funded_amnt, dtype=float)
    term = np.asarray(term_months, dtype=float)
    payment = np.asarray(installment, dtype=float)
    monthly_discount = (1 + float(annual_discount_rate)) ** (1 / 12) - 1
    good_pv = np.zeros_like(funded, dtype=float)
    for idx, months_to_pay in enumerate(term.astype(int)):
        discount = (1 + monthly_discount) ** np.arange(1, max(months_to_pay, 1) + 1)
        good_pv[idx] = (
            np.sum(payment[idx] / discount)
            - funded[idx]
            - (servicing_cost_rate * funded[idx] * months_to_pay / 12)
        ) * float(good_profit_haircut)
    bad_pv = -(lgd * funded) - (servicing_cost_rate * funded * term / 12)
    return (1 - p) * good_pv + p * bad_pv


def annualized_profit_rate(expected_profit_value, funded_amnt, term_months):
    expected = np.asarray(expected_profit_value, dtype=float)
    funded = np.asarray(funded_amnt, dtype=float)
    years = np.asarray(term_months, dtype=float) / 12.0
    return expected / funded / years


def approve(expected_profit_value, expected_return_value=None, required_return=None):
    if required_return is None:
        return np.asarray(expected_profit_value, dtype=float) > 0
    if expected_return_value is None:
        raise ValueError("expected_return_value is required when required_return is set")
    return np.asarray(expected_return_value, dtype=float) > required_return


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


def policy_metrics(
    df: pd.DataFrame,
    p_default,
    lgd: float = DEFAULT_LGD,
    required_return=None,
    use_npv: bool = False,
    annual_discount_rate: float = 0.08,
    servicing_cost_rate: float = 0.0,
    recovery_rate: float = 0.25,
    good_profit_haircut: float = 1.0,
) -> dict:
    profit_inputs = require_valid_profit_inputs(df)
    ep = expected_profit(
        p_default,
        profit_inputs["funded_amnt"],
        profit_inputs["term_months"],
        profit_inputs["installment"],
        lgd,
        good_profit_haircut=good_profit_haircut,
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
        "good_profit_haircut": good_profit_haircut,
    }
    if use_npv:
        npv = expected_npv_profit(
            p_default,
            profit_inputs["funded_amnt"],
            profit_inputs["term_months"],
            profit_inputs["installment"],
            lgd=lgd,
            annual_discount_rate=annual_discount_rate,
            servicing_cost_rate=servicing_cost_rate,
            good_profit_haircut=good_profit_haircut,
        )
        metrics["expected_npv_profit"] = float(np.sum(npv[decisions])) if decisions.any() else 0.0
        metrics["mean_expected_npv_profit"] = float(np.mean(npv[decisions])) if decisions.any() else 0.0
        metrics["annualized_return"] = float(
            np.mean(annualized_profit_rate(npv[decisions], approved_funded, profit_inputs.loc[decisions, "term_months"]))
        ) if decisions.any() and np.all(approved_funded > 0) else 0.0
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
    good_profit_haircuts=(1.00,),
    annual_discount_rates=(0.08,),
) -> list[dict]:
    rows = []
    for lgd in lgds:
        for required_return in required_returns:
            for good_profit_haircut in good_profit_haircuts:
                for annual_discount_rate in annual_discount_rates:
                    metrics = policy_metrics(
                        df,
                        p_default,
                        lgd=lgd,
                        required_return=required_return,
                        use_npv=True,
                        annual_discount_rate=annual_discount_rate,
                        good_profit_haircut=good_profit_haircut,
                    )
                    rows.append(
                        {
                            "lgd": lgd,
                            "required_return": required_return,
                            "good_profit_haircut": good_profit_haircut,
                            "annual_discount_rate": annual_discount_rate,
                            "approval_count": metrics["approval_count"],
                            "selection_rate": metrics["selection_rate"],
                            "expected_profit": metrics["expected_profit"],
                            "expected_return": metrics["expected_return"],
                            "expected_npv_profit": metrics.get("expected_npv_profit"),
                            "annualized_return": metrics.get("annualized_return"),
                            "total_realized_profit": metrics.get("total_realized_profit"),
                        }
                    )
    return rows
