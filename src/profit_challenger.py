from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import load_model_bundle
from .config import ACCEPTED_RISK_FEATURES, PROFIT_TARGET, TARGET
from .models import predict_raw_default
from .preprocessing import ensure_no_forbidden_features, prepare_accepted_loans
from .profit import policy_metrics


def validate_profit_features(feature_columns) -> None:
    ensure_no_forbidden_features(feature_columns)
    extra = sorted(set(feature_columns) - set(ACCEPTED_RISK_FEATURES))
    if extra:
        raise ValueError(f"non-allowlisted profit model features: {', '.join(extra)}")


def prepare_profit_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = prepare_accepted_loans(df)
    missing = [c for c in ["total_pymnt", "funded_amnt"] if c not in out.columns]
    if missing:
        raise ValueError(f"missing realized-profit columns: {', '.join(missing)}")
    values = out[["total_pymnt", "funded_amnt"]].apply(pd.to_numeric, errors="coerce")
    invalid = values.isna().any(axis=1) | (values["funded_amnt"] <= 0)
    if invalid.any():
        raise ValueError(f"invalid realized-profit target rows: {int(invalid.sum())}")
    out[PROFIT_TARGET] = values["total_pymnt"] - values["funded_amnt"]
    return out


def predict_profit(bundle, frame: pd.DataFrame):
    validate_profit_features(bundle.feature_columns)
    missing = [c for c in bundle.feature_columns if c not in frame.columns]
    if missing:
        raise ValueError(f"missing required profit scoring fields: {', '.join(missing)}")
    return bundle.model.predict(frame[bundle.feature_columns])


def policy_mask(predicted_profit, policy: dict):
    pred = np.asarray(predicted_profit, dtype=float)
    if policy["type"] == "threshold":
        return pred > float(policy["threshold"])
    if policy["type"] == "top_percent":
        pct = float(policy["top_percent"])
        if not 0 < pct <= 1:
            raise ValueError("top_percent policy must be in (0, 1]")
        approved = min(len(pred), math.ceil(len(pred) * pct))
        mask = np.zeros(len(pred), dtype=bool)
        if approved:
            mask[np.argsort(-pred)[:approved]] = True
        return mask
    raise ValueError(f"unknown profit policy type: {policy['type']}")


def decile_lift_table(df: pd.DataFrame, predicted_profit, bins: int = 10) -> list[dict]:
    frame = pd.DataFrame(
        {
            "predicted_profit": np.asarray(predicted_profit, dtype=float),
            PROFIT_TARGET: pd.to_numeric(df[PROFIT_TARGET], errors="coerce"),
            "funded_amnt": pd.to_numeric(df["funded_amnt"], errors="coerce"),
        },
        index=df.index,
    ).dropna()
    if frame.empty:
        return []
    try:
        raw_decile = pd.qcut(
            frame["predicted_profit"].rank(method="first"),
            q=min(bins, len(frame)),
            labels=False,
            duplicates="drop",
        )
        frame["decile"] = int(raw_decile.max()) + 1 - raw_decile.astype(int)
    except ValueError:
        frame["decile"] = 1
    if TARGET in df.columns:
        frame[TARGET] = df.loc[frame.index, TARGET]
    grouped = frame.groupby("decile", as_index=False).agg(
        count=(PROFIT_TARGET, "size"),
        average_predicted_profit=("predicted_profit", "mean"),
        total_realized_profit=(PROFIT_TARGET, "sum"),
        mean_realized_profit=(PROFIT_TARGET, "mean"),
        median_realized_profit=(PROFIT_TARGET, "median"),
        total_funded_amount=("funded_amnt", "sum"),
    )
    grouped["profit_per_dollar_funded"] = (
        grouped["total_realized_profit"] / grouped["total_funded_amount"]
    )
    if TARGET in frame.columns:
        default_rates = frame.groupby("decile")[TARGET].mean()
        grouped["realized_default_rate"] = grouped["decile"].map(default_rates)
    return grouped.to_dict(orient="records")


def profit_policy_metrics(df: pd.DataFrame, predicted_profit, policy: dict) -> dict:
    pred = np.asarray(predicted_profit, dtype=float)
    actual = pd.to_numeric(df[PROFIT_TARGET], errors="coerce").to_numpy(dtype=float)
    funded = pd.to_numeric(df["funded_amnt"], errors="coerce").to_numpy(dtype=float)
    mask = policy_mask(pred, policy)
    approved_actual = actual[mask]
    approved_funded = funded[mask]
    total_profit = float(np.sum(approved_actual)) if len(approved_actual) else 0.0
    total_funded = float(np.sum(approved_funded)) if len(approved_funded) else 0.0
    corr = None
    if len(pred) > 1 and np.std(pred) > 0 and np.std(actual) > 0:
        corr = float(np.corrcoef(pred, actual)[0, 1])
    metrics = {
        "policy": policy,
        "approval_count": int(mask.sum()),
        "selection_rate": float(mask.mean()) if len(mask) else 0.0,
        "total_realized_profit": total_profit,
        "mean_realized_profit": float(np.mean(approved_actual)) if len(approved_actual) else 0.0,
        "median_realized_profit": float(np.median(approved_actual)) if len(approved_actual) else 0.0,
        "total_funded_amount": total_funded,
        "profit_per_dollar_funded": total_profit / total_funded if total_funded else 0.0,
        "average_predicted_profit": float(np.mean(pred[mask])) if mask.any() else 0.0,
        "predicted_actual_correlation": corr,
        "decile_lift": decile_lift_table(df, pred),
    }
    if TARGET in df.columns:
        approved = df.loc[mask]
        metrics["realized_default_rate"] = float(approved[TARGET].mean()) if len(approved) else 0.0
    return metrics


def candidate_profit_policies(predicted_profit) -> list[dict]:
    pred = pd.Series(np.asarray(predicted_profit, dtype=float)).dropna()
    policies = [{"type": "threshold", "threshold": 0.0}]
    for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
        policies.append({"type": "threshold", "threshold": float(pred.quantile(q))})
    for pct in [0.1, 0.25, 0.5, 0.75, 1.0]:
        policies.append({"type": "top_percent", "top_percent": pct})
    seen = set()
    unique = []
    for policy in policies:
        key = tuple(sorted(policy.items()))
        if key not in seen:
            seen.add(key)
            unique.append(policy)
    return unique


def search_profit_policy(df: pd.DataFrame, predicted_profit) -> tuple[dict, list[dict]]:
    rows = []
    for policy in candidate_profit_policies(predicted_profit):
        metrics = profit_policy_metrics(df, predicted_profit, policy)
        rows.append({k: v for k, v in metrics.items() if k != "decile_lift"})
    best = max(
        rows,
        key=lambda m: (
            m["total_realized_profit"],
            m["profit_per_dollar_funded"],
            m["approval_count"],
        ),
    )
    return best["policy"], rows


def regression_summary(y_true, predicted) -> dict:
    actual = np.asarray(y_true, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    corr = None
    if len(pred) > 1 and np.std(pred) > 0 and np.std(actual) > 0:
        corr = float(np.corrcoef(pred, actual)[0, 1])
    return {
        "rmse": float(np.sqrt(np.mean((pred - actual) ** 2))),
        "mae": float(np.mean(np.abs(pred - actual))),
        "predicted_actual_correlation": corr,
    }


def default_risk_policy_metrics(frame: pd.DataFrame, accepted_bundle_path) -> dict:
    path = Path(accepted_bundle_path)
    if not path.exists():
        return {"available": False, "reason": f"missing accepted-risk bundle: {path}"}
    bundle = load_model_bundle(path)
    if bundle.calibrator is None:
        return {"available": False, "reason": "accepted-risk bundle is not calibrated"}
    raw = predict_raw_default(bundle.model, frame, bundle.feature_columns)
    p_default = bundle.calibrator.predict(raw)
    metrics = policy_metrics(
        frame,
        p_default,
        lgd=bundle.policy.get("lgd", 1.0),
        required_return=bundle.policy.get("required_return"),
    )
    return {"available": True, "bundle": str(path), "metrics": metrics}
