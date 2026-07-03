from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_mpl_config = Path(__file__).resolve().parents[1] / ".matplotlib"
_mpl_config.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_config))

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from .artifacts import file_fingerprint, load_model_bundle
from .calibration import calibration_summary
from .config import (
    ACCEPTED_CSV,
    ACCEPTED_TO_REJECTED_FEATURE_MAP,
    ACCEPTED_CATEGORICAL_RISK_FEATURES,
    ACCEPTED_NUMERIC_RISK_FEATURES,
    DEFAULT_ACCEPTED_BUNDLE,
    DEFAULT_LGD,
    DEFAULT_PROFIT_BUNDLE,
    DEFAULT_REJECTED_STYLE_BUNDLE,
    DEFAULT_REQUIRED_RETURN,
    DEFAULT_TARGET_HORIZON_MONTHS,
    DEFAULT_TARGET_MODE,
    EVALUATION_BOOTSTRAP_RANDOM_STATE,
    EVALUATION_BOOTSTRAP_SAMPLES,
    POST_PRICING_FIELDS,
    PRODUCT_MODE_POST_PRICING,
    PROFIT_INPUT_COLUMNS,
    PRODUCT_MODE_PRE_UNDERWRITING,
    REPORT_DIR,
    TARGET,
)
from .models import predict_raw_default
from .preprocessing import map_accepted_to_rejected_style, prepare_accepted_loans, split_chronological
from .profit import expected_profit, policy_metrics
from .profit_challenger import policy_mask, predict_profit


@dataclass(slots=True)
class EvaluationBundleResult:
    name: str
    path: Path
    bundle: Any
    frame: pd.DataFrame
    probability: pd.Series | None = None
    predicted_profit: pd.Series | None = None
    note: str = ""


def _repo_relative(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(Path(__file__).resolve().parents[1]))
    except ValueError:
        return str(resolved)


def _unique_columns(columns: list[str]) -> list[str]:
    return list(dict.fromkeys(columns))


def _target_config(bundle) -> dict:
    if bundle.metadata.get("target_mode", DEFAULT_TARGET_MODE) != "default_within_horizon":
        return {}
    target_summary = bundle.metadata.get("target_summary", {}) or {}
    return {
        "horizon_months": int(target_summary.get("horizon_months", DEFAULT_TARGET_HORIZON_MONTHS)),
    }


def _source_columns_for_bundle(bundle) -> list[str]:
    columns = list(bundle.feature_columns)
    columns += list(PROFIT_INPUT_COLUMNS)
    columns += [
        TARGET,
        "loan_status",
        "issue_d",
        "last_pymnt_d",
        "term",
        "total_pymnt",
        "funded_amnt",
        "id",
        "issue_dt",
        "issue_year",
        "issue_quarter",
        "issue_month",
        "title",
    ]
    if bundle.model_type == "rejected_style":
        columns += list(ACCEPTED_TO_REJECTED_FEATURE_MAP)
    return _unique_columns(columns)


def _read_source_frame(csv_path: str | Path, bundle, sample: int | None = None) -> pd.DataFrame:
    source = pd.read_csv(
        csv_path,
        usecols=lambda col: col in set(_source_columns_for_bundle(bundle)),
        low_memory=False,
        nrows=sample,
    )
    return source


def _prepare_validation_frame(bundle, csv_path: str | Path, stage: str, sample: int | None = None) -> tuple[pd.DataFrame, dict]:
    source = _read_source_frame(csv_path, bundle, sample=sample)
    prepared, target_summary = prepare_accepted_loans(
        source,
        target_mode=bundle.metadata.get("target_mode", DEFAULT_TARGET_MODE),
        target_config=_target_config(bundle),
        return_summary=True,
    )
    splits = split_chronological(prepared)
    if stage == "test":
        test_ids = set(bundle.metadata.get("split_manifest", {}).get("test_ids", []))
        if test_ids and "id" in prepared.columns:
            stage_frame = prepared.loc[prepared["id"].astype(str).isin(test_ids)].copy()
        else:
            stage_frame = splits["test"].copy()
    else:
        stage_frame = splits["validation"].copy()

    stage_summary = {
        "stage": stage,
        "row_count": int(len(stage_frame)),
        "default_rate": float(stage_frame[TARGET].mean()) if TARGET in stage_frame.columns and len(stage_frame) else None,
        "split_summary": [
            {
                "split": name,
                "rows": int(len(frame)),
                "default_rate": float(frame[TARGET].mean()) if TARGET in frame.columns and len(frame) else None,
                "date_min": frame["issue_dt"].min().isoformat() if "issue_dt" in frame.columns and len(frame) else None,
                "date_max": frame["issue_dt"].max().isoformat() if "issue_dt" in frame.columns and len(frame) else None,
            }
            for name, frame in splits.items()
        ],
        "target_summary": target_summary,
    }
    return stage_frame, stage_summary


def _safe_auc(y_true: pd.Series, score: pd.Series) -> float | None:
    y = pd.to_numeric(y_true, errors="coerce").dropna().astype(int)
    p = pd.to_numeric(score, errors="coerce").loc[y.index].astype(float)
    if y.nunique() < 2:
        return None
    return float(roc_auc_score(y, p))


def _safe_pr_auc(y_true: pd.Series, score: pd.Series) -> float | None:
    y = pd.to_numeric(y_true, errors="coerce").dropna().astype(int)
    p = pd.to_numeric(score, errors="coerce").loc[y.index].astype(float)
    if y.nunique() < 2:
        return None
    return float(average_precision_score(y, p))


def _safe_log_loss(y_true: pd.Series, score: pd.Series) -> float | None:
    y = pd.to_numeric(y_true, errors="coerce").dropna().astype(int)
    p = pd.to_numeric(score, errors="coerce").loc[y.index].astype(float)
    if y.empty or y.nunique() < 2:
        return None
    return float(log_loss(y, p, labels=[0, 1]))


def _policy_kwargs(policy: dict) -> dict:
    return {
        "lgd": policy.get("lgd", DEFAULT_LGD),
        "required_return": policy.get("required_return", DEFAULT_REQUIRED_RETURN),
        "use_npv": bool(policy.get("use_npv")),
        "annual_discount_rate": float(policy.get("annual_discount_rate", 0.08)),
        "servicing_cost_rate": float(policy.get("servicing_cost_rate", 0.0)),
        "good_profit_haircut": float(policy.get("good_profit_haircut", 1.0)),
    }


def _expected_profit_series(frame: pd.DataFrame, probability: pd.Series, bundle) -> pd.Series:
    profit_inputs = frame[PROFIT_INPUT_COLUMNS].apply(pd.to_numeric, errors="coerce")
    p = pd.to_numeric(probability, errors="coerce").to_numpy(dtype=float)
    values = expected_profit(
        p,
        profit_inputs["funded_amnt"],
        profit_inputs["term_months"],
        profit_inputs["installment"],
        lgd=bundle.policy.get("lgd", DEFAULT_LGD),
        good_profit_haircut=float(bundle.policy.get("good_profit_haircut", 1.0)),
    )
    return pd.Series(values, index=frame.index)


def _expected_return_series(frame: pd.DataFrame, probability: pd.Series, bundle) -> pd.Series:
    expected_profit = _expected_profit_series(frame, probability, bundle)
    return expected_profit / pd.to_numeric(frame["funded_amnt"], errors="coerce")


def _probability_bundle_result(name: str, path: Path, bundle, frame: pd.DataFrame) -> EvaluationBundleResult:
    if bundle.calibrator is None:
        raise ValueError(f"{name} bundle requires a calibrator for default-probability evaluation")
    raw = predict_raw_default(bundle.model, frame, bundle.feature_columns)
    probability = pd.Series(bundle.calibrator.predict(raw), index=frame.index)
    return EvaluationBundleResult(name=name, path=path, bundle=bundle, frame=frame, probability=probability)


def _profit_bundle_result(name: str, path: Path, bundle, frame: pd.DataFrame) -> EvaluationBundleResult:
    predicted_profit = pd.Series(predict_profit(bundle, frame), index=frame.index)
    return EvaluationBundleResult(name=name, path=path, bundle=bundle, frame=frame, predicted_profit=predicted_profit)


def _evaluate_review_only(bundle, frame: pd.DataFrame, probability: pd.Series) -> dict[str, Any]:
    summary = calibration_summary(frame[TARGET], probability)
    return {
        "calibration": summary,
        "approval_rate": None,
        "approved_default_rate": None,
        "expected_profit": None,
        "realized_profit": None,
        "profit_per_loan": None,
        "profit_per_funded_dollar": None,
        "selected_threshold": None,
        "selected_count": None,
    }


def _evaluate_probability_model(bundle, frame: pd.DataFrame, probability: pd.Series) -> dict[str, Any]:
    summary = calibration_summary(frame[TARGET], probability)
    metrics = policy_metrics(
        frame,
        probability,
        **_policy_kwargs(bundle.policy),
    )
    y = pd.to_numeric(frame[TARGET], errors="coerce")
    metrics.update(
        {
            "model_name": bundle.model_type,
            "calibration_method": bundle.metadata.get("calibration_method", "isotonic"),
            "roc_auc": summary["roc_auc"],
            "pr_auc": summary["pr_auc"],
            "brier_score": summary["brier_score"],
            "log_loss": _safe_log_loss(frame[TARGET], probability),
            "mean_predicted_default": summary["mean_predicted_default"],
            "observed_default_rate": summary["actual_default_rate"],
            "approval_rate": metrics["selection_rate"],
            "approved_default_rate": metrics.get("actual_default_rate"),
            "expected_profit": metrics["expected_profit"],
            "realized_profit": metrics.get("total_realized_profit"),
            "profit_per_loan": metrics["mean_expected_profit"],
            "profit_per_funded_dollar": metrics["expected_return"],
            "selected_threshold": bundle.policy.get("required_return", 0.0) if bundle.policy.get("required_return") is not None else 0.0,
            "selected_count": metrics["approval_count"],
        }
    )
    metrics["calibration_deciles"] = summary["deciles"]
    return metrics


def _evaluate_review_bundle(bundle, frame: pd.DataFrame, probability: pd.Series) -> dict[str, Any]:
    summary = calibration_summary(frame[TARGET], probability)
    return {
        "model_name": bundle.model_type,
        "calibration_method": bundle.metadata.get("calibration_method", "isotonic"),
        "roc_auc": summary["roc_auc"],
        "pr_auc": summary["pr_auc"],
        "brier_score": summary["brier_score"],
        "log_loss": _safe_log_loss(frame[TARGET], probability),
        "mean_predicted_default": summary["mean_predicted_default"],
        "observed_default_rate": summary["actual_default_rate"],
        "approval_rate": None,
        "approved_default_rate": None,
        "expected_profit": None,
        "realized_profit": None,
        "profit_per_loan": None,
        "profit_per_funded_dollar": None,
        "selected_threshold": None,
        "selected_count": None,
        "calibration_deciles": summary["deciles"],
    }


def _evaluate_profit_model(bundle, frame: pd.DataFrame, predicted_profit: pd.Series) -> dict[str, Any]:
    mask = pd.Series(policy_mask(predicted_profit, bundle.policy), index=frame.index)
    selected = frame.loc[mask]
    selected_pred = predicted_profit.loc[mask]
    funded = pd.to_numeric(selected.get("funded_amnt"), errors="coerce") if "funded_amnt" in selected.columns else pd.Series(dtype=float)
    realized = (
        pd.to_numeric(selected.get("realized_profit"), errors="coerce")
        if "realized_profit" in selected.columns
        else pd.Series(dtype=float)
    )
    if "realized_profit" not in selected.columns and {"total_pymnt", "funded_amnt"}.issubset(selected.columns):
        realized = pd.to_numeric(selected["total_pymnt"], errors="coerce") - pd.to_numeric(selected["funded_amnt"], errors="coerce")
    total_expected_profit = float(selected_pred.sum()) if len(selected_pred) else 0.0
    total_realized_profit = float(realized.sum()) if len(realized) else None
    total_funded = float(funded.sum()) if len(funded) else None
    default_rate = float(selected[TARGET].mean()) if TARGET in selected.columns and len(selected) else None
    return {
        "model_name": bundle.model_type,
        "calibration_method": "none",
        "roc_auc": None,
        "pr_auc": None,
        "brier_score": None,
        "log_loss": None,
        "mean_predicted_default": None,
        "observed_default_rate": float(frame[TARGET].mean()) if TARGET in frame.columns and len(frame) else None,
        "approval_rate": float(mask.mean()) if len(mask) else 0.0,
        "approved_default_rate": default_rate,
        "expected_profit": total_expected_profit,
        "realized_profit": total_realized_profit,
        "profit_per_loan": float(selected_pred.mean()) if len(selected_pred) else 0.0,
        "profit_per_funded_dollar": (total_expected_profit / total_funded) if total_funded else None,
        "selected_threshold": bundle.policy.get("threshold"),
        "selected_count": int(mask.sum()),
        "calibration_deciles": [],
    }


def _model_comparison_rows(results: list[EvaluationBundleResult]) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    rows = []
    detail_map: dict[str, dict[str, Any]] = {}
    for result in results:
        bundle = result.bundle
        if result.probability is not None:
            if bundle.model_type == "rejected_style" and bundle.policy.get("approval_rule", "").startswith("review only"):
                metrics = _evaluate_review_bundle(bundle, result.frame, result.probability)
            else:
                metrics = _evaluate_probability_model(bundle, result.frame, result.probability)
        elif result.predicted_profit is not None:
            metrics = _evaluate_profit_model(bundle, result.frame, result.predicted_profit)
        else:
            continue
        detail_map[result.name] = metrics
        rows.append(
            {
                "model_name": result.name,
                "bundle_type": bundle.model_type,
                "calibration_method": metrics["calibration_method"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "brier_score": metrics["brier_score"],
                "log_loss": metrics["log_loss"],
                "mean_predicted_default": metrics["mean_predicted_default"],
                "observed_default_rate": metrics["observed_default_rate"],
                "approval_rate": metrics["approval_rate"],
                "approved_default_rate": metrics["approved_default_rate"],
                "expected_profit": metrics["expected_profit"],
                "realized_profit": metrics["realized_profit"],
                "profit_per_loan": metrics["profit_per_loan"],
                "profit_per_funded_dollar": metrics["profit_per_funded_dollar"],
                "selected_threshold": metrics["selected_threshold"],
                "selected_count": metrics["selected_count"],
            }
        )
    comparison = pd.DataFrame(rows)
    if len(comparison):
        comparison = comparison.sort_values(
            by=["expected_profit", "realized_profit", "roc_auc"],
            ascending=[False, False, False],
            na_position="last",
        ).reset_index(drop=True)
    return comparison, detail_map


def _best_probability_model(comparison: pd.DataFrame) -> str | None:
    if comparison.empty:
        return None
    eligible = comparison.loc[
        (comparison["bundle_type"] != "direct_profit")
        & comparison["approval_rate"].notna()
    ].copy()
    if eligible.empty:
        return None
    eligible["_expected_profit_rank"] = pd.to_numeric(eligible["expected_profit"], errors="coerce").fillna(-np.inf)
    eligible["_brier_rank"] = pd.to_numeric(eligible["brier_score"], errors="coerce").fillna(np.inf)
    eligible["_roc_rank"] = pd.to_numeric(eligible["roc_auc"], errors="coerce").fillna(-np.inf)
    eligible = eligible.sort_values(
        by=["_expected_profit_rank", "_brier_rank", "_roc_rank"],
        ascending=[False, True, False],
    )
    return str(eligible.iloc[0]["model_name"])


def _best_profit_model(comparison: pd.DataFrame) -> str | None:
    if comparison.empty:
        return None
    eligible = comparison.loc[comparison["bundle_type"] == "direct_profit"].copy()
    if eligible.empty:
        return None
    eligible["_realized_rank"] = pd.to_numeric(eligible["realized_profit"], errors="coerce").fillna(-np.inf)
    eligible["_expected_rank"] = pd.to_numeric(eligible["expected_profit"], errors="coerce").fillna(-np.inf)
    eligible = eligible.sort_values(by=["_realized_rank", "_expected_rank"], ascending=[False, False])
    return str(eligible.iloc[0]["model_name"])


def _decile_frame(values: pd.Series, bins: int = 10) -> pd.Series:
    data = pd.to_numeric(values, errors="coerce")
    try:
        decile = pd.qcut(data.rank(method="first"), q=min(bins, len(data.dropna())), labels=False, duplicates="drop")
        return pd.Series(decile, index=data.index).fillna(0).astype(int) + 1
    except ValueError:
        return pd.Series(1, index=data.index)


def calibration_deciles(frame: pd.DataFrame, probability: pd.Series, bundle, metrics: dict[str, Any]) -> pd.DataFrame:
    data = frame.copy()
    data["p_default"] = probability
    data["decile"] = _decile_frame(data["p_default"], bins=10)
    grouped = data.groupby("decile", dropna=False)
    rows = []
    for decile, group in grouped:
        try:
            group_metrics = policy_metrics(
                group,
                group["p_default"],
                **_policy_kwargs(bundle.policy),
            )
            approval_rate = float(group_metrics["selection_rate"])
            mean_expected_profit = float(group_metrics["mean_expected_profit"])
        except Exception:
            approval_rate = None
            mean_expected_profit = None
        rows.append(
            {
                "decile": int(decile),
                "count": int(len(group)),
                "mean_predicted_default": float(group["p_default"].mean()) if len(group) else None,
                "observed_default_rate": float(group[TARGET].mean()) if TARGET in group.columns and len(group) else None,
                "default_count": int(group[TARGET].sum()) if TARGET in group.columns else None,
                "mean_expected_profit": mean_expected_profit,
                "mean_realized_profit": float(
                    (pd.to_numeric(group["total_pymnt"], errors="coerce") - pd.to_numeric(group["funded_amnt"], errors="coerce")).mean()
                )
                if {"total_pymnt", "funded_amnt"}.issubset(group.columns)
                else None,
                "approval_rate": approval_rate,
            }
        )
    return pd.DataFrame(rows)


def policy_threshold_curve(frame: pd.DataFrame, probability: pd.Series, bundle, thresholds: list[float]) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        metrics = policy_metrics(
            frame,
            probability,
            **{**_policy_kwargs(bundle.policy), "required_return": float(threshold)},
        )
        rows.append(
            {
                "threshold": float(threshold),
                "selected_count": int(metrics["approval_count"]),
                "approval_rate": float(metrics["selection_rate"]),
                "expected_profit": float(metrics["expected_profit"]),
                "realized_profit": metrics.get("total_realized_profit"),
                "approved_default_rate": metrics.get("actual_default_rate"),
                "profit_per_funded_dollar": float(metrics["expected_return"]),
            }
        )
    return pd.DataFrame(rows)


def top_percentile_curve(
    frame: pd.DataFrame,
    probability: pd.Series,
    bundle,
    percentiles: list[float],
) -> pd.DataFrame:
    expected_profit_per_row = _expected_profit_series(frame, probability, bundle)
    ordered = frame.copy()
    ordered["expected_return"] = expected_profit_per_row / pd.to_numeric(ordered["funded_amnt"], errors="coerce")
    ordered["expected_profit"] = expected_profit_per_row
    ordered = ordered.sort_values(["expected_return", "expected_profit"], ascending=[False, False])
    rows = []
    for selected_percent in percentiles:
        selected_count = max(1, int(np.ceil(len(ordered) * float(selected_percent))))
        selected = ordered.iloc[:selected_count].copy()
        realized_profit = None
        total_funded = float(pd.to_numeric(selected["funded_amnt"], errors="coerce").sum()) if "funded_amnt" in selected.columns else None
        if {"total_pymnt", "funded_amnt"}.issubset(selected.columns):
            realized_profit = float(
                (pd.to_numeric(selected["total_pymnt"], errors="coerce") - pd.to_numeric(selected["funded_amnt"], errors="coerce")).sum()
            )
        rows.append(
            {
                "selected_percent": float(selected_percent),
                "selected_count": int(selected_count),
                "expected_profit": float(selected["expected_profit"].sum()),
                "realized_profit": realized_profit,
                "mean_realized_profit": realized_profit / selected_count if realized_profit is not None else None,
                "profit_per_funded_dollar": (float(selected["expected_profit"].sum()) / total_funded) if total_funded else None,
                "approved_default_rate": float(selected[TARGET].mean()) if TARGET in selected.columns and len(selected) else None,
            }
        )
    return pd.DataFrame(rows)


def decile_lift(frame: pd.DataFrame, probability: pd.Series, bundle) -> pd.DataFrame:
    expected_profit = _expected_profit_series(frame, probability, bundle)
    expected_return = expected_profit / pd.to_numeric(frame["funded_amnt"], errors="coerce")
    risk_decile = _decile_frame(pd.Series(probability, index=frame.index), bins=10)
    profit_decile = _decile_frame(expected_return, bins=10)
    rows = []
    for decile_type, deciles in [("risk", risk_decile), ("expected_return", profit_decile)]:
        grouped = frame.assign(decile=deciles, expected_profit=expected_profit, expected_return=expected_return).groupby("decile", dropna=False)
        cumulative_realized = 0.0
        saw_realized = False
        for decile, group in grouped:
            mean_realized = None
            total_realized = None
            if {"total_pymnt", "funded_amnt"}.issubset(group.columns):
                realized = pd.to_numeric(group["total_pymnt"], errors="coerce") - pd.to_numeric(group["funded_amnt"], errors="coerce")
                mean_realized = float(realized.mean()) if len(realized) else None
                total_realized = float(realized.sum()) if len(realized) else None
                cumulative_realized += float(realized.sum()) if len(realized) else 0.0
                saw_realized = True
            rows.append(
                {
                    "decile_type": decile_type,
                    "decile": int(decile),
                    "count": int(len(group)),
                    "observed_default_rate": float(group[TARGET].mean()) if TARGET in group.columns and len(group) else None,
                    "mean_expected_profit": float(group["expected_profit"].mean()) if len(group) else None,
                    "mean_realized_profit": mean_realized,
                    "cumulative_realized_profit": cumulative_realized if saw_realized else None,
                }
            )
    return pd.DataFrame(rows)


def cohort_backtest(frame: pd.DataFrame, probability: pd.Series, bundle, include_month: bool = True) -> pd.DataFrame:
    rows = []
    if "issue_year" not in frame.columns and "issue_quarter" not in frame.columns:
        return pd.DataFrame(columns=[
            "cohort_type",
            "cohort",
            "count",
            "observed_default_rate",
            "mean_predicted_default",
            "brier_score",
            "roc_auc",
            "approval_rate",
            "approved_default_rate",
            "expected_profit",
            "realized_profit",
        ])
    cohort_cols = [("issue_year", "year"), ("issue_quarter", "quarter")]
    if include_month and "issue_month" in frame.columns:
        cohort_cols.append(("issue_month", "month"))
    for column, cohort_type in cohort_cols:
        grouped = frame.assign(p_default=probability).groupby(column, dropna=False)
        for cohort, group in grouped:
            if len(group) == 0:
                continue
            metrics = policy_metrics(
                group,
                group["p_default"],
                **_policy_kwargs(bundle.policy),
            )
            rows.append(
                {
                    "cohort_type": cohort_type,
                    "cohort": str(cohort),
                    "count": int(len(group)),
                    "observed_default_rate": float(group[TARGET].mean()) if TARGET in group.columns else None,
                    "mean_predicted_default": float(group["p_default"].mean()),
                    "brier_score": float(np.mean((pd.to_numeric(group[TARGET], errors="coerce") - group["p_default"]) ** 2))
                    if TARGET in group.columns and len(group)
                    else None,
                    "roc_auc": _safe_auc(group[TARGET], group["p_default"]) if TARGET in group.columns else None,
                    "approval_rate": float(metrics["selection_rate"]),
                    "approved_default_rate": metrics.get("actual_default_rate"),
                    "expected_profit": float(metrics["expected_profit"]),
                    "realized_profit": metrics.get("total_realized_profit"),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_intervals(frame: pd.DataFrame, probability: pd.Series, bundle, n_bootstrap: int = EVALUATION_BOOTSTRAP_SAMPLES, random_state: int = EVALUATION_BOOTSTRAP_RANDOM_STATE) -> pd.DataFrame:
    if len(frame) == 0:
        return pd.DataFrame(columns=["metric", "estimate", "lower", "upper", "bootstrap_samples", "random_state"])
    rng = np.random.default_rng(random_state)
    sample_metrics = []
    full_metrics = policy_metrics(
        frame,
        probability,
        **_policy_kwargs(bundle.policy),
    )
    full_prob = calibration_summary(frame[TARGET], probability)
    for _ in range(int(n_bootstrap)):
        sample_idx = rng.integers(0, len(frame), len(frame))
        sample = frame.iloc[sample_idx].reset_index(drop=True)
        sample_prob = pd.Series(probability).iloc[sample_idx].reset_index(drop=True)
        metrics = policy_metrics(
            sample,
            sample_prob,
            **_policy_kwargs(bundle.policy),
        )
        summary = calibration_summary(sample[TARGET], sample_prob)
        sample_metrics.append(
            {
                "roc_auc": summary["roc_auc"],
                "pr_auc": summary["pr_auc"],
                "brier_score": summary["brier_score"],
                "expected_profit": metrics["expected_profit"],
                "realized_profit": metrics.get("total_realized_profit"),
                "approval_rate": metrics["selection_rate"],
                "default_rate_approved": metrics.get("actual_default_rate"),
            }
        )
    rows = []
    estimate_map = {
        "roc_auc": full_prob["roc_auc"],
        "pr_auc": full_prob["pr_auc"],
        "brier_score": full_prob["brier_score"],
        "expected_profit": full_metrics["expected_profit"],
        "realized_profit": full_metrics.get("total_realized_profit"),
        "approval_rate": full_metrics["selection_rate"],
        "default_rate_approved": full_metrics.get("actual_default_rate"),
    }
    for metric in sample_metrics[0].keys():
        values = pd.to_numeric(pd.Series([row[metric] for row in sample_metrics]), errors="coerce").dropna()
        rows.append(
            {
                "metric": metric,
                "estimate": estimate_map.get(metric),
                "lower": float(values.quantile(0.025)) if len(values) else None,
                "upper": float(values.quantile(0.975)) if len(values) else None,
                "bootstrap_samples": int(n_bootstrap),
                "random_state": int(random_state),
            }
        )
    return pd.DataFrame(rows)


def _top_features(feature_importance_df: pd.DataFrame, limit: int = 5) -> list[dict[str, Any]]:
    if feature_importance_df.empty:
        return []
    return feature_importance_df.sort_values("importance_drop_roc_auc", ascending=False).head(limit).to_dict(orient="records")


def _compact_metrics(metrics: dict[str, Any] | None) -> dict[str, Any] | None:
    if not metrics:
        return None
    keys = [
        "model_name",
        "calibration_method",
        "roc_auc",
        "pr_auc",
        "brier_score",
        "log_loss",
        "mean_predicted_default",
        "observed_default_rate",
        "approval_rate",
        "approved_default_rate",
        "expected_profit",
        "realized_profit",
        "profit_per_loan",
        "profit_per_funded_dollar",
        "selected_threshold",
        "selected_count",
    ]
    return {key: metrics.get(key) for key in keys if key in metrics}


def feature_importance(frame: pd.DataFrame, probability: pd.Series, bundle, random_state: int = 42) -> pd.DataFrame:
    if bundle.calibrator is None or not bundle.feature_columns:
        return pd.DataFrame(columns=["feature", "importance_drop_roc_auc", "importance_drop_expected_profit", "importance_drop_realized_profit"])
    baseline_auc = _safe_auc(frame[TARGET], probability)
    baseline_metrics = policy_metrics(
        frame,
        probability,
        **_policy_kwargs(bundle.policy),
    )
    baseline_expected = baseline_metrics["expected_profit"]
    baseline_realized = baseline_metrics.get("total_realized_profit")
    rng = np.random.default_rng(random_state)
    rows = []
    for feature in bundle.feature_columns:
        if feature not in frame.columns:
            continue
        permuted = frame.copy()
        permuted[feature] = rng.permutation(permuted[feature].to_numpy())
        raw = predict_raw_default(bundle.model, permuted, bundle.feature_columns)
        permuted_probability = pd.Series(bundle.calibrator.predict(raw), index=frame.index)
        permuted_auc = _safe_auc(permuted[TARGET], permuted_probability)
        permuted_metrics = policy_metrics(
            permuted,
            permuted_probability,
            **_policy_kwargs(bundle.policy),
        )
        rows.append(
            {
                "feature": feature,
                "importance_drop_roc_auc": None if baseline_auc is None or permuted_auc is None else float(baseline_auc - permuted_auc),
                "importance_drop_expected_profit": float(baseline_expected - permuted_metrics["expected_profit"]),
                "importance_drop_realized_profit": (
                    None
                    if baseline_realized is None or permuted_metrics.get("total_realized_profit") is None
                    else float(baseline_realized - permuted_metrics.get("total_realized_profit"))
                ),
            }
        )
    return pd.DataFrame(rows)


def _plot_series(path: Path, x, ys: list[tuple[str, Any]], title: str, xlabel: str, ylabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    if len(x) and any(len(y[1]) for y in ys):
        for label, values in ys:
            ax.plot(x, values, marker="o", label=label)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_calibration_curve(path: Path, calibration_df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.55", linewidth=1)
    if len(calibration_df):
        ax.plot(calibration_df["mean_predicted_default"], calibration_df["observed_default_rate"], marker="o")
    else:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
    ax.set_xlabel("Mean predicted default rate")
    ax.set_ylabel("Observed default rate")
    ax.set_title("Calibration curve")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_threshold_curve(path: Path, curve_df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    if len(curve_df):
        ax.plot(curve_df["threshold"], curve_df["expected_profit"], marker="o", label="Expected profit")
        if curve_df["realized_profit"].notna().any():
            ax.plot(curve_df["threshold"], curve_df["realized_profit"], marker="o", label="Realized profit")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
    ax.set_xlabel("Required return threshold")
    ax.set_ylabel("Profit")
    ax.set_title("Policy threshold curve")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_cumulative_profit(path: Path, frame: pd.DataFrame, probability: pd.Series, bundle, use_realized: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    expected_profit = _expected_profit_series(frame, probability, bundle).to_numpy(dtype=float)
    expected_return = expected_profit / pd.to_numeric(frame["funded_amnt"], errors="coerce").to_numpy(dtype=float)
    order = np.argsort(-expected_return)
    ordered = frame.iloc[order].copy()
    ordered_expected = expected_profit[order].cumsum()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, len(ordered_expected) + 1), ordered_expected, label="Cumulative expected profit")
    if use_realized and {"total_pymnt", "funded_amnt"}.issubset(ordered.columns):
        realized = (pd.to_numeric(ordered["total_pymnt"], errors="coerce") - pd.to_numeric(ordered["funded_amnt"], errors="coerce")).to_numpy(dtype=float)
        ax.plot(np.arange(1, len(realized) + 1), realized.cumsum(), label="Cumulative realized profit")
    ax.set_xlabel("Loans selected")
    ax.set_ylabel("Cumulative profit")
    ax.set_title("Cumulative profit curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_decile_profit(path: Path, decile_df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    if len(decile_df):
        frame = decile_df.loc[decile_df["decile_type"] == "expected_return"]
        if frame.empty:
            frame = decile_df.loc[decile_df["decile_type"] == "risk"]
        ax.bar(frame["decile"].astype(str), frame["mean_realized_profit"].fillna(frame["mean_expected_profit"]))
    else:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
    ax.set_xlabel("Expected-return decile")
    ax.set_ylabel("Mean profit")
    ax.set_title("Profit by expected-return decile")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_risk_decile_default_rate(path: Path, calibration_df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    if len(calibration_df):
        ax.bar(calibration_df["decile"].astype(str), calibration_df["observed_default_rate"])
    else:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
    ax.set_xlabel("Risk decile")
    ax.set_ylabel("Observed default rate")
    ax.set_title("Default rate by risk decile")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _warnings_for_frame(frame: pd.DataFrame, bundle) -> list[str]:
    warnings = [
        "accepted-loan selection bias remains; rejected-applicant labels are unavailable",
        "profit estimates are scenario-based and depend on the stated LGD and policy assumptions",
        "locked test results are not generated by default",
    ]
    if "total_pymnt" not in frame.columns:
        warnings.append("realized profit is unavailable because total_pymnt is missing")
    if "issue_dt" not in frame.columns:
        warnings.append("cohort backtests are limited because issue dates are missing")
    if bundle.model_type == "rejected_style":
        warnings.append("rejected-style scoring is review-only and not a fair-lending validation")
    return warnings


def run_evaluation_suite(
    csv_path: str | Path = ACCEPTED_CSV,
    output_dir: str | Path = REPORT_DIR,
    stage: str = "validation",
    bundle_paths: dict[str, str | Path] | None = None,
    sample: int | None = None,
    n_bootstrap: int = EVALUATION_BOOTSTRAP_SAMPLES,
    random_state: int = EVALUATION_BOOTSTRAP_RANDOM_STATE,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    bundle_paths = bundle_paths or {
        "accepted": DEFAULT_ACCEPTED_BUNDLE,
        "rejected_style": DEFAULT_REJECTED_STYLE_BUNDLE,
        "direct_profit": DEFAULT_PROFIT_BUNDLE,
    }

    primary_path = Path(bundle_paths.get("accepted", DEFAULT_ACCEPTED_BUNDLE))
    primary_bundle = load_model_bundle(primary_path)
    if primary_bundle.metadata.get("source_fingerprint"):
        actual = file_fingerprint(csv_path)
        expected = primary_bundle.metadata["source_fingerprint"]
        if expected.get("sha256") != actual.get("sha256") or expected.get("size_bytes") != actual.get("size_bytes"):
            raise ValueError("source CSV fingerprint does not match the accepted bundle")

    primary_frame, stage_summary = _prepare_validation_frame(primary_bundle, csv_path, stage=stage, sample=sample)

    results: list[EvaluationBundleResult] = []
    for name, bundle_path in bundle_paths.items():
        path = Path(bundle_path)
        if not path.exists():
            continue
        bundle = load_model_bundle(path)
        if bundle.metadata.get("source_fingerprint"):
            expected = bundle.metadata["source_fingerprint"]
            actual = file_fingerprint(csv_path)
            if expected.get("sha256") != actual.get("sha256") or expected.get("size_bytes") != actual.get("size_bytes"):
                continue
        try:
            if bundle.model_type == "direct_profit":
                results.append(_profit_bundle_result(name, path, bundle, primary_frame))
            else:
                if bundle.model_type == "rejected_style":
                    evaluation_frame = map_accepted_to_rejected_style(primary_frame)
                else:
                    evaluation_frame = primary_frame
                results.append(_probability_bundle_result(name, path, bundle, evaluation_frame))
        except Exception:
            continue

    comparison_df, detail_map = _model_comparison_rows(results)
    if comparison_df.empty:
        raise ValueError("no compatible bundles were available for evaluation")

    best_probability_name = _best_probability_model(comparison_df)
    best_profit_name = _best_profit_model(comparison_df)
    best_probability = next((result for result in results if result.name == best_probability_name), None)
    best_profit = next((result for result in results if result.name == best_profit_name), None)
    if best_probability is None:
        raise ValueError("no probability model was available for calibration and policy evaluation")

    calibration_df = calibration_deciles(
        primary_frame,
        best_probability.probability,
        best_probability.bundle,
        detail_map[best_probability_name],
    )
    expected_return_values = _expected_return_series(primary_frame, best_probability.probability, best_probability.bundle)
    thresholds = [float(x) for x in np.unique(np.round(np.quantile(expected_return_values, np.linspace(0.0, 1.0, 11)), 6))]
    threshold_curve_df = policy_threshold_curve(primary_frame, best_probability.probability, best_probability.bundle, list(thresholds))
    percentiles = [round(x, 2) for x in np.linspace(0.1, 1.0, 10)]
    top_percentile_df = top_percentile_curve(primary_frame, best_probability.probability, best_probability.bundle, percentiles)
    decile_lift_df = decile_lift(primary_frame, best_probability.probability, best_probability.bundle)
    cohort_df = cohort_backtest(primary_frame, best_probability.probability, best_probability.bundle, include_month=True)
    bootstrap_df = bootstrap_intervals(primary_frame, best_probability.probability, best_probability.bundle, n_bootstrap=n_bootstrap, random_state=random_state)
    feature_df = feature_importance(primary_frame, best_probability.probability, best_probability.bundle, random_state=random_state)

    warnings = _warnings_for_frame(primary_frame, best_probability.bundle)
    if best_profit is not None and best_profit.bundle.model_type == "direct_profit":
        warnings.append("direct-profit selection is a separate regression problem, not a calibrated default model")
    if sample:
        warnings.append("evaluation ran on a smoke sample")

    output_files = {
        "model_comparison": output_dir / "model_comparison.csv",
        "calibration_deciles": output_dir / "calibration_deciles.csv",
        "policy_threshold_curve": output_dir / "policy_threshold_curve.csv",
        "top_percentile_curve": output_dir / "top_percentile_curve.csv",
        "decile_lift": output_dir / "decile_lift.csv",
        "cohort_backtest": output_dir / "cohort_backtest.csv",
        "feature_importance": output_dir / "feature_importance.csv",
        "evaluation_summary": output_dir / "evaluation_summary.json",
    }
    comparison_df.to_csv(output_files["model_comparison"], index=False)
    calibration_df.to_csv(output_files["calibration_deciles"], index=False)
    threshold_curve_df.to_csv(output_files["policy_threshold_curve"], index=False)
    top_percentile_df.to_csv(output_files["top_percentile_curve"], index=False)
    decile_lift_df.to_csv(output_files["decile_lift"], index=False)
    cohort_df.to_csv(output_files["cohort_backtest"], index=False)
    feature_df.to_csv(output_files["feature_importance"], index=False)

    _plot_calibration_curve(figures_dir / "calibration_curve.png", calibration_df)
    _plot_threshold_curve(figures_dir / "profit_threshold_curve.png", threshold_curve_df)
    _plot_cumulative_profit(figures_dir / "cumulative_profit_curve.png", primary_frame, best_probability.probability, best_probability.bundle)
    _plot_decile_profit(figures_dir / "profit_by_expected_return_decile.png", decile_lift_df)
    _plot_risk_decile_default_rate(figures_dir / "default_rate_by_risk_decile.png", calibration_df)

    evaluation_summary = {
        "stage": stage,
        "csv_path": _repo_relative(csv_path),
        "primary_bundle": _repo_relative(primary_path),
        "best_probability_model": best_probability_name,
        "best_probability_metrics": _compact_metrics(detail_map[best_probability_name]),
        "best_profit_model": best_profit_name,
        "best_profit_metrics": _compact_metrics(detail_map.get(best_profit_name)),
        "validation_row_count": int(len(primary_frame)),
        "validation_default_rate": float(primary_frame[TARGET].mean()) if TARGET in primary_frame.columns and len(primary_frame) else None,
        "split_summary": stage_summary["split_summary"],
        "feature_groups": {
            "numeric_risk": len([c for c in ACCEPTED_NUMERIC_RISK_FEATURES if c in best_probability.bundle.feature_columns]),
            "categorical_risk": len([c for c in ACCEPTED_CATEGORICAL_RISK_FEATURES if c in best_probability.bundle.feature_columns]),
            "pricing_fields": len([c for c in POST_PRICING_FIELDS if c in best_probability.bundle.feature_columns]),
            "product_mode": best_probability.bundle.metadata.get("product_mode", PRODUCT_MODE_POST_PRICING),
        },
        "generated_files": {name: _repo_relative(path) for name, path in output_files.items()},
        "generated_figures": {
            "calibration_curve": _repo_relative(figures_dir / "calibration_curve.png"),
            "profit_threshold_curve": _repo_relative(figures_dir / "profit_threshold_curve.png"),
            "cumulative_profit_curve": _repo_relative(figures_dir / "cumulative_profit_curve.png"),
            "profit_by_expected_return_decile": _repo_relative(figures_dir / "profit_by_expected_return_decile.png"),
            "default_rate_by_risk_decile": _repo_relative(figures_dir / "default_rate_by_risk_decile.png"),
        },
        "warnings": warnings,
        "top_features": _top_features(feature_df),
    }
    output_files["evaluation_summary"].write_text(json.dumps(evaluation_summary, indent=2, default=str), encoding="utf-8")
    return output_files


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Generate compact validation evaluation reports.")
    parser.add_argument("--csv", default=ACCEPTED_CSV)
    parser.add_argument("--output-dir", default=REPORT_DIR)
    parser.add_argument("--stage", choices=["validation", "test"], default="validation")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--accepted-bundle", default=DEFAULT_ACCEPTED_BUNDLE)
    parser.add_argument("--rejected-style-bundle", default=DEFAULT_REJECTED_STYLE_BUNDLE)
    parser.add_argument("--profit-bundle", default=DEFAULT_PROFIT_BUNDLE)
    args = parser.parse_args(argv)

    paths = run_evaluation_suite(
        csv_path=args.csv,
        output_dir=args.output_dir,
        stage=args.stage,
        bundle_paths={
            "accepted": args.accepted_bundle,
            "rejected_style": args.rejected_style_bundle,
            "direct_profit": args.profit_bundle,
        },
        sample=args.sample,
    )
    print(json.dumps({k: _repo_relative(v) for k, v in paths.items()}, indent=2))
    return 0
