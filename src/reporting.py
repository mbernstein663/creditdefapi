from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .calibration import calibration_summary
from .config import (
    ACCEPTED_CATEGORICAL_RISK_FEATURES,
    ACCEPTED_NUMERIC_RISK_FEATURES,
    DEFAULT_LGD,
    DEFAULT_REQUIRED_RETURN,
    EVALUATION_BOOTSTRAP_RANDOM_STATE,
    EVALUATION_BOOTSTRAP_SAMPLES,
    MIN_PROXY_GROUP_SIZE,
    PRODUCT_MODE_POST_PRICING,
    PROFIT_INPUT_COLUMNS,
    REPORT_DIR,
    TARGET,
)
from .profit import policy_metrics, policy_sensitivity


def _safe_auc(y_true, p_default):
    y = pd.Series(y_true).dropna().astype(int)
    p = pd.Series(p_default).loc[y.index].astype(float)
    if y.nunique() < 2:
        return None
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, p))


def _safe_pr_auc(y_true, p_default):
    y = pd.Series(y_true).dropna().astype(int)
    p = pd.Series(p_default).loc[y.index].astype(float)
    if y.nunique() < 2:
        return None
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y, p))


def calibration_table(frame: pd.DataFrame, p_default, bins: int = 10) -> pd.DataFrame:
    summary = calibration_summary(frame[TARGET], p_default, bins=bins)
    return pd.DataFrame(summary["deciles"])


def _policy_kwargs(policy: dict) -> dict:
    return {
        "lgd": policy.get("lgd", DEFAULT_LGD),
        "required_return": policy.get("required_return", DEFAULT_REQUIRED_RETURN),
        "use_npv": bool(policy.get("use_npv")),
        "annual_discount_rate": float(policy.get("annual_discount_rate", 0.08)),
        "servicing_cost_rate": float(policy.get("servicing_cost_rate", 0.0)),
        "good_profit_haircut": float(policy.get("good_profit_haircut", 1.0)),
    }


def bootstrap_intervals(
    frame: pd.DataFrame,
    p_default,
    policy: dict,
    n_bootstrap: int = EVALUATION_BOOTSTRAP_SAMPLES,
    random_state: int = EVALUATION_BOOTSTRAP_RANDOM_STATE,
) -> pd.DataFrame:
    if len(frame) == 0:
        return pd.DataFrame(columns=["metric", "estimate", "lower", "upper", "bootstrap_samples", "random_state"])

    full_calibration = calibration_summary(frame[TARGET], p_default)
    full_metrics = policy_metrics(
        frame,
        p_default,
        **_policy_kwargs(policy),
    )

    frame = frame.reset_index(drop=True)
    p_default = pd.Series(p_default).reset_index(drop=True)
    rng = np.random.default_rng(random_state)
    rows = []
    for _ in range(int(n_bootstrap)):
        sample_idx = rng.integers(0, len(frame), len(frame))
        sample = frame.iloc[sample_idx].reset_index(drop=True)
        p_sample = p_default.iloc[sample_idx].to_numpy()
        calibration = calibration_summary(sample[TARGET], p_sample)
        metrics = policy_metrics(
            sample,
            p_sample,
            **_policy_kwargs(policy),
        )
        rows.append(
            {
                "roc_auc": calibration["roc_auc"],
                "pr_auc": calibration["pr_auc"],
                "brier_score": calibration["brier_score"],
                "expected_profit": metrics["expected_profit"],
                "expected_return": metrics["expected_return"],
                "approval_rate": metrics["selection_rate"],
                "actual_default_rate_approved": metrics.get("actual_default_rate"),
                "total_realized_profit": metrics.get("total_realized_profit"),
                "mean_realized_profit": metrics.get("mean_realized_profit"),
            }
        )

    summary_rows = []
    estimate_map = {
        "roc_auc": full_calibration.get("roc_auc"),
        "pr_auc": full_calibration.get("pr_auc"),
        "brier_score": full_calibration.get("brier_score"),
        "expected_profit": full_metrics["expected_profit"],
        "expected_return": full_metrics["expected_return"],
        "approval_rate": full_metrics["selection_rate"],
        "actual_default_rate_approved": full_metrics.get("actual_default_rate"),
        "total_realized_profit": full_metrics.get("total_realized_profit"),
        "mean_realized_profit": full_metrics.get("mean_realized_profit"),
    }
    for column in rows[0].keys() if rows else []:
        values = pd.to_numeric(pd.Series([row[column] for row in rows]), errors="coerce").dropna()
        summary_rows.append(
            {
                "metric": column,
                "estimate": estimate_map.get(column),
                "lower": float(values.quantile(0.025)) if len(values) else None,
                "upper": float(values.quantile(0.975)) if len(values) else None,
                "bootstrap_samples": int(n_bootstrap),
                "random_state": int(random_state),
            }
        )
    return pd.DataFrame(summary_rows)


def cohort_backtest(frame: pd.DataFrame, p_default, policy: dict, include_month: bool = True) -> pd.DataFrame:
    data = frame.copy()
    data["p_default"] = pd.Series(p_default, index=data.index)
    rows = []
    cohorts = [
        ("issue_year", data.get("issue_year")),
        ("issue_quarter", data.get("issue_quarter")),
    ]
    if include_month:
        cohorts.append(("issue_month", data.get("issue_month")))
    for cohort_type, series in cohorts:
        if series is None:
            continue
        grouped = data.groupby(series, dropna=False)
        for cohort_value, group in grouped:
            y = group[TARGET] if TARGET in group.columns else pd.Series(dtype=float)
            p = group["p_default"]
            approval = policy_metrics(
                group,
                p,
                **_policy_kwargs(policy),
            )
            rows.append(
                {
                    "cohort_type": cohort_type,
                    "cohort_value": str(cohort_value),
                    "rows": int(len(group)),
                    "observed_default_rate": float(y.mean()) if len(y) else None,
                    "mean_predicted_default_rate": float(p.mean()) if len(p) else None,
                    "brier_score": float(np.mean((p.to_numpy(dtype=float) - y.to_numpy(dtype=float)) ** 2))
                    if len(y)
                    else None,
                    "auc": _safe_auc(y, p),
                    "approval_rate": float(approval["selection_rate"]),
                    "expected_profit": float(approval["expected_profit"]),
                    "realized_profit": approval.get("total_realized_profit"),
                }
            )
    return pd.DataFrame(rows)


def proxy_risk_diagnostics(
    frame: pd.DataFrame,
    p_default,
    policy: dict,
    min_group_size: int = MIN_PROXY_GROUP_SIZE,
) -> pd.DataFrame:
    data = frame.copy()
    data["p_default"] = pd.Series(p_default, index=data.index)
    groups = [
        column
        for column in ["addr_state", "zip_code", "home_ownership", "emp_length", "verification_status", "purpose"]
        if column in data.columns
    ]
    rows = []
    for column in groups:
        for value, group in data.groupby(column, dropna=False):
            count = int(len(group))
            suppressed = count < int(min_group_size)
            approval = policy_metrics(
                group,
                group["p_default"],
                **_policy_kwargs(policy),
            )
            total_expected_profit = float(approval["expected_profit"])
            rows.append(
                {
                    "field": column,
                    "group_value": str(value),
                    "count": count,
                    "suppressed": suppressed,
                    "approval_rate": None if suppressed else float(approval["selection_rate"]),
                    "observed_default_rate": None if suppressed or TARGET not in group.columns else float(group[TARGET].mean()),
                    "mean_predicted_default_rate": None if suppressed else float(group["p_default"].mean()),
                    "average_expected_profit": None
                    if suppressed or not approval["approval_count"]
                    else float(total_expected_profit / approval["approval_count"]),
                    "total_expected_profit": None if suppressed else total_expected_profit,
                }
            )
    return pd.DataFrame(rows)


def _feature_groups(feature_columns: list[str]) -> dict[str, list[str]]:
    feature_set = set(feature_columns)
    known = set(ACCEPTED_NUMERIC_RISK_FEATURES + ACCEPTED_CATEGORICAL_RISK_FEATURES + PROFIT_INPUT_COLUMNS)
    return {
        "numeric_risk": [c for c in ACCEPTED_NUMERIC_RISK_FEATURES if c in feature_set],
        "categorical_risk": [c for c in ACCEPTED_CATEGORICAL_RISK_FEATURES if c in feature_set],
        "profit_inputs": [c for c in PROFIT_INPUT_COLUMNS if c in feature_set],
        "pricing_fields": [c for c in ["grade", "sub_grade", "int_rate", "initial_list_status"] if c in feature_set],
        "other": [c for c in feature_columns if c not in known],
    }


def _model_card(
    bundle,
    stage_summary: dict,
    calibration_df: pd.DataFrame,
    policy_df: pd.DataFrame,
    cohort_df: pd.DataFrame,
    bootstrap_df: pd.DataFrame,
    proxy_df: pd.DataFrame,
    stage_label: str,
    include_sensitivity: bool,
) -> str:
    metadata = bundle.metadata or {}
    feature_groups = _feature_groups(list(bundle.feature_columns))
    calibration_rows = calibration_df.to_dict(orient="records")
    lines = [
        "# Model Card",
        "",
        "## Data Source",
        f"- Source fingerprint: `{json.dumps(metadata.get('source_fingerprint', {}), default=str)}`",
        f"- Target summary: `{json.dumps(metadata.get('target_summary', {}), default=str)}`",
        "",
        f"- Model type: `{bundle.model_type}`",
        f"- Target mode: `{metadata.get('target_mode', 'resolved_default')}`",
        f"- Product mode: `{metadata.get('product_mode', PRODUCT_MODE_POST_PRICING)}`",
        f"- Calibration method: `{metadata.get('calibration_method', 'isotonic')}`",
        f"- Selected policy: `{json.dumps(bundle.policy, default=str)}`",
        f"- Source rows: `{stage_summary.get('row_count')}`",
        f"- Included rows: `{metadata.get('target_summary', {}).get('included_rows')}`",
        f"- Excluded rows: `{metadata.get('target_summary', {}).get('excluded_rows')}`",
        "",
        "## Split Summary",
    ]
    for split in stage_summary.get("split_summary", []):
        lines.append(
            f"- {split['split']}: rows={split['rows']}, default_rate={split['default_rate']}, "
            f"date_range={split['date_min']} to {split['date_max']}"
        )
    lines += [
        "",
        "## Features",
        f"- Feature count: `{len(bundle.feature_columns)}`",
        f"- Numeric risk features: `{', '.join(feature_groups['numeric_risk'])}`",
        f"- Categorical risk features: `{', '.join(feature_groups['categorical_risk'])}`",
        f"- Pricing fields: `{', '.join(feature_groups['pricing_fields'])}`",
        f"- Leakage exclusions: `{', '.join(sorted(metadata.get('forbidden_feature_columns', [])))}`",
        "",
        f"## {stage_label} Metrics",
    ]
    for key in ["roc_auc", "pr_auc", "brier_score", "mean_predicted_default", "actual_default_rate"]:
        if key in stage_summary:
            lines.append(f"- {key}: `{stage_summary[key]}`")
    lines += ["", "## Calibration Table"]
    lines.append("| decile | count | mean_predicted_default | observed_default_rate |")
    lines.append("| --- | --- | --- | --- |")
    for row in calibration_rows:
        lines.append(
            f"| {row.get('decile')} | {row.get('count')} | {row.get('mean_predicted_default')} | {row.get('observed_default_rate')} |"
        )
    lines += [
        "",
        "## Policy",
        f"- Approval count: `{policy_df.iloc[0]['approval_count'] if len(policy_df) else None}`",
        f"- Selection rate: `{policy_df.iloc[0]['selection_rate'] if len(policy_df) else None}`",
        f"- Expected profit: `{policy_df.iloc[0]['expected_profit'] if len(policy_df) else None}`",
        f"- Realized profit: `{policy_df.iloc[0].get('total_realized_profit') if len(policy_df) else None}`",
    ]
    if include_sensitivity:
        lines += [
            "",
            "## Sensitivity",
            policy_df.to_csv(index=False),
        ]
    lines += [
        "",
        "## Cohort Backtest",
        cohort_df.to_csv(index=False),
        "",
        "## Bootstrap Intervals",
        bootstrap_df.to_csv(index=False),
        "",
        "## Proxy Risk Diagnostics",
        proxy_df.to_csv(index=False),
        "",
        "## Known Limitations",
        "- accepted-loan selection bias",
        "- rejected-loan labels unavailable",
        "- resolved-outcome or horizon-label limitations",
        "- simplified profit assumptions",
        "- not production underwriting",
        "- no fair-lending approval",
    ]
    return "\n".join(lines)


def generate_report_suite(
    bundle,
    frame: pd.DataFrame,
    p_default,
    output_dir: str | Path = REPORT_DIR,
    stage_summary: dict | None = None,
    stage: str = "validation",
    include_sensitivity: bool = True,
    n_bootstrap: int = EVALUATION_BOOTSTRAP_SAMPLES,
    random_state: int = EVALUATION_BOOTSTRAP_RANDOM_STATE,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_summary = stage_summary or {}
    calibration_df = calibration_table(frame, p_default)
    policy_df = pd.DataFrame(
        [
            policy_metrics(
                frame,
                p_default,
                **_policy_kwargs(bundle.policy),
            )
        ]
    )
    sensitivity_df = (
        pd.DataFrame(
            policy_sensitivity(
                frame,
                p_default,
                lgds=(0.60, 0.75, 1.00),
                required_returns=(0.00, 0.05, 0.10),
                good_profit_haircuts=(float(bundle.policy.get("good_profit_haircut", 1.0)),),
                annual_discount_rates=tuple(dict.fromkeys((0.08, float(bundle.policy.get("annual_discount_rate", 0.08))))),
            )
        )
        if include_sensitivity
        else pd.DataFrame()
    )
    cohort_df = cohort_backtest(frame, p_default, bundle.policy)
    bootstrap_df = bootstrap_intervals(frame, p_default, bundle.policy, n_bootstrap=n_bootstrap, random_state=random_state)
    proxy_df = proxy_risk_diagnostics(frame, p_default, bundle.policy)
    evaluation_summary = {
        "bundle_type": bundle.model_type,
        "stage": stage,
        "target_mode": bundle.metadata.get("target_mode", "resolved_default"),
        "product_mode": bundle.metadata.get("product_mode", PRODUCT_MODE_POST_PRICING),
        "calibration_method": bundle.metadata.get("calibration_method", "isotonic"),
        "source_fingerprint": bundle.metadata.get("source_fingerprint"),
        "target_summary": bundle.metadata.get("target_summary"),
        "policy": bundle.policy,
        "calibration": calibration_df.to_dict(orient="records"),
        "policy_summary": policy_df.to_dict(orient="records"),
        "sensitivity_summary": sensitivity_df.to_dict(orient="records") if include_sensitivity else [],
        "cohort_backtest_rows": int(len(cohort_df)),
        "bootstrap_rows": int(len(bootstrap_df)),
        "proxy_rows": int(len(proxy_df)),
        "stage_summary": stage_summary,
    }
    paths = {
        "evaluation_summary": output_dir / "evaluation_summary.json",
        "calibration_table": output_dir / "calibration_table.csv",
        "policy_summary": output_dir / "policy_summary.csv",
        "sensitivity_summary": output_dir / "sensitivity_summary.csv",
        "cohort_backtest": output_dir / "cohort_backtest.csv",
        "bootstrap_intervals": output_dir / "bootstrap_intervals.csv",
        "proxy_risk_diagnostics": output_dir / "proxy_risk_diagnostics.csv",
        "fairness_caveat": output_dir / "fairness_caveat.md",
        "model_card": output_dir / "model_card.md",
    }
    paths["evaluation_summary"].write_text(json.dumps(evaluation_summary, indent=2, default=str), encoding="utf-8")
    calibration_df.to_csv(paths["calibration_table"], index=False)
    policy_df.to_csv(paths["policy_summary"], index=False)
    if include_sensitivity:
        sensitivity_df.to_csv(paths["sensitivity_summary"], index=False)
    else:
        paths["sensitivity_summary"].write_text("", encoding="utf-8")
    cohort_df.to_csv(paths["cohort_backtest"], index=False)
    bootstrap_df.to_csv(paths["bootstrap_intervals"], index=False)
    proxy_df.to_csv(paths["proxy_risk_diagnostics"], index=False)
    paths["fairness_caveat"].write_text(
        "# Fairness Caveat\n\nThese are proxy-risk diagnostics only. They are not a fair-lending validation, compliance test, or approval justification.\n",
        encoding="utf-8",
    )
    stage_summary = dict(stage_summary)
    stage_summary.setdefault("row_count", len(frame))
    stage_summary.setdefault("split_summary", bundle.metadata.get("split_summary", []))
    stage_summary.update(calibration_df.iloc[0].to_dict() if len(calibration_df) else {})
    paths["model_card"].write_text(
        _model_card(
            bundle,
            stage_summary,
            calibration_df,
            policy_df,
            cohort_df,
            bootstrap_df,
            proxy_df,
            stage_label=stage.capitalize(),
            include_sensitivity=include_sensitivity,
        ),
        encoding="utf-8",
    )
    return paths
