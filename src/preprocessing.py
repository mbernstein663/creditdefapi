from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd

from .config import (
    ACCEPTED_TO_REJECTED_FEATURE_MAP,
    ACCEPTED_CATEGORICAL_RISK_FEATURES,
    ACCEPTED_NUMERIC_RISK_FEATURES,
    BAD_STATUSES,
    FORBIDDEN_FEATURE_COLUMNS,
    GOOD_STATUSES,
    DEFAULT_TARGET_DATE_COLUMNS,
    DEFAULT_TARGET_HORIZON_MONTHS,
    DEFAULT_TARGET_MODE,
    PRE_UNDERWRITING_FORBIDDEN_FIELDS,
    PROFIT_INPUT_COLUMNS,
    PRODUCT_MODE_POST_PRICING,
    PRODUCT_MODE_PRE_UNDERWRITING,
    REJECTED_STYLE_NUMERIC_RISK_FEATURES,
    REJECTED_RAW_ALIASES,
    TARGET,
    TARGET_MODES,
    UNRESOLVED_STATUSES,
)


def parse_percent(value):
    if pd.isna(value):
        return pd.NA
    if isinstance(value, str):
        value = value.strip().replace("%", "")
    return pd.to_numeric(value, errors="coerce")


def parse_term_months(value):
    if pd.isna(value):
        return pd.NA
    match = re.search(r"\d+", str(value))
    return int(match.group(0)) if match else pd.NA


def parse_date(value):
    if pd.isna(value):
        return pd.NaT
    return pd.to_datetime(value, format="%b-%Y", errors="coerce")


def _target_summary(mode: str, included: int, excluded: int, total: int, extra: dict | None = None) -> dict:
    summary = {
        "mode": mode,
        "included_rows": int(included),
        "excluded_rows": int(excluded),
        "total_rows": int(total),
        "included_rate": float(included / total) if total else 0.0,
    }
    if extra:
        summary.update(extra)
    return summary


def _default_within_horizon_target(df: pd.DataFrame, horizon_months: int, issue_date_column: str, last_payment_date_column: str, loan_status_column: str) -> tuple[pd.DataFrame, dict]:
    required = [issue_date_column, last_payment_date_column, loan_status_column]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing horizon target columns: {', '.join(missing)}")
    out = df.copy()
    issue_dt = out[issue_date_column].map(parse_date)
    last_payment_dt = out[last_payment_date_column].map(parse_date)
    status = out[loan_status_column].fillna("").astype(str).str.strip()
    known = GOOD_STATUSES | BAD_STATUSES | UNRESOLVED_STATUSES
    unknown = sorted(set(status.unique()) - known)
    if unknown:
        raise ValueError(f"unknown loan_status values: {unknown}")
    horizon = pd.to_timedelta(int(horizon_months) * 30, unit="D")
    cutoff = issue_dt + horizon
    observed = last_payment_dt.notna()
    enough_observation = observed & last_payment_dt.ge(cutoff)
    bad = status.isin(BAD_STATUSES) & observed & last_payment_dt.le(cutoff)
    good = status.isin(GOOD_STATUSES) & enough_observation
    included = bad | good
    out = out.loc[included].copy()
    out[TARGET] = bad.loc[included].astype(int)
    summary = _target_summary(
        "default_within_horizon",
        len(out),
        len(df) - len(out),
        len(df),
        {
            "horizon_months": int(horizon_months),
            "issue_date_column": issue_date_column,
            "last_payment_date_column": last_payment_date_column,
            "loan_status_column": loan_status_column,
            "bad_label_count": int(bad.sum()),
            "good_label_count": int(good.sum()),
        },
    )
    return out, summary


def construct_target(
    df: pd.DataFrame,
    mode: str = DEFAULT_TARGET_MODE,
    target_config: dict | None = None,
    return_summary: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    config = {
        **DEFAULT_TARGET_DATE_COLUMNS,
        "horizon_months": DEFAULT_TARGET_HORIZON_MONTHS,
    }
    if target_config:
        config.update(target_config)
    if mode not in TARGET_MODES:
        raise ValueError(f"unknown target mode: {mode}")
    if mode == "resolved_default":
        if "loan_status" not in df.columns:
            raise ValueError("accepted loan data must include loan_status")

        out = df.copy()
        status = out["loan_status"].fillna("").astype(str).str.strip()
        known = GOOD_STATUSES | BAD_STATUSES | UNRESOLVED_STATUSES
        unknown = sorted(set(status.unique()) - known)
        if unknown:
            raise ValueError(f"unknown loan_status values: {unknown}")
        resolved = status.isin(GOOD_STATUSES | BAD_STATUSES)
        out = out.loc[resolved].copy()
        status = status.loc[resolved]
        out[TARGET] = status.isin(BAD_STATUSES).astype(int)
        summary = _target_summary(mode, len(out), len(df) - len(out), len(df))
    else:
        out, summary = _default_within_horizon_target(
            df,
            horizon_months=int(config["horizon_months"]),
            issue_date_column=config["issue_date_column"],
            last_payment_date_column=config["last_payment_date_column"],
            loan_status_column=config["loan_status_column"],
        )
    if return_summary:
        return out, summary
    out.attrs["target_summary"] = summary
    return out


def prepare_accepted_loans(
    df: pd.DataFrame,
    target_mode: str = DEFAULT_TARGET_MODE,
    target_config: dict | None = None,
    return_summary: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    result = construct_target(df, mode=target_mode, target_config=target_config, return_summary=True)
    out, summary = result
    if "term" in out.columns and "term_months" not in out.columns:
        out["term_months"] = out["term"].map(parse_term_months)
    if "issue_d" in out.columns:
        out["issue_dt"] = out["issue_d"].map(parse_date)
        out["issue_year"] = out["issue_dt"].dt.year
        out["issue_quarter"] = out["issue_dt"].dt.to_period("Q").astype(str)
        out["issue_month"] = out["issue_dt"].dt.to_period("M").astype(str)
    if "last_pymnt_d" in out.columns:
        out["last_pymnt_dt"] = out["last_pymnt_d"].map(parse_date)

    numeric_columns = set(ACCEPTED_NUMERIC_RISK_FEATURES + PROFIT_INPUT_COLUMNS)
    for column in numeric_columns & set(out.columns):
        if column in {"int_rate", "revol_util"}:
            out[column] = pd.to_numeric(out[column].map(parse_percent), errors="coerce")
        else:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    if return_summary:
        return out, summary
    out.attrs["target_summary"] = summary
    return out


def feature_columns_for_product_mode(product_mode: str = PRODUCT_MODE_POST_PRICING) -> list[str]:
    base = ACCEPTED_NUMERIC_RISK_FEATURES + ACCEPTED_CATEGORICAL_RISK_FEATURES
    if product_mode == PRODUCT_MODE_POST_PRICING:
        return list(base)
    if product_mode == PRODUCT_MODE_PRE_UNDERWRITING:
        forbidden = set(PRE_UNDERWRITING_FORBIDDEN_FIELDS)
        return [column for column in base if column not in forbidden]
    raise ValueError(f"unknown product mode: {product_mode}")


def ensure_no_forbidden_features(feature_columns: Iterable[str], product_mode: str = PRODUCT_MODE_POST_PRICING) -> None:
    forbidden = set(FORBIDDEN_FEATURE_COLUMNS)
    if product_mode == PRODUCT_MODE_PRE_UNDERWRITING:
        forbidden |= set(PRE_UNDERWRITING_FORBIDDEN_FIELDS)
    forbidden = sorted(set(feature_columns) & forbidden)
    if forbidden:
        raise ValueError(f"forbidden model features: {', '.join(forbidden)}")


def split_chronological(
    df: pd.DataFrame,
    date_col: str = "issue_dt",
    ratios: tuple[float, float, float, float] = (0.6, 0.15, 0.15, 0.10),
) -> dict[str, pd.DataFrame]:
    if date_col not in df.columns:
        raise ValueError(f"missing split date column: {date_col}")
    if round(sum(ratios), 8) != 1:
        raise ValueError("split ratios must sum to 1")
    missing_dates = int(df[date_col].isna().sum())
    if missing_dates:
        raise ValueError(f"{date_col} has {missing_dates} missing values")

    ordered = df.sort_values([date_col, "id"] if "id" in df.columns else [date_col]).copy()
    n = len(ordered)
    train_end = int(n * ratios[0])
    calib_end = train_end + int(n * ratios[1])
    valid_end = calib_end + int(n * ratios[2])
    if n == 0:
        return {name: ordered.copy() for name in ["train", "calibration", "validation", "test"]}
    if ordered[date_col].nunique(dropna=True) < 4:
        return {
            "train": ordered.iloc[:train_end].copy(),
            "calibration": ordered.iloc[train_end:calib_end].copy(),
            "validation": ordered.iloc[calib_end:valid_end].copy(),
            "test": ordered.iloc[valid_end:].copy(),
        }

    split_names = ["train", "calibration", "validation", "test"]
    targets = [train_end, calib_end, valid_end, n]
    split_for_index = {}
    row_count = 0
    split_idx = 0
    for _, group in ordered.groupby(date_col, sort=True):
        while split_idx < 3 and row_count >= targets[split_idx]:
            split_idx += 1
        split_for_index.update({idx: split_names[split_idx] for idx in group.index})
        row_count += len(group)

    assigned = pd.Series(split_for_index)
    return {
        name: ordered.loc[assigned[assigned == name].index].copy()
        for name in split_names
    }


def split_manifest(splits: dict[str, pd.DataFrame]) -> dict:
    manifest = {"row_counts": {}, "date_ranges": {}}
    for name, frame in splits.items():
        manifest["row_counts"][name] = int(len(frame))
        if "issue_dt" in frame.columns and len(frame):
            manifest["date_ranges"][name] = {
                "min": frame["issue_dt"].min().isoformat(),
                "max": frame["issue_dt"].max().isoformat(),
            }
        else:
            manifest["date_ranges"][name] = {"min": None, "max": None}
        if name == "test" and "id" in frame.columns:
            manifest["test_ids"] = frame["id"].astype(str).tolist()
    return manifest


def split_count_report(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for split, frame in splits.items():
        by_year = frame.groupby("issue_year", dropna=False)[TARGET].agg(["count", "sum"])
        for issue_year, values in by_year.iterrows():
            defaults = int(values["sum"])
            total = int(values["count"])
            rows.append(
                {
                    "split": split,
                    "issue_year": issue_year,
                    "rows": total,
                    "defaults": defaults,
                    "non_defaults": total - defaults,
                }
            )
    return pd.DataFrame(rows)


def split_summary(splits: dict[str, pd.DataFrame]) -> list[dict]:
    rows = []
    for split, frame in splits.items():
        row = {"split": split, "rows": int(len(frame))}
        if TARGET in frame.columns and len(frame):
            row["default_rate"] = float(frame[TARGET].mean())
        else:
            row["default_rate"] = None
        if "issue_dt" in frame.columns and len(frame):
            row["date_min"] = frame["issue_dt"].min().isoformat()
            row["date_max"] = frame["issue_dt"].max().isoformat()
        else:
            row["date_min"] = None
            row["date_max"] = None
        rows.append(row)
    return rows


def split_row_count_report(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for split, frame in splits.items():
        by_year = frame.groupby("issue_year", dropna=False).size()
        for issue_year, count in by_year.items():
            rows.append({"split": split, "issue_year": issue_year, "rows": int(count)})
    return pd.DataFrame(rows)


def map_accepted_to_rejected_style(df: pd.DataFrame) -> pd.DataFrame:
    required = [c for c in ACCEPTED_TO_REJECTED_FEATURE_MAP if c != "policy_code"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing accepted columns for rejected-style map: {missing}")

    out = pd.DataFrame(index=df.index)
    for source, target in ACCEPTED_TO_REJECTED_FEATURE_MAP.items():
        if source in df.columns:
            out[target] = df[source]
    out["risk_score"] = pd.to_numeric(
        df[["fico_range_low", "fico_range_high"]].mean(axis=1), errors="coerce"
    )
    for column in REJECTED_STYLE_NUMERIC_RISK_FEATURES:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    if TARGET in df.columns:
        out[TARGET] = df[TARGET]
    if "issue_dt" in df.columns:
        out["issue_dt"] = df["issue_dt"]
        out["issue_year"] = df.get("issue_year")
    return out


def normalize_rejected_input(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(columns={k: v for k, v in REJECTED_RAW_ALIASES.items() if k in df.columns}).copy()
    if "dti" in out.columns:
        out["dti"] = out["dti"].map(parse_percent)
    for column in ["amount_requested", "risk_score", "funded_amnt", "installment", "term_months"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out
