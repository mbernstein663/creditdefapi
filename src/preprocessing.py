from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd

from .config import (
    ACCEPTED_TO_REJECTED_FEATURE_MAP,
    ACCEPTED_NUMERIC_RISK_FEATURES,
    BAD_STATUSES,
    FORBIDDEN_FEATURE_COLUMNS,
    GOOD_STATUSES,
    PROFIT_INPUT_COLUMNS,
    REJECTED_STYLE_NUMERIC_RISK_FEATURES,
    REJECTED_RAW_ALIASES,
    TARGET,
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


def construct_target(df: pd.DataFrame) -> pd.DataFrame:
    if "loan_status" not in df.columns:
        raise ValueError("accepted loan data must include loan_status")

    out = df.copy()
    status = out["loan_status"].fillna("").astype(str).str.strip()
    resolved = status.isin(GOOD_STATUSES | BAD_STATUSES)
    out = out.loc[resolved].copy()
    status = status.loc[resolved]
    out[TARGET] = status.isin(BAD_STATUSES).astype(int)
    return out


def prepare_accepted_loans(df: pd.DataFrame) -> pd.DataFrame:
    out = construct_target(df)
    if "term" in out.columns and "term_months" not in out.columns:
        out["term_months"] = out["term"].map(parse_term_months)
    if "issue_d" in out.columns:
        out["issue_dt"] = pd.to_datetime(out["issue_d"], format="%b-%Y", errors="coerce")
        out["issue_year"] = out["issue_dt"].dt.year

    numeric_columns = set(ACCEPTED_NUMERIC_RISK_FEATURES + PROFIT_INPUT_COLUMNS)
    for column in numeric_columns & set(out.columns):
        if column in {"int_rate", "revol_util"}:
            out[column] = pd.to_numeric(out[column].map(parse_percent), errors="coerce")
        else:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def ensure_no_forbidden_features(feature_columns: Iterable[str]) -> None:
    forbidden = sorted(set(feature_columns) & FORBIDDEN_FEATURE_COLUMNS)
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
