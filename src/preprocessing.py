from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .artifacts import file_fingerprint
from .config import (
    ACCEPTED_CATEGORICAL_RISK_FEATURES,
    ACCEPTED_CSV,
    ACCEPTED_NUMERIC_RISK_FEATURES,
    BAD_STATUSES,
    DEFAULT_PREPROCESSED_ACCEPTED_BUNDLE,
    FORBIDDEN_FEATURE_COLUMNS,
    GOOD_STATUSES,
    RISK_NUMERIC_CLIP_RANGES,
    RISK_NUMERIC_LOG_TRANSFORMS,
    TARGET,
    UNRESOLVED_STATUSES,
)


@dataclass(slots=True)
class PreprocessedAcceptedLoans:
    source_fingerprint: dict
    target_summary: dict
    accepted: pd.DataFrame
    splits: dict[str, pd.DataFrame]
    manifest: dict
    split_summary: list[dict]
    selected_features: list[str]
    sample_rows_requested: int | None


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


def apply_numeric_feature_transforms(df: pd.DataFrame) -> pd.DataFrame:
    transformed = df.copy()
    for column, (lower, upper) in RISK_NUMERIC_CLIP_RANGES.items():
        if column in transformed.columns:
            transformed[column] = pd.to_numeric(transformed[column], errors="coerce").clip(lower=lower, upper=upper)
    for column in RISK_NUMERIC_LOG_TRANSFORMS:
        if column in transformed.columns:
            values = pd.to_numeric(transformed[column], errors="coerce").clip(lower=0)
            transformed[column] = np.log1p(values)
    return transformed


def construct_target(df: pd.DataFrame, return_summary: bool = False) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
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
    out[TARGET] = status.loc[resolved].isin(BAD_STATUSES).astype(int)
    summary = {
        "target": TARGET,
        "good_statuses": sorted(GOOD_STATUSES),
        "bad_statuses": sorted(BAD_STATUSES),
        "dropped_statuses": sorted(UNRESOLVED_STATUSES),
        "included_rows": int(len(out)),
        "excluded_rows": int(len(df) - len(out)),
        "total_rows": int(len(df)),
    }
    if return_summary:
        return out, summary
    out.attrs["target_summary"] = summary
    return out


def prepare_accepted_loans(df: pd.DataFrame, return_summary: bool = False) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    prepared, summary = construct_target(df, return_summary=True)
    if "term" in prepared.columns and "term_months" not in prepared.columns:
        prepared["term_months"] = prepared["term"].map(parse_term_months)
    if "issue_d" in prepared.columns:
        prepared["issue_dt"] = prepared["issue_d"].map(parse_date)
        prepared["issue_year"] = prepared["issue_dt"].dt.year
        prepared["issue_quarter"] = prepared["issue_dt"].dt.to_period("Q").astype(str)
        prepared["issue_month"] = prepared["issue_dt"].dt.to_period("M").astype(str)

    numeric_columns = set(ACCEPTED_NUMERIC_RISK_FEATURES) & set(prepared.columns)
    for column in numeric_columns:
        if column in {"int_rate", "revol_util"}:
            prepared[column] = pd.to_numeric(prepared[column].map(parse_percent), errors="coerce")
        else:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    prepared = apply_numeric_feature_transforms(prepared)

    if return_summary:
        return prepared, summary
    prepared.attrs["target_summary"] = summary
    return prepared


def feature_columns() -> list[str]:
    return ACCEPTED_NUMERIC_RISK_FEATURES + ACCEPTED_CATEGORICAL_RISK_FEATURES


def ensure_no_forbidden_features(feature_columns: Iterable[str]) -> None:
    forbidden = sorted(set(feature_columns) & set(FORBIDDEN_FEATURE_COLUMNS))
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
    if ordered.empty:
        return {name: ordered.copy() for name in ["train", "calibration", "validation", "test"]}

    unique_dates = ordered[date_col].dropna().nunique()
    if unique_dates < 4:
        raise ValueError("chronological split requires at least 4 distinct issue dates")

    n = len(ordered)
    targets = [
        int(n * ratios[0]),
        int(n * (ratios[0] + ratios[1])),
        int(n * (ratios[0] + ratios[1] + ratios[2])),
        n,
    ]
    split_names = ["train", "calibration", "validation", "test"]
    rows_by_split: dict[str, list[int]] = {name: [] for name in split_names}
    split_idx = 0
    assigned = 0

    for _, group in ordered.groupby(date_col, sort=True):
        while split_idx < 3 and assigned >= targets[split_idx]:
            split_idx += 1
        rows_by_split[split_names[split_idx]].extend(group.index.tolist())
        assigned += len(group)

    return {name: ordered.loc[indexes].copy() for name, indexes in rows_by_split.items()}


def split_manifest(splits: dict[str, pd.DataFrame]) -> dict:
    manifest = {"row_counts": {}, "date_ranges": {}, "default_rates": {}}
    for name, frame in splits.items():
        manifest["row_counts"][name] = int(len(frame))
        manifest["default_rates"][name] = float(frame[TARGET].mean()) if TARGET in frame.columns and len(frame) else None
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


def split_summary(splits: dict[str, pd.DataFrame]) -> list[dict]:
    rows = []
    for split, frame in splits.items():
        rows.append(
            {
                "split": split,
                "rows": int(len(frame)),
                "default_rate": float(frame[TARGET].mean()) if TARGET in frame.columns and len(frame) else None,
                "date_min": frame["issue_dt"].min().isoformat() if "issue_dt" in frame.columns and len(frame) else None,
                "date_max": frame["issue_dt"].max().isoformat() if "issue_dt" in frame.columns and len(frame) else None,
            }
        )
    return rows


def split_count_report(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for split, frame in splits.items():
        grouped = frame.groupby("issue_year", dropna=False)[TARGET].agg(["count", "sum"])
        for issue_year, values in grouped.iterrows():
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


def normalize_rejected_input(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.rename(
        columns={
            "Amount Requested": "amount_requested",
            "Application Date": "application_date",
            "Loan Title": "loan_title",
            "Risk_Score": "risk_score",
            "Debt-To-Income Ratio": "dti",
            "Zip Code": "zip_code",
            "State": "state",
            "Employment Length": "employment_length",
        }
    ).copy()
    if "dti" in renamed.columns:
        renamed["dti"] = renamed["dti"].map(parse_percent)
    return renamed


def _read_accepted(path, selected_features, sample=None):
    needed = set(selected_features) | {"id", "loan_status", "issue_d", "term"}
    if sample is None:
        return pd.read_csv(path, usecols=lambda col: col in needed, low_memory=False)

    chunk_size = max(sample, 50000)
    frames = []
    for chunk in pd.read_csv(path, usecols=lambda col: col in needed, low_memory=False, chunksize=chunk_size):
        frames.append(chunk)
        candidate = pd.concat(frames, ignore_index=True)
        prepared = prepare_accepted_loans(candidate)
        try:
            splits = split_chronological(prepared)
        except ValueError:
            continue
        if len(prepared) >= sample and all(len(frame) > 0 for frame in splits.values()):
            return candidate
    return pd.concat(frames, ignore_index=True)


def preprocess_accepted_loans(csv_path=ACCEPTED_CSV, sample=None, selected_features: list[str] | None = None) -> PreprocessedAcceptedLoans:
    selected_features = list(selected_features or feature_columns())
    source = _read_accepted(csv_path, selected_features, sample=sample)
    accepted, target_summary = prepare_accepted_loans(source, return_summary=True)
    splits = split_chronological(accepted)
    return PreprocessedAcceptedLoans(
        source_fingerprint=file_fingerprint(csv_path),
        target_summary=target_summary,
        accepted=accepted,
        splits=splits,
        manifest=split_manifest(splits),
        split_summary=split_summary(splits),
        selected_features=selected_features,
        sample_rows_requested=sample,
    )


def save_preprocessed_accepted_loans(bundle: PreprocessedAcceptedLoans, path: str | Path = DEFAULT_PREPROCESSED_ACCEPTED_BUNDLE) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output)
    return output


def load_preprocessed_accepted_loans(path: str | Path = DEFAULT_PREPROCESSED_ACCEPTED_BUNDLE) -> PreprocessedAcceptedLoans:
    return joblib.load(Path(path))


def main():
    parser = argparse.ArgumentParser(description="Prepare accepted LendingClub loans for default-risk training.")
    parser.add_argument("--csv", default=ACCEPTED_CSV)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--output", default=DEFAULT_PREPROCESSED_ACCEPTED_BUNDLE)
    args = parser.parse_args()
    result = preprocess_accepted_loans(args.csv, sample=args.sample)
    saved = save_preprocessed_accepted_loans(result, args.output)
    print(
        json.dumps(
            {
                "output": str(saved),
                "source_fingerprint": result.source_fingerprint,
                "target_summary": result.target_summary,
                "split_manifest": result.manifest,
                "split_summary": result.split_summary,
                "selected_features": result.selected_features,
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
