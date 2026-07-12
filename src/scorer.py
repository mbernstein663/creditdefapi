from __future__ import annotations

import numpy as np
import pandas as pd

from .artifacts import ModelBundle
from .config import ACCEPTED_NUMERIC_RISK_FEATURES, ACCEPTED_RISK_FEATURES
from .preprocessing import apply_numeric_feature_transforms, ensure_no_forbidden_features, parse_percent

SCORING_NOTE = (
    "Probability is calibrated for accepted/funded LendingClub-style loans with resolved historical outcomes. "
    "decision_margin is 2 * abs(p_default - 0.5): scaled distance from a 50/50 default probability, not statistical confidence."
)

"""
Takes model bundle and establishes scoring layer of pipeline.
Returns p_default, risk_band, etc and other important prediction data for a set of records.
Used for defaulting prediction inference in frontend and batch evaluation.
"""

def _risk_band(p_default: pd.Series, thresholds: dict | None) -> pd.Series:
    low_max = float((thresholds or {}).get("low_max", 0.10))
    medium_max = float((thresholds or {}).get("medium_max", 0.20))
    return pd.Series(
        [
            "low" if value < low_max else "medium" if value < medium_max else "high"
            for value in pd.to_numeric(p_default, errors="coerce")
        ],
        index=p_default.index,
    )


def _validate_bundle(bundle: ModelBundle) -> None:
    if not set(bundle.feature_columns).issubset(set(ACCEPTED_RISK_FEATURES)):
        raise ValueError("bundle feature columns do not match the configured accepted-loan feature allowlist")
    ensure_no_forbidden_features(bundle.feature_columns)
    if bundle.calibrator is None:
        raise ValueError("scoring requires a calibrated model bundle")


def _prepare_scoring_frame(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    prepared = frame.copy()
    numeric_columns = set(feature_columns) & set(ACCEPTED_NUMERIC_RISK_FEATURES)
    for column in numeric_columns:
        if column not in prepared.columns:
            continue
        if column in {"int_rate", "revol_util"}:
            prepared[column] = pd.to_numeric(prepared[column].map(parse_percent), errors="coerce")
        else:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    return apply_numeric_feature_transforms(prepared)


def _decision_margin(p_default: pd.Series) -> pd.Series:
    values = pd.to_numeric(p_default, errors="coerce").astype(float)
    return (values.sub(0.5).abs() * 2).clip(0, 1)


def predict_default(bundle: ModelBundle, frame: pd.DataFrame) -> pd.Series:
    _validate_bundle(bundle)
    prepared = _prepare_scoring_frame(frame, bundle.feature_columns)
    missing = [column for column in bundle.feature_columns if column not in prepared.columns]
    if missing:
        raise ValueError(f"missing required scoring fields: {', '.join(missing)}")
    raw = bundle.model.predict_proba(prepared[bundle.feature_columns])[:, 1]
    return pd.Series(bundle.calibrator.predict(raw), index=frame.index)


def score_frame(frame: pd.DataFrame, bundle: ModelBundle) -> pd.DataFrame:
    prepared = _prepare_scoring_frame(frame, bundle.feature_columns)
    p_default = predict_default(bundle, prepared)
    metadata = bundle.metadata or {}
    scored = prepared.copy()
    scored["p_default"] = p_default
    scored["p_non_default"] = 1 - pd.to_numeric(p_default, errors="coerce")
    scored["decision_margin"] = _decision_margin(p_default)
    scored["risk_band"] = _risk_band(p_default, metadata.get("risk_band_thresholds"))
    scored["model_version"] = metadata.get("model_version")
    scored["model_type"] = metadata.get("selected_model_type", bundle.model_type)
    scored["calibration_method"] = metadata.get("calibration_method")
    scored["scoring_note"] = SCORING_NOTE
    return scored


def score_records(records, bundle: ModelBundle) -> list[dict]:
    frame = pd.DataFrame(records)
    scored = score_frame(frame, bundle)
    columns = [
        "p_default",
        "p_non_default",
        "decision_margin",
        "risk_band",
        "model_version",
        "model_type",
        "calibration_method",
        "scoring_note",
    ]
    result = scored[columns].astype(object)
    result = result.where(pd.notna(result), None)
    return result.to_dict(orient="records")
