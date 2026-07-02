from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    ACCEPTED_CATEGORICAL_RISK_FEATURES,
    ACCEPTED_NUMERIC_RISK_FEATURES,
    REJECTED_STYLE_CATEGORICAL_RISK_FEATURES,
    REJECTED_STYLE_NUMERIC_RISK_FEATURES,
    TARGET,
)
from .preprocessing import ensure_no_forbidden_features


def _one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def build_preprocessor(kind: str) -> ColumnTransformer:
    if kind == "accepted":
        numeric = ACCEPTED_NUMERIC_RISK_FEATURES
        categorical = ACCEPTED_CATEGORICAL_RISK_FEATURES
    elif kind == "rejected_style":
        numeric = REJECTED_STYLE_NUMERIC_RISK_FEATURES
        categorical = REJECTED_STYLE_CATEGORICAL_RISK_FEATURES
    else:
        raise ValueError(f"unknown model kind: {kind}")

    features = numeric + categorical
    ensure_no_forbidden_features(features)
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", _one_hot_encoder()),
                    ]
                ),
                categorical,
            ),
        ]
    )


def build_logistic_model(kind: str, class_weight="balanced") -> Pipeline:
    return Pipeline(
        [
            ("preprocess", build_preprocessor(kind)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=500,
                    class_weight=class_weight,
                    solver="saga",
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )


def fit_model(train_df, feature_columns, kind: str, class_weight="balanced"):
    ensure_no_forbidden_features(feature_columns)
    model = build_logistic_model(kind, class_weight=class_weight)
    return model.fit(train_df[feature_columns], train_df[TARGET])


def predict_raw_default(model, df, feature_columns):
    return model.predict_proba(df[feature_columns])[:, 1]
