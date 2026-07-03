from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .config import (
    ACCEPTED_CATEGORICAL_RISK_FEATURES,
    ACCEPTED_NUMERIC_RISK_FEATURES,
    PRODUCT_MODE_POST_PRICING,
    REJECTED_STYLE_CATEGORICAL_RISK_FEATURES,
    REJECTED_STYLE_NUMERIC_RISK_FEATURES,
    TARGET,
)
from .preprocessing import ensure_no_forbidden_features, feature_columns_for_product_mode


def _one_hot_encoder(sparse=True):
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=sparse)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=sparse)


def _feature_groups(kind: str, product_mode: str):
    if kind == "accepted":
        columns = feature_columns_for_product_mode(product_mode)
        numeric = [c for c in ACCEPTED_NUMERIC_RISK_FEATURES if c in columns]
        categorical = [c for c in ACCEPTED_CATEGORICAL_RISK_FEATURES if c in columns]
    elif kind == "rejected_style":
        numeric = REJECTED_STYLE_NUMERIC_RISK_FEATURES
        categorical = REJECTED_STYLE_CATEGORICAL_RISK_FEATURES
    else:
        raise ValueError(f"unknown model kind: {kind}")

    features = numeric + categorical
    ensure_no_forbidden_features(features, product_mode=product_mode)
    return numeric, categorical


def build_preprocessor(kind: str, product_mode: str = PRODUCT_MODE_POST_PRICING, sparse=True) -> ColumnTransformer:
    numeric, categorical = _feature_groups(kind, product_mode)
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
                        ("onehot", _one_hot_encoder(sparse=sparse)),
                    ]
                ),
                categorical,
            ),
        ]
    )


def build_tree_preprocessor(kind: str, product_mode: str = PRODUCT_MODE_POST_PRICING) -> ColumnTransformer:
    numeric, categorical = _feature_groups(kind, product_mode)
    return ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), numeric),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                    ]
                ),
                categorical,
            ),
        ]
    )


def build_logistic_model(kind: str, class_weight="balanced", product_mode: str = PRODUCT_MODE_POST_PRICING) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", build_preprocessor(kind, product_mode=product_mode)),
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


def build_default_model(name: str, kind: str, product_mode: str = PRODUCT_MODE_POST_PRICING) -> Pipeline:
    if name == "logistic_balanced":
        return build_logistic_model(kind, class_weight="balanced", product_mode=product_mode)
    if name == "logistic":
        return build_logistic_model(kind, class_weight=None, product_mode=product_mode)
    if name == "random_forest":
        return Pipeline(
            [
                ("preprocess", build_tree_preprocessor(kind, product_mode=product_mode)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=80,
                        max_depth=12,
                        min_samples_leaf=100,
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        )
    if name == "hist_gradient_boosting":
        return Pipeline(
            [
                ("preprocess", build_tree_preprocessor(kind, product_mode=product_mode)),
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        max_iter=120,
                        learning_rate=0.08,
                        l2_regularization=0.1,
                        random_state=42,
                    ),
                ),
            ]
        )
    raise ValueError(f"unknown default model candidate: {name}")


def fit_model(
    train_df,
    feature_columns,
    kind: str,
    class_weight="balanced",
    product_mode: str = PRODUCT_MODE_POST_PRICING,
    candidate_name: str | None = None,
):
    ensure_no_forbidden_features(feature_columns, product_mode=product_mode)
    model = (
        build_default_model(candidate_name, kind, product_mode=product_mode)
        if candidate_name
        else build_logistic_model(kind, class_weight=class_weight, product_mode=product_mode)
    )
    return model.fit(train_df[feature_columns], train_df[TARGET])


def predict_raw_default(model, df, feature_columns):
    return model.predict_proba(df[feature_columns])[:, 1]
