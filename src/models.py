from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .config import ACCEPTED_CATEGORICAL_RISK_FEATURES, ACCEPTED_NUMERIC_RISK_FEATURES, TARGET
from .preprocessing import ensure_no_forbidden_features


def _one_hot_encoder(sparse=True):
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=sparse)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=sparse)


def _feature_groups(feature_columns: list[str]) -> tuple[list[str], list[str]]:
    ensure_no_forbidden_features(feature_columns)
    numeric = [column for column in ACCEPTED_NUMERIC_RISK_FEATURES if column in feature_columns]
    categorical = [column for column in ACCEPTED_CATEGORICAL_RISK_FEATURES if column in feature_columns]
    return numeric, categorical


def build_preprocessor(feature_columns: list[str], sparse=True) -> ColumnTransformer:
    numeric, categorical = _feature_groups(feature_columns)
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())]),
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


def build_tree_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    numeric, categorical = _feature_groups(feature_columns)
    return ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median", add_indicator=True), numeric),
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


def build_model(candidate_name: str, feature_columns: list[str]) -> Pipeline:
    if candidate_name == "logistic_balanced":
        classifier = LogisticRegression(
            max_iter=500,
            class_weight="balanced",
            solver="saga",
            random_state=42,
        )
        return Pipeline([("preprocess", build_preprocessor(feature_columns)), ("classifier", classifier)])
    if candidate_name == "logistic":
        classifier = LogisticRegression(
            max_iter=500,
            class_weight=None,
            solver="saga",
            random_state=42,
        )
        return Pipeline([("preprocess", build_preprocessor(feature_columns)), ("classifier", classifier)])
    if candidate_name == "random_forest":
        return Pipeline(
            [
                ("preprocess", build_tree_preprocessor(feature_columns)),
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
    if candidate_name == "hist_gradient_boosting":
        return Pipeline(
            [
                ("preprocess", build_tree_preprocessor(feature_columns)),
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
    raise ValueError(f"unknown default model candidate: {candidate_name}")


def fit_model(train_df, feature_columns: list[str], candidate_name: str) -> Pipeline:
    ensure_no_forbidden_features(feature_columns)
    model = build_model(candidate_name, feature_columns)
    return model.fit(train_df[feature_columns], train_df[TARGET])


def predict_raw_default(model, df, feature_columns: list[str]):
    return model.predict_proba(df[feature_columns])[:, 1]
