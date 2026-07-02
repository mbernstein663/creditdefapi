from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.artifacts import ModelBundle, file_fingerprint, package_versions, save_model_bundle
from src.config import (
    ACCEPTED_CATEGORICAL_RISK_FEATURES,
    ACCEPTED_CSV,
    ACCEPTED_NUMERIC_RISK_FEATURES,
    ACCEPTED_RISK_FEATURES,
    ARTIFACT_DIR,
    DEFAULT_ACCEPTED_BUNDLE,
    DEFAULT_PROFIT_BUNDLE,
    FORBIDDEN_FEATURE_COLUMNS,
    PROFIT_INPUT_COLUMNS,
    PROFIT_TARGET,
    REPORT_DIR,
    TARGET,
)
from src.preprocessing import split_chronological, split_manifest, split_summary
from src.profit_challenger import (
    default_risk_policy_metrics,
    predict_profit,
    prepare_profit_frame,
    profit_policy_metrics,
    regression_summary,
    search_profit_policy,
    validate_profit_features,
)


def _read_profit_data(path, sample=None):
    needed = set(ACCEPTED_RISK_FEATURES + PROFIT_INPUT_COLUMNS)
    needed |= {"id", "term", "loan_status", "issue_d", "total_pymnt"}
    return pd.read_csv(path, usecols=lambda col: col in needed, low_memory=False, nrows=sample)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _paths(output_path, sample):
    report_dir = REPORT_DIR / "smoke" if sample else REPORT_DIR
    output = DEFAULT_PROFIT_BUNDLE.with_name("direct_profit_model_smoke.joblib") if sample else output_path
    return output, report_dir


def _one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def _linear_preprocessor():
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                ACCEPTED_NUMERIC_RISK_FEATURES,
            ),
            (
                "cat",
                Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", _one_hot_encoder())]),
                ACCEPTED_CATEGORICAL_RISK_FEATURES,
            ),
        ]
    )


def _tree_preprocessor():
    return ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), ACCEPTED_NUMERIC_RISK_FEATURES),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "ordinal",
                            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                        ),
                    ]
                ),
                ACCEPTED_CATEGORICAL_RISK_FEATURES,
            ),
        ]
    )


def candidate_models():
    return [
        (
            "ridge",
            Pipeline([("preprocess", _linear_preprocessor()), ("regressor", Ridge(alpha=10.0))]),
        ),
        (
            "random_forest",
            Pipeline(
                [
                    ("preprocess", _tree_preprocessor()),
                    (
                        "regressor",
                        RandomForestRegressor(
                            n_estimators=50,
                            max_depth=10,
                            min_samples_leaf=50,
                            max_samples=0.35,
                            n_jobs=-1,
                            random_state=42,
                        ),
                    ),
                ]
            ),
        ),
        (
            "hist_gradient_boosting",
            Pipeline(
                [
                    ("preprocess", _tree_preprocessor()),
                    (
                        "regressor",
                        HistGradientBoostingRegressor(
                            max_iter=120,
                            learning_rate=0.08,
                            l2_regularization=0.1,
                            early_stopping=True,
                            random_state=42,
                        ),
                    ),
                ]
            ),
        ),
    ]


def _small_metrics(metrics):
    return {k: v for k, v in metrics.items() if k != "decile_lift"}


def select_profit_candidate(train_df, validation_df):
    validate_profit_features(ACCEPTED_RISK_FEATURES)
    candidates = []
    for name, model in candidate_models():
        model.fit(train_df[ACCEPTED_RISK_FEATURES], train_df[PROFIT_TARGET])
        predicted = model.predict(validation_df[ACCEPTED_RISK_FEATURES])
        policy, policy_candidates = search_profit_policy(validation_df, predicted)
        selected_policy_metrics = profit_policy_metrics(validation_df, predicted, policy)
        candidates.append(
            {
                "name": name,
                "model": model,
                "policy": policy,
                "regression": regression_summary(validation_df[PROFIT_TARGET], predicted),
                "selected_policy_metrics": selected_policy_metrics,
                "policy_candidates": policy_candidates,
            }
        )
    return max(
        candidates,
        key=lambda c: (
            c["selected_policy_metrics"]["total_realized_profit"],
            c["selected_policy_metrics"]["profit_per_dollar_funded"],
            -c["regression"]["rmse"],
        ),
    ), candidates


def train_profit_model(
    csv_path=ACCEPTED_CSV,
    output_path=DEFAULT_PROFIT_BUNDLE,
    accepted_bundle_path=DEFAULT_ACCEPTED_BUNDLE,
    sample=None,
):
    output_path, report_dir = _paths(output_path, sample)
    report_dir.mkdir(parents=True, exist_ok=True)
    source_fingerprint = file_fingerprint(csv_path)
    training_timestamp = datetime.now(timezone.utc).isoformat()
    versions = package_versions()

    frame = prepare_profit_frame(_read_profit_data(csv_path, sample=sample))
    splits = split_chronological(frame)
    manifest = split_manifest(splits)
    train_df = splits["train"]
    validation_df = splits["validation"]
    selected, candidates = select_profit_candidate(train_df, validation_df)

    bundle = ModelBundle(
        model=selected["model"],
        calibrator=None,
        feature_columns=list(ACCEPTED_RISK_FEATURES),
        model_type="direct_profit",
        policy={
            **selected["policy"],
            "selection_rule": "max validation total_realized_profit; tie-break profit_per_dollar_funded",
        },
        required_input_schema={"risk_features": list(ACCEPTED_RISK_FEATURES)},
        metadata={
            "target": PROFIT_TARGET,
            "target_definition": "realized_profit = total_pymnt - funded_amnt on resolved accepted funded loans",
            "source_fingerprint": source_fingerprint,
            "split_manifest": manifest,
            "split_rule": "chronological split via src.preprocessing.split_chronological",
            "selected_candidate": selected["name"],
            "selection_rule": "model and policy selected on validation realized-profit policy performance",
            "forbidden_feature_columns": sorted(FORBIDDEN_FEATURE_COLUMNS),
            "rejected_data_handling": "rejected applications have no realized-profit labels and are excluded",
            "random_state": 42,
            "package_versions": versions,
            "training_timestamp": training_timestamp,
            "is_smoke_sample": bool(sample),
            "sample_rows_requested": sample,
        },
    )
    saved = save_model_bundle(bundle, output_path)
    selected_pred = predict_profit(bundle, validation_df)
    selected_metrics = profit_policy_metrics(validation_df, selected_pred, bundle.policy)

    report = {
        "is_smoke_sample": bool(sample),
        "sample_rows_requested": sample,
        "artifact_path": str(saved),
        "training_timestamp": training_timestamp,
        "package_versions": versions,
        "source_fingerprint": source_fingerprint,
        "resolved_row_count": int(len(frame)),
        "split_manifest": {k: v for k, v in manifest.items() if k != "test_ids"},
        "split_summary": split_summary(splits),
        "feature_columns": list(ACCEPTED_RISK_FEATURES),
        "target": PROFIT_TARGET,
        "selected_candidate": selected["name"],
        "selected_policy": bundle.policy,
        "selected_validation_metrics": _small_metrics(selected_metrics),
        "candidates": [
            {
                "name": c["name"],
                "policy": c["policy"],
                "regression": c["regression"],
                "selected_policy_metrics": _small_metrics(c["selected_policy_metrics"]),
                "policy_candidates": c["policy_candidates"],
            }
            for c in candidates
        ],
    }
    _write_json(report_dir / "direct_profit_validation_metrics.json", report)
    pd.DataFrame(selected_metrics["decile_lift"]).to_csv(
        report_dir / "direct_profit_validation_deciles.csv", index=False
    )
    _write_json(
        report_dir / "profit_policy_comparison_validation.json",
        {
            "note": "validation comparison only; no test-set selection",
            "direct_profit": _small_metrics(selected_metrics),
            "default_risk": default_risk_policy_metrics(validation_df, accepted_bundle_path),
        },
    )
    return saved


def main():
    parser = argparse.ArgumentParser(description="Train direct realized-profit challenger model.")
    parser.add_argument("--csv", default=ACCEPTED_CSV)
    parser.add_argument("--output", default=DEFAULT_PROFIT_BUNDLE)
    parser.add_argument("--accepted-bundle", default=DEFAULT_ACCEPTED_BUNDLE)
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    print(train_profit_model(args.csv, args.output, args.accepted_bundle, sample=args.sample))


if __name__ == "__main__":
    main()
