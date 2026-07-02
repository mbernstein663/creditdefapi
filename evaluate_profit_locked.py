from __future__ import annotations

import argparse
import json

import pandas as pd

from src.artifacts import file_fingerprint, load_model_bundle
from src.config import (
    ACCEPTED_CSV,
    ACCEPTED_RISK_FEATURES,
    DEFAULT_ACCEPTED_BUNDLE,
    DEFAULT_PROFIT_BUNDLE,
    PROFIT_INPUT_COLUMNS,
    REPORT_DIR,
)
from src.profit_challenger import (
    default_risk_policy_metrics,
    predict_profit,
    prepare_profit_frame,
    profit_policy_metrics,
)


def _read_profit_data(path, feature_columns, sample=None):
    needed = set(feature_columns + PROFIT_INPUT_COLUMNS)
    needed |= {"id", "term", "loan_status", "issue_d", "total_pymnt"}
    return pd.read_csv(path, usecols=lambda col: col in needed, low_memory=False, nrows=sample)


def _small_metrics(metrics):
    return {k: v for k, v in metrics.items() if k != "decile_lift"}


def _fingerprint_matches(expected, actual):
    return expected and expected.get("sha256") == actual.get("sha256") and expected.get("size_bytes") == actual.get(
        "size_bytes"
    )


def evaluate_locked_profit_model(
    bundle_path=DEFAULT_PROFIT_BUNDLE,
    csv_path=ACCEPTED_CSV,
    accepted_bundle_path=DEFAULT_ACCEPTED_BUNDLE,
    sample=None,
):
    bundle = load_model_bundle(bundle_path)
    if bundle.model_type != "direct_profit":
        raise ValueError("locked profit evaluation requires a direct_profit bundle")
    expected_source = bundle.metadata.get("source_fingerprint")
    actual_source = file_fingerprint(csv_path)
    if expected_source and not _fingerprint_matches(expected_source, actual_source):
        raise ValueError("source CSV fingerprint does not match the locked profit bundle")

    test_ids = set(bundle.metadata.get("split_manifest", {}).get("test_ids", []))
    if not test_ids:
        raise ValueError("locked profit bundle is missing saved test split IDs")

    frame = prepare_profit_frame(_read_profit_data(csv_path, bundle.feature_columns, sample=sample))
    test_df = frame.loc[frame["id"].astype(str).isin(test_ids)].copy()
    if len(test_df) != len(test_ids):
        raise ValueError("source CSV does not contain every saved profit test split ID")

    predicted = predict_profit(bundle, test_df)
    metrics = profit_policy_metrics(test_df, predicted, bundle.policy)
    comparison = {
        "note": "locked test comparison; no model, threshold, or policy selection happens here",
        "direct_profit": _small_metrics(metrics),
        "default_risk": default_risk_policy_metrics(test_df, accepted_bundle_path),
    }
    result = {
        "model_type": bundle.model_type,
        "selected_candidate": bundle.metadata.get("selected_candidate"),
        "locked_policy": bundle.policy,
        "profit_policy": _small_metrics(metrics),
        "comparison": comparison,
        "artifact_path": str(bundle_path),
        "training_timestamp": bundle.metadata.get("training_timestamp"),
        "source_fingerprint": actual_source,
        "test_row_count": int(len(test_df)),
        "note": "test set is evaluated after loading the locked direct-profit bundle",
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    is_smoke = bool(sample or bundle.metadata.get("is_smoke_sample"))
    metrics_path = REPORT_DIR / ("direct_profit_locked_test_metrics_smoke.json" if is_smoke else "direct_profit_locked_test_metrics.json")
    deciles_path = REPORT_DIR / ("direct_profit_locked_test_deciles_smoke.csv" if is_smoke else "direct_profit_locked_test_deciles.csv")
    comparison_path = REPORT_DIR / (
        "profit_policy_comparison_locked_test_smoke.json" if is_smoke else "profit_policy_comparison_locked_test.json"
    )
    metrics_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    comparison_path.write_text(json.dumps(comparison, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(metrics["decile_lift"]).to_csv(deciles_path, index=False)
    return metrics_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate locked direct realized-profit challenger model.")
    parser.add_argument("--bundle", default=DEFAULT_PROFIT_BUNDLE)
    parser.add_argument("--csv", default=ACCEPTED_CSV)
    parser.add_argument("--accepted-bundle", default=DEFAULT_ACCEPTED_BUNDLE)
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()
    print(evaluate_locked_profit_model(args.bundle, args.csv, args.accepted_bundle, sample=args.sample))


if __name__ == "__main__":
    main()
