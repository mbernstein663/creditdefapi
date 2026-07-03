from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.artifacts import ModelBundle, file_fingerprint, package_versions, save_model_bundle
from src.calibration import (
    ProbabilityCalibrator,
    calibration_summary,
    save_reliability_plot,
    subgroup_calibration_summary,
)
from src.config import (
    ACCEPTED_CSV,
    ARTIFACT_DIR,
    DEFAULT_ACCEPTED_BUNDLE,
    DEFAULT_LGD,
    DEFAULT_TARGET_HORIZON_MONTHS,
    DEFAULT_TARGET_MODE,
    FORBIDDEN_FEATURE_COLUMNS,
    PRODUCT_MODE_POST_PRICING,
    PROFIT_INPUT_COLUMNS,
    REPORT_DIR,
    ROOT,
    TARGET,
)
from src.models import fit_model, predict_raw_default
from src.preprocessing import (
    feature_columns_for_product_mode,
    prepare_accepted_loans,
    split_chronological,
    split_count_report,
    split_manifest,
    split_row_count_report,
    split_summary,
)
from src.profit import policy_metrics, policy_sensitivity
from src.profit import expected_profit, expected_return
from src.reporting import generate_report_suite


BASE_REQUIRED_RETURN_CANDIDATES = [0.00, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20]
GOOD_PROFIT_HAIRCUT_CANDIDATES = [0.25, 0.35, 0.50, 0.65, 0.80, 1.00]
DEFAULT_MODEL_CANDIDATES = ["logistic_balanced", "logistic", "random_forest", "hist_gradient_boosting"]


def _read_accepted(path, feature_columns, sample=None, target_mode=DEFAULT_TARGET_MODE):
    needed = set(feature_columns + PROFIT_INPUT_COLUMNS)
    needed |= {"id", "term", "loan_status", "issue_d", "last_pymnt_d", "total_pymnt"}
    if target_mode == "default_within_horizon":
        needed |= {"issue_d", "last_pymnt_d", "loan_status"}
    return pd.read_csv(path, usecols=lambda col: col in needed, low_memory=False, nrows=sample)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _repo_path(path):
    p = Path(path).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def _repo_fingerprint(fingerprint):
    out = dict(fingerprint)
    if out.get("path"):
        out["path"] = _repo_path(out["path"])
    return out


def _paths(output_path, sample):
    report_dir = REPORT_DIR / "validation" / "smoke" if sample else REPORT_DIR / "validation"
    output = DEFAULT_ACCEPTED_BUNDLE.with_name("accepted_model_smoke.joblib") if sample else output_path
    return output, report_dir


def _policy_thresholds(validation_df, p_default, lgd=DEFAULT_LGD, good_profit_haircut: float = 1.0) -> list[float]:
    inputs = validation_df[PROFIT_INPUT_COLUMNS].apply(pd.to_numeric, errors="coerce")
    ep = expected_profit(
        p_default,
        inputs["funded_amnt"],
        inputs["term_months"],
        inputs["installment"],
        lgd=lgd,
        good_profit_haircut=good_profit_haircut,
    )
    er = pd.Series(expected_return(ep, inputs["funded_amnt"])).replace([np.inf, -np.inf], np.nan).dropna()
    values = list(BASE_REQUIRED_RETURN_CANDIDATES)
    if not er.empty:
        values += [float(er.quantile(q)) for q in (0.70, 0.80, 0.90, 0.95)]
        values.append(float(er.max()) + 1e-9)
    return sorted({round(float(v), 12) for v in values if float(v) >= 0})


def select_required_return_policy(validation_df, p_default, lgd=DEFAULT_LGD) -> tuple[dict, list[dict]]:
    rows = []
    for good_profit_haircut in GOOD_PROFIT_HAIRCUT_CANDIDATES:
        for threshold in _policy_thresholds(
            validation_df,
            p_default,
            lgd=lgd,
            good_profit_haircut=good_profit_haircut,
        ):
            metrics = policy_metrics(
                validation_df,
                p_default,
                lgd=lgd,
                required_return=threshold,
                good_profit_haircut=good_profit_haircut,
            )
            rows.append(
                {
                    "good_profit_haircut": good_profit_haircut,
                    "required_return": threshold,
                    "approval_count": metrics["approval_count"],
                    "selection_rate": metrics["selection_rate"],
                    "expected_profit": metrics["expected_profit"],
                    "expected_return": metrics["expected_return"],
                    "actual_default_rate": metrics.get("actual_default_rate"),
                    "total_realized_profit": metrics.get("total_realized_profit", 0.0),
                    "mean_realized_profit": metrics.get("mean_realized_profit", 0.0),
                }
            )
    eligible = [
        row
        for row in rows
        if row["approval_count"] > 0
        and row["required_return"] >= 0
        and row["expected_profit"] >= 0
        and row["expected_return"] >= 0
        and row["total_realized_profit"] > 0
    ]
    selected = max(
        eligible or [row for row in rows if row["approval_count"] == 0],
        key=lambda r: (
            r["total_realized_profit"],
            r["expected_profit"],
            -(r["actual_default_rate"] or 0.0),
            r["approval_count"],
        ),
    )
    policy = {
        "lgd": lgd,
        "good_profit_haircut": selected["good_profit_haircut"],
        "required_return": selected["required_return"],
        "approval_rule": "expected_return > required_return",
        "use_npv": False,
        "annual_discount_rate": 0.08,
        "servicing_cost_rate": 0.0,
        "recovery_rate": 0.25,
        "selection_rule": (
            "max validation total_realized_profit among non-empty policies with non-negative scenario expected_profit, "
            "non-negative expected_return, non-negative required_return, and positive realized validation profit; "
            "tie-break expected_profit, approved default rate, approval count"
        ),
        "validation_expected_profit": selected["expected_profit"],
        "validation_realized_profit": selected["total_realized_profit"],
        "validation_approval_count": selected["approval_count"],
        "validation_selection_rate": selected["selection_rate"],
        "validation_approved_default_rate": selected["actual_default_rate"],
    }
    if selected["approval_count"] == 0:
        policy["profit_warning"] = "no non-empty validation policy passed non-negative scenario EV and positive realized-profit gates"
    return policy, rows


def select_accepted_candidate(train_df, calibration_df, validation_df, feature_columns, product_mode):
    candidates = []
    for name in DEFAULT_MODEL_CANDIDATES:
        model = fit_model(
            train_df,
            feature_columns,
            kind="accepted",
            candidate_name=name,
            product_mode=product_mode,
        )
        raw_calibration = predict_raw_default(model, calibration_df, feature_columns)
        calibrator = ProbabilityCalibrator("isotonic").fit(raw_calibration, calibration_df[TARGET])
        raw_validation = predict_raw_default(model, validation_df, feature_columns)
        p_validation = calibrator.predict(raw_validation)
        probability = calibration_summary(validation_df[TARGET], p_validation)
        policy, policy_candidates = select_required_return_policy(validation_df, p_validation, lgd=DEFAULT_LGD)
        profit = policy_metrics(
            validation_df,
            p_validation,
            lgd=DEFAULT_LGD,
            required_return=policy["required_return"],
            good_profit_haircut=policy["good_profit_haircut"],
        )
        candidates.append(
            {
                "name": name,
                "model": model,
                "calibrator": calibrator,
                "probability": probability,
                "selected_policy": policy,
                "policy_candidates": policy_candidates,
                "profit_policy": profit,
            }
        )
    return max(
        candidates,
        key=lambda c: (
            c["profit_policy"].get("total_realized_profit", 0.0),
            c["profit_policy"]["expected_profit"],
            -c["probability"]["brier_score"],
        ),
    ), candidates


def train_accepted_model(
    csv_path=ACCEPTED_CSV,
    output_path=DEFAULT_ACCEPTED_BUNDLE,
    sample=None,
    target_mode: str = DEFAULT_TARGET_MODE,
    horizon_months: int = DEFAULT_TARGET_HORIZON_MONTHS,
    product_mode: str = PRODUCT_MODE_POST_PRICING,
):
    output_path, report_dir = _paths(output_path, sample)
    report_dir.mkdir(parents=True, exist_ok=True)
    source_fingerprint = file_fingerprint(csv_path)
    training_timestamp = datetime.now(timezone.utc).isoformat()
    versions = package_versions()
    feature_columns = feature_columns_for_product_mode(product_mode)
    accepted, target_summary = prepare_accepted_loans(
        _read_accepted(csv_path, feature_columns, sample=sample, target_mode=target_mode),
        target_mode=target_mode,
        target_config={"horizon_months": horizon_months},
        return_summary=True,
    )
    splits = split_chronological(accepted)
    manifest = split_manifest(splits)
    split_row_count_report(splits).to_csv(report_dir / "supervised_row_counts_by_split_year.csv", index=False)
    split_count_report({k: v for k, v in splits.items() if k != "test"}).to_csv(
        report_dir / "non_test_label_counts_by_split_year.csv", index=False
    )

    train_df = splits["train"].dropna(subset=[TARGET])
    calibration_df = splits["calibration"].dropna(subset=[TARGET])
    validation_df = splits["validation"].dropna(subset=[TARGET])

    selected, candidates = select_accepted_candidate(
        train_df,
        calibration_df,
        validation_df,
        feature_columns,
        product_mode,
    )
    raw_selected = predict_raw_default(selected["model"], validation_df, feature_columns)
    p_selected = selected["calibrator"].predict(raw_selected)
    probability_summary = calibration_summary(validation_df[TARGET], p_selected)
    bundle = ModelBundle(
        model=selected["model"],
        calibrator=selected["calibrator"],
        feature_columns=list(feature_columns),
        model_type="accepted",
        policy=selected["selected_policy"],
        required_input_schema={
            "risk_features": list(feature_columns),
            "profit_inputs": list(PROFIT_INPUT_COLUMNS),
        },
        metadata={
            "target": TARGET,
            "target_mode": target_mode,
            "target_definition": "resolved funded accepted-loan default target"
            if target_mode == "resolved_default"
            else f"default within {horizon_months} months using issue_d and last_pymnt_d as conservative observation proxies",
            "target_limitation": "predicts eventual resolved default among accepted/funded loans, not all applicant risk",
            "target_summary": target_summary,
            "fixed_horizon_extension": "conservative fixed-horizon labeling uses issue_d plus last_pymnt_d and excludes censored rows",
            "is_smoke_sample": bool(sample),
            "sample_rows_requested": sample,
            "source_fingerprint": _repo_fingerprint(source_fingerprint),
            "resolved_row_count": int(len(accepted)),
            "split_manifest": manifest,
            "split_summary": split_summary(splits),
            "selected_candidate": selected["name"],
            "selection_rule": (
                "select calibrated model, good_profit_haircut, and required_return on validation total_realized_profit "
                "after non-negative scenario-EV gates; tie-break expected_profit and brier_score"
            ),
            "target_good_statuses": "see src.config.GOOD_STATUSES",
            "target_bad_statuses": "see src.config.BAD_STATUSES",
            "forbidden_feature_columns": sorted(FORBIDDEN_FEATURE_COLUMNS),
            "product_mode": product_mode,
            "scoring_moment": "post-pricing/post-underwriting only",
            "random_state": 42,
            "package_versions": versions,
            "split_rule": "chronological 60/15/15/10 by issue_d after dropping unresolved loans",
            "rejected_data_handling": "rejected applications are unlabeled and excluded",
            "calibration_method": "isotonic",
            "training_timestamp": training_timestamp,
        },
    )
    saved = save_model_bundle(bundle, output_path)
    validation_report = {
        "is_smoke_sample": bool(sample),
        "sample_rows_requested": sample,
        "artifact_path": str(saved),
        "training_timestamp": training_timestamp,
        "package_versions": versions,
        "source_fingerprint": source_fingerprint,
        "resolved_row_count": int(len(accepted)),
        "target_summary": target_summary,
        "split_manifest": {k: v for k, v in manifest.items() if k != "test_ids"},
        "split_summary": split_summary(splits),
        "feature_columns": list(feature_columns),
        "selected_candidate": selected["name"],
        "selection_rule": bundle.metadata["selection_rule"],
        "selected_policy": bundle.policy,
        "selected_subgroup_calibration": subgroup_calibration_summary(validation_df, TARGET, p_selected),
        "selected_policy_sensitivity": policy_sensitivity(
            validation_df,
            p_selected,
            good_profit_haircuts=(selected["selected_policy"]["good_profit_haircut"],),
        ),
        "candidates": [
            {
                "name": c["name"],
                "probability": c["probability"],
                "selected_policy": c["selected_policy"],
                "profit_policy": c["profit_policy"],
            }
            for c in candidates
        ],
    }
    _write_json(report_dir / "accepted_validation_metrics.json", validation_report)
    pd.DataFrame(
        [
            {
                "model_name": c["name"],
                **row,
                "selected_model": c["name"] == selected["name"],
                "selected": c["name"] == selected["name"]
                and row["required_return"] == c["selected_policy"]["required_return"]
                and row["good_profit_haircut"] == c["selected_policy"]["good_profit_haircut"],
            }
            for c in candidates
            for row in c["policy_candidates"]
        ]
    ).to_csv(report_dir / "policy_selection_validation.csv", index=False)
    generate_report_suite(
        bundle,
        validation_df,
        p_selected,
        output_dir=report_dir,
        stage="validation",
        include_sensitivity=True,
        stage_summary={
            "row_count": int(len(accepted)),
            "split_summary": split_summary(splits),
            "roc_auc": probability_summary.get("roc_auc"),
            "pr_auc": probability_summary.get("pr_auc"),
            "brier_score": probability_summary.get("brier_score"),
            "mean_predicted_default": probability_summary.get("mean_predicted_default"),
            "actual_default_rate": probability_summary.get("actual_default_rate"),
        },
    )
    _write_json(
        ROOT / "docs" / "accepted_model_card.json",
        {
            "model_type": bundle.model_type,
            "target": bundle.metadata["target_definition"],
            "target_limitation": bundle.metadata["target_limitation"],
            "scoring_moment": bundle.metadata["scoring_moment"],
            "artifact_path": _repo_path(saved),
            "feature_columns": bundle.feature_columns,
            "policy": bundle.policy,
            "selected_candidate": selected["name"],
            "selection_rule": bundle.metadata["selection_rule"],
            "split_summary": validation_report["split_summary"],
            "source_fingerprint": _repo_fingerprint(source_fingerprint),
            "training_timestamp": training_timestamp,
            "package_versions": versions,
        },
    )
    save_reliability_plot(validation_df[TARGET], p_selected, report_dir / "accepted_reliability.png")
    return saved


def main():
    parser = argparse.ArgumentParser(description="Train accepted-loan default/profit model.")
    parser.add_argument("--csv", default=ACCEPTED_CSV)
    parser.add_argument("--output", default=DEFAULT_ACCEPTED_BUNDLE)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--target-mode", default=DEFAULT_TARGET_MODE, choices=["resolved_default", "default_within_horizon"])
    parser.add_argument("--horizon-months", type=int, default=DEFAULT_TARGET_HORIZON_MONTHS)
    parser.add_argument("--product-mode", default=PRODUCT_MODE_POST_PRICING, choices=[PRODUCT_MODE_POST_PRICING, "pre_underwriting_applicant"])
    args = parser.parse_args()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    print(
        train_accepted_model(
            args.csv,
            args.output,
            sample=args.sample,
            target_mode=args.target_mode,
            horizon_months=args.horizon_months,
            product_mode=args.product_mode,
        )
    )


if __name__ == "__main__":
    main()
