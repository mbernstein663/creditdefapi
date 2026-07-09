from __future__ import annotations

import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.artifacts import ModelBundle, file_fingerprint, package_versions, save_model_bundle
from src.calibration import ProbabilityCalibrator, calibration_summary
from src.config import (
    ACCEPTED_CSV,
    ARTIFACT_DIR,
    DEFAULT_ACCEPTED_BUNDLE,
    DEFAULT_CALIBRATION_METHODS,
    DEFAULT_FRONTEND_BUNDLE,
    DEFAULT_PREPROCESSED_ACCEPTED_BUNDLE,
    DEFAULT_TRAINING_MODELS,
    FORBIDDEN_FEATURE_COLUMNS,
    FRONTEND_TOP_FEATURE_COUNT,
    MODEL_VERSION,
    REPORT_DIR,
    TARGET,
    load_training_config,
)
from src.evaluation import generate_evaluation_reports
from src.models import fit_model, predict_raw_default
from src.preprocessing import feature_columns

from .preprocessing import load_preprocessed_accepted_loans, preprocess_accepted_loans, save_preprocessed_accepted_loans


def _paths(output_path, sample):
    report_dir = REPORT_DIR / "smoke" / "validation" if sample else REPORT_DIR / "validation"
    output = DEFAULT_ACCEPTED_BUNDLE.with_name("accepted_model_smoke.joblib") if sample else Path(output_path)
    return output, report_dir


def _frontend_output_path(output_path: Path, sample):
    return (
        DEFAULT_FRONTEND_BUNDLE.with_name("frontend_model_smoke.joblib")
        if sample
        else output_path.with_name("frontend_model.joblib")
    )


def _artifact_data_context(sample: int | None) -> str:
    return "smoke_sample" if sample else "full_lendingclub_local"


def _load_or_build_preprocessed(csv_path, sample, selected_features):
    try:
        cached = load_preprocessed_accepted_loans(DEFAULT_PREPROCESSED_ACCEPTED_BUNDLE)
        if cached.source_fingerprint == file_fingerprint(csv_path) and getattr(cached, "sample_rows_requested", None) == sample:
            return cached
    except Exception:
        pass
    built = preprocess_accepted_loans(csv_path, sample=sample, selected_features=selected_features)
    save_preprocessed_accepted_loans(built, DEFAULT_PREPROCESSED_ACCEPTED_BUNDLE)
    return built


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def _candidate_rank(candidate: dict, model_candidates: list[str]) -> tuple[float, float, float, float, float, float, float, int]:
    def _as_float(value, default):
        return float(default if value is None else value)

    metrics = candidate["metrics"]
    cv_metrics = candidate.get("cv_metrics") or {}
    deciles = metrics["deciles"]
    mean_gap = float(pd.DataFrame(deciles)["absolute_calibration_gap"].mean()) if deciles else float("inf")
    return (
        mean_gap,
        _as_float(metrics["brier_score"], float("inf")),
        _as_float(metrics["log_loss"], float("inf")),
        _as_float(cv_metrics.get("mean_absolute_calibration_gap"), float("inf")),
        _as_float(cv_metrics.get("mean_brier_score"), float("inf")),
        -_as_float(metrics["roc_auc"], float("-inf")),
        -_as_float(metrics["pr_auc"], float("-inf")),
        model_candidates.index(candidate["model_name"]),
    )


def _risk_band_thresholds(p_default) -> dict:
    values = pd.Series(p_default).astype(float)
    return {
        "low_max": round(float(values.quantile(1 / 3)), 6),
        "medium_max": round(float(values.quantile(2 / 3)), 6),
    }


def _permutation_feature_importance(model, calibrator, frame: pd.DataFrame, selected_features: list[str]) -> list[dict]:
    baseline_raw = predict_raw_default(model, frame, selected_features)
    baseline_p = calibrator.predict(baseline_raw)
    actual = frame[TARGET].to_numpy(dtype=float)
    baseline_brier = float(np.mean((np.asarray(baseline_p, dtype=float) - actual) ** 2))
    rows = []
    for column in selected_features:
        permuted = frame.copy()
        permuted[column] = permuted[column].sample(frac=1, random_state=42).to_numpy()
        raw = predict_raw_default(model, permuted, selected_features)
        p_default = calibrator.predict(raw)
        permuted_brier = float(np.mean((np.asarray(p_default, dtype=float) - actual) ** 2))
        rows.append(
            {
                "feature": column,
                "importance": permuted_brier - baseline_brier,
            }
        )
    return sorted(rows, key=lambda row: abs(row["importance"]), reverse=True)


def _cross_validation_summary(train_df: pd.DataFrame, selected_features: list[str], model_name: str) -> dict:
    ordered = train_df.sort_values(["issue_dt", "id"] if "id" in train_df.columns else ["issue_dt"]).reset_index(drop=True)
    if len(ordered) < 8 or ordered["issue_dt"].dropna().nunique() < 4:
        return {"fold_count": 0, "folds": []}

    n_splits = min(4, max(2, len(ordered) // 10))
    splitter = TimeSeriesSplit(n_splits=n_splits)
    fold_rows = []
    for fold_number, (train_idx, validation_idx) in enumerate(splitter.split(ordered), start=1):
        fold_train = ordered.iloc[train_idx]
        fold_validation = ordered.iloc[validation_idx]
        if fold_train[TARGET].nunique() < 2 or fold_validation[TARGET].nunique() < 2:
            continue
        model = fit_model(fold_train, selected_features, model_name)
        raw_validation = predict_raw_default(model, fold_validation, selected_features)
        summary = calibration_summary(fold_validation[TARGET], raw_validation)
        fold_rows.append(
            {
                "fold": fold_number,
                "train_rows": int(len(fold_train)),
                "validation_rows": int(len(fold_validation)),
                "roc_auc": summary["roc_auc"],
                "pr_auc": summary["pr_auc"],
                "brier_score": summary["brier_score"],
                "log_loss": summary["log_loss"],
                "mean_predicted_default": summary["mean_predicted_default"],
                "actual_default_rate": summary["actual_default_rate"],
                "mean_absolute_calibration_gap": float(
                    pd.DataFrame(summary["deciles"])["absolute_calibration_gap"].mean()
                )
                if summary["deciles"]
                else None,
            }
        )
    if not fold_rows:
        return {"fold_count": 0, "folds": []}

    fold_frame = pd.DataFrame(fold_rows)
    summary = {
        "fold_count": int(len(fold_rows)),
        "folds": fold_rows,
        "mean_roc_auc": float(fold_frame["roc_auc"].mean()) if fold_frame["roc_auc"].notna().any() else None,
        "mean_pr_auc": float(fold_frame["pr_auc"].mean()) if fold_frame["pr_auc"].notna().any() else None,
        "mean_brier_score": float(fold_frame["brier_score"].mean()),
        "mean_log_loss": float(fold_frame["log_loss"].mean()) if fold_frame["log_loss"].notna().any() else None,
        "mean_absolute_calibration_gap": float(fold_frame["mean_absolute_calibration_gap"].mean())
        if fold_frame["mean_absolute_calibration_gap"].notna().any()
        else None,
        "mean_predicted_default": float(fold_frame["mean_predicted_default"].mean()),
        "mean_actual_default_rate": float(fold_frame["actual_default_rate"].mean()),
    }
    return summary


def _print_validation_summary(candidates: list[dict], selected: dict) -> None:
    comparison = pd.DataFrame(_candidate_comparison_rows(candidates)).sort_values(["brier_score", "log_loss"])
    print(
        "\n".join(
            [
                f"best model: {selected['model_name']} + {selected['calibration_method']}",
                f"validation summary: {selected['metrics']['actual_default_rate']:.4f} observed default rate, "
                f"{selected['metrics']['mean_predicted_default']:.4f} mean predicted default",
                comparison.to_string(index=False),
            ]
        )
    )


def select_candidate(
    train_df,
    calibration_df,
    validation_df,
    selected_features,
    model_candidates: list[str] | None = None,
    calibration_methods: list[str] | None = None,
    include_cv: bool = True,
    selected_model_name: str | None = None,
):
    model_candidates = list(model_candidates or DEFAULT_TRAINING_MODELS)
    calibration_methods = list(calibration_methods or DEFAULT_CALIBRATION_METHODS)
    candidates = []
    for model_name in model_candidates:
        cv_metrics = _cross_validation_summary(train_df, selected_features, model_name) if include_cv else {}
        model = fit_model(train_df, selected_features, model_name)
        raw_calibration = predict_raw_default(model, calibration_df, selected_features)
        raw_validation = predict_raw_default(model, validation_df, selected_features)
        for calibration_method in calibration_methods:
            calibrator = ProbabilityCalibrator(calibration_method).fit(raw_calibration, calibration_df[TARGET])
            p_validation = calibrator.predict(raw_validation)
            candidates.append(
                {
                    "model_name": model_name,
                    "calibration_method": calibration_method,
                    "model": model,
                    "calibrator": calibrator,
                    "p_validation": p_validation,
                    "metrics": calibration_summary(validation_df[TARGET], p_validation),
                    "cv_metrics": cv_metrics,
                }
            )
    selectable = [candidate for candidate in candidates if selected_model_name in {None, candidate["model_name"]}]
    selected = min(selectable, key=lambda candidate: _candidate_rank(candidate, model_candidates))
    return selected, candidates


def _candidate_comparison_rows(candidates: list[dict]) -> list[dict]:
    rows = []
    for candidate in candidates:
        metrics = candidate["metrics"]
        rows.append(
            {
                "model_name": candidate["model_name"],
                "calibration_method": candidate["calibration_method"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "brier_score": metrics["brier_score"],
                "log_loss": metrics["log_loss"],
                "mean_predicted_default_rate": metrics["mean_predicted_default"],
                "observed_default_rate": metrics["actual_default_rate"],
                "cv_mean_brier_score": candidate["cv_metrics"].get("mean_brier_score"),
                "cv_mean_log_loss": candidate["cv_metrics"].get("mean_log_loss"),
                "cv_mean_absolute_calibration_gap": candidate["cv_metrics"].get("mean_absolute_calibration_gap"),
                "cv_fold_count": candidate["cv_metrics"].get("fold_count"),
            }
        )
    return rows


def _build_bundle(
    selected,
    selected_features,
    target_summary,
    source_fingerprint,
    manifest,
    split_details,
    validation_row_count,
    forbidden_columns,
    versions,
    git_commit,
    training_timestamp,
    frontend_fields: list[str] | None = None,
    feature_importance: list[dict] | None = None,
    cross_validation_summary: dict | None = None,
    model_version: str = MODEL_VERSION,
    artifact_data_context: str = "full_lendingclub_local",
    sample_rows_requested: int | None = None,
):
    selected_metrics = selected["metrics"]
    return ModelBundle(
        model=selected["model"],
        calibrator=selected["calibrator"],
        feature_columns=list(selected_features),
        model_type=f"calibrated_{selected['model_name']}",
        required_input_schema={
            "schema_version": 1,
            "required_fields": list(selected_features),
            "forbidden_fields": forbidden_columns,
        },
        metadata={
            "bundle_schema_version": 1,
            "model_version": model_version,
            "target_name": TARGET,
            "target_definition": "resolved accepted/funded loan default target",
            "good_statuses": sorted(target_summary["good_statuses"]),
            "bad_statuses": sorted(target_summary["bad_statuses"]),
            "dropped_statuses": sorted(target_summary["dropped_statuses"]),
            "target_summary": target_summary,
            "source_fingerprint": source_fingerprint,
            "split_manifest": manifest,
            "split_summary": split_details,
            "split_date_boundaries": manifest["date_ranges"],
            "split_row_counts": manifest["row_counts"],
            "split_default_rates": manifest["default_rates"],
            "selected_model_name": selected["model_name"],
            "selected_model_type": f"calibrated_{selected['model_name']}",
            "calibration_method": selected["calibration_method"],
            "validation_metrics_summary": {
                "rows": int(validation_row_count),
                "observed_default_rate": selected_metrics["actual_default_rate"],
                "mean_predicted_default_rate": selected_metrics["mean_predicted_default"],
                "roc_auc": selected_metrics["roc_auc"],
                "pr_auc": selected_metrics["pr_auc"],
                "brier_score": selected_metrics["brier_score"],
                "log_loss": selected_metrics["log_loss"],
                "selected_model": selected["model_name"],
                "calibration_method": selected["calibration_method"],
            },
            "cross_validation_summary": cross_validation_summary or {},
            "risk_band_thresholds": _risk_band_thresholds(selected["p_validation"]),
            "feature_columns": list(selected_features),
            "forbidden_feature_columns": forbidden_columns,
            "feature_importance": feature_importance or [],
            "frontend_fields": frontend_fields or [],
            "limitations": [
                "accepted-loan selection bias",
                "rejected applications are unlabeled and excluded from supervised default modeling",
                "not validated for production underwriting or fair-lending use",
            ],
            "rejected_data_handling": "Rejected applications are not labeled as defaults or non-defaults and are excluded from supervised training, calibration, validation, and test evaluation.",
            "package_versions": versions,
            "git_commit": git_commit,
            "training_timestamp": training_timestamp,
            "artifact_data_context": artifact_data_context,
            "sample_rows_requested": sample_rows_requested,
        },
    )


def train_accepted_model(csv_path=ACCEPTED_CSV, output_path=DEFAULT_ACCEPTED_BUNDLE, sample=None, validation_only=False, config_path=None):
    output_path, report_dir = _paths(output_path, sample)
    frontend_output_path = _frontend_output_path(output_path, sample)
    report_dir.mkdir(parents=True, exist_ok=True)
    training_config = load_training_config(config_path)
    model_candidates = training_config["models"]
    selected_model_name = training_config["selected_model"]
    calibration_methods = training_config["calibration_methods"]
    include_cv = training_config["cross_validation"]
    selected_features = feature_columns()
    training_timestamp = datetime.now(timezone.utc).isoformat()
    versions = package_versions()
    git_commit = _git_commit()

    preprocessing = _load_or_build_preprocessed(csv_path, sample, selected_features)
    source_fingerprint = preprocessing.source_fingerprint
    target_summary = preprocessing.target_summary
    splits = preprocessing.splits
    manifest = preprocessing.manifest
    split_details = preprocessing.split_summary

    train_df = splits["train"].dropna(subset=[TARGET])
    calibration_df = splits["calibration"].dropna(subset=[TARGET])
    validation_df = splits["validation"].dropna(subset=[TARGET])
    selected, candidates = select_candidate(
        train_df,
        calibration_df,
        validation_df,
        selected_features,
        model_candidates=model_candidates,
        calibration_methods=calibration_methods,
        include_cv=include_cv,
        selected_model_name=selected_model_name,
    )
    forbidden_columns = sorted(FORBIDDEN_FEATURE_COLUMNS)
    pd.DataFrame(_candidate_comparison_rows(candidates)).to_csv(report_dir / "model_validation_results.csv", index=False)

    cross_validation_summary = {
        "enabled": include_cv,
        "selected_model_name": selected["model_name"],
        "selected_calibration_method": selected["calibration_method"],
        "candidate_summaries": [],
    }
    feature_importance = []
    frontend_fields = []
    if selected is not None and not validation_only:
        cross_validation_summary["candidate_summaries"] = [
            {
                "model_name": candidate["model_name"],
                "calibration_method": candidate["calibration_method"],
                **candidate["cv_metrics"],
            }
            for candidate in candidates
        ]
        feature_importance = _permutation_feature_importance(
            selected["model"],
            selected["calibrator"],
            validation_df,
            selected_features,
        )
        frontend_fields = [row["feature"] for row in feature_importance[:FRONTEND_TOP_FEATURE_COUNT]]

        frontend_selected, frontend_candidates = select_candidate(
            train_df,
            calibration_df,
            validation_df,
            frontend_fields,
            model_candidates=model_candidates,
            calibration_methods=calibration_methods,
            include_cv=include_cv,
            selected_model_name=selected_model_name,
        )
        frontend_cross_validation_summary = {
            "enabled": include_cv,
            "selected_model_name": frontend_selected["model_name"],
            "selected_calibration_method": frontend_selected["calibration_method"],
            "candidate_summaries": [
                {
                    "model_name": candidate["model_name"],
                    "calibration_method": candidate["calibration_method"],
                    **candidate["cv_metrics"],
                }
                for candidate in frontend_candidates
            ],
        }
        frontend_bundle = _build_bundle(
            frontend_selected,
            frontend_fields,
            target_summary,
            source_fingerprint,
            manifest,
            split_details,
            len(validation_df),
            forbidden_columns,
            versions,
            git_commit,
            training_timestamp,
            frontend_fields=frontend_fields,
            feature_importance=feature_importance,
            cross_validation_summary=frontend_cross_validation_summary,
            model_version=f"{MODEL_VERSION}-frontend",
            artifact_data_context=_artifact_data_context(sample),
            sample_rows_requested=sample,
        )
        save_model_bundle(frontend_bundle, frontend_output_path)

    bundle = _build_bundle(
        selected,
        selected_features,
        target_summary,
        source_fingerprint,
        manifest,
        split_details,
        len(validation_df),
        forbidden_columns,
        versions,
        git_commit,
        training_timestamp,
        frontend_fields=frontend_fields,
        feature_importance=feature_importance,
        cross_validation_summary=cross_validation_summary,
        artifact_data_context=_artifact_data_context(sample),
        sample_rows_requested=sample,
    )
    saved = save_model_bundle(bundle, output_path)
    generate_evaluation_reports(bundle, validation_df, selected["p_validation"], report_dir, "validation")
    _print_validation_summary(candidates, selected)
    print(f"saved best model: {selected['model_name']} + {selected['calibration_method']} -> {saved}")
    return saved


def main():
    parser = argparse.ArgumentParser(description="Train an accepted-loan calibrated default-risk model.")
    parser.add_argument("--csv", default=ACCEPTED_CSV)
    parser.add_argument("--output", default=DEFAULT_ACCEPTED_BUNDLE)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    print(
        train_accepted_model(
            args.csv,
            args.output,
            sample=args.sample,
            validation_only=args.validation_only,
            config_path=args.config,
        )
    )


if __name__ == "__main__":
    main()
