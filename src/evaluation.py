from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import load_model_bundle
from .calibration import calibration_summary
from .config import ACCEPTED_CSV, DEFAULT_ACCEPTED_BUNDLE, REPORT_DIR, TARGET
from .models import predict_raw_default
from .preprocessing import prepare_accepted_loans, split_chronological

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _group_calibration(frame: pd.DataFrame, p_default, column: str) -> pd.DataFrame:
    grouped = (
        pd.DataFrame({column: frame[column], "p_default": p_default, TARGET: frame[TARGET]})
        .dropna(subset=[column])
        .groupby(column, as_index=False)
        .agg(
            count=(TARGET, "size"),
            mean_predicted_default=("p_default", "mean"),
            observed_default_rate=(TARGET, "mean"),
        )
    )
    if grouped.empty:
        return grouped
    grouped["absolute_calibration_gap"] = (
        grouped["mean_predicted_default"] - grouped["observed_default_rate"]
    ).abs()
    return grouped


def _risk_decile_lift(frame: pd.DataFrame, p_default, bins: int = 10) -> pd.DataFrame:
    data = pd.DataFrame({TARGET: frame[TARGET], "p_default": p_default}).dropna()
    if data.empty:
        return pd.DataFrame(
            columns=[
                "predicted_risk_decile",
                "count",
                "observed_default_rate",
                "lift_versus_portfolio_default_rate",
                "cumulative_share_of_defaults_captured",
            ]
        )
    try:
        raw_decile = pd.qcut(
            data["p_default"].rank(method="first"),
            q=min(bins, len(data)),
            labels=False,
            duplicates="drop",
        )
        data["predicted_risk_decile"] = int(raw_decile.max()) + 1 - raw_decile.astype(int)
    except ValueError:
        data["predicted_risk_decile"] = 1
    grouped = (
        data.groupby("predicted_risk_decile", as_index=False)
        .agg(count=(TARGET, "size"), defaults=(TARGET, "sum"), observed_default_rate=(TARGET, "mean"))
        .sort_values("predicted_risk_decile")
    )
    portfolio_default_rate = float(data[TARGET].mean()) if len(data) else 0.0
    total_defaults = float(grouped["defaults"].sum()) if len(grouped) else 0.0
    grouped["lift_versus_portfolio_default_rate"] = grouped["observed_default_rate"] / portfolio_default_rate if portfolio_default_rate else np.nan
    grouped["cumulative_share_of_defaults_captured"] = grouped["defaults"].cumsum() / total_defaults if total_defaults else 0.0
    return grouped.drop(columns=["defaults"])


def _curve_frames(y_true, p_default) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = pd.Series(y_true).astype(int)
    p = pd.Series(p_default).astype(float)
    if y.nunique() < 2:
        empty_roc = pd.DataFrame(columns=["fpr", "tpr", "threshold"])
        empty_pr = pd.DataFrame(columns=["precision", "recall", "threshold"])
        return empty_roc, empty_pr
    fpr, tpr, roc_thresholds = roc_curve(y, p)
    precision, recall, pr_thresholds = precision_recall_curve(y, p)
    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_thresholds})
    pr_df = pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
            "threshold": np.append(pr_thresholds, np.nan),
        }
    )
    return roc_df, pr_df


def _plot_reliability(path: Path, calibration_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.6", linewidth=1)
    if len(calibration_df):
        ax.plot(calibration_df["mean_predicted_default"], calibration_df["observed_default_rate"], marker="o")
    ax.set_xlabel("Mean predicted default")
    ax.set_ylabel("Observed default rate")
    ax.set_title("Reliability Plot")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_curve(path: Path, frame: pd.DataFrame, x: str, y: str, title: str, xlabel: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    if len(frame):
        ax.plot(frame[x], frame[y], linewidth=2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _model_card(bundle, stage: str, metrics_summary: dict) -> str:
    metadata = bundle.metadata or {}
    validation_metrics = metadata.get("validation_metrics_summary", {})
    locked_test_metrics = metrics_summary if stage == "test" else metadata.get("locked_test_metrics_summary", {})
    return "\n".join(
        [
            "# Model Card",
            "",
            "## Intended Use",
            "Calibrated default-risk prediction for accepted/funded LendingClub loans with resolved outcomes.",
            "",
            "## Not Intended Use",
            "This project is not production underwriting, fair-lending validation, or rejected-applicant outcome prediction.",
            "",
            "## Target Definition",
            f"- Target: `{metadata.get('target_name', TARGET)}`",
            f"- Good statuses: `{', '.join(metadata.get('good_statuses', []))}`",
            f"- Bad statuses: `{', '.join(metadata.get('bad_statuses', []))}`",
            f"- Dropped statuses: `{', '.join(metadata.get('dropped_statuses', []))}`",
            "",
            "## Data Limitations",
            f"- Rejected applications: `{metadata.get('rejected_data_handling')}`",
            f"- Limitations: `{'; '.join(metadata.get('limitations', []))}`",
            "",
            "## Leakage Controls",
            f"- Forbidden columns: `{', '.join(metadata.get('forbidden_feature_columns', []))}`",
            "",
            "## Split Strategy",
            f"- Split summary: `{json.dumps(metadata.get('split_summary', []), default=str)}`",
            "",
            "## Cross Validation",
            f"- Summary: `{json.dumps(metadata.get('cross_validation_summary', {}), default=str)}`",
            "",
            "## Calibration Method",
            f"- Method: `{metadata.get('calibration_method')}`",
            "",
            "## Validation Performance",
            f"- Metrics: `{json.dumps(validation_metrics, default=str)}`",
            "",
            "## Locked Test Performance",
            f"- Metrics: `{json.dumps(locked_test_metrics, default=str)}`",
            "",
            "## API Usage",
            "- `GET /health`",
            "- `GET /ready`",
            "- `GET /model-card`",
            "- `GET /frontend-config`",
            "- `POST /score`",
            "- `POST /score-frontend`",
            "- `POST /score-batch`",
            "",
            "## Frontend Fields",
            f"- Reduced-feature model fields: `{', '.join(metadata.get('frontend_fields', []))}`",
            "",
            "## Known Risks",
            "- accepted-loan selection bias",
            "- unresolved outcomes are excluded rather than inferred",
            "- no fair-lending validation",
        ]
    )


def generate_evaluation_reports(bundle, frame: pd.DataFrame, p_default, output_dir: str | Path, stage: str) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = calibration_summary(frame[TARGET], p_default)
    metrics_summary = {
        "rows": int(len(frame)),
        "observed_default_rate": summary["actual_default_rate"],
        "mean_predicted_default_rate": summary["mean_predicted_default"],
        "roc_auc": summary["roc_auc"],
        "pr_auc": summary["pr_auc"],
        "brier_score": summary["brier_score"],
        "log_loss": summary["log_loss"],
        "selected_model": bundle.metadata.get("selected_model_name"),
        "calibration_method": bundle.metadata.get("calibration_method"),
    }
    deciles = pd.DataFrame(summary["deciles"])
    lift = _risk_decile_lift(frame, p_default)
    by_issue_year = _group_calibration(frame, p_default, "issue_year") if "issue_year" in frame.columns else pd.DataFrame()
    by_grade = _group_calibration(frame, p_default, "grade") if "grade" in bundle.feature_columns and "grade" in frame.columns else pd.DataFrame()
    by_term = _group_calibration(frame, p_default, "term") if "term" in bundle.feature_columns and "term" in frame.columns else pd.DataFrame()
    roc_df, pr_df = _curve_frames(frame[TARGET], p_default)

    files = {
        "metrics_summary": output_dir / "metrics_summary.json",
        "calibration_deciles": output_dir / "calibration_deciles.csv",
        "risk_decile_lift": output_dir / "risk_decile_lift.csv",
        "calibration_by_issue_year": output_dir / "calibration_by_issue_year.csv",
        "roc_curve": output_dir / "roc_curve.csv",
        "pr_curve": output_dir / "pr_curve.csv",
        "reliability_plot": output_dir / "reliability_plot.png",
        "roc_curve_plot": output_dir / "roc_curve.png",
        "pr_curve_plot": output_dir / "pr_curve.png",
        "model_card": output_dir / "model_card.md",
    }
    if not by_grade.empty:
        files["calibration_by_grade"] = output_dir / "calibration_by_grade.csv"
    if not by_term.empty:
        files["calibration_by_term"] = output_dir / "calibration_by_term.csv"

    _write_json(files["metrics_summary"], metrics_summary)
    deciles.to_csv(files["calibration_deciles"], index=False)
    lift.to_csv(files["risk_decile_lift"], index=False)
    by_issue_year.to_csv(files["calibration_by_issue_year"], index=False)
    if not by_grade.empty:
        by_grade.to_csv(files["calibration_by_grade"], index=False)
    if not by_term.empty:
        by_term.to_csv(files["calibration_by_term"], index=False)
    roc_df.to_csv(files["roc_curve"], index=False)
    pr_df.to_csv(files["pr_curve"], index=False)
    _plot_reliability(files["reliability_plot"], deciles)
    _plot_curve(files["roc_curve_plot"], roc_df, "fpr", "tpr", "ROC Curve", "False Positive Rate", "True Positive Rate")
    _plot_curve(files["pr_curve_plot"], pr_df, "recall", "precision", "PR Curve", "Recall", "Precision")
    files["model_card"].write_text(_model_card(bundle, stage, metrics_summary), encoding="utf-8")

    if stage == "validation":
        bundle.metadata["validation_metrics_summary"] = metrics_summary
    elif stage == "test":
        bundle.metadata["locked_test_metrics_summary"] = metrics_summary
    return files


def evaluate_bundle_on_split(bundle_path=DEFAULT_ACCEPTED_BUNDLE, csv_path=ACCEPTED_CSV, stage: str = "validation", sample: int | None = None) -> dict[str, Path]:
    bundle = load_model_bundle(bundle_path)
    needed = set(bundle.feature_columns) | {"id", "loan_status", "issue_d"}
    source = pd.read_csv(csv_path, usecols=lambda col: col in needed, low_memory=False, nrows=sample)
    prepared = prepare_accepted_loans(source)
    splits = split_chronological(prepared)
    frame = splits[stage].copy() if stage != "test" else splits["test"].copy()
    raw = predict_raw_default(bundle.model, frame, bundle.feature_columns)
    p_default = bundle.calibrator.predict(raw)
    return generate_evaluation_reports(bundle, frame, p_default, REPORT_DIR / stage, stage)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Generate default-risk evaluation reports from a saved bundle.")
    parser.add_argument("--bundle", default=DEFAULT_ACCEPTED_BUNDLE)
    parser.add_argument("--csv", default=ACCEPTED_CSV)
    parser.add_argument("--stage", choices=["validation", "test"], default="validation")
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args(argv)
    outputs = evaluate_bundle_on_split(args.bundle, args.csv, stage=args.stage, sample=args.sample)
    print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))
    return 0
