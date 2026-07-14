"""Evaluate a saved default-risk model bundle.

Runs the saved bundle on the validation or locked test split and writes a small
set of artifacts for the README/API.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from .artifacts import load_model_bundle
from .calibration import calibration_summary
from .config import ACCEPTED_CSV, DEFAULT_ACCEPTED_BUNDLE, REPORT_DIR, TARGET
from .models import fit_model, predict_raw_default
from .preprocessing import prepare_accepted_loans, split_chronological

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _risk_decile_lift(frame: pd.DataFrame, p_default, bins: int = 10) -> pd.DataFrame:
    columns = [
        "predicted_risk_decile",
        "count",
        "observed_default_rate",
        "lift_versus_portfolio_default_rate",
        "cumulative_share_of_defaults_captured",
    ]
    data = pd.DataFrame({TARGET: frame[TARGET], "p_default": p_default}).dropna()
    if data.empty:
        return pd.DataFrame(columns=columns)

    ranked = data["p_default"].rank(method="first", ascending=False)
    data["predicted_risk_decile"] = np.ceil(ranked / len(data) * bins).astype(int).clip(1, bins)

    grouped = (
        data.groupby("predicted_risk_decile", as_index=False)
        .agg(count=(TARGET, "size"), defaults=(TARGET, "sum"))
        .set_index("predicted_risk_decile")
        .reindex(range(1, bins + 1), fill_value=0)
        .reset_index()
    )
    grouped["observed_default_rate"] = np.where(grouped["count"] > 0, grouped["defaults"] / grouped["count"], 0.0)
    portfolio_rate = float(data[TARGET].mean())
    total_defaults = float(grouped["defaults"].sum())
    grouped["lift_versus_portfolio_default_rate"] = (
        grouped["observed_default_rate"] / portfolio_rate if portfolio_rate else np.nan
    )
    grouped["cumulative_share_of_defaults_captured"] = (
        grouped["defaults"].cumsum() / total_defaults if total_defaults else 0.0
    )
    return grouped.drop(columns="defaults")


def _curve_frames(y_true, p_default) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = pd.Series(y_true).astype(int)
    p = pd.Series(p_default).astype(float)
    if y.nunique() < 2:
        roc_df = pd.DataFrame(columns=["fpr", "tpr", "threshold"])
        pr_df = pd.DataFrame(columns=["precision", "recall", "threshold"])
        return roc_df, pr_df

    fpr, tpr, roc_thresholds = roc_curve(y, p)
    precision, recall, pr_thresholds = precision_recall_curve(y, p)
    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_thresholds})
    pr_df = pd.DataFrame({
        "precision": precision,
        "recall": recall,
        "threshold": np.append(pr_thresholds, np.nan),
    })
    return roc_df, pr_df


def _plot_reliability(path: Path, calibration_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.6", linewidth=1)
    if len(calibration_df):
        ax.plot(
            calibration_df["mean_predicted_default"],
            calibration_df["observed_default_rate"],
            marker="o",
        )
    ax.set_title("Reliability Plot")
    ax.set_xlabel("Mean predicted default")
    ax.set_ylabel("Observed default rate")
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


def _fmt(value) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def _metric_row(model_name: str, y_true, p_default, model_role: str) -> dict:
    y = pd.Series(y_true).reset_index(drop=True)
    p = pd.Series(p_default).reset_index(drop=True)
    summary = calibration_summary(y, p)
    return {
        "model_name": model_name,
        "model_role": model_role,
        "rows": int(len(y.dropna())),
        "observed_default_rate": summary["actual_default_rate"],
        "mean_predicted_default_rate": summary["mean_predicted_default"],
        "roc_auc": summary["roc_auc"],
        "pr_auc": summary["pr_auc"],
        "brier_score": summary["brier_score"],
        "log_loss": summary["log_loss"],
    }


def _markdown_table(rows: list[dict], columns: list[tuple[str, str]]) -> list[str]:
    if not rows:
        return ["not recorded"]
    header = "| " + " | ".join(label for _, label in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, divider]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(key)) for key, _ in columns) + " |")
    return lines


def _artifact_status(metadata: dict) -> tuple[str, str]:
    context = metadata.get("artifact_data_context")
    if context == "full_lendingclub_local":
        return (
            "Full LendingClub local data",
            "Example local-run evidence from user-supplied raw LendingClub files. Raw data is not committed.",
        )
    if context == "smoke_sample":
        sample_rows = metadata.get("sample_rows_requested")
        sample_note = f" Sample rows requested: {sample_rows}." if sample_rows else ""
        return (
            "Smoke-test sample",
            "Smoke-test artifact from a sampled LendingClub run. Not final model evidence." + sample_note,
        )
    if context == "synthetic_test_fixture":
        return (
            "Synthetic/test fixture",
            "Synthetic or test-fixture artifact used for CI or unit tests. Not LendingClub model evidence.",
        )
    return (
        "Unlabeled artifact",
        "Artifact context was not recorded. Treat results as non-final until rerun with current code.",
    )


def _split_rows(metadata: dict) -> list[str]:
    counts = metadata.get("split_row_counts") or {}
    if not counts:
        return ["- Split rows: `not recorded`"]
    order = ["train", "calibration", "validation", "test"]
    return [f"- {name.title()}: `{counts.get(name, 'n/a')}`" for name in order if name in counts]


def _baseline_table_rows(metrics: dict) -> list[dict]:
    return metrics.get("baseline_comparison") or []


def _model_card(bundle, stage: str, metrics: dict) -> str:
    metadata = bundle.metadata or {}
    artifact_label, artifact_note = _artifact_status(metadata)
    metric_lines = [
        f"- Rows: `{metrics['rows']}`",
        f"- Observed default rate: `{_fmt(metrics['observed_default_rate'])}`",
        f"- Mean predicted default rate: `{_fmt(metrics['mean_predicted_default_rate'])}`",
        f"- ROC-AUC: `{_fmt(metrics['roc_auc'])}`",
        f"- PR-AUC: `{_fmt(metrics['pr_auc'])}`",
        f"- Brier score: `{_fmt(metrics['brier_score'])}`",
        f"- Log loss: `{_fmt(metrics['log_loss'])}`",
    ]
    lines = [
        "# Model Card",
        "",
        "## Artifact Status",
        f"- Evidence label: `{artifact_label}`",
        f"- Evidence note: {artifact_note}",
        f"- Evaluation split: `{stage}`",
        f"- Training timestamp: `{metadata.get('training_timestamp', 'n/a')}`",
        "",
        "## Dataset Splits",
        "",
        *_split_rows(metadata),
        "",
        "## Model",
        "",
        f"- Selected model: `{metrics.get('selected_model')}`",
        f"- Calibration method: `{metrics.get('calibration_method')}`",
        f"- Feature count: `{len(getattr(bundle, 'feature_columns', []) or [])}`",
        "",
        f"## {stage.title()} Metrics",
        "",
        *metric_lines,
    ]
    if _baseline_table_rows(metrics):
        lines += [
            "",
            "## Baseline Comparison",
            "",
            *_markdown_table(
                _baseline_table_rows(metrics),
                [
                    ("model_name", "Model"),
                    ("model_role", "Role"),
                    ("roc_auc", "ROC-AUC"),
                    ("pr_auc", "PR-AUC"),
                    ("brier_score", "Brier"),
                    ("log_loss", "Log Loss"),
                    ("mean_predicted_default_rate", "Mean PD"),
                ],
            ),
        ]
    return "\n".join(lines) + "\n"


def generate_evaluation_reports(
    bundle,
    frame: pd.DataFrame,
    p_default,
    output_dir: str | Path,
    stage: str,
    baseline_comparison: list[dict] | None = None,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = bundle.metadata or {}
    summary = calibration_summary(frame[TARGET], p_default)
    artifact_label, artifact_note = _artifact_status(metadata)
    metrics = {
        "stage": stage,
        "evaluation_split": stage,
        "rows": int(len(frame)),
        "observed_default_rate": summary["actual_default_rate"],
        "mean_predicted_default_rate": summary["mean_predicted_default"],
        "roc_auc": summary["roc_auc"],
        "pr_auc": summary["pr_auc"],
        "brier_score": summary["brier_score"],
        "log_loss": summary["log_loss"],
        "selected_model": metadata.get("selected_model_name"),
        "model_type": metadata.get("selected_model_type", bundle.model_type),
        "calibration_method": metadata.get("calibration_method"),
        "target_name": metadata.get("target_name", TARGET),
        "training_timestamp": metadata.get("training_timestamp"),
        "artifact_data_context": metadata.get("artifact_data_context"),
        "artifact_label": artifact_label,
        "artifact_note": artifact_note,
        "sample_rows_requested": metadata.get("sample_rows_requested"),
        "split_row_counts": metadata.get("split_row_counts"),
        "split_summary": metadata.get("split_summary"),
        "limitations": metadata.get("limitations"),
        "baseline_comparison": baseline_comparison or [],
    }
    lift = _risk_decile_lift(frame, p_default)
    deciles = pd.DataFrame(summary["deciles"])
    roc_df, pr_df = _curve_frames(frame[TARGET], p_default)

    files = {
        "metrics_summary": output_dir / "metrics_summary.json",
        "calibration_deciles": output_dir / "calibration_deciles.csv",
        "risk_decile_lift": output_dir / "risk_decile_lift.csv",
        "roc_curve": output_dir / "roc_curve.csv",
        "pr_curve": output_dir / "pr_curve.csv",
        "baseline_comparison_csv": output_dir / "baseline_comparison.csv",
        "baseline_comparison_json": output_dir / "baseline_comparison.json",
        "reliability_plot": output_dir / "reliability_plot.png",
        "roc_curve_plot": output_dir / "roc_curve.png",
        "pr_curve_plot": output_dir / "pr_curve.png",
        "model_card": output_dir / "model_card.md",
    }

    _write_json(files["metrics_summary"], metrics)
    deciles.to_csv(files["calibration_deciles"], index=False)
    lift.to_csv(files["risk_decile_lift"], index=False)
    roc_df.to_csv(files["roc_curve"], index=False)
    pr_df.to_csv(files["pr_curve"], index=False)
    pd.DataFrame(baseline_comparison or []).to_csv(files["baseline_comparison_csv"], index=False)
    _write_json(files["baseline_comparison_json"], {"models": baseline_comparison or []})
    _plot_reliability(files["reliability_plot"], deciles)
    _plot_curve(files["roc_curve_plot"], roc_df, "fpr", "tpr", "ROC Curve", "False Positive Rate", "True Positive Rate")
    _plot_curve(files["pr_curve_plot"], pr_df, "recall", "precision", "Precision-Recall Curve", "Recall", "Precision")
    files["model_card"].write_text(_model_card(bundle, stage, metrics), encoding="utf-8")
    return files


def _base_rate_from_metadata(metadata: dict, fit_frame: pd.DataFrame) -> float:
    rates = metadata.get("split_default_rates") or {}
    for split in ["train", "calibration"]:
        if rates.get(split) is not None:
            return float(rates[split])
    return float(fit_frame[TARGET].mean())


def _group_rate_predictions(fit_frame: pd.DataFrame, test_frame: pd.DataFrame, column: str, fallback_rate: float):
    rates = fit_frame.groupby(column)[TARGET].mean()
    return test_frame[column].map(rates).fillna(fallback_rate).astype(float).to_numpy()


def baseline_comparison_metrics(bundle, splits: dict[str, pd.DataFrame], final_p_default) -> list[dict]:
    train_calibration = pd.concat([splits["train"], splits["calibration"]], ignore_index=True)
    test = splits["test"]
    base_rate = _base_rate_from_metadata(bundle.metadata or {}, train_calibration)
    selected_name = (bundle.metadata or {}).get("selected_model_name") or "selected_model"
    rows = [
        _metric_row(selected_name, test[TARGET], final_p_default, "final_model"),
        _metric_row("base_rate", test[TARGET], np.full(len(test), base_rate), "baseline"),
    ]

    if train_calibration[TARGET].nunique() >= 2:
        try:
            logistic = fit_model(train_calibration, bundle.feature_columns, "logistic")
            rows.append(
                _metric_row(
                    "logistic_regression",
                    test[TARGET],
                    predict_raw_default(logistic, test, bundle.feature_columns),
                    "baseline",
                )
            )
        except Exception as exc:
            rows.append({"model_name": "logistic_regression", "model_role": "baseline_unavailable", "error": str(exc)})

    for column in ["grade", "sub_grade"]:
        if column in train_calibration.columns and column in test.columns:
            rows.append(
                _metric_row(
                    f"{column}_historical_rate",
                    test[TARGET],
                    _group_rate_predictions(train_calibration, test, column, base_rate),
                    "baseline",
                )
            )
    return rows


def evaluate_bundle_on_split(
    bundle_path=DEFAULT_ACCEPTED_BUNDLE,
    csv_path=ACCEPTED_CSV,
    stage: str = "validation",
    sample: int | None = None,
) -> dict[str, Path]:
    bundle = load_model_bundle(bundle_path)
    needed = set(bundle.feature_columns) | {"id", "loan_status", "issue_d"}
    source = pd.read_csv(csv_path, usecols=lambda col: col in needed, low_memory=False, nrows=sample)
    prepared = prepare_accepted_loans(source)
    frame = split_chronological(prepared)[stage].copy()

    raw = predict_raw_default(bundle.model, frame, bundle.feature_columns)
    p_default = bundle.calibrator.predict(raw)
    return generate_evaluation_reports(bundle, frame, p_default, REPORT_DIR / stage, stage)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Generate compact default-risk evaluation reports.")
    parser.add_argument("--bundle", default=DEFAULT_ACCEPTED_BUNDLE)
    parser.add_argument("--csv", default=ACCEPTED_CSV)
    parser.add_argument("--stage", choices=["validation", "test"], default="validation")
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args(argv)

    outputs = evaluate_bundle_on_split(args.bundle, args.csv, args.stage, args.sample)
    print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
