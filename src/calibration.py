from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_mpl_config = Path(__file__).resolve().parents[1] / ".matplotlib"
_mpl_config.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_config))

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


@dataclass # decorator, modifies the class by adding dataclass
class ProbabilityCalibrator:
    method: str = "isotonic"
    model: Any | None = None

    """
    Applies class calibration 
    -> ingests raw probabilities and adjusts them to better match reality in calibration set
    """

    def fit(self, raw_probability, y):
        raw = np.asarray(raw_probability, dtype=float)
        target = np.asarray(y, dtype=int)
        if len(np.unique(target)) < 2:
            raise ValueError("calibration requires both classes")
        if self.method == "identity":
            self.model = None
            return self
        if self.method == "sigmoid":
            self.model = LogisticRegression(max_iter=2000, solver="liblinear").fit(raw.reshape(-1, 1), target)
            return self
        if self.method != "isotonic":
            raise ValueError(f"unknown calibration method: {self.method}")
        self.model = IsotonicRegression(out_of_bounds="clip").fit(raw, target)
        return self

    def predict(self, raw_probability):
        raw = np.asarray(raw_probability, dtype=float)
        if self.model is None:
            return np.clip(raw, 0, 1)
        if self.method == "sigmoid":
            return self.model.predict_proba(raw.reshape(-1, 1))[:, 1]
        return np.clip(self.model.predict(raw), 0, 1)

# Returns calibration metrics across bins as dictionary
def calibration_summary(y_true, p_default, bins: int = 10) -> dict:
    frame = pd.DataFrame({"actual": y_true, "predicted": p_default}).dropna()
    if frame.empty:
        raise ValueError("cannot summarize empty calibration frame")
    try:
        frame["decile"] = pd.qcut(
            frame["predicted"], q=min(bins, len(frame)), labels=False, duplicates="drop"
        ) + 1
    except ValueError:
        frame["decile"] = 1
    frame["decile"] = frame["decile"].fillna(1).astype(int)
    deciles = (
        frame.groupby("decile", as_index=False)
        .agg(
            count=("actual", "size"),
            mean_predicted_default=("predicted", "mean"),
            observed_default_rate=("actual", "mean"),
        )
        .assign(absolute_calibration_gap=lambda x: (x["mean_predicted_default"] - x["observed_default_rate"]).abs())
        .to_dict(orient="records")
    )
    actual = frame["actual"].astype(int)
    predicted = frame["predicted"].astype(float)
    roc_auc = roc_auc_score(actual, predicted) if actual.nunique() == 2 else None
    pr_auc = average_precision_score(actual, predicted) if actual.nunique() == 2 else None
    return {
        "roc_auc": float(roc_auc) if roc_auc is not None else None,
        "pr_auc": float(pr_auc) if pr_auc is not None else None,
        "brier_score": float(np.mean((predicted - actual) ** 2)),
        "log_loss": float(log_loss(actual, predicted, labels=[0, 1])) if actual.nunique() == 2 else None,
        "mean_predicted_default": float(predicted.mean()),
        "actual_default_rate": float(actual.mean()),
        "deciles": deciles,
    }



# takes default prediction probabilities and actual default rates to return dictionary of subgroups
def subgroup_calibration_summary(frame: pd.DataFrame, y_col: str, p_default) -> dict[str, list[dict]]:
    data = pd.DataFrame({"actual": frame[y_col], "predicted": p_default}, index=frame.index).dropna()
    out: dict[str, list[dict]] = {}

    def summarize(name: str, groups) -> None:
        grouped = data.assign(group=groups).dropna(subset=["group"])
        if grouped.empty:
            return
        rows = (
            grouped.groupby("group", as_index=False)
            .agg(
                count=("actual", "size"),
                mean_predicted_default=("predicted", "mean"),
                observed_default_rate=("actual", "mean"),
                brier_score=("predicted", lambda p: float(np.mean((p - grouped.loc[p.index, "actual"]) ** 2))),
            )
            .to_dict(orient="records")
        )
        out[name] = rows

    for column in ["term_months", "grade", "sub_grade"]:
        if column in frame.columns:
            summarize(column, frame[column])
    amount_col = "loan_amnt" if "loan_amnt" in frame.columns else "amount_requested" if "amount_requested" in frame.columns else None
    if amount_col:
        amount = pd.to_numeric(frame[amount_col], errors="coerce")
        try:
            bands = pd.qcut(amount, q=4, duplicates="drop").astype(str)
        except ValueError:
            bands = amount.astype(str)
        summarize(f"{amount_col}_band", bands)
    return out

# build plot of decile calibration across subgroups
def save_reliability_plot(y_true, p_default, path, bins: int = 10):
    summary = calibration_summary(y_true, p_default, bins=bins)
    deciles = pd.DataFrame(summary["deciles"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([0, 1], [0, 1], color="0.55", linestyle="--", linewidth=1)
    ax.plot(
        deciles["mean_predicted_default"],
        deciles["observed_default_rate"],
        marker="o",
    )
    ax.set_xlabel("Mean predicted default rate")
    ax.set_ylabel("Observed default rate")
    ax.set_title("Reliability plot")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
