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


@dataclass
class ProbabilityCalibrator:
    method: str = "isotonic"
    model: Any | None = None

    def fit(self, raw_probability, y):
        raw = np.asarray(raw_probability, dtype=float)
        target = np.asarray(y, dtype=int)
        if len(np.unique(target)) < 2:
            raise ValueError("calibration requires both classes")
        if self.method == "identity":
            self.model = None
            return self
        if self.method == "sigmoid":
            self.model = LogisticRegression(max_iter=1000).fit(raw.reshape(-1, 1), target)
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
        .to_dict(orient="records")
    )
    return {
        "brier_score": float(np.mean((frame["predicted"] - frame["actual"]) ** 2)),
        "mean_predicted_default": float(frame["predicted"].mean()),
        "actual_default_rate": float(frame["actual"].mean()),
        "deciles": deciles,
    }


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
