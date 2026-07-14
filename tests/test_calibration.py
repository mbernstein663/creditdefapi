import numpy as np

from src.calibration import ProbabilityCalibrator, calibration_summary


def test_calibration_summary_reports_brier_log_loss_and_deciles():
    summary = calibration_summary([0, 1, 0, 1], [0.1, 0.8, 0.2, 0.7], bins=2)

    assert summary["brier_score"] < 0.1
    assert summary["roc_auc"] == 1.0
    assert summary["pr_auc"] == 1.0
    assert summary["log_loss"] is not None
    assert summary["mean_predicted_default"] == 0.45
    assert summary["actual_default_rate"] == 0.5
    assert len(summary["deciles"]) == 2
    assert "absolute_calibration_gap" in summary["deciles"][0]


def test_probability_calibrator_outputs_probabilities_between_zero_and_one():
    calibrator = ProbabilityCalibrator("sigmoid").fit(np.array([0.1, 0.2, 0.7, 0.9]), np.array([0, 0, 1, 1]))
    predicted = calibrator.predict(np.array([-10.0, 0.5, 10.0]))

    assert ((predicted >= 0) & (predicted <= 1)).all()
