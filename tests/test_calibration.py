from src.calibration import calibration_summary


def test_calibration_summary_reports_brier_and_deciles():
    summary = calibration_summary([0, 1, 0, 1], [0.1, 0.8, 0.2, 0.7], bins=2)

    assert summary["brier_score"] < 0.1
    assert summary["roc_auc"] == 1.0
    assert summary["pr_auc"] == 1.0
    assert summary["mean_predicted_default"] == 0.45
    assert summary["actual_default_rate"] == 0.5
    assert len(summary["deciles"]) == 2
