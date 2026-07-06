import json

import pandas as pd
import pytest

import evaluate_locked
import train
from src.artifacts import file_fingerprint, load_model_bundle


def _training_frame(rows=40):
    issue_dates = pd.date_range("2015-01-01", periods=rows, freq="MS")
    data = []
    for i, issue_date in enumerate(issue_dates):
        default = int(i % 4 == 0)
        data.append(
            {
                "id": str(i + 1),
                "loan_amnt": 1000 + 10 * i,
                "int_rate": 9 + (i % 5),
                "annual_inc": 50000 + 1000 * i,
                "dti": 10 + (i % 7),
                "fico_range_low": 680 + (i % 15),
                "fico_range_high": 684 + (i % 15),
                "delinq_2yrs": i % 2,
                "inq_last_6mths": i % 3,
                "open_acc": 8 + (i % 4),
                "pub_rec": 0,
                "revol_bal": 1000 + 50 * i,
                "revol_util": f"{20 + (i % 40)}%",
                "total_acc": 12 + (i % 10),
                "mort_acc": i % 3,
                "acc_open_past_24mths": i % 5,
                "pub_rec_bankruptcies": 0,
                "grade": "A" if i % 2 == 0 else "B",
                "sub_grade": "A1" if i % 2 == 0 else "B2",
                "emp_length": "4 years",
                "home_ownership": "RENT" if i % 2 == 0 else "MORTGAGE",
                "verification_status": "Verified" if i % 2 == 0 else "Not Verified",
                "purpose": "debt_consolidation",
                "addr_state": "NY" if i % 2 == 0 else "CA",
                "application_type": "Individual",
                "initial_list_status": "w",
                "term": " 36 months",
                "loan_status": "Charged Off" if default else "Fully Paid",
                "issue_d": issue_date.strftime("%b-%Y"),
            }
        )
    return pd.DataFrame(data)


def test_sample_training_uses_smoke_artifact_and_report_paths():
    output, report_dir = train._paths(train.DEFAULT_ACCEPTED_BUNDLE, sample=100)

    assert output.name == "accepted_model_smoke.joblib"
    assert report_dir.name == "smoke"


def test_train_pipeline_saves_split_provenance_and_validation_reports(tmp_path, monkeypatch):
    csv_path = tmp_path / "accepted.csv"
    bundle_path = tmp_path / "bundle.joblib"
    _training_frame().to_csv(csv_path, index=False)
    monkeypatch.setattr(train, "REPORT_DIR", tmp_path / "reports")

    saved = train.train_accepted_model(csv_path=csv_path, output_path=bundle_path)
    bundle = load_model_bundle(saved)

    assert bundle.metadata["model_version"] == "accepted-default-v1"
    assert bundle.metadata["split_manifest"]["row_counts"]["test"] > 0
    assert bundle.metadata["split_date_boundaries"]["validation"]["min"] is not None
    assert bundle.metadata["validation_metrics_summary"]["rows"] == bundle.metadata["split_row_counts"]["validation"]
    assert bundle.metadata["cross_validation_summary"]["selected_model_name"] == bundle.metadata["selected_model_name"]
    assert (tmp_path / "reports" / "validation" / "metrics_summary.json").exists()
    assert (tmp_path / "reports" / "validation" / "cross_validation_summary.csv").exists()


def test_locked_evaluation_uses_saved_test_ids_and_writes_metrics(tmp_path, monkeypatch):
    csv_path = tmp_path / "accepted.csv"
    bundle_path = tmp_path / "bundle.joblib"
    _training_frame().to_csv(csv_path, index=False)
    monkeypatch.setattr(train, "REPORT_DIR", tmp_path / "reports")
    train.train_accepted_model(csv_path=csv_path, output_path=bundle_path)
    monkeypatch.setattr(evaluate_locked, "REPORT_DIR", tmp_path / "reports")

    output = evaluate_locked.evaluate_locked_model(bundle_path, csv_path)
    summary = json.loads(output.read_text(encoding="utf-8"))

    assert output.name == "metrics_summary.json"
    assert summary["rows"] > 0
    assert (tmp_path / "reports" / "test" / "model_card.md").exists()


def test_locked_evaluation_fails_on_source_fingerprint_drift(tmp_path):
    csv_path = tmp_path / "accepted.csv"
    bundle_path = tmp_path / "bundle.joblib"
    _training_frame().to_csv(csv_path, index=False)
    train.train_accepted_model(csv_path=csv_path, output_path=bundle_path)

    changed = _training_frame()
    changed.loc[0, "annual_inc"] = 999999
    changed.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="source CSV fingerprint does not match"):
        evaluate_locked.evaluate_locked_model(bundle_path, csv_path)


def test_file_fingerprint_changes_when_same_size_content_changes(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("abc\n", encoding="utf-8")
    first = file_fingerprint(path)
    path.write_text("abd\n", encoding="utf-8")
    second = file_fingerprint(path)

    assert first["size_bytes"] == second["size_bytes"]
    assert first["sha256"] != second["sha256"]
