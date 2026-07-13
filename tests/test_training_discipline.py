import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest

import evaluate_locked
from src import train
from src.artifacts import file_fingerprint, load_model_bundle, save_model_bundle
from src.config import load_training_config
from src.preprocessing import preprocess_accepted_loans, save_preprocessed_accepted_loans

SYNTHETIC_CONTEXT = "synthetic_test_fixture"


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


def _isolate_reports(monkeypatch, tmp_path):
    report_dir = tmp_path / "reports"
    cache_path = tmp_path / "accepted_preprocessed.joblib"
    monkeypatch.setattr(train, "REPORT_DIR", report_dir)
    monkeypatch.setattr(evaluate_locked, "REPORT_DIR", report_dir)
    monkeypatch.setattr(train, "DEFAULT_PREPROCESSED_ACCEPTED_BUNDLE", cache_path)
    monkeypatch.setattr(evaluate_locked, "DEFAULT_PREPROCESSED_ACCEPTED_BUNDLE", cache_path)
    return report_dir


def _tracked_report_snapshot():
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "ls-files", "reports/validation", "reports/test"],
        check=True,
        capture_output=True,
        text=True,
        cwd=repo,
    )
    paths = [repo / line for line in result.stdout.splitlines() if line]
    return {str(path.relative_to(repo)): path.read_bytes() for path in paths}


def test_sample_training_uses_smoke_artifact_and_report_paths():
    output, report_dir = train._paths(train.DEFAULT_ACCEPTED_BUNDLE, sample=100)

    assert output.name == "accepted_model_smoke.joblib"
    assert report_dir.parts[-2:] == ("smoke", "validation")


def test_artifact_context_defaults_and_explicit_fixture_context():
    assert train._artifact_data_context(None) == "full_lendingclub_local"
    assert train._artifact_data_context(100) == "smoke_sample"
    assert train._artifact_data_context(None, SYNTHETIC_CONTEXT) == SYNTHETIC_CONTEXT


def test_train_pipeline_saves_split_provenance_and_validation_reports(tmp_path, monkeypatch, capsys):
    csv_path = tmp_path / "accepted.csv"
    bundle_path = tmp_path / "bundle.joblib"
    _training_frame().to_csv(csv_path, index=False)
    _isolate_reports(monkeypatch, tmp_path)

    saved = train.train_accepted_model(csv_path=csv_path, output_path=bundle_path, artifact_data_context=SYNTHETIC_CONTEXT)
    bundle = load_model_bundle(saved)

    assert bundle.metadata["model_version"] == "accepted-default-v1"
    assert bundle.metadata["artifact_data_context"] == SYNTHETIC_CONTEXT
    assert bundle.metadata["split_manifest"]["row_counts"]["test"] > 0
    assert bundle.metadata["split_date_boundaries"]["validation"]["min"] is not None
    assert bundle.metadata["validation_metrics_summary"]["rows"] == bundle.metadata["split_row_counts"]["validation"]
    assert bundle.metadata["cross_validation_summary"]["selected_model_name"] == bundle.metadata["selected_model_name"]
    assert (tmp_path / "reports" / "validation" / "metrics_summary.json").exists()
    assert (tmp_path / "reports" / "validation" / "model_validation_results.csv").exists()
    out = capsys.readouterr().out
    assert "best model:" in out
    assert "saved best model:" in out


def test_custom_training_source_does_not_overwrite_shared_preprocessing_cache(tmp_path, monkeypatch):
    csv_path = tmp_path / "accepted.csv"
    bundle_path = tmp_path / "bundle.joblib"
    cache_path = tmp_path / "shared-cache.joblib"
    _training_frame().to_csv(csv_path, index=False)
    _isolate_reports(monkeypatch, tmp_path)
    monkeypatch.setattr(train, "DEFAULT_PREPROCESSED_ACCEPTED_BUNDLE", cache_path)

    train.train_accepted_model(csv_path=csv_path, output_path=bundle_path, artifact_data_context=SYNTHETIC_CONTEXT)

    assert not cache_path.exists()


def test_training_config_controls_models_and_cv(tmp_path, monkeypatch):
    csv_path = tmp_path / "accepted.csv"
    config_path = tmp_path / "config.yaml"
    bundle_path = tmp_path / "bundle.joblib"
    _training_frame().to_csv(csv_path, index=False)
    config_path.write_text(
        "\n".join(
            [
                "training:",
                "  cross_validation: false",
                "  models:",
                "    logistic regression: false",
                "    logistic balanced: true",
                "    random forest: false",
                "  calibration_methods:",
                "    isotonic: true",
                "    sigmoid: false",
            ]
        ),
        encoding="utf-8",
    )
    _isolate_reports(monkeypatch, tmp_path)

    loaded = load_training_config(config_path)
    saved = train.train_accepted_model(
        csv_path=csv_path,
        output_path=bundle_path,
        config_path=config_path,
        artifact_data_context=SYNTHETIC_CONTEXT,
    )
    bundle = load_model_bundle(saved)

    assert bundle.metadata["selected_model_name"] == "logistic_balanced"
    assert bundle.metadata["cross_validation_summary"]["enabled"] is False
    assert len(bundle.metadata["cross_validation_summary"]["candidate_summaries"]) == 1
    assert loaded["models"] == ["logistic_balanced"]
    assert loaded["calibration_methods"] == ["isotonic"]


def test_training_config_defaults_to_automatic_selection_and_all_enabled_models(tmp_path):
    pinned_config = tmp_path / "pinned.yaml"
    automatic_config = tmp_path / "automatic.yaml"
    pinned_config.write_text(
        "\n".join(
            [
                "training:",
                "  cross_validation: false",
                "  models:",
                "    logistic regression: true",
                "    logistic balanced: true",
                "  selected_model: logistic balanced",
            ]
        ),
        encoding="utf-8",
    )
    automatic_config.write_text(
        "\n".join(
            [
                "training:",
                "  cross_validation: false",
                "  models:",
                "    logistic regression: true",
                "    logistic balanced: true",
            ]
        ),
        encoding="utf-8",
    )

    pinned = load_training_config(pinned_config)
    automatic = load_training_config(automatic_config)

    assert pinned["selected_model"] == "logistic_balanced"
    assert automatic["selected_model"] is None
    assert automatic["models"] == ["logistic", "logistic_balanced"]


def test_full_training_does_not_reuse_smoke_preprocessed_cache(tmp_path, monkeypatch):
    csv_path = tmp_path / "accepted.csv"
    bundle_path = tmp_path / "bundle.joblib"
    cache_path = tmp_path / "accepted_preprocessed.joblib"
    _training_frame().to_csv(csv_path, index=False)
    save_preprocessed_accepted_loans(preprocess_accepted_loans(csv_path, sample=10), cache_path)
    _isolate_reports(monkeypatch, tmp_path)
    monkeypatch.setattr(train, "DEFAULT_PREPROCESSED_ACCEPTED_BUNDLE", cache_path)

    saved = train.train_accepted_model(csv_path=csv_path, output_path=bundle_path, artifact_data_context=SYNTHETIC_CONTEXT)
    bundle = load_model_bundle(saved)

    assert bundle.metadata["split_row_counts"]["train"] == 24


def test_validation_only_mode_writes_comparison_without_bundle(tmp_path, monkeypatch):
    csv_path = tmp_path / "accepted.csv"
    bundle_path = tmp_path / "bundle.joblib"
    _training_frame().to_csv(csv_path, index=False)
    _isolate_reports(monkeypatch, tmp_path)

    saved = train.train_accepted_model(
        csv_path=csv_path,
        output_path=bundle_path,
        validation_only=True,
        artifact_data_context=SYNTHETIC_CONTEXT,
    )

    assert saved == bundle_path
    assert bundle_path.exists()
    assert (tmp_path / "reports" / "validation" / "model_validation_results.csv").exists()


def test_locked_evaluation_uses_saved_test_ids_and_writes_metrics(tmp_path, monkeypatch):
    csv_path = tmp_path / "accepted.csv"
    bundle_path = tmp_path / "bundle.joblib"
    _training_frame().to_csv(csv_path, index=False)
    _isolate_reports(monkeypatch, tmp_path)
    train.train_accepted_model(csv_path=csv_path, output_path=bundle_path, artifact_data_context=SYNTHETIC_CONTEXT)
    bundle_sha256 = file_fingerprint(bundle_path)["sha256"]

    output = evaluate_locked.evaluate_locked_model(bundle_path, csv_path)
    summary = json.loads(output.read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "reports" / "test" / "evaluation_manifest.json").read_text(encoding="utf-8"))

    assert output.name == "metrics_summary.json"
    assert summary["rows"] > 0
    assert summary["artifact_data_context"] == SYNTHETIC_CONTEXT
    assert summary["baseline_comparison"]
    assert file_fingerprint(bundle_path)["sha256"] == bundle_sha256
    assert manifest["model_bundle_sha256"] == bundle_sha256
    assert manifest["locked_test_metrics"]["evaluation_split"] == "test"
    assert manifest["baseline_comparison"]
    for filename in [
        "metrics_summary.json",
        "model_card.md",
        "baseline_comparison.csv",
        "baseline_comparison.json",
        "calibration_deciles.csv",
        "risk_decile_lift.csv",
        "roc_curve.csv",
        "pr_curve.csv",
        "evaluation_manifest.json",
    ]:
        assert (tmp_path / "reports" / "test" / filename).exists()


def test_locked_evaluation_fails_when_test_id_set_differs_with_same_count(tmp_path, monkeypatch):
    csv_path = tmp_path / "accepted.csv"
    bundle_path = tmp_path / "bundle.joblib"
    _training_frame().to_csv(csv_path, index=False)
    _isolate_reports(monkeypatch, tmp_path)
    train.train_accepted_model(csv_path=csv_path, output_path=bundle_path, artifact_data_context=SYNTHETIC_CONTEXT)
    bundle = load_model_bundle(bundle_path)
    count = len(bundle.metadata["split_manifest"]["test_ids"])
    bundle.metadata["split_manifest"]["test_ids"] = [f"wrong-{i}" for i in range(count)]
    save_model_bundle(bundle, bundle_path)

    with pytest.raises(ValueError, match="test split IDs do not match"):
        evaluate_locked.evaluate_locked_model(bundle_path, csv_path)


def test_locked_evaluation_fails_when_validation_report_is_stale(tmp_path, monkeypatch):
    csv_path = tmp_path / "accepted.csv"
    bundle_path = tmp_path / "bundle.joblib"
    _training_frame().to_csv(csv_path, index=False)
    report_dir = _isolate_reports(monkeypatch, tmp_path)
    train.train_accepted_model(csv_path=csv_path, output_path=bundle_path, artifact_data_context=SYNTHETIC_CONTEXT)
    metrics_path = report_dir / "validation" / "metrics_summary.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["selected_model"] = "stale_model"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

    with pytest.raises(ValueError, match="validation report does not match locked bundle"):
        evaluate_locked.evaluate_locked_model(bundle_path, csv_path)


def test_locked_evaluation_sample_writes_smoke_reports(tmp_path, monkeypatch):
    csv_path = tmp_path / "accepted.csv"
    bundle_path = tmp_path / "bundle.joblib"
    _training_frame().to_csv(csv_path, index=False)
    _isolate_reports(monkeypatch, tmp_path)
    train.train_accepted_model(csv_path=csv_path, output_path=bundle_path, artifact_data_context=SYNTHETIC_CONTEXT)

    evaluate_locked.evaluate_locked_model(bundle_path, csv_path, sample=10)

    assert (tmp_path / "reports" / "smoke" / "test" / "metrics_summary.json").exists()


def test_locked_evaluation_fails_on_source_fingerprint_drift(tmp_path, monkeypatch):
    csv_path = tmp_path / "accepted.csv"
    bundle_path = tmp_path / "bundle.joblib"
    _training_frame().to_csv(csv_path, index=False)
    _isolate_reports(monkeypatch, tmp_path)
    train.train_accepted_model(csv_path=csv_path, output_path=bundle_path, artifact_data_context=SYNTHETIC_CONTEXT)

    changed = _training_frame()
    changed.loc[0, "annual_inc"] = 999999
    changed.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="source CSV fingerprint does not match"):
        evaluate_locked.evaluate_locked_model(bundle_path, csv_path)


def test_fixture_training_and_locked_evaluation_do_not_touch_tracked_reports(tmp_path, monkeypatch):
    before = _tracked_report_snapshot()
    assert before
    csv_path = tmp_path / "accepted.csv"
    bundle_path = tmp_path / "bundle.joblib"
    cache_path = tmp_path / "accepted_preprocessed.joblib"
    _training_frame().to_csv(csv_path, index=False)
    _isolate_reports(monkeypatch, tmp_path)
    monkeypatch.setattr(train, "DEFAULT_PREPROCESSED_ACCEPTED_BUNDLE", cache_path)
    monkeypatch.setattr(evaluate_locked, "DEFAULT_PREPROCESSED_ACCEPTED_BUNDLE", cache_path)

    train.train_accepted_model(csv_path=csv_path, output_path=bundle_path, artifact_data_context=SYNTHETIC_CONTEXT)
    evaluate_locked.evaluate_locked_model(bundle_path, csv_path)

    assert _tracked_report_snapshot() == before


def test_file_fingerprint_changes_when_same_size_content_changes(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("abc\n", encoding="utf-8")
    first = file_fingerprint(path)
    path.write_text("abd\n", encoding="utf-8")
    second = file_fingerprint(path)

    assert first["size_bytes"] == second["size_bytes"]
    assert first["sha256"] != second["sha256"]
