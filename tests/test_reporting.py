import csv
import json
from pathlib import Path


def test_gitignore_allows_committed_locked_test_report_files():
    repo = Path(__file__).resolve().parents[1]
    lines = set((repo / ".gitignore").read_text(encoding="utf-8").splitlines())
    expected = {
        "!reports/test/model_card.md",
        "!reports/test/metrics_summary.json",
        "!reports/test/baseline_comparison.csv",
        "!reports/test/baseline_comparison.json",
        "!reports/test/calibration_deciles.csv",
        "!reports/test/risk_decile_lift.csv",
        "!reports/test/roc_curve.csv",
        "!reports/test/pr_curve.csv",
        "!reports/test/reliability_plot.png",
        "!reports/test/roc_curve.png",
        "!reports/test/pr_curve.png",
        "!reports/test/evaluation_manifest.json",
    }

    assert expected <= lines
    assert "!reports/test/model_validation_results.csv" not in lines


def test_demo_batch_output_preserves_submitted_features():
    repo = Path(__file__).resolve().parents[1]
    with (repo / "docs" / "demo" / "sample_batch_input.csv").open(encoding="utf-8", newline="") as handle:
        submitted = list(csv.DictReader(handle))
    with (repo / "docs" / "demo" / "sample_batch_output.csv").open(encoding="utf-8", newline="") as handle:
        scored = list(csv.DictReader(handle))

    assert len(scored) == len(submitted)
    assert all(
        {feature: scored_row[feature] for feature in submitted_row} == submitted_row
        for submitted_row, scored_row in zip(submitted, scored)
    )
    assert {"p_default", "risk_band", "model_version"} <= scored[0].keys()
    assert "decision_margin" not in scored[0]
    assert "scoring_note" not in scored[0]


def test_locked_evaluation_manifest_is_allowed_and_contains_bundle_sha():
    repo = Path(__file__).resolve().parents[1]
    manifest = json.loads((repo / "reports" / "test" / "evaluation_manifest.json").read_text(encoding="utf-8"))

    assert "!reports/test/evaluation_manifest.json" in (repo / ".gitignore").read_text(encoding="utf-8")
    assert len(manifest["model_bundle_sha256"]) == 64
    assert all(character in "0123456789abcdef" for character in manifest["model_bundle_sha256"])


def test_frontend_accessibility_and_font_contract():
    frontend = (Path(__file__).resolve().parents[1] / "frontend" / "index.html").read_text(encoding="utf-8")

    assert "font-family: Arial, Helvetica, sans-serif;" in frontend
    assert "overflow-wrap: anywhere;" in frontend
    assert '<div class="result" id="result" aria-live="polite">' in frontend
    assert 'font-family: Georgia, "Times New Roman", serif;' not in frontend


def test_frontend_uses_config_defaults_without_ranked_or_extra_score_output():
    frontend = (Path(__file__).resolve().parents[1] / "frontend" / "index.html").read_text(encoding="utf-8")

    assert "Top 5 attributes" not in frontend
    assert "rank-list" not in frontend
    assert "feature_importance" not in frontend
    assert "decision_margin" not in frontend
    assert "scoring_note" not in frontend
    assert 'name === "dti" ? "0.1"' in frontend
    assert "buildField(name, body.frontend_defaults)" in frontend
    assert "Number(defaults[name]).toFixed(1)" in frontend
