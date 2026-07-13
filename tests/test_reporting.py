import csv
import json
from pathlib import Path


def test_scope_guardrails_block_reintroduced_decision_or_business_terms():
    repo = Path(__file__).resolve().parents[1]
    files = [
        repo / "README.md",
        repo / "api.py",
        repo / "batch.py",
        repo / "evaluate_locked.py",
        repo / "src" / "train.py",
        *sorted((repo / "src").glob("*.py")),
    ]
    allowed = [
        "It is not a pro" "fit model or lending po" "licy engine.",
        "Does not meet the credit po" "licy. Status:Charged Off",
        "Does not meet the credit po" "licy. Status:Fully Paid",
        "the resolved `Does not meet the credit po" "licy. Status:Charged Off` variant",
        "the resolved `Does not meet the credit po" "licy. Status:Fully Paid` variant",
    ]
    banned = [
        "pro" "fit",
        "expected_" "pro" "fit",
        "realized_" "pro" "fit",
        "appro" "ve",
        "appro" "val",
        "po" "licy",
        "invest" "ment",
        "lg" "d",
        "required_" "return",
        "good_" "pro" "fit_" "haircut",
    ]

    text = "\n".join(path.read_text(encoding="utf-8") for path in files)
    for phrase in allowed:
        text = text.replace(phrase, "")

    lowered = text.lower()
    for term in banned:
        assert term not in lowered


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
