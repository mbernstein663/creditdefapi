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
    }

    assert expected <= lines
    assert "!reports/test/model_validation_results.csv" not in lines
