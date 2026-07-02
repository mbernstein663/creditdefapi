from fastapi.testclient import TestClient

import api
from src.artifacts import ModelBundle
from tests.test_artifacts_batch_scoring import DummyModel, IdentityCalibrator


def test_rejected_risk_valid_request_returns_review(monkeypatch):
    bundle = ModelBundle(
        DummyModel(),
        IdentityCalibrator(),
        ["amount_requested", "risk_score", "dti", "zip_code", "state", "employment_length"],
        "rejected_style",
    )
    monkeypatch.setattr(api, "rejected_style_bundle", lambda: bundle)
    client = TestClient(api.app)

    response = client.post(
        "/score/rejected-risk",
        json={
            "amount_requested": 1000,
            "risk_score": 700,
            "dti": 10,
            "zip_code": "123xx",
            "state": "NY",
            "employment_length": "4 years",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "review"
    assert body["model_note"].startswith("risk is estimated")
    assert body["lgd"] == 1.0


def test_ready_reports_missing_artifacts(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "DEFAULT_ACCEPTED_BUNDLE", tmp_path / "missing_a.joblib")
    monkeypatch.setattr(api, "DEFAULT_REJECTED_STYLE_BUNDLE", tmp_path / "missing_r.joblib")
    client = TestClient(api.app)

    response = client.get("/ready")

    assert response.status_code == 503
    assert "artifact_errors" in response.json()["detail"]


def test_rejected_risk_invalid_request_fails_validation():
    client = TestClient(api.app)

    response = client.post("/score/rejected-risk", json={"amount_requested": 1000})

    assert response.status_code == 422


def test_accepted_score_missing_model_artifact_is_clear(monkeypatch):
    def missing():
        raise FileNotFoundError("missing")

    monkeypatch.setattr(api, "accepted_bundle", missing)
    client = TestClient(api.app)

    response = client.post(
        "/score",
        json={
            "loan_amnt": 1000,
            "int_rate": 10,
            "annual_inc": 50000,
            "dti": 12,
            "fico_range_low": 700,
            "fico_range_high": 704,
            "grade": "A",
            "sub_grade": "A1",
            "emp_length": "4 years",
            "home_ownership": "RENT",
            "verification_status": "Not Verified",
            "purpose": "debt_consolidation",
            "addr_state": "NY",
        },
    )

    assert response.status_code == 503
    assert "train model first" in response.json()["detail"]
