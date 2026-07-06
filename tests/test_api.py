from fastapi.testclient import TestClient

import api
from src.artifacts import save_model_bundle
from tests.test_artifacts_batch_scoring import accepted_bundle, accepted_row


def test_health_works():
    client = TestClient(api.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_reports_missing_artifact(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "DEFAULT_ACCEPTED_BUNDLE", tmp_path / "missing.joblib")
    monkeypatch.setattr(api, "DEFAULT_FRONTEND_BUNDLE", tmp_path / "missing-frontend.joblib")
    client = TestClient(api.app)

    response = client.get("/ready")

    assert response.status_code == 503
    assert "artifact_errors" in response.json()["detail"]


def test_ready_reports_incomplete_artifact(monkeypatch, tmp_path):
    path = tmp_path / "incomplete.joblib"
    save_model_bundle(accepted_bundle(metadata={"model_version": "accepted-default-v1"}), path)
    monkeypatch.setattr(api, "DEFAULT_ACCEPTED_BUNDLE", path)
    monkeypatch.setattr(api, "DEFAULT_FRONTEND_BUNDLE", path)
    client = TestClient(api.app)

    response = client.get("/ready")

    assert response.status_code == 503
    assert "missing metadata field" in str(response.json()["detail"])


def test_model_card_returns_metadata(monkeypatch):
    monkeypatch.setattr(api, "accepted_bundle", lambda: accepted_bundle())
    client = TestClient(api.app)

    response = client.get("/model-card")

    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "accepted-default-v1"
    assert body["calibration_method"] == "isotonic"
    assert "cross_validation_summary" in body


def test_frontend_config_returns_top_fields(monkeypatch):
    monkeypatch.setattr(
        api,
        "frontend_bundle",
        lambda: accepted_bundle(feature_columns=["loan_amnt", "int_rate", "annual_inc", "dti", "fico_range_low"]),
    )
    client = TestClient(api.app)

    response = client.get("/frontend-config")

    assert response.status_code == 200
    body = response.json()
    assert len(body["frontend_fields"]) == 5


def test_score_returns_risk_only_schema(monkeypatch):
    monkeypatch.setattr(api, "accepted_bundle", lambda: accepted_bundle())
    client = TestClient(api.app)
    payload = accepted_row()
    payload.pop("id")

    response = client.post("/score", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "p_default",
        "p_non_default",
        "confidence",
        "risk_band",
        "model_version",
        "model_type",
        "calibration_method",
        "scoring_note",
    }


def test_score_frontend_returns_subset_model_prediction(monkeypatch):
    monkeypatch.setattr(
        api,
        "frontend_bundle",
        lambda: accepted_bundle(feature_columns=["loan_amnt", "int_rate", "annual_inc", "dti", "fico_range_low"]),
    )
    client = TestClient(api.app)

    response = client.post(
        "/score-frontend",
        json={
            "loan_amnt": 1000,
            "int_rate": 10,
            "annual_inc": 50000,
            "dti": 12,
            "fico_range_low": 700,
        },
    )

    assert response.status_code == 200
    assert response.json()["risk_band"] == "medium"


def test_invalid_input_returns_validation_error():
    client = TestClient(api.app)

    response = client.post("/score", json={"loan_amnt": 1000})

    assert response.status_code == 422


def test_rejected_style_payload_fails_on_score_endpoint():
    client = TestClient(api.app)

    response = client.post(
        "/score",
        json={
            "amount_requested": 1000,
            "risk_score": 700,
            "dti": 10,
            "zip_code": "123xx",
            "state": "NY",
            "employment_length": "4 years",
        },
    )

    assert response.status_code == 422


def test_score_batch_returns_csv(monkeypatch):
    monkeypatch.setattr(api, "accepted_bundle", lambda: accepted_bundle())
    client = TestClient(api.app)
    frame = "id,loan_amnt,int_rate,annual_inc,dti,fico_range_low,fico_range_high,delinq_2yrs,inq_last_6mths,open_acc,pub_rec,revol_bal,revol_util,total_acc,mort_acc,acc_open_past_24mths,pub_rec_bankruptcies,grade,sub_grade,emp_length,home_ownership,verification_status,purpose,addr_state,application_type,initial_list_status\n1,1000,10,50000,12,700,704,0,0,8,0,1000,20,12,0,1,0,A,A1,4 years,RENT,Not Verified,debt_consolidation,NY,Individual,w\n"

    response = client.post(
        "/score-batch",
        files={"file": ("loans.csv", frame.encode("utf-8"), "text/csv")},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "p_default" in response.text
