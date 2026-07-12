from pathlib import Path

from fastapi.testclient import TestClient

import api
from scripts.generate_demo_artifacts import generate
from src.artifacts import load_model_bundle


def test_demo_generator_writes_valid_synthetic_bundles(monkeypatch, tmp_path):
    paths = generate(tmp_path)

    assert {path.name for path in paths} == {"accepted_model.joblib", "frontend_model.joblib"}
    for path in paths:
        bundle = load_model_bundle(path)
        assert bundle.calibrator is not None
        assert bundle.required_input_schema
        assert bundle.metadata["artifact_data_context"] == "synthetic_test_fixture"

    monkeypatch.setattr(api, "DEFAULT_ACCEPTED_BUNDLE", paths[0])
    monkeypatch.setattr(api, "DEFAULT_FRONTEND_BUNDLE", paths[1])
    assert TestClient(api.app).get("/ready").status_code == 200


def test_container_uses_external_production_artifact_paths():
    root = Path(__file__).resolve().parents[1]
    dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")
    compose = (root / "compose.yaml").read_text(encoding="utf-8")

    assert "ACCEPTED_MODEL_BUNDLE=/app/artifacts/accepted_model.joblib" in dockerfile
    assert "FRONTEND_MODEL_BUNDLE=/app/artifacts/frontend_model.joblib" in dockerfile
    assert "./artifacts:/app/artifacts:ro" in compose
