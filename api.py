from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from src.artifacts import load_model_bundle
from src.config import DEFAULT_ACCEPTED_BUNDLE, DEFAULT_REJECTED_STYLE_BUNDLE, ROOT
from src.schemas import AcceptedScoreRequest, RejectedRiskRequest, ScoreResponse
from src.scorer import score_records

app = FastAPI(title="Credit Risk Scoring API")


@lru_cache(maxsize=2)
def accepted_bundle():
    return load_model_bundle(DEFAULT_ACCEPTED_BUNDLE)


@lru_cache(maxsize=2)
def rejected_style_bundle():
    return load_model_bundle(DEFAULT_REJECTED_STYLE_BUNDLE)


def _artifact_errors(path, require_test_ids=False):
    if not Path(path).exists():
        return [f"missing artifact: {path}"]
    bundle = load_model_bundle(path)
    errors = []
    if not bundle.metadata.get("source_fingerprint"):
        errors.append(f"incomplete artifact metadata: {path}")
    if require_test_ids and not bundle.metadata.get("split_manifest", {}).get("test_ids"):
        errors.append(f"missing locked test split IDs: {path}")
    return errors


def _score(record, bundle_loader, note: str) -> ScoreResponse:
    try:
        bundle = bundle_loader()
        result = score_records([record], bundle)[0]
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="model artifact not found; train model first") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    result["model_note"] = note
    return ScoreResponse(**result)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    errors = _artifact_errors(DEFAULT_ACCEPTED_BUNDLE, require_test_ids=True)
    errors += _artifact_errors(DEFAULT_REJECTED_STYLE_BUNDLE)
    if errors:
        raise HTTPException(status_code=503, detail={"artifact_errors": errors})
    return {"status": "ready"}


@app.post("/score", response_model=ScoreResponse)
def score_accepted(request: AcceptedScoreRequest):
    return _score(
        request.model_dump(exclude_none=True),
        accepted_bundle,
        "accepted-loan model; profit decision only when profit inputs are supplied",
    )


@app.post("/score/rejected-risk", response_model=ScoreResponse)
def score_rejected_risk(request: RejectedRiskRequest):
    return _score(
        request.model_dump(exclude_none=True),
        rejected_style_bundle,
        "risk is estimated from resolved accepted loans mapped to rejected-style fields",
    )


frontend_dir = ROOT / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
