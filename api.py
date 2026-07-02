from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from src.artifacts import load_model_bundle
from src.config import DEFAULT_ACCEPTED_BUNDLE, DEFAULT_REJECTED_STYLE_BUNDLE, PROFIT_INPUT_COLUMNS, ROOT
from src.schemas import AcceptedScoreRequest, RejectedRiskRequest, ScoreResponse
from src.scorer import score_records

app = FastAPI(title="Credit Risk Scoring API")


@lru_cache(maxsize=2)
def accepted_bundle():
    return load_model_bundle(DEFAULT_ACCEPTED_BUNDLE)


@lru_cache(maxsize=2)
def rejected_style_bundle():
    return load_model_bundle(DEFAULT_REJECTED_STYLE_BUNDLE)


def _artifact_errors(path):
    if not Path(path).exists():
        return [f"missing artifact: {path}"]
    bundle = load_model_bundle(path)
    errors = []
    if not bundle.metadata.get("source_fingerprint"):
        errors.append(f"incomplete artifact metadata: {path}")
    if bundle.calibrator is None:
        errors.append(f"missing calibrator: {path}")
    if not bundle.feature_columns:
        errors.append(f"missing feature columns: {path}")
    if not bundle.required_input_schema:
        errors.append(f"missing required input schema: {path}")
    if not bundle.policy:
        errors.append(f"missing scoring policy: {path}")
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
    errors = _artifact_errors(DEFAULT_ACCEPTED_BUNDLE)
    errors += _artifact_errors(DEFAULT_REJECTED_STYLE_BUNDLE)
    if errors:
        raise HTTPException(status_code=503, detail={"artifact_errors": errors})
    return {"status": "ready"}


@app.post("/score", response_model=ScoreResponse)
def score_accepted(request: AcceptedScoreRequest):
    record = request.model_dump()
    for column in PROFIT_INPUT_COLUMNS:
        if record.get(column) is None:
            record.pop(column)
    return _score(
        record,
        accepted_bundle,
        "post-pricing accepted-loan model; valid only after LendingClub grade/rate fields are available",
    )


@app.post("/score/rejected-risk", response_model=ScoreResponse)
def score_rejected_risk(request: RejectedRiskRequest):
    return _score(
        request.model_dump(exclude_none=True),
        rejected_style_bundle,
        "limited-field risk estimate using accepted-loan outcomes projected onto rejected-application-style inputs",
    )


frontend_dir = ROOT / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
