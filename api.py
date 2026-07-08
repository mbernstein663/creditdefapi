from __future__ import annotations

from functools import lru_cache
from io import StringIO
from pathlib import Path

import pandas as pd
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.artifacts import load_model_bundle
from src.config import DEFAULT_ACCEPTED_BUNDLE, DEFAULT_FRONTEND_BUNDLE, ROOT
from src.schemas import AcceptedScoreRequest, ScoreResponse
from src.scorer import score_frame, score_records

app = FastAPI(title="Credit Default Risk API")

REQUIRED_METADATA_FIELDS = [
    "bundle_schema_version",
    "model_version",
    "source_fingerprint",
    "split_manifest",
    "split_summary",
    "target_definition",
    "forbidden_feature_columns",
    "training_timestamp",
    "package_versions",
]


@lru_cache(maxsize=1)
def accepted_bundle():
    return load_model_bundle(DEFAULT_ACCEPTED_BUNDLE)


@lru_cache(maxsize=1)
def frontend_bundle():
    return load_model_bundle(DEFAULT_FRONTEND_BUNDLE)


def _artifact_errors(path):
    if not Path(path).exists():
        return [f"missing artifact: {path}"]
    bundle = load_model_bundle(path)
    errors = []
    if bundle.calibrator is None:
        errors.append(f"missing calibrator: {path}")
    if not bundle.feature_columns:
        errors.append(f"missing feature columns: {path}")
    if not bundle.required_input_schema:
        errors.append(f"missing required input schema: {path}")
    metadata = bundle.metadata or {}
    for field in REQUIRED_METADATA_FIELDS:
        if not metadata.get(field):
            errors.append(f"missing metadata field `{field}`: {path}")
    return errors


def _model_card_payload(bundle) -> dict:
    metadata = bundle.metadata or {}
    return {
        "model_version": metadata.get("model_version"),
        "model_type": metadata.get("selected_model_type", bundle.model_type),
        "selected_model": metadata.get("selected_model_name"),
        "calibration_method": metadata.get("calibration_method"),
        "target_definition": metadata.get("target_definition"),
        "feature_columns": bundle.feature_columns,
        "split_summary": metadata.get("split_summary"),
        "cross_validation_summary": metadata.get("cross_validation_summary"),
        "validation_metrics_summary": metadata.get("validation_metrics_summary"),
        "locked_test_metrics_summary": metadata.get("locked_test_metrics_summary"),
        "frontend_fields": metadata.get("frontend_fields"),
        "feature_importance": metadata.get("feature_importance"),
        "limitations": metadata.get("limitations"),
        "training_timestamp": metadata.get("training_timestamp"),
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    errors = _artifact_errors(DEFAULT_ACCEPTED_BUNDLE)
    errors += _artifact_errors(DEFAULT_FRONTEND_BUNDLE)
    if errors:
        raise HTTPException(status_code=503, detail={"artifact_errors": errors})
    return {"status": "ready"}


@app.get("/model-card")
def model_card():
    try:
        bundle = accepted_bundle()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="model artifact not found; train model first") from exc
    return _model_card_payload(bundle)


@app.get("/frontend-config")
def frontend_config():
    try:
        bundle = frontend_bundle()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="frontend model artifact not found; train model first") from exc
    metadata = bundle.metadata or {}
    return {
        "model_version": metadata.get("model_version"),
        "model_type": metadata.get("selected_model_type", bundle.model_type),
        "calibration_method": metadata.get("calibration_method"),
        "frontend_fields": bundle.feature_columns,
        "feature_importance": metadata.get("feature_importance", []),
        "scoring_note": "Frontend demo uses a separate reduced-feature model trained on exactly the displayed fields.",
    }


@app.post("/score", response_model=ScoreResponse)
def score_accepted(request: AcceptedScoreRequest):
    try:
        bundle = accepted_bundle()
        result = score_records([request.model_dump()], bundle)[0]
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="model artifact not found; train model first") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return ScoreResponse(**result)


@app.post("/score-frontend", response_model=ScoreResponse)
def score_frontend(payload: dict = Body(...)):
    try:
        bundle = frontend_bundle()
        result = score_records([payload], bundle)[0]
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="frontend model artifact not found; train model first") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return ScoreResponse(**result)


@app.post("/score-batch")
async def score_batch(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=422, detail="batch scoring requires a .csv upload")
    try:
        bundle = accepted_bundle()
        content = await file.read()
        frame = pd.read_csv(StringIO(content.decode("utf-8")))
        scored = score_frame(frame, bundle)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="model artifact not found; train model first") from exc
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=422, detail="uploaded CSV must be UTF-8 encoded") from exc
    except pd.errors.EmptyDataError as exc:
        raise HTTPException(status_code=422, detail="uploaded CSV is empty") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    output = StringIO()
    scored.to_csv(output, index=False)
    output.seek(0)
    filename = f"{Path(file.filename).stem}_scored.csv"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return StreamingResponse(iter([output.getvalue()]), media_type="text/csv", headers=headers)


frontend_dir = ROOT / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
