# Runbook

## Default Paths

- Accepted data: `./accepted_2007_to_2018Q4.csv`
- Rejected data: `./rejected_2007_to_2018Q4.csv`
- Preprocessed cache: `./artifacts/accepted_preprocessed.joblib`
- Main bundle: `./artifacts/accepted_model.joblib`
- Frontend bundle: `./artifacts/frontend_model.joblib`

Raw LendingClub CSVs are intentionally not committed.

## Fresh Clone To Local API

1. Install dependencies.

```bash
python -m venv .venv
. .venv/Scripts/activate
python -m pip install -r requirements.txt
python -m pip install -e .[dev]
```

2. Place `accepted_2007_to_2018Q4.csv` at the repo root.

3. Preprocess the accepted-loan data.

```bash
python -m src.preprocessing
```

4. Train the calibrated default-risk bundles and write validation reports.

```bash
python -m src.train
```

5. Start the API.

```bash
uvicorn api:app --reload
```

6. Check service status.

```bash
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

`/ready` returns `503` until both saved bundles exist. In Docker it also returns `503` if `./artifacts` was not mounted.

7. Score the demo batch file.

```bash
python batch.py docs/demo/sample_batch_input.csv docs/demo/sample_batch_output.csv --api-url http://127.0.0.1:8000
```

## Optional Locked Test

Run only after validation-stage model selection is complete:

```bash
python evaluate_locked.py
```

## Docker

Build:

```bash
docker build -t credit-default-api .
```

Run with mounted artifacts:

```bash
docker run --rm -p 8000:8000 -v ./artifacts:/app/artifacts credit-default-api
```

The image build excludes raw CSVs, artifacts, and reports by default. Train on the host first, then mount `./artifacts`.

## Smoke Runs

Smoke/sample runs are for quick checks only and are not final model evidence:

```bash
python -m src.preprocessing --sample 5000
python -m src.train --sample 5000
```

Those write smoke artifacts under `reports/smoke/validation/` and smoke bundles under `artifacts/*_smoke.joblib`.
