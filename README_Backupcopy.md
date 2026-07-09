# Credit Default Risk API

This repo is a calibrated, API-backed LendingClub default-risk project. It uses previously accepted/funded loans with resolved outcomes to improve current default predictions and ultimately devise more profitable lending policies. Our hope is that we can gather uncaptured signal from previous lending years and improve LendingClub's current business model.

## Overview

This repo is not production underwriting; included is leakage-controlled modeling, chronological validation, class calibration, saved model artifacts, FastAPI-backed scoring operations with batch support, and adequate testing: all containerized in Docker. 

- Supervised modeling uses accepted/funded loans with resolved outcomes only.
- The target is binary `default`.
- Outputs are calibrated `p_default`
- Features are limited to application-time or underwriting-time fields.
- Post-origination repayment fields are forbidden model inputs.

## Scope

- Supervised modeling uses accepted/funded loans with resolved outcomes only.
- The target is binary `default`.
- Outputs are calibrated `p_default`
- Features are limited to application-time or underwriting-time fields.
- Post-origination repayment fields are forbidden model inputs.

## Data & Target Definition

We wanted to ensure a fair grading without cutting too many of our samples. We started with 2,260,701 previously accepted loans from June 2007 to December 2018 with 35 consumer attributes.

After preprocessing, we ended up using 25 attributes and 1,348,099 loans with known outomes. We purposely did not use non-accepted loans due to counterfactual realized success rate.

**Default** (Bad Outcomes):

- `Charged Off`
- `Default`
- `Does not meet the credit policy. Status:Charged Off`

**Non-Default** (Good outcome):

- `Fully Paid`
- `Does not meet the credit policy. Status:Fully Paid`

Dropped from supervised modeling:

- `Current`
- `In Grace Period`
- `Late (16-30 days)`
- `Late (31-120 days)`
- `Issued`
- other blank or unresolved statuses

### Setup

Raw LendingClub files are intentionally not committed.

Default expected path at repo root:

`./accepted_2007_to_2018Q4.csv`

The active default-risk pipeline requires the accepted-loan file only. Raw LendingClub CSVs are intentionally not committed. Obtain the dataset from a public LendingClub archive or mirror, then place the accepted-loan file at the exact filename above, or pass a custom path with `--csv`.

## Reproducibility Path (Non-Docker)

Requires `python` on local path.

```bash
# create venv and install dependencies
python -m venv .venv
# PowerShell:
.\.venv\Scripts\Activate.ps1
# Git Bash:
. .venv/Scripts/activate
python -m pip install -r requirements.txt
python -m pip install -e .[dev]

python -m src.preprocessing
python -m src.train

"""
expects accepted-loan CSV at `./accepted_2007_to_2018Q4.csv`.
"""

# this writes the preprocessing artifact at: `artifacts/accepted_preprocessed.joblib`
python -m src.preprocessing

# Trains the calibrated models and write validation artifacts
python train.py

HOW IS THE BEST MODEL CHOSEN??
```

This writes:

- `artifacts/accepted_model.joblib`
- `artifacts/frontend_model.joblib`
- validation reports under `reports/validation/`

Review validation results.

- `reports/validation/metrics_summary.json`
- `reports/validation/model_card.md`
- `reports/validation/model_validation_results.csv`
- `reports/validation/risk_decile_lift.csv`
- ROC/PR/reliability plots

The committed validation evidence is meant to be readable, not exhaustive. Any large-data metrics shown in the README are example local-run results, not a promise that the committed CSVs include the full historical dataset.

To run the locked test evaluation later, once model selection is finished.

```bash
python evaluate_locked.py
```

That writes `reports/test/`. Do not treat locked test as part of the regular iteration loop.

```bash
# Start the api
uvicorn api:app --reload
```

```bash
#Check liveness and readiness
curl http://localhost:8000/health
curl http://localhost:8000/ready

"""
`/health` only checks that the service is up. `/ready` checks that both saved bundles exist and contain the required metadata. It will return `503` before training has produced artifacts, or when Docker is running without mounted artifacts.
"""

# Grade a sample default prediction

curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "loan_amnt": 10000,
    "int_rate": 12.5,
    "annual_inc": 78000,
    "dti": 15.2,
    "fico_range_low": 690,
    "fico_range_high": 694,
    "delinq_2yrs": 0,
    "inq_last_6mths": 1,
    "open_acc": 10,
    "pub_rec": 0,
    "revol_bal": 12000,
    "revol_util": 44.1,
    "total_acc": 24,
    "mort_acc": 1,
    "acc_open_past_24mths": 3,
    "pub_rec_bankruptcies": 0,
    "grade": "C",
    "sub_grade": "C2",
    "emp_length": "10+ years",
    "home_ownership": "MORTGAGE",
    "verification_status": "Verified",
    "purpose": "debt_consolidation",
    "addr_state": "CA",
    "application_type": "Individual",
    "initial_list_status": "w"
  }'
```

```bash
python batch.py docs/demo/sample_batch_input.csv docs/demo/sample_batch_output.csv --api-url http://127.0.0.1:8000
```

Look in `docs/demo/` for demo API tests.

## API Endpoints

- `GET /health`
- `GET /ready`
- `GET /model-card`
- `GET /frontend-config`
- `POST /score`
- `POST /score-frontend`
- `POST /score-batch`

The reduced frontend endpoint uses a separate saved bundle trained on the top five application-time features from the main validation feature-importance pass. It is still a default-risk model, not a separate business-decision system.

## Docker

Build:

```bash
docker build -t credit-default-api .
```

Run:

```bash
docker run --rm -p 8000:8000 -v ./artifacts:/app/artifacts credit-default-api
```

The Docker build excludes raw CSVs, `artifacts/`, and `reports/` by default through `.dockerignore`. That means container `/ready` will fail unless you either:

1. train on the host first and mount `./artifacts`, or
2. mount an existing `artifacts/` directory produced elsewhere.

### Reporting Methodology

- `reports/validation/` is the main path for validation-stage review.
- `reports/test/` should only be committed after a deliberate locked test run with `evaluate_locked.py`.
- `reports/smoke/` is for smoke/sample runs.

## Test Metrics

Current test metrics from local run:

- Rows: `134273`
- Observed default rate: `0.1989`
- Mean predicted default rate: `0.2223`
- ROC-AUC: `0.7092`
- PR-AUC: `0.3579`
- Brier score: `0.1463`
- Log loss: `0.4568`

Those files are intentionally small enough to review in git. Raw data and model binaries remain uncommitted.

## Demo Files

Committed demo files:

- `docs/demo/sample_batch_input.csv`
- `docs/demo/sample_batch_output.csv`
- `docs/demo/README.md`

## Tests

Run:

```bash
python -m pytest
```

The suite covers target construction, leakage prevention, split discipline, calibration outputs, artifact round-tripping, batch scoring, API readiness, and repo-level scope guardrails.

## Split Details

| Split | Rows | Default Rate | Date Min | Date Max |
| --- | --- | --- | --- | --- |
| train | 829355 | 0.1846 | 2007-06-01T00:00:00 | 2015-12-01T00:00:00 |
| calibration | 196607 | 0.2265 | 2016-01-01T00:00:00 | 2016-07-01T00:00:00 |
| validation | 187864 | 0.2399 | 2016-08-01T00:00:00 | 2017-06-01T00:00:00 |
| test | 134273 | 0.1989 | 2017-07-01T00:00:00 | 2018-12-01T00:00:00 |

## Limitations

- Test set may not be representative of current 2026 lending practices
- Rejected applications are excluded because repayment outcomes are not observed.
- The repo demonstrates disciplined modeling and artifact serving without production deployment controls.
- Fair-lending validation, monitoring, and operational controls are not in scope.