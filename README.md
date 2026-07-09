# Credit Default Risk API

This repo is a calibrated, API-backed LendingClub default-risk project. It uses previously accepted/funded loans with resolved outcomes to estimate probability of default with leakage-controlled modeling, chronological validation, saved artifacts, API-backed scoring, batch scoring, Dockerized serving, and tests.

## Overview

This repo is not production underwriting or a rejected-applicant model.

## Scope

- Supervised modeling uses accepted/funded loans with resolved outcomes only.
- The target is binary `default`.
- Outputs are calibrated `p_default`
- Features are limited to application-time or underwriting-time fields.
- Post-origination repayment fields are forbidden model inputs.

## Target Definition

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

## Data Setup

Raw LendingClub files are intentionally not committed.

Default expected path at repo root:

- `./accepted_2007_to_2018Q4.csv`

The active default-risk pipeline requires the accepted-loan file only. Raw LendingClub CSVs are intentionally not committed. Obtain the dataset from a public LendingClub archive or mirror, then place the accepted-loan file at the exact filename above, or pass a custom path with `--csv`.

## Local Venv Run

Default paths:

- Accepted data: `./accepted_2007_to_2018Q4.csv`
- Preprocessed cache: `./artifacts/accepted_preprocessed.joblib`
- Main bundle: `./artifacts/accepted_model.joblib`
- Frontend bundle: `./artifacts/frontend_model.joblib`

Requires `python` on local path and the accepted-loan CSV at the repo root.

```bash
python -m venv .venv
# PowerShell:
.\.venv\Scripts\Activate.ps1
# Git Bash:
. .venv/Scripts/activate
python -m pip install -r requirements.txt
python -m pip install -e .[dev]

python -m src.preprocessing
python -m src.train
```

This writes:

- `artifacts/accepted_model.joblib`
- `artifacts/frontend_model.joblib`
- validation reports under `reports/validation/`

Review validation results for model selection:

- `reports/validation/metrics_summary.json`
- `reports/validation/model_card.md`
- `reports/validation/model_validation_results.csv`
- `reports/validation/risk_decile_lift.csv`
- ROC/PR/reliability plots

Validation compares the configured candidate runs and confirms/tunes the selected model and calibration choice. In the committed config, `selected_model` pins histogram gradient boosting while still retaining candidate comparison output. Validation is not the final held-out performance claim.

Deliberately run the locked test evaluation after model selection is finished:

```bash
python evaluate_locked.py
```

That writes the final committed model evidence under `reports/test/`. Do not treat locked test as part of the regular iteration loop.

Expected locked-test report files include:

- `reports/test/model_card.md`
- `reports/test/metrics_summary.json`
- `reports/test/baseline_comparison.csv`
- `reports/test/baseline_comparison.json`
- `reports/test/calibration_deciles.csv`
- `reports/test/risk_decile_lift.csv`
- `reports/test/roc_curve.csv`
- `reports/test/pr_curve.csv`
- `reports/test/reliability_plot.png`
- `reports/test/roc_curve.png`
- `reports/test/pr_curve.png`

Start the API:

```bash
uvicorn api:app --reload
```

Check liveness and readiness:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

`/health` only checks that the service is up. `/ready` checks that both saved bundles exist and contain the required metadata. It will return `503` before training has produced artifacts, or when Docker is running without mounted artifacts.

Score a single row or batch file:

```bash
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

Smoke/sample runs are for quick checks only and are not final model evidence:

```bash
python -m src.preprocessing --sample 5000
python -m src.train --sample 5000
```

## API

Endpoints:

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
docker run --rm -p 8000:8000 -v "${PWD}/artifacts:/app/artifacts" credit-default-api
```

On Windows PowerShell, this mount form is usually clearer:

```powershell
docker run --rm -p 8000:8000 -v "${PWD}\artifacts:/app/artifacts" credit-default-api
```

The Docker build excludes raw CSVs, `artifacts/`, and `reports/` by default through `.dockerignore`. Train on the host first, then mount `./artifacts`; otherwise container `/ready` will fail because the saved bundles are absent.

## Reporting Methodology

- `reports/test/model_card.md` is the main final evidence after the locked test run.
- `reports/validation/` is secondary evidence used to compare configured candidates and confirm/tune the selected model/calibration choice.
- `reports/test/` should only be regenerated by deliberately running `python evaluate_locked.py` after selection is complete.
- `reports/smoke/` is for smoke/sample runs and is not model evidence.
- `metrics_summary.json` and `model_card.md` label each report as one of:
  - full LendingClub local data
  - smoke-test sample
  - synthetic/test fixture

## Final Evidence

The final model evidence is produced by:

```bash
python evaluate_locked.py
```

The locked test report compares the selected calibrated model against simple baselines fit only on train/calibration rows: base-rate, logistic regression when feasible, and historical grade/sub_grade rates when those fields are available.

Raw data and model binaries remain uncommitted.

## Demo Files

Committed demo files:

- `docs/demo/sample_batch_input.csv`
- `docs/demo/sample_batch_output.csv`
- `docs/demo/README.md`

These files demonstrate API/batch mechanics only. They are not evidence for model performance.

## Tests

Run:

```bash
python -m pytest
```

The suite covers target construction, leakage prevention, split discipline, calibration outputs, artifact round-tripping, batch scoring, API readiness, and repo-level scope guardrails.

## Limitations

- Accepted-loan selection bias remains because the supervised population is accepted/funded loans only.
- Rejected applications are excluded because repayment outcomes are not observed.
- The repo demonstrates disciplined modeling and artifact-backed serving, not production deployment controls.
- Fair-lending validation, monitoring, drift management, and operational controls are out of scope.
