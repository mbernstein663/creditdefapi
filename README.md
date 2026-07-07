# Calibrated Credit Default Risk API

## Overview

This project predicts calibrated probability of default for accepted/funded LendingClub loans. It is built as a clean portfolio repo around leakage-controlled modeling, chronological validation, saved model artifacts, API-backed scoring, batch scoring, Dockerized serving, and automated tests.

## What the project does

The active workflow trains default-risk models on accepted/funded loans with resolved repayment outcomes, calibrates probabilities on a dedicated calibration split, selects the final model on validation metrics, saves a versioned bundle, and serves that bundle through FastAPI and batch scoring.

Outputs are calibrated `p_default` values and a display-only `risk_band`. The project demonstrates ML pipeline discipline, calibration reporting, artifact-based serving, and test coverage.

The repo also trains a separate reduced-feature frontend model on exactly the top five ranked application-time attributes from the main validation feature-importance pass. That frontend model is served through its own FastAPI scoring endpoint.

## What the project does not claim

- It is not production underwriting.
- It is not fair-lending validated.
- It is not rejected-applicant outcome prediction.
- It is not a profit model or lending policy engine.

Rejected applications are not labeled as defaults or non-defaults because their repayment outcomes are not observed.

## Data and target definition

The supervised target is `default`.

Bad outcome:

- `Charged Off`
- `Default`
- the resolved `Does not meet the credit policy. Status:Charged Off` variant

Good outcome:

- `Fully Paid`
- the resolved `Does not meet the credit policy. Status:Fully Paid` variant

Dropped from supervised training and evaluation:

- `Current`
- `In Grace Period`
- late or active statuses
- `Issued`
- blank or unresolved statuses

Only accepted/funded loans with resolved outcomes are used for supervised training, calibration, validation, and locked test evaluation.

## Modeling approach

The repo trains simple scikit-learn classifiers for default-risk prediction, including logistic regression, random forest, and histogram gradient boosting. Feature handling is limited to application or underwriting-time fields only. Post-origination repayment or outcome-derived columns are explicitly excluded from model features.

## Calibration and validation design

Splits are chronological by `issue_d`:

1. `train`
2. `calibration`
3. `validation`
4. `test`

The training split fits preprocessing and candidate models. The calibration split fits post-hoc calibrators only. The validation split selects model family, calibration method, and display cutoffs such as risk bands. The locked test split is evaluated only after the final bundle has been saved.

Model selection is calibration-first, using:

- Brier score
- log loss
- ROC AUC
- PR AUC
- mean predicted default rate versus observed default rate
- decile calibration gaps
- risk-decile lift

## Reports generated

Validation reports are written to `reports/validation/` and locked test reports to `reports/test/`.

Each stage writes:

- `metrics_summary.json`
- `calibration_deciles.csv`
- `risk_decile_lift.csv`
- `calibration_by_issue_year.csv`
- `calibration_by_grade.csv` when `grade` is in the feature set
- `roc_curve.csv`
- `pr_curve.csv`
- `reliability_plot.png`
- `roc_curve.png`
- `pr_curve.png`
- `model_card.md`

## API usage

Run the API locally:

```bash
uvicorn api:app --reload
```

Health check:

```bash
curl http://localhost:8000/health
```

Example score request:

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

Expected response shape:

```json
{
  "p_default": 0.184,
  "risk_band": "medium",
  "model_version": "accepted-default-v1",
  "model_type": "calibrated_hist_gradient_boosting",
  "calibration_method": "isotonic",
  "scoring_note": "Probability is calibrated for accepted/funded LendingClub-style loans with resolved historical outcomes."
}
```

Reduced frontend scoring:

```bash
curl http://localhost:8000/frontend-config
curl -X POST http://localhost:8000/score-frontend \
  -H "Content-Type: application/json" \
  -d '{
    "loan_amnt": 10000,
    "int_rate": 12.5,
    "annual_inc": 78000,
    "dti": 15.2,
    "fico_range_low": 690
  }'
```

## Batch scoring usage

```bash
python batch.py input.csv output.csv --api-url http://127.0.0.1:8000
```

The CLI uploads the CSV to the FastAPI batch endpoint, validates the data shape and required attributes, and writes a scored CSV that preserves row order. The response CSV includes the original columns plus:

- `p_default`
- `p_non_default`
- `confidence`
- `risk_band`
- `model_version`
- `model_type`
- `calibration_method`

## Docker usage

Build and run:

```bash
docker build -t credit-default-api .
docker run --rm -p 8000:8000 -v ./artifacts:/app/artifacts credit-default-api
```

Then check:

```bash
curl http://localhost:8000/health
```

## Runbook Notes

- `config.yaml` controls training cost and candidate selection.
- `cross_validation: false` skips CV; `true` enables it.
- `selected_model:` can pin the saved bundle to an enabled candidate while still reporting every candidate.
- `models:` now uses boolean flags such as `logistic regression: true` and `random forest: false`.
- `calibration_methods:` uses boolean flags such as `isotonic: true` and `sigmoid: false`.
- Run `python preprocessing.py` once to write `artifacts/accepted_preprocessed.joblib`; `train.py` and `evaluate_locked.py` reuse it when the source CSV fingerprint matches.
- If Docker is installed but `docker build` fails with engine or pipe errors, start Docker Desktop and wait for the engine to come online.
- On Windows, `Start-Process "$env:ProgramFiles\Docker\Docker\Docker Desktop.exe"` starts Docker Desktop.
- `formatt.txt` is the copy-paste command list for the full local runbook.

## Tests

Run:

```bash
python -m pytest
```

The suite covers target construction, leakage prevention, split discipline, calibration outputs, artifact round-tripping, API scoring, batch scoring, and repo guardrails against reintroducing business-decision scope.

## Repo structure

```text
api.py
batch.py
evaluate_locked.py
evaluation.py
train.py
src/
  artifacts.py
  calibration.py
  config.py
  evaluation.py
  models.py
  preprocessing.py
  scorer.py
  schemas.py
tests/
Dockerfile
```

## Limitations

- The training population is accepted/funded loans only, so accepted-loan selection bias remains.
- Rejected applications are excluded from supervised default modeling because outcomes are not observed.
- The repo demonstrates disciplined modeling and serving, not production deployment controls.
- The repo does not include fairness, monitoring, drift management, or operational risk controls needed for real lending use.

## Latest Validation Model Results

Saved CSV: `reports/validation/model_validation_results.csv`

| model | calibration | ROC AUC | PR AUC | Brier | Log loss | mean predicted default | observed default rate | selected |
|---|---|---:|---:|---:|---:|---:|---:|---|
| logistic | isotonic | 0.693867 | 0.394177 | 0.167080 | 0.507400 | 0.235246 | 0.239892 | no |
| logistic_balanced | isotonic | 0.696190 | 0.397752 | 0.166573 | 0.506146 | 0.231322 | 0.239892 | no |
| random_forest | isotonic | 0.690086 | 0.395406 | 0.167419 | 0.508428 | 0.239207 | 0.239892 | no |
| hist_gradient_boosting | isotonic | 0.697473 | 0.404248 | 0.166195 | 0.505037 | 0.233484 | 0.239892 | yes |

## Locked Test Results

Saved reports: `reports/test/`

| model | calibration | rows | ROC AUC | PR AUC | Brier | Log loss | mean predicted default | observed default rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| hist_gradient_boosting | isotonic | 134273 | 0.709191 | 0.357852 | 0.146283 | 0.456757 | 0.222296 | 0.198856 |
