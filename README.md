# Credit Risk ML Pipeline

This repo builds a statistically disciplined LendingClub-style credit-risk pipeline:

- accepted funded-loan default modeling
- post-hoc probability calibration
- expected-profit approval logic
- rejected-application handling without fake labels
- batch scoring
- FastAPI scoring
- small rejected-style risk/review frontend

## Data Rules

Supervised default modeling uses only accepted funded loans with resolved outcomes.

Default:

- `Charged Off`
- `Default`
- `Does not meet the credit policy. Status:Charged Off`

Non-default:

- `Fully Paid`
- `Does not meet the credit policy. Status:Fully Paid`

Active, late, blank, and unresolved statuses are dropped before supervised splits.

Rejected applications are never assigned `default = 0` or `default = 1`. They are used only for schema checks and rejected-style risk input design.

## Pipeline

Install:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -m pip install -r requirements.txt
```

Smoke-test accepted-loan training without overwriting production artifact names:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" train.py --sample 200000
```

Smoke-test rejected-style training:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" train_rejected_style.py --sample 200000
```

Smoke outputs go to `artifacts/*_smoke.joblib` and `reports/smoke/`.

Train final accepted-loan profit model first:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" train.py
```

Train final rejected-style risk/review model second:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" train_rejected_style.py
```

Evaluate the locked accepted-loan model on test data only after training is complete:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" evaluate_locked.py
```

Locked evaluation uses the saved test IDs and source-file fingerprint from the model bundle.

Run API:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -m uvicorn api:app --reload
```

Open `http://127.0.0.1:8000/` for the frontend.

Docker serving does not bake raw CSVs or ignored local artifacts into the image. Mount artifacts or set explicit paths:

```powershell
docker build -t credit-risk-api .
docker run --rm -p 8000:8000 -v ${PWD}\artifacts:/app/artifacts credit-risk-api
```

Optional env vars:

```text
ACCEPTED_MODEL_BUNDLE=/app/artifacts/accepted_model.joblib
REJECTED_STYLE_MODEL_BUNDLE=/app/artifacts/rejected_style_model.joblib
```

Use `/ready` to verify required artifacts exist before scoring.

## Scoring Outputs

Risk features estimate calibrated `p_default`.

Profit inputs are separate and used only after prediction:

- `funded_amnt`
- `term_months`
- `installment`

Expected profit:

```text
(1 - p_default) * ((installment * term_months) - funded_amnt)
+ p_default * (-(LGD * funded_amnt))
```

Default policy:

```text
approve = expected_profit > 0
```

The baseline LGD is `1.00`.

Rejected-style frontend/API output is `review` risk-only unless profit inputs are supplied. Even with supplied profit inputs, it is still an accepted-loan-trained risk estimate, not proof of rejected-loan repayment performance.

## Reports

Training writes ignored local reports under `reports/`:

- supervised row counts by issue year/split after unresolved loans are dropped
- non-test default/non-default counts by issue year/split
- validation calibration metrics
- validation expected-profit metrics
- reliability plots

Artifacts are written under `artifacts/` and ignored by git.

## Tests

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -m pytest
```

Tests use synthetic data and cover target mapping, leakage guards, split discipline, feature mapping, expected-profit math, artifact loading, batch scoring, API validation, and rejected-style risk/review behavior.

## Limits

- No reject inference is implemented.
- No realized-profit claims are made for rejected applications.
- The accepted-loan model may use LendingClub underwriting fields like `grade`, `sub_grade`, and `int_rate`; use it only when those fields are available at the scoring moment.
- Candidate selection is intentionally small: balanced logistic regression versus unweighted logistic regression.
