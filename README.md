# Credit Risk Scoring Pipeline

This repo is an accepted/funded-loan risk and profit scoring system. It is not a production underwriting engine.

It supports two label modes for accepted loans with observed repayment outcomes:

- `resolved_default`
- `default_within_horizon`

It also supports two scoring/product modes:

- `post_pricing_investment`
- `pre_underwriting_applicant`

Rejected applications are not treated as observed defaults or non-defaults. They are review-only unless real repayment outcomes exist.

## Scope

- Accepted/funded loan default modeling
- Probability calibration
- Expected-profit policy logic
- Batch scoring and FastAPI scoring from saved artifacts
- Compact model and evaluation reports
- Limited-field risk scoring for rejected-style inputs

## Target Modes

`resolved_default` is the default mode. It uses resolved good/bad loan statuses only.

`default_within_horizon` is conservative:

- labels default only when a bad status appears within the chosen horizon
- labels non-default only when there is enough observation time
- excludes censored rows instead of guessing

The default horizon is 36 months.

## Product Modes

`post_pricing_investment`

- current accepted/funded loan scoring path
- may use lender-generated pricing fields such as `grade`, `sub_grade`, `int_rate`, and `initial_list_status`
- this is the current default path in the repo

`pre_underwriting_applicant`

- excludes lender-generated pricing fields
- risk/review only unless the needed economic inputs are available
- do not claim it is a complete underwriting engine

## Profit Logic

The decision rule is expected-profit based:

```text
Expected Profit_i =
(1 - p_default_i) × [good_profit_haircut × ((installment_i × term_months_i) - funded_amnt_i)]
+ p_default_i × [-(LGD × funded_amnt_i)]
```

Expected return is:

```text
expected_return = expected_profit / funded_amnt
```

Optional NPV-style economics are supported behind config/policy metadata, but the simple EV formula remains the default.

Training selects a conservative `good_profit_haircut` and `required_return` on validation data only. The haircut is a simple, documented scenario assumption for the fact that many fully paid loans do not pay the full scheduled term; it is not a calibrated cash-profit model. Candidate policies must have non-negative scenario expected profit, non-negative expected return, non-negative required return, and positive realized validation profit. A loan is approved when:

```text
expected_return > required_return
```

If every non-empty validation policy loses realized money, training may lock a reject-all policy and record that warning in the model artifact.

Realized validation profit is backtest evidence. Expected-profit dollars are scenario estimates used for screening, not exact profit forecasts.

## Reports

Training and locked evaluation generate compact outputs under `reports/`:

- `model_card.md`
- `evaluation_summary.json`
- `calibration_table.csv`
- `policy_summary.csv`
- `sensitivity_summary.csv`
- `cohort_backtest.csv`
- `bootstrap_intervals.csv`
- `proxy_risk_diagnostics.csv`
- `fairness_caveat.md`
- `policy_selection_validation.csv`

These are summary artifacts, not compliance claims.

## How To Run

Install:

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .[dev]
```

Preprocess, train, calibrate, and select policy for the accepted-loan model:

```powershell
python train.py
```

Fixed-horizon accepted-loan training:

```powershell
python train.py --target-mode default_within_horizon --horizon-months 36
```

Train the limited-field review model:

```powershell
python train_rejected_style.py
```

Train the direct-profit challenger:

```powershell
python train_profit.py
```

Generate compact validation evaluation reports:

```powershell
python evaluation.py
```

Locked evaluation of the accepted-loan bundle:

```powershell
python evaluate_locked.py
```

Locked evaluation of the direct-profit challenger:

```powershell
python evaluate_profit_locked.py
```

Launch the API:

```powershell
python -m uvicorn api:app --reload
```

Batch scoring:

```powershell
python batch.py input.csv output.csv --bundle artifacts\accepted_model.joblib
```

Tests:

```powershell
python -m pytest
```

## Artifact Locations

Large data and model artifacts should live outside version control:

- `data/`
- `artifacts/`
- `models/`

Compact report outputs live under `reports/` and are intended to be easy to inspect and, when appropriate, commit.

## What Not To Claim

- Do not claim production underwriting.
- Do not claim rejected-applicant outcome prediction.
- Do not claim fair-lending validation.
- Do not claim selection-bias is solved.
- Do not claim locked test results were used to choose the model or policy.
- Do not claim profit estimates are exact. They are scenario-based and depend on the stated LGD and policy assumptions.
- Do not claim the fixed-horizon target is a perfect default-time model. It uses conservative observation proxies.

## Notes

- Accepted-loan training and evaluation use resolved repayment outcomes only.
- Rejected-style scoring is review-only unless a row actually has repayment outcomes.
- The repo keeps the current accepted-loan pipeline intact while adding the new horizon, reporting, and diagnostic paths behind config and metadata.
