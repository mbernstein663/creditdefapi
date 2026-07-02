# LendingClub Accepted-Loan Risk/Profit Pipeline

This repo is an accepted funded-loan default and profit scoring pipeline. It is not a full rejected-applicant underwriting engine.

It trains calibrated default-risk models from LendingClub accepted loans with resolved repayment outcomes, then applies a simplified expected-value policy to funded-loan terms.

## Scope

- Accepted funded-loan default modeling
- Post-hoc probability calibration
- Simplified expected-profit and expected-return policy logic
- Batch scoring and FastAPI scoring from saved artifacts
- Limited-field risk estimate using accepted-loan outcomes projected onto rejected-application-style inputs

Rejected applications do not have repayment outcomes in this dataset. They are never assigned fake `default = 0` or `default = 1` labels and are not used for supervised training, calibration, validation, locked test evaluation, or realized-profit backtesting.

## Target

The baseline target is eventual resolved default among accepted funded loans.

Default:

- `Charged Off`
- `Default`
- `Does not meet the credit policy. Status:Charged Off`

Non-default:

- `Fully Paid`
- `Does not meet the credit policy. Status:Fully Paid`

Active, late, blank, unknown, and unresolved statuses are excluded from the baseline target. Unknown statuses fail explicitly instead of being silently labeled.

Limitation: this target predicts resolved default among accepted/funded loans. It does not estimate risk for all applicants, and it does not solve selection bias from rejected applications.

Fixed-horizon extension point: a future target such as default/charge-off within 36 months of issue date should add explicit issue-date plus performance-window logic before labeling current or unresolved loans. Do not label unresolved/current loans as good unless the fixed performance window is correctly implemented.

## Scoring Moment

| Model path | Valid scoring moment | Fields allowed | Output limits |
| --- | --- | --- | --- |
| Full accepted model, `/score` | Post-pricing/post-underwriting, after LendingClub grade/rate fields exist | May use `grade`, `sub_grade`, `int_rate`, `initial_list_status` | Calibrated accepted-loan risk plus profit policy when profit inputs are present |
| Pre-underwriting model | Before LendingClub pricing/grade assignment | Must exclude LendingClub-generated grade/rate/listing fields | Not implemented in this repo |
| Limited-field model, `/score/rejected-risk` and frontend | Rejected-application-style triage only | Uses `amount_requested`, `risk_score`, `dti`, `zip_code`, `state`, `employment_length` | Review-only limited-field risk estimate; no true rejected-applicant default labels or realized-profit claims |

## Leakage Controls

Risk-model features are explicit allowlists in `src/config.py`. Post-origination status, payment, recovery, settlement, hardship, collection, and realized-outcome fields are forbidden as model features.

Profit inputs are separate from risk features:

- `funded_amnt`
- `term_months`
- `installment`

`total_pymnt` is used only for realized-profit backtesting on accepted funded loans with observed outcomes.

## Profit Policy

Expected profit is a simplified EV approximation:

```text
(1 - p_default) * ((installment * term_months) - funded_amnt)
+ p_default * (-(LGD * funded_amnt))
```

Expected return:

```text
expected_return = expected_profit / funded_amnt
```

The selected baseline policy uses:

```text
approve = expected_return >= required_return
```

The default LGD is `1.00` and the default required return is `0.00`. Validation reports include LGD and required-return sensitivity. The simplified EV math does not model prepayment, discounting, servicing cost, recoveries, cost of capital, or timing of default.

## Direct Profit Model Challenger

The direct profit challenger predicts realized loan profit directly:

```text
realized_profit = total_pymnt - funded_amnt
```

It trains only on accepted funded loans with observed repayment outcomes. Rejected loans are not used as profit labels because their repayment outcomes are unobserved.

This is a challenger to the calibrated default-risk policy, not a replacement. It uses the same accepted-loan origination-time feature allowlist as the full accepted model, selects both model and approval policy on validation only, then evaluates the locked policy on the saved test split.

Supported challenger policies:

- approve if `predicted_profit > threshold`
- approve top X percent by `predicted_profit`

Reports compare the default-risk policy and direct-profit policy on realized profit, approval rate, default rate, and profit per dollar funded. Do not claim the challenger is better unless validation and locked-test results support it.

## Quick Start

In PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run the full pipeline:

```powershell
python train.py
python train_rejected_style.py
python train_profit.py
python evaluate_locked.py
python evaluate_profit_locked.py
```

Run the API and frontend:

```powershell
python -m uvicorn api:app --reload
```

Open `http://127.0.0.1:8000/`.

## Pipeline

Install:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -m pip install -r requirements.txt
```

Smoke-test accepted-loan training:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" train.py --sample 200000
```

Smoke-test the limited-field model:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" train_rejected_style.py --sample 200000
```

Train final accepted-loan model:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" train.py
```

Train final limited-field risk model:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" train_rejected_style.py
```

Smoke-test the direct profit challenger:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" train_profit.py --sample 20000
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" evaluate_profit_locked.py --bundle artifacts\direct_profit_model_smoke.joblib
```

Train final direct profit challenger:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" train_profit.py
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" evaluate_profit_locked.py
```

Evaluate the locked accepted-loan model on the test set only after training is complete:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" evaluate_locked.py
```

Run API:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -m uvicorn api:app --reload
```

Open `http://127.0.0.1:8000/` for the limited-field review frontend.

## Reports

Training writes ignored local reports under `reports/` and compact model cards under `docs/`:

- split date ranges, row counts, and default rates
- selected candidate and policy
- ROC AUC, PR AUC, Brier score, mean predicted default rate, actual default rate
- decile calibration table
- subgroup calibration where available: term, grade/sub-grade, amount band
- expected and realized validation profit metrics where available
- LGD and required-return sensitivity on validation data
- direct-profit challenger validation, locked-test, decile lift, and comparison outputs
- source CSV SHA-256 fingerprint, feature list, package versions, and training timestamp

Large CSVs, model binaries, and bulky generated reports are ignored by git.

## Tests

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -m pytest
```

Tests use synthetic data and cover target mapping, unresolved-status exclusion, no fake rejected labels, leakage guards, chronological split discipline, required-return policy behavior, artifact hashing/loading, batch scoring, API validation, and missing-field handling.

## Limits

- No reject inference is implemented.
- No pre-underwriting model is implemented.
- No realized-profit claims are made for rejected applications.
- Candidate selection is intentionally small: balanced logistic regression versus unweighted logistic regression.
- The full accepted model is valid only when LendingClub-generated grade/rate fields are available at the scoring moment.
- The direct-profit challenger is trained on historical realized profit and can learn historical servicing, prepayment, and selection artifacts; treat it as an empirical challenger, not causal underwriting proof.
