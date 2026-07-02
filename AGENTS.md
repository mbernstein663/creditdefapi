# Goal

You are building a statistically defensible, modular credit-risk ML pipeline repo. It has multiple purposes:

1. A full ML pipeline of python files that import, preprocess, and fit a model that maximizes expected profits across loans with an adjusted lending policy.
2. A small and well-labeled API ingestion through an api.py script that uses simple CRUD to evaluate single and batch loan approval.
3. A small frontend that provides simple loan evaluation by collecting only the fields required by the deployed scoring model. If the frontend uses a reduced input set, then a separate simplified model must be trained on exactly those frontend-supported fields.
4. Careful handling of accepted funded loans versus rejected loan applications.

It should contain:

1. data/preprocessing
2. model scoring
3. probability calibration
4. expected-profit decision logic
5. accepted/rejected loan handling
6. batch scoring + API integration
7. clean evaluation frontend

## Target Definition

The binary target is `default`.

Use only resolved funded-loan outcomes for supervised default modeling.

Default / bad outcome:

* Charged Off
* Default
* Does not meet the credit policy. Status:Charged Off

Non-default / good outcome:

* Fully Paid
* Does not meet the credit policy. Status:Fully Paid

Drop unresolved or active statuses from supervised training/evaluation:

* Current
* In Grace Period
* Late
* Issued
* Any status without a final repayment outcome

Do not silently include ambiguous statuses.

Rejected loan applications are different from resolved funded loans. A rejected application is not an observed default and is not an observed non-default. Do not assign `default = 1` or `default = 0` to rejected applications unless the data contains a real repayment outcome.

Rejected applications may be used for schema design, feature availability checks, accepted-vs-rejected distribution analysis, API/frontend input design, and optional reject-inference experiments. They must not be used for supervised default training, calibration, validation, test evaluation, or realized-profit backtesting unless true repayment outcomes exist.

## Tech Stack

Do not overengineer. Keep the repo understandable for a portfolio reviewer.

Use:

```text
- FastAPI
- Docker
- Python
- scikit-learn
- numpy
- pandas
- matplotlib
- joblib for artifact serialization
```

## Approach

### Split Discipline

Use four logical datasets for accepted funded loans with resolved outcomes:

1. Train

   * Fit preprocessing
   * Fit candidate models

2. Calibration

   * Calibrate predicted default probabilities
   * Do not use for model fitting except calibration

3. Validation / Policy Selection

   * Compare candidate models
   * Choose the final model
   * Choose LGD assumption if being tuned
   * Choose expected-profit threshold or required-return policy

4. Test

   * Evaluate the single locked final model and single locked final policy exactly once
   * Do not tune, compare, select, calibrate, debug, or adjust based on test results

Any code that chooses a model, threshold, feature set, calibration method, LGD assumption, or policy using the test set is invalid.

Rejected applications should not be included in these supervised default splits unless they have true repayment outcomes. If rejected applications are analyzed, keep that analysis separate from accepted-loan model training and evaluation.

## Lending Policy and Evaluation

The lending policy must be based on expected profit, not a raw default-probability cutoff.

Use calibrated predicted default probabilities to compute loan-level expected profit:

```text
Expected Profit_i =
(1 - p_default_i) × [(installment_i × term_months_i) - funded_amnt_i]
+
p_default_i × [-(LGD × funded_amnt_i)]
```

For historical backtesting, realized profit may be computed as:

```text
realized_profit_i = total_pymnt_i - funded_amnt_i
```

This is for evaluation only. `total_pymnt` and related repayment fields must never be used as model features or approval-decision inputs.

Where:

```text
p_default_i      = calibrated predicted probability that loan i defaults
1 - p_default_i  = probability that loan i does not default
installment_i    = monthly payment
term_months_i    = loan term in months, usually 36 or 60
funded_amnt_i    = funded loan amount / principal
LGD              = loss given default assumption, such as 1.00 for full principal loss,
                   0.75 for 75% loss, or 0.60 for 60% loss
```

Also compute:

```text
expected_return_i = expected_profit_i / funded_amnt_i
```

The baseline approval policy is:

```text
approve_i = expected_profit_i > 0
```

If a required return threshold is used, the approval policy becomes:

```text
approve_i = expected_return_i > required_return
```

If a capital budget is introduced, rank approved loans by expected_return descending and approve until the budget is exhausted.

Do not use a raw default-probability threshold as the primary lending policy unless explicitly requested, because default probability alone ignores interest rate, term, funded amount, and expected loss.

Test-set evaluation is forbidden until the final model, calibration method, LGD assumption, required-return threshold if used, and lending policy have already been selected using non-test data.

The final evaluation should report both expected and realized performance on accepted funded loans with resolved outcomes:

```text
expected_profit
expected_return
approval_count
selection_rate
actual_default_rate among approved loans
mean_expected_profit among approved loans
mean_expected_return among approved loans
total_realized_profit among approved loans
mean_realized_profit among approved loans
```

Rejected applications may be scored only if they contain the required scoring fields, but they must not be used for realized-profit claims because they do not have observed repayment outcomes.

## Calibration Requirements

Evaluate calibrated probabilities using:

* Brier score
* calibration curve / reliability plot
* mean predicted default rate vs actual default rate
* decile-level observed default rates

AUC may be reported but is not sufficient for profit-based lending decisions.

Do not treat class-weighted or resampled model probabilities as calibrated without post-hoc calibration.

Calibration must use accepted funded loans with resolved outcomes only. Rejected applications must not be used for calibration unless they have true repayment outcomes.

## Leakage Rules

Features must only use information available at loan application / underwriting time.

Never use post-origination or outcome-derived columns as model features, including but not limited to:

* loan_status
* total_pymnt
* total_pymnt_inv
* recoveries
* collection_recovery_fee
* last_pymnt_d
* last_pymnt_amnt
* next_pymnt_d
* last_credit_pull_d
* out_prncp
* out_prncp_inv
* total_rec_prncp
* total_rec_int
* total_rec_late_fee
* settlement_status
* hardship_flag
* debt_settlement_flag
* any column directly created from repayment performance after origination

These columns may be used only for target construction or realized-profit evaluation, never for model training/scoring.

Rejected-loan decision fields should not be used as default-model features. Rejected applications can help define which application-time fields are realistically available, but they cannot create default labels by assumption.

## Model Selection

Select the final model using validation-set evidence.

Primary criteria:

* calibrated probability quality
* validation expected profit
* validation realized profit backtest
* approval rate stability
* simplicity/explainability for portfolio use

Secondary criteria:

* ROC AUC
* PR AUC
* accuracy
* F1
* brier score

Do not select the final model based only on AUC.

If the deployed API/frontend is intended to score rejected-application-style records, the model must use fields available in that scoring population or clearly reject/flag unsupported inputs. Do not silently fake unavailable fields.

### Subagent Audit Gate

Before API/frontend work, run a read-only audit using subagents.

Required subagents:

1. Data Leakage Reviewer
2. Modeling/Statistics Reviewer
3. Profit Math Reviewer
4. Rejected-Loan / Selection-Bias Reviewer
5. Architecture/AI-Slop Reviewer
6. Testing/Reproducibility Reviewer

### Repo Structure

`src`:

* `preprocessing.py`: separates accepted funded loans from rejected applications, sets binary target variable for resolved funded loans, and gets data ready for modeling
* `models.py`: trains simple candidate classifiers that estimate probability of default, such as logistic regression, random forest, gradient boosting, and optionally linear SVM with calibration. Use joblib only for artifact serialization.
* `calibration.py`: calibrates predicted default probabilities so predicted risk levels better match observed default rates. It must not use the final test set or rejected applications without outcomes.
* `profit.py`: computes expected_profit, expected_return, approval decisions, and realized-profit backtests. It must not train models or access the test set for policy selection.
* `evaluation.py`: evaluates a locked model and locked policy. Test-set evaluation is allowed only after model, calibration method, LGD assumption, and approval policy are finalized on non-test data.
* `artifacts.py`: save/load model bundle
* `scorer.py`: shared scoring logic used by batch.py and api.py
* `schemas.py`: API/batch input/output validation
* `config.py`: central paths/columns/settings

Training must save a single versioned model bundle containing:

* fitted preprocessor
* fitted classifier
* fitted calibrator or calibrated classifier
* selected feature columns in exact training order
* required input schema
* target definition metadata
* accepted/rejected data handling notes
* forbidden leakage columns
* selected approval policy
* LGD assumption
* required_return or expected_profit threshold if used
* package/version metadata where practical
* training timestamp

## Expected Final Deliverables

By the end, the repo should have:

1. FastAPI integration
2. cleaned src/
3. tested expected-value math
4. leakage prevention
5. split/calibration discipline documented
6. rejected-loan handling documented
7. tests for math, leakage, splits, rejected-loan handling, and batch scoring
8. updated README with honest claims

Work incrementally.

Make small commits conceptually.

Do not hide uncertainty.

## Required Tests

Add tests for:

1. Target construction

   * resolved good statuses map to 0
   * resolved bad statuses map to 1
   * unresolved statuses are dropped
   * rejected applications are not assigned default labels by assumption

2. Leakage prevention

   * forbidden post-origination columns are excluded from model features
   * rejected-loan decision fields are not used as default-model features

3. Split discipline

   * train/calibration/validation/test are disjoint
   * test set is not used in model selection or threshold selection
   * rejected applications do not contaminate supervised accepted-loan evaluation

4. Expected profit math

   * expected_profit matches hand-calculated examples
   * expected_return = expected_profit / funded_amnt
   * approval rule behaves correctly

5. Artifact loading

   * saved model bundle loads successfully
   * batch scoring uses saved artifacts without refitting

6. Batch scoring

   * batch scoring returns p_default, expected_profit, expected_return, decision
   * output row count matches input row count
   * rejected-style inputs are scored only when required fields are present

7. API scoring

   * valid request returns approve/deny/review
   * invalid request fails with a clear validation error
   * unsupported rejected-style applications are flagged or rejected clearly

## Non-Negotiable Statistical Rules

This project is invalid if any of these rules are broken:

1. The final test set must not be used for model selection, threshold selection, policy selection, feature selection, calibration, LGD tuning, debugging, or iteration.
2. Model features must only use information available at loan application / underwriting time.
3. Post-origination repayment fields must never enter model features.
4. Class-weighted, resampled, or boosted model probabilities must not be treated as calibrated without calibration.
5. AUC is not the final objective. The final objective is statistically defensible profit-based approval using calibrated default probabilities.
6. Expected profit is for decision-making. Realized profit is for historical backtesting only.
7. Batch scoring and API scoring must load saved artifacts and must not refit preprocessing or models.
8. Any simplified frontend must use a model trained on exactly the fields the frontend collects.
9. All profit claims must state the LGD assumption and approval policy used.
10. If uncertainty exists, document it instead of inventing precision.
11. Rejected applications must not be treated as defaults or non-defaults unless true repayment outcomes exist.
12. Rejected applications must not be used for calibration, realized-profit backtesting, or final model evaluation unless true repayment outcomes exist.

**Acceptable initial LGD values:**

* 1.00 conservative full principal loss
* 0.60 or 0.75 as documented sensitivity assumptions
