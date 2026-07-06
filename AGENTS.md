# Goal

Build a statistically defensible, portfolio-ready calibrated default-risk repo centered on accepted/funded LendingClub loans.

Primary claim:

`Calibrated default-risk prediction for accepted/funded LendingClub loans, with leakage-controlled modeling, chronological validation, saved model artifacts, API-backed scoring, batch scoring, Dockerized serving, and automated tests.`

## Scope

The active repo should focus on:

1. target construction for accepted/funded loans with resolved outcomes
2. chronological train/calibration/validation/test splits
3. calibrated probability-of-default modeling
4. locked-artifact scoring through API and batch paths
5. default-risk and calibration reporting
6. rejected-application safeguards
7. clean repo hygiene and reproducibility metadata

Do not build or preserve active code paths for business-return optimization or decision-engine framing.

## Target Definition

The binary target is `default`.

Bad outcome:

* Charged Off
* Default
* Does not meet the credit policy. Status:Charged Off

Good outcome:

* Fully Paid
* Does not meet the credit policy. Status:Fully Paid

Drop unresolved or ambiguous statuses:

* Current
* In Grace Period
* Late
* Issued
* blank or unresolved statuses

Never infer outcomes for rejected applications.

## Statistical Rules

1. Use only accepted/funded loans with resolved outcomes for supervised default modeling.
2. Do not use rejected applications for supervised training, calibration, validation, or locked test evaluation.
3. Features must only use information available at application or underwriting time.
4. Post-origination repayment fields must never be model features.
5. Calibrate final probability outputs after model fitting.
6. Split chronologically by issue date into train, calibration, validation, and test.
7. Fit preprocessing and base models on train only.
8. Fit calibrators on calibration only.
9. Select model family, hyperparameters, calibration method, and display cutoffs on validation only.
10. The locked test split is for one final evaluation after selection is complete.

## Active Workflow

`train.py`

* builds target
* fits preprocessing on train only
* fits candidate default-risk models
* calibrates on calibration only
* selects the final model on validation metrics only
* saves a single bundle
* writes validation reports to `reports/validation/`

`evaluate_locked.py`

* loads the saved bundle
* verifies source fingerprint
* evaluates once on the saved locked test split
* writes test reports to `reports/test/`

`batch.py`

* loads the saved bundle only
* scores CSV rows without refitting
* writes `p_default`, `risk_band`, and model metadata

`api.py`

* loads the saved bundle only
* serves default-risk outputs only

## Required Metadata

Saved bundle metadata must include:

* target definition
* good, bad, and dropped status mappings
* split date boundaries
* split row counts
* observed default rates by split
* selected model type
* calibration method
* feature list
* forbidden leakage columns
* training timestamp
* package versions
* model version
* limitations

## Required Reports

Generate under `reports/validation/` and `reports/test/`:

* `metrics_summary.json`
* `calibration_deciles.csv`
* `risk_decile_lift.csv`
* `calibration_by_issue_year.csv`
* `calibration_by_grade.csv` when `grade` is a model feature
* `calibration_by_term.csv` when `term` is a model feature
* `roc_curve.csv`
* `pr_curve.csv`
* `reliability_plot.png`
* `roc_curve.png`
* `pr_curve.png`
* `model_card.md`

## Tests

Keep tests focused on:

* target construction
* leakage prevention
* chronological split discipline
* calibration and probability bounds
* artifact round-tripping
* batch scoring from saved artifacts only
* API scoring and readiness
* repo-level guardrails against reintroducing business-decision scope
