# Model Card

## Artifact Status
- Evidence label: `Full LendingClub local data`
- Evidence note: Example local-run evidence from user-supplied raw LendingClub files. Raw data is not committed.
- Evaluation split: `validation`
- Training timestamp: `2026-07-09T21:07:20.724803+00:00`
- Sample rows requested: `None`

## Purpose
Calibrated default-risk prediction for accepted LendingClub loans with resolved outcomes.

## Scope
Evaluated on accepted/funded loans only. Not a production underwriting system or rejected-applicant outcome model.

## Target
- Target: `default`
- Good statuses: `Does not meet the credit policy. Status:Fully Paid, Fully Paid`
- Bad statuses: `Charged Off, Default, Does not meet the credit policy. Status:Charged Off`
- Dropped statuses: `, Current, In Grace Period, Issued, Late (16-30 days), Late (31-120 days)`

## Dataset Splits
- Train: `829355`
- Calibration: `196607`
- Validation: `187864`
- Test: `134273`

## Model
- Model type: `calibrated_hist_gradient_boosting`
- Selected model: `hist_gradient_boosting`
- Calibration method: `isotonic`
- Feature count: `25`

## Validation Metrics
- Rows: `187864`
- Observed default rate: `0.2399`
- Mean predicted default rate: `0.2335`
- ROC-AUC: `0.6975`
- PR-AUC: `0.4042`
- Brier score: `0.1662`
- Log loss: `0.5050`

## Split Strategy
`[{"split": "train", "rows": 829355, "default_rate": 0.18455908507213437, "date_min": "2007-06-01T00:00:00", "date_max": "2015-12-01T00:00:00"}, {"split": "calibration", "rows": 196607, "default_rate": 0.22647718545117926, "date_min": "2016-01-01T00:00:00", "date_max": "2016-07-01T00:00:00"}, {"split": "validation", "rows": 187864, "default_rate": 0.2398916237278031, "date_min": "2016-08-01T00:00:00", "date_max": "2017-06-01T00:00:00"}, {"split": "test", "rows": 134273, "default_rate": 0.19885606190373345, "date_min": "2017-07-01T00:00:00", "date_max": "2018-12-01T00:00:00"}]`

## Known Limits
- accepted-loan selection bias
- rejected applications are unlabeled and excluded from supervised default modeling
- not validated for production underwriting or fair-lending use
