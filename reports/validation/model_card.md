# Model Card

## Artifact Status
- Evidence label: `Full LendingClub local data`
- Evidence note: Example local-run evidence from user-supplied raw LendingClub files. Raw data is not committed.
- Evaluation split: `validation`
- Training timestamp: `2026-07-08T23:56:18.180820+00:00`
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
- Train: `24`
- Calibration: `6`
- Validation: `6`
- Test: `4`

## Model
- Model type: `calibrated_hist_gradient_boosting`
- Selected model: `hist_gradient_boosting`
- Calibration method: `isotonic`
- Feature count: `25`

## Validation Metrics
- Rows: `6`
- Observed default rate: `0.1667`
- Mean predicted default rate: `0.3333`
- ROC-AUC: `0.5000`
- PR-AUC: `0.1667`
- Brier score: `0.1667`
- Log loss: `0.5210`

## Split Strategy
`[{"split": "train", "rows": 24, "default_rate": 0.25, "date_min": "2015-01-01T00:00:00", "date_max": "2016-12-01T00:00:00"}, {"split": "calibration", "rows": 6, "default_rate": 0.3333333333333333, "date_min": "2017-01-01T00:00:00", "date_max": "2017-06-01T00:00:00"}, {"split": "validation", "rows": 6, "default_rate": 0.16666666666666666, "date_min": "2017-07-01T00:00:00", "date_max": "2017-12-01T00:00:00"}, {"split": "test", "rows": 4, "default_rate": 0.25, "date_min": "2018-01-01T00:00:00", "date_max": "2018-04-01T00:00:00"}]`

## Known Limits
- accepted-loan selection bias
- rejected applications are unlabeled and excluded from supervised default modeling
- not validated for production underwriting or fair-lending use
