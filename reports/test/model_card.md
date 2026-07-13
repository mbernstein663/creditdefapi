# Model Card

## Artifact Status
- Evidence label: `Full LendingClub local data`
- Evidence note: Example local-run evidence from user-supplied raw LendingClub files. Raw data is not committed.
- Evaluation split: `test`
- Training timestamp: `2026-07-13T01:06:54.665260+00:00`

## Dataset Splits

- Train: `829355`
- Calibration: `196607`
- Validation: `187864`
- Test: `134273`

## Model

- Selected model: `hist_gradient_boosting`
- Calibration method: `isotonic`
- Feature count: `25`

## Test Metrics

- Rows: `134273`
- Observed default rate: `0.1989`
- Mean predicted default rate: `0.2223`
- ROC-AUC: `0.7092`
- PR-AUC: `0.3579`
- Brier score: `0.1463`
- Log loss: `0.4568`

## Baseline Comparison

| Model | Role | ROC-AUC | PR-AUC | Brier | Log Loss | Mean PD |
| --- | --- | --- | --- | --- | --- | --- |
| hist_gradient_boosting | final_model | 0.7092 | 0.3579 | 0.1463 | 0.4568 | 0.2223 |
| base_rate | baseline | 0.5000 | 0.1989 | 0.1595 | 0.4995 | 0.1846 |
| logistic_regression | baseline | 0.7080 | 0.3567 | 0.1458 | 0.4554 | 0.1907 |
| grade_historical_rate | baseline | 0.6759 | 0.3044 | 0.1492 | 0.4660 | 0.1893 |
| sub_grade_historical_rate | baseline | 0.6853 | 0.3248 | 0.1487 | 0.4639 | 0.1908 |
