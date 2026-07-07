# Model Card

## Intended Use
Calibrated default-risk prediction for accepted/funded LendingClub loans with resolved outcomes.

## Not Intended Use
This project is not production underwriting, fair-lending validation, or rejected-applicant outcome prediction.

## Target Definition
- Target: `default`
- Good statuses: `Does not meet the credit policy. Status:Fully Paid, Fully Paid`
- Bad statuses: `Charged Off, Default, Does not meet the credit policy. Status:Charged Off`
- Dropped statuses: `, Current, In Grace Period, Issued, Late (16-30 days), Late (31-120 days)`

## Data Limitations
- Rejected applications: `Rejected applications are not labeled as defaults or non-defaults and are excluded from supervised training, calibration, validation, and test evaluation.`
- Limitations: `accepted-loan selection bias; rejected applications are unlabeled and excluded from supervised default modeling; not validated for production underwriting or fair-lending use`

## Leakage Controls
- Forbidden columns: `collection_recovery_fee, debt_settlement_flag, debt_settlement_flag_date, default, hardship_flag, hardship_loan_status, hardship_reason, hardship_status, hardship_type, last_credit_pull_d, last_pymnt_amnt, last_pymnt_d, loan_status, next_pymnt_d, out_prncp, out_prncp_inv, recoveries, settlement_amount, settlement_date, settlement_percentage, settlement_status, settlement_term, total_pymnt, total_pymnt_inv, total_rec_int, total_rec_late_fee, total_rec_prncp`

## Split Strategy
- Split summary: `[{"split": "train", "rows": 24, "default_rate": 0.25, "date_min": "2015-01-01T00:00:00", "date_max": "2016-12-01T00:00:00"}, {"split": "calibration", "rows": 6, "default_rate": 0.3333333333333333, "date_min": "2017-01-01T00:00:00", "date_max": "2017-06-01T00:00:00"}, {"split": "validation", "rows": 6, "default_rate": 0.16666666666666666, "date_min": "2017-07-01T00:00:00", "date_max": "2017-12-01T00:00:00"}, {"split": "test", "rows": 4, "default_rate": 0.25, "date_min": "2018-01-01T00:00:00", "date_max": "2018-04-01T00:00:00"}]`

## Cross Validation
- Summary: `{"enabled": false, "selected_model_name": "hist_gradient_boosting", "selected_calibration_method": "isotonic", "candidate_summaries": [{"model_name": "logistic", "calibration_method": "isotonic"}, {"model_name": "logistic_balanced", "calibration_method": "isotonic"}, {"model_name": "random_forest", "calibration_method": "isotonic"}, {"model_name": "hist_gradient_boosting", "calibration_method": "isotonic"}]}`

## Calibration Method
- Method: `isotonic`

## Validation Performance
- Metrics: `{"rows": 6, "observed_default_rate": 0.16666666666666666, "mean_predicted_default_rate": 0.3333333333333333, "roc_auc": 0.5, "pr_auc": 0.16666666666666666, "brier_score": 0.1666666666666667, "log_loss": 0.5209896382014885, "selected_model": "hist_gradient_boosting", "calibration_method": "isotonic"}`

## Locked Test Performance
- Metrics: `{}`

## API Usage
- `GET /health`
- `GET /ready`
- `GET /model-card`
- `GET /frontend-config`
- `POST /score`
- `POST /score-frontend`
- `POST /score-batch`

## Frontend Fields
- Reduced-feature model fields: `loan_amnt, int_rate, annual_inc, dti, fico_range_low`

## Known Risks
- accepted-loan selection bias
- unresolved outcomes are excluded rather than inferred
- no fair-lending validation