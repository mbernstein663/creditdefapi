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
- Summary: `{"selected_model_name": "logistic", "selected_calibration_method": "isotonic", "candidate_summaries": [{"model_name": "logistic_balanced", "calibration_method": "isotonic", "fold_count": 2, "folds": [{"fold": 1, "train_rows": 8, "validation_rows": 8, "roc_auc": 1.0, "pr_auc": 1.0, "brier_score": 0.12130438990311498, "log_loss": 0.3171311898402286, "mean_predicted_default": 0.09804810956289021, "actual_default_rate": 0.25, "mean_absolute_calibration_gap": 0.19449174710970044}, {"fold": 2, "train_rows": 16, "validation_rows": 8, "roc_auc": 1.0, "pr_auc": 1.0, "brier_score": 0.05739722001642732, "log_loss": 0.21036103952964896, "mean_predicted_default": 0.2512413379471673, "actual_default_rate": 0.25, "mean_absolute_calibration_gap": 0.17046007745443092}], "mean_roc_auc": 1.0, "mean_pr_auc": 1.0, "mean_brier_score": 0.08935080495977116, "mean_log_loss": 0.2637461146849388, "mean_absolute_calibration_gap": 0.18247591228206567, "mean_predicted_default": 0.17464472375502876, "mean_actual_default_rate": 0.25}, {"model_name": "logistic_balanced", "calibration_method": "sigmoid", "fold_count": 2, "folds": [{"fold": 1, "train_rows": 8, "validation_rows": 8, "roc_auc": 1.0, "pr_auc": 1.0, "brier_score": 0.12130438990311498, "log_loss": 0.3171311898402286, "mean_predicted_default": 0.09804810956289021, "actual_default_rate": 0.25, "mean_absolute_calibration_gap": 0.19449174710970044}, {"fold": 2, "train_rows": 16, "validation_rows": 8, "roc_auc": 1.0, "pr_auc": 1.0, "brier_score": 0.05739722001642732, "log_loss": 0.21036103952964896, "mean_predicted_default": 0.2512413379471673, "actual_default_rate": 0.25, "mean_absolute_calibration_gap": 0.17046007745443092}], "mean_roc_auc": 1.0, "mean_pr_auc": 1.0, "mean_brier_score": 0.08935080495977116, "mean_log_loss": 0.2637461146849388, "mean_absolute_calibration_gap": 0.18247591228206567, "mean_predicted_default": 0.17464472375502876, "mean_actual_default_rate": 0.25}, {"model_name": "logistic", "calibration_method": "isotonic", "fold_count": 2, "folds": [{"fold": 1, "train_rows": 8, "validation_rows": 8, "roc_auc": 1.0, "pr_auc": 1.0, "brier_score": 0.16585578623980632, "log_loss": 0.4328187954888952, "mean_predicted_default": 0.05953411926362692, "actual_default_rate": 0.25, "mean_absolute_calibration_gap": 0.21622893415236075}, {"fold": 2, "train_rows": 16, "validation_rows": 8, "roc_auc": 1.0, "pr_auc": 1.0, "brier_score": 0.07770985574186527, "log_loss": 0.24436346752888016, "mean_predicted_default": 0.17825429928141, "actual_default_rate": 0.25, "mean_absolute_calibration_gap": 0.17870412716520467}], "mean_roc_auc": 1.0, "mean_pr_auc": 1.0, "mean_brier_score": 0.12178282099083579, "mean_log_loss": 0.33859113150888764, "mean_absolute_calibration_gap": 0.19746653065878272, "mean_predicted_default": 0.11889420927251847, "mean_actual_default_rate": 0.25}, {"model_name": "logistic", "calibration_method": "sigmoid", "fold_count": 2, "folds": [{"fold": 1, "train_rows": 8, "validation_rows": 8, "roc_auc": 1.0, "pr_auc": 1.0, "brier_score": 0.16585578623980632, "log_loss": 0.4328187954888952, "mean_predicted_default": 0.05953411926362692, "actual_default_rate": 0.25, "mean_absolute_calibration_gap": 0.21622893415236075}, {"fold": 2, "train_rows": 16, "validation_rows": 8, "roc_auc": 1.0, "pr_auc": 1.0, "brier_score": 0.07770985574186527, "log_loss": 0.24436346752888016, "mean_predicted_default": 0.17825429928141, "actual_default_rate": 0.25, "mean_absolute_calibration_gap": 0.17870412716520467}], "mean_roc_auc": 1.0, "mean_pr_auc": 1.0, "mean_brier_score": 0.12178282099083579, "mean_log_loss": 0.33859113150888764, "mean_absolute_calibration_gap": 0.19746653065878272, "mean_predicted_default": 0.11889420927251847, "mean_actual_default_rate": 0.25}, {"model_name": "random_forest", "calibration_method": "isotonic", "fold_count": 2, "folds": [{"fold": 1, "train_rows": 8, "validation_rows": 8, "roc_auc": 0.5, "pr_auc": 0.25, "brier_score": 0.18769775390625001, "log_loss": 0.5628499271106359, "mean_predicted_default": 0.2640625, "actual_default_rate": 0.25, "mean_absolute_calibration_gap": 0.014062499999999978}, {"fold": 2, "train_rows": 16, "validation_rows": 8, "roc_auc": 0.5, "pr_auc": 0.25, "brier_score": 0.187744140625, "log_loss": 0.5629690460630739, "mean_predicted_default": 0.265625, "actual_default_rate": 0.25, "mean_absolute_calibration_gap": 0.015625}], "mean_roc_auc": 0.5, "mean_pr_auc": 0.25, "mean_brier_score": 0.187720947265625, "mean_log_loss": 0.5629094865868549, "mean_absolute_calibration_gap": 0.014843749999999989, "mean_predicted_default": 0.26484375, "mean_actual_default_rate": 0.25}, {"model_name": "random_forest", "calibration_method": "sigmoid", "fold_count": 2, "folds": [{"fold": 1, "train_rows": 8, "validation_rows": 8, "roc_auc": 0.5, "pr_auc": 0.25, "brier_score": 0.18769775390625001, "log_loss": 0.5628499271106359, "mean_predicted_default": 0.2640625, "actual_default_rate": 0.25, "mean_absolute_calibration_gap": 0.014062499999999978}, {"fold": 2, "train_rows": 16, "validation_rows": 8, "roc_auc": 0.5, "pr_auc": 0.25, "brier_score": 0.187744140625, "log_loss": 0.5629690460630739, "mean_predicted_default": 0.265625, "actual_default_rate": 0.25, "mean_absolute_calibration_gap": 0.015625}], "mean_roc_auc": 0.5, "mean_pr_auc": 0.25, "mean_brier_score": 0.187720947265625, "mean_log_loss": 0.5629094865868549, "mean_absolute_calibration_gap": 0.014843749999999989, "mean_predicted_default": 0.26484375, "mean_actual_default_rate": 0.25}, {"model_name": "hist_gradient_boosting", "calibration_method": "isotonic", "fold_count": 2, "folds": [{"fold": 1, "train_rows": 8, "validation_rows": 8, "roc_auc": 0.5, "pr_auc": 0.25, "brier_score": 0.1875, "log_loss": 0.5623351446188083, "mean_predicted_default": 0.25, "actual_default_rate": 0.25, "mean_absolute_calibration_gap": 0.0}, {"fold": 2, "train_rows": 16, "validation_rows": 8, "roc_auc": 0.5, "pr_auc": 0.25, "brier_score": 0.1875, "log_loss": 0.5623351446188083, "mean_predicted_default": 0.25, "actual_default_rate": 0.25, "mean_absolute_calibration_gap": 0.0}], "mean_roc_auc": 0.5, "mean_pr_auc": 0.25, "mean_brier_score": 0.1875, "mean_log_loss": 0.5623351446188083, "mean_absolute_calibration_gap": 0.0, "mean_predicted_default": 0.25, "mean_actual_default_rate": 0.25}, {"model_name": "hist_gradient_boosting", "calibration_method": "sigmoid", "fold_count": 2, "folds": [{"fold": 1, "train_rows": 8, "validation_rows": 8, "roc_auc": 0.5, "pr_auc": 0.25, "brier_score": 0.1875, "log_loss": 0.5623351446188083, "mean_predicted_default": 0.25, "actual_default_rate": 0.25, "mean_absolute_calibration_gap": 0.0}, {"fold": 2, "train_rows": 16, "validation_rows": 8, "roc_auc": 0.5, "pr_auc": 0.25, "brier_score": 0.1875, "log_loss": 0.5623351446188083, "mean_predicted_default": 0.25, "actual_default_rate": 0.25, "mean_absolute_calibration_gap": 0.0}], "mean_roc_auc": 0.5, "mean_pr_auc": 0.25, "mean_brier_score": 0.1875, "mean_log_loss": 0.5623351446188083, "mean_absolute_calibration_gap": 0.0, "mean_predicted_default": 0.25, "mean_actual_default_rate": 0.25}]}`

## Calibration Method
- Method: `isotonic`

## Validation Performance
- Metrics: `{"rows": 6, "observed_default_rate": 0.16666666666666666, "mean_predicted_default_rate": 0.18948970832406573, "roc_auc": 1.0, "pr_auc": 1.0, "brier_score": 0.0017062759971493196, "log_loss": 0.02372474595422038, "selected_model": "logistic", "calibration_method": "isotonic"}`

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