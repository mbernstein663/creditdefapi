# Model Card

## Data Source
- Source fingerprint: `{"path": "C:\\Users\\micro\\Documents\\Projects\\APICreditTrial3\\.pytest_tmp\\test_locked_evaluation_uses_sa0\\accepted.csv", "size_bytes": 231, "sha256": "be8e2614bf84cc41d168a0001ac6e73be2fdb5170b027c5a5a6f331edb634e80"}`
- Target summary: `{}`

- Model type: `accepted`
- Target mode: `resolved_default`
- Product mode: `post_pricing_investment`
- Calibration method: `isotonic`
- Selected policy: `{"lgd": 1.0, "required_return": null, "approval_rule": "expected_profit > 0"}`
- Source rows: `1`
- Included rows: `None`
- Excluded rows: `None`

## Split Summary

## Features
- Feature count: `1`
- Numeric risk features: `loan_amnt`
- Categorical risk features: ``
- Pricing fields: ``
- Leakage exclusions: ``

## Validation Metrics
- mean_predicted_default: `0.1`

## Calibration Table
| decile | count | mean_predicted_default | observed_default_rate |
| --- | --- | --- | --- |
| 1 | 1 | 0.1 | 1.0 |

## Policy
- Approval count: `1`
- Selection rate: `1.0`
- Expected profit: `296.0`
- Realized profit: `-800.0`

## Sensitivity
expected_profit,expected_return,approval_count,selection_rate,mean_expected_profit,mean_expected_return,lgd,required_return,actual_default_rate,total_realized_profit,mean_realized_profit
296.0,0.296,1,1.0,296.0,0.296,1.0,,1.0,-800.0,-800.0


## Cohort Backtest
cohort_type,cohort_value,rows,observed_default_rate,mean_predicted_default_rate,brier_score,auc,approval_rate,expected_profit,realized_profit
issue_year,2018,1,1.0,0.1,0.81,,1.0,296.0,-800.0
issue_quarter,2018Q1,1,1.0,0.1,0.81,,1.0,296.0,-800.0
issue_month,2018-02,1,1.0,0.1,0.81,,1.0,296.0,-800.0


## Bootstrap Intervals
metric,estimate,lower,upper,bootstrap_samples,random_state
roc_auc,,,,200,42
pr_auc,,,,200,42
brier_score,0.8100000000000003,0.81,0.81,200,42
expected_profit,296.0,296.0,296.0,200,42
expected_return,0.29599999999999993,0.296,0.296,200,42
approval_rate,1.0,1.0,1.0,200,42
actual_default_rate_approved,1.0,1.0,1.0,200,42
total_realized_profit,-800.0,-800.0,-800.0,200,42
mean_realized_profit,-800.0,-800.0,-800.0,200,42


## Proxy Risk Diagnostics



## Known Limitations
- accepted-loan selection bias
- rejected-loan labels unavailable
- resolved-outcome or horizon-label limitations
- simplified profit assumptions
- not production underwriting
- no fair-lending approval