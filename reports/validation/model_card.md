# Model Card

## Artifact Status
- Evidence label: `Full LendingClub local data`
- Evidence note: Example local-run evidence from user-supplied raw LendingClub files. Raw data is not committed.
- Evaluation split: `validation`
- Training timestamp: `2026-07-12T22:03:49.460174+00:00`

## Dataset Splits

- Train: `829355`
- Calibration: `196607`
- Validation: `187864`
- Test: `134273`

## Model

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
