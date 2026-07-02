# Subagent Audit Gate

Read-only audit was run before API/frontend work.

Roles:

- Data Leakage Reviewer
- Modeling/Statistics Reviewer
- Profit Math Reviewer
- Rejected-Loan / Selection-Bias Reviewer
- Architecture/AI-Slop Reviewer
- Testing/Reproducibility Reviewer

Changes made from audit findings:

- Rejected-style scoring states that risk is trained on resolved accepted loans mapped to rejected-style fields.
- `policy_code` is mapped for documentation/schema only and is not required as a model feature.
- Training reports test row counts only; non-test label counts are reported separately.
- Scoring/evaluation fail closed if a bundle has no calibrator.
- API/batch scoring uses the locked bundle LGD by default.
- Profit inputs must be present and positive before expected-profit decisions are made.
- Model training sets `random_state=42`.
- `--sample` uses `nrows` so smoke runs do not read the full CSV.
- `--sample` writes smoke artifacts/reports instead of production artifact/report names.
- Locked evaluation uses saved test IDs from the trained bundle and checks the source CSV fingerprint.
- Scoring returns LGD and approval policy metadata with profit outputs.
- Batch scoring validates profit inputs per row.
- Docker excludes raw CSVs, reports, caches, and ignored artifacts from the image context.
