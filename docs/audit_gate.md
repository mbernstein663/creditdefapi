# Subagent Audit Gate

Read-only audit was run before API/frontend changes.

Roles:

- Data Leakage Reviewer
- Modeling/Statistics Reviewer
- Profit Math Reviewer
- Rejected-Loan / Selection-Bias Reviewer
- Architecture/AI-Slop Reviewer
- Testing/Reproducibility Reviewer

Changes made from audit findings:

- Full accepted scoring is documented as post-pricing/post-underwriting because it uses LendingClub-generated fields such as `int_rate`, `grade`, `sub_grade`, and `initial_list_status`.
- Limited-field scoring language now says: limited-field risk estimate using accepted-loan outcomes projected onto rejected-application-style inputs.
- Limited-field scoring remains `review` even when scenario profit inputs are supplied.
- API accepted-score optional fields no longer default missing borrower/loan fields to artificial zeros.
- Serving validates bundle feature columns against configured allowlists.
- Unknown loan statuses and missing split dates fail explicitly.
- Profit policy uses `expected_return >= required_return` when a required return is set.
- Locked accepted-loan test evaluation uses the saved test IDs, content-hash source fingerprint, locked LGD, and locked required-return policy.
- Training and locked-test reports include source SHA-256, split summaries, selected policy, calibration metrics, subgroup calibration where available, and compact model-card artifacts.
