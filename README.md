# Credit Default Risk API

This repo is a LendingClub loan risk-grading ML project. It uses historical accepted and funded loans with resolved outcomes to identify additional predictive signal beyond LendingClub’s existing risk classification by producing stronger default-risk estimates.

This repo is not a production underwriting system. It is a default-risk modeling and deployment project that uses loans with resolved outcomes only. It includes leakage-controlled modeling, chronological validation, class calibration, model artifact preservation, FastAPI-backed scoring operations (with batch scoring), and adequate testing. There is optional Docker containerization as well.

In a real business setting, these calibrated risk estimates could support underwriter review.

## Results:

We ran four different models with cross validation and isotonic/sigmoid calibration tuning:

- Histogram gradient boosted trees
- Random forest
- Logistic regression
- Class-weighted logistic regression

**Selected Model:** HGB with isotonic calibration was selected during validation. It is now pinned as the final configuration in `config.yaml`.

Resulting test metrics from local run:

- Rows: `134273`
- Observed default rate: `0.1989`
- Mean predicted default rate: `0.2223`
- ROC-AUC: `0.7092`
- PR-AUC: `0.3579`
- Brier score: `0.1463`
- Log loss: `0.4568`

### Baseline comparison

LendingClub uses grade (coarse) and sub-grade (detailed) risk classification to rank loans historically. We also used a constant $p_{\text{default}} = \frac{\text{training defaults}}{\text{training loans}}$ as a base rate comparison:

| Model | ROC-AUC | PR-AUC | Brier | Log Loss |
| --- | ---: | ---: | ---: | ---: |
| Final calibrated hist gradient boosting | 0.7092 | 0.3579 | 0.1463 | 0.4568 |
| Base rate | 0.5000 | 0.1989 | 0.1595 | 0.4995 |
| Grade historical rate (A-G)| 0.6759 | 0.3044 | 0.1492 | 0.4660 |
| Sub-grade historical rate (A1-G5)| 0.6853 | 0.3248 | 0.1487 | 0.4639 |

**Verdict:** Our selected model learns useful information beyond LendingClub's current grading system.

## Data & Target Definition

We wanted to ensure a fair grading without cutting too many of our samples. We started with 2,260,701 previously accepted loans from June 2007 to December 2018 with 35 consumer attributes.

- The target is binary `default`.
- Outputs are calibrated `p_default`
- Features are limited to application-time or underwriting-time fields.
- Post-origination repayment fields are forbidden model inputs.

After preprocessing, we ended up using 25 attributes and 1,348,099 loans with known outcomes. We purposely did not use non-accepted loans because their realized success rate is unknown and we cannot optimize default rates that way.

**Default** (Bad Outcomes):

- `Charged Off`
- `Default`
- `Does not meet the credit policy. Status:Charged Off`

**Non-Default** (Good outcome):

- `Fully Paid`
- `Does not meet the credit policy. Status:Fully Paid`

Dropped from supervised modeling:

- `Current`
- `In Grace Period`
- `Late (16-30 days)`
- `Late (31-120 days)`
- `Issued`
- other blank or unresolved statuses

### Dataset Setup

Raw LendingClub files are not committed. 

Default expected path at repo root:

`./accepted_2007_to_2018Q4.csv`

The active default-risk pipeline requires the accepted-loan file only. Obtain the dataset from a public LendingClub archive or mirror, then place the accepted-loan file at the exact filename above, or pass a custom path with `--csv`.

**KaggleHub download:**

```
import kagglehub

# Download latest version
path = kagglehub.dataset_download("wordsforthewise/lending-club")

print("Path to dataset files:", path)
```

Link: https://www.kaggle.com/datasets/wordsforthewise/lending-club

## Run Using Venv:

Requires `python` on local path and the accepted-loan CSV at the repo root.

```bash
python -m venv .venv
# PowerShell:
.\.venv\Scripts\Activate.ps1
# Git Bash:
. .venv/Scripts/activate
python -m pip install -r requirements.txt
python -m pip install -e .[dev]

python -m src.preprocessing
python -m src.train

# expects accepted-loan CSV at `./accepted_2007_to_2018Q4.csv`.
```

This writes:

- `artifacts/accepted_model.joblib`
- `artifacts/frontend_model.joblib`
- validation reports under `reports/validation/`

### Configuration:

Use `config.yaml` to enable/disable cross validation, model types, or calibration types.

If `training.selected_model` is omitted, candidates are selected automatically by validation mean absolute calibration gap, Brier score, log loss, CV calibration gap, CV Brier score, ROC-AUC, PR-AUC, then configured model order as the final tie-breaker. Set `training.selected_model` to one enabled model to pin it manually.

To run the locked test evaluation later, once model selection is finished:

```bash
python -m evaluate_locked
```

That writes `reports/test/`. Do not treat locked test as part of the regular iteration loop.

## API + Frontend Viewing

Ensuring your `venv` is still activated, start the API:

```bash
uvicorn api:app --reload
```

Once the API server is started, open a new terminal to perform scoring/health checks

### Frontend

The frontend is a demo of an application that would help with quick underwriting decisions. Fill in five loan application attributes, it gets graded, then returns a default risk score. The endpoint uses a separate saved bundle trained on the top five application-time features from the main validation feature-importance pass. 

`/frontend/index.html` is a static-file server. To run the frontend, start the API then open http://127.0.0.1:8000/ in your browser:

```
http://127.0.0.1:8000/
```


### API Health Checks + Samples

```bash
#Check liveness and readiness
curl http://localhost:8000/health
curl http://localhost:8000/ready


# `/health` only checks that the service is up. `/ready` checks that both saved bundles exist and contain the required metadata. It will return `503` before training has produced artifacts, or when Docker is running without mounted artifacts.


# Score a sample loan
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "loan_amnt": 10000,
    "int_rate": 12.5,
    "annual_inc": 78000,
    "dti": 15.2,
    "fico_range_low": 690,
    "fico_range_high": 694,
    "delinq_2yrs": 0,
    "inq_last_6mths": 1,
    "open_acc": 10,
    "pub_rec": 0,
    "revol_bal": 12000,
    "revol_util": 44.1,
    "total_acc": 24,
    "mort_acc": 1,
    "acc_open_past_24mths": 3,
    "pub_rec_bankruptcies": 0,
    "grade": "C",
    "sub_grade": "C2",
    "emp_length": "10+ years",
    "home_ownership": "MORTGAGE",
    "verification_status": "Verified",
    "purpose": "debt_consolidation",
    "addr_state": "CA",
    "application_type": "Individual",
    "initial_list_status": "w"
  }'
```

Batch scoring sample:

```bash
python -m batch docs/demo/sample_batch_input.csv docs/demo/sample_batch_output.csv --api-url http://127.0.0.1:8000
```

### API Endpoints

- `GET /health`
- `GET /ready`
- `GET /model-card`
- `GET /frontend-config`
- `POST /score`
- `POST /score-frontend`
- `POST /score-batch`

## Docker

This repo has optional Docker support if you prefer that over a venv. To build & run:

```bash
docker build -t credit-default-api .

docker run --rm -p 8000:8000 -v ./artifacts:/app/artifacts credit-default-api
```

The Docker build excludes raw CSVs, `artifacts/`, and `reports/` by default through `.dockerignore`. That means container `/ready` will fail unless you either:

1. train on the host first and mount `./artifacts`, or
2. mount an existing `artifacts/` directory produced elsewhere.

## Methodology

- `reports/validation/` is the main path for validation-stage review.
- `reports/test/` should only be committed after a deliberate locked test run with `python -m evaluate_locked`.
- `reports/smoke/` is for smoke/sample runs.

### Split Details

**Time Based Split:** four way chronological: 60-15-15-10

| Split | Rows | Default Rate | Date Min | Date Max |
| --- | --- | --- | --- | --- |
| train | 829355 | 0.1846 | 2007-06-01T00:00:00 | 2015-12-01T00:00:00 |
| calibration | 196607 | 0.2265 | 2016-01-01T00:00:00 | 2016-07-01T00:00:00 |
| validation | 187864 | 0.2399 | 2016-08-01T00:00:00 | 2017-06-01T00:00:00 |
| test | 134273 | 0.1989 | 2017-07-01T00:00:00 | 2018-12-01T00:00:00 |

### Demo Files


Look in `docs/demo/` for demo API tests.

Committed demo files:

- `docs/demo/sample_batch_input.csv`
- `docs/demo/sample_batch_output.csv`
- `docs/demo/README.md`

### Tests

Run:

```bash
python -m pytest
```

The suite covers target construction, leakage prevention, split discipline, calibration outputs, artifact round-tripping, batch scoring, API readiness, and repo-level scope guardrails.

## Limitations

- Test set may not be representative of current 2026 lending practices
- Rejected applications are excluded because repayment outcomes are not observed.
- No production deployment controls.
- Fair-lending validation, monitoring, and operational controls are not in scope.
- No policy optimization is included, just default predictions.
