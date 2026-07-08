# Cold Start Runbook

Fresh clone to local API:

1. Install dependencies.

```bash
python -m pip install -r requirements.txt
python -m pip install -e .[dev]
```

2. Preprocess the accepted-loan source data.

```bash
python preprocessing.py
```

3. Train the calibrated default-risk bundle and write validation reports.

```bash
python train.py
```

4. Evaluate the locked test split with the saved bundle.

```bash
python evaluate_locked.py
```

5. Run the API locally.

```bash
uvicorn api:app --reload
```

6. Score the committed demo CSV through the batch API.

```bash
python batch.py docs/demo/sample_batch_input.csv docs/demo/sample_batch_output.csv --api-url http://127.0.0.1:8000
```

7. Build and run Docker with mounted artifacts.

```bash
docker build -t credit-default-api .
docker run --rm -p 8000:8000 -v ./artifacts:/app/artifacts credit-default-api
```

Smoke/demo mode writes separate outputs under `reports/smoke/validation/` and `reports/smoke/test/`:

```bash
python train.py --sample 5000
python evaluate_locked.py --sample 5000
```
