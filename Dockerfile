FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY api.py batch.py evaluate_locked.py train.py train_rejected_style.py README.md ./
COPY src ./src
COPY frontend ./frontend

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
