FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ACCEPTED_MODEL_BUNDLE=/app/artifacts/accepted_model.joblib \
    FRONTEND_MODEL_BUNDLE=/app/artifacts/frontend_model.joblib

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && groupadd --system app \
    && useradd --system --gid app --home-dir /app app \
    && mkdir -p /app/artifacts /app/.matplotlib \
    && chown app:app /app/artifacts /app/.matplotlib

COPY --chown=app:app config.yaml api.py ./
COPY --chown=app:app frontend ./frontend
COPY --chown=app:app src ./src

USER app
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=2)"]
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

FROM runtime AS demo
COPY --chown=app:app scripts/generate_demo_artifacts.py ./scripts/generate_demo_artifacts.py
RUN python -m scripts.generate_demo_artifacts /app/artifacts

FROM runtime AS production
