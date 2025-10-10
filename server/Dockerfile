FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy app code and artifacts
COPY server /app/server
COPY artifacts /app/artifacts

ENV MODEL_PATH=/app/artifacts/csk_prematch_random_forest.pkl

# Hugging Face Spaces default port is 7860 (but PORT is provided at runtime)
EXPOSE 7860
CMD ["sh", "-c", "uvicorn server.main:app --host 0.0.0.0 --port ${PORT:-7860}"]


