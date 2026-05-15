FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY frontend ./frontend
COPY evals ./evals
COPY scripts ./scripts
COPY docs ./docs

ENV PORT=8000
EXPOSE 8000 8501

# Honors $PORT so the same image runs under docker-compose and Cloud Run.
CMD ["sh", "-c", "python -m src.db.seed && uvicorn src.main:app --host 0.0.0.0 --port ${PORT}"]
