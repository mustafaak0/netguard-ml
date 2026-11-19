FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# CPU-only torch first so the layer cache stays useful
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN grep -v '^torch' requirements.txt | pip install -r /dev/stdin

COPY pyproject.toml README.md ./
COPY netguard ./netguard
RUN pip install --no-deps .

# Train on a small synthetic dataset so the image is self-contained
RUN netguard generate-data --samples 20000 --out /tmp/sample.csv \
    && netguard train --data /tmp/sample.csv --out /artifacts \
    && rm /tmp/sample.csv

ENV NETGUARD_MODEL_DIR=/artifacts
EXPOSE 8000

CMD ["uvicorn", "netguard.api:app", "--host", "0.0.0.0", "--port", "8000"]
