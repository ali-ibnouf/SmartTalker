# =============================================================================
# SmartTalker Central — Multi-stage Dockerfile (Phase 1: Cloud APIs)
# DashScope ASR/LLM/TTS (cloud), RunPod GPU (serverless), Cloudflare R2 (storage)
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Builder — install Python dependencies
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ---------------------------------------------------------------------------
# Stage 2: Runtime — slim image with application
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    tesseract-ocr \
    tesseract-ocr-ara \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Create all directories the app needs (including volume mount points)
RUN mkdir -p /app/logs /app/files /app/clips /app/storage \
    /app/outputs /app/data/kb /app/avatars /app/voices

COPY src/ /app/src/
COPY frontend/ /app/frontend/

# Non-root user — create BEFORE chown so volumes inherit correct ownership
RUN groupadd -r smarttalker && useradd -r -g smarttalker -d /app smarttalker \
    && chown -R smarttalker:smarttalker /app

USER smarttalker

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=30s \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
