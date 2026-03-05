# =============================================================================
# SmartTalker Central — Multi-stage Dockerfile
# DashScope LLM (cloud), CosyVoice TTS (local CPU), FunASR ASR (local CPU)
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Builder — install Python dependencies
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System dependencies for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install CosyVoice from source
RUN git clone --depth 1 https://github.com/FunAudioLLM/CosyVoice.git /tmp/cosyvoice \
    && cd /tmp/cosyvoice \
    && pip install --no-cache-dir -e . \
    && rm -rf /tmp/cosyvoice/.git

# ---------------------------------------------------------------------------
# Stage 2: Runtime — slim image with application
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/venv/bin:$PATH"

# Runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create application directory
WORKDIR /app

# Create required directories
RUN mkdir -p /app/models/cosyvoice /app/models/funasr /app/voices /app/outputs \
    /app/logs /app/files /app/clips /app/kb

# Copy application code
COPY src/ /app/src/
COPY frontend/ /app/frontend/

# Pre-download FunASR SenseVoice model
RUN python -c "\
from funasr import AutoModel; \
model = AutoModel(model='iic/SenseVoiceSmall', device='cpu', hub='ms', model_dir='/app/models/funasr'); \
print('FunASR SenseVoice model downloaded')" || echo "FunASR download skipped (will download at runtime)"

# Create non-root user
RUN groupadd -r smarttalker && useradd -r -g smarttalker -d /app smarttalker \
    && chown -R smarttalker:smarttalker /app

USER smarttalker

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=120s \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Run the application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
