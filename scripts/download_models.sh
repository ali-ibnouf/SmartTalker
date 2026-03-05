#!/usr/bin/env bash
# =============================================================================
# SmartTalker — Model Download Script
# Downloads all required AI models for the pipeline
# Usage: bash scripts/download_models.sh
# =============================================================================

set -euo pipefail

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[✔]${NC} $1"; }
info() { echo -e "${CYAN}[i]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="${PROJECT_DIR}/models"

echo ""
echo "============================================="
echo "  SmartTalker — Model Downloader"
echo "============================================="
echo ""

# Create model directories
mkdir -p "${MODELS_DIR}/funasr"
mkdir -p "${MODELS_DIR}/cosyvoice"

# ── 1. FunASR SenseVoice ────────────────────────────────────────────────────
info "Downloading FunASR SenseVoice model..."
if [[ -d "${MODELS_DIR}/funasr/FunAudioLLM" ]]; then
    warn "FunASR model already exists — skipping"
else
    pip install modelscope 2>/dev/null || true
    python3 -c "
from modelscope import snapshot_download
snapshot_download('FunAudioLLM/Fun-ASR-Nano-2512', local_dir='${MODELS_DIR}/funasr/FunAudioLLM')
print('FunASR SenseVoice downloaded successfully')
" || warn "FunASR download failed — install modelscope and retry"
    log "FunASR SenseVoice model ready"
fi

# ── 2. FunASR VAD Model ─────────────────────────────────────────────────────
info "Downloading FSMN-VAD model..."
if [[ -d "${MODELS_DIR}/funasr/fsmn-vad" ]]; then
    warn "FSMN-VAD model already exists — skipping"
else
    python3 -c "
from modelscope import snapshot_download
snapshot_download('iic/speech_fsmn_vad_zh-cn-16k-common-pytorch', local_dir='${MODELS_DIR}/funasr/fsmn-vad')
print('FSMN-VAD downloaded successfully')
" || warn "FSMN-VAD download failed"
    log "FSMN-VAD model ready"
fi

# ── 3. CosyVoice ────────────────────────────────────────────────────────────
info "Downloading CosyVoice..."
if [[ -d "${MODELS_DIR}/cosyvoice/CosyVoice" ]]; then
    warn "CosyVoice already exists — skipping"
else
    cd "${MODELS_DIR}/cosyvoice"
    rm -rf /tmp/CosyVoice
    git clone --depth 1 https://github.com/FunAudioLLM/CosyVoice.git /tmp/CosyVoice
    cp -r /tmp/CosyVoice ./CosyVoice
    cd CosyVoice

    # Install CosyVoice dependencies
    pip install -r requirements.txt 2>/dev/null || warn "Some CosyVoice deps may need manual install"

    # Download pre-trained models
    python3 -c "
from modelscope import snapshot_download
snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
print('CosyVoice models downloaded successfully')
" || warn "CosyVoice model download failed"
    log "CosyVoice ready"
fi

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "============================================="
echo "  ✅ Model Download Complete!"
echo "============================================="
echo ""
echo "  Models directory: ${MODELS_DIR}"
echo "  Note: LLM runs via DashScope API — no local model download needed."
echo ""
du -sh "${MODELS_DIR}"/* 2>/dev/null || true
echo ""
echo "============================================="
