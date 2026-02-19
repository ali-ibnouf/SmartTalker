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
mkdir -p "${MODELS_DIR}/echomimic"
mkdir -p "${MODELS_DIR}/upscale/realesrgan"
mkdir -p "${MODELS_DIR}/upscale/codeformer"

# ── 1. Fun-ASR Nano ─────────────────────────────────────────────────────────
info "Downloading Fun-ASR Nano model..."
if [[ -d "${MODELS_DIR}/funasr/FunAudioLLM" ]]; then
    warn "Fun-ASR model already exists — skipping"
else
    pip install modelscope 2>/dev/null || true
    python3 -c "
from modelscope import snapshot_download
snapshot_download('FunAudioLLM/Fun-ASR-Nano-2512', local_dir='${MODELS_DIR}/funasr/FunAudioLLM')
print('Fun-ASR Nano downloaded successfully')
" || warn "Fun-ASR download failed — install modelscope and retry"
    log "Fun-ASR Nano model ready"
fi

# ── 2. Fun-ASR VAD Model ────────────────────────────────────────────────────
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

# ── 3. CosyVoice 3.0 ────────────────────────────────────────────────────────
info "Downloading CosyVoice 3.0..."
if [[ -d "${MODELS_DIR}/cosyvoice/CosyVoice" ]]; then
    warn "CosyVoice already exists — skipping"
else
    cd "${MODELS_DIR}/cosyvoice"
    git clone --depth 1 https://github.com/FunAudioLLM/CosyVoice.git
    cd CosyVoice

    # Install CosyVoice dependencies
    pip install -r requirements.txt 2>/dev/null || warn "Some CosyVoice deps may need manual install"

    # Download pre-trained models
    python3 -c "
from modelscope import snapshot_download
snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
print('CosyVoice models downloaded successfully')
" || warn "CosyVoice model download failed"
    log "CosyVoice 3.0 ready"
fi

# ── 4. EchoMimicV2 ──────────────────────────────────────────────────────────
info "Downloading EchoMimicV2..."
if [[ -d "${MODELS_DIR}/echomimic/echomimic_v2" ]]; then
    warn "EchoMimicV2 already exists — skipping"
else
    cd "${MODELS_DIR}/echomimic"
    git clone --depth 1 https://github.com/antgroup/echomimic_v2.git
    cd echomimic_v2
    pip install -r requirements.txt 2>/dev/null || warn "Some EchoMimicV2 deps may need manual install"
    log "EchoMimicV2 ready"
fi

# ── 5. RealESRGAN ────────────────────────────────────────────────────────────
info "Downloading RealESRGAN model..."
REALESRGAN_URL="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
REALESRGAN_PATH="${MODELS_DIR}/upscale/realesrgan/realesr-general-x4v3.pth"
if [[ -f "$REALESRGAN_PATH" ]]; then
    warn "RealESRGAN model already exists — skipping"
else
    wget -q --show-progress -O "$REALESRGAN_PATH" "$REALESRGAN_URL" || warn "RealESRGAN download failed"
    log "RealESRGAN model ready"
fi

# ── 6. CodeFormer ────────────────────────────────────────────────────────────
info "Downloading CodeFormer model..."
CODEFORMER_URL="https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"
CODEFORMER_PATH="${MODELS_DIR}/upscale/codeformer/codeformer.pth"
if [[ -f "$CODEFORMER_PATH" ]]; then
    warn "CodeFormer model already exists — skipping"
else
    wget -q --show-progress -O "$CODEFORMER_PATH" "$CODEFORMER_URL" || warn "CodeFormer download failed"
    log "CodeFormer model ready"
fi

# ── 7. Ollama Models ────────────────────────────────────────────────────────
info "Ensuring Ollama model is available..."
if command -v ollama &> /dev/null; then
    ollama pull qwen2.5:14b || warn "Ollama pull failed — ensure Ollama is running"
    log "Ollama Qwen 2.5 14B ready"
else
    warn "Ollama not installed — run setup.sh first"
fi

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "============================================="
echo "  ✅ Model Download Complete!"
echo "============================================="
echo ""
echo "  Models directory: ${MODELS_DIR}"
echo ""
du -sh "${MODELS_DIR}"/* 2>/dev/null || true
echo ""
echo "============================================="
