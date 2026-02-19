#!/usr/bin/env bash
# =============================================================================
# SmartTalker — One-Click Setup Script
# Target: Fresh Ubuntu 22.04 + NVIDIA GPU
# Usage: chmod +x setup.sh && sudo ./setup.sh
# =============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log()    { echo -e "${GREEN}[✔]${NC} $1"; }
warn()   { echo -e "${YELLOW}[!]${NC} $1"; }
error()  { echo -e "${RED}[✘]${NC} $1"; exit 1; }
info()   { echo -e "${CYAN}[i]${NC} $1"; }

echo ""
echo "============================================="
echo "  SmartTalker — Automated Setup"
echo "  Digital Human AI Agent Platform"
echo "============================================="
echo ""

# ── Check root ───────────────────────────────────────────────────────────────
if [[ $EUID -ne 0 ]]; then
    error "This script must be run as root (sudo ./setup.sh)"
fi

INSTALL_USER="${SUDO_USER:-$USER}"
INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

info "Installing as user: $INSTALL_USER"
info "Install directory: $INSTALL_DIR"

# ── 1. System Dependencies ──────────────────────────────────────────────────
log "Installing system dependencies..."
apt-get update -qq
apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    ffmpeg \
    libsndfile1 \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    jq \
    unzip
log "System dependencies installed"

# ── 2. NVIDIA Driver Check ──────────────────────────────────────────────────
info "Checking NVIDIA GPU..."
if ! command -v nvidia-smi &> /dev/null; then
    error "nvidia-smi not found. Install NVIDIA drivers first:
    sudo apt install nvidia-driver-545
    Then reboot and re-run this script."
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
log "GPU detected: $GPU_NAME ($GPU_MEM)"

# ── 3. Docker ────────────────────────────────────────────────────────────────
if ! command -v docker &> /dev/null; then
    info "Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    usermod -aG docker "$INSTALL_USER"
    systemctl enable docker
    systemctl start docker
    log "Docker installed"
else
    log "Docker already installed: $(docker --version)"
fi

# ── 4. NVIDIA Container Toolkit ──────────────────────────────────────────────
if ! dpkg -l | grep -q nvidia-container-toolkit; then
    info "Installing NVIDIA Container Toolkit..."
    distribution=$(. /etc/os-release; echo "${ID}${VERSION_ID}")
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
        gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L "https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list" | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
    apt-get update -qq
    apt-get install -y nvidia-container-toolkit
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
    log "NVIDIA Container Toolkit installed"
else
    log "NVIDIA Container Toolkit already installed"
fi

# ── 5. Python Virtual Environment ────────────────────────────────────────────
info "Setting up Python virtual environment..."
cd "$INSTALL_DIR"

sudo -u "$INSTALL_USER" python3.10 -m venv venv
sudo -u "$INSTALL_USER" bash -c "
    source venv/bin/activate
    pip install --upgrade pip setuptools wheel
    pip install -r requirements.txt
"
log "Python environment ready"

# ── 6. Create Directories ───────────────────────────────────────────────────
info "Creating required directories..."
sudo -u "$INSTALL_USER" mkdir -p models avatars voices outputs logs files
log "Directories created"

# ── 7. Environment File ─────────────────────────────────────────────────────
if [[ ! -f .env ]]; then
    sudo -u "$INSTALL_USER" cp .env.example .env
    log "Created .env from .env.example — edit with your settings"
else
    warn ".env already exists — skipping"
fi

# ── 8. Ollama ────────────────────────────────────────────────────────────────
if ! command -v ollama &> /dev/null; then
    info "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    log "Ollama installed"
else
    log "Ollama already installed: $(ollama --version)"
fi

info "Pulling Qwen 2.5 14B model (this may take a while)..."
sudo -u "$INSTALL_USER" ollama pull qwen2.5:14b || warn "Failed to pull model — you can pull it later with: ollama pull qwen2.5:14b"

# ── 9. Summary ───────────────────────────────────────────────────────────────
echo ""
echo "============================================="
echo "  ✅ SmartTalker Setup Complete!"
echo "============================================="
echo ""
echo "  Next steps:"
echo "  1. Edit .env with your configuration"
echo "  2. Download AI models:  bash scripts/download_models.sh"
echo "  3. Start with Docker:   docker compose up -d"
echo "     OR locally:          source venv/bin/activate && make dev"
echo ""
echo "  API will be available at: http://localhost:8000"
echo "  API docs at:              http://localhost:8000/docs"
echo ""
echo "  GPU: $GPU_NAME ($GPU_MEM)"
echo "============================================="
