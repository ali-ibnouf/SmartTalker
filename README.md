<div dir="ltr">

# 🗣️ SmartTalker

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/tests-171%20passed-brightgreen.svg)]()

**Digital Human AI Agent Platform — Arabic-First, Open-Source AI Stack**

SmartTalker is an end-to-end platform for building real-time digital human AI agents. It takes speech or text input and produces a talking avatar video response — powered entirely by open-source AI models. The platform is designed with Arabic as the primary language, targeting MENA markets, but supports multilingual use cases out of the box.

### Key Features

- **Full Speech Pipeline** — ASR, LLM reasoning, TTS, and talking-head video generation in a single API call
- **Arabic-First** — Native Arabic support across all pipeline layers (ASR, LLM, TTS)
- **Real-Time Communication** — REST API, WebSocket, and WebRTC interfaces for flexible integration
- **WhatsApp Integration** — Built-in WhatsApp Business API client for conversational AI over messaging
- **Voice Cloning** — Clone voices from 3–10 second reference audio samples
- **Emotion-Aware** — Detects and applies emotion to both speech synthesis and avatar animation
- **Production-Ready** — Redis rate limiting, API key auth, Prometheus metrics, Docker deployment, and structured JSON logging
- **Cost-Efficient** — Runs on a single GPU server at $50–150/month using fully open-source models

> **First Client:** BusTickets Pro — WhatsApp bus booking assistant
> **Cost Target:** $50–150/month operational

---

## 🏗️ Architecture

SmartTalker uses a 6-layer pipeline architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                        SmartTalker Pipeline                     │
│                                                                 │
│  🎤 Audio In                                          🎬 Video Out │
│      │                                                    ▲     │
│      ▼                                                    │     │
│  ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐  ┌────────┐ │
│  │  ASR   │──▶│  LLM   │──▶│  TTS   │──▶│ Video  │──▶│Upscale │ │
│  │Fun-ASR │   │Qwen 2.5│   │CosyVoice│  │EchoMimic│  │RealESR │ │
│  │  Nano  │   │  14B   │   │  3.0   │   │  V2    │  │  GAN   │ │
│  └────────┘   └────────┘   └────────┘   └────────┘  └────────┘ │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │          Orchestrator: FastAPI + WebSocket + Redis        │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

| Layer | Tool | Purpose |
|-------|------|---------|
| 1. ASR | [Fun-ASR Nano](https://github.com/modelscope/FunASR) | Speech → Text |
| 2. LLM | [Qwen 2.5 14B](https://ollama.com/library/qwen2.5) via Ollama | Reasoning & Response |
| 3. TTS | [CosyVoice 3.0](https://github.com/FunAudioLLM/CosyVoice) | Text → Speech |
| 4. Video | [EchoMimicV2](https://github.com/antgroup/echomimic_v2) | Audio → Talking Head |
| 5. Upscale | [RealESRGAN](https://github.com/xinntao/Real-ESRGAN) + CodeFormer | Quality Enhancement |
| 6. Orchestrator | FastAPI + WebSocket + Redis | Coordination |

---

## 🚀 Quick Start

### Prerequisites

- **OS:** Ubuntu 22.04 LTS
- **GPU:** NVIDIA RTX 4090 (24GB VRAM) or equivalent
- **NVIDIA Driver:** 545+
- **Docker:** 24.0+
- **Python:** 3.10+

### Option 1: One-Click Setup (Recommended)

```bash
git clone https://github.com/ali-ibnouf/SmartTalker.git
cd SmartTalker
chmod +x setup.sh
sudo ./setup.sh
```

### Option 2: Docker Compose

```bash
# Clone the repo
git clone https://github.com/ali-ibnouf/SmartTalker.git
cd SmartTalker

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Build and run
docker compose up -d

# Pull the LLM model
docker exec smarttalker-ollama ollama pull qwen2.5:14b

# Download AI models
bash scripts/download_models.sh
```

### Option 3: Local Development

```bash
# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure
cp .env.example .env

# Download models
bash scripts/download_models.sh

# Start Ollama (separate terminal)
ollama serve

# Run the app
make dev
```

### Verify Installation

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Test text-to-speech
curl -X POST http://localhost:8000/api/v1/text-to-speech \
  -H "Content-Type: application/json" \
  -d '{"text": "مرحباً بكم في سمارت توكر", "language": "ar"}'
```

---

## 📁 Project Structure

```
SmartTalker/
├── src/
│   ├── config.py           # Pydantic Settings
│   ├── main.py             # FastAPI application
│   ├── pipeline/           # AI processing engines
│   │   ├── orchestrator.py # Pipeline coordinator
│   │   ├── asr.py          # Fun-ASR Nano
│   │   ├── llm.py          # Qwen 2.5 via Ollama
│   │   ├── tts.py          # CosyVoice 3.0
│   │   ├── video.py        # EchoMimicV2
│   │   ├── upscale.py      # RealESRGAN + CodeFormer
│   │   └── emotions.py     # Emotion detection
│   ├── api/                # REST + WebSocket API
│   ├── integrations/       # WhatsApp, WebRTC, Storage
│   └── utils/              # Audio, video, logging
├── tests/                  # Test suite
├── scripts/                # Setup & maintenance scripts
├── avatars/                # Avatar reference images
├── voices/                 # Voice reference audio
├── docs/                   # Documentation
├── docker-compose.yml      # 3-service stack
├── Dockerfile              # Multi-stage build
├── Makefile                # Build targets
└── requirements.txt        # Pinned dependencies
```

---

## 🔧 Make Targets

```bash
make setup          # Initial setup (Linux)
make setup-win      # Initial setup (Windows)
make build          # Build Docker images
make run            # Start all services
make dev            # Run locally with hot reload
make test           # Run test suite
make lint           # Run linters
make format         # Format code
make download-models # Download AI models
make clean          # Clean generated files
make help           # Show all targets
```

---

## 📖 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/text-to-speech` | Text → Audio |
| POST | `/api/v1/audio-chat` | Audio → Audio |
| POST | `/api/v1/text-to-video` | Text → Video |
| POST | `/api/v1/voice-clone` | Clone a voice |
| GET | `/api/v1/voices` | List voices |
| GET | `/api/v1/health` | System health |
| WS | `/ws/chat/{avatar_id}` | Real-time chat |

Full API docs: http://localhost:8000/docs

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

</div>

<div dir="rtl">

## 🌍 سمارت توكر — بالعربية

### نظرة عامة

**سمارت توكر** هو منصة وكيل ذكاء اصطناعي رقمي بشري، مصمم خصيصاً للأسواق العربية في منطقة الشرق الأوسط وشمال أفريقيا (MENA).

### المميزات الرئيسية

- 🎤 **التعرف على الكلام** — دعم كامل للغة العربية باستخدام Fun-ASR
- 🧠 **الذكاء الاصطناعي** — محادثة طبيعية بالعربية مع Qwen 2.5
- 🗣️ **تحويل النص إلى كلام** — صوت عربي طبيعي مع CosyVoice
- 🎬 **فيديو ذكي** — أفاتار متحرك واقعي مع EchoMimicV2
- 📱 **واتساب** — تكامل مباشر مع واتساب للأعمال

### العميل الأول: BusTickets Pro

نظام حجز تذاكر الحافلات عبر واتساب — يتحدث العربية بطلاقة ويوفر تجربة حجز سهلة وسريعة.

### البدء السريع

```bash
# استنساخ المشروع
git clone https://github.com/ali-ibnouf/SmartTalker.git
cd SmartTalker

# الإعداد التلقائي
chmod +x setup.sh
sudo ./setup.sh

# تشغيل الخدمات
docker compose up -d
```

### التكلفة التشغيلية

الهدف: **50–150 دولار شهرياً** — باستخدام أدوات مفتوحة المصدر بالكامل.

</div>
