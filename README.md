<div dir="ltr">

# 🗣️ SmartTalker

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com/)
[![CI](https://github.com/ali-ibnouf/SmartTalker/actions/workflows/tests.yml/badge.svg)](https://github.com/ali-ibnouf/SmartTalker/actions/workflows/tests.yml)

**Maskki AI Platform — Arabic-First Conversational AI Engine**

SmartTalker is the core AI engine that powers Maskki — an AI-powered conversational agent with real-time 3D VRM avatars. SmartTalker handles **everything**: conversation, voice synthesis, emotion detection, lip-sync visemes, and knowledge retrieval — all running on a basic VPS with **no GPU required** for runtime.

### Key Features

- **No GPU Required** — Runs on a $5–10/mo VPS using CPU-only inference
- **Arabic-First** — Native Arabic support across all pipeline layers (ASR, LLM, TTS)
- **Cloud LLM** — Qwen via DashScope API (OpenAI-compatible); optional local Ollama for development
- **Knowledge Base (RAG)** — ChromaDB-backed retrieval-augmented generation with DashScope embeddings
- **Training Engine** — Q&A pair management, escalation handling, and go-live readiness tracking
- **Real-Time Communication** — REST API, WebSocket streaming, and WebRTC interfaces
- **WhatsApp Integration** — Built-in WhatsApp Business API client
- **Voice Cloning** — Clone voices from 3–10 second reference audio samples via CosyVoice zero-shot
- **Emotion-Aware** — Detects user sentiment and adapts response tone accordingly
- **Billing & Multi-Tenant** — Per-second billing, subscription plans, kill switch, node management
- **Production-Ready** — PostgreSQL, Redis caching, API key auth, Prometheus metrics, Grafana dashboards, Docker deployment
- **Cost-Efficient** — $0.001/sec covers conversation + voice + lip-sync

---

## 🏗️ Architecture

### SmartTalker Server (CPU-Only Runtime)

- Runs on customer's basic VPS ($5–10/mo) — **NO GPU needed**
- LLM inference via DashScope cloud API (Ollama optional for local dev)
- ASR and TTS run locally on CPU

```
┌─────────────────────────────────────────────────────────────────┐
│                     SmartTalker Server (CPU)                     │
│                                                                  │
│  🎤 Audio/Text In                              🔊 Audio + Visemes│
│      │                                                  ▲       │
│      ▼                                                  │       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐│
│  │   ASR    │─▶│   LLM    │─▶│   TTS    │─▶│    Visemes +     ││
│  │ FunASR   │  │  Qwen    │  │CosyVoice │  │   Lip Params     ││
│  │SenseVoice│  │DashScope │  │  (CPU)   │  │                  ││
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘│
│       │              │                                          │
│  ┌────┴─────┐ ┌──────┴──────┐  ┌──────────────┐               │
│  │ Emotion  │ │ Knowledge   │  │   Training   │               │
│  │Detection │ │ Base (RAG)  │  │   Engine     │               │
│  └──────────┘ └─────────────┘  └──────────────┘               │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │   Orchestrator: FastAPI + WebSocket + Redis + PostgreSQL  │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

| Layer | Tool | Purpose |
|-------|------|---------|
| 1. ASR | [FunASR SenseVoice](https://github.com/modelscope/FunASR) | Speech → Text (CPU) |
| 2. LLM | [Qwen](https://dashscope.aliyuncs.com) via DashScope API | Reasoning & Response |
| 3. TTS | [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) | Text → Speech (CPU) |
| 4. Emotion | Keyword + DistilRoBERTa | Sentiment Detection |
| 5. KB/RAG | ChromaDB + DashScope Embeddings | Knowledge Retrieval |
| 6. Orchestrator | FastAPI + WebSocket + Redis | Coordination |

### Optional: RunPod Worker (one-time avatar generation)

- Runs on RunPod GPU cloud for avatar clip generation
- Located in `workers/avatar-generation/`
- Cost: ~$0.25 per customer (serverless — $0 when idle)

---

## 🚀 Quick Start

### Prerequisites

- **OS:** Ubuntu 22.04 LTS (or Windows with WSL2)
- **No GPU required** — runs entirely on CPU
- **Docker:** 24.0+
- **Python:** 3.10+

### Option 1: Docker Compose (Recommended)

```bash
git clone https://github.com/ali-ibnouf/SmartTalker.git
cd SmartTalker

# Configure environment
cp .env.example .env
# Edit .env to add your DASHSCOPE_API_KEY

# Build and run
docker compose up -d

# (Optional) For local LLM via Ollama instead of DashScope:
# docker compose --profile local-llm up -d
# docker exec smarttalker-ollama ollama pull qwen2.5:14b
```

### Option 2: Local Development

```bash
# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure
cp .env.example .env

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
│   ├── config.py              # Pydantic Settings (env-driven)
│   ├── main.py                # FastAPI application + lifespan
│   ├── pipeline/              # AI processing engines
│   │   ├── orchestrator.py    # Pipeline coordinator + streaming
│   │   ├── asr.py             # FunASR SenseVoice (CPU)
│   │   ├── llm.py             # Qwen via DashScope API
│   │   ├── tts.py             # CosyVoice (CPU, voice cloning)
│   │   ├── emotions.py        # Emotion detection
│   │   ├── knowledge_base.py  # RAG with ChromaDB
│   │   ├── training.py        # Q&A training + escalation
│   │   ├── persona.py         # Avatar persona management
│   │   ├── billing.py         # Per-second billing engine
│   │   ├── visemes.py         # Lip animation viseme extraction
│   │   ├── kill_switch.py     # Emergency kill switch
│   │   └── node_manager.py    # GPU render node management
│   ├── api/                   # REST + WebSocket API
│   │   ├── routes.py          # REST endpoints
│   │   ├── websocket.py       # WebSocket chat handler
│   │   ├── operator_ws.py     # Operator dashboard WS
│   │   ├── dashboard_routes.py # Dashboard API routes
│   │   ├── schemas.py         # Pydantic models
│   │   └── middleware.py      # Auth, rate limiting
│   ├── integrations/          # WhatsApp, WebRTC, Storage
│   ├── db/                    # Database models + sessions
│   └── utils/                 # Audio, video, ffmpeg, logging
├── workers/
│   └── avatar-generation/     # RunPod GPU worker (optional)
├── frontend/                  # Web client (HTML + JS + CSS)
├── clips/                     # Avatar video clips (per avatar)
├── voices/                    # Voice reference audio
├── avatars/                   # Avatar reference images
├── tests/                     # pytest test suite
├── docs/                      # Architecture, API, Deployment docs
├── docker-compose.yml         # Multi-service stack (CPU only)
├── Dockerfile                 # Multi-stage build
├── Makefile                   # Build targets
└── requirements.txt           # Dependencies
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
make clean          # Clean generated files
make help           # Show all targets
```

---

## 📖 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/text-to-speech` | Text → LLM → TTS pipeline |
| POST | `/api/v1/audio-chat` | Audio → ASR → LLM → TTS |
| POST | `/api/v1/voice-clone` | Clone a voice |
| GET | `/api/v1/voices` | List voices |
| GET | `/api/v1/health` | System health |
| POST | `/api/v1/avatars/{id}/upload/{state}` | Upload avatar clip |
| GET | `/api/v1/avatars` | List avatars |
| GET | `/api/v1/avatars/{id}` | Get avatar clips |
| WS | `/ws/chat` | Real-time chat |
| WS | `/ws/rtc` | WebRTC signaling |

Full API docs: http://localhost:8000/docs

---

## 💰 Cost Model

| Component | Cost | When |
|-----------|------|------|
| Avatar Generation (RunPod) | ~$0.25/customer | One-time |
| SmartTalker Server (VPS) | $5–10/month | Always on |
| Runtime Processing | $0.001/sec | Per interaction |
| RunPod Idle | $0 | Serverless |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

</div>

<div dir="rtl">

## 🌍 سمارت توكر — بالعربية

### نظرة عامة

**سمارت توكر** هو محرك الذكاء الاصطناعي الأساسي الذي يشغّل الموظف الرقمي — وكيل محادثة ذكي مع صور أفاتار واقعية. يعمل على خادم VPS بسيط **بدون GPU** وبتكلفة **5–10 دولار شهرياً**.

### المميزات الرئيسية

- 🤖 **نموذج لغوي سحابي** — Qwen عبر DashScope API (أو Ollama محلياً)
- 🎤 **التعرف على الكلام** — دعم كامل للعربية باستخدام FunASR SenseVoice
- 🧠 **المحادثة الذكية** — محادثة طبيعية بالعربية مع قاعدة معرفة (RAG)
- 🗣️ **تحويل النص إلى كلام** — صوت عربي طبيعي مع CosyVoice
- 🎓 **محرك التدريب** — إدارة الأسئلة والأجوبة والتصعيد
- 📱 **واتساب** — تكامل مباشر مع واتساب للأعمال
- 💰 **تكلفة منخفضة** — 5–10 دولار شهرياً فقط

### البدء السريع

```bash
# استنساخ المشروع
git clone https://github.com/ali-ibnouf/SmartTalker.git
cd SmartTalker

# تشغيل الخدمات
docker compose up -d

# تحميل نموذج الذكاء الاصطناعي
docker exec smarttalker-ollama ollama pull qwen2.5:14b
```

</div>
