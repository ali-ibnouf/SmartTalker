# SmartTalker Architecture

## Overview
SmartTalker is a real-time conversational AI platform that processes speech/text input through a multi-layer pipeline to produce intelligent voice responses with lip-sync visemes. Built for MENA markets with Arabic-first support. Runs entirely on CPU — no GPU required for runtime.

```
Audio In → ASR → Emotion → LLM → TTS → Visemes → Audio + Lip Params Out
Text In  →        Emotion → LLM → TTS → Visemes → Audio + Lip Params Out
```

## Core Pipeline

| Layer | Engine | Model | Device | Purpose |
|-------|--------|-------|--------|---------|
| 1. ASR | `ASREngine` | FunASR SenseVoice | CPU | Arabic/English speech → text |
| 2. LLM | `LLMEngine` | Qwen via DashScope API | Cloud | Text → intelligent response |
| 3. TTS | `TTSEngine` | CosyVoice (SFT + zero-shot) | CPU | Text → natural speech |
| 4. Emotion | `EmotionEngine` | DistilRoBERTa / keyword | CPU | Text → emotion detection |
| 5. Visemes | `VisemeExtractor` | Character mapping | CPU | Text → lip animation hints |

## Supporting Engines

| Engine | Purpose |
|--------|---------|
| `KnowledgeBaseEngine` | RAG with ChromaDB + DashScope embeddings |
| `TrainingEngine` | Q&A pair management, escalation, go-live tracking |
| `PersonaEngine` | Avatar persona and behavior configuration |
| `BillingEngine` | Per-second usage billing, subscription plans |
| `KillSwitch` | Emergency activation/deactivation |
| `NodeManager` | GPU render node registration and streaming relay |

## System Architecture

```
┌─────────────────────────────────────────────────────┐
│                    FastAPI Server                     │
│  ┌──────────────────────────────────────────────┐    │
│  │ Middleware: RequestID → Auth → Logging → CORS │    │
│  └──────────────────────────────────────────────┘    │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────┐    │
│  │ REST Routes  │  │  WebSocket   │  │ WhatsApp │    │
│  │ + Dashboard  │  │ + Operator   │  │          │    │
│  └──────┬──────┘  └──────┬───────┘  └────┬─────┘    │
│         │                │               │           │
│  ┌──────┴────────────────┴───────────────┴─────┐    │
│  │          SmartTalkerPipeline                  │    │
│  │  ASR → Emotion → LLM → TTS → Visemes         │    │
│  │  + KB/RAG + Training + Persona + Billing      │    │
│  └──────────────────────────────────────────────┘    │
│  ┌──────────────────────────────────────────────┐    │
│  │ Storage Manager │ Config │ Logger │ Metrics   │    │
│  └──────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
         │              │               │
    ┌────┴────┐   ┌─────┴─────┐   ┌────┴──────┐
    │DashScope│   │   Redis   │   │PostgreSQL │
    │  (LLM)  │   │  (Cache)  │   │   (DB)    │
    └─────────┘   └───────────┘   └───────────┘
```

## Directory Structure

```
SmartTalker/
├── src/
│   ├── config.py              # Pydantic Settings (env-driven)
│   ├── main.py                # FastAPI app + lifespan
│   ├── api/
│   │   ├── routes.py          # REST endpoints
│   │   ├── schemas.py         # Pydantic models
│   │   ├── middleware.py      # Request ID + Auth + Logging
│   │   ├── websocket.py       # WebSocket chat handler
│   │   ├── operator_ws.py     # Operator dashboard WS
│   │   └── dashboard_routes.py # Dashboard API routes
│   ├── pipeline/
│   │   ├── orchestrator.py    # Pipeline coordinator + streaming
│   │   ├── asr.py             # FunASR SenseVoice (CPU)
│   │   ├── llm.py             # Qwen via DashScope API
│   │   ├── tts.py             # CosyVoice (CPU)
│   │   ├── emotions.py        # Emotion detection
│   │   ├── knowledge_base.py  # RAG with ChromaDB
│   │   ├── training.py        # Q&A training + escalation
│   │   ├── persona.py         # Avatar persona management
│   │   ├── billing.py         # Per-second billing engine
│   │   ├── visemes.py         # Lip animation viseme extraction
│   │   ├── kill_switch.py     # Emergency kill switch
│   │   └── node_manager.py    # GPU render node management
│   ├── integrations/
│   │   ├── whatsapp.py        # WhatsApp Business API
│   │   ├── webrtc.py          # WebRTC signaling & peer connections
│   │   └── storage.py         # File lifecycle manager
│   ├── db/                    # Database models + async sessions
│   └── utils/
│       ├── audio.py           # Audio utilities
│       ├── video.py           # Video utilities
│       ├── ffmpeg.py          # FFmpeg wrapper
│       ├── exceptions.py      # Custom exception hierarchy
│       ├── logger.py          # Structured JSON logging
│       └── metrics.py         # Prometheus metrics definitions
├── tests/                     # pytest test suite
├── scripts/                   # Setup + model download scripts
├── avatars/                   # Reference images per avatar
├── voices/                    # Voice clone reference audio
├── models/                    # AI model weights (gitignored)
└── outputs/                   # Generated files (gitignored)
```

## Key Design Decisions

1. **Arabic-First**: All system prompts default to Arabic. Emotion keywords include Arabic vocabulary.
2. **CPU-Only Runtime**: ASR and TTS run on CPU. LLM is cloud-based via DashScope API.
3. **Async Pipeline**: LLM uses `httpx.AsyncClient`, streaming TTS yields chunks for real-time delivery.
4. **Structured Logging**: JSON format with correlation IDs via `contextvars` for async safety.
5. **Error Hierarchy**: `SmartTalkerError` base with per-layer subclasses (never bare `except`).
6. **Config via Environment**: Pydantic `BaseSettings` loads from `.env` with validators.
7. **Multi-Tenant**: PostgreSQL-backed with per-tenant billing, subscription plans, and node management.

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 4GB | 8GB |
| CPU | 2-core | 4-core |
| Storage | 20GB SSD | 50GB SSD |
| OS | Ubuntu 22.04 | Ubuntu 22.04 |
| GPU | Not required | Not required |
