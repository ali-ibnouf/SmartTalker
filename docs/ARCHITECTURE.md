# SmartTalker Architecture

## Overview
SmartTalker is the core engine for **Maskki**, a real-time conversational AI platform that processes speech/text input through a multi-layer pipeline to produce intelligent voice responses with 3D avatar lip-sync. Built for global markets with native Arabic, English, French, and Turkish support.

```
Audio In → ASR → Emotion → LLM → TTS → Visemes → Audio + Lip Params Out
Text In  →        Emotion → LLM → TTS → Visemes → Audio + Lip Params Out
```

## Core Pipeline

| Layer | Engine | Model | Device | Purpose |
|-------|--------|-------|--------|---------|
| 1. ASR | `ASREngine` | FunASR SenseVoice | CPU | Multilingual speech → text (V3 Streaming) |
| 2. LLM | `LLMEngine` | Qwen via DashScope API | Cloud | Text → intelligent response |
| 3. TTS | `TTSEngine` | CosyVoice (SFT + zero-shot) | CPU | Text → natural speech |
| 4. Emotion | `EmotionEngine` | DistilRoBERTa / keyword | CPU | Text → emotion detection |
| 5. Visemes | `VisemeExtractor` | Character mapping | CPU | Text → lip animation hints |
| 6. Render | `NodeManager` | MuseTalk / LivePortrait | GPU | Visemes + Audio → Video Sync |

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
┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   Visitors   │     │ Customer Dashboard│     │ Admin Dashboard  │
│  (VRM Widget)│     │ (OpenNext/Pages)  │     │ (OpenNext/Pages)  │
└──────┬───────┘     └────────┬─────────┘     └────────┬─────────┘
       │ wss://               │ /api/proxy             │ /api/proxy
       │              ┌───────▼─────────┐              │
       │              │  Workers API    │              │
       │              │ (CF Workers)    │──────────────┘
       │              └───────┬─────────┘
       │                      │ /api/v1/* proxy
       │              ┌───────▼─────────┐     ┌──────────────────┐
       └──────────────▶ Central Server  │────▶│  GPU Render Node │
                      │ (Hetzner VPS)   │     │ (Edge/Serverless)│
                      └────────┬────────┘     └──────────────────┘
         ┌─────────────────────┼─────────────────────┐
    ┌────┴────┐          ┌─────┴─────┐          ┌────┴──────┐
    │DashScope│          │ Cloudflare│          │ Cloudflare│
    │ (AI API)│          │ (D1 / KV) │          │ (R2 / S3) │
    └─────────┘          └───────────┘          └───────────┘
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

1. **Global Support**: Native support for Arabic, English, French, and Turkish.
2. **Hybrid Rendering**: Core AI (ASR/TTS) runs on CPU for cost efficiency, while complex lip-sync rendering uses dedicated GPU Nodes (RunPod/Edge).
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
| GPU | Required for RenderNode | NVIDIA T4 / A10G |
