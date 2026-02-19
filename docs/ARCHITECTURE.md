# SmartTalker Architecture

## Overview
SmartTalker is a real-time talking avatar platform that converts speech input to video output through a 6-layer AI pipeline. Built for MENA markets with Arabic-first support.

```
Audio In → ASR → Emotion → LLM → TTS → Video → Upscale → Video Out
Text In  →        Emotion → LLM → TTS → Video → Upscale → Video Out
```

## 6-Layer Pipeline

| Layer | Engine | Model | Device | Purpose |
|-------|--------|-------|--------|---------|
| 1. ASR | `ASREngine` | Fun-ASR Nano | GPU:0 | Arabic/English speech → text |
| 2. LLM | `LLMEngine` | Qwen 2.5 14B via Ollama | GPU:0 | Text → intelligent response |
| 3. TTS | `TTSEngine` | CosyVoice 3.0 | GPU:0 | Text → natural speech |
| 4. Video | `VideoEngine` | EchoMimicV2 | GPU:0 | Audio + image → talking head |
| 5. Upscale | `UpscaleEngine` | RealESRGAN + CodeFormer | GPU:0 | 512px → 1080p enhancement |
| 6. Emotion | `EmotionEngine` | DistilRoBERTa / keyword | CPU/GPU | Text → emotion detection |

## System Architecture

```
┌─────────────────────────────────────────────────────┐
│                    FastAPI Server                     │
│  ┌──────────────────────────────────────────────┐    │
│  │ Middleware: RequestID → Logging → CORS        │    │
│  └──────────────────────────────────────────────┘    │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────┐    │
│  │ REST Routes  │  │  WebSocket   │  │ WhatsApp │    │
│  └──────┬──────┘  └──────┬───────┘  └────┬─────┘    │
│         │                │               │           │
│  ┌──────┴────────────────┴───────────────┴─────┐    │
│  │          SmartTalkerPipeline                  │    │
│  │  ASR → Emotion → LLM → TTS → Video → Upscale│    │
│  └──────────────────────────────────────────────┘    │
│  ┌──────────────────────────────────────────────┐    │
│  │  Storage Manager  │  Config  │  Logger        │    │
│  └──────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
         │              │               │
    ┌────┴────┐   ┌─────┴─────┐   ┌────┴────┐
    │  Ollama  │   │   Redis   │   │   GPU   │
    │  (LLM)   │   │  (Cache)  │   │ (CUDA)  │
    └──────────┘   └───────────┘   └─────────┘
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
│   │   ├── middleware.py      # Request ID + Logging
│   │   └── websocket.py       # WebSocket handler
│   ├── pipeline/
│   │   ├── orchestrator.py    # Pipeline coordinator
│   │   ├── asr.py             # Fun-ASR Nano
│   │   ├── llm.py             # Qwen 2.5 via Ollama
│   │   ├── tts.py             # CosyVoice 3.0
│   │   ├── video.py           # EchoMimicV2
│   │   ├── upscale.py         # RealESRGAN + CodeFormer
│   │   └── emotions.py        # Emotion detection
│   ├── integrations/
│   │   ├── whatsapp.py        # WhatsApp Business API
│   │   ├── webrtc.py          # WebRTC signaling & peer connections
│   │   └── storage.py         # File lifecycle manager
│   └── utils/
│       ├── audio.py           # ffmpeg audio utilities
│       ├── video.py           # ffmpeg video utilities
│       ├── exceptions.py      # Custom exception hierarchy
│       ├── logger.py          # Structured JSON logging
│       └── metrics.py         # Prometheus metrics definitions
├── tests/                     # pytest test suite
├── scripts/                   # Setup + benchmark scripts
├── avatars/                   # Reference images per avatar
├── voices/                    # Voice clone reference audio
├── models/                    # AI model weights (gitignored)
└── outputs/                   # Generated files (gitignored)
```

## Key Design Decisions

1. **Arabic-First**: All system prompts default to Arabic. Emotion keywords include Arabic vocabulary.
2. **GPU Memory**: Models loaded lazily, explicit `unload()` with `torch.cuda.empty_cache()`.
3. **Async Pipeline**: LLM uses `httpx.AsyncClient`, Video uses `asyncio.create_subprocess_exec`.
4. **Structured Logging**: JSON format with correlation IDs via `contextvars` for async safety.
5. **Error Hierarchy**: `SmartTalkerError` base with per-layer subclasses (never bare `except`).
6. **Config via Environment**: Pydantic `BaseSettings` loads from `.env` with validators.

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | RTX 3090 (24GB) | RTX 4090 (24GB) |
| RAM | 32GB | 64GB |
| CPU | 8-core | 16-core |
| Storage | 100GB SSD | 500GB NVMe |
| OS | Ubuntu 22.04 | Ubuntu 22.04 |
