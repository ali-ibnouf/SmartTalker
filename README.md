<div dir="ltr">

# 🗣️ Maskki (SmartTalker) — Production AI Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/Tests-1020+-green.svg)](https://github.com/ali-ibnouf/SmartTalker/actions)

**Maskki AI Platform — Production-Ready Arabic-First Conversational AI**

SmartTalker is the core AI engine that powers **Maskki** — a digital employee platform with real-time 3D VRM avatars. It orchestrates a multi-stage pipeline: V3 streaming ASR, high-latency LLM reasoning, emotional TTS, and Cloud/Edge GPU lipsync rendering.

### 🌟 Key Features

- **Production-Ready** — 100% feature complete and optimized for production.
- **Multilingual** — Native support for Arabic, English, French, and Turkish.
- **Cloud-Native Pipeline** — DashScope (ASR/LLM/TTS) + Hybrid GPU Rendering.
- **Multichannel** — Unified routing for Web Widget, WhatsApp Cloud API, and Telegram.
- **Workflow Engine** — 8 step types (Ask, Decision, Tool, Escalation) for structured automation.
- **Auto-Learning** — Confidence-based Q&A extraction and Visitor Memory from transcripts.
- **Security Hardened** — SSRF Protection, Guardrails (PII/Content), and Rate Limiting.
- **Self-Healing** — AI Optimization Agent monitors 40+ rules and applies auto-fixes.
- **Operator Takeover** — Support for live human intervention with cloned voice synthesis.
- **1020+ Tests** — Comprehensive test suite with 100% success rate.

---

## 🏗️ Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   Visitors   │     │ Customer Dashboard│     │ Admin Dashboard  │
│  (VRM Widget)│     │ (OpenNext/Workers)│     │ (OpenNext/Workers)│
└──────┬───────┘     └────────┬─────────┘     └────────┬─────────┘
       │ wss://               │ /api/proxy             │ /api/proxy
       │              ┌───────▼─────────┐              │
       │              │  Workers API    │              │
       │              │ (CF Workers)    │──────────────┘
       │              └───────┬─────────┘
       │                      │ /api/v1/* proxy
       │              ┌───────▼─────────┐     ┌──────────────────┐
       └──────────────▶ Central Server  │────▶│  RunPod Worker   │
                      │  (FastAPI)      │     │  (Serverless GPU)│
                      └────────┬────────┘     └──────────────────┘
```

| Component | Tool | Purpose |
|-----------|------|---------|
| **Pipeline** | DashScope (Qwen3) | ASR, LLM (max), and TTS (vc) |
| **GPU Render**| RunPod Serverless | Real-time MuseTalk/LivePortrait Lipsync |
| **Storage**   | Cloudflare R2 | S3-compatible storage for VRM & Media |
| **Identity**  | SQLite (D1) | User & Subscription management |
| **Cache**     | Redis 7 | Rate limiting & WebSocket sessions |
| **DB**        | PostgreSQL 16 | App data, Knowledge Base, and Logs |

---

## 🚀 Deployment

The platform is designed for **Docker Compose** on a Ubuntu 22.04 VPS.

### 1. Prerequisites
- **Hetzner CX31** (4 vCPU, 8 GB RAM)
- **Docker** 24.0+
- **Let's Encrypt** SSL certificates for `ws.maskki.com`

### 2. Configure
```bash
cp .env.production.example .env.production
# Fill in real keys for DashScope, RunPod, R2, and Paddle
```

### 3. Launch
```bash
docker compose -f docker-compose.prod.yml up -d
```

---

## 💰 Production Metrics

| Metric | Value |
|--------|-------|
| **Central VPS** | ~$12 / month (Hetzner) |
| **Edge GPU** | $0 (Local/Serverless Hybrid) |
| **API Proxy** | ~$0 (Cloudflare Free/Paid) |
| **Runtime Cost**| ~$0.002 / second |
| **Scalability** | 20+ concurrent voice sessions |
| **Tests** | 1020+ Passing |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

</div>

<div dir="rtl">

## 🌍 سمارت توكر — بالعربية

### نظرة عامة

**سمارت توكر** هو المحرك الأساسي لمنصة Maskki — الموظف الرقمي الذكي مع صور أفاتار واقعية. تم تحويل المشروع بالكامل ليعمل بنظام سحابي متكامل يجمع بين قوة **DashScope** للمحادثة و **RunPod** للمعالجة الرسومية.

### المميزات الرئيسية
- 🚀 **جاهز للإنتاج** — اكتمال التطوير بنسبة 100%.
- 🎤 **متعدد اللغات** — دعم كامل للعربية، الإنجليزية، الفرنسية، والتركية.
- 🖼️ **افاتار VRM** — ودجت ثلاثية الأبعاد تفاعلية مدمجة.
- 📱 **قنوات متعددة** — واتساب، تيليجرام، وودجت الموقع.
- 🛡️ **حماية متقدمة** — أنظمة حماية SSRF و Guardrails وفلترة المحتوى.
- 🧠 **تعلم ذاتي** — استخراج المعرفة والذاكرة تلقائياً من المحادثات.

</div>
