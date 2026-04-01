# Maskki (SmartTalker) API Reference

## Base URL
```
https://ws.maskki.com
```

## Authentication

Standard API requests require `X-API-Key`. Administrative requests require `X-Admin-API-Key`. In **production** mode, these are mandatory.

```
X-API-Key: your-customer-api-key
X-Admin-API-Key: your-admin-api-key
```

The `/health` endpoint is always accessible without authentication. WhatsApp webhook endpoints use Meta's signature verification.

## Interactive Docs
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## Endpoints

### POST `/api/v1/text-to-speech`
Send text, receive AI-generated audio response.

**Request Body:**
```json
{
  "text": "مرحبا، كيف يمكنني مساعدتك؟",
  "avatar_id": "default",
  "emotion": "neutral",
  "language": "ar",
  "voice_id": null
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `text` | string | ✅ | — | Input text (1–2000 chars) |
| `avatar_id` | string | — | `"default"` | Avatar for video rendering |
| `emotion` | string | — | `"neutral"` | Emotion label |
| `language` | string | — | `"ar"` | Code: `ar`, `en`, `fr`, `tr` |
| `voice_id` | string | — | `null` | Voice clone ID |

**Response (200):**
```json
{
  "audio_url": "http://localhost:8000/files/tts_abc123.wav",
  "video_url": null,
  "response_text": "أهلاً وسهلاً! كيف أقدر أساعدك؟",
  "total_latency_ms": 850,
  "breakdown": { "llm_ms": 500, "tts_ms": 350 },
  "request_id": "a1b2c3d4"
}
```

---

### POST `/api/v1/audio-chat`
Send audio, receive AI-generated audio response (ASR → LLM → TTS).

**Request:** Multipart form-data

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `audio` | file | ✅ | Audio file (WAV, MP3, OGG, WEBM, M4A) |
| `avatar_id` | string | — | Avatar ID |
| `emotion` | string | — | Emotion label |
| `language` | string | — | Language code |
| `voice_id` | string | — | Voice clone ID |

**Response:** Same as text-to-speech, plus `asr_ms` in breakdown.

---

### POST `/api/v1/voice-clone`
Upload 3–10 second reference audio to create a cloned voice.

**Request:** Multipart form-data

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `audio` | file | ✅ | Reference audio (3–10 seconds, WAV) |
| `voice_name` | string | ✅ | Name for the voice (1–100 chars) |

**Response (200):**
```json
{
  "voice_id": "voice_a1b2c3d4",
  "name": "Salem",
  "message": "Voice cloned successfully"
}
```

---

### GET `/api/v1/voices`
List all available voices.

**Response (200):**
```json
{
  "voices": [
    { "voice_id": "voice_a1b2c3d4", "name": "Salem", "language": "ar", "description": "" }
  ],
  "count": 1
}
```

---

### POST `/api/v1/webhooks/paddle`
Listener for Paddle billing events. Requires `Paddle-Signature` verification.

**Supported Events:**
- `subscription.created`
- `subscription.updated`
- `subscription.canceled`
- `transaction.completed`

**Response (200):** `{"status": "ok"}`

---

### GET `/api/v1/health`
System health check.

**Response (200):**
```json
{
  "status": "healthy",
  "models_loaded": { "asr": true, "tts": true, "emotion": true, "llm": true, "kb": true },
  "services": { "db": "ok", "redis": "ok", "r2": "ok", "d1": "ok" },
  "uptime_s": 3600.0
}
```

---

## Error Responses

All errors return:
```json
{
  "error": "Error message",
  "detail": "Additional context",
  "request_id": "a1b2c3d4"
}
```

| Status Code | Meaning |
|-------------|---------|
| 400 | Bad request (invalid input) |
| 422 | Validation error |
| 500 | Internal server error |

## Headers
- **X-Request-ID**: Correlation ID (auto-generated or echoed from client)
