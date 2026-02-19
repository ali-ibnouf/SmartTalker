# SmartTalker API Reference

## Base URL
```
http://localhost:8000
```

## Authentication

API key authentication via the `X-API-Key` header. In **development** mode (`APP_ENV=development`), authentication is optional. In **production** mode, a valid `API_KEY` must be configured and all requests must include the header.

```
X-API-Key: your-api-key-here
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
| `avatar_id` | string | — | `"default"` | Avatar for video |
| `emotion` | string | — | `"neutral"` | Emotion label |
| `language` | string | — | `"ar"` | Response language |
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

### POST `/api/v1/text-to-video`
Send text, receive AI-generated video with talking avatar (LLM → TTS → Video → Upscale).

Requires `VIDEO_ENABLED=true` and a valid avatar reference image.

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
| `text` | string | Yes | — | Input text (1–2000 chars) |
| `avatar_id` | string | — | `"default"` | Avatar for video |
| `emotion` | string | — | `"neutral"` | Emotion label |
| `language` | string | — | `"ar"` | Response language |
| `voice_id` | string | — | `null` | Voice clone ID |

**Response (200):**
```json
{
  "audio_url": "http://localhost:8000/files/tts_abc123.wav",
  "video_url": "http://localhost:8000/files/video_abc123.mp4",
  "response_text": "أهلاً وسهلاً! كيف أقدر أساعدك؟",
  "total_latency_ms": 12500,
  "breakdown": { "llm_ms": 500, "tts_ms": 350, "video_ms": 10000, "upscale_ms": 1650 },
  "request_id": "a1b2c3d4"
}
```

> **Note:** `video_url` will be `null` if `VIDEO_ENABLED=false` or if no avatar reference image is found.

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

### GET `/api/v1/health`
System health check.

**Response (200):**
```json
{
  "status": "healthy",
  "gpu_available": true,
  "gpu_memory_used_mb": 8192.0,
  "models_loaded": { "asr": true, "tts": true, "emotion": true, "video": false, "upscale": false },
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
