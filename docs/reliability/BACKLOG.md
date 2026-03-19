# Reliability Backlog

> P3 issues and deferred work items to address in future sessions.

---

## P3 — Low Priority

### DashScope ASR/TTS reconnection logic
- **Location:** `src/pipeline/asr.py`, `src/pipeline/tts.py`
- **Issue:** Single WebSocket connection per session. If connection drops mid-stream, entire pipeline fails with no fallback or retry.
- **Recommended fix:** Add circuit breaker pattern with exponential backoff. Consider fallback to a simpler response mode.

### Customer dashboard hardcoded avatarId
- **Location:** `customer-dashboard/src/app/analytics/page.tsx:69`
- **Issue:** `avatarId = "default"` is hardcoded instead of loaded from URL params or localStorage.
- **Recommended fix:** Load from route params or user context.
