# Reliability Issue Log

> Append every resolved issue here. Most recent first.

---

## [2026-03-19] [P1] Fire-and-forget background tasks lose exceptions silently
- **Root cause:** `asyncio.create_task()` called without storing the reference or attaching a done callback. If the task raises, the exception is only logged by the event loop default handler (easily missed).
- **Fix applied:** Added `task.add_done_callback(_bg_task_error_handler)` to both fire-and-forget sites. Added `_bg_task_error_handler()` helper that logs exceptions via structured logger.
- **Files modified:** `src/api/ws_visitor.py`, `src/api/operator_ws.py`
- **Prevention:** All future `create_task()` calls must either store the task reference or attach a done callback. See PATTERNS.md.
- **Verified:** Yes — 1020 tests passed

---

## [2026-03-19] [P1] WhatsApp webhook crashes on malformed payload (IndexError)
- **Root cause:** Direct `[0]` indexing on `entry`, `changes`, and `messages` arrays without bounds checking. Malformed or empty webhook payloads cause unhandled IndexError.
- **Fix applied:** Replaced direct indexing with `.get()` + `isinstance` + `len()` checks. Raises descriptive `ValueError` on invalid payloads.
- **Files modified:** `src/channels/whatsapp.py`
- **Prevention:** Never use direct `[0]` indexing on external API payloads. Always validate array existence and length first. See PATTERNS.md.
- **Verified:** Yes — 1020 tests passed

---

## [2026-03-19] [P1] DashScope WebSocket connect can hang indefinitely
- **Root cause:** `websockets.connect()` has no connection-level timeout. If DashScope is unreachable or slow, the coroutine blocks forever, hanging the entire visitor session.
- **Fix applied:** Wrapped `websockets.connect()` in `asyncio.wait_for(timeout=15.0)` for both ASR and TTS engines.
- **Files modified:** `src/pipeline/asr.py`, `src/pipeline/tts.py`
- **Prevention:** All external WebSocket connections must use `asyncio.wait_for()` with an explicit timeout. See PATTERNS.md.
- **Verified:** Yes — 1020 tests passed

---

## [2026-03-19] [P1] Unguarded .json().get() on WhatsApp media API responses
- **Root cause:** `response.json().get("url", "")` assumes the response is always a dict. If the API returns a non-dict (e.g., error string, array), this raises `AttributeError`.
- **Fix applied:** Added `isinstance(data, dict)` guard before calling `.get()` in both `channels/whatsapp.py` and `integrations/whatsapp.py`.
- **Files modified:** `src/channels/whatsapp.py`, `src/integrations/whatsapp.py`
- **Prevention:** Always guard `.get()` with `isinstance(result, dict)` when consuming external API JSON responses. See PATTERNS.md.
- **Verified:** Yes — 1020 tests passed

---

## [2026-03-19] [P2] Production server starts with empty critical API keys
- **Root cause:** `validate_production_config()` only checked `API_KEY`, `DEBUG`, and WhatsApp secrets. Critical service keys (DASHSCOPE_API_KEY, DATABASE_URL) had empty defaults and no production validation — server starts but all AI features silently fail with 401.
- **Fix applied:** Added production-mode validation for `DASHSCOPE_API_KEY` and `DATABASE_URL`. Server now raises `ValueError` at startup if these are empty in production.
- **Files modified:** `src/config.py`
- **Prevention:** Every new critical env var must be added to `required_in_prod` in the production validator.
- **Verified:** Yes — 1020 tests passed, manual config validation confirmed

---

## [2026-03-19] [P2] Embedding API crashes on empty response (IndexError)
- **Root cause:** `data["data"][0]["embedding"]` accessed without checking if `data["data"]` exists or is non-empty. Empty API response causes IndexError.
- **Fix applied:** Added bounds check with descriptive `KnowledgeBaseError` before indexing.
- **Files modified:** `src/pipeline/knowledge_base.py`
- **Prevention:** Same as WhatsApp indexing — never use direct indexing on external API arrays without bounds checking.
- **Verified:** Yes — 1020 tests passed

---

## [2026-03-19] [P3] WebSocket receive() without timeout
- **Root cause:** Main WebSocket message loops call `websocket.receive()` with no timeout, which can cause connection leaks if clients disconnect uncleanly.
- **Fix applied:** Wrapped `receive()` in `asyncio.wait_for(timeout=60.0)` in both customer and operator WebSocket loops.
- **Files modified:** `src/api/websocket.py`, `src/api/operator_ws.py`
- **Prevention:** All infinite `receive()` loops must have explicit timeouts to avoid zombie connections.
- **Verified:** Yes — 1020 tests passed

---

## [2026-03-19] [P3] TTS HTTP client not properly closed on shutdown
- **Root cause:** The `unload()` method in TTSEngine was synchronous but needed to close an async HTTP client, relying on garbage collection.
- **Fix applied:** Converted `unload()` to an `async def` and explicitly awaited `self._http_client.aclose()`.
- **Files modified:** `src/pipeline/tts.py`, `src/pipeline/orchestrator.py`
- **Prevention:** Any class holding async resources must have an `async` teardown method that explicitly closes them.
- **Verified:** Yes — 1020 tests passed

---

## [2026-03-19] [P3] RunPod poll request missing per-request timeout
- **Root cause:** Individual `client.get(status_url)` calls during polling lacked explicit timeouts, allowing a single hung request to delay the outer timeout detection.
- **Fix applied:** Added `timeout=10.0` to the polling GET requests in `RunPodServerless`.
- **Files modified:** `src/services/runpod_client.py`
- **Prevention:** All HTTP client calls must have explicit timeouts, especially inside polling loops.
- **Verified:** Yes — 1020 tests passed

---

## [2026-03-19] [P3] LOG_LEVEL read directly from environment
- **Root cause:** `setup_logger` read `LOG_LEVEL` straight from `os.environ`, bypassing Pydantic config validation and type coercion.
- **Fix applied:** Updated `logger.py` to retrieve the `log_level` from `get_settings()` with a fallback.
- **Files modified:** `src/utils/logger.py`
- **Prevention:** Utility modules should read configuration via `get_settings()` when possible.
- **Verified:** Yes — 1020 tests passed

---

## [2026-03-19] [P3] TESTING=1 flag not guarded for production
- **Root cause:** Booting with `TESTING=1` skipped all service startup logic. If accidentally set in production, the server would start in a broken state without connections.
- **Fix applied:** Added a guard in `main.py` lifespan that raises a `ValueError` if `TESTING=1` is set while `APP_ENV=production`.
- **Files modified:** `src/main.py`
- **Prevention:** Dangerous flags (like testing mode) must be explicitly forbidden in production code paths.
- **Verified:** Yes — 1020 tests passed

---

## [2026-03-19] [P3] Missing critical variables in production config
- **Root cause:** Production critical keys (e.g., RunPod API Key, R2 secrets) were not checked on startup.
- **Fix applied:** Added these critical variables to the `required_in_prod` dict in `Settings.validate_production_config()`.
- **Files modified:** `src/config.py`
- **Prevention:** Any new API key required for core functionality must be added to production validation checks.
- **Verified:** Yes — 1020 tests passed

---

## [2026-03-19] [P3] KnowledgeBase HTTP client not properly closed on shutdown
- **Root cause:** Found during preventive hardening — identical pattern to the TTS bug. `unload()` was sync but used a workaround (`loop.create_task(aclose())`) that fires-and-forgets the close coroutine.
- **Fix applied:** Converted `unload()` to `async def` and explicitly awaited `self._http_client.aclose()`. Updated `orchestrator.py` to `await self._kb.unload()`.
- **Files modified:** `src/pipeline/knowledge_base.py`, `src/pipeline/orchestrator.py`, `tests/test_knowledge_base.py`
- **Prevention:** See PATTERNS.md "Synchronous Teardown of Async Resources".
- **Verified:** Yes — 1020 tests passed
