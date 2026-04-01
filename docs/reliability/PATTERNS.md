# Reliability Patterns Library

> Class-level bug patterns and prevention rules learned from incidents.

---

## Pattern: Unobserved Background Tasks

- **Description:** `asyncio.create_task()` called without storing the reference or attaching a done callback. Exceptions from the task are silently lost.
- **How to detect:** `grep -rn "asyncio.create_task(" --include="*.py" | grep -v "task ="` (finds create_task calls that don't store the return value)
- **How to prevent:** Always either (a) store the task reference and await it later, or (b) attach `task.add_done_callback(error_handler)` immediately after creation.
- **Example fix:** See ISSUE_LOG 2026-03-19 "Fire-and-forget background tasks"

---

## Pattern: Unchecked Indexing on External API Payloads

- **Description:** Direct `array[0]` or `dict["key"]` on data from external APIs (webhooks, REST responses). If the payload is malformed, missing, or empty, this crashes with IndexError or KeyError.
- **How to detect:** `grep -rn "\[0\]" --include="*.py" src/ | grep -v test` — look for direct indexing on data from external sources
- **How to prevent:** Always use `.get()` for dicts and check `len() > 0` for arrays before indexing. Raise descriptive errors on invalid payloads.
- **Example fix:** See ISSUE_LOG 2026-03-19 "WhatsApp webhook crashes on malformed payload"

---

## Pattern: Missing Connection Timeout on External WebSockets

- **Description:** `websockets.connect()` or similar async connect calls without an overall timeout. If the remote service is unreachable, the coroutine blocks indefinitely.
- **How to detect:** `grep -rn "websockets.connect\|aiohttp.ClientSession\|httpx.AsyncClient" --include="*.py" src/` — check for missing timeout parameters
- **How to prevent:** Wrap all external connection attempts in `asyncio.wait_for(coro, timeout=N)`. Use 10-20s for initial connections.
- **Example fix:** See ISSUE_LOG 2026-03-19 "DashScope WebSocket connect can hang indefinitely"

---

## Pattern: Unguarded .get() on External JSON Responses

- **Description:** Calling `.json().get("key")` on HTTP responses without first verifying the response is a dict. External APIs may return non-dict types (strings, arrays, None) on errors.
- **How to detect:** `grep -rn "\.json()\.get\|\.json()\[" --include="*.py" src/` — find chained access on JSON responses
- **How to prevent:** Always assign `data = response.json()` first, then guard with `if isinstance(data, dict)` before using `.get()`.
- **Example fix:** See ISSUE_LOG 2026-03-19 "Unguarded .json().get() on WhatsApp media API"

---

## Pattern: Silent Startup Degradation

- **Description:** Critical services (DB, Redis, API keys) fail to connect at startup but the server catches the exception and continues with `None` references. Server appears healthy but all requests fail at runtime.
- **How to detect:** Search for `except Exception` in startup/lifespan code that sets services to `None`
- **How to prevent:** In production mode, critical service failures must raise and halt startup. Use config validators for required credentials.
- **Example fix:** See ISSUE_LOG 2026-03-19 "Production server starts with empty critical API keys"

---

## Pattern: Infinite Block on WebSocket Receive

- **Description:** A message loop using `await websocket.receive()` without a timeout. A silent or bad client disconnect can leave the server coroutine blocked indefinitely.
- **How to detect:** `grep -rn "websocket.receive()" --include="*.py" src/`
- **How to prevent:** Always wrap long-running `receive()` calls inside `asyncio.wait_for(timeout=...)` and handle `asyncio.TimeoutError`.
- **Example fix:** See ISSUE_LOG 2026-03-19 "WebSocket receive() without timeout"

---

## Pattern: Synchronous Teardown of Async Resources

- **Description:** Using a synchronous function (e.g. `def unload()`) to clean up async resources (e.g. `aiohttp.ClientSession` or `httpx.AsyncClient`). GC is unreliable for connections.
- **How to detect:** Look for synchronous `def close()` or `def unload()` methods that set clients to None without checking for `aclose` or `close`.
- **How to prevent:** Any class initializing async clients must have an `async def` teardown function.
- **Example fix:** See ISSUE_LOG 2026-03-19 "TTS HTTP client not properly closed on shutdown"

---

## Pattern: SSRF in User-Provided Webhook or Tool URLs

- **Description:** Allowing the AI agent to call arbitrary URLs provided by users or LLM tool calls without validation. This can be used to scan internal networks or access metadata services.
- **How to detect:** `grep -rn "httpx.get\|httpx.post" --include="*.py" src/` — look for requests where the URL comes from a variable.
- **How to prevent:** Use a dedicated `SSRF_PROTECTOR` utility to validate URLs against a denylist of internal IP ranges (127.0.0.1, 10.0.0.0/8, etc.) and restricted domains.
- **Example fix:** See ISSUE_LOG 2026-03-24 "SSRF vulnerability in ToolRegistry execution"

---

## Pattern: Unsafe Indexing on Paddle Callback Payloads

- **Description:** Directly accessing `payload['data']['items'][0]` in Paddle webhooks. Paddle payloads can vary significantly between event types.
- **How to detect:** `grep -rn "\[" src/api/webhooks/paddle.py`
- **How to prevent:** Always use `.get()` and check list length. For Paddle, use their official SDK or a robust Pydantic model for validation.
- **Example fix:** See ISSUE_LOG 2026-03-27 "Paddle webhook crash on empty transaction"

---

## Pattern: Centralized Background Task Exception Handling

- **Description:** Using `asyncio.create_task` scattered across the codebase with varying levels of error handling.
- **How to detect:** Search for any `create_task` call.
- **How to prevent:** Use `src.utils.async_utils.safe_create_task(coro, context_id)` which automatically attaches a logger-aware error handler.
- **Example fix:** See ISSUE_LOG 2026-03-28 "Consolidate background task handling"
