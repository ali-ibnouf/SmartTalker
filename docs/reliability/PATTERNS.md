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
