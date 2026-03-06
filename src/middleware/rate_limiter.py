"""Rate limiting middleware for FastAPI endpoints.

Uses Redis for distributed counters with sliding window approach.
Falls back to in-memory counters if Redis is unavailable.

Rate limits are applied per customer_id or IP address.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.utils.logger import setup_logger

logger = setup_logger("middleware.rate_limiter")


# Rate limit configurations
RATE_LIMITS = {
    "api_default":     {"requests": 100, "window": 60},      # 100 req/min
    "voice_clone":     {"requests": 5,   "window": 3600},    # 5/hour
    "tool_test":       {"requests": 20,  "window": 60},      # 20/min
    "knowledge_bulk":  {"requests": 10,  "window": 3600},    # 10/hour
    "ws_session":      {"requests": 50,  "window": 60},      # 50 msg/min per session
    "admin":           {"requests": 200, "window": 60},      # 200 req/min for admin
}

# Path → limit type mapping
PATH_LIMITS: dict[str, str] = {
    "/api/v1/avatars/voice-enroll": "voice_clone",
    "/api/v1/tools/test": "tool_test",
    "/api/v1/knowledge/bulk": "knowledge_bulk",
    "/api/v1/admin/": "admin",
}


class RateLimiter:
    """Rate limiter with Redis backend and in-memory fallback."""

    def __init__(self, redis: Any = None) -> None:
        self._redis = redis
        # In-memory fallback: key → list of timestamps
        self._memory: dict[str, list[float]] = defaultdict(list)

    async def check(self, key: str, limit_type: str = "api_default") -> bool:
        """Check if a request is within rate limits.

        Args:
            key: Identifier (customer_id or IP).
            limit_type: The rate limit category.

        Returns:
            True if the request is allowed, False if rate limited.
        """
        limit = RATE_LIMITS.get(limit_type, RATE_LIMITS["api_default"])
        redis_key = f"ratelimit:{limit_type}:{key}"

        if self._redis is not None:
            return await self._check_redis(redis_key, limit)
        return self._check_memory(redis_key, limit)

    async def _check_redis(self, redis_key: str, limit: dict) -> bool:
        """Check rate limit using Redis INCR + EXPIRE."""
        try:
            current = await self._redis.incr(redis_key)
            if current == 1:
                await self._redis.expire(redis_key, limit["window"])
            return current <= limit["requests"]
        except Exception as exc:
            logger.debug(f"Redis rate limit check failed: {exc}")
            return True  # Fail open if Redis is down

    def _check_memory(self, key: str, limit: dict) -> bool:
        """Check rate limit using in-memory sliding window."""
        now = time.time()
        window_start = now - limit["window"]

        # Remove expired timestamps
        timestamps = self._memory[key]
        self._memory[key] = [t for t in timestamps if t > window_start]

        if len(self._memory[key]) >= limit["requests"]:
            return False

        self._memory[key].append(now)
        return True

    def get_remaining(self, key: str, limit_type: str = "api_default") -> int:
        """Get remaining requests (memory-based only)."""
        limit = RATE_LIMITS.get(limit_type, RATE_LIMITS["api_default"])
        redis_key = f"ratelimit:{limit_type}:{key}"
        now = time.time()
        window_start = now - limit["window"]
        timestamps = [t for t in self._memory.get(redis_key, []) if t > window_start]
        return max(0, limit["requests"] - len(timestamps))


def _get_limit_type(path: str) -> str:
    """Determine the rate limit type from the request path."""
    for prefix, limit_type in PATH_LIMITS.items():
        if path.startswith(prefix):
            return limit_type
    return "api_default"


def _get_client_key(request: Request) -> str:
    """Extract the client identifier from the request."""
    # Prefer customer_id from auth middleware
    customer_id = getattr(request.state, "customer_id", None)
    if customer_id:
        return f"cust:{customer_id}"

    # Fall back to API key
    api_key = request.headers.get("X-API-Key", "")
    if api_key:
        return f"key:{api_key[:16]}"

    # Fall back to IP
    client_ip = request.client.host if request.client else "unknown"
    return f"ip:{client_ip}"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware that applies rate limiting to all requests."""

    def __init__(self, app: Any, rate_limiter: RateLimiter) -> None:
        super().__init__(app)
        self._limiter = rate_limiter

    async def dispatch(self, request: Request, call_next) -> Response:
        # Skip rate limiting for health endpoints and WebSocket upgrades
        path = request.url.path
        if path.startswith("/health") or request.headers.get("upgrade") == "websocket":
            return await call_next(request)

        client_key = _get_client_key(request)
        limit_type = _get_limit_type(path)

        allowed = await self._limiter.check(client_key, limit_type)

        if not allowed:
            limit = RATE_LIMITS.get(limit_type, RATE_LIMITS["api_default"])
            logger.warning(
                "Rate limit exceeded",
                extra={"client": client_key, "limit_type": limit_type, "path": path},
            )
            return Response(
                content='{"error": "Rate limit exceeded", "retry_after": '
                f'{limit["window"]}}}',
                status_code=429,
                headers={
                    "Content-Type": "application/json",
                    "Retry-After": str(limit["window"]),
                },
            )

        return await call_next(request)
