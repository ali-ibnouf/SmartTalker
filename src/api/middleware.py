"""FastAPI middleware -- request IDs, logging, auth, rate limiting, and security headers.

RequestIDMiddleware: adds X-Request-ID to every request.
LoggingMiddleware: logs method, path, status, and latency.
APIKeyAuthMiddleware: rejects requests without valid X-API-Key.
RedisRateLimitMiddleware: sliding-window rate limiter backed by Redis (in-memory fallback).
SecurityHeadersMiddleware: injects security headers on every response.
"""

from __future__ import annotations

import hmac
import time
import uuid
from typing import Any, Optional

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response, JSONResponse

from src.config import Settings
from src.utils.logger import set_correlation_id, setup_logger

logger = setup_logger("api.middleware")

# Paths excluded from auth and rate limiting
_EXCLUDED_PATHS = {"/api/v1/health", "/docs", "/openapi.json", "/redoc", "/metrics"}


def _is_excluded(path: str) -> bool:
    """Check if a request path should bypass auth/rate limiting."""
    if path in _EXCLUDED_PATHS:
        return True
    if path.startswith("/ws/"):
        return True
    return False


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a unique X-Request-ID to every request and response.

    If the client supplies an X-Request-ID header, it is reused;
    otherwise a new UUID4 is generated. The ID is also stored
    in the async-safe correlation_id context variable.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Process the request and inject the request ID.

        Args:
            request: Incoming HTTP request.
            call_next: Next middleware/handler in the chain.

        Returns:
            HTTP response with X-Request-ID header.
        """
        request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex)

        # Store in context for structured logging
        set_correlation_id(request_id)

        # Attach to request state for downstream access
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    """Log every request with method, path, status code, and latency."""

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Log the request and response timing.

        Args:
            request: Incoming HTTP request.
            call_next: Next middleware/handler in the chain.

        Returns:
            HTTP response (unchanged).
        """
        start = time.perf_counter()

        response = await call_next(request)

        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            "Request processed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "latency_ms": latency_ms,
            },
        )
        return response


# ── CORS Configuration ──────────────────────────────────────────────────────

def get_cors_config(origins: str = "*") -> dict:
    """Build CORS middleware configuration dictionary.

    Args:
        origins: Comma-separated allowed origins, or "*" for all.

    Returns:
        Dictionary ready for CORSMiddleware(**config).
    """
    if origins.strip() == "*":
        allow_origins = ["*"]
    else:
        allow_origins = [o.strip() for o in origins.split(",") if o.strip()]

    return {
        "allow_origins": allow_origins,
        "allow_credentials": True,
        "allow_methods": ["*"],
        "allow_headers": ["*"],
        "expose_headers": ["X-Request-ID"],
    }


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """Reject requests without a valid X-API-Key header."""

    _dev_warning_logged: bool = False

    def __init__(self, app: Any, config: Settings):
        super().__init__(app)
        self.config = config

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        if _is_excluded(request.url.path):
            return await call_next(request)

        # No API key configured — dev mode
        if not self.config.api_key:
            if not APIKeyAuthMiddleware._dev_warning_logged:
                logger.warning("No API key configured — auth disabled (dev mode)")
                APIKeyAuthMiddleware._dev_warning_logged = True
            return await call_next(request)

        # Validate key (constant-time comparison to prevent timing attacks)
        client_key = request.headers.get("X-API-Key", "")
        if not hmac.compare_digest(client_key, self.config.api_key):
            return JSONResponse(
                status_code=403,
                content={"detail": "Invalid or missing API Key"},
            )

        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiter (sliding window)."""

    def __init__(self, app: Any, config: Settings):
        super().__init__(app)
        self.limit = config.rate_limit_per_minute
        self._requests: dict[str, list[float]] = {}  # IP -> updated timestamps

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        if _is_excluded(request.url.path):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()

        # Prune old timestamps
        if client_ip in self._requests:
            self._requests[client_ip] = [
                t for t in self._requests[client_ip]
                if now - t < 60.0
            ]

        # Prune stale IPs with no recent requests
        if len(self._requests) > 500:
            stale = [ip for ip, ts in self._requests.items() if not ts]
            for ip in stale:
                del self._requests[ip]

        # Check limit
        current_count = len(self._requests.get(client_ip, []))
        if current_count >= self.limit:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded"},
            )

        # Record request
        if client_ip not in self._requests:
            self._requests[client_ip] = []
        self._requests[client_ip].append(now)

        return await call_next(request)


class RedisRateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window rate limiter backed by Redis sorted sets.

    Falls back to in-memory rate limiting if Redis is unavailable.
    """

    _fallback_warned: bool = False

    def __init__(self, app: Any, config: Settings, redis: Optional[Any] = None):
        super().__init__(app)
        self.limit = config.rate_limit_per_minute
        self.redis = redis
        # In-memory fallback
        self._requests: dict[str, list[float]] = {}

    async def _check_redis(self, client_ip: str, now: float) -> tuple[bool, int]:
        """Check rate limit using Redis sorted sets.

        Returns:
            (allowed, current_count) tuple.
        """
        key = f"rate_limit:{client_ip}"
        window_start = now - 60.0

        pipe = self.redis.pipeline()
        pipe.zremrangebyscore(key, 0, window_start)
        pipe.zadd(key, {f"{now}": now})
        pipe.zcard(key)
        pipe.expire(key, 120)
        results = await pipe.execute()

        current_count = results[2]
        if current_count > self.limit:
            # Remove the just-added entry since we're rejecting
            await self.redis.zrem(key, f"{now}")
            return False, current_count
        return True, current_count

    def _check_memory(self, client_ip: str, now: float) -> tuple[bool, int]:
        """Check rate limit using in-memory sliding window."""
        if client_ip in self._requests:
            self._requests[client_ip] = [
                t for t in self._requests[client_ip]
                if now - t < 60.0
            ]

        # Prune stale IPs with no recent requests (every 500 IPs)
        if len(self._requests) > 500:
            stale = [ip for ip, ts in self._requests.items() if not ts]
            for ip in stale:
                del self._requests[ip]

        current_count = len(self._requests.get(client_ip, []))
        if current_count >= self.limit:
            return False, current_count

        if client_ip not in self._requests:
            self._requests[client_ip] = []
        self._requests[client_ip].append(now)
        return True, current_count + 1

    def _resolve_redis(self, request: Request) -> Optional[Any]:
        """Resolve Redis client — prefer constructor arg, then app.state."""
        if self.redis:
            return self.redis
        try:
            return request.app.state.redis
        except AttributeError:
            return None

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        if _is_excluded(request.url.path):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()

        redis = self._resolve_redis(request)
        allowed = True
        if redis:
            try:
                # Use resolved redis instead of self.redis
                self.redis = redis
                allowed, _ = await self._check_redis(client_ip, now)
            except Exception:
                if not RedisRateLimitMiddleware._fallback_warned:
                    logger.warning(
                        "Redis rate limiting unavailable, falling back to in-memory"
                    )
                    RedisRateLimitMiddleware._fallback_warned = True
                allowed, _ = self._check_memory(client_ip, now)
        else:
            allowed, _ = self._check_memory(client_ip, now)

        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded"},
                headers={"Retry-After": "60"},
            )

        return await call_next(request)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Inject security headers on every response."""

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response
