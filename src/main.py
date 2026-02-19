"""FastAPI application entry point.

Creates the app with lifespan management, mounts routers,
serves static files, and configures global exception handling.
Run with: uvicorn src.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from src.api.middleware import (
    APIKeyAuthMiddleware,
    LoggingMiddleware,
    RateLimitMiddleware,
    RedisRateLimitMiddleware,
    RequestIDMiddleware,
    SecurityHeadersMiddleware,
    get_cors_config,
)
from src.api.routes import router as api_router
from src.api.schemas import ErrorResponse
from src.api.websocket import WebSocketManager, websocket_chat_endpoint
from src.config import get_settings
from src.pipeline.orchestrator import SmartTalkerPipeline
from src.utils.exceptions import SmartTalkerError
from src.utils.logger import setup_logger
from src.integrations.whatsapp import WhatsAppClient
from src.integrations.webrtc import WebRTCSignalingHandler, webrtc_signaling_endpoint

logger = setup_logger("main")


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown lifecycle.

    On startup: load configuration and initialize pipeline models.
    On shutdown: unload all models and free GPU memory.

    Args:
        application: The FastAPI application instance.

    Yields:
        Control back to the application during its lifetime.
    """
    # ── Startup ──────────────────────────────────────────────────────────
    logger.info("SmartTalker starting up...")

    config = get_settings()
    application.state.config = config

    # Initialize Redis client
    redis_client = None
    try:
        import redis.asyncio as aioredis
        redis_client = aioredis.from_url(
            config.redis_url,
            decode_responses=True,
        )
        await redis_client.ping()
        logger.info("Redis connected", extra={"url": config.redis_url})
    except Exception as exc:
        logger.warning(f"Redis connection failed, using in-memory fallback: {exc}")
        redis_client = None
    application.state.redis = redis_client

    pipeline = SmartTalkerPipeline(config)
    application.state.pipeline = pipeline

    # Initialize WhatsApp client
    whatsapp = WhatsAppClient(config)
    application.state.whatsapp = whatsapp

    # Initialize WebSocket manager (with api_key for auth)
    ws_storage = config.storage_base_dir / "ws_audio"
    ws_manager = WebSocketManager(
        pipeline=pipeline,
        storage_dir=ws_storage,
        api_key=config.api_key,
    )
    application.state.ws_manager = ws_manager

    # Initialize WebRTC handler (if enabled)
    if config.webrtc_enabled:
        webrtc_handler = WebRTCSignalingHandler(pipeline=pipeline, config=config)
        application.state.webrtc_handler = webrtc_handler
    else:
        application.state.webrtc_handler = None

    try:
        pipeline.load_all()
        logger.info("Pipeline models loaded successfully")
    except Exception as exc:
        logger.warning(f"Some models failed to load: {exc}")

    # Ensure static files directory exists
    config.static_files_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "SmartTalker ready",
        extra={"host": config.api_host, "port": config.api_port},
    )

    yield

    # ── Shutdown ─────────────────────────────────────────────────────────
    logger.info("SmartTalker shutting down...")
    await pipeline.unload_all()
    await whatsapp.close()
    if redis_client:
        await redis_client.close()
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance.
    """
    config = get_settings()

    application = FastAPI(
        title="SmartTalker API",
        description=(
            "Digital Human AI Agent Platform — "
            "Real-time talking avatar with Arabic-first support. "
            "Speech-in, video-out pipeline using Chinese open-source AI tools."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # ── Middleware (order matters: outermost first) ───────────────────────

    # Inner middlewares (Auth, RateLimit)
    application.add_middleware(APIKeyAuthMiddleware, config=config)
    application.add_middleware(RedisRateLimitMiddleware, config=config)

    # Outer middlewares
    cors_config = get_cors_config(config.cors_origins)
    application.add_middleware(CORSMiddleware, **cors_config)
    application.add_middleware(SecurityHeadersMiddleware)
    application.add_middleware(LoggingMiddleware)
    application.add_middleware(RequestIDMiddleware)

    # ── Routers ──────────────────────────────────────────────────────────
    application.include_router(api_router)

    # ── WebSocket Endpoint ───────────────────────────────────────────────
    @application.websocket("/ws/chat")
    async def ws_chat(websocket: WebSocket):
        manager = application.state.ws_manager
        await websocket_chat_endpoint(websocket, manager)

    # ── WebRTC Signaling ─────────────────────────────────────────────────
    if config.webrtc_enabled:
        @application.websocket("/ws/rtc")
        async def ws_rtc(websocket: WebSocket):
            handler = application.state.webrtc_handler
            if handler:
                await webrtc_signaling_endpoint(websocket, handler)

    # ── Static Files ─────────────────────────────────────────────────────
    config.static_files_dir.mkdir(parents=True, exist_ok=True)
    application.mount(
        "/files",
        StaticFiles(directory=str(config.static_files_dir)),
        name="files",
    )

    # ── Global Exception Handlers ────────────────────────────────────────

    @application.exception_handler(SmartTalkerError)
    async def smarttalker_error_handler(
        request: Request,
        exc: SmartTalkerError,
    ) -> JSONResponse:
        """Handle all SmartTalkerError subclasses.

        Args:
            request: The incoming request that caused the error.
            exc: The SmartTalkerError exception.

        Returns:
            JSONResponse with ErrorResponse body and 500 status.
        """
        request_id = getattr(request.state, "request_id", None)
        logger.error(
            f"Pipeline error: {exc.message}",
            extra={"detail": exc.detail, "request_id": request_id},
        )
        error_response = ErrorResponse(
            error=exc.message,
            detail=exc.detail,
            request_id=request_id,
        )
        return JSONResponse(
            status_code=500,
            content=error_response.model_dump(),
        )

    @application.exception_handler(Exception)
    async def general_error_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        """Handle unexpected exceptions.

        Args:
            request: The incoming request.
            exc: The unhandled exception.

        Returns:
            JSONResponse with generic error message and 500 status.
        """
        request_id = getattr(request.state, "request_id", None)
        logger.error(
            f"Unhandled error: {exc}",
            extra={"type": type(exc).__name__, "request_id": request_id},
        )
        error_response = ErrorResponse(
            error="Internal server error",
            detail=str(exc) if config.debug else None,
            request_id=request_id,
        )
        return JSONResponse(
            status_code=500,
            content=error_response.model_dump(),
        )

    # ── Frontend UI ────────────────────────────────────────────────────
    frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
    if frontend_dir.is_dir():
        application.mount(
            "/app",
            StaticFiles(directory=str(frontend_dir), html=True),
            name="frontend",
        )

    # Prometheus Instrumentation
    from prometheus_fastapi_instrumentator import Instrumentator
    Instrumentator().instrument(application).expose(application, tags=["monitoring"])

    return application


# Create the app instance
app = create_app()
