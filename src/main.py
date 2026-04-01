"""FastAPI application entry point.

Creates the app with lifespan management, mounts routers,
serves static files, and configures global exception handling.
Run with: uvicorn src.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.api.middleware import (
    APIKeyAuthMiddleware,
    CORSCleanupMiddleware,
    LoggingMiddleware,
    RedisRateLimitMiddleware,
    RequestBodyLimitMiddleware,
    RequestIDMiddleware,
    SecurityHeadersMiddleware,
    get_cors_config,
)
from src.api.routes import router as api_router
from src.api.dashboard_routes import router as dashboard_router
from src.api.tool_routes import router as tool_router
from src.api.workflow_routes import router as workflow_router
from src.api.learning_routes import router as learning_router
from src.api.admin_cost_routes import router as admin_cost_router
from src.api.admin_session_routes import router as admin_session_router
from src.api.admin_guardian_routes import router as admin_guardian_router
from src.services.ai_agent.routes import router as agent_router
from src.api.channel_routes import router as channel_router
from src.api.onboarding_routes import router as onboarding_router
from src.api.visitor_routes import router as visitor_router
from src.api.session_links import router as session_links_router
from src.api.webhooks.whatsapp import router as wa_webhook_router
from src.api.webhooks.telegram import router as tg_webhook_router
from src.api.webhooks.paddle import router as paddle_webhook_router
from src.api.schemas import ErrorResponse
from src.api.websocket import WebSocketManager, websocket_chat_endpoint
from src.api.operator_ws import OperatorWebSocketManager, operator_websocket_endpoint
from src.api.ws_visitor import visitor_session_handler
from src.config import get_settings
from src.db import Database
from src.pipeline.billing import BillingEngine
from src.pipeline.kill_switch import KillSwitch
from src.pipeline.orchestrator import SmartTalkerPipeline
from src.pipeline.persona import PersonaEngine
from src.pipeline.learning_analytics import LearningAnalytics
from src.pipeline.supervisor import SupervisorEngine
from src.pipeline.analytics import AnalyticsEngine
from src.utils.exceptions import SmartTalkerError
from src.utils.async_utils import background_task_error_handler
from src.utils.logger import setup_logger
from src.integrations.whatsapp import WhatsAppClient
from src.integrations.storage import StorageManager
from src.integrations.webrtc import WebRTCSignalingHandler, webrtc_signaling_endpoint

logger = setup_logger("main")


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown lifecycle.

    On startup: load configuration and initialize pipeline models.
    On shutdown: unload all models and free resources.

    Args:
        application: The FastAPI application instance.

    Yields:
        Control back to the application during its lifetime.
    """
    # ── Startup ──────────────────────────────────────────────────────────

    # In test mode, skip all real service connections
    if os.environ.get("TESTING"):
        if get_settings().app_env == "production":
            raise ValueError("TESTING flag cannot be set in production environment")
        logger.info("Test mode — skipping service connections")
        application.state.config = get_settings()
        application.state.db = None
        application.state.redis = None
        application.state.pipeline = None
        application.state.billing = None
        application.state.whatsapp = None
        application.state.ws_manager = None
        application.state.operator_manager = None
        application.state.webrtc_handler = None
        application.state.supervisor = None
        application.state.analytics = None
        application.state.guardrails = None
        application.state.persona_engine = None
        application.state.kill_switch = None
        application.state.ai_agent = None
        application.state.learning_analytics = None
        application.state.learning_engine = None
        application.state.storage = None
        application.state.cost_guardian = None
        yield
        return

    logger.info("SmartTalker starting up...")

    config = get_settings()
    application.state.config = config

    # Initialize PostgreSQL database
    db = Database(url=config.database_url, echo=config.debug)
    try:
        await db.connect()
        logger.info("PostgreSQL database connected")
    except Exception as exc:
        logger.warning(f"PostgreSQL connection failed: {exc}")
        db = None
    application.state.db = db

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

    pipeline = SmartTalkerPipeline(config, db=db)
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

    # Initialize Operator WebSocket manager
    operator_storage = config.storage_base_dir / "operator"
    operator_manager = OperatorWebSocketManager(
        customer_ws_manager=ws_manager,
        storage_dir=operator_storage,
        api_key=config.api_key,
        pipeline=pipeline,
    )
    ws_manager.set_operator_manager(operator_manager)
    application.state.operator_manager = operator_manager

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

    # Load Training engine (async — requires aiosqlite or db)
    if pipeline._training is not None:
        try:
            await pipeline._training.load()
            logger.info("Training engine loaded successfully")
        except Exception as exc:
            logger.warning(f"Training engine failed to load: {exc}")
            
    # Initialize Learning Analytics
    learning_analytics = LearningAnalytics(config, db=db)
    try:
        await learning_analytics.load()
        logger.info("LearningAnalytics loaded")
        if pipeline._training is not None:
            pipeline._training._analytics = learning_analytics
    except Exception as exc:
        logger.warning(f"LearningAnalytics failed to load: {exc}")
    application.state.learning_analytics = learning_analytics

    # Seed industry categories for cross-learning
    if db:
        try:
            from src.agent.cross_learning import CrossLearningEngine
            cross_learning = CrossLearningEngine(db)
            seeded = await cross_learning.seed_industries()
            if seeded:
                logger.info(f"Seeded {seeded} industry categories")
        except Exception as exc:
            logger.warning(f"Industry seeding failed: {exc}")

    # Initialize Supervisor
    supervisor = SupervisorEngine(config, db=db)
    try:
        await supervisor.load()
        logger.info("SupervisorEngine loaded")
    except Exception as exc:
        logger.warning(f"SupervisorEngine failed to load: {exc}")
    application.state.supervisor = supervisor
    supervisor.set_ws_manager(ws_manager)

    # Initialize Analytics
    analytics_engine = AnalyticsEngine(config, db=db)
    try:
        await analytics_engine.load()
        logger.info("AnalyticsEngine loaded")
    except Exception as exc:
        logger.warning(f"AnalyticsEngine failed to load: {exc}")
    application.state.analytics = analytics_engine

    # Map Guardrails (pipeline's or standalone fallback)
    if pipeline._guardrails is not None:
        try:
            await pipeline._guardrails.load()
            logger.info("GuardrailsEngine loaded (pipeline)")
        except Exception as exc:
            logger.warning(f"GuardrailsEngine failed to load: {exc}")
        application.state.guardrails = pipeline._guardrails
    else:
        from src.agent.guardrails import GuardrailsEngine as StandaloneGuardrails
        application.state.guardrails = StandaloneGuardrails()
        logger.info("GuardrailsEngine loaded (standalone)")

    # Initialize Billing engine
    billing = BillingEngine(config, db=db)
    try:
        await billing.load()
        logger.info("BillingEngine loaded")
    except Exception as exc:
        logger.warning(f"BillingEngine failed to load: {exc}")
    application.state.billing = billing

    # Initialize Persona engine
    persona_engine = PersonaEngine(config, db=db)
    try:
        await persona_engine.load()
        logger.info("PersonaEngine loaded")
    except Exception as exc:
        logger.warning(f"PersonaEngine failed to load: {exc}")
    application.state.persona_engine = persona_engine

    # Initialize Auto-Learning Engine
    from src.pipeline.auto_learning import AutoLearningEngine
    learning_engine = AutoLearningEngine(db=db, config=config)
    application.state.learning_engine = learning_engine
    logger.info("AutoLearningEngine loaded")

    # Initialize Kill Switch
    kill_switch = KillSwitch(db=db, redis=redis_client, ws_manager=ws_manager)
    application.state.kill_switch = kill_switch

    # Initialize AI Optimization Agent
    from src.services.ai_agent import AIAgent
    from src.services.ai_agent.config import AgentSettings
    from src.services.ai_agent.rules import AgentContext

    agent_config = AgentSettings()
    agent_ctx = AgentContext(
        db=db,
        redis=redis_client,
        pipeline=pipeline,
        config=config,
        agent_config=agent_config,
        operator_manager=operator_manager,
    )
    ai_agent = AIAgent(agent_ctx)
    if agent_config.agent_enabled:
        await ai_agent.start()
        logger.info("AI Optimization Agent started")
    application.state.ai_agent = ai_agent

    # Initialize Cost Guardian
    from src.services.cost_guardian import CostGuardian
    cost_guardian = CostGuardian(
        db=db,
        redis=redis_client,
        resend_api_key=config.resend_api_key,
        runpod_client=getattr(pipeline, "_runpod", None),
    )
    guardian_task = asyncio.create_task(cost_guardian.start())
    guardian_task.add_done_callback(background_task_error_handler)
    application.state.cost_guardian = cost_guardian
    logger.info("Cost Guardian started")

    # Initialize storage manager and schedule periodic cleanup
    storage = StorageManager(config)
    application.state.storage = storage

    # Ensure static files directory exists
    config.static_files_dir.mkdir(parents=True, exist_ok=True)

    # Background cleanup task — runs every hour
    async def _periodic_cleanup():
        import asyncio as _asyncio
        while True:
            await _asyncio.sleep(3600)
            try:
                deleted = storage.cleanup_old_files()
                if deleted:
                    logger.info("Periodic cleanup", extra={"deleted": deleted})
            except Exception as exc:
                logger.warning(f"Periodic cleanup failed: {exc}")

    cleanup_task = asyncio.create_task(_periodic_cleanup())
    cleanup_task.add_done_callback(background_task_error_handler)

    # Weekly Cross-Learning CRON Task — runs every 7 days
    async def _weekly_cross_learning():
        import asyncio as _asyncio
        while True:
            # Sleep 7 days (604800 seconds)
            await _asyncio.sleep(604800)
            if db:
                try:
                    from src.agent.cross_learning import CrossLearningEngine
                    cross_learning = CrossLearningEngine(db)
                    
                    # While the seed logic runs on startup, a future generalization engine
                    # processing phase could be hooked here for dynamic insights extraction.
                    # Currently, we trigger a stats log to ensure the cycle proves active.
                    stats = await cross_learning.get_stats()
                    logger.info("Weekly Cross-Learning Cycle Executed", extra={"stats": stats})
                except Exception as exc:
                    logger.warning(f"Cross-learning CRON cycle failed: {exc}")

    cross_learning_task = asyncio.create_task(_weekly_cross_learning())
    cross_learning_task.add_done_callback(background_task_error_handler)

    logger.info(
        "SmartTalker ready",
        extra={"host": config.api_host, "port": config.api_port},
    )

    yield

    # ── Shutdown ─────────────────────────────────────────────────────────
    logger.info("SmartTalker shutting down...")
    cleanup_task.cancel()
    cross_learning_task.cancel()
    guardian_task.cancel()
    try:
        await cleanup_task
        await cross_learning_task
        await guardian_task
    except asyncio.CancelledError:
        pass

    # Stop Cost Guardian + AI Agent
    await cost_guardian.stop()
    await ai_agent.stop()

    # Unload engines
    await learning_analytics.unload()
    await supervisor.unload()
    await analytics_engine.unload()
    if pipeline._guardrails is not None:
        await pipeline._guardrails.unload()
        
    await billing.unload()
    await learning_engine.close()
    await persona_engine.unload()
    await pipeline.unload_all()
    await whatsapp.close()
    if redis_client:
        await redis_client.close()
    if db:
        await db.disconnect()
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
            "Qwen3 LLM, DashScope TTS/ASR, RunPod GPU rendering."
        ),
        version="3.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # ── Middleware (order matters: outermost first) ───────────────────────

    # Inner middlewares (Auth, RateLimit, Body size)
    application.add_middleware(RequestBodyLimitMiddleware, max_bytes=10 * 1024 * 1024)
    application.add_middleware(APIKeyAuthMiddleware, config=config)
    application.add_middleware(RedisRateLimitMiddleware, config=config)

    # Outer middlewares
    cors_config = get_cors_config(config.cors_origins)
    application.add_middleware(CORSMiddleware, **cors_config)
    application.add_middleware(CORSCleanupMiddleware)
    application.add_middleware(SecurityHeadersMiddleware)
    application.add_middleware(LoggingMiddleware)
    application.add_middleware(RequestIDMiddleware)

    # ── Routers ──────────────────────────────────────────────────────────
    application.include_router(api_router)
    application.include_router(dashboard_router)
    application.include_router(tool_router)
    application.include_router(workflow_router)
    application.include_router(learning_router)
    application.include_router(admin_cost_router)
    application.include_router(admin_session_router)
    application.include_router(admin_guardian_router)
    application.include_router(agent_router)
    application.include_router(channel_router)
    application.include_router(onboarding_router)
    application.include_router(visitor_router)
    application.include_router(session_links_router)
    application.include_router(wa_webhook_router)
    application.include_router(tg_webhook_router)
    application.include_router(paddle_webhook_router)

    # ── WebSocket Endpoint ───────────────────────────────────────────────
    @application.websocket("/ws/chat")
    async def ws_chat(websocket: WebSocket):
        manager = application.state.ws_manager
        await websocket_chat_endpoint(websocket, manager)

    # ── Visitor Direct Session ────────────────────────────────────────────
    @application.websocket("/session")
    async def ws_session(websocket: WebSocket):
        await visitor_session_handler(websocket)

    # ── Operator WebSocket ───────────────────────────────────────────────
    @application.websocket("/ws/operator")
    async def ws_operator(websocket: WebSocket):
        manager = application.state.operator_manager
        await operator_websocket_endpoint(websocket, manager)

    # ── WebRTC Signaling ─────────────────────────────────────────────────
    if config.webrtc_enabled:
        @application.websocket("/ws/rtc")
        async def ws_rtc(websocket: WebSocket):
            handler = application.state.webrtc_handler
            if handler:
                await webrtc_signaling_endpoint(
                    websocket, handler, api_key=config.api_key,
                )

    # ── Static Files ─────────────────────────────────────────────────────
    config.static_files_dir.mkdir(parents=True, exist_ok=True)

    # Documents directory
    docs_dir = config.storage_base_dir / "operator" / "documents"
    docs_dir.mkdir(parents=True, exist_ok=True)
    application.mount(
        "/files/documents",
        StaticFiles(directory=str(docs_dir)),
        name="documents",
    )

    # VRM models directory
    vrm_dir = config.static_files_dir / "vrm"
    vrm_dir.mkdir(parents=True, exist_ok=True)
    application.mount(
        "/files/vrm",
        StaticFiles(directory=str(vrm_dir)),
        name="vrm_files",
    )

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

    # Avatar video clips (pre-generated via RunPod worker)
    clips_dir = config.clips_dir
    clips_dir.mkdir(parents=True, exist_ok=True)
    application.mount(
        "/clips",
        StaticFiles(directory=str(clips_dir)),
        name="clips",
    )

    # Operator dashboard — explicit route for /operator
    operator_html = frontend_dir / "operator.html"
    if operator_html.is_file():
        @application.get("/operator", include_in_schema=False)
        async def serve_operator():
            return FileResponse(str(operator_html))

    # Main customer frontend (catch-all — must be last)
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
