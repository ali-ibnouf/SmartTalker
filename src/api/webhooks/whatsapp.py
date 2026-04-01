"""WhatsApp webhook endpoints for Meta Cloud API.

GET  /webhooks/whatsapp/{employee_id}  — Meta webhook verification
POST /webhooks/whatsapp/{employee_id}  — Receive incoming messages

When a visitor has an active document-collection flow in Redis,
messages are routed through DocumentFlowHandler instead of the
normal ChannelRouter → AgentEngine pipeline.
"""

from __future__ import annotations

from fastapi import APIRouter, Request, Response

from src.utils.logger import setup_logger

logger = setup_logger("webhooks.whatsapp")

router = APIRouter(tags=["webhooks"])


async def _get_channel_config(employee_id: str, db):
    """Load WhatsApp channel config for this employee."""
    if db is None:
        return None

    from sqlalchemy import select
    from src.db.models import EmployeeChannel

    async with db.session_ctx() as session:
        stmt = select(EmployeeChannel).where(
            EmployeeChannel.employee_id == employee_id,
            EmployeeChannel.channel_type == "whatsapp",
            EmployeeChannel.enabled == True,  # noqa: E712
        )
        result = await session.execute(stmt)
        return result.scalar()


@router.get("/webhooks/whatsapp/{employee_id}")
async def verify_webhook(employee_id: str, request: Request):
    """Meta webhook verification (GET)."""
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    db = getattr(request.app.state, "db", None)
    config = await _get_channel_config(employee_id, db)

    if config is None:
        logger.warning(f"No WA config for employee {employee_id}")
        return Response(status_code=404)

    if mode == "subscribe" and token == config.wa_verify_token:
        logger.info(f"WA webhook verified for employee {employee_id}")
        return Response(content=challenge, media_type="text/plain")

    logger.warning(f"WA webhook verification failed for {employee_id}")
    return Response(status_code=403)


@router.post("/webhooks/whatsapp/{employee_id}")
async def receive_message(employee_id: str, request: Request):
    """Receive WhatsApp message (POST)."""
    body = await request.json()

    # Verify it's a message (not a status update)
    entries = body.get("entry", [])
    if not isinstance(entries, list) or not entries:
        return {"status": "ok"}

    entry = entries[0]
    if not isinstance(entry, dict):
        return {"status": "ok"}

    changes = entry.get("changes", [])
    if not isinstance(changes, list) or not changes:
        return {"status": "ok"}

    change = changes[0]
    if not isinstance(change, dict):
        return {"status": "ok"}

    value = change.get("value", {})
    if not isinstance(value, dict) or not value.get("messages"):
        return {"status": "ok"}

    db = getattr(request.app.state, "db", None)
    config = await _get_channel_config(employee_id, db)

    if config is None:
        logger.warning(f"No WA config for employee {employee_id}")
        return {"status": "ignored", "reason": "no_config"}

    try:
        from src.channels.base import ChannelType
        from src.channels.whatsapp import WhatsAppAdapter

        # Get or create adapter
        channel_router = getattr(request.app.state, "channel_router", None)
        adapter = None
        if channel_router is not None:
            adapter = channel_router.get_adapter(ChannelType.WHATSAPP)
        if adapter is None:
            adapter = WhatsAppAdapter()

        # Cache config on adapter for send_response
        adapter._cached_config = config

        # Inject media processing dependencies (R2 + OCR)
        if getattr(adapter, "_r2", None) is None:
            adapter._r2 = _get_r2(request)
        if getattr(adapter, "_ocr", None) is None:
            app_config = getattr(request.app.state, "config", None)
            api_key = getattr(app_config, "dashscope_api_key", "") if app_config else ""
            if api_key:
                from src.services.ocr_service import HybridOCRService
                adapter._ocr = HybridOCRService(api_key)

        message = await adapter.parse_incoming(body, config)

        # ── Document flow intercept ──────────────────────────────
        # If the visitor has an active document flow, route through
        # DocumentFlowHandler instead of normal agent pipeline.
        handled = await _try_document_flow(request, adapter, message, config)
        if not handled:
            if channel_router is not None:
                await channel_router.handle_message(message)
            else:
                logger.warning("ChannelRouter not initialized, no fallback")
                return {"status": "ignored", "reason": "no_router"}

        logger.info(
            "WA message processed",
            extra={
                "employee_id": employee_id,
                "phone": message.metadata.get("phone"),
                "doc_flow": handled,
            },
        )
    except Exception as exc:
        logger.error(f"WA message processing failed: {exc}")
        # Return 500 so Meta retries delivery; returning 200 causes silent message loss
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=500,
            content={"status": "error", "reason": "processing_failed"},
        )

    return {"status": "ok"}


async def _try_document_flow(
    request: Request,
    adapter: "WhatsAppAdapter",
    message: "IncomingMessage",
    config: object,
) -> bool:
    """Attempt to handle message via document flow.

    Returns True if the message was handled (active flow exists or
    the message starts a new flow via a government-service keyword).
    Returns False if we should fall through to the normal pipeline.
    """
    redis = getattr(request.app.state, "redis", None)
    if redis is None:
        return False

    from src.services.document_flow_handler import FLOW_REDIS_PREFIX

    visitor_id = message.visitor_id
    flow_key = f"{FLOW_REDIS_PREFIX}{visitor_id}"

    # Check if visitor already has an active flow
    flow_data = await redis.get(flow_key)
    has_active_flow = flow_data is not None

    # If no active flow and the message is not text, skip (no way to start a flow)
    if not has_active_flow and message.message_type != "text":
        return False

    # If no active flow, check if the text looks like a government-service request
    if not has_active_flow:
        if not _looks_like_service_request(message.text or ""):
            return False

    # Build the handler
    handler = _build_doc_flow_handler(request, redis)
    if handler is None:
        return False

    # Build the message dict for the handler
    msg_dict = _incoming_to_flow_message(message)

    result = await handler.handle(
        visitor_id=visitor_id,
        customer_id=message.customer_id,
        avatar_id=message.employee_id,
        message=msg_dict,
    )

    # Send the voice response back via WhatsApp
    from src.channels.base import OutgoingMessage

    outgoing = OutgoingMessage(text=result.voice_text)

    # TTS: convert voice_text to audio if possible
    tts_rest = _get_tts_rest(request)
    r2 = _get_r2(request)

    if tts_rest and r2:
        try:
            audio_url = await tts_rest.synthesize_to_url(
                text=result.voice_text,
                language=result.language,
                r2_storage=r2,
                session_id=message.channel_session_id,
            )
            outgoing.audio_url = audio_url
        except Exception as exc:
            logger.warning("Doc flow TTS failed, sending text only", extra={"error": str(exc)})

    await adapter.send_response(message.channel_session_id, outgoing)

    # If there's a session link, send it as a separate text message
    if result.session_link:
        link_msg = OutgoingMessage(text=result.session_link)
        await adapter.send_response(message.channel_session_id, link_msg)

    logger.info(
        "Document flow handled",
        extra={
            "visitor_prefix": visitor_id[:8],
            "state": result.state,
            "has_link": result.session_link is not None,
            "notify_supervisor": result.notify_supervisor,
        },
    )
    return True


def _looks_like_service_request(text: str) -> bool:
    """Quick keyword check — does the text mention a government service?"""
    if not text:
        return False
    text_lower = text.lower().strip()

    # Arabic keywords for government services
    keywords = [
        "رخصة", "تجديد", "إقامة", "اقامة", "تأشيرة", "تاشيرة",
        "مركبة", "سيارة", "فيزا", "visa", "license",
        "residency", "vehicle", "registration", "renewal",
        "خدمة", "معاملة", "وثيقة",
    ]
    return any(kw in text_lower for kw in keywords)


def _build_doc_flow_handler(request: Request, redis):
    """Construct DocumentFlowHandler from app state dependencies."""
    config = getattr(request.app.state, "config", None)

    # LLM service — lives inside the pipeline
    pipeline = getattr(request.app.state, "pipeline", None)
    llm = getattr(pipeline, "_llm", None) if pipeline else None
    if llm is None:
        return None

    # OCR service
    ocr = None
    api_key = getattr(config, "dashscope_api_key", "") if config else ""
    if api_key:
        from src.services.ocr_service import HybridOCRService
        ocr = HybridOCRService(api_key)

    # Session link service (lazy — construct from redis + config)
    session_links = getattr(request.app.state, "session_link_service", None)
    if session_links is None and redis and config:
        from src.services.session_link_service import SessionLinkService
        base_url = getattr(config, "session_link_base_url", "https://app.maskki.com")
        session_links = SessionLinkService(redis, base_url)
        request.app.state.session_link_service = session_links

    from src.services.document_flow_handler import DocumentFlowHandler

    return DocumentFlowHandler(
        redis_client=redis,
        llm_service=llm,
        ocr_service=ocr,
        session_link_service=session_links,
    )


def _get_tts_rest(request: Request):
    """Get or create TTSRestService from app state."""
    tts_rest = getattr(request.app.state, "tts_rest", None)
    if tts_rest is not None:
        return tts_rest

    config = getattr(request.app.state, "config", None)
    api_key = getattr(config, "dashscope_api_key", "") if config else ""
    if api_key:
        from src.services.tts_rest_service import TTSRestService
        tts_rest = TTSRestService(api_key)
        request.app.state.tts_rest = tts_rest
        return tts_rest

    return None


def _get_r2(request: Request):
    """Get or create R2Storage from app state."""
    r2 = getattr(request.app.state, "r2_docflow", None)
    if r2 is not None:
        return r2

    config = getattr(request.app.state, "config", None)
    if config and getattr(config, "r2_account_id", ""):
        from src.services.r2_storage import R2Storage
        r2 = R2Storage(config)
        request.app.state.r2_docflow = r2
        return r2

    return None


def _incoming_to_flow_message(message: "IncomingMessage") -> dict:
    """Convert IncomingMessage to the dict format DocumentFlowHandler expects."""
    from src.channels.base import IncomingMessage

    if message.message_type == "text":
        return {
            "type": "text",
            "content": message.text or "",
        }

    # Image messages (from OCR handler) have ocr_result in metadata
    if message.metadata.get("media_type") == "image":
        return {
            "type": "image",
            "content": message.metadata.get("image_bytes", b""),
            "format": message.metadata.get("mime_type", "image/jpeg").split("/")[-1],
            "r2_key": message.metadata.get("media_url", ""),
        }

    # Document messages — if image-type, treat as image
    if message.metadata.get("media_type") == "document":
        mime = message.metadata.get("mime_type", "")
        if mime.startswith("image/"):
            return {
                "type": "image",
                "content": message.metadata.get("image_bytes", b""),
                "format": mime.split("/")[-1],
                "r2_key": message.metadata.get("media_url", ""),
            }

    # Default: treat as text
    return {
        "type": "text",
        "content": message.text or "",
    }
