"""WhatsApp webhook endpoints for Meta Cloud API.

GET  /webhooks/whatsapp/{employee_id}  — Meta webhook verification
POST /webhooks/whatsapp/{employee_id}  — Receive incoming messages
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

    # Get the channel router from app state
    channel_router = getattr(request.app.state, "channel_router", None)
    if channel_router is None:
        logger.warning("ChannelRouter not initialized")
        return {"status": "ignored", "reason": "no_router"}

    try:
        from src.channels.base import ChannelType
        from src.channels.whatsapp import WhatsAppAdapter

        adapter = channel_router.get_adapter(ChannelType.WHATSAPP)
        if adapter is None:
            adapter = WhatsAppAdapter()

        # Cache config on adapter for send_response
        adapter._cached_config = config

        message = await adapter.parse_incoming(body, config)
        await channel_router.handle_message(message)

        logger.info(
            "WA message processed",
            extra={"employee_id": employee_id, "phone": message.metadata.get("phone")},
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
