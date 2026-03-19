"""Telegram webhook endpoint for Bot API.

POST /webhooks/telegram/{employee_id}  — Receive Telegram updates
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from src.utils.logger import setup_logger

logger = setup_logger("webhooks.telegram")

router = APIRouter(tags=["webhooks"])


async def _get_channel_config(employee_id: str, db):
    """Load Telegram channel config for this employee."""
    if db is None:
        return None

    from sqlalchemy import select
    from src.db.models import EmployeeChannel

    async with db.session_ctx() as session:
        stmt = select(EmployeeChannel).where(
            EmployeeChannel.employee_id == employee_id,
            EmployeeChannel.channel_type == "telegram",
            EmployeeChannel.enabled == True,  # noqa: E712
        )
        result = await session.execute(stmt)
        return result.scalar()


@router.post("/webhooks/telegram/{employee_id}")
async def telegram_webhook(employee_id: str, request: Request):
    """Receive Telegram update (POST)."""
    body = await request.json()

    if "message" not in body:
        return {"status": "ok"}

    db = getattr(request.app.state, "db", None)
    config = await _get_channel_config(employee_id, db)

    if config is None:
        logger.warning(f"No TG config for employee {employee_id}")
        return {"status": "ignored", "reason": "no_config"}

    channel_router = getattr(request.app.state, "channel_router", None)
    if channel_router is None:
        logger.warning("ChannelRouter not initialized")
        return {"status": "ignored", "reason": "no_router"}

    try:
        from src.channels.base import ChannelType
        from src.channels.telegram import TelegramAdapter

        adapter = channel_router.get_adapter(ChannelType.TELEGRAM)
        if adapter is None:
            adapter = TelegramAdapter()

        # Cache config on adapter for send_response
        adapter._cached_config = config

        message = await adapter.parse_incoming(body, config)
        await channel_router.handle_message(message)

        logger.info(
            "TG message processed",
            extra={"employee_id": employee_id, "chat_id": message.metadata.get("chat_id")},
        )
    except Exception as exc:
        logger.error(f"TG message processing failed: {exc}")
        # Return 500 so Telegram retries delivery; returning 200 causes silent message loss
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=500,
            content={"status": "error", "reason": "processing_failed"},
        )

    return {"status": "ok"}
