"""Telegram Bot API channel adapter.

Each customer creates their own Telegram bot via @BotFather and
provides the token. Maskki auto-registers the webhook.
"""

from __future__ import annotations

from typing import Any

import httpx

from src.channels.base import ChannelAdapter, ChannelType, IncomingMessage, OutgoingMessage
from src.channels.visitor_resolver import VisitorResolver
from src.utils.logger import setup_logger

logger = setup_logger("channels.telegram")

TG_API = "https://api.telegram.org"


class TelegramAdapter(ChannelAdapter):
    """Telegram Bot API adapter."""

    def __init__(self, visitor_resolver: VisitorResolver | None = None) -> None:
        self._visitor_resolver = visitor_resolver

    def supports_voice(self) -> bool:
        return True

    def supports_video(self) -> bool:
        return False  # text + voice only

    async def parse_incoming(
        self, raw_data: dict[str, Any], channel_config: Any
    ) -> IncomingMessage:
        """Parse Telegram webhook update."""
        message = raw_data.get("message", {})
        chat_id = str(message["chat"]["id"])
        user_id = str(message["from"]["id"])

        employee_id = getattr(channel_config, "employee_id", "")
        customer_id = getattr(channel_config, "customer_id", "")

        visitor_id = user_id  # default to tg user_id
        if self._visitor_resolver:
            visitor_id = await self._visitor_resolver.resolve_visitor(
                "telegram", user_id, employee_id
            )

        incoming = IncomingMessage(
            channel=ChannelType.TELEGRAM,
            channel_session_id=f"tg_{chat_id}_{employee_id}",
            employee_id=employee_id,
            customer_id=customer_id,
            visitor_id=visitor_id,
            message_type="text",
            metadata={"chat_id": chat_id, "user_id": user_id},
        )

        if "text" in message:
            text = message["text"]
            # Handle /start command — extract employee_id parameter
            if text.startswith("/start"):
                parts = text.split(maxsplit=1)
                if len(parts) > 1:
                    incoming.metadata["start_param"] = parts[1]
                incoming.text = f"Hi, I'd like to chat!"
            else:
                incoming.text = text
        elif "voice" in message:
            incoming.message_type = "voice"
            file_id = message["voice"]["file_id"]
            bot_token = getattr(channel_config, "tg_bot_token", "")
            incoming.audio_url = await self._get_file_url(file_id, bot_token)

        return incoming

    async def send_response(
        self, channel_session_id: str, response: OutgoingMessage
    ) -> None:
        """Send response back via Telegram Bot API."""
        parts = channel_session_id.split("_", 2)
        if len(parts) < 3:
            logger.error(f"Invalid TG session ID: {channel_session_id}")
            return

        chat_id = parts[1]
        employee_id = parts[2]
        config = await self._get_channel_config(employee_id, "telegram")
        if config is None:
            logger.error(f"No TG config for employee {employee_id}")
            return

        bot_token = getattr(config, "tg_bot_token", "")
        base = f"{TG_API}/bot{bot_token}"

        async with httpx.AsyncClient(timeout=10.0) as client:
            # Text
            if response.text:
                await client.post(f"{base}/sendMessage", json={
                    "chat_id": chat_id,
                    "text": response.text,
                    "parse_mode": "Markdown",
                })

            # Voice
            if response.audio_url:
                await client.post(f"{base}/sendVoice", json={
                    "chat_id": chat_id,
                    "voice": response.audio_url,
                })

            # Quick replies as keyboard
            if response.quick_replies:
                keyboard = [[{"text": r}] for r in response.quick_replies[:4]]
                await client.post(f"{base}/sendMessage", json={
                    "chat_id": chat_id,
                    "text": response.text or "Choose an option:",
                    "reply_markup": {
                        "keyboard": keyboard,
                        "one_time_keyboard": True,
                        "resize_keyboard": True,
                    },
                })

    async def _get_file_url(self, file_id: str, bot_token: str) -> str:
        """Get download URL for a Telegram file."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{TG_API}/bot{bot_token}/getFile",
                params={"file_id": file_id},
            )
            resp.raise_for_status()
            file_path = resp.json()["result"]["file_path"]
            return f"{TG_API}/file/bot{bot_token}/{file_path}"

    async def _get_channel_config(
        self, employee_id: str, channel_type: str
    ) -> Any:
        """Look up channel config from DB."""
        return getattr(self, "_cached_config", None)

    @staticmethod
    async def register_webhook(
        bot_token: str, webhook_url: str
    ) -> dict[str, Any]:
        """Register webhook URL with Telegram Bot API."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{TG_API}/bot{bot_token}/setWebhook",
                json={"url": webhook_url, "allowed_updates": ["message"]},
            )
            resp.raise_for_status()
            result = resp.json()
            logger.info(
                "Telegram webhook registered",
                extra={"url": webhook_url, "ok": result.get("ok")},
            )
            return result
