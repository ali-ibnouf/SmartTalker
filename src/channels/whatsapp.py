"""WhatsApp Cloud API channel adapter.

Uses Meta's WhatsApp Business API (Cloud API) to send/receive messages.
Each customer provides their own WhatsApp Business phone number and credentials
configured via the employee_channels table.
"""

from __future__ import annotations

from typing import Any

import httpx

from src.channels.base import ChannelAdapter, ChannelType, IncomingMessage, OutgoingMessage
from src.channels.visitor_resolver import VisitorResolver
from src.utils.logger import setup_logger

logger = setup_logger("channels.whatsapp")

META_API = "https://graph.facebook.com/v21.0"


class WhatsAppAdapter(ChannelAdapter):
    """WhatsApp Cloud API adapter."""

    def __init__(self, visitor_resolver: VisitorResolver | None = None) -> None:
        self._visitor_resolver = visitor_resolver

    def supports_voice(self) -> bool:
        return True  # WhatsApp supports voice messages

    def supports_video(self) -> bool:
        return False  # No video avatar in WhatsApp (text + voice only)

    async def parse_incoming(
        self, raw_data: dict[str, Any], channel_config: Any
    ) -> IncomingMessage:
        """Parse WhatsApp webhook payload."""
        entries = raw_data.get("entry")
        if not entries or not isinstance(entries, list) or len(entries) == 0:
            raise ValueError("Invalid WhatsApp webhook: missing or empty 'entry'")
        entry = entries[0]

        changes = entry.get("changes")
        if not changes or not isinstance(changes, list) or len(changes) == 0:
            raise ValueError("Invalid WhatsApp webhook: missing or empty 'changes'")
        value = changes[0].get("value", {})

        messages = value.get("messages")
        if not messages or not isinstance(messages, list) or len(messages) == 0:
            raise ValueError("Invalid WhatsApp webhook: missing or empty 'messages'")
        msg = messages[0]

        phone = msg.get("from", "")
        msg_type = msg.get("type", "text")

        # Resolve or create unified visitor_id
        employee_id = getattr(channel_config, "employee_id", "")
        customer_id = getattr(channel_config, "customer_id", "")

        visitor_id = phone  # default to phone
        if self._visitor_resolver:
            visitor_id = await self._visitor_resolver.resolve_visitor(
                "whatsapp", phone, employee_id
            )

        incoming = IncomingMessage(
            channel=ChannelType.WHATSAPP,
            channel_session_id=f"wa_{phone}_{employee_id}",
            employee_id=employee_id,
            customer_id=customer_id,
            visitor_id=visitor_id,
            message_type="text",
            metadata={"phone": phone, "wa_message_id": msg["id"]},
        )

        if msg_type == "text":
            incoming.text = msg["text"]["body"]
        elif msg_type == "audio":
            media_id = msg["audio"]["id"]
            incoming.message_type = "voice"
            wa_token = getattr(channel_config, "wa_access_token", "")
            incoming.audio_url = await self._get_media_url(media_id, wa_token)

        return incoming

    async def send_response(
        self, channel_session_id: str, response: OutgoingMessage
    ) -> None:
        """Send response back via WhatsApp."""
        parts = channel_session_id.split("_", 2)
        if len(parts) < 3:
            logger.error(f"Invalid WA session ID: {channel_session_id}")
            return

        phone = parts[1]
        employee_id = parts[2]
        config = await self._get_channel_config(employee_id, "whatsapp")
        if config is None:
            logger.error(f"No WA config for employee {employee_id}")
            return

        wa_token = getattr(config, "wa_access_token", "")
        phone_number_id = getattr(config, "wa_phone_number_id", "")
        headers = {
            "Authorization": f"Bearer {wa_token}",
            "Content-Type": "application/json",
        }

        # 1. Send text
        if response.text:
            await self._send_wa_message(phone_number_id, phone, {
                "messaging_product": "whatsapp",
                "to": phone,
                "type": "text",
                "text": {"body": response.text},
            }, headers)

        # 2. Send voice note (if available)
        if response.audio_url:
            await self._send_wa_message(phone_number_id, phone, {
                "messaging_product": "whatsapp",
                "to": phone,
                "type": "audio",
                "audio": {"link": response.audio_url},
            }, headers)

        # 3. Quick replies (if any)
        if response.quick_replies and response.text:
            buttons = [
                {"type": "reply", "reply": {"id": f"qr_{i}", "title": r[:20]}}
                for i, r in enumerate(response.quick_replies[:3])
            ]
            await self._send_wa_message(phone_number_id, phone, {
                "messaging_product": "whatsapp",
                "to": phone,
                "type": "interactive",
                "interactive": {
                    "type": "button",
                    "body": {"text": response.text},
                    "action": {"buttons": buttons},
                },
            }, headers)

    async def _send_wa_message(
        self,
        phone_number_id: str,
        to: str,
        body: dict[str, Any],
        headers: dict[str, str],
    ) -> None:
        """Send a single message via WhatsApp Cloud API."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{META_API}/{phone_number_id}/messages",
                    headers=headers,
                    json=body,
                )
                resp.raise_for_status()
        except Exception as exc:
            logger.error(f"Failed to send WA message: {exc}")

    async def _get_media_url(self, media_id: str, access_token: str) -> str:
        """Get download URL for a WhatsApp media file."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{META_API}/{media_id}",
                headers={"Authorization": f"Bearer {access_token}"},
            )
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict):
                return data.get("url", "")
            return ""

    async def _get_channel_config(
        self, employee_id: str, channel_type: str
    ) -> Any:
        """Look up channel config from DB. Returns None if not found."""
        # This is resolved by the webhook handler which passes the config.
        # For send_response, we need to look it up again.
        # In practice, the webhook handler caches this on the adapter.
        return getattr(self, "_cached_config", None)
