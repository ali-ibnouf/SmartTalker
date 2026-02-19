"""WhatsApp Business API integration via Meta Graph API.

Handles webhook verification, incoming message parsing,
audio message download, and response sending (text + audio).
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional

import httpx

from src.config import Settings
from src.utils.exceptions import WhatsAppError
from src.utils.logger import setup_logger, log_with_latency

logger = setup_logger("integrations.whatsapp")


class WhatsAppClient:
    """Meta WhatsApp Business API client.

    Manages webhook verification, message parsing, media download,
    and sending text/audio replies.

    Args:
        config: Application settings with WhatsApp credentials.
    """

    def __init__(self, config: Settings) -> None:
        """Initialize the WhatsApp client.

        Args:
            config: Application settings instance.
        """
        self._config = config
        self._verify_token = config.whatsapp_verify_token or ""
        self._access_token = config.whatsapp_access_token or ""
        self._phone_number_id = config.whatsapp_phone_number_id or ""
        self._app_secret = config.whatsapp_app_secret or ""
        self._api_version = config.whatsapp_api_version
        self._base_url = f"https://graph.facebook.com/{self._api_version}"
        self._media_dir = config.storage_base_dir / "whatsapp_media"
        self._media_dir.mkdir(parents=True, exist_ok=True)

        self._media_dir.mkdir(parents=True, exist_ok=True)

        self._client: Optional[httpx.AsyncClient] = None
        self._processed_ids: deque = deque(maxlen=1000)

        logger.info(
            "WhatsAppClient initialized",
            extra={"phone_id": self._phone_number_id, "api_version": self._api_version},
        )

    def is_duplicate(self, message_id: str) -> bool:
        """Check if a message ID has already been processed.

        Args:
            message_id: Unique message ID from WhatsApp.

        Returns:
            True if duplicate, False otherwise (and marks as processed).
        """
        if message_id in self._processed_ids:
            return True
        self._processed_ids.append(message_id)
        return False

    @property
    def is_configured(self) -> bool:
        """Check if all required credentials are set."""
        return bool(self._access_token and self._phone_number_id)

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers={
                    "Authorization": f"Bearer {self._access_token}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(30.0, connect=10.0),
            )
        return self._client

    # ── Webhook Verification ─────────────────────────────────────────────

    def verify_webhook(
        self,
        mode: Optional[str],
        token: Optional[str],
        challenge: Optional[str],
    ) -> Optional[str]:
        """Verify a webhook subscription request from Meta.

        Args:
            mode: hub.mode parameter (should be "subscribe").
            token: hub.verify_token parameter.
            challenge: hub.challenge parameter to echo back.

        Returns:
            The challenge string if verification succeeds, None otherwise.
        """
        if mode == "subscribe" and token == self._verify_token:
            logger.info("Webhook verified successfully")
            return challenge
        logger.warning("Webhook verification failed", extra={"mode": mode})
        return None

    def verify_signature(self, payload: bytes, signature: str) -> bool:
        """Verify the X-Hub-Signature-256 header.

        Args:
            payload: Raw request body bytes.
            signature: Signature from X-Hub-Signature-256 header.

        Returns:
            True if signature is valid.
        """
        if not self._app_secret:
            logger.warning("App secret not configured — rejecting unverifiable request")
            return False

        expected = "sha256=" + hmac.new(
            self._app_secret.encode(),
            payload,
            hashlib.sha256,
        ).hexdigest()

        return hmac.compare_digest(expected, signature)

    # ── Message Parsing ──────────────────────────────────────────────────

    @staticmethod
    def parse_incoming(payload: dict) -> list[dict[str, Any]]:
        """Parse incoming webhook payload into structured messages.

        Args:
            payload: Raw webhook JSON payload from Meta.

        Returns:
            List of parsed message dicts with keys:
            - from_number: Sender's phone number
            - message_id: Unique message ID
            - type: Message type (text, audio, image, etc.)
            - text: Text content (for text messages)
            - media_id: Media ID (for audio/image messages)
            - timestamp: Message timestamp
        """
        messages: list[dict[str, Any]] = []

        try:
            entries = payload.get("entry", [])
            for entry in entries:
                changes = entry.get("changes", [])
                for change in changes:
                    value = change.get("value", {})
                    raw_messages = value.get("messages", [])

                    for msg in raw_messages:
                        parsed: dict[str, Any] = {
                            "from_number": msg.get("from", ""),
                            "message_id": msg.get("id", ""),
                            "type": msg.get("type", ""),
                            "timestamp": msg.get("timestamp", ""),
                        }

                        msg_type = msg.get("type", "")

                        if msg_type == "text":
                            parsed["text"] = msg.get("text", {}).get("body", "")
                        elif msg_type == "audio":
                            parsed["media_id"] = msg.get("audio", {}).get("id", "")
                            parsed["mime_type"] = msg.get("audio", {}).get("mime_type", "")
                        elif msg_type == "image":
                            parsed["media_id"] = msg.get("image", {}).get("id", "")
                            parsed["mime_type"] = msg.get("image", {}).get("mime_type", "")

                        messages.append(parsed)

        except Exception as exc:
            logger.error(f"Failed to parse webhook payload: {exc}")

        return messages

    # ── Media Download ───────────────────────────────────────────────────

    async def download_media(self, media_id: str, mime_type: str = "") -> Path:
        """Download a media file from the WhatsApp API.

        Args:
            media_id: Media ID from the webhook payload.
            mime_type: MIME type (used for extension detection).

        Returns:
            Path to the downloaded media file.

        Raises:
            WhatsAppError: If download fails.
        """
        start = time.perf_counter()

        try:
            client = await self._get_client()

            # Step 1: Get media URL
            url_response = await client.get(f"/{media_id}")
            url_response.raise_for_status()
            media_url = url_response.json().get("url", "")

            if not media_url:
                raise WhatsAppError(message=f"No URL returned for media {media_id}")

            # Step 2: Download media binary
            media_response = await client.get(media_url)
            media_response.raise_for_status()

            # Determine file extension
            ext = self._mime_to_ext(mime_type)
            file_path = self._media_dir / f"{media_id}{ext}"
            file_path.write_bytes(media_response.content)

            elapsed_ms = int((time.perf_counter() - start) * 1000)
            log_with_latency(
                logger, "Media downloaded", elapsed_ms,
                extra={"media_id": media_id, "size_bytes": len(media_response.content)},
            )
            return file_path

        except WhatsAppError:
            raise
        except Exception as exc:
            raise WhatsAppError(
                message=f"Failed to download media {media_id}",
                detail=str(exc),
                original_exception=exc,
            ) from exc

    # ── Sending Messages ─────────────────────────────────────────────────

    async def send_text(self, to_number: str, text: str) -> dict:
        """Send a text message via WhatsApp.

        Args:
            to_number: Recipient phone number (E.164 format).
            text: Message text.

        Returns:
            API response dict.

        Raises:
            WhatsAppError: If sending fails.
        """
        payload = {
            "messaging_product": "whatsapp",
            "to": to_number,
            "type": "text",
            "text": {"body": text},
        }
        return await self._send_message(payload)

    async def send_audio(self, to_number: str, audio_url: str) -> dict:
        """Send an audio message via WhatsApp.

        Args:
            to_number: Recipient phone number (E.164 format).
            audio_url: Publicly accessible URL to the audio file.

        Returns:
            API response dict.

        Raises:
            WhatsAppError: If sending fails.
        """
        payload = {
            "messaging_product": "whatsapp",
            "to": to_number,
            "type": "audio",
            "audio": {"link": audio_url},
        }
        return await self._send_message(payload)

    async def mark_read(self, message_id: str) -> None:
        """Mark a message as read.

        Args:
            message_id: ID of the message to mark as read.
        """
        try:
            payload = {
                "messaging_product": "whatsapp",
                "status": "read",
                "message_id": message_id,
            }
            await self._send_message(payload)
        except Exception as exc:
            logger.warning(f"Failed to mark message as read: {exc}")

    async def _send_message(self, payload: dict) -> dict:
        """Send a message via the WhatsApp API.

        Args:
            payload: Message payload dict.

        Returns:
            API response dict.

        Raises:
            WhatsAppError: If the API call fails.
        """
        for attempt in range(3):
            try:
                client = await self._get_client()
                response = await client.post(
                    f"/{self._phone_number_id}/messages",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

                logger.info(
                    "Message sent",
                    extra={"to": payload.get("to", ""), "type": payload.get("type", "")},
                )
                return data

            except httpx.HTTPStatusError as exc:
                # Don't retry 4xx errors (client error)
                if 400 <= exc.response.status_code < 500:
                    raise WhatsAppError(
                        message=f"WhatsApp API error: {exc.response.status_code}",
                        detail=exc.response.text,
                        original_exception=exc,
                    ) from exc
                
                # Retry 5xx errors
                if attempt == 2:
                    raise WhatsAppError(
                        message=f"WhatsApp API error: {exc.response.status_code}",
                        detail=exc.response.text,
                        original_exception=exc,
                    ) from exc

            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                if attempt == 2:
                    raise WhatsAppError(
                        message="Failed to send WhatsApp message",
                        detail=str(exc),
                        original_exception=exc,
                    ) from exc
                
                # Exponential backoff: 0.5s, 1.0s, ...
                await asyncio.sleep(0.5 * (2 ** attempt))

            except Exception as exc:
                 raise WhatsAppError(
                    message="Failed to send WhatsApp message",
                    detail=str(exc),
                    original_exception=exc,
                ) from exc
        
        # Should not be reached
        raise WhatsAppError(message="Failed to send WhatsApp message after retries")

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _mime_to_ext(mime_type: str) -> str:
        """Convert MIME type to file extension.

        Args:
            mime_type: MIME type string.

        Returns:
            File extension with dot prefix.
        """
        mapping = {
            "audio/ogg": ".ogg",
            "audio/mpeg": ".mp3",
            "audio/mp4": ".m4a",
            "audio/wav": ".wav",
            "audio/webm": ".webm",
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
        }
        return mapping.get(mime_type, ".bin")

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
        logger.info("WhatsApp client closed")
