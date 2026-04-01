"""WhatsApp Cloud API channel adapter.

Uses Meta's WhatsApp Business API (Cloud API) to send/receive messages.
Each customer provides their own WhatsApp Business phone number and credentials
configured via the employee_channels table.

Supported incoming message types:
- text     → passes text to agent
- audio    → ASR transcription → agent
- video    → extract audio → ASR → agent
- image    → OCR (Tesseract + Qwen-VL) → agent
- document → PDF text extraction / OCR → agent
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from typing import Any

import httpx

from src.channels.base import ChannelAdapter, ChannelType, IncomingMessage, OutgoingMessage
from src.channels.visitor_resolver import VisitorResolver
from src.utils.logger import setup_logger

logger = setup_logger("channels.whatsapp")

META_API = "https://graph.facebook.com/v21.0"


class WhatsAppAdapter(ChannelAdapter):
    """WhatsApp Cloud API adapter."""

    def __init__(
        self,
        visitor_resolver: VisitorResolver | None = None,
        r2_storage: Any | None = None,
        ocr_service: Any | None = None,
    ) -> None:
        self._visitor_resolver = visitor_resolver
        self._r2 = r2_storage
        self._ocr = ocr_service

    def supports_voice(self) -> bool:
        return True  # WhatsApp supports voice messages

    def supports_video(self) -> bool:
        return False  # No video avatar in WhatsApp (text + voice only)

    # ── Incoming message parsing ──────────────────────────────────────────

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

        wa_token = getattr(channel_config, "wa_access_token", "")

        if msg_type == "text":
            incoming.text = msg["text"]["body"]
        elif msg_type == "audio":
            media_id = msg["audio"]["id"]
            incoming.message_type = "voice"
            incoming.audio_url = await self._get_media_url(media_id, wa_token)
        elif msg_type == "video":
            await self._handle_video(msg, wa_token, incoming)
        elif msg_type == "image":
            await self._handle_image(msg, wa_token, incoming)
        elif msg_type == "document":
            await self._handle_document(msg, wa_token, incoming)
        else:
            incoming.text = (
                "عذراً، لا أستطيع معالجة هذا النوع من الرسائل. "
                "يرجى ارسال نص أو صوت أو صورة أو مستند."
            )
            logger.info("Unsupported WA message type", extra={"msg_type": msg_type})

        return incoming

    # ── Media handlers ────────────────────────────────────────────────────

    async def _handle_video(
        self, msg: dict[str, Any], wa_token: str, incoming: IncomingMessage
    ) -> None:
        """Handle incoming video: extract audio for ASR transcription."""
        video_data = msg.get("video", {})
        media_id = video_data.get("id")
        if not media_id:
            logger.warning("Video message has no media ID")
            return

        mime_type = video_data.get("mime_type", "video/mp4")

        try:
            video_bytes = await self._download_media(media_id, wa_token)
            logger.info(
                "Video downloaded",
                extra={"size_bytes": len(video_bytes), "mime": mime_type},
            )

            # Archive to R2 (fire-and-forget)
            await self._archive_media(
                incoming.channel_session_id, video_bytes,
                "video", mime_type, self._ext_from_mime(mime_type, "mp4"),
            )

            # Extract audio (sync function, run in executor)
            from src.utils.video import extract_audio_from_bytes

            loop = asyncio.get_running_loop()
            video_format = self._ext_from_mime(mime_type, "mp4")
            audio_bytes = await loop.run_in_executor(
                None, extract_audio_from_bytes, video_bytes, video_format,
            )

            incoming.message_type = "voice"
            incoming.audio_bytes = audio_bytes
            incoming.metadata["media_type"] = "video"
            incoming.metadata["mime_type"] = mime_type
            logger.info("Audio extracted from video", extra={"audio_bytes": len(audio_bytes)})

        except Exception as exc:
            logger.error("Video processing failed", extra={"error": str(exc)})
            incoming.message_type = "text"
            incoming.text = (
                "ارسلت فيديو ولكن لم أتمكن من معالجته. "
                "يرجى ارسال رسالة صوتية أو نصية."
            )
            incoming.metadata["media_error"] = str(exc)

    async def _handle_image(
        self, msg: dict[str, Any], wa_token: str, incoming: IncomingMessage
    ) -> None:
        """Handle incoming image: OCR extraction, return as text."""
        image_data = msg.get("image", {})
        media_id = image_data.get("id")
        if not media_id:
            logger.warning("Image message has no media ID")
            return

        mime_type = image_data.get("mime_type", "image/jpeg")
        caption = image_data.get("caption", "")

        try:
            image_bytes = await self._download_media(media_id, wa_token)
            logger.info(
                "Image downloaded",
                extra={"size_bytes": len(image_bytes), "mime": mime_type},
            )

            # Archive to R2
            r2_url = await self._archive_media(
                incoming.channel_session_id, image_bytes,
                "image", mime_type, self._ext_from_mime(mime_type, "jpg"),
            )

            # OCR
            ocr_text = ""
            if self._ocr:
                image_format = self._ext_from_mime(mime_type, "jpeg")
                if image_format == "jpg":
                    image_format = "jpeg"
                ocr_result = await self._ocr.analyze_image(
                    image_bytes, image_format, extract_fields=True,
                )
                ocr_text = ocr_result.text or ""
                incoming.metadata["ocr_result"] = ocr_result.to_dict()
                logger.info(
                    "Image OCR complete",
                    extra={
                        "doc_type": ocr_result.document_type.value,
                        "confidence": round(ocr_result.confidence, 2),
                        "method": ocr_result.method,
                    },
                )

            # Build context text for the agent
            parts: list[str] = []
            if caption:
                parts.append(caption)
            if ocr_text:
                parts.append(f"[محتوى المستند المرفق]\n{ocr_text[:2000]}")

            incoming.message_type = "text"
            incoming.text = "\n\n".join(parts) if parts else "أرسل الزبون صورة"
            incoming.metadata["media_type"] = "image"
            incoming.metadata["mime_type"] = mime_type
            incoming.metadata["image_bytes"] = image_bytes
            if r2_url:
                incoming.metadata["media_url"] = r2_url

        except Exception as exc:
            logger.error("Image processing failed", extra={"error": str(exc)})
            incoming.message_type = "text"
            incoming.text = caption if caption else (
                "أرسلت صورة ولكن لم أتمكن من قراءتها. يرجى ارسال النص مباشرة."
            )
            incoming.metadata["media_error"] = str(exc)

    async def _handle_document(
        self, msg: dict[str, Any], wa_token: str, incoming: IncomingMessage
    ) -> None:
        """Handle incoming document: extract text from PDF/images."""
        doc_data = msg.get("document", {})
        media_id = doc_data.get("id")
        if not media_id:
            logger.warning("Document message has no media ID")
            return

        mime_type = doc_data.get("mime_type", "application/octet-stream")
        filename = doc_data.get("filename", "document")
        caption = doc_data.get("caption", "")

        try:
            doc_bytes = await self._download_media(media_id, wa_token)
            logger.info(
                "Document downloaded",
                extra={"doc_filename": filename, "size_bytes": len(doc_bytes), "mime": mime_type},
            )

            # Archive to R2
            ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "bin"
            r2_url = await self._archive_media(
                incoming.channel_session_id, doc_bytes,
                "document", mime_type, ext,
            )

            # Extract text based on type
            extracted_text = ""

            if mime_type == "application/pdf" or ext == "pdf":
                loop = asyncio.get_running_loop()
                extracted_text = await loop.run_in_executor(
                    None, self._parse_pdf_bytes, doc_bytes,
                )

            elif mime_type.startswith("image/"):
                if self._ocr:
                    image_format = self._ext_from_mime(mime_type, "jpeg")
                    if image_format == "jpg":
                        image_format = "jpeg"
                    ocr_result = await self._ocr.analyze_image(
                        doc_bytes, image_format, extract_fields=True,
                    )
                    extracted_text = ocr_result.text or ""
                    incoming.metadata["ocr_result"] = ocr_result.to_dict()

            else:
                extracted_text = f"[مستند من نوع {mime_type} — لا يمكن معالجته حالياً]"

            # Build context
            parts: list[str] = []
            if caption:
                parts.append(caption)
            if extracted_text:
                parts.append(f"[محتوى المستند المرفق: {filename}]\n{extracted_text[:2000]}")

            incoming.message_type = "text"
            incoming.text = "\n\n".join(parts) if parts else f"أرسل الزبون مستند: {filename}"
            incoming.metadata["media_type"] = "document"
            incoming.metadata["mime_type"] = mime_type
            incoming.metadata["filename"] = filename
            if r2_url:
                incoming.metadata["media_url"] = r2_url

        except Exception as exc:
            logger.error("Document processing failed", extra={"error": str(exc), "doc_filename": filename})
            incoming.message_type = "text"
            incoming.text = caption if caption else f"أرسلت مستند ({filename}) ولكن لم أتمكن من قراءته."
            incoming.metadata["media_error"] = str(exc)

    # ── Outgoing ──────────────────────────────────────────────────────────

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

    # ── Private helpers ───────────────────────────────────────────────────

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

    async def _download_media(self, media_id: str, access_token: str) -> bytes:
        """Download media bytes from Meta CDN (requires Bearer auth)."""
        url = await self._get_media_url(media_id, access_token)
        if not url:
            raise ValueError(f"Could not get media URL for {media_id}")
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                url, headers={"Authorization": f"Bearer {access_token}"},
            )
            resp.raise_for_status()
            return resp.content

    async def _archive_media(
        self,
        session_id: str,
        media_bytes: bytes,
        media_type: str,
        mime_type: str,
        ext: str,
    ) -> str | None:
        """Upload media to R2 for supervisor reference (non-blocking)."""
        if self._r2 is None:
            return None
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None,
                self._r2.upload_incoming_media,
                session_id, media_bytes, media_type, mime_type, ext,
            )
        except Exception as exc:
            logger.warning("R2 archive failed (non-blocking)", extra={"error": str(exc)})
            return None

    @staticmethod
    def _ext_from_mime(mime_type: str, default: str = "bin") -> str:
        """Extract file extension from MIME type."""
        parts = mime_type.split("/")
        if len(parts) == 2:
            sub = parts[1].split(";")[0].strip()
            if sub and sub != "*":
                return sub
        return default

    @staticmethod
    def _parse_pdf_bytes(pdf_bytes: bytes) -> str:
        """Parse PDF from bytes. Sync method for use in executor."""
        fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        try:
            with open(tmp_path, "wb") as f:
                f.write(pdf_bytes)

            from PyPDF2 import PdfReader

            reader = PdfReader(tmp_path)
            pages: list[str] = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
            return "\n\n".join(pages)
        except ImportError:
            logger.warning("PyPDF2 not installed — cannot extract PDF text")
            return "[لم يتمكن من استخراج النص — PyPDF2 غير مثبت]"
        except Exception as exc:
            logger.warning("PDF parsing failed", extra={"error": str(exc)})
            return "[لم يتمكن من استخراج النص من الملف]"
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    async def _get_channel_config(
        self, employee_id: str, channel_type: str
    ) -> Any:
        """Look up channel config from DB. Returns None if not found."""
        # This is resolved by the webhook handler which passes the config.
        # For send_response, we need to look it up again.
        # In practice, the webhook handler caches this on the adapter.
        return getattr(self, "_cached_config", None)
