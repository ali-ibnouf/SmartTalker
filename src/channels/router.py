"""Channel router — routes messages from any channel to the Agent Engine.

The Agent Engine processes ALL channels identically — same brain, same tools,
same memory. Only the I/O adapter changes per channel.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import httpx

from src.channels.base import ChannelAdapter, ChannelType, IncomingMessage, OutgoingMessage
from src.utils.logger import setup_logger

logger = setup_logger("channels.router")


class ChannelRouter:
    """Routes messages from any channel to Agent Engine."""

    def __init__(
        self,
        agent_engine: Any,
        tts: Any,
        asr: Any,
        runpod: Any,
        r2: Any,
        redis: Any = None,
    ) -> None:
        self.agent = agent_engine
        self.tts = tts
        self.asr = asr
        self.runpod = runpod
        self.r2 = r2
        self.redis = redis
        self._adapters: dict[ChannelType, ChannelAdapter] = {}

    def register_adapter(
        self, channel_type: ChannelType, adapter: ChannelAdapter
    ) -> None:
        """Register a channel adapter."""
        self._adapters[channel_type] = adapter
        logger.info(f"Registered adapter: {channel_type.value}")

    def get_adapter(self, channel_type: ChannelType) -> ChannelAdapter | None:
        """Get the adapter for a channel type."""
        return self._adapters.get(channel_type)

    async def _record_metric(self, key: str) -> None:
        """Increment a Redis counter for agent monitoring (fire-and-forget)."""
        if self.redis is None:
            return
        try:
            hour = datetime.utcnow().strftime("%Y%m%d%H")
            full_key = f"{key}:{hour}"
            await self.redis.incr(full_key)
            await self.redis.expire(full_key, 7200)  # 2 hour TTL
        except Exception:
            pass  # Never fail the main flow for metrics

    async def handle_message(self, message: IncomingMessage) -> OutgoingMessage:
        """Process any incoming message regardless of channel."""
        adapter = self._adapters.get(message.channel)
        if adapter is None:
            raise ValueError(f"No adapter registered for {message.channel.value}")

        # 1. Voice → text (if audio)
        if message.message_type == "voice" and message.audio_bytes:
            asr_session = await self.asr.create_session()
            await asr_session.send_audio(message.audio_bytes)
            result = await asr_session.finish()
            message.text = result.text

        elif message.message_type == "voice" and message.audio_url:
            audio_bytes = await self._download_audio(message.audio_url)
            asr_session = await self.asr.create_session()
            await asr_session.send_audio(audio_bytes)
            result = await asr_session.finish()
            message.text = result.text
            message.audio_bytes = audio_bytes

        # 2. Agent thinks (same engine for all channels)
        agent_response = await self.agent.handle_message(
            session_id=message.channel_session_id,
            visitor_message=message.text or "",
            employee_id=message.employee_id,
            visitor_id=message.visitor_id,
            customer_id=message.customer_id,
        )

        # 3. Build outgoing response (AgentEngine returns a string)
        response_text = str(agent_response)
        outgoing = OutgoingMessage(text=response_text)

        # 4. TTS (if visitor sent voice and employee has voice)
        employee = await self._get_employee(message.employee_id)
        voice_id = getattr(employee, "voice_id", None) if employee else None

        if message.message_type == "voice" and voice_id and self.tts:
            try:
                tts_stream = await self.tts.synthesize_stream(response_text, voice_id)
                audio_bytes = await tts_stream.collect_all()

                # Upload to R2 (WhatsApp/Telegram need URL, widget gets bytes)
                if self.r2:
                    audio_url = await self.r2.upload_audio(
                        message.channel_session_id, audio_bytes
                    )
                    outgoing.audio_url = audio_url

                outgoing.audio_bytes = audio_bytes
            except Exception as exc:
                logger.warning(f"TTS failed, sending text only: {exc}")

        # 5. Video (widget only)
        if (
            message.channel == ChannelType.WIDGET
            and employee
            and getattr(employee, "avatar_mode", "") == "video"
            and getattr(employee, "face_data_url", None)
            and outgoing.audio_bytes
            and outgoing.audio_url
            and self.runpod
        ):
            try:
                render_result = await self.runpod.render_lipsync(
                    audio_url=outgoing.audio_url,
                    face_data_url=employee.face_data_url,
                    employee_id=employee.id,
                    session_id=message.channel_session_id,
                )
                outgoing.video_url = render_result.video_url
            except Exception as exc:
                logger.warning(f"RunPod render failed: {exc}")

        # 7. Send response through the right channel
        try:
            await adapter.send_response(message.channel_session_id, outgoing)
            await self._record_metric(f"route_ok:{message.channel.value}")
        except Exception as exc:
            await self._record_metric(f"route_err:{message.channel.value}")
            await self._record_metric(f"channel_fail:{message.channel.value}:{message.employee_id}")
            logger.error(f"Channel send failed: {exc}")
            raise

        logger.info(
            "Message routed",
            extra={
                "channel": message.channel.value,
                "employee_id": message.employee_id,
                "has_audio": outgoing.audio_url is not None,
                "has_video": outgoing.video_url is not None,
            },
        )

        return outgoing

    async def _download_audio(self, url: str) -> bytes:
        """Download audio from a URL (WhatsApp/Telegram media)."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.content

    async def _get_employee(self, employee_id: str) -> Any:
        """Get employee from agent engine or DB."""
        if hasattr(self.agent, "get_employee"):
            return await self.agent.get_employee(employee_id)
        return None
