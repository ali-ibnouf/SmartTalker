"""Widget channel adapter — wraps the existing WebSocket visitor session."""

from __future__ import annotations

from typing import Any

from src.channels.base import ChannelAdapter, ChannelType, IncomingMessage, OutgoingMessage
from src.utils.logger import setup_logger

logger = setup_logger("channels.widget")


class WidgetAdapter(ChannelAdapter):
    """Website widget adapter (WebSocket-based).

    The widget communicates over WebSocket at /session.
    Audio is streamed as base64 PCM chunks; video is a RunPod-rendered URL.
    """

    def supports_voice(self) -> bool:
        return True

    def supports_video(self) -> bool:
        return True  # Widget supports video avatar rendering

    async def parse_incoming(
        self, raw_data: dict[str, Any], channel_config: Any
    ) -> IncomingMessage:
        """Parse a WebSocket message into unified format.

        For widget, raw_data is the parsed JSON message from the WS connection.
        """
        msg_type = raw_data.get("type", "text_message")
        employee_id = raw_data.get("employee_id", "")
        session_id = raw_data.get("session_id", "")

        incoming = IncomingMessage(
            channel=ChannelType.WIDGET,
            channel_session_id=session_id,
            employee_id=employee_id,
            customer_id=getattr(channel_config, "customer_id", ""),
            visitor_id=raw_data.get("visitor_id", session_id),
            message_type="text",
            metadata={"ws_type": msg_type},
        )

        if msg_type == "text_message":
            incoming.text = raw_data.get("text", "")
        elif msg_type in ("audio_chunk", "audio_end"):
            incoming.message_type = "voice"
            import base64
            audio_b64 = raw_data.get("audio", "")
            if audio_b64:
                incoming.audio_bytes = base64.b64decode(audio_b64)

        return incoming

    async def send_response(
        self, channel_session_id: str, response: OutgoingMessage
    ) -> None:
        """Send response back through the WebSocket.

        Note: The actual WebSocket send is handled by ws_visitor.py.
        This adapter prepares the response format; the WS handler
        calls send_json() directly. For webhook-triggered flows,
        the response is stored for pickup by the WS connection.
        """
        # Widget responses are streamed via the existing WS handler.
        # This method is a no-op for direct WS sessions — the router
        # returns the OutgoingMessage and ws_visitor.py sends it.
        logger.debug(
            "Widget response prepared",
            extra={"session_id": channel_session_id, "has_video": response.video_url is not None},
        )
