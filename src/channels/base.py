"""Channel abstraction — unified message types and adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ChannelType(Enum):
    WIDGET = "widget"
    WHATSAPP = "whatsapp"
    TELEGRAM = "telegram"


@dataclass
class IncomingMessage:
    """Unified message from any channel."""

    channel: ChannelType
    channel_session_id: str       # channel-specific ID (wa phone, tg chat_id, ws session)
    employee_id: str
    customer_id: str
    visitor_id: str               # unified visitor ID across channels
    message_type: str             # "text", "voice", "image"
    text: str | None = None
    audio_bytes: bytes | None = None
    audio_url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OutgoingMessage:
    """Unified response to any channel."""

    text: str
    audio_bytes: bytes | None = None
    audio_url: str | None = None      # R2 URL for audio
    video_url: str | None = None      # R2 URL for video (widget only)
    action_required: dict[str, Any] | None = None  # confirmation buttons
    quick_replies: list[str] | None = None          # suggested replies


class ChannelAdapter(ABC):
    """Base adapter — every channel implements this."""

    @abstractmethod
    async def send_response(
        self, channel_session_id: str, response: OutgoingMessage
    ) -> None:
        """Send response back to visitor through this channel."""

    @abstractmethod
    async def parse_incoming(
        self, raw_data: dict[str, Any], channel_config: Any
    ) -> IncomingMessage:
        """Parse channel-specific webhook/message into unified format."""

    @abstractmethod
    def supports_voice(self) -> bool:
        """Whether this channel supports voice messages."""

    @abstractmethod
    def supports_video(self) -> bool:
        """Whether this channel supports video responses."""
