"""Multi-channel integration layer.

Routes messages from any channel (Widget, WhatsApp, Telegram)
to the same Agent Engine. Only the I/O adapter changes per channel.
"""

from src.channels.base import ChannelAdapter, ChannelType, IncomingMessage, OutgoingMessage
from src.channels.router import ChannelRouter

__all__ = [
    "ChannelAdapter",
    "ChannelRouter",
    "ChannelType",
    "IncomingMessage",
    "OutgoingMessage",
]
