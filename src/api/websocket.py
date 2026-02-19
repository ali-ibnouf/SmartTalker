"""WebSocket handler for real-time bidirectional chat.

Manages WebSocket connections, audio streaming, and
real-time pipeline processing per avatar session.

Protocol (JSON control frames + binary audio):

Client -> Server:
    {"type": "text_chat", "text": "...", "language": "ar", "emotion": "neutral"}
    {"type": "audio_start", "format": "wav", "language": "ar"}
    <binary audio chunks>
    {"type": "audio_end"}
    {"type": "config", "avatar_id": "...", "voice_id": "...", "enable_video": false}
    {"type": "ping"}

Server -> Client:
    {"type": "session_init", "session_id": "...", "message": "Connected"}
    {"type": "text_response", "text": "...", "emotion": "...", "audio_url": "...", ...}
    {"type": "audio_ack", "bytes_received": N}
    {"type": "error", "error": "...", "detail": "..."}
    {"type": "pong"}
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from src.pipeline.orchestrator import SmartTalkerPipeline
from src.utils.exceptions import WebSocketError
from src.utils.logger import setup_logger, set_correlation_id

logger = setup_logger("api.websocket")

# Limits
_MAX_AUDIO_BYTES = 25 * 1024 * 1024  # 25 MB max audio upload
_MAX_TEXT_LENGTH = 2000
_AUDIO_TIMEOUT_S = 60  # Max seconds between audio_start and audio_end
_MAX_CONNECTIONS_PER_IP = 5


@dataclass
class SessionConfig:
    """Per-session configuration that clients can update mid-session.

    Attributes:
        avatar_id: Avatar identifier for video generation.
        voice_id: Optional voice clone ID for TTS.
        language: Target response language code.
        enable_video: Whether to generate video output.
    """

    avatar_id: str = "default"
    voice_id: Optional[str] = None
    language: str = "ar"
    enable_video: bool = False


@dataclass
class AudioBuffer:
    """Accumulates binary audio chunks during streaming.

    Attributes:
        chunks: List of raw byte chunks received.
        format: Audio format (wav, mp3, ogg, webm).
        language: Language hint for ASR.
        started_at: Timestamp when audio_start was received.
    """

    chunks: list[bytes] = field(default_factory=list)
    format: str = "wav"
    language: str = "ar"
    started_at: float = 0.0

    @property
    def total_bytes(self) -> int:
        return sum(len(c) for c in self.chunks)

    def reset(self) -> None:
        self.chunks.clear()
        self.started_at = 0.0


@dataclass
class Session:
    """Represents one active WebSocket session.

    Attributes:
        session_id: Unique identifier for this session.
        websocket: The FastAPI WebSocket connection.
        config: Mutable session configuration.
        audio_buffer: Audio accumulator for streaming mode.
        connected_at: Timestamp of connection establishment.
        client_ip: Remote client IP address.
        is_recording: Whether audio streaming is in progress.
    """

    session_id: str
    websocket: WebSocket
    config: SessionConfig = field(default_factory=SessionConfig)
    audio_buffer: AudioBuffer = field(default_factory=AudioBuffer)
    connected_at: float = field(default_factory=time.time)
    client_ip: str = "unknown"
    is_recording: bool = False


class WebSocketManager:
    """Manages active WebSocket sessions and connection lifecycle.

    Tracks all connected clients, enforces per-IP connection limits,
    and provides broadcast/cleanup capabilities.

    Args:
        pipeline: The SmartTalkerPipeline instance for processing.
        storage_dir: Directory for temporary audio files.
        api_key: Optional API key for WebSocket authentication.
    """

    def __init__(
        self,
        pipeline: SmartTalkerPipeline,
        storage_dir: Path,
        api_key: Optional[str] = None,
    ) -> None:
        self._pipeline = pipeline
        self._storage_dir = storage_dir
        self._api_key = api_key
        self._sessions: dict[str, Session] = {}
        self._ip_counts: dict[str, int] = {}
        self._lock = asyncio.Lock()

        self._storage_dir.mkdir(parents=True, exist_ok=True)

    @property
    def active_count(self) -> int:
        """Number of active WebSocket sessions."""
        return len(self._sessions)

    async def connect(
        self, websocket: WebSocket, already_accepted: bool = False,
    ) -> Session:
        """Accept a WebSocket connection and create a session.

        Args:
            websocket: Incoming WebSocket connection.
            already_accepted: If True, skip the accept() call (already done for auth).

        Returns:
            The newly created Session.

        Raises:
            WebSocketError: If the per-IP connection limit is exceeded.
        """
        client_ip = websocket.client.host if websocket.client else "unknown"

        async with self._lock:
            ip_count = self._ip_counts.get(client_ip, 0)
            if ip_count >= _MAX_CONNECTIONS_PER_IP:
                await websocket.close(
                    code=1008,
                    reason="Too many connections from this IP",
                )
                raise WebSocketError(
                    message="Connection limit exceeded",
                    detail={"ip": client_ip, "limit": _MAX_CONNECTIONS_PER_IP},
                )

            if not already_accepted:
                await websocket.accept()

            session_id = uuid.uuid4().hex[:16]
            session = Session(
                session_id=session_id,
                websocket=websocket,
                client_ip=client_ip,
            )

            self._sessions[session_id] = session
            self._ip_counts[client_ip] = ip_count + 1

        logger.info(
            "WebSocket connected",
            extra={
                "session_id": session_id,
                "client_ip": client_ip,
                "active_sessions": self.active_count,
            },
        )

        # Send session initialization
        await self._send_json(websocket, {
            "type": "session_init",
            "session_id": session_id,
            "message": "Connected to SmartTalker",
        })

        return session

    async def disconnect(self, session: Session) -> None:
        """Clean up a disconnected session.

        Args:
            session: The session to remove.
        """
        async with self._lock:
            self._sessions.pop(session.session_id, None)
            ip_count = self._ip_counts.get(session.client_ip, 1)
            if ip_count <= 1:
                self._ip_counts.pop(session.client_ip, None)
            else:
                self._ip_counts[session.client_ip] = ip_count - 1

        # Clean up any in-progress audio buffer
        session.audio_buffer.reset()

        logger.info(
            "WebSocket disconnected",
            extra={
                "session_id": session.session_id,
                "client_ip": session.client_ip,
                "active_sessions": self.active_count,
            },
        )

    async def handle_session(self, session: Session) -> None:
        """Main receive loop for a WebSocket session.

        Dispatches incoming messages to the appropriate handler
        based on message type. Runs until the client disconnects.

        Args:
            session: The active session to handle.
        """
        set_correlation_id(session.session_id)

        try:
            while True:
                message = await session.websocket.receive()

                if message["type"] == "websocket.disconnect":
                    break

                if "bytes" in message and message["bytes"]:
                    await self._handle_audio_chunk(session, message["bytes"])
                elif "text" in message and message["text"]:
                    await self._handle_text_message(session, message["text"])

        except WebSocketDisconnect:
            pass
        except Exception as exc:
            logger.error(
                f"Session error: {exc}",
                extra={"session_id": session.session_id},
            )
            await self._send_error(
                session.websocket,
                "Internal server error",
            )

    # ═════════════════════════════════════════════════════════════════════
    # Message Dispatching
    # ═════════════════════════════════════════════════════════════════════

    async def _handle_text_message(self, session: Session, raw: str) -> None:
        """Parse and dispatch a JSON text message.

        Args:
            session: The source session.
            raw: Raw JSON string from the client.
        """
        try:
            import json
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            await self._send_error(
                session.websocket,
                "Invalid JSON",
                "Message must be valid JSON",
            )
            return

        msg_type = data.get("type", "")

        handlers = {
            "text_chat": self._handle_text_chat,
            "audio_start": self._handle_audio_start,
            "audio_end": self._handle_audio_end,
            "config": self._handle_config_update,
            "ping": self._handle_ping,
        }

        handler = handlers.get(msg_type)
        if not handler:
            await self._send_error(
                session.websocket,
                "Unknown message type",
                f"Unsupported type: '{msg_type}'. "
                f"Valid types: {', '.join(handlers.keys())}",
            )
            return

        await handler(session, data)

    # ═════════════════════════════════════════════════════════════════════
    # Text Chat
    # ═════════════════════════════════════════════════════════════════════

    async def _handle_text_chat(self, session: Session, data: dict) -> None:
        """Process a text chat message through the pipeline.

        Args:
            session: The source session.
            data: Parsed message with "text", optional "language", "emotion".
        """
        text = data.get("text", "").strip()
        if not text:
            await self._send_error(
                session.websocket,
                "Empty text",
                "The 'text' field is required and cannot be empty",
            )
            return

        if len(text) > _MAX_TEXT_LENGTH:
            await self._send_error(
                session.websocket,
                "Text too long",
                f"Maximum {_MAX_TEXT_LENGTH} characters allowed",
            )
            return

        language = data.get("language", session.config.language)
        emotion = data.get("emotion", "neutral")

        logger.info(
            "Processing text chat",
            extra={
                "session_id": session.session_id,
                "text_length": len(text),
                "language": language,
            },
        )

        try:
            result = await self._pipeline.process_text(
                text=text,
                avatar_id=session.config.avatar_id,
                voice_id=session.config.voice_id,
                emotion=emotion,
                language=language,
                enable_video=session.config.enable_video,
                session_id=session.session_id,
            )

            await self._send_json(session.websocket, {
                "type": "text_response",
                "text": result.response_text,
                "emotion": result.detected_emotion,
                "audio_url": f"/files/{Path(result.audio_path).name}" if result.audio_path else None,
                "video_url": f"/files/{Path(result.video_path).name}" if result.video_path else None,
                "latency_ms": result.total_latency_ms,
                "breakdown": result.breakdown,
            })

        except Exception as exc:
            logger.error(
                f"Text chat pipeline error: {exc}",
                extra={"session_id": session.session_id},
            )
            await self._send_error(
                session.websocket,
                "Pipeline processing failed",
            )

    # ═════════════════════════════════════════════════════════════════════
    # Audio Streaming
    # ═════════════════════════════════════════════════════════════════════

    async def _handle_audio_start(self, session: Session, data: dict) -> None:
        """Begin audio streaming — initialize the audio buffer.

        Args:
            session: The source session.
            data: Parsed message with optional "format" and "language".
        """
        if session.is_recording:
            await self._send_error(
                session.websocket,
                "Already recording",
                "Send 'audio_end' before starting a new recording",
            )
            return

        session.audio_buffer.reset()
        session.audio_buffer.format = data.get("format", "wav")
        session.audio_buffer.language = data.get("language", session.config.language)
        session.audio_buffer.started_at = time.time()
        session.is_recording = True

        logger.info(
            "Audio recording started",
            extra={
                "session_id": session.session_id,
                "format": session.audio_buffer.format,
            },
        )

    async def _handle_audio_chunk(self, session: Session, chunk: bytes) -> None:
        """Accumulate a binary audio chunk.

        Args:
            session: The source session.
            chunk: Raw audio bytes from the client.
        """
        if not session.is_recording:
            await self._send_error(
                session.websocket,
                "Not recording",
                "Send 'audio_start' before sending audio data",
            )
            return

        # Check timeout
        elapsed = time.time() - session.audio_buffer.started_at
        if elapsed > _AUDIO_TIMEOUT_S:
            session.is_recording = False
            session.audio_buffer.reset()
            await self._send_error(
                session.websocket,
                "Audio timeout",
                f"Recording exceeded {_AUDIO_TIMEOUT_S}s limit",
            )
            return

        # Check size limit
        if session.audio_buffer.total_bytes + len(chunk) > _MAX_AUDIO_BYTES:
            session.is_recording = False
            session.audio_buffer.reset()
            await self._send_error(
                session.websocket,
                "Audio too large",
                f"Maximum {_MAX_AUDIO_BYTES // (1024 * 1024)}MB allowed",
            )
            return

        session.audio_buffer.chunks.append(chunk)

    async def _handle_audio_end(self, session: Session, data: dict) -> None:
        """Finalize audio streaming and process through the pipeline.

        Saves accumulated audio to a temp file, runs the audio pipeline,
        then sends the response back to the client.

        Args:
            session: The source session.
            data: Parsed message (no additional fields needed).
        """
        if not session.is_recording:
            await self._send_error(
                session.websocket,
                "Not recording",
                "No active recording to finalize",
            )
            return

        session.is_recording = False
        buffer = session.audio_buffer

        if not buffer.chunks:
            buffer.reset()
            await self._send_error(
                session.websocket,
                "Empty audio",
                "No audio data received between audio_start and audio_end",
            )
            return

        total_bytes = buffer.total_bytes
        # Capture buffer metadata before reset() clears state
        audio_language = buffer.language
        audio_format = buffer.format

        # Acknowledge receipt
        await self._send_json(session.websocket, {
            "type": "audio_ack",
            "bytes_received": total_bytes,
        })

        # Save to temp file
        ext = f".{audio_format}" if not audio_format.startswith(".") else audio_format
        temp_path = self._storage_dir / f"ws_{session.session_id}_{uuid.uuid4().hex[:8]}{ext}"

        try:
            audio_data = b"".join(buffer.chunks)
            temp_path.write_bytes(audio_data)
        except OSError as exc:
            buffer.reset()
            await self._send_error(
                session.websocket,
                "Failed to save audio",
                str(exc),
            )
            return
        finally:
            buffer.reset()

        logger.info(
            "Audio received, processing",
            extra={
                "session_id": session.session_id,
                "bytes": total_bytes,
                "format": ext,
            },
        )

        # Process through pipeline
        try:
            result = await self._pipeline.process_audio(
                audio_path=str(temp_path),
                avatar_id=session.config.avatar_id,
                voice_id=session.config.voice_id,
                emotion="neutral",
                language=audio_language,
                enable_video=session.config.enable_video,
                session_id=session.session_id,
            )

            await self._send_json(session.websocket, {
                "type": "text_response",
                "text": result.response_text,
                "emotion": result.detected_emotion,
                "audio_url": f"/files/{Path(result.audio_path).name}" if result.audio_path else None,
                "video_url": f"/files/{Path(result.video_path).name}" if result.video_path else None,
                "latency_ms": result.total_latency_ms,
                "breakdown": result.breakdown,
            })

        except Exception as exc:
            logger.error(
                f"Audio pipeline error: {exc}",
                extra={"session_id": session.session_id},
            )
            await self._send_error(
                session.websocket,
                "Pipeline processing failed",
            )
        finally:
            # Clean up temp file
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass

    # ═════════════════════════════════════════════════════════════════════
    # Config & Ping
    # ═════════════════════════════════════════════════════════════════════

    async def _handle_config_update(self, session: Session, data: dict) -> None:
        """Update session configuration mid-session.

        Args:
            session: The source session.
            data: Parsed message with optional config fields.
        """
        cfg = session.config

        if "avatar_id" in data:
            cfg.avatar_id = str(data["avatar_id"])
        if "voice_id" in data:
            cfg.voice_id = data["voice_id"]
        if "language" in data:
            cfg.language = str(data["language"])
        if "enable_video" in data:
            cfg.enable_video = bool(data["enable_video"])

        logger.info(
            "Session config updated",
            extra={
                "session_id": session.session_id,
                "avatar_id": cfg.avatar_id,
                "voice_id": cfg.voice_id,
                "language": cfg.language,
                "enable_video": cfg.enable_video,
            },
        )

        await self._send_json(session.websocket, {
            "type": "config_ack",
            "avatar_id": cfg.avatar_id,
            "voice_id": cfg.voice_id,
            "language": cfg.language,
            "enable_video": cfg.enable_video,
        })

    async def _handle_ping(self, session: Session, data: dict) -> None:
        """Respond to a client ping with a pong.

        Args:
            session: The source session.
            data: Parsed message (no additional fields needed).
        """
        await self._send_json(session.websocket, {
            "type": "pong",
            "timestamp": time.time(),
        })

    # ═════════════════════════════════════════════════════════════════════
    # Send Helpers
    # ═════════════════════════════════════════════════════════════════════

    @staticmethod
    async def _send_json(websocket: WebSocket, data: dict[str, Any]) -> None:
        """Send a JSON message to a WebSocket client.

        Logs send failures at debug level instead of silently swallowing.

        Args:
            websocket: Target WebSocket connection.
            data: Dictionary to serialize and send.
        """
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_json(data)
        except Exception as exc:
            logger.debug(f"WebSocket send failed: {exc}")

    @staticmethod
    async def _send_error(
        websocket: WebSocket,
        error: str,
        detail: Optional[str] = None,
    ) -> None:
        """Send an error message to a WebSocket client.

        Args:
            websocket: Target WebSocket connection.
            error: Short error description.
            detail: Additional context about the error.
        """
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                msg: dict[str, Any] = {"type": "error", "error": error}
                if detail:
                    msg["detail"] = detail
                await websocket.send_json(msg)
        except Exception as exc:
            logger.debug(f"WebSocket error send failed: {exc}")


# ═════════════════════════════════════════════════════════════════════════
# WebSocket Endpoint (mounted by main.py)
# ═════════════════════════════════════════════════════════════════════════


async def _authenticate_websocket(
    websocket: WebSocket,
    api_key: str,
) -> bool:
    """Authenticate a WebSocket connection.

    Checks query param first, then waits for an auth message (5s timeout).

    Args:
        websocket: The accepted WebSocket connection.
        api_key: The expected API key.

    Returns:
        True if authenticated successfully.
    """
    import json as _json

    # Check query parameter first
    query_key = websocket.query_params.get("api_key")
    if query_key == api_key:
        return True

    # Wait for auth message with 5 second timeout
    try:
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
        data = _json.loads(raw)
        if data.get("type") == "auth" and data.get("api_key") == api_key:
            return True
    except (asyncio.TimeoutError, _json.JSONDecodeError, TypeError):
        pass

    return False


async def websocket_chat_endpoint(
    websocket: WebSocket,
    manager: WebSocketManager,
) -> None:
    """FastAPI WebSocket endpoint for real-time chat.

    Handles the full connection lifecycle: authenticate, accept,
    process messages in a loop, and clean up on disconnect.

    Args:
        websocket: The incoming WebSocket connection.
        manager: The WebSocketManager instance.
    """
    # Auth check before connecting through manager
    if manager._api_key:
        # Accept first so we can exchange messages for auth
        await websocket.accept()
        authenticated = await _authenticate_websocket(websocket, manager._api_key)
        if not authenticated:
            await websocket.close(code=4003, reason="Authentication required")
            logger.warning(
                "WebSocket auth failed",
                extra={"client": websocket.client.host if websocket.client else "unknown"},
            )
            return
        # Pass to manager with already_accepted=True
        session: Optional[Session] = None
        try:
            session = await manager.connect(websocket, already_accepted=True)
            await manager.handle_session(session)
        except WebSocketError as exc:
            logger.warning(f"WebSocket rejected: {exc.message}")
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            logger.error(f"Unexpected WebSocket error: {exc}")
        finally:
            if session:
                await manager.disconnect(session)
    else:
        # No auth required (dev mode)
        session = None
        try:
            session = await manager.connect(websocket)
            await manager.handle_session(session)
        except WebSocketError as exc:
            logger.warning(f"WebSocket rejected: {exc.message}")
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            logger.error(f"Unexpected WebSocket error: {exc}")
        finally:
            if session:
                await manager.disconnect(session)
