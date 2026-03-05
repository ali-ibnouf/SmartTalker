"""WebSocket handler for real-time bidirectional chat.

Manages WebSocket connections, audio streaming, and
real-time pipeline processing per avatar session.

Protocol (JSON control frames + binary audio):

Client -> Server:
    {"type": "text_chat", "text": "...", "language": "ar", "emotion": "neutral"}
    {"type": "audio_start", "format": "wav", "language": "ar"}
    <binary audio chunks>
    {"type": "audio_end"}
    {"type": "config", "avatar_id": "...", "voice_id": "...", "language": "ar", "training_mode": "digital"}
    {"type": "training_mode", "mode": "digital|human"}
    {"type": "set_state", "state": "idle|thinking|talking_happy|talking_sad"}
    {"type": "stop"}
    {"type": "ping"}

Server -> Client:
    {"type": "session_init", "session_id": "...", "message": "Connected"}
    {"type": "thinking"}
    {"type": "body_state", "state": "...", "clip_url": "..."}
    {"type": "voice", "audio_url": "...", "lip_sync": {...}}
    {"type": "response", "text": "...", "emotion": "...", "latency_ms": N, "breakdown": {...},
             "kb_confidence": 0.0, "kb_sources": [], "escalated": false, "escalation_id": null}
    {"type": "escalation_alert", "escalation_id": "...", "question": "...", "kb_confidence": 0.0}
    {"type": "awaiting_operator", "text": "...", "session_id": "..."}
    {"type": "training_mode_ack", "mode": "digital|human"}
    {"type": "audio_ack", "bytes_received": N}
    {"type": "error", "error": "...", "detail": "..."}
    {"type": "pong"}
"""

from __future__ import annotations

import asyncio
import base64
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
    """

    avatar_id: str = "default"
    voice_id: Optional[str] = None
    language: str = "ar"
    training_mode: str = "digital"  # "digital" = AI answers, "human" = operator answers
    avatar_type: str = "video"  # "video" = RunPod rendered, "vrm" = direct browser streaming


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
    ai_paused: bool = False
    operator_id: Optional[str] = None
    takeover_at: Optional[str] = None


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
        self._operator_manager: Optional[Any] = None

        self._storage_dir.mkdir(parents=True, exist_ok=True)

    def set_operator_manager(self, operator_manager: Any) -> None:
        """Set the operator manager for relaying messages."""
        self._operator_manager = operator_manager

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

        # Notify operators that this session ended
        if self._operator_manager:
            await self._operator_manager.notify_session_ended(session.session_id)

        logger.info(
            "WebSocket disconnected",
            extra={
                "session_id": session.session_id,
                "client_ip": session.client_ip,
                "active_sessions": self.active_count,
            },
        )

    _MAX_TEXT_MSG = 256 * 1024   # 256 KB for JSON text messages
    _MAX_AUDIO_MSG = 1024 * 1024  # 1 MB for binary audio chunks

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
                    if len(message["bytes"]) > self._MAX_AUDIO_MSG:
                        await self._send_error(session.websocket, "Audio chunk too large (1MB limit)")
                        continue
                    await self._handle_audio_chunk(session, message["bytes"])
                elif "text" in message and message["text"]:
                    if len(message["text"]) > self._MAX_TEXT_MSG:
                        await self._send_error(session.websocket, "Message too large (256KB limit)")
                        continue
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
            "set_state": self._handle_set_state,
            "stop": self._handle_stop,
            "training_mode": self._handle_training_mode,
            "video_frame": self._handle_video_frame,
            "document_upload": self._handle_document_upload,
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
            # Relay user message to operator
            if self._operator_manager:
                await self._operator_manager.relay_chat_message(
                    session.session_id, "user", text,
                )

            # In human training mode or when AI is paused (operator takeover),
            # relay to operator and wait — don't run pipeline
            if session.config.training_mode == "human" or session.ai_paused:
                reason = "operator_takeover" if session.ai_paused else "human_training_mode"
                await self._send_json(session.websocket, {
                    "type": "awaiting_operator",
                    "text": text,
                    "session_id": session.session_id,
                })
                if self._operator_manager:
                    await self._operator_manager.relay_escalation(
                        session.session_id,
                        text,
                        reason=reason,
                    )
                return

            # 1. Signal thinking state
            await self._send_json(session.websocket, {"type": "thinking"})

            result = await self._pipeline.process_text(
                text=text,
                avatar_id=session.config.avatar_id,
                voice_id=session.config.voice_id,
                emotion=emotion,
                language=language,
                session_id=session.session_id,
            )

            # VRM mode: stream audio + visemes directly to browser
            if session.config.avatar_type == "vrm":
                await self._send_vrm_response(session, result)
            else:
                # 2. Send body state (which clip to play)
                clip_url = f"/clips/{session.config.avatar_id}/{result.body_state}.mp4"
                await self._send_json(session.websocket, {
                    "type": "body_state",
                    "state": result.body_state,
                    "clip_url": clip_url,
                })

                # 3. Send voice audio + lip sync
                audio_url = f"/files/{Path(result.audio_path).name}" if result.audio_path else None
                await self._send_json(session.websocket, {
                    "type": "voice",
                    "audio_url": audio_url,
                    "lip_sync": result.lip_sync,
                })

            # 4. Send text response with KB/escalation fields
            response_msg = {
                "type": "response",
                "text": result.response_text,
                "emotion": result.detected_emotion,
                "latency_ms": result.total_latency_ms,
                "breakdown": result.breakdown,
                "kb_confidence": getattr(result, "kb_confidence", 0.0),
                "kb_sources": getattr(result, "kb_sources", []),
                "escalated": getattr(result, "escalated", False),
                "escalation_id": getattr(result, "escalation_id", None),
            }
            await self._send_json(session.websocket, response_msg)

            # 5. Send escalation alert if needed
            if getattr(result, "escalated", False):
                await self._send_json(session.websocket, {
                    "type": "escalation_alert",
                    "escalation_id": result.escalation_id,
                    "question": text,
                    "kb_confidence": result.kb_confidence,
                })
                if self._operator_manager:
                    await self._operator_manager.relay_escalation(
                        session.session_id,
                        text,
                        reason="low_confidence",
                        escalation_id=result.escalation_id,
                        confidence=result.kb_confidence,
                    )

            # Relay bot response to operator
            if self._operator_manager:
                await self._operator_manager.relay_chat_message(
                    session.session_id, "bot", result.response_text,
                    {"emotion": result.detected_emotion, "latency_ms": result.total_latency_ms},
                )

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
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, temp_path.write_bytes, audio_data)
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
            # Signal thinking state
            await self._send_json(session.websocket, {"type": "thinking"})

            result = await self._pipeline.process_audio(
                audio_path=str(temp_path),
                avatar_id=session.config.avatar_id,
                voice_id=session.config.voice_id,
                emotion="neutral",
                language=audio_language,
                session_id=session.session_id,
            )

            # VRM mode: stream audio + visemes directly to browser
            if session.config.avatar_type == "vrm":
                await self._send_vrm_response(session, result)
            else:
                # Send body state
                clip_url = f"/clips/{session.config.avatar_id}/{result.body_state}.mp4"
                await self._send_json(session.websocket, {
                    "type": "body_state",
                    "state": result.body_state,
                    "clip_url": clip_url,
                })

                # Send voice audio + lip sync
                audio_url = f"/files/{Path(result.audio_path).name}" if result.audio_path else None
                await self._send_json(session.websocket, {
                    "type": "voice",
                    "audio_url": audio_url,
                    "lip_sync": result.lip_sync,
                })

            # Send text response
            await self._send_json(session.websocket, {
                "type": "response",
                "text": result.response_text,
                "emotion": result.detected_emotion,
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
    # VRM Direct Streaming
    # ═════════════════════════════════════════════════════════════════════

    async def _send_vrm_response(self, session: Session, result: Any) -> None:
        """Stream audio chunks with viseme data directly to browser for VRM rendering.

        Instead of routing through a RenderNode, this sends PCM audio + timed
        viseme frames so the browser can drive its own VRM lip-sync animation.

        Args:
            session: The active WebSocket session.
            result: PipelineResult from the orchestrator.
        """
        from src.pipeline.visemes import VisemeExtractor

        audio_path = result.audio_path
        if not audio_path:
            await self._send_json(session.websocket, {"type": "vrm_audio_end"})
            return

        # Read the full audio file (PCM 16-bit mono 22050Hz WAV)
        try:
            audio_data = Path(audio_path).read_bytes()
        except (OSError, FileNotFoundError):
            await self._send_json(session.websocket, {"type": "vrm_audio_end"})
            return

        # Skip WAV header (44 bytes) if present
        if audio_data[:4] == b"RIFF":
            audio_data = audio_data[44:]

        if not audio_data:
            await self._send_json(session.websocket, {"type": "vrm_audio_end"})
            return

        # Split into ~200ms chunks (22050 samples/s * 2 bytes * 0.2s = 8820 bytes)
        sample_rate = 22050
        bytes_per_sample = 2
        chunk_duration_s = 0.2
        chunk_size = int(sample_rate * bytes_per_sample * chunk_duration_s)

        # Get word-level timings from TTS if available
        word_timings = []
        lip_sync = getattr(result, "lip_sync", None)
        if lip_sync and isinstance(lip_sync, dict):
            word_timings = lip_sync.get("words", [])

        total_bytes = len(audio_data)
        total_duration_ms = int(total_bytes / (sample_rate * bytes_per_sample) * 1000)

        seq = 0
        for offset in range(0, total_bytes, chunk_size):
            chunk = audio_data[offset:offset + chunk_size]
            chunk_dur_ms = int(len(chunk) / (sample_rate * bytes_per_sample) * 1000)
            chunk_start_ms = int(offset / (sample_rate * bytes_per_sample) * 1000)
            chunk_end_ms = chunk_start_ms + chunk_dur_ms

            # Compute visemes for this time window
            if word_timings:
                visemes = VisemeExtractor.extract_from_word_timings(
                    word_timings, chunk_start_ms, chunk_end_ms,
                )
            else:
                # Fallback: estimate from text proportional to chunk position
                text = result.response_text or ""
                text_frac = len(chunk) / max(total_bytes, 1)
                char_start = int(len(text) * offset / max(total_bytes, 1))
                char_end = int(len(text) * (offset + len(chunk)) / max(total_bytes, 1))
                chunk_text = text[char_start:char_end]
                visemes = VisemeExtractor.extract_timed(chunk_text, chunk_dur_ms)

            await self._send_json(session.websocket, {
                "type": "vrm_audio",
                "seq": seq,
                "audio_b64": base64.b64encode(chunk).decode("ascii"),
                "duration_ms": chunk_dur_ms,
                "visemes": [
                    {"time_ms": v.start_ms, "viseme": v.viseme, "weight": v.weight}
                    for v in visemes
                ],
                "emotion": getattr(result, "detected_emotion", "neutral"),
            })
            seq += 1

        # Signal end of VRM audio stream
        await self._send_json(session.websocket, {"type": "vrm_audio_end"})

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
        if "training_mode" in data:
            mode = str(data["training_mode"])
            if mode in ("digital", "human"):
                cfg.training_mode = mode
        if "avatar_type" in data:
            atype = str(data["avatar_type"])
            if atype in ("video", "vrm"):
                cfg.avatar_type = atype

        logger.info(
            "Session config updated",
            extra={
                "session_id": session.session_id,
                "avatar_id": cfg.avatar_id,
                "voice_id": cfg.voice_id,
                "language": cfg.language,
                "training_mode": cfg.training_mode,
                "avatar_type": cfg.avatar_type,
            },
        )

        await self._send_json(session.websocket, {
            "type": "config_ack",
            "avatar_id": cfg.avatar_id,
            "voice_id": cfg.voice_id,
            "language": cfg.language,
            "training_mode": cfg.training_mode,
            "avatar_type": cfg.avatar_type,
        })

    async def _handle_training_mode(self, session: Session, data: dict) -> None:
        """Switch between digital (AI) and human (operator) training mode.

        Args:
            session: The source session.
            data: Parsed message with "mode" field ("digital" or "human").
        """
        mode = data.get("mode", "")
        if mode not in ("digital", "human"):
            await self._send_error(
                session.websocket,
                "Invalid training mode",
                "Mode must be 'digital' or 'human'",
            )
            return

        session.config.training_mode = mode
        await self._send_json(session.websocket, {
            "type": "training_mode_ack",
            "mode": mode,
        })

        logger.info(
            "Training mode changed",
            extra={
                "session_id": session.session_id,
                "mode": mode,
            },
        )

    async def _handle_set_state(self, session: Session, data: dict) -> None:
        """Manually switch the avatar clip state.

        Args:
            session: The source session.
            data: Parsed message with "state" field.
        """
        valid_states = {"idle", "thinking", "talking_happy", "talking_sad"}
        requested = data.get("state", "")
        if requested not in valid_states:
            await self._send_error(
                session.websocket,
                "Invalid state",
                f"Valid states: {', '.join(sorted(valid_states))}",
            )
            return

        clip_url = f"/clips/{session.config.avatar_id}/{requested}.mp4"
        await self._send_json(session.websocket, {
            "type": "body_state",
            "state": requested,
            "clip_url": clip_url,
        })

    async def _handle_stop(self, session: Session, data: dict) -> None:
        """Return the avatar to idle state.

        Args:
            session: The source session.
            data: Parsed message (no additional fields needed).
        """
        clip_url = f"/clips/{session.config.avatar_id}/idle.mp4"
        await self._send_json(session.websocket, {
            "type": "body_state",
            "state": "idle",
            "clip_url": clip_url,
        })

    async def _handle_video_frame(self, session: Session, data: dict) -> None:
        """Relay a customer video frame to subscribed operators.

        Args:
            session: The source session.
            data: Parsed message with "frame" (base64-encoded image data).
        """
        frame = data.get("frame", "")
        if not frame:
            return

        if self._operator_manager:
            await self._operator_manager.relay_video_frame(
                session.session_id, frame,
            )

    async def _handle_document_upload(self, session: Session, data: dict) -> None:
        """Handle a scanned document upload from the customer.

        Args:
            session: The source session.
            data: Parsed message with "image" (base64) and "filename".
        """
        import base64

        image_b64 = data.get("image", "")
        filename = data.get("filename", "document.jpg")

        if not image_b64:
            await self._send_error(
                session.websocket,
                "Empty image",
                "The 'image' field (base64) is required",
            )
            return

        try:
            image_data = base64.b64decode(image_b64)
        except Exception:
            await self._send_error(
                session.websocket,
                "Invalid image data",
                "Could not decode base64 image",
            )
            return

        # Size limit: 10MB
        if len(image_data) > 10 * 1024 * 1024:
            await self._send_error(
                session.websocket,
                "Image too large",
                "Maximum 10MB per document image",
            )
            return

        url = ""
        if self._operator_manager:
            url = await self._operator_manager.relay_document(
                session.session_id, image_data, filename,
            )

        await self._send_json(session.websocket, {
            "type": "document_saved",
            "filename": filename,
            "url": url,
            "size_bytes": len(image_data),
        })

        logger.info(
            "Document uploaded",
            extra={
                "session_id": session.session_id,
                "filename": filename,
                "size_bytes": len(image_data),
            },
        )

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
    import hmac as _hmac
    import json as _json

    # Check query parameter first (constant-time comparison)
    query_key = websocket.query_params.get("api_key", "")
    if query_key and _hmac.compare_digest(query_key, api_key):
        return True

    # Wait for auth message with 5 second timeout
    try:
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
        if len(raw) > 4096:  # Auth messages should be tiny
            return False
        data = _json.loads(raw)
        client_key = data.get("api_key", "")
        if data.get("type") == "auth" and client_key and _hmac.compare_digest(client_key, api_key):
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
