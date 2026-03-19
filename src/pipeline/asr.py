"""Automatic Speech Recognition engine using DashScope qwen3-asr.

Connects to DashScope WebSocket for real-time streaming ASR.
Model: qwen3-asr-flash-realtime ($0.008/min).
Replaces the old FunASR local engine.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from src.config import Settings
from src.utils.exceptions import ASRError
from src.utils.logger import setup_logger, log_with_latency

logger = setup_logger("pipeline.asr")

# Cost: $0.008 per minute of audio
COST_PER_MINUTE = 0.008


@dataclass
class TranscriptionResult:
    """Result of an ASR transcription.

    Attributes:
        text: Transcribed text output.
        language: Detected language code (e.g., "ar", "en").
        confidence: Average confidence score (0.0-1.0).
        latency_ms: Processing time in milliseconds.
        segments: Word/phrase-level segments with timestamps.
        cost_usd: Cost of this transcription.
    """

    text: str
    language: str = "ar"
    confidence: float = 0.0
    latency_ms: int = 0
    segments: list[dict[str, Any]] = field(default_factory=list)
    cost_usd: float = 0.0


class ASRSession:
    """A single streaming ASR session connected to DashScope WebSocket.

    Usage:
        session = await asr_engine.create_session("ar")
        await session.send_audio(pcm_chunk_bytes)
        await session.send_audio(more_bytes)
        result = await session.finish()
    """

    def __init__(self, ws: Any, session_id: str, language: str) -> None:
        self._ws = ws
        self._session_id = session_id
        self._language = language
        self._transcript_parts: list[str] = []
        self._segments: list[dict[str, Any]] = []
        self._audio_duration_s = 0.0
        self._closed = False

    async def close(self) -> None:
        """Close the underlying WebSocket connection without waiting for results."""
        if not self._closed:
            self._closed = True
            try:
                await self._ws.close()
            except Exception:
                pass

    async def send_audio(self, pcm_bytes: bytes) -> None:
        """Send a chunk of PCM audio (16-bit signed mono) to the ASR session.

        Args:
            pcm_bytes: Raw PCM audio bytes. 16kHz 16-bit mono expected.
        """
        if self._closed:
            raise ASRError(message="ASR session already closed")

        # Track duration: 16kHz 16-bit mono = 32000 bytes/sec
        self._audio_duration_s += len(pcm_bytes) / 32000.0

        encoded = base64.b64encode(pcm_bytes).decode("ascii")
        msg = {
            "type": "input_audio_buffer.append",
            "audio": encoded,
        }
        await self._ws.send(json.dumps(msg))

    async def finish(self) -> TranscriptionResult:
        """Commit the audio buffer and wait for the final transcript.

        Returns:
            TranscriptionResult with the complete transcription.
        """
        if self._closed:
            raise ASRError(message="ASR session already closed")

        self._closed = True

        # Signal end of audio
        await self._ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        # Collect responses until session.finished
        try:
            async for raw_msg in self._ws:
                data = json.loads(raw_msg)
                msg_type = data.get("type", "")

                if msg_type == "transcription.text":
                    text = data.get("text", "")
                    if text:
                        self._transcript_parts.append(text)

                elif msg_type == "transcription.segment":
                    self._segments.append({
                        "start": data.get("start", 0),
                        "end": data.get("end", 0),
                        "text": data.get("text", ""),
                    })

                elif msg_type == "session.finished":
                    break

                elif msg_type == "error":
                    raise ASRError(
                        message="DashScope ASR error",
                        detail=data.get("message", str(data)),
                    )
        finally:
            await self._ws.close()

        full_text = " ".join(self._transcript_parts).strip()
        detected_lang = ASREngine._detect_language(full_text) if full_text else self._language
        cost = self._audio_duration_s / 60.0 * COST_PER_MINUTE

        return TranscriptionResult(
            text=full_text,
            language=detected_lang,
            confidence=0.9 if full_text else 0.0,
            latency_ms=0,  # Caller measures end-to-end latency
            segments=self._segments,
            cost_usd=cost,
        )


class ASREngine:
    """DashScope qwen3-asr streaming ASR engine.

    Connects to DashScope WebSocket for real-time speech recognition.
    Replaces the old FunASR local engine while keeping the same public interface.
    """

    def __init__(self, config: Settings) -> None:
        self._config = config
        self._api_key = config.dashscope_api_key
        self._ws_url = config.dashscope_ws_url
        self._model = config.asr_model
        self._loaded = True  # No local model to load — always ready

        # Rolling buffers for monitoring
        from collections import deque
        self._recent_latencies: deque[float] = deque(maxlen=50)
        self._recent_errors: deque[float] = deque(maxlen=20)  # timestamps of WS failures

        logger.info(
            "ASREngine initialized (DashScope WebSocket)",
            extra={"model": self._model, "ws_url": self._ws_url},
        )

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        """No-op — DashScope ASR is cloud-based, no local model to load."""
        self._loaded = True
        logger.info("ASREngine ready (DashScope cloud, no local model)")

    async def create_session(self, language: str = "ar") -> ASRSession:
        """Open a new streaming ASR session via DashScope WebSocket.

        Args:
            language: Hint for the expected language.

        Returns:
            ASRSession that accepts audio chunks and returns a transcript.
        """
        try:
            import websockets
        except ImportError:
            raise ASRError(
                message="websockets not installed",
                detail="Install with: pip install websockets",
            )

        session_id = uuid.uuid4().hex[:16]

        headers = {
            "Authorization": f"Bearer {self._api_key}",
        }

        try:
            ws = await asyncio.wait_for(
                websockets.connect(
                    self._ws_url,
                    additional_headers=headers,
                    ping_interval=20,
                    ping_timeout=10,
                ),
                timeout=15.0,
            )

            # Send session.update to configure the ASR session
            config_msg = {
                "type": "session.update",
                "session": {
                    "model": self._model,
                    "input_audio_format": "pcm16",
                    "sample_rate": 16000,
                    "language": language,
                },
            }
            await ws.send(json.dumps(config_msg))

            # Wait for session.created acknowledgment
            ack_raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
            ack = json.loads(ack_raw)
            if ack.get("type") == "error":
                raise ASRError(
                    message="DashScope ASR session creation failed",
                    detail=ack.get("message", str(ack)),
                )

            logger.info(
                "ASR session created",
                extra={"session_id": session_id, "language": language},
            )
            return ASRSession(ws, session_id, language)

        except ASRError:
            self._recent_errors.append(time.time())
            raise
        except Exception as exc:
            self._recent_errors.append(time.time())
            raise ASRError(
                message="Failed to create ASR session",
                detail=str(exc),
                original_exception=exc,
            ) from exc

    async def transcribe(self, audio_path: str) -> TranscriptionResult:
        """Transcribe an audio file (convenience method for non-streaming use).

        Opens a session, sends the entire file, and returns the result.
        Maintains backward compatibility with the old FunASR interface.

        Args:
            audio_path: Path to the audio file.

        Returns:
            TranscriptionResult with text, language, and segments.
        """
        from pathlib import Path

        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise ASRError(message=f"Audio file not found: {audio_path}")

        start = time.perf_counter()

        session = await self.create_session()

        # Read and send audio in 32KB chunks (1 second of 16kHz 16-bit mono)
        chunk_size = 32000
        with open(audio_file, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                await session.send_audio(chunk)

        result = await session.finish()
        result.latency_ms = int((time.perf_counter() - start) * 1000)
        self._recent_latencies.append(result.latency_ms / 1000.0)

        log_with_latency(
            logger,
            "Transcription complete",
            result.latency_ms,
            extra={
                "input_path": str(audio_file),
                "text_length": len(result.text),
                "language": result.language,
                "cost_usd": f"{result.cost_usd:.6f}",
            },
        )
        return result

    # Turkish-unique characters (not shared with French or English)
    _TURKISH_UNIQUE = frozenset("ğĞışİŞ")
    # French accented characters
    _FRENCH_MARKERS = frozenset("éèêëàâäùûüôöïîçÉÈÊËÀÂÄÙÛÜÔÖÏÎÇœŒæÆ")

    @staticmethod
    def _detect_language(text: str) -> str:
        """Detect language from transcribed text (AR/EN/FR/TR).

        Arabic is identified by Unicode Arabic block characters.
        Among Latin-script languages, Turkish is identified by unique
        characters (ğ, ı, ş, İ), French by accented characters
        (é, è, ê, etc.), and English is the fallback.
        """
        if not text:
            return "unknown"

        arabic_chars = sum(
            1 for c in text if "\u0600" <= c <= "\u06FF" or "\u0750" <= c <= "\u077F"
        )
        latin_chars = sum(1 for c in text if "A" <= c <= "Z" or "a" <= c <= "z")
        total = arabic_chars + latin_chars

        if total == 0:
            return "unknown"
        if arabic_chars / total > 0.5:
            return "ar"
        if latin_chars / total > 0.3:
            turkish_unique = sum(1 for c in text if c in ASREngine._TURKISH_UNIQUE)
            if turkish_unique > 0:
                return "tr"
            french_markers = sum(1 for c in text if c in ASREngine._FRENCH_MARKERS)
            if french_markers > 0:
                return "fr"
            return "en"
        return "mixed"

    def unload(self) -> None:
        """No-op — no local resources to free."""
        self._loaded = False
        logger.info("ASR engine unloaded")
