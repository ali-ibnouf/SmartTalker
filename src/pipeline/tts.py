"""Text-to-Speech engine using DashScope qwen3-tts.

Connects to DashScope WebSocket for real-time streaming TTS with voice cloning.
Model: qwen3-tts-vc-realtime ($0.015/min audio output).
Voice enrollment: $0.20/voice via REST API.
Replaces the old CosyVoice local engine.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional

import httpx

from src.config import Settings
from src.utils.exceptions import TTSError
from src.utils.logger import setup_logger, log_with_latency

logger = setup_logger("pipeline.tts")

# DashScope TTS pricing
COST_PER_MINUTE = 0.015  # $0.015 per minute of audio output
VOICE_ENROLLMENT_COST = 0.20  # $0.20 per voice enrollment

# Emotion-to-TTS parameter mapping (speed multiplier for emotion control)
EMOTION_PARAMS: dict[str, dict[str, Any]] = {
    "neutral": {"speed": 1.0},
    "happy": {"speed": 1.1},
    "sad": {"speed": 0.9},
    "angry": {"speed": 1.2},
    "surprised": {"speed": 1.15},
    "fearful": {"speed": 0.85},
    "disgusted": {"speed": 0.95},
    "contempt": {"speed": 0.9},
}


@dataclass
class TTSChunk:
    """A single audio chunk from streaming TTS synthesis."""

    seq: int
    audio_bytes: bytes
    duration_ms: int
    sample_rate: int = 48000


@dataclass
class TTSResult:
    """Result of a TTS synthesis operation."""
    audio_path: str
    duration_s: float = 0.0
    sample_rate: int = 48000
    latency_ms: int = 0
    lip_sync: dict = None  # type: ignore[assignment]
    cost_usd: float = 0.0

    def __post_init__(self) -> None:
        if self.lip_sync is None:
            self.lip_sync = {}


@dataclass
class VoiceInfo:
    """Metadata for a registered voice."""
    voice_id: str
    name: str
    language: str = "ar"
    reference_audio: str = ""
    description: str = ""


class TTSStream:
    """Async iterator over TTS audio chunks from a DashScope WebSocket session.

    Usage:
        stream = await tts_engine.synthesize_stream("Hello", voice_id="v123")
        async for chunk in stream:
            # chunk.audio_bytes is raw PCM 48kHz 16-bit mono
            ...
        all_audio = await stream.collect_all()
    """

    def __init__(self, ws: Any) -> None:
        self._ws = ws
        self._chunks: list[bytes] = []
        self._total_bytes = 0
        self._finished = False
        self._seq = 0

    @property
    def duration_seconds(self) -> float:
        """Calculated duration from total audio bytes (48kHz 16-bit mono = 96000 bytes/sec)."""
        return self._total_bytes / 96000.0

    @property
    def cost_usd(self) -> float:
        return self.duration_seconds / 60.0 * COST_PER_MINUTE

    async def collect_all(self) -> bytes:
        """Consume all chunks and return concatenated audio bytes."""
        all_bytes = bytearray()
        async for chunk in self:
            all_bytes.extend(chunk.audio_bytes)
        return bytes(all_bytes)

    def __aiter__(self) -> AsyncIterator[TTSChunk]:
        return self

    async def __anext__(self) -> TTSChunk:
        if self._finished:
            raise StopAsyncIteration

        try:
            while True:
                raw_msg = await self._ws.recv()
                data = json.loads(raw_msg)
                msg_type = data.get("type", "")

                if msg_type == "audio.delta":
                    audio_b64 = data.get("audio", "")
                    if audio_b64:
                        audio_bytes = base64.b64decode(audio_b64)
                        self._total_bytes += len(audio_bytes)
                        self._chunks.append(audio_bytes)
                        duration_ms = int(len(audio_bytes) / 96000.0 * 1000)
                        chunk = TTSChunk(
                            seq=self._seq,
                            audio_bytes=audio_bytes,
                            duration_ms=duration_ms,
                            sample_rate=48000,
                        )
                        self._seq += 1
                        return chunk

                elif msg_type == "session.finished":
                    self._finished = True
                    await self._ws.close()
                    raise StopAsyncIteration

                elif msg_type == "error":
                    self._finished = True
                    await self._ws.close()
                    raise TTSError(
                        message="DashScope TTS error",
                        detail=data.get("message", str(data)),
                    )

        except StopAsyncIteration:
            raise
        except TTSError:
            raise
        except Exception as exc:
            self._finished = True
            raise TTSError(
                message="TTS stream error",
                detail=str(exc),
                original_exception=exc,
            ) from exc


class TTSEngine:
    """DashScope qwen3-tts streaming TTS engine with voice cloning.

    Connects to DashScope WebSocket for real-time text-to-speech synthesis.
    Supports voice enrollment via REST API for voice cloning.
    Replaces the old CosyVoice local engine.
    """

    def __init__(self, config: Settings) -> None:
        self._config = config
        self._api_key = config.dashscope_api_key
        self._ws_url = config.dashscope_ws_url
        self._base_url = config.dashscope_base_url
        self._model = config.tts_model
        self._max_text_length = config.tts_max_text_length
        self._loaded = True  # Cloud-based — always ready
        self._voices: dict[str, VoiceInfo] = {}
        self._http_client: Optional[httpx.AsyncClient] = None

        logger.info(
            "TTSEngine initialized (DashScope WebSocket)",
            extra={"model": self._model},
        )

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        """No-op — DashScope TTS is cloud-based."""
        self._loaded = True
        logger.info("TTSEngine ready (DashScope cloud, no local model)")

    async def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(30.0, connect=10.0),
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._http_client

    async def clone_voice(
        self, audio_url: str, prefix: str = "", language: str = "ar"
    ) -> str:
        """Enroll a new voice via DashScope voice-enrollment REST API.

        Args:
            audio_url: URL to the reference audio file (must be publicly accessible or R2 URL).
            prefix: Optional prefix for the voice name.
            language: Voice language code.

        Returns:
            voice_id: The enrolled voice ID for use in synthesis.

        Cost: $0.20 per voice enrollment.
        """
        client = await self._get_http_client()

        voice_name = f"{prefix}_{uuid.uuid4().hex[:8]}" if prefix else uuid.uuid4().hex[:12]

        try:
            response = await client.post(
                "/services/tts/voice-enrollment",
                json={
                    "model": self._model,
                    "audio_url": audio_url,
                    "voice_name": voice_name,
                    "language": language,
                },
            )
            response.raise_for_status()
            data = response.json()
            voice_id = data.get("voice_id", voice_name)

            self._voices[voice_id] = VoiceInfo(
                voice_id=voice_id,
                name=voice_name,
                language=language,
                reference_audio=audio_url,
                description=f"DashScope enrolled voice ({language})",
            )

            logger.info(
                "Voice enrolled",
                extra={"voice_id": voice_id, "cost_usd": VOICE_ENROLLMENT_COST},
            )
            return voice_id

        except httpx.HTTPStatusError as exc:
            raise TTSError(
                message=f"Voice enrollment failed: {exc.response.status_code}",
                detail=exc.response.text,
                original_exception=exc,
            ) from exc
        except Exception as exc:
            raise TTSError(
                message="Voice enrollment failed",
                detail=str(exc),
                original_exception=exc,
            ) from exc

    async def synthesize_stream(
        self,
        text: str,
        voice_id: Optional[str] = None,
        emotion: str = "neutral",
        speed: float = 1.0,
        language: str = "ar",
    ) -> TTSStream:
        """Open a streaming TTS session and return a TTSStream async iterator.

        Args:
            text: Text to synthesize.
            voice_id: DashScope voice ID (from clone_voice or built-in).
            emotion: Emotion for speed adjustment.
            speed: Base speech speed multiplier.
            language: Target language code.

        Returns:
            TTSStream that yields TTSChunk objects.
        """
        if not text or not text.strip():
            raise TTSError(message="Text cannot be empty")

        if len(text) > self._max_text_length:
            raise TTSError(f"Text too long: {len(text)} chars (max {self._max_text_length})")

        try:
            import websockets
        except ImportError:
            raise TTSError(
                message="websockets not installed",
                detail="Install with: pip install websockets",
            )

        # Apply emotion speed adjustment
        emotion_params = EMOTION_PARAMS.get(emotion, EMOTION_PARAMS["neutral"])
        effective_speed = speed * emotion_params["speed"]

        headers = {
            "Authorization": f"Bearer {self._api_key}",
        }

        try:
            ws = await websockets.connect(
                self._ws_url,
                additional_headers=headers,
                ping_interval=20,
                ping_timeout=10,
            )

            # Configure TTS session
            session_config: dict[str, Any] = {
                "model": self._model,
                "output_audio_format": "pcm16",
                "sample_rate": 48000,
                "speed": effective_speed,
            }
            if voice_id:
                session_config["voice"] = voice_id

            config_msg = {
                "type": "session.update",
                "session": session_config,
            }
            await ws.send(json.dumps(config_msg))

            # Wait for session acknowledgment
            ack_raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
            ack = json.loads(ack_raw)
            if ack.get("type") == "error":
                raise TTSError(
                    message="DashScope TTS session creation failed",
                    detail=ack.get("message", str(ack)),
                )

            # Send the text for synthesis
            text_msg = {
                "type": "input_text.append",
                "text": text,
            }
            await ws.send(json.dumps(text_msg))

            # Signal that all text has been sent
            await ws.send(json.dumps({"type": "input_text.commit"}))

            logger.info(
                "TTS stream started",
                extra={
                    "text_length": len(text),
                    "voice_id": voice_id or "default",
                    "emotion": emotion,
                    "speed": effective_speed,
                },
            )

            return TTSStream(ws)

        except TTSError:
            raise
        except Exception as exc:
            raise TTSError(
                message="Failed to start TTS stream",
                detail=str(exc),
                original_exception=exc,
            ) from exc

    async def synthesize(
        self,
        text: str,
        voice_id: Optional[str] = None,
        emotion: str = "neutral",
        speed: float = 1.0,
        language: str = "ar",
    ) -> TTSResult:
        """Synthesize speech and return all audio at once (non-streaming convenience).

        Backward compatible with old CosyVoice interface.
        """
        import struct
        from pathlib import Path

        start = time.perf_counter()

        stream = await self.synthesize_stream(text, voice_id, emotion, speed, language)
        all_audio = await stream.collect_all()

        elapsed_ms = int((time.perf_counter() - start) * 1000)

        # Save to WAV file
        output_dir = self._config.storage_base_dir / "tts"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"tts_{uuid.uuid4().hex[:12]}.wav"

        # Write WAV header + PCM data
        sample_rate = 48000
        num_channels = 1
        bits_per_sample = 16
        data_size = len(all_audio)
        with open(output_path, "wb") as f:
            f.write(b"RIFF")
            f.write(struct.pack("<I", 36 + data_size))
            f.write(b"WAVE")
            f.write(b"fmt ")
            f.write(struct.pack("<I", 16))
            f.write(struct.pack("<H", 1))  # PCM
            f.write(struct.pack("<H", num_channels))
            f.write(struct.pack("<I", sample_rate))
            f.write(struct.pack("<I", sample_rate * num_channels * bits_per_sample // 8))
            f.write(struct.pack("<H", num_channels * bits_per_sample // 8))
            f.write(struct.pack("<H", bits_per_sample))
            f.write(b"data")
            f.write(struct.pack("<I", data_size))
            f.write(all_audio)

        duration_s = stream.duration_seconds
        lip_sync = self._prepare_lip_sync(text, language, duration_s)

        result = TTSResult(
            audio_path=str(output_path),
            duration_s=round(duration_s, 3),
            sample_rate=sample_rate,
            latency_ms=elapsed_ms,
            lip_sync=lip_sync,
            cost_usd=stream.cost_usd,
        )

        log_with_latency(
            logger, "TTS synthesis complete", elapsed_ms,
            extra={
                "text_length": len(text),
                "emotion": emotion,
                "language": language,
                "cost_usd": f"{stream.cost_usd:.6f}",
            },
        )
        return result

    @staticmethod
    def _estimate_phoneme_weight(word: str) -> float:
        """Estimate a word's spoken duration weight using phoneme-aware heuristics."""
        import unicodedata

        weight = 0.0
        for ch in word:
            cat = unicodedata.category(ch)
            if cat == "Mn":
                weight += 0.3
            elif ch.lower() in "aeiou\u0627\u0648\u064a":
                weight += 1.0
            elif ch.lower() in "rlmnwy\u0631\u0644\u0645\u0646":
                weight += 0.7
            elif ch.lower() in "szfv\u0633\u0632\u0641\u0634":
                weight += 0.6
            else:
                weight += 0.4
        return max(weight, 0.1)

    def _prepare_lip_sync(self, text: str, language: str, duration_s: float) -> dict:
        """Generate word-level timing data for lip sync."""
        import re as _re

        words = _re.split(r"[\s\u200b\u200c\u200d]+", text.strip())
        words = [w for w in words if w]

        if not words or duration_s <= 0:
            return {"words": []}

        word_weights = [self._estimate_phoneme_weight(w) for w in words]
        total_weight = sum(word_weights)
        if total_weight <= 0:
            return {"words": []}

        current_time = 0.0
        word_timings = []

        for word, wt in zip(words, word_weights):
            word_duration = (wt / total_weight) * duration_s
            word_timings.append({
                "word": word,
                "start": round(current_time, 3),
                "end": round(current_time + word_duration, 3),
            })
            current_time += word_duration

        return {"words": word_timings}

    def list_voices(self) -> list[VoiceInfo]:
        return list(self._voices.values())

    def unload(self) -> None:
        """Clean up HTTP client."""
        self._loaded = False
        if self._http_client is not None and not self._http_client.is_closed:
            # Cannot await in sync method — client will be garbage collected
            self._http_client = None
        logger.info("TTS engine unloaded")
