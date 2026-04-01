"""Standalone TTS service using DashScope REST API (non-WebSocket).

More reliable than WebSocket for fire-and-forget WhatsApp voice replies.
Uses qwen3-tts-flash model via REST endpoint.

Cost: ~$0.015 per minute of audio output (same rate as streaming).
"""

from __future__ import annotations

import time
from typing import Any, Optional

import httpx

from src.utils.logger import setup_logger, log_with_latency

logger = setup_logger("services.tts_rest")

DASHSCOPE_TTS_REST_URL = (
    "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/"
    "multimodal-generation/generation"
)
TTS_MODEL = "qwen3-tts-flash"
DEFAULT_VOICE = "Cherry"
COST_PER_MINUTE = 0.015


class TTSRestService:
    """DashScope REST-based TTS — synthesize text to audio URL or bytes.

    Unlike the WebSocket TTSEngine, this is a simple request/response:
    send text → receive a temporary audio URL.  No streaming, no session management.
    Ideal for WhatsApp voice replies where we need the full audio at once.
    """

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(60.0, connect=10.0),
            )
        return self._client

    async def synthesize(
        self,
        text: str,
        voice_id: Optional[str] = None,
        language: str = "ar",
    ) -> bytes:
        """Synthesize text to WAV audio bytes.

        Args:
            text: Text to speak.
            voice_id: DashScope voice ID (enrolled or built-in).
            language: Language code (for logging only — model detects language).

        Returns:
            WAV audio bytes downloaded from DashScope's temporary URL.
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        start = time.perf_counter()
        client = await self._get_client()

        # DashScope TTS REST format: input.text + input.voice
        payload: dict[str, Any] = {
            "model": TTS_MODEL,
            "input": {
                "text": text,
                "voice": voice_id or DEFAULT_VOICE,
            },
        }

        try:
            resp = await client.post(
                DASHSCOPE_TTS_REST_URL,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "TTS REST API error",
                extra={"status": exc.response.status_code, "body": exc.response.text[:200]},
            )
            raise RuntimeError(f"TTS REST failed: {exc.response.status_code}") from exc
        except Exception as exc:
            logger.error("TTS REST request failed", extra={"error": str(exc)})
            raise RuntimeError(f"TTS REST failed: {exc}") from exc

        # Extract audio URL from response
        output = data.get("output", {})
        audio_info = output.get("audio", {}) if isinstance(output, dict) else {}
        audio_url = audio_info.get("url", "") if isinstance(audio_info, dict) else ""

        if not audio_url:
            logger.error("TTS REST returned no audio URL", extra={"response_keys": list(data.keys())})
            raise RuntimeError("TTS REST returned no audio URL")

        # Download the audio file from the temporary URL
        try:
            audio_resp = await client.get(audio_url)
            audio_resp.raise_for_status()
            wav_bytes = audio_resp.content
        except Exception as exc:
            logger.error("TTS audio download failed", extra={"error": str(exc)})
            raise RuntimeError(f"TTS audio download failed: {exc}") from exc

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        # Estimate duration from file size (WAV: 48kHz, 16-bit mono = 96000 bytes/sec)
        # Subtract 44-byte WAV header
        audio_data_size = max(0, len(wav_bytes) - 44)
        duration_s = audio_data_size / 96000.0
        cost = duration_s / 60.0 * COST_PER_MINUTE

        log_with_latency(
            logger, "TTS REST synthesis complete", elapsed_ms,
            extra={
                "text_length": len(text),
                "audio_bytes": len(wav_bytes),
                "duration_s": round(duration_s, 2),
                "cost_usd": f"{cost:.6f}",
                "language": language,
            },
        )

        return wav_bytes

    async def synthesize_to_url(
        self,
        text: str,
        voice_id: Optional[str] = None,
        language: str = "ar",
        r2_storage: Any = None,
        session_id: str = "tts_rest",
    ) -> str:
        """Synthesize and upload to R2, returning a public URL.

        Args:
            text: Text to speak.
            voice_id: DashScope voice ID.
            language: Language code.
            r2_storage: R2Storage instance with upload_audio() method.
            session_id: Session ID for the R2 key path.

        Returns:
            Public URL to the uploaded audio file.
        """
        wav_bytes = await self.synthesize(text, voice_id, language)

        if r2_storage is None:
            raise ValueError("r2_storage is required for synthesize_to_url")

        import asyncio

        loop = asyncio.get_running_loop()
        url = await loop.run_in_executor(
            None, r2_storage.upload_audio, session_id, wav_bytes,
        )

        logger.info(
            "TTS audio uploaded to R2",
            extra={"session_id": session_id, "url_prefix": url[:60] if url else ""},
        )
        return url

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
