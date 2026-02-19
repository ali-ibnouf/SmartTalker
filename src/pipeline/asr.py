"""Automatic Speech Recognition engine using Fun-ASR Nano.

Supports Arabic and multilingual transcription with VAD
(Voice Activity Detection) and word-level timestamps.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from src.config import Settings
from src.utils.exceptions import ASRError
from src.utils.logger import setup_logger, log_with_latency

logger = setup_logger("pipeline.asr")


@dataclass
class TranscriptionResult:
    """Result of an ASR transcription.

    Attributes:
        text: Transcribed text output.
        language: Detected language code (e.g., "ar", "en").
        confidence: Average confidence score (0.0–1.0).
        latency_ms: Processing time in milliseconds.
        segments: Word/phrase-level segments with timestamps.
    """

    text: str
    language: str = "ar"
    confidence: float = 0.0
    latency_ms: int = 0
    segments: list[dict[str, Any]] = field(default_factory=list)


class ASREngine:
    """Fun-ASR Nano speech recognition engine.

    Manages model lifecycle, GPU memory, and provides
    transcription with VAD and language detection.

    Args:
        config: Application settings with ASR configuration.
    """

    def __init__(self, config: Settings) -> None:
        """Initialize the ASR engine.

        Args:
            config: Application settings instance.
        """
        self._config = config
        self._model: Any = None
        self._loaded = False
        logger.info(
            "ASREngine initialized",
            extra={"model_id": config.asr_model_id, "device": config.asr_device},
        )

    @property
    def is_loaded(self) -> bool:
        """Check if the model is currently loaded."""
        return self._loaded

    def load(self) -> None:
        """Load Fun-ASR model and VAD model into GPU memory.

        Raises:
            ASRError: If model loading fails.
        """
        if self._loaded:
            logger.info("ASR model already loaded — skipping")
            return

        start = time.perf_counter()
        try:
            from funasr import AutoModel  # type: ignore[import-untyped]

            self._model = AutoModel(
                model=self._config.asr_model_id,
                vad_model=self._config.asr_vad_model,
                device=self._config.asr_device,
                model_path=str(self._config.asr_model_dir),
            )
            self._loaded = True

            elapsed = (time.perf_counter() - start) * 1000
            log_with_latency(logger, "ASR model loaded", elapsed)

        except ImportError as exc:
            raise ASRError(
                message="funasr package not installed",
                detail="Install with: pip install funasr modelscope",
                original_exception=exc,
            ) from exc
        except Exception as exc:
            raise ASRError(
                message="Failed to load ASR model",
                detail=str(exc),
                original_exception=exc,
            ) from exc

    def transcribe(self, audio_path: str) -> TranscriptionResult:
        """Transcribe an audio file to text.

        Args:
            audio_path: Path to the audio file (WAV, MP3, OGG, etc.).

        Returns:
            TranscriptionResult with text, language, confidence, and segments.

        Raises:
            ASRError: If the model is not loaded or transcription fails.
        """
        if not self._loaded or self._model is None:
            raise ASRError(message="ASR model not loaded — call load() first")

        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise ASRError(
                message=f"Audio file not found: {audio_path}",
            )

        start = time.perf_counter()
        try:
            result = self._model.generate(input=str(audio_file))
            elapsed_ms = int((time.perf_counter() - start) * 1000)

            # Parse Fun-ASR output format
            parsed = self._parse_result(result)
            parsed.latency_ms = elapsed_ms

            log_with_latency(
                logger,
                "Transcription complete",
                elapsed_ms,
                extra={
                    "input_path": str(audio_file),
                    "text_length": len(parsed.text),
                    "language": parsed.language,
                },
            )
            return parsed

        except ASRError:
            raise
        except Exception as exc:
            raise ASRError(
                message="Transcription failed",
                detail=str(exc),
                original_exception=exc,
            ) from exc

    def _parse_result(self, raw_result: Any) -> TranscriptionResult:
        """Parse the raw Fun-ASR output into a TranscriptionResult.

        Args:
            raw_result: Raw output from Fun-ASR model.generate().

        Returns:
            Parsed TranscriptionResult.
        """
        # Fun-ASR returns a list of dicts with 'text' and optional 'timestamp'
        if not raw_result:
            return TranscriptionResult(text="", confidence=0.0)

        # Handle list format: [{"text": "...", "timestamp": [...]}]
        if isinstance(raw_result, list):
            segments: list[dict[str, Any]] = []
            full_text_parts: list[str] = []
            total_confidence = 0.0
            count = 0

            for item in raw_result:
                if isinstance(item, dict):
                    text = item.get("text", "")
                    full_text_parts.append(text)

                    # Extract segments with timestamps if available
                    timestamps = item.get("timestamp", [])
                    for ts in timestamps:
                        seg = {
                            "text": ts.get("text", ""),
                            "start": ts.get("start", 0),
                            "end": ts.get("end", 0),
                        }
                        segments.append(seg)

                    # Accumulate confidence
                    conf = item.get("confidence", 0.0)
                    if conf:
                        total_confidence += conf
                        count += 1

            full_text = " ".join(full_text_parts).strip()
            avg_confidence = total_confidence / count if count > 0 else 0.0

            return TranscriptionResult(
                text=full_text,
                language=self._detect_language(full_text),
                confidence=round(avg_confidence, 4),
                segments=segments,
            )

        # Fallback: treat as string
        text = str(raw_result).strip()
        return TranscriptionResult(
            text=text,
            language=self._detect_language(text),
            confidence=0.0,
        )

    @staticmethod
    def _detect_language(text: str) -> str:
        """Simple language detection based on character ranges.

        Args:
            text: Transcribed text to analyze.

        Returns:
            Language code: "ar" for Arabic, "en" for English, "mixed" otherwise.
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
        if latin_chars / total > 0.5:
            return "en"
        return "mixed"

    def unload(self) -> None:
        """Free GPU memory by unloading the ASR model.

        Safe to call even if the model is not loaded.
        """
        try:
            if self._model is not None:
                del self._model
                self._model = None

            self._loaded = False

            # Attempt to free CUDA memory
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

            logger.info("ASR model unloaded and GPU memory freed")

        except Exception as exc:
            logger.warning(
                "Error during ASR model unload",
                extra={"error": str(exc)},
            )
            self._loaded = False
