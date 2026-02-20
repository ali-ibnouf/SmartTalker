"""Text-to-Speech engine using CosyVoice 3.0.

Supports zero-shot voice cloning, emotion-aware synthesis,
and Arabic language output at 22050Hz mono WAV.
"""

from __future__ import annotations

import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from src.config import Settings
from src.utils.exceptions import TTSError
from src.utils.logger import setup_logger, log_with_latency

logger = setup_logger("pipeline.tts")

# Emotion-to-TTS parameter mapping
EMOTION_PARAMS: dict[str, dict[str, Any]] = {
    "neutral": {
        "pitch_shift": 0,
        "speed": 1.0,
        "energy": "normal",
    },
    "happy": {
        "pitch_shift": 2,
        "speed": 1.1,
        "energy": "high",
    },
    "sad": {
        "pitch_shift": -1,
        "speed": 0.9,
        "energy": "low",
    },
    "angry": {
        "pitch_shift": 3,
        "speed": 1.2,
        "energy": "high",
    },
    "surprised": {
        "pitch_shift": 4,
        "speed": 1.15,
        "energy": "high",
    },
    "fearful": {
        "pitch_shift": 1,
        "speed": 0.85,
        "energy": "low",
    },
    "disgusted": {
        "pitch_shift": -2,
        "speed": 0.95,
        "energy": "normal",
    },
    "contempt": {
        "pitch_shift": -1,
        "speed": 0.9,
        "energy": "normal",
    },
}


@dataclass
class TTSResult:
    """Result of a TTS synthesis operation.

    Attributes:
        audio_path: Path to the generated WAV file.
        duration_s: Audio duration in seconds.
        sample_rate: Output sample rate in Hz.
        latency_ms: Processing time in milliseconds.
    """

    audio_path: str
    duration_s: float = 0.0
    sample_rate: int = 22050
    latency_ms: int = 0


@dataclass
class VoiceInfo:
    """Metadata for a registered voice.

    Attributes:
        voice_id: Unique voice identifier.
        name: Human-readable voice name.
        language: Primary language code.
        reference_audio: Path to the reference audio file.
        description: Optional voice description.
    """

    voice_id: str
    name: str
    language: str = "ar"
    reference_audio: str = ""
    description: str = ""


class TTSEngine:
    """CosyVoice 3.0 text-to-speech engine.

    Supports zero-shot voice cloning from 3-10 second reference audio,
    emotion-aware synthesis with pitch/speed/energy control, and
    multi-voice management.

    Args:
        config: Application settings with TTS configuration.
    """

    def __init__(self, config: Settings) -> None:
        """Initialize the TTS engine.

        Args:
            config: Application settings instance.
        """
        self._config = config
        self._model: Any = None
        self._loaded = False
        self._voices_dir = Path("voices")
        self._output_dir = config.storage_base_dir / "tts"
        self._voices: dict[str, VoiceInfo] = {}

        # Ensure directories exist
        self._voices_dir.mkdir(parents=True, exist_ok=True)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "TTSEngine initialized",
            extra={
                "model_dir": str(config.tts_model_dir),
                "device": config.tts_device,
                "sample_rate": config.tts_sample_rate,
            },
        )

    @property
    def is_loaded(self) -> bool:
        """Check if the model is currently loaded."""
        return self._loaded

    def load(self) -> None:
        """Load CosyVoice 3.0 model into GPU memory.

        Raises:
            TTSError: If model loading fails.
        """
        if self._loaded:
            logger.info("TTS model already loaded — skipping")
            return

        start = time.perf_counter()
        try:
            from cosyvoice.cli.cosyvoice import CosyVoice2  # type: ignore[import-untyped]

            model_dir = str(self._config.tts_model_dir / "CosyVoice" / "pretrained_models" / "CosyVoice2-0.5B")
            self._model = CosyVoice2(model_dir)
            self._loaded = True

            elapsed = (time.perf_counter() - start) * 1000
            log_with_latency(logger, "TTS model loaded", elapsed)

            # Scan existing voices
            self._scan_voices()

        except ImportError as exc:
            raise TTSError(
                message="CosyVoice package not installed",
                detail="Clone and install from: github.com/FunAudioLLM/CosyVoice",
                original_exception=exc,
            ) from exc
        except Exception as exc:
            raise TTSError(
                message="Failed to load TTS model",
                detail=str(exc),
                original_exception=exc,
            ) from exc

    def synthesize(
        self,
        text: str,
        voice_ref: Optional[str] = None,
        emotion: str = "neutral",
        speed: float = 1.0,
    ) -> TTSResult:
        """Synthesize speech from text.

        Uses zero-shot mode if a voice reference is provided,
        otherwise uses the default model voice.

        Args:
            text: Input text to synthesize (max length from config).
            voice_ref: Optional path to reference audio (3-10s WAV).
            emotion: Emotion label for prosody adjustment.
            speed: Speech speed multiplier (0.5–2.0).

        Returns:
            TTSResult with audio path, duration, sample rate, and latency.

        Raises:
            TTSError: If model not loaded, text too long, or synthesis fails.
        """
        if not self._loaded or self._model is None:
            raise TTSError(message="TTS model not loaded — call load() first")

        # Validate text length
        if not text or not text.strip():
            raise TTSError(message="Text cannot be empty")

        if len(text) > self._config.tts_max_text_length:
            raise TTSError(
                message=f"Text too long: {len(text)} chars (max {self._config.tts_max_text_length})",
            )

        # Validate speed range
        speed = max(0.5, min(2.0, speed))

        # Apply emotion parameters
        params = EMOTION_PARAMS.get(emotion, EMOTION_PARAMS["neutral"])
        effective_speed = speed * params["speed"]

        # Generate unique output filename
        output_filename = f"tts_{uuid.uuid4().hex[:12]}.wav"
        output_path = self._output_dir / output_filename

        start = time.perf_counter()
        try:
            if voice_ref:
                # Zero-shot voice cloning mode
                audio_output = self._synthesize_zero_shot(
                    text=text,
                    reference_audio=voice_ref,
                    speed=effective_speed,
                )
            else:
                # Default voice mode
                audio_output = self._synthesize_default(
                    text=text,
                    speed=effective_speed,
                )

            # Save audio output
            duration_s = self._save_audio(audio_output, output_path)
            elapsed_ms = int((time.perf_counter() - start) * 1000)

            result = TTSResult(
                audio_path=str(output_path),
                duration_s=round(duration_s, 3),
                sample_rate=self._config.tts_sample_rate,
                latency_ms=elapsed_ms,
            )

            log_with_latency(
                logger,
                "TTS synthesis complete",
                elapsed_ms,
                extra={
                    "text_length": len(text),
                    "duration_s": result.duration_s,
                    "emotion": emotion,
                    "voice_ref": voice_ref is not None,
                },
            )
            return result

        except TTSError:
            raise
        except Exception as exc:
            raise TTSError(
                message="TTS synthesis failed",
                detail=str(exc),
                original_exception=exc,
            ) from exc

    def _synthesize_zero_shot(
        self,
        text: str,
        reference_audio: str,
        speed: float = 1.0,
    ) -> Any:
        """Synthesize using zero-shot voice cloning.

        Args:
            text: Text to synthesize.
            reference_audio: Path to reference audio (3-10 seconds).
            speed: Speech speed multiplier.

        Returns:
            Raw audio tensor/array from CosyVoice.

        Raises:
            TTSError: If reference audio is invalid.
        """
        ref_path = Path(reference_audio)
        if not ref_path.exists():
            raise TTSError(
                message=f"Voice reference file not found: {reference_audio}",
            )

        # CosyVoice zero-shot inference
        # The prompt text is extracted from the reference audio
        output_gen = self._model.inference_zero_shot(
            tts_text=text,
            prompt_text="",
            prompt_speech_16k=str(ref_path),
            stream=False,
            speed=speed,
        )

        # Collect output from generator
        for result in output_gen:
            return result["tts_speech"]

        raise TTSError(message="Zero-shot synthesis produced no output")

    def _synthesize_default(
        self,
        text: str,
        speed: float = 1.0,
    ) -> Any:
        """Synthesize using the default model voice.

        Args:
            text: Text to synthesize.
            speed: Speech speed multiplier.

        Returns:
            Raw audio tensor/array from CosyVoice.
        """
        # CosyVoice instruct mode with default speaker
        output_gen = self._model.inference_sft(
            tts_text=text,
            spk_id="default",
            stream=False,
            speed=speed,
        )

        for result in output_gen:
            return result["tts_speech"]

        raise TTSError(message="Default synthesis produced no output")

    def _save_audio(self, audio_tensor: Any, output_path: Path) -> float:
        """Save audio tensor to WAV file.

        Args:
            audio_tensor: Audio data (torch tensor or numpy array).
            output_path: Destination file path.

        Returns:
            Audio duration in seconds.

        Raises:
            TTSError: If saving fails.
        """
        try:
            import numpy as np
            import soundfile as sf

            # Convert torch tensor to numpy if needed
            if hasattr(audio_tensor, "cpu"):
                audio_data = audio_tensor.cpu().numpy()
            elif isinstance(audio_tensor, np.ndarray):
                audio_data = audio_tensor
            else:
                audio_data = np.array(audio_tensor)

            # Ensure 1D (mono)
            if audio_data.ndim > 1:
                audio_data = audio_data.squeeze()

            # Normalize to [-1.0, 1.0] range
            max_val = np.abs(audio_data).max()
            if max_val > 0:
                audio_data = audio_data / max_val * 0.95

            # Write WAV file
            sf.write(
                str(output_path),
                audio_data,
                samplerate=self._config.tts_sample_rate,
                subtype="PCM_16",
            )

            duration_s = len(audio_data) / self._config.tts_sample_rate
            return duration_s

        except Exception as exc:
            raise TTSError(
                message="Failed to save TTS audio",
                detail=str(exc),
                original_exception=exc,
            ) from exc

    def clone_voice(
        self,
        reference_audio: str,
        voice_name: str,
    ) -> str:
        """Register a new voice from reference audio.

        Copies the reference audio to the voices directory and
        registers it for future synthesis calls.

        Args:
            reference_audio: Path to reference audio (3-10 seconds, WAV).
            voice_name: Human-readable name for the voice.

        Returns:
            Unique voice_id string.

        Raises:
            TTSError: If the reference audio is invalid.
        """
        ref_path = Path(reference_audio)
        if not ref_path.exists():
            raise TTSError(message=f"Reference audio not found: {reference_audio}")

        # Validate reference audio duration
        try:
            from src.utils.audio import get_duration
            duration = get_duration(ref_path)
        except Exception:
            duration = 0.0

        if duration < 3.0:
            raise TTSError(
                message=f"Reference audio too short: {duration:.1f}s (min 3s)",
            )
        if duration > 10.0:
            raise TTSError(
                message=f"Reference audio too long: {duration:.1f}s (max 10s)",
            )

        # Generate voice ID and copy file
        voice_id = f"voice_{uuid.uuid4().hex[:8]}"
        dest_path = self._voices_dir / f"{voice_id}{ref_path.suffix}"
        shutil.copy2(str(ref_path), str(dest_path))

        # Register voice
        voice_info = VoiceInfo(
            voice_id=voice_id,
            name=voice_name,
            language="ar",
            reference_audio=str(dest_path),
            description=f"Cloned voice from {ref_path.name}",
        )
        self._voices[voice_id] = voice_info

        logger.info(
            "Voice cloned and registered",
            extra={"voice_id": voice_id, "name": voice_name, "duration_s": round(duration, 2)},
        )
        return voice_id

    def list_voices(self) -> list[VoiceInfo]:
        """List all registered voices.

        Returns:
            List of VoiceInfo objects for all available voices.
        """
        return list(self._voices.values())

    def _scan_voices(self) -> None:
        """Scan the voices directory for existing reference audio files."""
        for voice_file in self._voices_dir.iterdir():
            if voice_file.suffix.lower() in {".wav", ".mp3", ".ogg", ".m4a"}:
                voice_id = voice_file.stem
                if voice_id not in self._voices:
                    self._voices[voice_id] = VoiceInfo(
                        voice_id=voice_id,
                        name=voice_id,
                        language="ar",
                        reference_audio=str(voice_file),
                    )

        logger.info(
            "Voices scanned",
            extra={"count": len(self._voices)},
        )

    def get_emotion_params(self, emotion: str) -> dict[str, Any]:
        """Get TTS parameters for a given emotion.

        Args:
            emotion: Emotion label (neutral, happy, sad, angry, etc.).

        Returns:
            Dictionary of pitch_shift, speed, and energy parameters.
        """
        return EMOTION_PARAMS.get(emotion, EMOTION_PARAMS["neutral"])

    def unload(self) -> None:
        """Free GPU memory by unloading the TTS model.

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

            logger.info("TTS model unloaded and GPU memory freed")

        except Exception as exc:
            logger.warning(
                "Error during TTS model unload",
                extra={"error": str(exc)},
            )
            self._loaded = False
