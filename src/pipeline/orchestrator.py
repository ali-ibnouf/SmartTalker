"""Pipeline orchestrator — coordinates ASR -> LLM -> TTS -> Video -> Upscale.

Manages model lifecycle, GPU memory, and end-to-end processing
for text-to-speech, audio-chat, and full video generation pipelines.

Audit fixes applied:
- Sync GPU calls (ASR, TTS, Upscale) wrapped in run_in_executor()
- asyncio.Semaphore for GPU concurrency control
- Per-request session_id for LLM history isolation
- Public clone_voice() and list_voices() methods (no private access from routes)
"""

from __future__ import annotations

import asyncio
import functools
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from src.config import Settings
from src.pipeline.asr import ASREngine
from src.pipeline.emotions import EmotionEngine
from src.pipeline.llm import LLMEngine
from src.pipeline.tts import TTSEngine, VoiceInfo
from src.pipeline.upscale import UpscaleEngine
from src.pipeline.video import VideoEngine
from src.utils.exceptions import PipelineError
from src.utils.logger import setup_logger, log_with_latency
from src.utils.metrics import (
    ACTIVE_SESSIONS,
    GPU_MEMORY_USAGE,
    GPU_QUEUE_DEPTH,
    INFERENCE_LATENCY,
)
logger = setup_logger("pipeline.orchestrator")

# Default avatar reference image path
DEFAULT_AVATAR_DIR = Path("avatars")

# Max concurrent GPU inferences (prevents OOM on RTX 4090)
_DEFAULT_GPU_CONCURRENCY = 2


@dataclass
class PipelineResult:
    """Result of a full pipeline execution.

    Attributes:
        audio_path: Path to the generated audio file.
        video_path: Path to the generated video file (if video enabled).
        response_text: LLM-generated response text.
        detected_emotion: Detected emotion from the input.
        total_latency_ms: Total end-to-end processing time.
        breakdown: Per-layer latency breakdown.
    """

    audio_path: str = ""
    video_path: Optional[str] = None
    response_text: str = ""
    detected_emotion: str = "neutral"
    total_latency_ms: int = 0
    breakdown: dict[str, int] = field(default_factory=dict)


class SmartTalkerPipeline:
    """Main pipeline orchestrator for SmartTalker.

    Coordinates the full processing flow:
    - Text input:  Emotion -> LLM -> TTS (-> Video -> Upscale)
    - Audio input: ASR -> Emotion -> LLM -> TTS (-> Video -> Upscale)

    Video and Upscale layers are optional.
    All sync GPU calls are offloaded to a thread pool executor
    so the async event loop is never blocked.

    Args:
        config: Application settings.
    """

    def __init__(self, config: Settings) -> None:
        """Initialize the pipeline with all engine instances.

        Args:
            config: Application settings instance.
        """
        self._config = config
        self._asr = ASREngine(config)
        self._llm = LLMEngine(config)
        self._tts = TTSEngine(config)
        self._video = VideoEngine(config)
        self._upscale = UpscaleEngine(config)
        self._emotion = EmotionEngine(config)
        self._start_time = time.time()

        # Flags for optional layers
        self._video_enabled = config.video_enabled
        self._upscale_enabled = config.upscale_enabled

        # GPU concurrency semaphore — limits parallel GPU inferences
        self._gpu_semaphore = asyncio.Semaphore(_DEFAULT_GPU_CONCURRENCY)

        # Enforce GPU memory fraction limit
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.set_per_process_memory_fraction(
                    config.gpu_memory_fraction
                )
                logger.info(
                    "GPU memory fraction set",
                    extra={"fraction": config.gpu_memory_fraction},
                )
        except (ImportError, RuntimeError) as exc:
            logger.debug(f"Could not set GPU memory fraction: {exc}")

        logger.info(
            "SmartTalkerPipeline initialized",
            extra={
                "video_enabled": self._video_enabled,
                "upscale_enabled": self._upscale_enabled,
                "gpu_concurrency": _DEFAULT_GPU_CONCURRENCY,
            },
        )

    @property
    def uptime_seconds(self) -> float:
        """Seconds since pipeline was initialized."""
        return round(time.time() - self._start_time, 1)

    def load_all(self) -> None:
        """Load all pipeline models.

        Non-critical models are loaded with graceful fallbacks.
        """
        logger.info("Loading pipeline models...")
        start = time.perf_counter()

        # Core models (warn but don't fail)
        for name, loader in [
            ("ASR", self._asr.load),
            ("TTS", self._tts.load),
        ]:
            try:
                loader()
            except PipelineError as exc: # Changed from SmartTalkerError to PipelineError
                logger.warning(f"{name} model failed to load: {exc.message}")

        # Optional: Emotion model (lightweight)
        try:
            self._emotion.load()
        except Exception as exc:
            logger.warning(f"Emotion model failed to load: {exc}")

        # Optional: Video and Upscale
        if self._video_enabled:
            try:
                self._video.load()
            except PipelineError as exc: # Changed from SmartTalkerError to PipelineError
                logger.warning(f"Video model failed to load: {exc.message}")
                self._video_enabled = False

        if self._upscale_enabled:
            try:
                self._upscale.load()
            except PipelineError as exc: # Changed from SmartTalkerError to PipelineError
                logger.warning(f"Upscale model failed to load: {exc.message}")
                self._upscale_enabled = False

        elapsed = int((time.perf_counter() - start) * 1000)
        log_with_latency(logger, "Pipeline models loaded", elapsed)

    # ═════════════════════════════════════════════════════════════════════
    # GPU executor helper
    # ═════════════════════════════════════════════════════════════════════

    async def _run_on_gpu(self, fn, *args, **kwargs):
        """Run a sync GPU-bound function in a thread pool, guarded by semaphore.

        Prevents event-loop blocking and limits GPU concurrency.
        Updates GPU_QUEUE_DEPTH metric to track waiting tasks.

        Args:
            fn: Synchronous callable to execute.
            *args: Positional arguments for fn.
            **kwargs: Keyword arguments for fn.

        Returns:
            Return value of fn.
        """
        # Track how many tasks are waiting for the GPU
        waiting = _DEFAULT_GPU_CONCURRENCY - self._gpu_semaphore._value
        GPU_QUEUE_DEPTH.set(waiting)

        loop = asyncio.get_running_loop()
        async with self._gpu_semaphore:
            GPU_QUEUE_DEPTH.set(_DEFAULT_GPU_CONCURRENCY - self._gpu_semaphore._value)
            try:
                return await loop.run_in_executor(
                    None, functools.partial(fn, *args, **kwargs)
                )
            finally:
                GPU_QUEUE_DEPTH.set(_DEFAULT_GPU_CONCURRENCY - self._gpu_semaphore._value)

    # ═════════════════════════════════════════════════════════════════════
    # Text Pipeline
    # ═════════════════════════════════════════════════════════════════════

    async def process_text(
        self,
        text: str,
        avatar_id: str = "default",
        voice_id: Optional[str] = None,
        emotion: str = "neutral",
        language: str = "ar",
        enable_video: bool = False,
        session_id: str = "default",
    ) -> PipelineResult:
        """Process text input through Emotion -> LLM -> TTS (-> Video -> Upscale).

        Args:
            text: User's text input.
            avatar_id: Avatar identifier for video generation.
            voice_id: Optional voice ID for TTS cloning.
            emotion: Emotion label (auto-detected if "neutral").
            language: Target response language.
            enable_video: Whether to generate video output.
            session_id: Unique session identifier for conversation isolation.

        Returns:
            PipelineResult with audio/video paths, text, and latency breakdown.
        """
        start = time.perf_counter()
        breakdown: dict[str, int] = {}

        # ── Emotion Detection (lightweight — OK on event loop) ────────────
        if emotion == "neutral":
            with INFERENCE_LATENCY.labels(model_type="emotion").time():
                emo_result = self._emotion.detect_from_text(text)
            emotion = emo_result.primary_emotion
            breakdown["emotion_ms"] = emo_result.latency_ms

        # ── LLM (already async via httpx) ─────────────────────────────────
        with INFERENCE_LATENCY.labels(model_type="llm").time():
            llm_result = await self._llm.generate(
                user_text=text,
                emotion=emotion,
                language=language,
                session_id=session_id,
            )
        breakdown["llm_ms"] = llm_result.latency_ms

        # Update Request Metrics
        ACTIVE_SESSIONS.set(len(self._llm._sessions))
        try:
            import torch
            if torch.cuda.is_available():
                GPU_MEMORY_USAGE.labels(device_index="0").set(torch.cuda.memory_allocated(0))
        except ImportError:
            pass


        # ── TTS (sync GPU → executor) ────────────────────────────────────
        voice_ref = self._resolve_voice_ref(voice_id)
        with INFERENCE_LATENCY.labels(model_type="tts").time():
            tts_result = await self._run_on_gpu(
                self._tts.synthesize,
                text=llm_result.text,
                voice_ref=voice_ref,
                emotion=emotion,
            )
        breakdown["tts_ms"] = tts_result.latency_ms

        # ── Video (optional) ─────────────────────────────────────────────
        video_path: Optional[str] = None
        if enable_video and self._video_enabled and self._video.is_loaded:
            with INFERENCE_LATENCY.labels(model_type="video").time():
                video_path = await self._generate_video(
                    audio_path=tts_result.audio_path,
                    avatar_id=avatar_id,
                    breakdown=breakdown,
                )

        total_ms = int((time.perf_counter() - start) * 1000)

        result = PipelineResult(
            audio_path=tts_result.audio_path,
            video_path=video_path,
            response_text=llm_result.text,
            detected_emotion=emotion,
            total_latency_ms=total_ms,
            breakdown=breakdown,
        )

        log_with_latency(
            logger, "Text pipeline complete", total_ms,
            extra={"input_length": len(text), "emotion": emotion, "video": video_path is not None},
        )
        return result

    # ═════════════════════════════════════════════════════════════════════
    # Audio Pipeline
    # ═════════════════════════════════════════════════════════════════════

    async def process_audio(
        self,
        audio_path: str,
        avatar_id: str = "default",
        voice_id: Optional[str] = None,
        emotion: str = "neutral",
        language: str = "ar",
        enable_video: bool = False,
        session_id: str = "default",
    ) -> PipelineResult:
        """Process audio input through ASR -> Emotion -> LLM -> TTS (-> Video -> Upscale).

        Args:
            audio_path: Path to the user's audio file.
            avatar_id: Avatar identifier for video generation.
            voice_id: Optional voice ID for TTS cloning.
            emotion: Emotion label (auto-detected if "neutral").
            language: Target response language.
            enable_video: Whether to generate video output.
            session_id: Unique session identifier for conversation isolation.

        Returns:
            PipelineResult with audio/video paths, text, and latency breakdown.
        """
        start = time.perf_counter()
        breakdown: dict[str, int] = {}

        # ── ASR (sync GPU → executor) ────────────────────────────────────
        with INFERENCE_LATENCY.labels(model_type="asr").time():
            asr_result = await self._run_on_gpu(self._asr.transcribe, audio_path)
        breakdown["asr_ms"] = asr_result.latency_ms
        detected_lang = asr_result.language if asr_result.language != "unknown" else language

        # ── Emotion Detection ────────────────────────────────────────────
        if emotion == "neutral":
            emo_result = self._emotion.detect_from_text(asr_result.text)
            emotion = emo_result.primary_emotion
            breakdown["emotion_ms"] = emo_result.latency_ms

        # ── LLM (already async) ──────────────────────────────────────────
        llm_result = await self._llm.generate(
            user_text=asr_result.text,
            emotion=emotion,
            language=detected_lang,
            session_id=session_id,
        )
        breakdown["llm_ms"] = llm_result.latency_ms

        # ── TTS (sync GPU → executor) ────────────────────────────────────
        voice_ref = self._resolve_voice_ref(voice_id)
        tts_result = await self._run_on_gpu(
            self._tts.synthesize,
            text=llm_result.text,
            voice_ref=voice_ref,
            emotion=emotion,
        )
        breakdown["tts_ms"] = tts_result.latency_ms

        # ── Video (optional) ─────────────────────────────────────────────
        video_path: Optional[str] = None
        if enable_video and self._video_enabled and self._video.is_loaded:
            video_path = await self._generate_video(
                audio_path=tts_result.audio_path,
                avatar_id=avatar_id,
                breakdown=breakdown,
            )

        total_ms = int((time.perf_counter() - start) * 1000)

        result = PipelineResult(
            audio_path=tts_result.audio_path,
            video_path=video_path,
            response_text=llm_result.text,
            detected_emotion=emotion,
            total_latency_ms=total_ms,
            breakdown=breakdown,
        )

        log_with_latency(
            logger, "Audio pipeline complete", total_ms,
            extra={"transcribed": asr_result.text[:80], "emotion": emotion},
        )
        return result

    # ═════════════════════════════════════════════════════════════════════
    # Video sub-pipeline
    # ═════════════════════════════════════════════════════════════════════

    async def _generate_video(
        self,
        audio_path: str,
        avatar_id: str,
        breakdown: dict[str, int],
    ) -> Optional[str]:
        """Generate and optionally upscale a talking-head video.

        Args:
            audio_path: Path to the TTS audio output.
            avatar_id: Avatar identifier for reference image lookup.
            breakdown: Latency breakdown dict to update.

        Returns:
            Path to the final video, or None if generation fails.
        """
        # Resolve avatar reference image
        ref_image = self._resolve_avatar_image(avatar_id)
        if not ref_image:
            logger.warning(f"No reference image for avatar '{avatar_id}' — skipping video")
            return None

        try:
            # Generate raw video (already async via subprocess)
            video_result = await self._video.generate(
                audio_path=audio_path,
                reference_image=ref_image,
            )
            breakdown["video_ms"] = video_result.latency_ms

            # Upscale if enabled (async via executor)
            if self._upscale_enabled and self._upscale.is_loaded:
                enhanced = await self._upscale.enhance(
                    video_path=video_result.video_path,
                    target_resolution=self._config.upscale_target_resolution,
                )
                breakdown["upscale_ms"] = enhanced.latency_ms
                return enhanced.video_path

            return video_result.video_path

        except PipelineError as exc:
            logger.error(f"Video generation failed: {exc.message}")
            return None

    # ═════════════════════════════════════════════════════════════════════
    # Public API (used by routes — no private member access)
    # ═════════════════════════════════════════════════════════════════════

    def clone_voice(self, reference_audio: str, voice_name: str) -> str:
        """Register a new cloned voice from reference audio.

        Args:
            reference_audio: Path to reference audio (3-10 seconds).
            voice_name: Human-readable name for the voice.

        Returns:
            Unique voice_id string.
        """
        return self._tts.clone_voice(
            reference_audio=reference_audio,
            voice_name=voice_name,
        )

    def list_voices(self) -> list[VoiceInfo]:
        """List all registered voices.

        Returns:
            List of VoiceInfo objects.
        """
        return self._tts.list_voices()

    # ═════════════════════════════════════════════════════════════════════
    # Helpers
    # ═════════════════════════════════════════════════════════════════════

    def _resolve_voice_ref(self, voice_id: Optional[str]) -> Optional[str]:
        """Resolve a voice ID to a reference audio path.

        Args:
            voice_id: Voice identifier to look up.

        Returns:
            Path to reference audio, or None for default voice.
        """
        if not voice_id:
            return None

        voices = {v.voice_id: v for v in self._tts.list_voices()}
        voice_info = voices.get(voice_id)
        if voice_info and voice_info.reference_audio:
            return voice_info.reference_audio
        return None

    @staticmethod
    def _resolve_avatar_image(avatar_id: str) -> Optional[str]:
        """Resolve an avatar ID to a reference image path.

        Looks for files in avatars/<avatar_id>/ directory.

        Args:
            avatar_id: Avatar identifier.

        Returns:
            Path to the reference image, or None if not found.
        """
        # Reject path traversal characters
        if not avatar_id or "/" in avatar_id or "\\" in avatar_id or ".." in avatar_id:
            logger.warning("Invalid avatar_id rejected", extra={"avatar_id": avatar_id})
            return None

        avatar_dir = DEFAULT_AVATAR_DIR / avatar_id

        # Verify resolved path stays inside the avatars directory
        try:
            avatar_dir.resolve().relative_to(DEFAULT_AVATAR_DIR.resolve())
        except ValueError:
            logger.warning("Avatar path traversal blocked", extra={"avatar_id": avatar_id})
            return None

        if not avatar_dir.exists():
            return None

        # Look for common image formats
        for ext in [".png", ".jpg", ".jpeg", ".webp"]:
            for pattern in [f"reference{ext}", f"avatar{ext}", f"{avatar_id}{ext}"]:
                candidate = avatar_dir / pattern
                if candidate.exists():
                    return str(candidate)

        # Fallback: first image file found
        for f in avatar_dir.iterdir():
            if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                return str(f)

        return None

    async def health_check(self) -> dict[str, Any]:
        """Check system health, model status, and Ollama connectivity.

        Returns:
            Dictionary with status, GPU availability, loaded models, and Ollama reachability.
        """
        gpu_available = False
        gpu_memory_used_mb = 0.0

        try:
            import torch
            gpu_available = torch.cuda.is_available()
            if gpu_available:
                gpu_memory_used_mb = round(
                    torch.cuda.memory_allocated() / (1024 * 1024), 1
                )
        except ImportError:
            pass

        # Check Ollama connectivity
        ollama_reachable = False
        try:
            client = await self._llm._get_client()
            resp = await client.get("/api/tags")
            ollama_reachable = resp.status_code == 200
        except Exception:
            ollama_reachable = False

        models_loaded = {
            "asr": self._asr.is_loaded,
            "tts": self._tts.is_loaded,
            "emotion": self._emotion.is_loaded,
            "video": self._video.is_loaded,
            "upscale": self._upscale.is_loaded,
            "ollama": ollama_reachable,
        }

        all_ok = any(models_loaded.values()) and ollama_reachable
        status = "healthy" if all_ok else "degraded"

        return {
            "status": status,
            "gpu_available": gpu_available,
            "gpu_memory_used_mb": gpu_memory_used_mb,
            "models_loaded": models_loaded,
            "video_enabled": self._video_enabled,
            "upscale_enabled": self._upscale_enabled,
            "uptime_s": self.uptime_seconds,
        }

    async def unload_all(self) -> None:
        """Unload all models and free GPU memory."""
        logger.info("Unloading all pipeline models...")

        self._asr.unload()
        self._tts.unload()
        self._emotion.unload()
        await self._llm.close()

        if self._video.is_loaded:
            self._video.unload()
        if self._upscale.is_loaded:
            self._upscale.unload()

        logger.info("All pipeline models unloaded")
