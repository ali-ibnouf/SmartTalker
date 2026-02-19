"""Video generation engine using EchoMimicV2.

Generates talking-head video from audio and a reference image,
with optional pose control and lip-sync.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from src.config import Settings
from src.utils.exceptions import VideoError
from src.utils.logger import setup_logger, log_with_latency

logger = setup_logger("pipeline.video")


@dataclass
class VideoResult:
    """Result of a video generation operation.

    Attributes:
        video_path: Path to the generated MP4 video.
        fps: Output frames per second.
        resolution: Output resolution string (WxH).
        duration_s: Video duration in seconds.
        latency_ms: Processing time in milliseconds.
    """

    video_path: str
    fps: int = 25
    resolution: str = "512x512"
    duration_s: float = 0.0
    latency_ms: int = 0


class VideoEngine:
    """EchoMimicV2 talking-head video generation engine.

    Generates realistic lip-synced video from audio input and
    a reference portrait image. Supports optional pose control.

    Args:
        config: Application settings with Video configuration.
    """

    def __init__(self, config: Settings) -> None:
        """Initialize the video engine.

        Args:
            config: Application settings instance.
        """
        self._config = config
        self._model: Any = None
        self._loaded = False
        self._output_dir = config.storage_base_dir / "video"
        self._output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "VideoEngine initialized",
            extra={
                "model_dir": str(config.video_model_dir),
                "device": config.video_device,
                "fps": config.video_fps,
                "resolution": config.video_resolution,
            },
        )

    @property
    def is_loaded(self) -> bool:
        """Check if the model is currently loaded."""
        return self._loaded

    def load(self) -> None:
        """Load EchoMimicV2 model into GPU memory.

        Raises:
            VideoError: If model loading fails.
        """
        if self._loaded:
            logger.info("Video model already loaded — skipping")
            return

        start = time.perf_counter()
        try:
            import torch
            from diffusers import AutoencoderKL, DDIMScheduler  # type: ignore[import-untyped]
            from transformers import CLIPVisionModelWithProjection  # type: ignore[import-untyped]

            model_dir = Path(self._config.video_model_dir) / "echomimic_v2"
            if not model_dir.exists():
                raise VideoError(
                    message=f"EchoMimicV2 model directory not found: {model_dir}",
                    detail="Run: bash scripts/download_models.sh",
                )
            
            # Verify key model files
            required_files = ["reference_unet.pth", "denoising_unet.pth", "motion_module.pth", "face_locator.pth"]
            missing = [f for f in required_files if not (model_dir / f).exists()]
            if missing:
                raise VideoError(
                    message=f"Missing EchoMimicV2 model files: {missing}",
                    detail="Please re-download models.",
                )

            # Store model components for generation
            self._model = {
                "model_dir": str(model_dir),
                "device": self._config.video_device,
                "dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            }
            self._loaded = True

            elapsed = (time.perf_counter() - start) * 1000
            log_with_latency(logger, "Video model loaded", elapsed)

        except ImportError as exc:
            raise VideoError(
                message="Video dependencies not installed",
                detail="Install: pip install diffusers transformers accelerate",
                original_exception=exc,
            ) from exc
        except VideoError:
            raise
        except Exception as exc:
            raise VideoError(
                message="Failed to load video model",
                detail=str(exc),
                original_exception=exc,
            ) from exc

    async def generate(
        self,
        audio_path: str,
        reference_image: str,
        pose_data: Optional[str] = None,
    ) -> VideoResult:
        """Generate a talking-head video from audio and reference image.

        Args:
            audio_path: Path to the driving audio file (WAV).
            reference_image: Path to the reference portrait image.
            pose_data: Optional path to pose sequence data.

        Returns:
            VideoResult with video path, FPS, resolution, duration, and latency.

        Raises:
            VideoError: If model not loaded, inputs invalid, or generation fails.
        """
        if not self._loaded or self._model is None:
            raise VideoError(message="Video model not loaded — call load() first")

        # Validate inputs
        audio_file = Path(audio_path)
        image_file = Path(reference_image)

        if not audio_file.exists():
            raise VideoError(message=f"Audio file not found: {audio_path}")
        if not image_file.exists():
            raise VideoError(message=f"Reference image not found: {reference_image}")

        # Generate unique output filename
        output_filename = f"video_{uuid.uuid4().hex[:12]}.mp4"
        output_path = self._output_dir / output_filename

        start = time.perf_counter()
        try:
            # Run EchoMimicV2 inference
            duration_s = await self._run_inference(
                audio_path=str(audio_file),
                image_path=str(image_file),
                output_path=str(output_path),
                pose_data=pose_data,
            )

            elapsed_ms = int((time.perf_counter() - start) * 1000)

            result = VideoResult(
                video_path=str(output_path),
                fps=self._config.video_fps,
                resolution=self._config.video_resolution,
                duration_s=round(duration_s, 3),
                latency_ms=elapsed_ms,
            )

            log_with_latency(
                logger,
                "Video generation complete",
                elapsed_ms,
                extra={
                    "duration_s": result.duration_s,
                    "resolution": result.resolution,
                    "output": str(output_path),
                },
            )
            return result

        except VideoError:
            raise
        except Exception as exc:
            raise VideoError(
                message="Video generation failed",
                detail=str(exc),
                original_exception=exc,
            ) from exc

    async def _run_inference(
        self,
        audio_path: str,
        image_path: str,
        output_path: str,
        pose_data: Optional[str] = None,
    ) -> float:
        """Execute the EchoMimicV2 inference pipeline.

        Args:
            audio_path: Path to driving audio.
            image_path: Path to reference portrait.
            output_path: Path for output video.
            pose_data: Optional pose data path.

        Returns:
            Duration of generated video in seconds.

        Raises:
            VideoError: If inference fails.
        """
        import subprocess
        import asyncio

        model_dir = self._model["model_dir"]
        w, h = self._config.video_resolution_tuple

        # Build inference command
        cmd = [
            "python", "-m", "echomimic_v2.inference",
            "--model_dir", model_dir,
            "--audio_path", audio_path,
            "--image_path", image_path,
            "--output_path", output_path,
            "--fps", str(self._config.video_fps),
            "--width", str(w),
            "--height", str(h),
        ]

        if pose_data:
            cmd.extend(["--pose_data", pose_data])

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=300)

            if process.returncode != 0:
                raise VideoError(
                    message="EchoMimicV2 inference failed",
                    detail=stderr.decode() if stderr else "Unknown error",
                )

            # Estimate duration from audio
            from src.utils.audio import get_duration
            duration = get_duration(Path(audio_path))
            return duration

        except asyncio.TimeoutError:
            raise VideoError(message="Video generation timed out (300s limit)")
        except VideoError:
            raise
        except Exception as exc:
            raise VideoError(
                message="Video inference subprocess failed",
                detail=str(exc),
                original_exception=exc,
            ) from exc

    def unload(self) -> None:
        """Free GPU memory by unloading the video model.

        Safe to call even if the model is not loaded.
        """
        try:
            if self._model is not None:
                self._model = None

            self._loaded = False

            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

            logger.info("Video model unloaded and GPU memory freed")

        except Exception as exc:
            logger.warning(
                "Error during video model unload",
                extra={"error": str(exc)},
            )
            self._loaded = False
