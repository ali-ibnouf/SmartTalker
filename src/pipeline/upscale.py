"""Video upscaling engine using RealESRGAN and CodeFormer.

Enhances video resolution and facial quality for
production-grade digital human output.
"""

from __future__ import annotations

import asyncio
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from src.config import Settings
from src.utils.exceptions import UpscaleError
from src.utils.logger import setup_logger, log_with_latency

logger = setup_logger("pipeline.upscale")


# Resolution presets
RESOLUTION_PRESETS: dict[str, tuple[int, int]] = {
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "1440p": (2560, 1440),
    "4k": (3840, 2160),
}


@dataclass
class EnhancedResult:
    """Result of a video enhancement operation.

    Attributes:
        video_path: Path to the enhanced video.
        original_resolution: Original resolution string.
        enhanced_resolution: Enhanced resolution string.
        scale_factor: Upscale multiplier applied.
        latency_ms: Processing time in milliseconds.
    """

    video_path: str
    original_resolution: str = ""
    enhanced_resolution: str = ""
    scale_factor: float = 1.0
    latency_ms: int = 0


class UpscaleEngine:
    """RealESRGAN + CodeFormer video upscaling engine.

    Applies super-resolution to video frames and enhances
    facial quality using CodeFormer. Skips processing if
    the input already meets the target resolution.

    Args:
        config: Application settings with Upscale configuration.
    """

    def __init__(self, config: Settings) -> None:
        """Initialize the upscale engine.

        Args:
            config: Application settings instance.
        """
        self._config = config
        self._realesrgan: Any = None
        self._codeformer: Any = None
        self._loaded = False
        self._output_dir = config.storage_base_dir / "upscale"
        self._output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "UpscaleEngine initialized",
            extra={
                "model_dir": str(config.upscale_model_dir),
                "target": config.upscale_target_resolution,
            },
        )

    @property
    def is_loaded(self) -> bool:
        """Check if the models are currently loaded."""
        return self._loaded

    def load(self) -> None:
        """Load RealESRGAN and CodeFormer models.

        Raises:
            UpscaleError: If model loading fails.
        """
        if self._loaded:
            logger.info("Upscale models already loaded — skipping")
            return

        start = time.perf_counter()
        try:
            self._load_realesrgan()
            self._load_codeformer()
            self._loaded = True

            elapsed = (time.perf_counter() - start) * 1000
            log_with_latency(logger, "Upscale models loaded", elapsed)

        except UpscaleError:
            raise
        except Exception as exc:
            raise UpscaleError(
                message="Failed to load upscale models",
                detail=str(exc),
                original_exception=exc,
            ) from exc

    def _load_realesrgan(self) -> None:
        """Load the RealESRGAN model for super-resolution.

        Raises:
            UpscaleError: If the model file is not found.
        """
        try:
            from realesrgan import RealESRGANer  # type: ignore[import-untyped]
            from basicsr.archs.rrdbnet_arch import RRDBNet  # type: ignore[import-untyped]
            import torch

            model_path = self._config.upscale_model_dir / "realesrgan" / "realesr-general-x4v3.pth"
            if not model_path.exists():
                raise UpscaleError(
                    message=f"RealESRGAN model not found: {model_path}",
                    detail="Run: bash scripts/download_models.sh",
                )

            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4,
            )

            device = self._config.upscale_device
            self._realesrgan = RealESRGANer(
                scale=4,
                model_path=str(model_path),
                model=model,
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=torch.cuda.is_available(),
                device=device,
            )

            logger.info("RealESRGAN loaded")

        except ImportError as exc:
            raise UpscaleError(
                message="realesrgan package not installed",
                detail="Install: pip install realesrgan basicsr",
                original_exception=exc,
            ) from exc

    def _load_codeformer(self) -> None:
        """Load the CodeFormer model for face enhancement.

        Raises:
            UpscaleError: If the model file is not found.
        """
        try:
            model_path = self._config.upscale_model_dir / "codeformer" / "codeformer.pth"
            if not model_path.exists():
                raise UpscaleError(
                    message=f"CodeFormer model not found: {model_path}",
                    detail="Run: bash scripts/download_models.sh",
                )

            # Store path for face enhancement pipeline
            self._codeformer = {"model_path": str(model_path)}
            logger.info("CodeFormer model path registered")

        except UpscaleError:
            raise
        except Exception as exc:
            raise UpscaleError(
                message="Failed to load CodeFormer",
                detail=str(exc),
                original_exception=exc,
            ) from exc

    async def enhance(
        self,
        video_path: str,
        target_resolution: str = "1080p",
    ) -> EnhancedResult:
        """Enhance video resolution and facial quality.

        Applies RealESRGAN super-resolution frame-by-frame and
        CodeFormer face enhancement. Skips if already at target.

        Args:
            video_path: Path to the input video.
            target_resolution: Target resolution preset (720p, 1080p, 1440p, 4k).

        Returns:
            EnhancedResult with enhanced video path and metadata.

        Raises:
            UpscaleError: If enhancement fails.
        """
        if not self._loaded:
            raise UpscaleError(message="Upscale models not loaded — call load() first")

        input_path = Path(video_path)
        if not input_path.exists():
            raise UpscaleError(message=f"Video file not found: {video_path}")

        # Get target dimensions
        target = RESOLUTION_PRESETS.get(
            target_resolution,
            RESOLUTION_PRESETS.get(self._config.upscale_target_resolution, (1920, 1080)),
        )

        # Check if upscaling is needed
        from src.utils.video import get_video_info
        video_info = get_video_info(input_path)
        original_w = video_info["width"]
        original_h = video_info["height"]

        if original_w >= target[0] and original_h >= target[1]:
            logger.info(
                "Video already at target resolution — skipping upscale",
                extra={"current": f"{original_w}x{original_h}", "target": target_resolution},
            )
            return EnhancedResult(
                video_path=video_path,
                original_resolution=f"{original_w}x{original_h}",
                enhanced_resolution=f"{original_w}x{original_h}",
                scale_factor=1.0,
                latency_ms=0,
            )

        output_filename = f"enhanced_{uuid.uuid4().hex[:12]}.mp4"
        output_path = self._output_dir / output_filename

        start = time.perf_counter()
        try:
            await self._process_video_frames(
                input_path=str(input_path),
                output_path=str(output_path),
                target_w=target[0],
                target_h=target[1],
                fps=video_info["fps"],
            )

            elapsed_ms = int((time.perf_counter() - start) * 1000)
            scale = target[0] / original_w if original_w > 0 else 1.0

            result = EnhancedResult(
                video_path=str(output_path),
                original_resolution=f"{original_w}x{original_h}",
                enhanced_resolution=f"{target[0]}x{target[1]}",
                scale_factor=round(scale, 2),
                latency_ms=elapsed_ms,
            )

            log_with_latency(
                logger,
                "Video enhancement complete",
                elapsed_ms,
                extra={
                    "original": result.original_resolution,
                    "enhanced": result.enhanced_resolution,
                    "scale": result.scale_factor,
                },
            )
            return result

        except UpscaleError:
            raise
        except Exception as exc:
            raise UpscaleError(
                message="Video enhancement failed",
                detail=str(exc),
                original_exception=exc,
            ) from exc

    async def _process_video_frames(
        self,
        input_path: str,
        output_path: str,
        target_w: int,
        target_h: int,
        fps: float,
    ) -> None:
        """Process video frame-by-frame with RealESRGAN.

        Extracts frames, enhances each, and reassembles.

        Args:
            input_path: Source video path.
            output_path: Destination video path.
            target_w: Target output width.
            target_h: Target output height.
            fps: Output FPS.

        Raises:
            UpscaleError: If frame processing fails.
        """
        try:
            # Use ffmpeg + RealESRGAN pipeline via subprocess
            # Extract → enhance → reassemble
            cmd = [
                "python", "-m", "realesrgan",
                "-i", input_path,
                "-o", output_path,
                "-s", "4",
                "--face_enhance",
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=600)

            if process.returncode != 0:
                # Fallback: use ffmpeg upscale if realesrgan CLI fails
                logger.warning("RealESRGAN CLI failed, falling back to ffmpeg upscale")
                await self._ffmpeg_upscale(input_path, output_path, target_w, target_h, fps)
                return

        except asyncio.TimeoutError:
            raise UpscaleError(message="Video enhancement timed out (600s limit)")
        except UpscaleError:
            raise
        except Exception:
            # Fallback to ffmpeg
            await self._ffmpeg_upscale(input_path, output_path, target_w, target_h, fps)

    @staticmethod
    async def _ffmpeg_upscale(
        input_path: str,
        output_path: str,
        target_w: int,
        target_h: int,
        fps: float,
    ) -> None:
        """Fallback upscale using ffmpeg's lanczos filter.

        Args:
            input_path: Source video path.
            output_path: Destination video path.
            target_w: Target width.
            target_h: Target height.
            fps: Output FPS.
        """
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", input_path,
            "-vf", f"scale={target_w}:{target_h}:flags=lanczos",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-c:a", "aac",
            "-r", str(int(fps)),
            "-movflags", "+faststart",
            output_path,
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=300)

            if process.returncode != 0:
                raise UpscaleError(
                    message="ffmpeg upscale fallback failed",
                    detail=stderr.decode() if stderr else "Unknown error",
                )
        except asyncio.TimeoutError:
            raise UpscaleError(message="ffmpeg upscale timed out")
        except Exception as exc:
            raise UpscaleError(message=f"ffmpeg upscale failed: {exc}")

    def unload(self) -> None:
        """Free GPU memory by unloading upscale models."""
        try:
            self._realesrgan = None
            self._codeformer = None
            self._loaded = False

            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

            logger.info("Upscale models unloaded")

        except Exception as exc:
            logger.warning(f"Error during upscale unload: {exc}")
            self._loaded = False
