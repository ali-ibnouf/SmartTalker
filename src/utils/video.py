"""Video processing utilities.

Functions: combine_audio_video, get_video_info,
validate_image, resize_image.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from src.utils.exceptions import SmartTalkerError
from src.utils.ffmpeg import run_ffmpeg as _run_ffmpeg, run_ffprobe as _run_ffprobe
from src.utils.logger import setup_logger

logger = setup_logger("utils.video")

# Supported image formats for avatar references
SUPPORTED_IMAGE_FORMATS: set[str] = {"png", "jpg", "jpeg", "webp"}

# Image constraints
MAX_IMAGE_SIZE_MB: int = 10
MIN_IMAGE_DIMENSION: int = 128
MAX_IMAGE_DIMENSION: int = 4096


def combine_audio_video(
    audio_path: Path,
    video_path: Path,
    output_path: Optional[Path] = None,
) -> Path:
    """Combine an audio file with a video file.

    The output is encoded as H.264 video with AAC audio.

    Args:
        audio_path: Path to the audio file.
        video_path: Path to the video file (may be silent).
        output_path: Optional output path. Defaults to video dir with '_combined' suffix.

    Returns:
        Path to the combined output video.

    Raises:
        SmartTalkerError: If either input is missing or combining fails.
    """
    audio_path = Path(audio_path)
    video_path = Path(video_path)

    if not audio_path.exists():
        raise SmartTalkerError(message=f"Audio file not found: {audio_path}")
    if not video_path.exists():
        raise SmartTalkerError(message=f"Video file not found: {video_path}")

    if output_path is None:
        output_path = video_path.parent / f"{video_path.stem}_combined.mp4"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _run_ffmpeg([
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "libx264",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        str(output_path),
    ])

    logger.info(
        "Audio and video combined",
        extra={"audio": str(audio_path), "video": str(video_path), "output": str(output_path)},
    )
    return output_path


def get_video_info(video_path: Path) -> dict[str, Any]:
    """Get metadata for a video file.

    Args:
        video_path: Path to the video file.

    Returns:
        Dictionary with fps, duration, resolution (WxH), width, height, codec.

    Raises:
        SmartTalkerError: If the file cannot be probed.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise SmartTalkerError(message=f"Video file not found: {video_path}")

    try:
        stdout = _run_ffprobe([
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(video_path),
        ], timeout=30)
        data = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise SmartTalkerError(
            message="Failed to parse video probe output",
            detail=str(exc),
            original_exception=exc,
        ) from exc

    # Find the video stream
    video_stream: Optional[dict[str, Any]] = None
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            video_stream = stream
            break

    if video_stream is None:
        raise SmartTalkerError(
            message="No video stream found in file",
            detail=str(video_path),
        )

    # Parse FPS from r_frame_rate (e.g., "25/1") — guard against "0/0"
    fps_parts = video_stream.get("r_frame_rate", "25/1").split("/")
    if len(fps_parts) == 2:
        num, den = float(fps_parts[0]), float(fps_parts[1])
        fps = num / den if den != 0 else 25.0
    else:
        fps = 25.0

    width = int(video_stream.get("width", 0))
    height = int(video_stream.get("height", 0))
    duration = float(data.get("format", {}).get("duration", 0))

    return {
        "fps": round(fps, 2),
        "duration": round(duration, 2),
        "resolution": f"{width}x{height}",
        "width": width,
        "height": height,
        "codec": video_stream.get("codec_name", "unknown"),
    }


def validate_image(file_path: Path) -> bool:
    """Validate an image file for format, size, and dimensions.

    Args:
        file_path: Path to the image file.

    Returns:
        True if the image passes all validation checks.

    Raises:
        SmartTalkerError: If any validation check fails.
    """
    file_path = Path(file_path)

    # Check existence
    if not file_path.exists():
        raise SmartTalkerError(message=f"Image file not found: {file_path}")

    # Check format
    ext = file_path.suffix.lower().lstrip(".")
    if ext not in SUPPORTED_IMAGE_FORMATS:
        raise SmartTalkerError(
            message=f"Unsupported image format: {ext}",
            detail=f"Supported: {SUPPORTED_IMAGE_FORMATS}",
        )

    # Check file size
    size_mb = file_path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_IMAGE_SIZE_MB:
        raise SmartTalkerError(
            message=f"Image too large: {size_mb:.1f}MB (max {MAX_IMAGE_SIZE_MB}MB)",
        )

    # Check dimensions using ffprobe
    try:
        info = _get_image_dimensions(file_path)
        w, h = info["width"], info["height"]
    except SmartTalkerError:
        raise
    except Exception as exc:
        raise SmartTalkerError(
            message="Could not read image dimensions",
            original_exception=exc,
        ) from exc

    if w < MIN_IMAGE_DIMENSION or h < MIN_IMAGE_DIMENSION:
        raise SmartTalkerError(
            message=f"Image too small: {w}x{h} (min {MIN_IMAGE_DIMENSION}x{MIN_IMAGE_DIMENSION})",
        )
    if w > MAX_IMAGE_DIMENSION or h > MAX_IMAGE_DIMENSION:
        raise SmartTalkerError(
            message=f"Image too large: {w}x{h} (max {MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION})",
        )

    logger.info(
        "Image validated",
        extra={"path": str(file_path), "size_mb": round(size_mb, 2), "dimensions": f"{w}x{h}"},
    )
    return True


def _get_image_dimensions(image_path: Path) -> dict[str, int]:
    """Get width and height of an image using ffprobe.

    Args:
        image_path: Path to the image file.

    Returns:
        Dict with 'width' and 'height' keys.
    """
    try:
        stdout = _run_ffprobe([
            "-v", "quiet",
            "-show_entries", "stream=width,height",
            "-of", "json",
            str(image_path),
        ], timeout=15)
        data = json.loads(stdout)
        stream = data["streams"][0]
        return {"width": int(stream["width"]), "height": int(stream["height"])}
    except (json.JSONDecodeError, KeyError, IndexError) as exc:
        raise SmartTalkerError(
            message="Failed to read image dimensions",
            original_exception=exc,
        ) from exc


def resize_image(
    image_path: Path,
    target_size: tuple[int, int],
    output_path: Optional[Path] = None,
) -> Path:
    """Resize an image to the target dimensions.

    Maintains aspect ratio by padding with black if needed.

    Args:
        image_path: Path to the source image.
        target_size: Target (width, height) tuple.
        output_path: Optional output path. Defaults to input dir with '_resized' suffix.

    Returns:
        Path to the resized image.

    Raises:
        SmartTalkerError: If resizing fails.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise SmartTalkerError(message=f"Image not found: {image_path}")

    if output_path is None:
        output_path = image_path.parent / f"{image_path.stem}_resized{image_path.suffix}"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    w, h = target_size
    scale_filter = (
        f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black"
    )

    _run_ffmpeg([
        "-i", str(image_path),
        "-vf", scale_filter,
        str(output_path),
    ])

    logger.info(
        "Image resized",
        extra={"input": str(image_path), "target": f"{w}x{h}", "output": str(output_path)},
    )
    return output_path
