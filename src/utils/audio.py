"""Audio processing utilities.

Functions: convert_format, get_duration, validate_audio,
normalize_audio. Supports WAV, OGG, MP3, WEBM, M4A.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Optional

from src.utils.exceptions import SmartTalkerError
from src.utils.ffmpeg import run_ffmpeg as _run_ffmpeg, run_ffprobe as _run_ffprobe
from src.utils.logger import setup_logger

logger = setup_logger("utils.audio")

# Supported audio formats for input/output
SUPPORTED_FORMATS: set[str] = {"wav", "ogg", "mp3", "webm", "m4a"}

# Constraints
MAX_FILE_SIZE_MB: int = 25
MAX_DURATION_SECONDS: float = 300.0  # 5 minutes
MIN_DURATION_SECONDS: float = 0.1


def convert_format(
    input_path: Path,
    output_format: str,
    output_dir: Optional[Path] = None,
) -> Path:
    """Convert an audio file to the specified format.

    Args:
        input_path: Path to the source audio file.
        output_format: Target format (wav, ogg, mp3, webm, m4a).
        output_dir: Optional output directory. Defaults to same as input.

    Returns:
        Path to the converted audio file.

    Raises:
        SmartTalkerError: If format is unsupported or conversion fails.
    """
    input_path = Path(input_path)
    output_format = output_format.lower().lstrip(".")

    if output_format not in SUPPORTED_FORMATS:
        raise SmartTalkerError(
            message=f"Unsupported output format: {output_format}",
            detail=f"Supported formats: {SUPPORTED_FORMATS}",
        )

    if not input_path.exists():
        raise SmartTalkerError(message=f"Input file not found: {input_path}")

    dest_dir = output_dir or input_path.parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    output_path = dest_dir / f"{input_path.stem}.{output_format}"

    _run_ffmpeg(["-i", str(input_path), str(output_path)])

    logger.info(
        "Audio converted",
        extra={"src": str(input_path), "dst": str(output_path)},
    )
    return output_path


def get_duration(audio_path: Path) -> float:
    """Get the duration of an audio file in seconds.

    Args:
        audio_path: Path to the audio file.

    Returns:
        Duration in seconds as a float.

    Raises:
        SmartTalkerError: If the file cannot be probed.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise SmartTalkerError(message=f"Audio file not found: {audio_path}")

    stdout = _run_ffprobe([
        "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ])

    try:
        return float(stdout)
    except ValueError:
        raise SmartTalkerError(
            message="Could not parse audio duration",
            detail=f"ffprobe output: {stdout}",
        ) from None


def validate_audio(file_path: Path) -> bool:
    """Validate an audio file for format, size, and duration.

    Args:
        file_path: Path to the audio file to validate.

    Returns:
        True if the file passes all validation checks.

    Raises:
        SmartTalkerError: If any validation check fails.
    """
    file_path = Path(file_path)

    # Check existence
    if not file_path.exists():
        raise SmartTalkerError(message=f"File not found: {file_path}")

    # Check format
    ext = file_path.suffix.lower().lstrip(".")
    if ext not in SUPPORTED_FORMATS:
        raise SmartTalkerError(
            message=f"Unsupported audio format: {ext}",
            detail=f"Supported: {SUPPORTED_FORMATS}",
        )

    # Check file size
    size_mb = file_path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise SmartTalkerError(
            message=f"File too large: {size_mb:.1f}MB (max {MAX_FILE_SIZE_MB}MB)",
        )

    # Check duration
    duration = get_duration(file_path)
    if duration < MIN_DURATION_SECONDS:
        raise SmartTalkerError(
            message=f"Audio too short: {duration:.2f}s (min {MIN_DURATION_SECONDS}s)",
        )
    if duration > MAX_DURATION_SECONDS:
        raise SmartTalkerError(
            message=f"Audio too long: {duration:.1f}s (max {MAX_DURATION_SECONDS}s)",
        )

    logger.info(
        "Audio validated",
        extra={"path": str(file_path), "size_mb": round(size_mb, 2), "duration_s": round(duration, 2)},
    )
    return True


def normalize_audio(audio_path: Path, output_dir: Optional[Path] = None) -> Path:
    """Normalize audio volume using ffmpeg loudnorm filter.

    Args:
        audio_path: Path to the input audio file.
        output_dir: Optional output directory. Defaults to same as input.

    Returns:
        Path to the normalized audio file.

    Raises:
        SmartTalkerError: If normalization fails.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise SmartTalkerError(message=f"Audio file not found: {audio_path}")

    dest_dir = output_dir or audio_path.parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    output_path = dest_dir / f"{audio_path.stem}_normalized{audio_path.suffix}"

    _run_ffmpeg([
        "-i", str(audio_path),
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-ar", "22050",
        "-ac", "1",
        str(output_path),
    ])

    logger.info("Audio normalized", extra={"output": str(output_path)})
    return output_path


def normalize_arabic_text(text: str) -> str:
    """Normalize Arabic text using Unicode NFC normalization.

    Args:
        text: Raw Arabic text string.

    Returns:
        NFC-normalized text string.
    """
    return unicodedata.normalize("NFC", text)
