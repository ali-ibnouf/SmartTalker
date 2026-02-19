"""Shared ffmpeg/ffprobe runner used by audio and video utilities."""

from __future__ import annotations

import subprocess

from src.utils.exceptions import SmartTalkerError


def run_ffmpeg(args: list[str], timeout: int = 120) -> subprocess.CompletedProcess[str]:
    """Run an ffmpeg command and return the result.

    Args:
        args: Command-line arguments for ffmpeg (without the 'ffmpeg' prefix).
        timeout: Max seconds before the process is killed.

    Returns:
        Completed process result.

    Raises:
        SmartTalkerError: If ffmpeg is not installed or command fails.
    """
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args]
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        raise SmartTalkerError(
            message="ffmpeg not found",
            detail="Install ffmpeg: apt-get install ffmpeg",
        ) from None
    except subprocess.CalledProcessError as exc:
        raise SmartTalkerError(
            message="ffmpeg command failed",
            detail=exc.stderr,
            original_exception=exc,
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise SmartTalkerError(
            message="ffmpeg command timed out",
            original_exception=exc,
        ) from exc


def run_ffprobe(args: list[str], timeout: int = 30) -> str:
    """Run an ffprobe command and return stdout.

    Args:
        args: Command-line arguments for ffprobe.
        timeout: Max seconds before the process is killed.

    Returns:
        Stdout string from ffprobe.

    Raises:
        SmartTalkerError: If ffprobe fails.
    """
    cmd = ["ffprobe", "-hide_banner", *args]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout,
        )
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise SmartTalkerError(
            message="ffprobe command failed",
            detail=str(exc),
            original_exception=exc if isinstance(exc, Exception) else None,
        ) from exc
