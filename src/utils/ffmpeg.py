"""Shared ffmpeg/ffprobe runner used by audio and video utilities."""

from __future__ import annotations

import subprocess

from src.utils.exceptions import SmartTalkerError


def _run_cmd(cmd: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    """Run a subprocess with proper timeout and cleanup.

    Uses Popen to ensure the child process tree is killed
    on timeout rather than relying on subprocess.run's cleanup.

    Args:
        cmd: Full command list.
        timeout: Max seconds before the process is killed.

    Returns:
        Completed process result.

    Raises:
        FileNotFoundError: If the binary is not found.
        subprocess.CalledProcessError: If the command exits non-zero.
        subprocess.TimeoutExpired: If the process exceeds timeout.
    """
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        raise
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(
            proc.returncode, cmd, output=stdout, stderr=stderr,
        )
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)


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
        return _run_cmd(cmd, timeout)
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
        result = _run_cmd(cmd, timeout)
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise SmartTalkerError(
            message="ffprobe command failed",
            detail=str(exc),
            original_exception=exc,
        ) from exc
