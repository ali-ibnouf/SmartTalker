"""Tests for video audio extraction utilities.

Tests that require ffmpeg are skipped if ffmpeg is not installed.
Unit tests for validation logic (format/size checks) always run.
"""

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.utils.exceptions import SmartTalkerError
from src.utils.video import (
    extract_audio,
    extract_audio_from_bytes,
    probe_media_info,
    SUPPORTED_VIDEO_FORMATS,
    MAX_VIDEO_SIZE_MB,
)

HAS_FFMPEG = shutil.which("ffmpeg") is not None


# ─── Validation tests (no ffmpeg needed) ─────────────────────────────────────


class TestExtractAudioValidation:
    """Test input validation without needing ffmpeg."""

    def test_file_not_found(self):
        with pytest.raises(SmartTalkerError, match="not found"):
            extract_audio(Path("/nonexistent/video.mp4"))

    def test_unsupported_format(self, tmp_path):
        bad = tmp_path / "file.xyz"
        bad.write_bytes(b"fake")
        with pytest.raises(SmartTalkerError, match="Unsupported video format"):
            extract_audio(bad)

    def test_file_too_large(self, tmp_path):
        big = tmp_path / "huge.mp4"
        big.write_bytes(b"x")
        with patch.object(Path, "stat") as mock_stat:
            mock_stat.return_value = MagicMock(st_size=(MAX_VIDEO_SIZE_MB + 1) * 1024 * 1024)
            with pytest.raises(SmartTalkerError, match="too large"):
                extract_audio(big)

    def test_supported_formats_set(self):
        assert ".mp4" in SUPPORTED_VIDEO_FORMATS
        assert ".webm" in SUPPORTED_VIDEO_FORMATS
        assert ".3gp" in SUPPORTED_VIDEO_FORMATS
        assert ".mov" in SUPPORTED_VIDEO_FORMATS
        assert ".txt" not in SUPPORTED_VIDEO_FORMATS


class TestExtractAudioFromBytesValidation:
    """Test bytes-based extraction validation."""

    def test_empty_bytes_creates_small_file(self, tmp_path):
        """Empty bytes should fail at ffmpeg or size check."""
        with pytest.raises((SmartTalkerError, Exception)):
            extract_audio_from_bytes(b"", "mp4")


class TestProbeMediaInfo:
    """Test probe_media_info validation."""

    def test_file_not_found(self):
        with pytest.raises(SmartTalkerError, match="not found"):
            probe_media_info(Path("/nonexistent/file.mp4"))


# ─── Integration tests (require ffmpeg) ──────────────────────────────────────


@pytest.fixture(scope="module")
def test_video_with_audio():
    """Create a test video with a sine wave audio track using ffmpeg."""
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg not installed")

    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)

    import subprocess
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
            "-f", "lavfi", "-i", "color=c=blue:size=320x240:rate=25",
            "-shortest",
            "-c:v", "libx264", "-c:a", "aac",
            path,
        ],
        capture_output=True,
        timeout=30,
    )
    if result.returncode != 0:
        os.unlink(path)
        pytest.skip(f"Could not create test video: {result.stderr.decode()[:200]}")

    yield Path(path)
    os.unlink(path)


@pytest.fixture(scope="module")
def test_video_no_audio():
    """Create a test video without an audio track."""
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg not installed")

    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)

    import subprocess
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "color=c=red:size=320x240:rate=25",
            "-t", "2", "-an",
            "-c:v", "libx264",
            path,
        ],
        capture_output=True,
        timeout=30,
    )
    if result.returncode != 0:
        os.unlink(path)
        pytest.skip(f"Could not create test video: {result.stderr.decode()[:200]}")

    yield Path(path)
    os.unlink(path)


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
class TestExtractAudioIntegration:
    """Integration tests that require ffmpeg."""

    def test_extract_audio_wav(self, test_video_with_audio):
        output = extract_audio(test_video_with_audio, output_format="wav")
        try:
            assert output.exists()
            assert output.stat().st_size > 1000
            assert output.suffix == ".wav"
        finally:
            output.unlink(missing_ok=True)

    def test_extract_audio_custom_sample_rate(self, test_video_with_audio):
        output = extract_audio(test_video_with_audio, sample_rate=8000)
        try:
            assert output.exists()
            assert output.stat().st_size > 100
        finally:
            output.unlink(missing_ok=True)

    def test_extract_audio_to_specific_path(self, test_video_with_audio, tmp_path):
        out = tmp_path / "extracted.wav"
        result = extract_audio(test_video_with_audio, output_path=out)
        assert result == out
        assert out.exists()
        assert out.stat().st_size > 1000

    def test_no_audio_track_raises(self, test_video_no_audio):
        with pytest.raises(SmartTalkerError, match="no audio track"):
            extract_audio(test_video_no_audio)


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
class TestExtractAudioFromBytesIntegration:
    """Integration tests for bytes-based extraction."""

    def test_extract_from_bytes(self, test_video_with_audio):
        video_bytes = test_video_with_audio.read_bytes()
        audio_bytes = extract_audio_from_bytes(video_bytes, "mp4")
        assert len(audio_bytes) > 1000

    def test_extract_from_bytes_cleans_temp(self, test_video_with_audio):
        """Temp files should be cleaned up after extraction."""
        before = set(Path(tempfile.gettempdir()).glob("*.mp4"))
        video_bytes = test_video_with_audio.read_bytes()
        extract_audio_from_bytes(video_bytes, "mp4")
        after = set(Path(tempfile.gettempdir()).glob("*.mp4"))
        # No new .mp4 files left behind
        assert after == before


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
class TestProbeMediaInfoIntegration:
    """Integration tests for probe_media_info."""

    def test_probe_video_with_audio(self, test_video_with_audio):
        info = probe_media_info(test_video_with_audio)
        assert info["has_audio"] is True
        assert info["has_video"] is True
        assert info["duration"] > 0
        assert info["size_mb"] > 0

    def test_probe_video_no_audio(self, test_video_no_audio):
        info = probe_media_info(test_video_no_audio)
        assert info["has_audio"] is False
        assert info["has_video"] is True
