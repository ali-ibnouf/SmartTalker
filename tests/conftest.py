"""Shared test fixtures for SmartTalker."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from typing import Generator

import pytest


@pytest.fixture(autouse=True)
def env_override(tmp_path: Path) -> Generator[None, None, None]:
    """Override environment variables for test isolation.

    Sets storage and model directories to temp paths.
    """
    overrides = {
        "STORAGE_BASE_DIR": str(tmp_path / "outputs"),
        "STATIC_FILES_DIR": str(tmp_path / "files"),
        "ASR_MODEL_DIR": str(tmp_path / "models" / "funasr"),
        "TTS_MODEL_DIR": str(tmp_path / "models" / "cosyvoice"),
        "VIDEO_MODEL_DIR": str(tmp_path / "models" / "echomimic"),
        "UPSCALE_MODEL_DIR": str(tmp_path / "models" / "upscale"),
        "VIDEO_ENABLED": "false",
        "UPSCALE_ENABLED": "false",
        "DEBUG": "true",
        "LOG_LEVEL": "DEBUG",
    }
    for key, val in overrides.items():
        os.environ[key] = val
    yield
    for key in overrides:
        os.environ.pop(key, None)


@pytest.fixture
def config() -> "Settings":
    """Create a test Settings instance."""
    from src.config import Settings
    return Settings()


@pytest.fixture
def mock_pipeline() -> MagicMock:
    """Create a mock SmartTalkerPipeline with all engines mocked."""
    pipeline = MagicMock()

    # Mock ASR
    pipeline._asr.is_loaded = True
    pipeline._asr.transcribe.return_value = MagicMock(
        text="test transcription",
        language="ar",
        confidence=0.95,
        latency_ms=100,
        segments=[],
    )

    # Mock LLM
    pipeline._llm = AsyncMock()

    # Mock TTS
    pipeline._tts.is_loaded = True
    pipeline._tts.list_voices.return_value = []

    # Mock health
    pipeline.health_check.return_value = {
        "status": "healthy",
        "gpu_available": False,
        "gpu_memory_used_mb": 0.0,
        "models_loaded": {"asr": True, "tts": True, "emotion": False, "video": False, "upscale": False},
        "video_enabled": False,
        "upscale_enabled": False,
        "uptime_s": 10.0,
    }

    return pipeline


@pytest.fixture
def sample_audio(tmp_path: Path) -> Path:
    """Create a minimal WAV file for testing."""
    import struct

    audio_path = tmp_path / "test.wav"
    sample_rate = 22050
    duration_s = 1.0
    num_samples = int(sample_rate * duration_s)

    # Generate silence (minimal valid WAV)
    with open(audio_path, "wb") as f:
        # RIFF header
        data_size = num_samples * 2  # 16-bit PCM
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        # fmt chunk
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))       # chunk size
        f.write(struct.pack("<H", 1))        # PCM
        f.write(struct.pack("<H", 1))        # mono
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", sample_rate * 2))  # byte rate
        f.write(struct.pack("<H", 2))        # block align
        f.write(struct.pack("<H", 16))       # bits per sample
        # data chunk
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(b"\x00" * data_size)         # silence

    return audio_path


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    """Create a minimal PNG file for testing."""
    img_path = tmp_path / "test.png"
    # 1x1 pixel white PNG
    png_data = (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde"
        b"\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05"
        b"\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    img_path.write_bytes(png_data)
    return img_path
