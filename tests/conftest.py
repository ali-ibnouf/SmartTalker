"""Shared test fixtures for SmartTalker."""

from __future__ import annotations

import os

# Set TESTING before any app imports to skip lifespan service connections
os.environ["TESTING"] = "1"

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from typing import Generator

import pytest


@pytest.fixture(autouse=True)
def env_override(tmp_path: Path) -> Generator[None, None, None]:
    """Override environment variables for test isolation.

    Sets storage and model directories to temp paths.
    Resets the Settings singleton so each test gets fresh config.
    """
    import src.config as cfg_module
    cfg_module._settings_instance = None  # Reset before env changes

    overrides = {
        "STORAGE_BASE_DIR": str(tmp_path / "outputs"),
        "STATIC_FILES_DIR": str(tmp_path / "files"),
        "ASR_MODEL_DIR": str(tmp_path / "models" / "funasr"),
        "TTS_MODEL_DIR": str(tmp_path / "models" / "cosyvoice"),
        "CLIPS_DIR": str(tmp_path / "clips"),
        "KB_STORAGE_DIR": str(tmp_path / "kb"),
        "TRAINING_DB_PATH": str(tmp_path / "training.db"),
        "DATABASE_URL": "sqlite+aiosqlite://",
        "LLM_API_KEY": "test-dashscope-key",  # DashScope API key for Qwen LLM
        "KB_EMBEDDING_API_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "KB_EMBEDDING_API_KEY": "test-dashscope-key",  # DashScope API key for embeddings
        "DEBUG": "true",
        "LOG_LEVEL": "DEBUG",
    }
    for key, val in overrides.items():
        os.environ[key] = val
    yield
    for key in overrides:
        os.environ.pop(key, None)
    cfg_module._settings_instance = None  # Reset after cleanup


@pytest.fixture
def config():
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

    # Mock KB engine
    mock_kb = MagicMock()
    mock_kb.is_loaded = True
    mock_kb.query = AsyncMock(return_value=MagicMock(
        context="", confidence=0.0, has_answer=False, source_chunks=[], latency_ms=0,
    ))
    mock_kb.search = AsyncMock(return_value=MagicMock(
        chunks=[], query="", top_similarity=0.0, latency_ms=0,
    ))
    mock_kb.list_documents = MagicMock(return_value=[])
    pipeline._kb = mock_kb

    # Mock Training engine
    mock_training = MagicMock()
    mock_training.is_loaded = True
    mock_training.should_escalate = AsyncMock(return_value=(False, ""))
    mock_training.get_status = AsyncMock(return_value=MagicMock(
        avatar_id="default", skills=[], overall_progress=0.0,
        is_live=False, total_qa_pairs=0, total_escalations=0, unresolved_escalations=0,
    ))
    pipeline._training = mock_training

    # Mock health (async)
    pipeline.health_check = AsyncMock(return_value={
        "status": "healthy",
        "models_loaded": {
            "asr": True, "tts": True, "emotion": False, "llm": True,
            "kb": True, "training": True,
        },
        "uptime_s": 10.0,
    })

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
