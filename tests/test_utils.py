"""Tests for utility modules: exceptions, logger, storage."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# =============================================================================
# Exception Hierarchy Tests
# =============================================================================


class TestSmartTalkerError:
    """Tests for the base SmartTalkerError."""

    def test_basic(self):
        """Basic error with message only."""
        from src.utils.exceptions import SmartTalkerError
        exc = SmartTalkerError("Something went wrong")
        assert exc.message == "Something went wrong"
        assert exc.detail is None
        assert exc.original_exception is None

    def test_with_detail(self):
        """Error with detail context."""
        from src.utils.exceptions import SmartTalkerError
        exc = SmartTalkerError("Error", detail={"key": "value"})
        assert exc.detail == {"key": "value"}

    def test_with_original_exception(self):
        """Error wrapping an original exception."""
        from src.utils.exceptions import SmartTalkerError
        original = ValueError("root cause")
        exc = SmartTalkerError("Wrapped", original_exception=original)
        assert exc.original_exception is original

    def test_str_representation(self):
        """String includes message, detail, and cause."""
        from src.utils.exceptions import SmartTalkerError
        original = RuntimeError("boom")
        exc = SmartTalkerError("Failed", detail="context", original_exception=original)
        s = str(exc)
        assert "Failed" in s
        assert "context" in s
        assert "RuntimeError" in s
        assert "boom" in s

    def test_str_message_only(self):
        """String with message only has no pipe separators for detail."""
        from src.utils.exceptions import SmartTalkerError
        exc = SmartTalkerError("Simple error")
        assert str(exc) == "Simple error"

    def test_to_dict(self):
        """to_dict serializes correctly."""
        from src.utils.exceptions import SmartTalkerError
        exc = SmartTalkerError("Error", detail="more info")
        d = exc.to_dict()
        assert d["error"] == "Error"
        assert d["detail"] == "more info"

    def test_to_dict_with_cause(self):
        """to_dict includes cause when original exception set."""
        from src.utils.exceptions import SmartTalkerError
        original = TypeError("bad type")
        exc = SmartTalkerError("Wrapped", original_exception=original)
        d = exc.to_dict()
        assert "cause" in d
        assert "bad type" in d["cause"]

    def test_to_dict_minimal(self):
        """to_dict with no detail or cause is minimal."""
        from src.utils.exceptions import SmartTalkerError
        exc = SmartTalkerError("Oops")
        d = exc.to_dict()
        assert d == {"error": "Oops"}

    def test_is_exception(self):
        """SmartTalkerError is a proper Exception."""
        from src.utils.exceptions import SmartTalkerError
        exc = SmartTalkerError("test")
        assert isinstance(exc, Exception)


class TestLayerExceptions:
    """Tests for layer-specific exception subclasses."""

    def test_asr_error(self):
        """ASRError has correct default message."""
        from src.utils.exceptions import ASRError, SmartTalkerError
        exc = ASRError()
        assert "ASR" in exc.message
        assert isinstance(exc, SmartTalkerError)

    def test_llm_error(self):
        """LLMError has correct default message."""
        from src.utils.exceptions import LLMError, SmartTalkerError
        exc = LLMError()
        assert "LLM" in exc.message
        assert isinstance(exc, SmartTalkerError)

    def test_tts_error(self):
        """TTSError has correct default message."""
        from src.utils.exceptions import TTSError, SmartTalkerError
        exc = TTSError()
        assert "TTS" in exc.message
        assert isinstance(exc, SmartTalkerError)

    def test_video_error(self):
        """VideoError has correct default message."""
        from src.utils.exceptions import VideoError, SmartTalkerError
        exc = VideoError()
        assert "Video" in exc.message
        assert isinstance(exc, SmartTalkerError)

    def test_upscale_error(self):
        """UpscaleError has correct default message."""
        from src.utils.exceptions import UpscaleError, SmartTalkerError
        exc = UpscaleError()
        assert "Upscale" in exc.message
        assert isinstance(exc, SmartTalkerError)

    def test_storage_error(self):
        """StorageError has correct default message."""
        from src.utils.exceptions import StorageError, SmartTalkerError
        exc = StorageError()
        assert "Storage" in exc.message
        assert isinstance(exc, SmartTalkerError)

    def test_whatsapp_error(self):
        """WhatsAppError has correct default message."""
        from src.utils.exceptions import WhatsAppError, SmartTalkerError
        exc = WhatsAppError()
        assert "WhatsApp" in exc.message
        assert isinstance(exc, SmartTalkerError)

    def test_websocket_error(self):
        """WebSocketError has correct default message."""
        from src.utils.exceptions import WebSocketError, SmartTalkerError
        exc = WebSocketError()
        assert "WebSocket" in exc.message
        assert isinstance(exc, SmartTalkerError)

    def test_custom_message_override(self):
        """Layer exceptions accept custom messages."""
        from src.utils.exceptions import ASRError
        exc = ASRError(message="Custom ASR failure", detail="model crashed")
        assert exc.message == "Custom ASR failure"
        assert exc.detail == "model crashed"

    def test_exception_chaining(self):
        """Layer exceptions can wrap original exceptions."""
        from src.utils.exceptions import TTSError
        original = FileNotFoundError("model.pth")
        exc = TTSError(
            message="Model load failed",
            original_exception=original,
        )
        assert exc.original_exception is original
        assert "model.pth" in str(exc)


# =============================================================================
# Logger Tests
# =============================================================================


class TestLogger:
    """Tests for structured JSON logger."""

    def test_setup_logger_returns_logger(self):
        """setup_logger returns a Logger instance."""
        from src.utils.logger import setup_logger
        lgr = setup_logger("test.module.unique1")
        assert isinstance(lgr, logging.Logger)
        assert lgr.name == "test.module.unique1"

    def test_setup_logger_no_duplicate_handlers(self):
        """Calling setup_logger twice doesn't add duplicate handlers."""
        from src.utils.logger import setup_logger
        lgr1 = setup_logger("test.module.unique2")
        handler_count = len(lgr1.handlers)
        lgr2 = setup_logger("test.module.unique2")
        assert len(lgr2.handlers) == handler_count
        assert lgr1 is lgr2

    def test_correlation_id_context(self):
        """Correlation ID can be set and retrieved."""
        from src.utils.logger import set_correlation_id, get_correlation_id
        set_correlation_id("test-corr-123")
        assert get_correlation_id() == "test-corr-123"
        # Reset
        set_correlation_id(None)

    def test_log_with_latency(self):
        """log_with_latency calls the logger without raising."""
        from unittest.mock import patch, MagicMock
        from src.utils.logger import log_with_latency
        mock_logger = MagicMock()
        log_with_latency(mock_logger, "Test operation", 42.5)
        mock_logger.info.assert_called_once()
        call_kwargs = mock_logger.info.call_args
        assert call_kwargs[1]["extra"]["latency_ms"] == 42.5

    def test_setup_logger_level(self):
        """Logger respects the level parameter."""
        from src.utils.logger import setup_logger
        lgr = setup_logger("test.level.unique4", level="WARNING")
        assert lgr.level == logging.WARNING


# =============================================================================
# Storage Manager Tests
# =============================================================================


class TestStorageManager:
    """Tests for StorageManager."""

    def test_init_creates_subdirs(self, config):
        """Initialization creates required subdirectories."""
        from src.integrations.storage import StorageManager
        manager = StorageManager(config)
        base = config.storage_base_dir
        for subdir in ["tts", "video", "upscale", "uploads", "whatsapp_media"]:
            assert (base / subdir).is_dir()

    def test_get_stats_empty(self, config):
        """Stats on empty storage returns zeros."""
        from src.integrations.storage import StorageManager
        manager = StorageManager(config)
        stats = manager.get_stats()
        assert stats.total_files == 0
        assert stats.total_size_mb == 0.0

    def test_get_stats_with_files(self, config):
        """Stats reflect actual files."""
        from src.integrations.storage import StorageManager
        manager = StorageManager(config)

        # Create test files large enough to register in MB rounding
        test_file = config.storage_base_dir / "tts" / "test.wav"
        test_file.write_bytes(b"\x00" * (1024 * 1024))  # 1 MB

        stats = manager.get_stats()
        assert stats.total_files >= 1
        assert stats.total_size_mb >= 1.0

    def test_resolve_path_valid(self, config):
        """Valid relative path resolves within base."""
        from src.integrations.storage import StorageManager
        manager = StorageManager(config)
        resolved = manager.resolve_path("tts/output.wav")
        assert str(config.storage_base_dir.resolve()) in str(resolved)

    def test_resolve_path_traversal_blocked(self, config):
        """Path traversal attempt raises ValueError."""
        from src.integrations.storage import StorageManager
        manager = StorageManager(config)
        with pytest.raises(ValueError, match="traversal"):
            manager.resolve_path("../../etc/passwd")

    def test_get_file_url(self, config):
        """File URL is generated correctly."""
        from src.integrations.storage import StorageManager
        manager = StorageManager(config)
        file_path = config.storage_base_dir / "tts" / "output.wav"
        url = manager.get_file_url(file_path, base_url="http://localhost:8000")
        assert url == "http://localhost:8000/files/output.wav"

    def test_get_file_url_no_base(self, config):
        """File URL without base_url uses relative path."""
        from src.integrations.storage import StorageManager
        manager = StorageManager(config)
        file_path = config.storage_base_dir / "tts" / "output.wav"
        url = manager.get_file_url(file_path)
        assert url == "/files/output.wav"

    def test_get_file_url_outside_base(self, config, tmp_path):
        """File outside base dir still gets a URL (fallback)."""
        from src.integrations.storage import StorageManager
        manager = StorageManager(config)
        file_path = tmp_path / "external" / "file.mp4"
        url = manager.get_file_url(file_path)
        assert "/files/file.mp4" in url

    def test_cleanup_old_files(self, config):
        """Cleanup removes old files."""
        from src.integrations.storage import StorageManager
        import os
        manager = StorageManager(config)

        # Create a file and backdate its modification time
        old_file = config.storage_base_dir / "tts" / "old.wav"
        old_file.write_bytes(b"\x00" * 100)
        old_time = time.time() - (config.storage_max_file_age_hours * 3600 + 100)
        os.utime(old_file, (old_time, old_time))

        deleted = manager.cleanup_old_files()
        assert deleted >= 1
        assert not old_file.exists()

    def test_cleanup_keeps_recent_files(self, config):
        """Cleanup keeps recently created files."""
        from src.integrations.storage import StorageManager
        manager = StorageManager(config)

        recent_file = config.storage_base_dir / "tts" / "recent.wav"
        recent_file.write_bytes(b"\x00" * 100)

        deleted = manager.cleanup_old_files()
        assert recent_file.exists()

    def test_clear_all(self, config):
        """clear_all removes all files in subdirectories."""
        from src.integrations.storage import StorageManager
        manager = StorageManager(config)

        # Create files in multiple subdirs
        (config.storage_base_dir / "tts" / "a.wav").write_bytes(b"\x00")
        (config.storage_base_dir / "video" / "b.mp4").write_bytes(b"\x00")

        manager.clear_all()

        # Verify files are gone but directories remain
        assert not (config.storage_base_dir / "tts" / "a.wav").exists()
        assert not (config.storage_base_dir / "video" / "b.mp4").exists()
        assert (config.storage_base_dir / "tts").is_dir()


# =============================================================================
# StorageStats Tests
# =============================================================================


class TestStorageStats:
    """Tests for StorageStats dataclass."""

    def test_defaults(self):
        """StorageStats has correct defaults."""
        from src.integrations.storage import StorageStats
        stats = StorageStats()
        assert stats.total_files == 0
        assert stats.total_size_mb == 0.0
        assert stats.oldest_file_age_hours == 0.0

    def test_custom(self):
        """StorageStats accepts custom values."""
        from src.integrations.storage import StorageStats
        stats = StorageStats(total_files=42, total_size_mb=128.5, oldest_file_age_hours=12.3)
        assert stats.total_files == 42
        assert stats.total_size_mb == 128.5
