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


# =============================================================================
# Shared FFmpeg Utility Tests
# =============================================================================


class TestSharedFFmpeg:
    """Tests for the shared ffmpeg runner module."""

    def test_run_ffmpeg_import(self):
        """Shared run_ffmpeg can be imported."""
        from src.utils.ffmpeg import run_ffmpeg, run_ffprobe
        assert callable(run_ffmpeg)
        assert callable(run_ffprobe)

    def test_audio_uses_shared_ffmpeg(self):
        """audio.py imports _run_ffmpeg from shared module."""
        from src.utils import audio
        from src.utils.ffmpeg import run_ffmpeg
        assert audio._run_ffmpeg is run_ffmpeg

    def test_video_uses_shared_ffmpeg(self):
        """video.py imports _run_ffmpeg from shared module."""
        from src.utils import video
        from src.utils.ffmpeg import run_ffmpeg
        assert video._run_ffmpeg is run_ffmpeg

    def test_run_ffmpeg_not_found_raises(self):
        """run_ffmpeg raises SmartTalkerError when ffmpeg not found."""
        from unittest.mock import patch
        from src.utils.ffmpeg import run_ffmpeg
        from src.utils.exceptions import SmartTalkerError
        with patch("subprocess.Popen", side_effect=FileNotFoundError):
            with pytest.raises(SmartTalkerError, match="ffmpeg not found"):
                run_ffmpeg(["-version"])


# =============================================================================
# Rate Limiter IP Pruning Tests
# =============================================================================


class TestRateLimiterPruning:
    """Tests for in-memory rate limiter stale IP cleanup."""

    def test_stale_ips_pruned(self, config):
        """IPs with empty timestamp lists are pruned when dict grows large."""
        from src.api.middleware import RedisRateLimitMiddleware
        middleware = RedisRateLimitMiddleware(app=MagicMock(), config=config)

        # Populate with 600 stale IPs (empty lists)
        for i in range(600):
            middleware._requests[f"192.168.1.{i}"] = []

        # Run a check which should trigger pruning
        allowed, count = middleware._check_memory("10.0.0.1", time.time())
        assert allowed
        # All stale IPs should have been pruned
        assert len(middleware._requests) < 100


# =============================================================================
# WebSocket AudioBuffer Tests
# =============================================================================


class TestAudioBufferReset:
    """Tests for AudioBuffer.reset() preserving language/format."""

    def test_reset_clears_chunks(self):
        """reset() clears chunks and started_at."""
        from src.api.websocket import AudioBuffer
        buf = AudioBuffer()
        buf.chunks.append(b"\x00" * 100)
        buf.started_at = 12345.0
        buf.language = "en"
        buf.format = "ogg"
        buf.reset()
        assert buf.chunks == []
        assert buf.started_at == 0.0

    def test_reset_preserves_language_and_format(self):
        """reset() does not clear language or format fields."""
        from src.api.websocket import AudioBuffer
        buf = AudioBuffer()
        buf.language = "en"
        buf.format = "ogg"
        buf.reset()
        # Language and format are preserved after reset
        assert buf.language == "en"
        assert buf.format == "ogg"


# =============================================================================
# CORS Configuration Tests
# =============================================================================


class TestCORSConfig:
    """Tests for CORS configuration builder."""

    def test_wildcard_disables_credentials(self):
        """Wildcard origins must disable allow_credentials (CORS spec)."""
        from src.api.middleware import get_cors_config
        cfg = get_cors_config("*")
        assert cfg["allow_origins"] == ["*"]
        assert cfg["allow_credentials"] is False

    def test_specific_origins_enable_credentials(self):
        """Specific origins can use allow_credentials."""
        from src.api.middleware import get_cors_config
        cfg = get_cors_config("http://localhost:3000,http://example.com")
        assert "http://localhost:3000" in cfg["allow_origins"]
        assert cfg["allow_credentials"] is True


# =============================================================================
# Auth Excluded Paths Tests
# =============================================================================


class TestExcludedPaths:
    """Tests for auth/rate-limit excluded paths."""

    def test_whatsapp_webhook_excluded(self):
        """WhatsApp webhook path is excluded from auth."""
        from src.api.middleware import _is_excluded
        assert _is_excluded("/api/v1/whatsapp/webhook")

    def test_health_excluded(self):
        """Health endpoint is excluded from auth."""
        from src.api.middleware import _is_excluded
        assert _is_excluded("/api/v1/health")

    def test_websocket_paths_excluded(self):
        """WebSocket paths are excluded from HTTP middleware."""
        from src.api.middleware import _is_excluded
        assert _is_excluded("/ws/chat")
        assert _is_excluded("/ws/rtc")

    def test_api_endpoints_not_excluded(self):
        """Normal API endpoints are not excluded."""
        from src.api.middleware import _is_excluded
        assert not _is_excluded("/api/v1/text-to-speech")
        assert not _is_excluded("/api/v1/audio-chat")


# =============================================================================
# Logger LOG_LEVEL Environment Variable Tests
# =============================================================================


class TestLoggerEnvLevel:
    """Tests for logger LOG_LEVEL environment variable support."""

    def test_setup_logger_respects_env(self, monkeypatch):
        """setup_logger reads LOG_LEVEL env var when no explicit level."""
        monkeypatch.setenv("LOG_LEVEL", "WARNING")
        from src.utils.logger import setup_logger
        lgr = setup_logger("test.env.level.unique99")
        assert lgr.level == logging.WARNING

    def test_setup_logger_explicit_overrides_env(self, monkeypatch):
        """Explicit level parameter overrides LOG_LEVEL env var."""
        monkeypatch.setenv("LOG_LEVEL", "WARNING")
        from src.utils.logger import setup_logger
        lgr = setup_logger("test.env.level.unique100", level="ERROR")
        assert lgr.level == logging.ERROR
