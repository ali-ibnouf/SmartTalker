"""Tests for SmartTalker pipeline engines and orchestrator."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# ASR Engine Tests
# =============================================================================


class TestASREngine:
    """Tests for ASREngine."""

    def test_init(self, config):
        """Engine initializes without loading model."""
        from src.pipeline.asr import ASREngine
        engine = ASREngine(config)
        assert not engine.is_loaded

    def test_transcribe_without_load_raises(self, config):
        """Transcribing without load() raises ASRError."""
        from src.pipeline.asr import ASREngine
        from src.utils.exceptions import ASRError
        engine = ASREngine(config)
        with pytest.raises(ASRError, match="not loaded"):
            engine.transcribe("fake.wav")

    def test_transcribe_missing_file_raises(self, config):
        """Transcribing a non-existent file raises ASRError."""
        from src.pipeline.asr import ASREngine
        from src.utils.exceptions import ASRError
        engine = ASREngine(config)
        engine._loaded = True
        engine._model = MagicMock()
        with pytest.raises(ASRError, match="not found"):
            engine.transcribe("nonexistent.wav")

    def test_detect_language_arabic(self, config):
        """Arabic text is detected correctly."""
        from src.pipeline.asr import ASREngine
        engine = ASREngine(config)
        assert engine._detect_language("\u0645\u0631\u062d\u0628\u0627 \u0628\u0643\u0645") == "ar"

    def test_detect_language_english(self, config):
        """English text is detected correctly."""
        from src.pipeline.asr import ASREngine
        engine = ASREngine(config)
        assert engine._detect_language("Hello World") == "en"

    def test_detect_language_empty(self, config):
        """Empty text returns unknown."""
        from src.pipeline.asr import ASREngine
        engine = ASREngine(config)
        assert engine._detect_language("") == "unknown"

    def test_parse_result_empty(self, config):
        """Empty result returns empty TranscriptionResult."""
        from src.pipeline.asr import ASREngine
        engine = ASREngine(config)
        result = engine._parse_result([])
        assert result.text == ""
        assert result.confidence == 0.0

    def test_parse_result_list(self, config):
        """List result is parsed correctly."""
        from src.pipeline.asr import ASREngine
        engine = ASREngine(config)
        raw = [{"text": "hello world", "confidence": 0.95}]
        result = engine._parse_result(raw)
        assert result.text == "hello world"
        assert result.confidence == 0.95

    def test_unload_safe(self, config):
        """Unload is safe when model not loaded."""
        from src.pipeline.asr import ASREngine
        engine = ASREngine(config)
        engine.unload()
        assert not engine.is_loaded


# =============================================================================
# LLM Engine Tests
# =============================================================================


class TestLLMEngine:
    """Tests for LLMEngine."""

    def test_init(self, config):
        """Engine initializes with correct config."""
        from src.pipeline.llm import LLMEngine
        engine = LLMEngine(config)
        assert engine._model == config.llm_model_name

    @pytest.mark.asyncio
    async def test_generate_empty_text_raises(self, config):
        """Empty text raises LLMError."""
        from src.pipeline.llm import LLMEngine
        from src.utils.exceptions import LLMError
        engine = LLMEngine(config)
        with pytest.raises(LLMError, match="empty"):
            await engine.generate(user_text="   ")

    def test_build_messages_arabic(self, config):
        """Arabic prompts are built correctly."""
        from src.pipeline.llm import LLMEngine
        engine = LLMEngine(config)
        messages = engine._build_messages("test", language="ar")
        assert messages[0]["role"] == "system"
        assert any("\u0623\u0646\u062a" in m["content"] for m in messages if m["role"] == "system")

    def test_build_messages_english(self, config):
        """English prompts are built correctly."""
        from src.pipeline.llm import LLMEngine
        engine = LLMEngine(config)
        messages = engine._build_messages("test", language="en")
        assert "helpful AI" in messages[0]["content"]

    def test_extract_text_valid(self, config):
        """Valid response text is extracted."""
        from src.pipeline.llm import LLMEngine
        data = {"message": {"content": "Hello!"}}
        assert LLMEngine._extract_text(data) == "Hello!"

    def test_extract_text_empty_raises(self, config):
        """Empty response raises LLMError."""
        from src.pipeline.llm import LLMEngine, LLMError
        from src.utils.exceptions import LLMError
        data = {"message": {"content": ""}}
        with pytest.raises(LLMError, match="Empty response"):
            LLMEngine._extract_text(data)

    def test_extract_tokens(self, config):
        """Token counts are summed correctly."""
        from src.pipeline.llm import LLMEngine
        data = {"prompt_eval_count": 100, "eval_count": 50}
        assert LLMEngine._extract_tokens(data) == 150

    def test_clear_history(self, config):
        """History clears correctly."""
        from src.pipeline.llm import LLMEngine
        engine = LLMEngine(config)
        engine._sessions["test-session"] = MagicMock()
        engine.clear_history()
        assert len(engine._sessions) == 0


# =============================================================================
# TTS Engine Tests
# =============================================================================


class TestTTSEngine:
    """Tests for TTSEngine."""

    def test_init(self, config):
        """Engine initializes without loading model."""
        from src.pipeline.tts import TTSEngine
        engine = TTSEngine(config)
        assert not engine.is_loaded

    def test_synthesize_without_load_raises(self, config):
        """Synthesize without load() raises TTSError."""
        from src.pipeline.tts import TTSEngine
        from src.utils.exceptions import TTSError
        engine = TTSEngine(config)
        with pytest.raises(TTSError, match="not loaded"):
            engine.synthesize("Hello")

    def test_synthesize_empty_text_raises(self, config):
        """Empty text raises TTSError."""
        from src.pipeline.tts import TTSEngine
        from src.utils.exceptions import TTSError
        engine = TTSEngine(config)
        engine._loaded = True
        engine._model = MagicMock()
        with pytest.raises(TTSError, match="empty"):
            engine.synthesize("")

    def test_synthesize_too_long_raises(self, config):
        """Text exceeding max length raises TTSError."""
        from src.pipeline.tts import TTSEngine
        from src.utils.exceptions import TTSError
        engine = TTSEngine(config)
        engine._loaded = True
        engine._model = MagicMock()
        with pytest.raises(TTSError, match="too long"):
            engine.synthesize("x" * (config.tts_max_text_length + 1))

    def test_unload_safe(self, config):
        """Unload is safe when model not loaded."""
        from src.pipeline.tts import TTSEngine
        engine = TTSEngine(config)
        engine.unload()
        assert not engine.is_loaded


# =============================================================================
# Emotion Engine Tests
# =============================================================================


class TestEmotionEngine:
    """Tests for EmotionEngine."""

    def test_init(self, config):
        """Engine initializes without model."""
        from src.pipeline.emotions import EmotionEngine
        engine = EmotionEngine(config)
        assert not engine.is_loaded

    def test_detect_empty_text(self, config):
        """Empty text returns neutral."""
        from src.pipeline.emotions import EmotionEngine
        engine = EmotionEngine(config)
        result = engine.detect_from_text("")
        assert result.primary_emotion == "neutral"

    def test_detect_arabic_happy(self, config):
        """Arabic happy keyword is detected."""
        from src.pipeline.emotions import EmotionEngine
        engine = EmotionEngine(config)
        result = engine.detect_from_text("\u0634\u0643\u0631\u0627 \u062c\u0632\u064a\u0644\u0627")
        assert result.primary_emotion == "happy"

    def test_detect_english_sad(self, config):
        """English sad keyword is detected."""
        from src.pipeline.emotions import EmotionEngine
        engine = EmotionEngine(config)
        result = engine.detect_from_text("I feel so sorry and sad")
        assert result.primary_emotion == "sad"

    def test_detect_neutral_unknown(self, config):
        """Text with no emotion keywords returns neutral."""
        from src.pipeline.emotions import EmotionEngine
        engine = EmotionEngine(config)
        result = engine.detect_from_text("12345")
        assert result.primary_emotion == "neutral"


# =============================================================================
# Orchestrator Tests
# =============================================================================


class TestSmartTalkerPipeline:
    """Tests for SmartTalkerPipeline."""

    def test_init(self, config):
        """Pipeline initializes all engines."""
        from src.pipeline.orchestrator import SmartTalkerPipeline
        pipeline = SmartTalkerPipeline(config)
        assert pipeline.uptime_seconds >= 0
        assert not pipeline._video_enabled

    def test_health_check(self, config):
        """Health check returns valid structure."""
        from src.pipeline.orchestrator import SmartTalkerPipeline
        pipeline = SmartTalkerPipeline(config)
        health = pipeline.health_check()
        assert "status" in health
        assert "models_loaded" in health
        assert isinstance(health["models_loaded"], dict)
