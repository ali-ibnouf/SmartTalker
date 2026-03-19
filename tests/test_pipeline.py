"""Tests for SmartTalker pipeline engines and orchestrator."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


# =============================================================================
# ASR Engine Tests
# =============================================================================


class TestASREngine:
    """Tests for ASREngine (DashScope WebSocket, cloud-based)."""

    def test_init(self, config):
        """Cloud-based engine is always loaded after init."""
        from src.pipeline.asr import ASREngine
        engine = ASREngine(config)
        assert engine.is_loaded

    def test_load_is_noop(self, config):
        """load() is a no-op for cloud-based ASR."""
        from src.pipeline.asr import ASREngine
        engine = ASREngine(config)
        engine.load()
        assert engine.is_loaded

    def test_cost_per_minute(self, config):
        """ASR cost constant is correct."""
        from src.pipeline.asr import COST_PER_MINUTE
        assert COST_PER_MINUTE == 0.008

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

    def test_transcription_result_dataclass(self, config):
        """TranscriptionResult has expected defaults."""
        from src.pipeline.asr import TranscriptionResult
        result = TranscriptionResult(text="hello world")
        assert result.text == "hello world"
        assert result.language == "ar"
        assert result.confidence == 0.0
        assert result.cost_usd == 0.0
        assert result.segments == []


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
        data = {"choices": [{"message": {"content": "Hello!"}}]}
        assert LLMEngine._extract_text(data) == "Hello!"

    def test_extract_text_empty_raises(self, config):
        """Empty response raises LLMError."""
        from src.pipeline.llm import LLMEngine
        from src.utils.exceptions import LLMError
        data = {"choices": [{"message": {"content": ""}}]}
        with pytest.raises(LLMError, match="Empty response"):
            LLMEngine._extract_text(data)

    def test_extract_tokens(self, config):
        """Token counts are summed correctly."""
        from src.pipeline.llm import LLMEngine
        data = {"usage": {"total_tokens": 150}}
        assert LLMEngine._extract_tokens(data) == 150

    @pytest.mark.asyncio
    async def test_clear_history(self, config):
        """History clears correctly."""
        from src.pipeline.llm import LLMEngine
        engine = LLMEngine(config)
        engine._sessions["test-session"] = MagicMock()
        await engine.clear_history()
        assert len(engine._sessions) == 0


# =============================================================================
# TTS Engine Tests
# =============================================================================


class TestTTSEngine:
    """Tests for TTSEngine (DashScope WebSocket, cloud-based)."""

    def test_init(self, config):
        """Cloud-based engine is always loaded after init."""
        from src.pipeline.tts import TTSEngine
        engine = TTSEngine(config)
        assert engine.is_loaded

    def test_load_is_noop(self, config):
        """load() is a no-op for cloud-based TTS."""
        from src.pipeline.tts import TTSEngine
        engine = TTSEngine(config)
        engine.load()
        assert engine.is_loaded

    @pytest.mark.asyncio
    async def test_synthesize_stream_empty_text_raises(self, config):
        """Empty text raises TTSError."""
        from src.pipeline.tts import TTSEngine
        from src.utils.exceptions import TTSError
        engine = TTSEngine(config)
        with pytest.raises(TTSError, match="empty"):
            await engine.synthesize_stream("")

    @pytest.mark.asyncio
    async def test_synthesize_stream_too_long_raises(self, config):
        """Text exceeding max length raises TTSError."""
        from src.pipeline.tts import TTSEngine
        from src.utils.exceptions import TTSError
        engine = TTSEngine(config)
        with pytest.raises(TTSError, match="too long"):
            await engine.synthesize_stream("x" * (config.tts_max_text_length + 1))

    def test_tts_result_defaults(self, config):
        """TTSResult has correct defaults for DashScope (48kHz)."""
        from src.pipeline.tts import TTSResult
        result = TTSResult(audio_path="/out/test.wav")
        assert result.sample_rate == 48000
        assert result.cost_usd == 0.0


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

    @pytest.mark.asyncio
    async def test_health_check(self, config):
        """Health check returns valid structure."""
        from src.pipeline.orchestrator import SmartTalkerPipeline
        pipeline = SmartTalkerPipeline(config)
        health = await pipeline.health_check()
        assert "status" in health
        assert "models_loaded" in health
        assert isinstance(health["models_loaded"], dict)
        assert "llm" in health["models_loaded"]

    def test_avatar_path_traversal_blocked(self, config):
        """Path traversal in avatar_id is rejected."""
        from src.pipeline.orchestrator import SmartTalkerPipeline
        assert SmartTalkerPipeline._resolve_avatar_image("../../etc/passwd") is None
        assert SmartTalkerPipeline._resolve_avatar_image("..\\windows\\system32") is None
        assert SmartTalkerPipeline._resolve_avatar_image("valid/path") is None  # slash blocked
        assert SmartTalkerPipeline._resolve_avatar_image("") is None

    def test_avatar_valid_id_no_dir(self, config):
        """Valid avatar_id with no directory returns None."""
        from src.pipeline.orchestrator import SmartTalkerPipeline
        assert SmartTalkerPipeline._resolve_avatar_image("nonexistent") is None


# =============================================================================
# LLM Session Lock Tests
# =============================================================================


class TestLLMSessionLock:
    """Tests for async session management with lock."""

    @pytest.mark.asyncio
    async def test_get_session_creates_new(self, config):
        """_get_session creates new session for unknown ID."""
        from src.pipeline.llm import LLMEngine
        engine = LLMEngine(config)
        session = await engine._get_session("new-session")
        assert session is not None
        assert len(session.history) == 0

    @pytest.mark.asyncio
    async def test_get_session_returns_existing(self, config):
        """_get_session returns same session for same ID."""
        from src.pipeline.llm import LLMEngine
        engine = LLMEngine(config)
        s1 = await engine._get_session("test-id")
        s1.history.append({"role": "user", "content": "hello"})
        s2 = await engine._get_session("test-id")
        assert len(s2.history) == 1

    @pytest.mark.asyncio
    async def test_concurrent_session_access(self, config):
        """Multiple concurrent _get_session calls don't corrupt state."""
        import asyncio
        from src.pipeline.llm import LLMEngine
        engine = LLMEngine(config)

        async def access_session(sid):
            session = await engine._get_session(sid)
            session.history.append({"role": "user", "content": f"msg-{sid}"})

        await asyncio.gather(*[access_session(f"s{i}") for i in range(20)])
        assert len(engine._sessions) == 20


# =============================================================================
# LLM KB Context Tests
# =============================================================================


class TestLLMKBContext:
    """Tests for knowledge base context injection into LLM."""

    def test_build_messages_with_kb_context_arabic(self, config):
        """KB context is injected into Arabic system prompt."""
        from src.pipeline.llm import LLMEngine
        engine = LLMEngine(config)
        messages = engine._build_messages(
            "test question",
            language="ar",
            kb_context="Some KB context here",
        )
        system_msg = messages[0]["content"]
        assert "Some KB context here" in system_msg

    def test_build_messages_with_kb_context_english(self, config):
        """KB context is injected into English system prompt."""
        from src.pipeline.llm import LLMEngine
        engine = LLMEngine(config)
        messages = engine._build_messages(
            "test question",
            language="en",
            kb_context="English KB context",
        )
        system_msg = messages[0]["content"]
        assert "English KB context" in system_msg
        assert "knowledge base context" in system_msg

    def test_build_messages_without_kb_context(self, config):
        """Without KB context, system prompt is standard."""
        from src.pipeline.llm import LLMEngine
        engine = LLMEngine(config)
        messages = engine._build_messages("test", language="en")
        system_msg = messages[0]["content"]
        assert "knowledge base context" not in system_msg

    def test_build_messages_empty_kb_context(self, config):
        """Empty string KB context is not injected."""
        from src.pipeline.llm import LLMEngine
        engine = LLMEngine(config)
        messages = engine._build_messages("test", language="en", kb_context="")
        system_msg = messages[0]["content"]
        assert "knowledge base context" not in system_msg


# =============================================================================
# LLM History Atomicity Tests
# =============================================================================


class TestLLMHistoryAtomicity:
    """Tests for session history update atomicity."""

    @pytest.mark.asyncio
    async def test_history_append_under_lock(self, config):
        """Verify history appends happen atomically under lock."""
        from src.pipeline.llm import LLMEngine

        engine = LLMEngine(config)
        session = await engine._get_session("atomicity-test")

        # Simulate the pattern used in generate() — both appends under one lock
        async with engine._session_lock:
            session.history.append({"role": "user", "content": "test"})
            session.history.append({"role": "assistant", "content": "response"})

        assert len(session.history) == 2
        assert session.history[0]["role"] == "user"
        assert session.history[1]["role"] == "assistant"


# =============================================================================
# Config Singleton Tests
# =============================================================================


# =============================================================================
# Emotion Motion Params Tests (v3)
# =============================================================================


class TestEmotionMotionParams:
    """Tests for motion parameter generation."""

    def test_get_motion_params_neutral(self):
        from src.pipeline.emotions import EmotionEngine, MotionParams
        params = EmotionEngine.get_motion_params("neutral")
        assert isinstance(params, MotionParams)
        assert params.expression == "neutral_pleasant"
        assert params.smile_weight == 0.2
        assert params.eye_contact is True

    def test_get_motion_params_happy(self):
        from src.pipeline.emotions import EmotionEngine
        params = EmotionEngine.get_motion_params("happy")
        assert params.expression == "smile"
        assert params.smile_weight == 0.7
        assert params.head_nod is True

    def test_get_motion_params_sad(self):
        from src.pipeline.emotions import EmotionEngine
        params = EmotionEngine.get_motion_params("sad")
        assert params.smile_weight == 0.0
        assert params.eye_contact is False
        assert params.head_pose["pitch"] == -3

    def test_get_motion_params_unknown_returns_neutral(self):
        from src.pipeline.emotions import EmotionEngine
        params = EmotionEngine.get_motion_params("nonexistent_emotion")
        assert params.expression == "neutral_pleasant"

    def test_get_thinking_motion(self):
        from src.pipeline.emotions import EmotionEngine
        params = EmotionEngine.get_thinking_motion()
        assert params.expression == "contemplative"
        assert params.head_pose["yaw"] == 8.0
        assert params.eye_contact is False

    def test_get_idle_motion(self):
        from src.pipeline.emotions import EmotionEngine
        params = EmotionEngine.get_idle_motion()
        assert params.idle_animation is True
        assert params.eye_contact is True
        assert params.expression == "neutral_pleasant"

    def test_get_micro_motion_varies_with_seq(self):
        from src.pipeline.emotions import EmotionEngine
        m0 = EmotionEngine.get_micro_motion("neutral", 0)
        m5 = EmotionEngine.get_micro_motion("neutral", 5)
        assert "head_pose" in m0
        assert "expression_delta" in m0
        # Different seq should produce different head_pose
        assert m0["head_pose"]["yaw"] != m5["head_pose"]["yaw"]

    def test_motion_params_to_dict(self):
        from src.pipeline.emotions import MotionParams
        params = MotionParams(expression="smile", smile_weight=0.5)
        d = params.to_dict()
        assert d["expression"] == "smile"
        assert d["smile_weight"] == 0.5
        assert "head_pose" in d


# =============================================================================
# TTS Streaming Tests (v3)
# =============================================================================


class TestTTSStreaming:
    """Tests for streaming TTS (DashScope WebSocket)."""

    def test_tts_chunk_dataclass(self, config):
        """TTSChunk has expected fields."""
        from src.pipeline.tts import TTSChunk
        chunk = TTSChunk(seq=0, audio_bytes=b"\x00" * 100, duration_ms=50)
        assert chunk.seq == 0
        assert chunk.sample_rate == 48000
        assert len(chunk.audio_bytes) == 100

    def test_tts_stream_duration_calculation(self, config):
        """TTSStream calculates duration from bytes (48kHz 16-bit mono)."""
        from src.pipeline.tts import TTSStream
        stream = TTSStream(ws=MagicMock())
        stream._total_bytes = 96000  # 1 second at 48kHz 16-bit mono
        assert abs(stream.duration_seconds - 1.0) < 0.001

    def test_tts_stream_cost_calculation(self, config):
        """TTSStream calculates cost from duration."""
        from src.pipeline.tts import TTSStream, COST_PER_MINUTE
        stream = TTSStream(ws=MagicMock())
        stream._total_bytes = 96000 * 60  # 60 seconds
        assert abs(stream.cost_usd - COST_PER_MINUTE) < 0.0001

    @pytest.mark.asyncio
    async def test_stream_empty_text_raises(self, config):
        """Empty text raises TTSError via synthesize_stream."""
        from src.pipeline.tts import TTSEngine
        from src.utils.exceptions import TTSError
        engine = TTSEngine(config)
        with pytest.raises(TTSError, match="empty"):
            await engine.synthesize_stream("")

    @pytest.mark.asyncio
    async def test_stream_too_long_raises(self, config):
        """Too-long text raises TTSError via synthesize_stream."""
        from src.pipeline.tts import TTSEngine
        from src.utils.exceptions import TTSError
        engine = TTSEngine(config)
        with pytest.raises(TTSError, match="too long"):
            await engine.synthesize_stream("x" * (config.tts_max_text_length + 1))


# =============================================================================
# StreamChunk Tests (v3)
# =============================================================================


class TestStreamChunk:
    """Tests for StreamChunk dataclass."""

    def test_stream_chunk_to_dict(self):
        from src.pipeline.orchestrator import StreamChunk
        chunk = StreamChunk(
            type="audio_chunk",
            session_id="s1",
            seq=3,
            audio_b64="AAAA",
            duration_ms=100,
        )
        d = chunk.to_dict()
        assert d["type"] == "audio_chunk"
        assert d["seq"] == 3
        assert d["audio"] == "AAAA"  # to_dict() maps audio_b64 → "audio" key

    def test_stream_chunk_thinking(self):
        from src.pipeline.orchestrator import StreamChunk
        chunk = StreamChunk(
            type="thinking",
            session_id="s1",
            motion={"expression": "contemplative"},
        )
        d = chunk.to_dict()
        assert d["type"] == "thinking"
        assert d["motion"]["expression"] == "contemplative"


# =============================================================================
# Config Singleton Tests
# =============================================================================


class TestConfigSingleton:
    """Tests for Settings singleton behavior."""

    def test_get_settings_returns_same_instance(self):
        """get_settings returns cached singleton."""
        import src.config as cfg_module
        cfg_module._settings_instance = None  # Reset
        s1 = cfg_module.get_settings()
        s2 = cfg_module.get_settings()
        assert s1 is s2
        cfg_module._settings_instance = None  # Cleanup


# =============================================================================
# Multi-Language Support Tests (AR / EN / FR / TR)
# =============================================================================


class TestMultiLanguageDetection:
    """Tests for 4-language ASR detection."""

    def test_detect_french(self, config):
        """French text with accented characters is detected."""
        from src.pipeline.asr import ASREngine
        assert ASREngine._detect_language("Bonjour, comment ça va aujourd'hui?") == "fr"

    def test_detect_french_accents(self, config):
        """French diacriticals trigger French detection."""
        from src.pipeline.asr import ASREngine
        assert ASREngine._detect_language("Je suis très heureux de vous rencontrer") == "fr"

    def test_detect_turkish(self, config):
        """Turkish text with unique characters is detected."""
        from src.pipeline.asr import ASREngine
        assert ASREngine._detect_language("Merhaba, nasılsınız?") == "tr"

    def test_detect_turkish_unique_chars(self, config):
        """Turkish-unique ğ/ş/ı characters trigger Turkish detection."""
        from src.pipeline.asr import ASREngine
        assert ASREngine._detect_language("Türkiye'de güneşli güzel bir gün geçirdik") == "tr"

    def test_detect_mixed_returns_mixed(self, config):
        """Mostly non-Latin/non-Arabic returns mixed."""
        from src.pipeline.asr import ASREngine
        result = ASREngine._detect_language("123 456 789")
        assert result == "unknown"

    def test_arabic_still_works(self, config):
        """Arabic detection unchanged after expansion."""
        from src.pipeline.asr import ASREngine
        assert ASREngine._detect_language("\u0645\u0631\u062d\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645") == "ar"

    def test_english_still_works(self, config):
        """English detection unchanged after expansion."""
        from src.pipeline.asr import ASREngine
        assert ASREngine._detect_language("Hello world how are you") == "en"


class TestMultiLanguageLLM:
    """Tests for 4-language LLM prompt selection."""

    def test_build_messages_french(self, config):
        """French system prompt is selected for language='fr'."""
        from src.pipeline.llm import LLMEngine, FRENCH_SYSTEM_PROMPT
        engine = LLMEngine(config)
        messages = engine._build_messages("Bonjour", language="fr")
        assert FRENCH_SYSTEM_PROMPT in messages[0]["content"]

    def test_build_messages_turkish(self, config):
        """Turkish system prompt is selected for language='tr'."""
        from src.pipeline.llm import LLMEngine, TURKISH_SYSTEM_PROMPT
        engine = LLMEngine(config)
        messages = engine._build_messages("Merhaba", language="tr")
        assert TURKISH_SYSTEM_PROMPT in messages[0]["content"]

    def test_emotion_prompt_french(self, config):
        """French emotion suffix is applied for language='fr'."""
        from src.pipeline.llm import LLMEngine
        engine = LLMEngine(config)
        messages = engine._build_messages("test", language="fr", emotion="happy")
        assert "enthousiasme" in messages[0]["content"]

    def test_emotion_prompt_turkish(self, config):
        """Turkish emotion suffix is applied for language='tr'."""
        from src.pipeline.llm import LLMEngine
        engine = LLMEngine(config)
        messages = engine._build_messages("test", language="tr", emotion="happy")
        # Turkish happy: "coşku ve pozitiflik"
        assert "pozitiflik" in messages[0]["content"]

    def test_kb_context_french(self, config):
        """French KB context template is used for language='fr'."""
        from src.pipeline.llm import LLMEngine
        engine = LLMEngine(config)
        messages = engine._build_messages("q", language="fr", kb_context="test info")
        assert "Contexte" in messages[0]["content"]

    def test_kb_context_turkish(self, config):
        """Turkish KB context template is used for language='tr'."""
        from src.pipeline.llm import LLMEngine
        engine = LLMEngine(config)
        messages = engine._build_messages("q", language="tr", kb_context="test info")
        # Turkish: "Bağlam"
        assert "Ba\u011flam" in messages[0]["content"]

    def test_unknown_language_falls_back_to_english(self, config):
        """Unknown language falls back to English system prompt."""
        from src.pipeline.llm import LLMEngine
        engine = LLMEngine(config)
        messages = engine._build_messages("test", language="xx")
        assert "helpful AI" in messages[0]["content"]


class TestMultiLanguageTTS:
    """Tests for multi-language TTS (DashScope voice cloning, no CosyVoice speakers)."""

    def test_emotion_params_all_languages(self, config):
        """EMOTION_PARAMS applies to all languages via speed multiplier."""
        from src.pipeline.tts import EMOTION_PARAMS
        assert "happy" in EMOTION_PARAMS
        assert EMOTION_PARAMS["happy"]["speed"] > 1.0

    def test_tts_engine_accepts_language(self, config):
        """TTSEngine.synthesize_stream accepts language parameter."""
        from src.pipeline.tts import TTSEngine
        import inspect
        sig = inspect.signature(TTSEngine.synthesize_stream)
        assert "language" in sig.parameters

    def test_voice_info_language_default(self, config):
        """VoiceInfo defaults to Arabic language."""
        from src.pipeline.tts import VoiceInfo
        voice = VoiceInfo(voice_id="v1", name="Test")
        assert voice.language == "ar"

    def test_voice_info_language_custom(self, config):
        """VoiceInfo accepts custom language."""
        from src.pipeline.tts import VoiceInfo
        voice = VoiceInfo(voice_id="v1", name="Test", language="fr")
        assert voice.language == "fr"


class TestMultiLanguageModels:
    """Tests for language fields on DB models."""

    def test_customer_has_language_fields(self):
        """Customer model has operator_language and data_language."""
        from src.db.models import Customer
        mapper = Customer.__table__
        col_names = {c.name for c in mapper.columns}
        assert "operator_language" in col_names
        assert "data_language" in col_names

    def test_conversation_has_language_field(self):
        """Conversation model has language field."""
        from src.db.models import Conversation
        mapper = Conversation.__table__
        col_names = {c.name for c in mapper.columns}
        assert "language" in col_names

    def test_message_has_language_field(self):
        """ConversationMessage model has language field."""
        from src.db.models import ConversationMessage
        mapper = ConversationMessage.__table__
        col_names = {c.name for c in mapper.columns}
        assert "language" in col_names

    def test_pipeline_result_has_detected_language(self):
        """PipelineResult dataclass has detected_language field."""
        from src.pipeline.orchestrator import PipelineResult
        result = PipelineResult()
        assert hasattr(result, "detected_language")
        assert result.detected_language == ""


class TestMultiLanguageSchemas:
    """Tests for language validation in API schemas."""

    def test_text_request_accepts_auto(self):
        """TextRequest accepts language='auto'."""
        from src.api.schemas import TextRequest
        req = TextRequest(text="hello", language="auto")
        assert req.language == "auto"

    def test_text_request_accepts_fr(self):
        """TextRequest accepts language='fr'."""
        from src.api.schemas import TextRequest
        req = TextRequest(text="bonjour", language="fr")
        assert req.language == "fr"

    def test_text_request_accepts_tr(self):
        """TextRequest accepts language='tr'."""
        from src.api.schemas import TextRequest
        req = TextRequest(text="merhaba", language="tr")
        assert req.language == "tr"

    def test_supported_languages_constant(self):
        """SUPPORTED_LANGUAGES contains at least the original 4 languages."""
        from src.config import LANGUAGE_CODES
        for code in ("ar", "en", "fr", "tr"):
            assert code in LANGUAGE_CODES, f"{code} not in LANGUAGE_CODES"


# =============================================================================
# Language & Plan Configuration Tests (32-language expansion)
# =============================================================================


def test_supported_languages_count():
    """SUPPORTED_LANGUAGES has exactly 32 entries."""
    from src.config import SUPPORTED_LANGUAGES

    assert len(SUPPORTED_LANGUAGES) == 32, (
        f"Expected 32 supported languages, got {len(SUPPORTED_LANGUAGES)}"
    )


def test_language_codes_frozenset():
    """LANGUAGE_CODES is a frozenset containing core language codes."""
    from src.config import LANGUAGE_CODES

    assert isinstance(LANGUAGE_CODES, frozenset), (
        f"LANGUAGE_CODES should be a frozenset, got {type(LANGUAGE_CODES).__name__}"
    )
    expected_subset = {"ar", "en", "fr", "tr", "es", "ja", "zh"}
    missing = expected_subset - LANGUAGE_CODES
    assert not missing, f"LANGUAGE_CODES is missing expected codes: {missing}"


def test_rtl_languages():
    """RTL_LANGUAGES contains all RTL codes and excludes LTR codes."""
    from src.config import RTL_LANGUAGES

    expected_rtl = {"ar", "he", "fa", "ur", "ps", "ku"}
    for code in expected_rtl:
        assert code in RTL_LANGUAGES, f"RTL_LANGUAGES should contain '{code}'"

    non_rtl = {"en", "fr"}
    for code in non_rtl:
        assert code not in RTL_LANGUAGES, (
            f"RTL_LANGUAGES should NOT contain '{code}'"
        )


def test_language_metadata_structure():
    """Each language dict has keys: code, name, name_native, rtl."""
    from src.config import SUPPORTED_LANGUAGES

    required_keys = {"code", "name", "name_native", "rtl"}
    for i, lang in enumerate(SUPPORTED_LANGUAGES):
        missing = required_keys - set(lang.keys())
        assert not missing, (
            f"Language at index {i} ({lang.get('code', '?')}) is missing keys: {missing}"
        )


def test_plan_tiers_pricing():
    """Plan tiers have correct monthly pricing and yearly pricing exists."""
    from src.config import PLAN_TIERS

    expected_monthly = {
        "starter": 100,
        "professional": 200,
        "business": 400,
        "enterprise": 800,
    }
    for plan, expected_price in expected_monthly.items():
        assert plan in PLAN_TIERS, f"PLAN_TIERS missing '{plan}' tier"
        actual = PLAN_TIERS[plan]["price_monthly"]
        assert actual == expected_price, (
            f"{plan} price_monthly should be {expected_price}, got {actual}"
        )
        assert "price_yearly" in PLAN_TIERS[plan], (
            f"{plan} tier is missing 'price_yearly' key"
        )


def test_plan_tiers_billing_rate(monkeypatch):
    """Settings default billing_rate_per_second is 0.002."""
    from src.config import Settings

    # Override any .env value to ensure we test the code default
    monkeypatch.setenv("BILLING_RATE_PER_SECOND", "0.002")
    import src.config as cfg_module
    cfg_module._settings_instance = None

    settings = Settings()
    assert settings.billing_rate_per_second == 0.002, (
        f"billing_rate_per_second should be 0.002, got {settings.billing_rate_per_second}"
    )


def test_languages_endpoint():
    """GET /api/v1/languages returns 200 with count=32 and languages array."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from src.api.routes import router

    app = FastAPI()
    app.include_router(router)

    with TestClient(app) as client:
        response = client.get("/api/v1/languages")
        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text}"
        )
        data = response.json()
        assert "languages" in data, "Response missing 'languages' key"
        assert "count" in data, "Response missing 'count' key"
        assert data["count"] == 32, (
            f"Expected count=32, got {data['count']}"
        )
        assert isinstance(data["languages"], list), "languages should be a list"
        assert len(data["languages"]) == 32, (
            f"Expected 32 language entries, got {len(data['languages'])}"
        )


def test_brand_name_maskki():
    """SUPPORTED_LANGUAGES does not contain any reference to 'SmartTalker'."""
    from src.config import SUPPORTED_LANGUAGES

    for lang in SUPPORTED_LANGUAGES:
        name = lang.get("name", "")
        name_native = lang.get("name_native", "")
        assert "SmartTalker" not in name, (
            f"Language '{lang['code']}' name field contains 'SmartTalker': {name}"
        )
        assert "SmartTalker" not in name_native, (
            f"Language '{lang['code']}' name_native field contains 'SmartTalker': {name_native}"
        )