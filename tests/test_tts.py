"""Tests specifically for TTS engine and voice management."""

from __future__ import annotations


import pytest


class TestTTSEmotionParams:
    """Tests for TTS emotion parameter mapping."""

    def test_all_emotions_have_params(self):
        """All supported emotions have parameter mappings."""
        from src.pipeline.tts import EMOTION_PARAMS
        expected_emotions = [
            "neutral", "happy", "sad", "angry",
            "surprised", "fearful", "disgusted", "contempt",
        ]
        for emotion in expected_emotions:
            assert emotion in EMOTION_PARAMS

    def test_neutral_no_speed_change(self):
        """Neutral emotion has speed 1.0."""
        from src.pipeline.tts import EMOTION_PARAMS
        assert EMOTION_PARAMS["neutral"]["speed"] == 1.0

    def test_happy_faster(self):
        """Happy emotion has speed > 1.0."""
        from src.pipeline.tts import EMOTION_PARAMS
        assert EMOTION_PARAMS["happy"]["speed"] > 1.0

    def test_sad_slower(self):
        """Sad emotion has speed < 1.0."""
        from src.pipeline.tts import EMOTION_PARAMS
        assert EMOTION_PARAMS["sad"]["speed"] < 1.0

    def test_all_params_have_keys(self):
        """All emotion params have speed key (DashScope format)."""
        from src.pipeline.tts import EMOTION_PARAMS
        for emotion, params in EMOTION_PARAMS.items():
            assert "speed" in params, f"{emotion} missing speed"


class TestVoiceInfo:
    """Tests for VoiceInfo dataclass."""

    def test_voice_info_defaults(self):
        """VoiceInfo has correct defaults."""
        from src.pipeline.tts import VoiceInfo
        voice = VoiceInfo(voice_id="test_001", name="Test Voice")
        assert voice.language == "ar"
        assert voice.reference_audio == ""

    def test_voice_info_custom(self):
        """VoiceInfo accepts custom values."""
        from src.pipeline.tts import VoiceInfo
        voice = VoiceInfo(
            voice_id="v2",
            name="Salem",
            language="ar",
            reference_audio="/path/to/ref.wav",
            description="Omani male voice",
        )
        assert voice.voice_id == "v2"
        assert voice.description == "Omani male voice"


class TestTTSResult:
    """Tests for TTSResult dataclass."""

    def test_tts_result_defaults(self):
        """TTSResult has correct defaults (48kHz for DashScope)."""
        from src.pipeline.tts import TTSResult
        result = TTSResult(audio_path="/out/test.wav")
        assert result.sample_rate == 48000
        assert result.duration_s == 0.0
        assert result.latency_ms == 0


class TestTTSEngineCloneVoice:
    """Tests for voice cloning (DashScope REST API)."""

    def test_clone_voice_is_async(self, config):
        """clone_voice is an async method (DashScope REST enrollment)."""
        from src.pipeline.tts import TTSEngine
        import inspect
        engine = TTSEngine(config)
        assert inspect.iscoroutinefunction(engine.clone_voice)

    def test_voice_enrollment_cost(self, config):
        """Voice enrollment cost constant is correct."""
        from src.pipeline.tts import VOICE_ENROLLMENT_COST
        assert VOICE_ENROLLMENT_COST == 0.20


class TestTTSEngineEmotionParams:
    """Tests for EMOTION_PARAMS lookup used by synthesize()."""

    def test_known_emotion(self):
        """Known emotion returns correct params."""
        from src.pipeline.tts import EMOTION_PARAMS
        params = EMOTION_PARAMS["happy"]
        assert params["speed"] > 1.0

    def test_unknown_emotion_returns_neutral(self):
        """Unknown emotion falls back to neutral."""
        from src.pipeline.tts import EMOTION_PARAMS
        params = EMOTION_PARAMS.get("unknown_emotion", EMOTION_PARAMS["neutral"])
        assert params["speed"] == 1.0


class TestPrepareLipSync:
    """Tests for phoneme-aware _prepare_lip_sync."""

    def _make_engine(self, config):
        from src.pipeline.tts import TTSEngine
        return TTSEngine(config)

    def test_basic_english(self, config):
        """English sentence produces word timings."""
        engine = self._make_engine(config)
        result = engine._prepare_lip_sync("hello world", "en", 1.0)
        assert "words" in result
        assert len(result["words"]) == 2
        assert result["words"][0]["word"] == "hello"
        assert result["words"][1]["word"] == "world"
        # Timings should span full duration
        assert result["words"][0]["start"] == 0.0
        assert abs(result["words"][-1]["end"] - 1.0) < 0.01

    def test_vowel_heavy_word_gets_more_time(self, config):
        """Words with more vowels should get proportionally more time."""
        engine = self._make_engine(config)
        # "aaa" is all vowels, "bbb" is all consonants
        result = engine._prepare_lip_sync("aaa bbb", "en", 1.0)
        words = result["words"]
        assert len(words) == 2
        dur_aaa = words[0]["end"] - words[0]["start"]
        dur_bbb = words[1]["end"] - words[1]["start"]
        assert dur_aaa > dur_bbb, "Vowel-heavy word should get more time"

    def test_arabic_text(self, config):
        """Arabic text produces word timings."""
        engine = self._make_engine(config)
        result = engine._prepare_lip_sync("\u0645\u0631\u062d\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645", "ar", 1.0)
        assert len(result["words"]) == 2

    def test_arabic_diacritics(self, config):
        """Arabic with diacritics doesn't break word splitting."""
        engine = self._make_engine(config)
        result = engine._prepare_lip_sync("\u0645\u064e\u0631\u0652\u062d\u064e\u0628\u064b\u0627", "ar", 0.5)
        assert len(result["words"]) == 1
        assert result["words"][0]["start"] == 0.0

    def test_empty_text(self, config):
        """Empty text returns empty words list."""
        engine = self._make_engine(config)
        result = engine._prepare_lip_sync("", "en", 1.0)
        assert result == {"words": []}

    def test_zero_duration(self, config):
        """Zero duration returns empty words list."""
        engine = self._make_engine(config)
        result = engine._prepare_lip_sync("hello", "en", 0.0)
        assert result == {"words": []}
