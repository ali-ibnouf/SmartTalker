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
        """All emotion params have pitch_shift, speed, energy."""
        from src.pipeline.tts import EMOTION_PARAMS
        for emotion, params in EMOTION_PARAMS.items():
            assert "pitch_shift" in params, f"{emotion} missing pitch_shift"
            assert "speed" in params, f"{emotion} missing speed"
            assert "energy" in params, f"{emotion} missing energy"


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
        """TTSResult has correct defaults."""
        from src.pipeline.tts import TTSResult
        result = TTSResult(audio_path="/out/test.wav")
        assert result.sample_rate == 22050
        assert result.duration_s == 0.0
        assert result.latency_ms == 0


class TestTTSEngineCloneVoice:
    """Tests for voice cloning validation."""

    def test_clone_missing_file_raises(self, config):
        """Cloning non-existent file raises TTSError."""
        from src.pipeline.tts import TTSEngine
        from src.utils.exceptions import TTSError
        engine = TTSEngine(config)
        with pytest.raises(TTSError, match="not found"):
            engine.clone_voice("nonexistent.wav", "test_voice")


class TestTTSEngineEmotionParams:
    """Tests for get_emotion_params method."""

    def test_known_emotion(self, config):
        """Known emotion returns correct params."""
        from src.pipeline.tts import TTSEngine
        engine = TTSEngine(config)
        params = engine.get_emotion_params("happy")
        assert params["speed"] > 1.0

    def test_unknown_emotion_returns_neutral(self, config):
        """Unknown emotion falls back to neutral."""
        from src.pipeline.tts import TTSEngine
        engine = TTSEngine(config)
        params = engine.get_emotion_params("unknown_emotion")
        assert params["speed"] == 1.0
