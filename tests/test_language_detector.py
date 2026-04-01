"""Tests for the langdetect-based language detection service."""

import pytest

from src.services.language_detector import (
    detect_language,
    detect_language_with_confidence,
    SUPPORTED_LANGUAGES,
    DEFAULT_LANGUAGE,
)


class TestDetectLanguage:
    """Test detect_language() with various inputs."""

    def test_arabic(self):
        assert detect_language("أريد تجديد رخصة القيادة") == "ar"

    def test_english(self):
        assert detect_language("I want to renew my driving license") == "en"

    def test_french(self):
        assert detect_language("Je veux renouveler mon permis de conduire") == "fr"

    def test_turkish(self):
        assert detect_language("Ehliyetimi yenilemek istiyorum lütfen") == "tr"

    def test_hindi(self):
        assert detect_language("मैं अपना ड्राइविंग लाइसेंस नवीनीकृत करना चाहता हूं") == "hi"

    def test_urdu(self):
        assert detect_language("میں اپنا ڈرائیونگ لائسنس تجدید کرنا چاہتا ہوں") in ("ur", "ar")

    def test_mixed_arabic_dominant(self):
        result = detect_language("مرحبا hello مرحبا")
        assert result == "ar"

    def test_mixed_english_dominant(self):
        result = detect_language("Hello world, this is a test with some English words in it")
        assert result == "en"

    def test_empty_string(self):
        assert detect_language("") == DEFAULT_LANGUAGE

    def test_none_like_short(self):
        assert detect_language("ok") == DEFAULT_LANGUAGE

    def test_whitespace_only(self):
        assert detect_language("   ") == DEFAULT_LANGUAGE

    def test_numbers_only(self):
        assert detect_language("123456") == DEFAULT_LANGUAGE

    def test_returns_supported_language(self):
        """All detected languages should be in the supported set or defaults."""
        samples = [
            "أريد تجديد رخصة القيادة",
            "I want to renew my license",
            "Je veux renouveler mon permis",
            "Ehliyetimi yenilemek istiyorum",
        ]
        for text in samples:
            lang = detect_language(text)
            assert lang in SUPPORTED_LANGUAGES, f"Unexpected lang '{lang}' for: {text}"


class TestDetectLanguageWithConfidence:
    """Test detect_language_with_confidence() return structure."""

    def test_returns_dict(self):
        result = detect_language_with_confidence("Hello world, how are you?")
        assert isinstance(result, dict)
        assert "language" in result
        assert "confidence" in result
        assert "method" in result

    def test_short_text_default(self):
        result = detect_language_with_confidence("ok")
        assert result["language"] == DEFAULT_LANGUAGE
        assert result["method"] == "default"

    def test_arabic_confidence(self):
        result = detect_language_with_confidence("أريد تجديد رخصة القيادة")
        assert result["language"] == "ar"
        assert result["confidence"] > 0.5
        assert result["method"] == "langdetect"

    def test_english_confidence(self):
        result = detect_language_with_confidence("I want to renew my driving license please")
        assert result["language"] == "en"
        assert result["confidence"] > 0.5

    def test_all_detected_field(self):
        result = detect_language_with_confidence("Bonjour, je veux renouveler mon permis de conduire")
        assert result["method"] == "langdetect"
        if "all_detected" in result:
            assert isinstance(result["all_detected"], list)
            assert len(result["all_detected"]) >= 1
