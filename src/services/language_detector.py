"""Language detection service using langdetect.

Replaces the heuristic Unicode-block detector in ASR with ML-based
detection from the langdetect library (port of Google's language-detection).

Supported languages match the target markets: Arabic, English, French,
Urdu, Hindi, Turkish, Persian, Filipino.
"""

from __future__ import annotations

from langdetect import detect_langs, LangDetectException
from langdetect import DetectorFactory

from src.utils.logger import setup_logger

# Deterministic results across calls
DetectorFactory.seed = 0

logger = setup_logger("services.language_detector")

SUPPORTED_LANGUAGES = {
    "ar": "Arabic",
    "en": "English",
    "fr": "French",
    "ur": "Urdu",
    "hi": "Hindi",
    "tr": "Turkish",
    "fa": "Persian",
    "tl": "Filipino",
}

DEFAULT_LANGUAGE = "ar"


def _arabic_ratio(text: str) -> float:
    """Return the ratio of Arabic-script characters in text."""
    if not text:
        return 0.0
    arabic = sum(1 for c in text if "\u0600" <= c <= "\u06FF" or "\u0750" <= c <= "\u077F")
    return arabic / max(len(text), 1)


def detect_language(text: str) -> str:
    """Detect language from text.

    Returns a language code (ar, en, fr, etc.).
    Falls back to 'ar' if detection fails or text is too short.
    """
    if not text or len(text.strip()) < 3:
        return DEFAULT_LANGUAGE

    try:
        langs = detect_langs(text)
        top = langs[0]

        # Low confidence → fall back to Arabic check
        if top.prob < 0.7:
            if _arabic_ratio(text) > 0.3:
                return "ar"
            return DEFAULT_LANGUAGE

        lang_code = top.lang

        if lang_code in SUPPORTED_LANGUAGES:
            return lang_code

        # Unsupported language → Arabic fallback or English
        if _arabic_ratio(text) > 0.3:
            return "ar"
        return "en"

    except LangDetectException:
        if _arabic_ratio(text) > 0.3:
            return "ar"
        return DEFAULT_LANGUAGE


def detect_language_with_confidence(text: str) -> dict:
    """Return language, confidence, and method for logging/debugging."""
    if not text or len(text.strip()) < 3:
        return {"language": DEFAULT_LANGUAGE, "confidence": 1.0, "method": "default"}

    try:
        langs = detect_langs(text)
        top = langs[0]
        lang = top.lang if top.lang in SUPPORTED_LANGUAGES else "en"
        return {
            "language": lang,
            "confidence": round(top.prob, 3),
            "method": "langdetect",
            "all_detected": [{"lang": l.lang, "prob": round(l.prob, 3)} for l in langs[:3]],
        }
    except LangDetectException:
        if _arabic_ratio(text) > 0.3:
            return {"language": "ar", "confidence": 0.95, "method": "unicode_fallback"}
        return {"language": DEFAULT_LANGUAGE, "confidence": 0.5, "method": "error_fallback"}
