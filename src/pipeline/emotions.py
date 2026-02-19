"""Emotion detection and classification engine.

Uses text-based sentiment analysis and optional audio prosody
analysis to detect emotions for pipeline routing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

from src.config import Settings
from src.utils.logger import setup_logger, log_with_latency

logger = setup_logger("pipeline.emotions")

# Supported emotion labels
EMOTION_LABELS: list[str] = [
    "neutral",
    "happy",
    "sad",
    "angry",
    "surprised",
    "fearful",
    "disgusted",
    "contempt",
]

# Arabic emotion keywords for rule-based fallback detection
ARABIC_EMOTION_KEYWORDS: dict[str, list[str]] = {
    "happy": [
        "\u0634\u0643\u0631\u0627",        # shukran (thanks)
        "\u0633\u0639\u064a\u062f",         # sa'id (happy)
        "\u0645\u0645\u062a\u0627\u0632",   # mumtaz (excellent)
        "\u0631\u0627\u0626\u0639",         # ra'i' (wonderful)
        "\u062c\u0645\u064a\u0644",         # jameel (beautiful)
        "\u0641\u0631\u062d",               # farah (joy)
        "\u0645\u0628\u0633\u0648\u0637",   # mabsut (pleased)
    ],
    "sad": [
        "\u062d\u0632\u064a\u0646",         # hazeen (sad)
        "\u0623\u0633\u0641",               # asif (sorry)
        "\u0645\u0624\u0633\u0641",         # mu'assif (sorry)
        "\u0645\u062d\u0632\u0646",         # muhzin (sad)
    ],
    "angry": [
        "\u063a\u0627\u0636\u0628",         # ghadib (angry)
        "\u0645\u0632\u0639\u062c",         # muz'ij (annoyed)
        "\u0627\u0633\u062a\u0641\u0632\u0627\u0632", # istifzaz (provocation)
    ],
    "surprised": [
        "\u0645\u0641\u0627\u062c\u0623\u0629", # mufaja'a (surprise)
        "\u0639\u062c\u064a\u0628",          # 'ajeeb (amazing)
        "\u063a\u0631\u064a\u0628",          # ghareeb (strange)
    ],
    "fearful": [
        "\u062e\u0627\u0626\u0641",         # kha'if (afraid)
        "\u0642\u0644\u0642",               # qalaq (worried)
        "\u062e\u0648\u0641",               # khawf (fear)
    ],
}


@dataclass
class EmotionResult:
    """Result of emotion detection.

    Attributes:
        primary_emotion: Strongest detected emotion.
        confidence: Confidence score (0.0-1.0).
        scores: Per-emotion confidence scores.
        method: Detection method used.
        latency_ms: Processing time in milliseconds.
    """

    primary_emotion: str = "neutral"
    confidence: float = 0.0
    scores: dict[str, float] = None  # type: ignore[assignment]
    method: str = "keyword"
    latency_ms: int = 0

    def __post_init__(self) -> None:
        if self.scores is None:
            self.scores = {label: 0.0 for label in EMOTION_LABELS}


class EmotionEngine:
    """Multi-modal emotion detection engine.

    Uses text-based keyword matching (always) and optional
    transformer-based sentiment analysis when loaded.

    Args:
        config: Application settings.
    """

    def __init__(self, config: Settings) -> None:
        """Initialize the emotion engine.

        Args:
            config: Application settings instance.
        """
        self._config = config
        self._model: Any = None
        self._loaded = False

        logger.info("EmotionEngine initialized")

    @property
    def is_loaded(self) -> bool:
        """Check if the ML model is loaded."""
        return self._loaded

    def load(self) -> None:
        """Load the transformer-based emotion model (optional).

        If loading fails, falls back to keyword-based detection.
        """
        try:
            from transformers import pipeline as hf_pipeline  # type: ignore[import-untyped]

            self._model = hf_pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                top_k=None,
                device=0 if self._config.video_device == "cuda" else -1,
            )
            self._loaded = True
            logger.info("Emotion model loaded (transformer-based)")

        except Exception as exc:
            logger.warning(
                f"Emotion model failed to load, using keyword fallback: {exc}"
            )
            self._loaded = False

    def detect_from_text(self, text: str) -> EmotionResult:
        """Detect emotion from text input.

        Uses transformer model if loaded, otherwise falls back
        to keyword matching.

        Args:
            text: Input text to analyze.

        Returns:
            EmotionResult with emotion label and confidence.
        """
        if not text or not text.strip():
            return EmotionResult(primary_emotion="neutral", confidence=1.0)

        start = time.perf_counter()

        if self._loaded and self._model is not None:
            result = self._detect_transformer(text)
        else:
            result = self._detect_keywords(text)

        result.latency_ms = int((time.perf_counter() - start) * 1000)

        log_with_latency(
            logger,
            "Emotion detected",
            result.latency_ms,
            extra={
                "emotion": result.primary_emotion,
                "confidence": result.confidence,
                "method": result.method,
            },
        )
        return result

    def _detect_transformer(self, text: str) -> EmotionResult:
        """Detect emotion using the transformer model.

        Args:
            text: Input text.

        Returns:
            EmotionResult from model prediction.
        """
        try:
            predictions = self._model(text[:512])
            scores: dict[str, float] = {}

            if predictions and isinstance(predictions, list):
                for pred_list in predictions:
                    if isinstance(pred_list, list):
                        for pred in pred_list:
                            label = pred["label"].lower()
                            score = pred["score"]
                            # Map model labels to our labels
                            mapped = self._map_label(label)
                            scores[mapped] = max(scores.get(mapped, 0.0), score)
                    elif isinstance(pred_list, dict):
                        label = pred_list["label"].lower()
                        score = pred_list["score"]
                        mapped = self._map_label(label)
                        scores[mapped] = max(scores.get(mapped, 0.0), score)

            if not scores:
                return EmotionResult(
                    primary_emotion="neutral",
                    confidence=1.0,
                    method="transformer",
                )

            primary = max(scores, key=scores.get)  # type: ignore[arg-type]
            return EmotionResult(
                primary_emotion=primary,
                confidence=round(scores[primary], 4),
                scores=scores,
                method="transformer",
            )

        except Exception as exc:
            logger.warning(f"Transformer detection failed, falling back: {exc}")
            return self._detect_keywords(text)

    def _detect_keywords(self, text: str) -> EmotionResult:
        """Keyword-based emotion detection — works for Arabic and English.

        Args:
            text: Input text.

        Returns:
            EmotionResult from keyword matching.
        """
        text_lower = text.lower()
        scores: dict[str, float] = {label: 0.0 for label in EMOTION_LABELS}

        # Arabic keyword matching
        for emotion, keywords in ARABIC_EMOTION_KEYWORDS.items():
            for word in keywords:
                if word in text:
                    scores[emotion] += 1.0

        # English keyword matching
        english_keywords: dict[str, list[str]] = {
            "happy": ["thank", "great", "wonderful", "excellent", "amazing", "love", "good"],
            "sad": ["sorry", "sad", "unfortunately", "miss", "lost"],
            "angry": ["angry", "furious", "annoying", "terrible", "worst"],
            "surprised": ["surprised", "unexpected", "wow", "shocking", "unbelievable"],
            "fearful": ["afraid", "scared", "worried", "nervous", "anxious"],
        }

        for emotion, keywords in english_keywords.items():
            for word in keywords:
                if word in text_lower:
                    scores[emotion] += 1.0

        # Find primary emotion
        max_score = max(scores.values())
        if max_score == 0:
            return EmotionResult(
                primary_emotion="neutral",
                confidence=1.0,
                scores=scores,
                method="keyword",
            )

        # Normalize scores
        total = sum(scores.values()) or 1.0
        scores = {k: round(v / total, 4) for k, v in scores.items()}

        primary = max(scores, key=scores.get)  # type: ignore[arg-type]
        return EmotionResult(
            primary_emotion=primary,
            confidence=scores[primary],
            scores=scores,
            method="keyword",
        )

    @staticmethod
    def _map_label(label: str) -> str:
        """Map transformer model labels to our standard labels.

        Args:
            label: Raw model label (e.g., "joy", "anger").

        Returns:
            Mapped standard emotion label.
        """
        mapping: dict[str, str] = {
            "joy": "happy",
            "happiness": "happy",
            "sadness": "sad",
            "anger": "angry",
            "surprise": "surprised",
            "fear": "fearful",
            "disgust": "disgusted",
            "neutral": "neutral",
        }
        return mapping.get(label, label if label in EMOTION_LABELS else "neutral")

    def unload(self) -> None:
        """Free resources by unloading the emotion model."""
        if self._model is not None:
            del self._model
            self._model = None
        self._loaded = False
        logger.info("Emotion model unloaded")
