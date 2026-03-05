"""Custom exception hierarchy for SmartTalker.

Base: SmartTalkerError. Layer-specific: ASRError, LLMError,
TTSError, StorageError,
WhatsAppError, WebSocketError.
"""

from __future__ import annotations

from typing import Any, Optional


class SmartTalkerError(Exception):
    """Base exception for all SmartTalker errors.

    Attributes:
        message: Human-readable error description.
        detail: Additional context or structured data about the error.
        original_exception: The underlying exception that caused this error.
    """

    def __init__(
        self,
        message: str = "An unexpected error occurred",
        detail: Optional[Any] = None,
        original_exception: Optional[Exception] = None,
    ) -> None:
        """Initialize SmartTalkerError.

        Args:
            message: Human-readable error description.
            detail: Additional context (dict, str, or any serializable data).
            original_exception: The underlying exception that triggered this error.
        """
        self.message = message
        self.detail = detail
        self.original_exception = original_exception
        super().__init__(self.message)

    def __str__(self) -> str:
        """Return a detailed string representation."""
        parts = [self.message]
        if self.detail:
            parts.append(f"Detail: {self.detail}")
        if self.original_exception:
            parts.append(f"Caused by: {type(self.original_exception).__name__}: {self.original_exception}")
        return " | ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the error for API responses.

        Returns:
            Dictionary with error, detail, and cause fields.
        """
        result: dict[str, Any] = {"error": self.message}
        if self.detail:
            result["detail"] = self.detail
        if self.original_exception:
            result["cause"] = str(self.original_exception)
        return result


# ── Pipeline Layer Errors ────────────────────────────────────────────────────


class PipelineError(SmartTalkerError):
    """Base error for all pipeline-layer failures.

    Catches ASR, LLM, and TTS errors uniformly.
    """

    def __init__(
        self,
        message: str = "Pipeline processing failed",
        detail: Optional[Any] = None,
        original_exception: Optional[Exception] = None,
    ) -> None:
        super().__init__(message=message, detail=detail, original_exception=original_exception)


class ASRError(PipelineError):
    """Error during Automatic Speech Recognition (Fun-ASR).

    Raised when: model loading fails, audio format is unsupported,
    transcription fails, or VAD encounters an issue.
    """

    def __init__(
        self,
        message: str = "ASR processing failed",
        detail: Optional[Any] = None,
        original_exception: Optional[Exception] = None,
    ) -> None:
        super().__init__(message=message, detail=detail, original_exception=original_exception)


class LLMError(PipelineError):
    """Error during Large Language Model inference (Qwen via DashScope).

    Raised when: API is unreachable, model is not available,
    generation times out, or response is malformed.
    """

    def __init__(
        self,
        message: str = "LLM generation failed",
        detail: Optional[Any] = None,
        original_exception: Optional[Exception] = None,
    ) -> None:
        super().__init__(message=message, detail=detail, original_exception=original_exception)


class TTSError(PipelineError):
    """Error during Text-to-Speech synthesis (CosyVoice).

    Raised when: model loading fails, text is too long,
    voice reference is invalid, or synthesis fails.
    """

    def __init__(
        self,
        message: str = "TTS synthesis failed",
        detail: Optional[Any] = None,
        original_exception: Optional[Exception] = None,
    ) -> None:
        super().__init__(message=message, detail=detail, original_exception=original_exception)


# ── Integration Errors ───────────────────────────────────────────────────────


class StorageError(SmartTalkerError):
    """Error during file storage operations.

    Raised when: file save/read fails, disk is full,
    path is invalid, or cleanup fails.
    """

    def __init__(
        self,
        message: str = "Storage operation failed",
        detail: Optional[Any] = None,
        original_exception: Optional[Exception] = None,
    ) -> None:
        super().__init__(message=message, detail=detail, original_exception=original_exception)


class WhatsAppError(SmartTalkerError):
    """Error during WhatsApp API interaction.

    Raised when: webhook verification fails, message sending fails,
    media download fails, or signature validation fails.
    """

    def __init__(
        self,
        message: str = "WhatsApp operation failed",
        detail: Optional[Any] = None,
        original_exception: Optional[Exception] = None,
    ) -> None:
        super().__init__(message=message, detail=detail, original_exception=original_exception)


class WebSocketError(SmartTalkerError):
    """Error during WebSocket communication.

    Raised when: connection drops unexpectedly, message format
    is invalid, or session state is corrupted.
    """

    def __init__(
        self,
        message: str = "WebSocket error",
        detail: Optional[Any] = None,
        original_exception: Optional[Exception] = None,
    ) -> None:
        super().__init__(message=message, detail=detail, original_exception=original_exception)


class WebRTCError(SmartTalkerError):
    """WebRTC signaling or peer connection errors."""

    def __init__(
        self,
        message: str = "WebRTC error",
        detail: Optional[Any] = None,
        original_exception: Optional[Exception] = None,
    ) -> None:
        super().__init__(message=message, detail=detail, original_exception=original_exception)


class KnowledgeBaseError(PipelineError):
    """Error during Knowledge Base / RAG operations.

    Raised when: document ingestion fails, embedding generation fails,
    vector search fails, or ChromaDB is unavailable.
    """

    def __init__(
        self,
        message: str = "Knowledge Base operation failed",
        detail: Optional[Any] = None,
        original_exception: Optional[Exception] = None,
    ) -> None:
        super().__init__(message=message, detail=detail, original_exception=original_exception)


class TrainingError(PipelineError):
    """Error during Training Engine operations.

    Raised when: skill tracking fails, learning pipeline fails,
    or escalation logic encounters an error.
    """

    def __init__(
        self,
        message: str = "Training operation failed",
        detail: Optional[Any] = None,
        original_exception: Optional[Exception] = None,
    ) -> None:
        super().__init__(message=message, detail=detail, original_exception=original_exception)


class GuardrailsError(PipelineError):
    """Error during content guardrails enforcement.

    Raised when: policy check fails, violation recording fails,
    or policy CRUD operations encounter an error.
    """

    def __init__(
        self,
        message: str = "Guardrails operation failed",
        detail: Optional[Any] = None,
        original_exception: Optional[Exception] = None,
    ) -> None:
        super().__init__(message=message, detail=detail, original_exception=original_exception)


class AnalyticsError(PipelineError):
    """Error during analytics computation.

    Raised when: KPI aggregation fails, time-series query fails,
    or drift detection encounters an error.
    """

    def __init__(
        self,
        message: str = "Analytics operation failed",
        detail: Optional[Any] = None,
        original_exception: Optional[Exception] = None,
    ) -> None:
        super().__init__(message=message, detail=detail, original_exception=original_exception)


class BillingError(SmartTalkerError):
    """Error during billing operations.

    Raised when: quota exceeded, metering fails, or usage
    recording encounters an error.
    """

    def __init__(
        self,
        message: str = "Billing operation failed",
        detail: Optional[Any] = None,
        original_exception: Optional[Exception] = None,
    ) -> None:
        super().__init__(message=message, detail=detail, original_exception=original_exception)


class DatabaseError(SmartTalkerError):
    """Error during database operations.

    Raised when: connection fails, query fails, or migration
    encounters an error.
    """

    def __init__(
        self,
        message: str = "Database operation failed",
        detail: Optional[Any] = None,
        original_exception: Optional[Exception] = None,
    ) -> None:
        super().__init__(message=message, detail=detail, original_exception=original_exception)


class AgentError(SmartTalkerError):
    """Error during AI Optimization Agent operations.

    Raised when: detection scan fails, auto-fix fails,
    or pattern tracking encounters an error.
    """

    def __init__(
        self,
        message: str = "Agent operation failed",
        detail: Optional[Any] = None,
        original_exception: Optional[Exception] = None,
    ) -> None:
        super().__init__(message=message, detail=detail, original_exception=original_exception)
