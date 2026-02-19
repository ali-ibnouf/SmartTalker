"""Custom exception hierarchy for SmartTalker.

Base: SmartTalkerError. Layer-specific: ASRError, LLMError,
TTSError, VideoError, UpscaleError, StorageError,
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

    Catches ASR, LLM, TTS, Video, and Upscale errors uniformly.
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
    """Error during Large Language Model inference (Qwen/Ollama).

    Raised when: Ollama is unreachable, model is not loaded,
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


class VideoError(PipelineError):
    """Error during video generation (EchoMimicV2).

    Raised when: model loading fails, reference image is invalid,
    audio mismatch, or rendering fails.
    """

    def __init__(
        self,
        message: str = "Video generation failed",
        detail: Optional[Any] = None,
        original_exception: Optional[Exception] = None,
    ) -> None:
        super().__init__(message=message, detail=detail, original_exception=original_exception)


class UpscaleError(PipelineError):
    """Error during video upscaling (RealESRGAN/CodeFormer).

    Raised when: model loading fails, input video is corrupt,
    or enhancement process fails.
    """

    def __init__(
        self,
        message: str = "Upscale processing failed",
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
