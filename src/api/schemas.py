"""Pydantic request/response schemas for the API.

Defines: TextRequest, AudioChatRequest, VoiceCloneRequest,
PipelineResponse, HealthResponse, ErrorResponse, AvatarInfo, VoiceInfo.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# =============================================================================
# Request Schemas
# =============================================================================


class TextRequest(BaseModel):
    """Request body for text-to-speech endpoint.

    Attributes:
        text: User's input text.
        avatar_id: Avatar identifier for video generation.
        emotion: Emotion label for response adjustment.
        language: Target response language code.
        voice_id: Optional voice clone ID.
    """

    text: str = Field(..., min_length=1, max_length=2000, description="Input text")
    avatar_id: str = Field(default="default", description="Avatar ID")
    emotion: str = Field(default="neutral", description="Emotion label")
    language: str = Field(default="ar", description="Language code")
    voice_id: Optional[str] = Field(default=None, description="Voice clone ID")


class AudioChatRequest(BaseModel):
    """Metadata for audio-chat multipart upload.

    The audio file is uploaded as a multipart form field.
    This schema handles the JSON metadata portion.

    Attributes:
        avatar_id: Avatar identifier for video generation.
        emotion: Emotion label for response adjustment.
        language: Target response language code.
        voice_id: Optional voice clone ID.
    """

    avatar_id: str = Field(default="default", description="Avatar ID")
    emotion: str = Field(default="neutral", description="Emotion label")
    language: str = Field(default="ar", description="Language code")
    voice_id: Optional[str] = Field(default=None, description="Voice clone ID")


class VoiceCloneRequest(BaseModel):
    """Metadata for voice-clone multipart upload.

    The reference audio file is uploaded as a multipart form field.

    Attributes:
        voice_name: Human-readable name for the new voice.
        language: Primary language of the voice.
    """

    voice_name: str = Field(..., min_length=1, max_length=100, description="Voice name")
    language: str = Field(default="ar", description="Voice language")


# =============================================================================
# Response Schemas
# =============================================================================


class PipelineResponse(BaseModel):
    """Standard response from pipeline processing endpoints.

    Attributes:
        audio_url: URL to the generated audio file.
        video_url: URL to the generated video file (if applicable).
        response_text: LLM-generated response text.
        total_latency_ms: End-to-end processing time.
        breakdown: Per-layer latency breakdown.
        request_id: Correlation ID for the request.
    """

    audio_url: str = Field(..., description="URL to generated audio")
    video_url: Optional[str] = Field(default=None, description="URL to generated video")
    response_text: str = Field(..., description="AI response text")
    total_latency_ms: int = Field(..., description="Total processing time (ms)")
    breakdown: dict[str, int] = Field(default_factory=dict, description="Per-layer latency")
    request_id: Optional[str] = Field(default=None, description="Request correlation ID")


class HealthResponse(BaseModel):
    """System health check response.

    Attributes:
        status: Overall system status (healthy/degraded/down).
        gpu_available: Whether CUDA GPU is available.
        gpu_memory_used_mb: GPU memory usage in MB.
        models_loaded: Per-model load status.
        uptime_s: Server uptime in seconds.
    """

    status: str = Field(..., description="System status")
    gpu_available: bool = Field(..., description="GPU availability")
    gpu_memory_used_mb: float = Field(default=0.0, description="GPU memory used (MB)")
    models_loaded: dict[str, bool] = Field(default_factory=dict, description="Model load status")
    uptime_s: float = Field(default=0.0, description="Uptime in seconds")


class ErrorResponse(BaseModel):
    """Standard error response body.

    Attributes:
        error: Error message.
        detail: Additional error context.
        request_id: Correlation ID for debugging.
    """

    error: str = Field(..., description="Error message")
    detail: Optional[Any] = Field(default=None, description="Error details")
    request_id: Optional[str] = Field(default=None, description="Request ID")


class VoiceCloneResponse(BaseModel):
    """Response after successful voice cloning.

    Attributes:
        voice_id: Unique identifier for the cloned voice.
        name: Human-readable voice name.
        message: Confirmation message.
    """

    voice_id: str = Field(..., description="New voice ID")
    name: str = Field(..., description="Voice name")
    message: str = Field(default="Voice cloned successfully", description="Status")


# =============================================================================
# Info Schemas
# =============================================================================


class AvatarInfo(BaseModel):
    """Metadata for an available avatar.

    Attributes:
        avatar_id: Unique avatar identifier.
        name: Human-readable avatar name.
        image_url: URL to the avatar reference image.
        description: Avatar description.
    """

    avatar_id: str = Field(..., description="Avatar ID")
    name: str = Field(..., description="Avatar name")
    image_url: Optional[str] = Field(default=None, description="Reference image URL")
    description: str = Field(default="", description="Description")


class VoiceInfoResponse(BaseModel):
    """Metadata for an available voice.

    Attributes:
        voice_id: Unique voice identifier.
        name: Human-readable voice name.
        language: Primary voice language.
        description: Voice description.
    """

    voice_id: str = Field(..., description="Voice ID")
    name: str = Field(..., description="Voice name")
    language: str = Field(default="ar", description="Language")
    description: str = Field(default="", description="Description")


class VoiceListResponse(BaseModel):
    """Response containing list of available voices.

    Attributes:
        voices: List of voice info objects.
        count: Total number of voices.
    """

    voices: list[VoiceInfoResponse] = Field(default_factory=list, description="Voices")
    count: int = Field(default=0, description="Total count")
