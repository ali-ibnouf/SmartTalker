"""Pydantic request/response schemas for the API.

Defines: TextRequest, AudioChatRequest, VoiceCloneRequest,
PipelineResponse, HealthResponse, ErrorResponse, AvatarInfo, VoiceInfo.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from pydantic import BaseModel, EmailStr, Field, field_validator

# Allowed emotion labels (matches EmotionEngine output + "neutral" default)
VALID_EMOTIONS = frozenset({
    "neutral", "happy", "sad", "angry", "surprised", "fearful", "disgusted",
})

# Language code pattern: 2-5 chars (ISO 639-1/BCP 47, e.g. "ar", "en", "zh-CN")
_LANG_RE = re.compile(r"^[a-zA-Z]{2,3}(-[a-zA-Z0-9]{1,8})?$")


# =============================================================================
# Request Schemas
# =============================================================================


class ChatRequest(BaseModel):
    """Request body for REST chat endpoints.

    Attributes:
        text: User's input text.
        avatar_id: Avatar identifier (optional, uses default if omitted).
        language: Target response language code.
    """

    text: str = Field(..., min_length=1, max_length=2000, description="Input text")
    avatar_id: str = Field(default="default", description="Avatar ID")
    language: str = Field(default="auto", description="Language code")

    @field_validator("language")
    @classmethod
    def _validate_language(cls, v: str) -> str:
        if v.lower() == "auto":
            return "auto"
        if not _LANG_RE.match(v):
            raise ValueError(f"Invalid language code: {v}")
        return v.lower()


class ChatResponse(BaseModel):
    """Response from REST chat endpoints.

    Attributes:
        text: AI-generated response text.
        emotion: Detected emotion.
        latency_ms: Total processing time.
        breakdown: Per-layer latency breakdown.
        kb_confidence: Knowledge base confidence score.
        escalated: Whether the query was escalated.
    """

    text: str = Field(..., description="AI response text")
    emotion: str = Field(default="neutral", description="Detected emotion")
    latency_ms: int = Field(default=0, description="Total processing time (ms)")
    breakdown: dict[str, int] = Field(default_factory=dict, description="Per-layer latency")
    kb_confidence: Optional[float] = Field(default=None, description="KB confidence score")
    escalated: bool = Field(default=False, description="Whether escalated to operator")


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

    @field_validator("emotion")
    @classmethod
    def _validate_emotion(cls, v: str) -> str:
        v = v.lower()
        if v not in VALID_EMOTIONS:
            raise ValueError(f"emotion must be one of: {', '.join(sorted(VALID_EMOTIONS))}")
        return v

    @field_validator("language")
    @classmethod
    def _validate_language(cls, v: str) -> str:
        if v.lower() == "auto":
            return "auto"
        if not _LANG_RE.match(v):
            raise ValueError("language must be 'auto' or a valid code (e.g. 'ar', 'en', 'fr', 'tr')")
        return v.lower()


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

    @field_validator("emotion")
    @classmethod
    def _validate_emotion(cls, v: str) -> str:
        v = v.lower()
        if v not in VALID_EMOTIONS:
            raise ValueError(f"emotion must be one of: {', '.join(sorted(VALID_EMOTIONS))}")
        return v

    @field_validator("language")
    @classmethod
    def _validate_language(cls, v: str) -> str:
        if v.lower() == "auto":
            return "auto"
        if not _LANG_RE.match(v):
            raise ValueError("language must be 'auto' or a valid code (e.g. 'ar', 'en', 'fr', 'tr')")
        return v.lower()


class VoiceCloneRequest(BaseModel):
    """Metadata for voice-clone multipart upload.

    The reference audio file is uploaded as a multipart form field.

    Attributes:
        voice_name: Human-readable name for the new voice.
        language: Primary language of the voice.
    """

    voice_name: str = Field(..., min_length=1, max_length=100, description="Voice name")
    language: str = Field(default="ar", description="Voice language")

    @field_validator("language")
    @classmethod
    def _validate_language(cls, v: str) -> str:
        if v.lower() == "auto":
            return "auto"
        if not _LANG_RE.match(v):
            raise ValueError("language must be 'auto' or a valid code (e.g. 'ar', 'en', 'fr', 'tr')")
        return v.lower()


# =============================================================================
# Response Schemas
# =============================================================================


class PipelineResponse(BaseModel):
    """Standard response from pipeline processing endpoints.

    Attributes:
        audio_url: URL to the generated audio file.
        response_text: LLM-generated response text.
        total_latency_ms: End-to-end processing time.
        breakdown: Per-layer latency breakdown.
        request_id: Correlation ID for the request.
    """

    audio_url: str = Field(..., description="URL to generated audio")
    response_text: str = Field(..., description="AI response text")
    total_latency_ms: int = Field(..., description="Total processing time (ms)")
    breakdown: dict[str, int] = Field(default_factory=dict, description="Per-layer latency")
    request_id: Optional[str] = Field(default=None, description="Request correlation ID")


class HealthResponse(BaseModel):
    """System health check response.

    Attributes:
        status: Overall system status (healthy/degraded/down).
        models_loaded: Per-model load status.
        uptime_s: Server uptime in seconds.
    """

    status: str = Field(..., description="System status")
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
    avatar_type: str = Field(default="video", description="Rendering type: video or vrm")
    vrm_url: Optional[str] = Field(default=None, description="URL to VRM model file")


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


# =============================================================================
# Avatar Clip Schemas
# =============================================================================

# Valid avatar clip states
VALID_CLIP_STATES = frozenset({"idle", "thinking", "talking_happy", "talking_sad"})


class AvatarClipResponse(BaseModel):
    """Metadata for a single avatar clip."""

    state: str = Field(..., description="Clip state (idle/thinking/talking_happy/talking_sad)")
    url: str = Field(..., description="URL to the clip file")
    size_bytes: int = Field(default=0, description="File size in bytes")


class AvatarDetailResponse(BaseModel):
    """Detailed avatar information with all clips."""

    avatar_id: str = Field(..., description="Avatar ID")
    clips: list[AvatarClipResponse] = Field(default_factory=list, description="Available clips")


class AvatarListResponse(BaseModel):
    """Response containing list of all avatars."""

    avatars: list[AvatarDetailResponse] = Field(default_factory=list, description="Avatars")
    count: int = Field(default=0, description="Total count")


# =============================================================================
# Knowledge Base Schemas
# =============================================================================


class KBUploadResponse(BaseModel):
    """Response after document upload/ingestion."""

    doc_id: str = Field(..., description="Document ID")
    filename: str = Field(..., description="Original filename")
    doc_type: str = Field(..., description="Document type")
    chunk_count: int = Field(..., description="Number of chunks created")
    message: str = Field(default="Document ingested successfully")


class KBDocumentResponse(BaseModel):
    """Single document metadata."""

    doc_id: str = Field(..., description="Document ID")
    filename: str = Field(..., description="Filename")
    doc_type: str = Field(..., description="Type")
    chunk_count: int = Field(default=0, description="Chunks")
    created_at: float = Field(default=0.0, description="Timestamp")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Extra metadata")


class KBListResponse(BaseModel):
    """List of KB documents."""

    documents: list[KBDocumentResponse] = Field(default_factory=list)
    count: int = Field(default=0)


class KBSearchRequest(BaseModel):
    """Search request for the knowledge base."""

    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of results")


class KBSearchResponse(BaseModel):
    """Search results from KB."""

    chunks: list[dict[str, Any]] = Field(default_factory=list)
    query: str = Field(default="")
    top_similarity: float = Field(default=0.0)
    latency_ms: int = Field(default=0)


# =============================================================================
# Training Schemas
# =============================================================================


class SkillCreateRequest(BaseModel):
    """Request to create a new skill."""

    avatar_id: str = Field(..., min_length=1, max_length=64, description="Avatar ID")
    name: str = Field(..., min_length=1, max_length=200, description="Skill name")
    description: str = Field(default="", max_length=1000, description="Skill description")
    target_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Confidence threshold")


class SkillResponse(BaseModel):
    """Single skill metadata."""

    skill_id: str
    avatar_id: str
    name: str
    description: str = ""
    target_threshold: float = 0.7
    progress: float = 0.0
    qa_count: int = 0


class SkillListResponse(BaseModel):
    """List of skills."""

    skills: list[SkillResponse] = Field(default_factory=list)
    count: int = Field(default=0)


class TrainingStatusResponse(BaseModel):
    """Overall training status for an avatar."""

    avatar_id: str
    skills: list[SkillResponse] = Field(default_factory=list)
    overall_progress: float = 0.0
    is_live: bool = False
    total_qa_pairs: int = 0
    total_escalations: int = 0
    unresolved_escalations: int = 0


class LearnRequest(BaseModel):
    """Request to learn from a human operator response."""

    avatar_id: str = Field(..., min_length=1, description="Avatar ID")
    skill_id: str = Field(..., min_length=1, description="Skill ID")
    question: str = Field(..., min_length=1, max_length=2000, description="Customer question")
    human_answer: str = Field(..., min_length=1, max_length=5000, description="Human response")
    ai_answer: str = Field(default="", max_length=5000, description="AI's original response")
    quality: str = Field(default="none", description="Quality rating: good/bad/none")

    @field_validator("quality")
    @classmethod
    def _validate_quality(cls, v: str) -> str:
        if v not in {"good", "bad", "none"}:
            raise ValueError("quality must be 'good', 'bad', or 'none'")
        return v


class LearnResponse(BaseModel):
    """Response after learning from a human."""

    qa_id: str
    skill_id: str
    updated_progress: float = 0.0
    message: str = "Learned successfully"


class EscalationResponse(BaseModel):
    """Single escalation event."""

    event_id: str
    session_id: str
    avatar_id: str
    skill_id: str
    question: str = ""
    confidence: float = 0.0
    resolved: bool = False
    resolution: str = ""
    created_at: float = 0.0


class EscalationListResponse(BaseModel):
    """List of escalation events."""

    escalations: list[EscalationResponse] = Field(default_factory=list)
    count: int = Field(default=0)


# =============================================================================
# Billing Schemas
# =============================================================================


class BillingUsageResponse(BaseModel):
    """Single usage record."""

    id: str
    session_id: str
    avatar_id: str = ""
    channel: str = ""
    duration_s: float = 0.0
    cost: float = 0.0
    started_at: Optional[str] = None
    ended_at: Optional[str] = None


class BillingUsageListResponse(BaseModel):
    """Usage records list."""

    records: list[BillingUsageResponse] = Field(default_factory=list)
    count: int = Field(default=0)


class BillingQuotaResponse(BaseModel):
    """Quota remaining for a customer."""

    customer_id: str
    remaining_seconds: float = 0.0


class BillingBalanceResponse(BaseModel):
    """Dual balance: plan seconds + extra seconds."""

    plan_seconds_remaining: int = 0
    plan_seconds_total: int = 0
    extra_seconds_remaining: int = 0
    total_remaining: int = 0
    plan_renewal_date: str | None = None
    usage_pct: float = 0.0


class TopupPackageResponse(BaseModel):
    """A top-up package."""

    id: str
    seconds: int
    price: int


class TopupPackageListResponse(BaseModel):
    """List of available top-up packages."""

    packages: list[TopupPackageResponse] = Field(default_factory=list)


class TopupCheckoutResponse(BaseModel):
    """Result of creating a top-up checkout session."""

    checkout_url: str = ""
    package_id: str = ""
    seconds: int = 0
    price: int = 0


class AddTopupRequest(BaseModel):
    """Admin request to add top-up seconds (from webhook callback)."""

    customer_id: str
    package_id: str
    transaction_id: str = ""


class BillingHistoryItem(BaseModel):
    """Single billing history entry."""

    date: str
    type: str  # "subscription" | "topup"
    amount: float
    seconds: int


class BillingHistoryResponse(BaseModel):
    """Billing history list."""

    items: list[BillingHistoryItem] = Field(default_factory=list)
    count: int = 0


# =============================================================================
# Persona Schemas
# =============================================================================


class PersonaResponse(BaseModel):
    """Single persona."""

    persona_id: str
    name: str
    industry: str = "general"
    description: str = ""
    skill_count: int = 0
    is_public: bool = False
    source_avatar_id: str = ""


class PersonaListResponse(BaseModel):
    """List of personas."""

    personas: list[PersonaResponse] = Field(default_factory=list)
    count: int = Field(default=0)


class PersonaExtractRequest(BaseModel):
    """Request to extract a persona from an avatar."""

    avatar_id: str = Field(..., min_length=1, description="Source avatar ID")
    name: str = Field(..., min_length=1, max_length=200, description="Persona name")
    industry: str = Field(default="general", max_length=100, description="Industry category")


class PersonaMatchRequest(BaseModel):
    """Request to find matching personas."""

    industry: str = Field(..., min_length=1, description="Target industry")
    skills_needed: list[str] = Field(default_factory=list, description="Required skills")


class PersonaMatchResponse(BaseModel):
    """Single persona match."""

    persona_id: str
    name: str
    industry: str = "general"
    match_score: float = 0.0
    pre_populated_skills: list[dict[str, str]] = Field(default_factory=list)


class PersonaMatchListResponse(BaseModel):
    """List of persona matches."""

    matches: list[PersonaMatchResponse] = Field(default_factory=list)
    count: int = Field(default=0)


class PersonaApplyRequest(BaseModel):
    """Request to apply a persona to an avatar."""

    avatar_id: str = Field(..., min_length=1, description="Target avatar ID")
    persona_id: str = Field(..., min_length=1, description="Persona ID to apply")


class PersonaApplyResponse(BaseModel):
    """Result of applying a persona."""

    avatar_id: str
    persona_id: str
    skills_applied: int = 0


# =============================================================================
# Node Manager Schemas
# =============================================================================


class RenderNodeResponse(BaseModel):
    """Single render node."""

    node_id: str
    hostname: str
    gpu_type: str = ""
    vram_mb: int = 0
    status: str = "offline"
    current_fps: float = 0.0
    last_heartbeat: float = 0.0
    registered_at: float = 0.0


class RenderNodeListResponse(BaseModel):
    """List of render nodes."""

    nodes: list[RenderNodeResponse] = Field(default_factory=list)
    count: int = Field(default=0)


# =============================================================================
# Admin Schemas
# =============================================================================


class SuspendRequest(BaseModel):
    """Request to suspend a customer."""

    reason: str = Field(default="", max_length=500, description="Suspension reason")


class SuspendResponse(BaseModel):
    """Suspension result."""

    customer_id: str
    suspended: bool
    reason: str = ""


class CustomerCreate(BaseModel):
    """Initial customer registration info."""

    name: str = Field(..., max_length=255)
    email: EmailStr
    company: Optional[str] = Field(default=None, max_length=255)
    operator_language: str = Field(default="ar", description="Language for Operator console mapping (ar, en, es, etc.)")
    data_language: str = Field(default="ar", description="Language for Analytics/Data mapping (ar, en, es, etc.)")


class CustomerResponse(BaseModel):
    """Customer account details."""

    customer_id: str
    name: str
    email: str
    api_key: str
    operator_language: str = "ar"
    data_language: str = "ar"


class AdminSubscriptionRequest(BaseModel):
    """Request to create or update a subscription."""

    plan: str = Field(..., description="Plan tier: starter, professional, business, enterprise")


class AdminSubscriptionResponse(BaseModel):
    """Result of subscription operation."""

    subscription_id: str
    customer_id: str
    plan: str
    monthly_seconds: int
    max_avatars: int
    max_concurrent_sessions: int
    price_monthly: float


# =============================================================================
# Learning Analytics Schemas
# =============================================================================


class QualityStatsResponse(BaseModel):
    """Quality statistics for a skill."""

    skill_id: str
    total: int = 0
    good: int = 0
    bad: int = 0
    none_count: int = 0
    correction_count: int = 0
    bad_ratio: float = 0.0
    improvement_rate: float = 0.0
    effective_threshold: float = 0.7


class ImprovementTimelinePoint(BaseModel):
    """Single point in improvement timeline."""

    date: str
    qa_added: int = 0
    good_count: int = 0
    bad_count: int = 0
    avg_confidence: float = 0.0


class ImprovementTimelineResponse(BaseModel):
    """Improvement timeline data."""

    avatar_id: str
    skill_id: Optional[str] = None
    days: int = 30
    timeline: list[ImprovementTimelinePoint] = Field(default_factory=list)


class ExportResponse(BaseModel):
    """Export result."""

    avatar_id: str
    skill_id: Optional[str] = None
    format: str = "jsonl"
    content: str = ""
    record_count: int = 0


class WeakAreaResponse(BaseModel):
    """Single weak area."""

    skill_id: str
    skill_name: str
    bad_ratio: float = 0.0
    correction_count: int = 0
    effective_threshold: float = 0.7


class WeakAreaListResponse(BaseModel):
    """List of weak areas."""

    avatar_id: str
    weak_areas: list[WeakAreaResponse] = Field(default_factory=list)
    count: int = 0


class ConsolidateResponse(BaseModel):
    """Daily consolidation result."""

    avatar_id: str
    date: str
    skills_consolidated: int = 0


# =============================================================================
# Guardrails Schemas
# =============================================================================


class PolicyRequest(BaseModel):
    """Request to set guardrail policy."""

    blocked_topics: list[str] = Field(default_factory=list, description="Blocked topic keywords")
    required_disclaimers: list[str] = Field(default_factory=list, description="Required disclaimer phrases")
    max_response_length: int = Field(default=2000, ge=100, le=10000, description="Max response length")
    escalation_keywords: list[str] = Field(default_factory=list, description="Keywords triggering escalation")


class PolicyResponse(BaseModel):
    """Guardrail policy configuration."""

    avatar_id: str
    blocked_topics: list[str] = Field(default_factory=list)
    required_disclaimers: list[str] = Field(default_factory=list)
    max_response_length: int = 2000
    escalation_keywords: list[str] = Field(default_factory=list)


class ViolationResponse(BaseModel):
    """Single policy violation."""

    id: str
    avatar_id: str
    session_id: str = ""
    violation_type: str = ""
    original_text: str = ""
    sanitized_text: str = ""
    severity: str = "medium"
    created_at: float = 0.0


class ViolationListResponse(BaseModel):
    """List of violations."""

    violations: list[ViolationResponse] = Field(default_factory=list)
    count: int = 0


class AuditTrailResponse(BaseModel):
    """Guardrails audit trail summary."""

    avatar_id: str
    total_checks: int = 0
    total_violations: int = 0
    violation_types: dict[str, int] = Field(default_factory=dict)
    recent_violations: list[ViolationResponse] = Field(default_factory=list)


# =============================================================================
# Supervisor Schemas
# =============================================================================


class OperatorMetricsResponse(BaseModel):
    """Operator performance metrics."""

    operator_id: str
    total_responses: int = 0
    avg_response_time_ms: float = 0.0
    escalations_resolved: int = 0
    corrections_made: int = 0
    sessions_handled: int = 0
    quality_score: float = 0.0


class OperatorMetricsListResponse(BaseModel):
    """List of operator metrics."""

    operators: list[OperatorMetricsResponse] = Field(default_factory=list)
    count: int = 0


class ActiveSessionSummary(BaseModel):
    """Active session summary."""

    session_id: str
    avatar_id: str = ""
    operator_id: Optional[str] = None
    started_at: float = 0.0
    message_count: int = 0


class ActiveSessionListResponse(BaseModel):
    """List of active sessions."""

    sessions: list[ActiveSessionSummary] = Field(default_factory=list)
    count: int = 0


class SessionTakeoverRequest(BaseModel):
    """Request to take over a session."""

    operator_id: str = Field(..., min_length=1, description="Operator taking over")


class SessionTakeoverResponse(BaseModel):
    """Response after taking over a session."""

    status: str
    session_id: str
    operator_id: str = ""


class SessionReturnResponse(BaseModel):
    """Response after returning a session to AI."""

    status: str
    session_id: str


class TrainingCorrectionItem(BaseModel):
    """A single correction in a training submission."""

    message_id: str = ""
    corrected_response: str = ""


class TrainingSubmitRequest(BaseModel):
    """Request to submit training data from operator review."""

    operator_id: str = Field(..., min_length=1, description="Reviewing operator")
    approved_messages: list[str] = Field(default_factory=list, description="IDs of approved messages")
    corrections: list[TrainingCorrectionItem] = Field(default_factory=list, description="Message corrections")
    notes: str = Field(default="", description="Operator notes")


class TrainingSubmitResponse(BaseModel):
    """Response after submitting training data."""

    status: str
    session_id: str
    approved_count: int = 0
    correction_count: int = 0


class DecisionReviewResponse(BaseModel):
    """Single decision review item."""

    review_id: str
    session_id: str
    avatar_id: str
    question: str = ""
    ai_response: str = ""
    confidence: float = 0.0
    flagged_reason: str = ""
    reviewed: bool = False
    reviewer_id: Optional[str] = None
    verdict: Optional[str] = None
    corrected_response: Optional[str] = None
    created_at: float = 0.0


class ReviewQueueResponse(BaseModel):
    """Review queue."""

    reviews: list[DecisionReviewResponse] = Field(default_factory=list)
    count: int = 0


class ReviewSubmitRequest(BaseModel):
    """Request to submit a review verdict."""

    reviewer_id: str = Field(..., min_length=1, description="Reviewer operator ID")
    verdict: str = Field(..., description="approved or corrected")
    corrected_response: Optional[str] = Field(default=None, max_length=5000, description="Corrected response text")

    @field_validator("verdict")
    @classmethod
    def _validate_verdict(cls, v: str) -> str:
        if v not in {"approved", "corrected"}:
            raise ValueError("verdict must be 'approved' or 'corrected'")
        return v


class ActivityTimelineEntry(BaseModel):
    """Single activity timeline entry."""

    action_id: str
    operator_id: str
    action_type: str = ""
    session_id: str = ""
    details: dict[str, Any] = Field(default_factory=dict)
    created_at: float = 0.0


class ActivityTimelineResponse(BaseModel):
    """Activity timeline."""

    entries: list[ActivityTimelineEntry] = Field(default_factory=list)
    count: int = 0


# =============================================================================
# Analytics Schemas
# =============================================================================


class KPISnapshotResponse(BaseModel):
    """KPI snapshot."""

    avatar_id: str
    period: str = "daily"
    total_conversations: int = 0
    total_messages: int = 0
    avg_response_time_ms: float = 0.0
    avg_kb_confidence: float = 0.0
    escalation_rate: float = 0.0
    autonomy_percent: float = 0.0
    resolution_time_avg_s: float = 0.0
    accuracy_score: float = 0.0
    total_cost: float = 0.0


class TimeseriesPoint(BaseModel):
    """Single timeseries point."""

    date: str
    value: float = 0.0


class TimeseriesResponse(BaseModel):
    """Timeseries data."""

    avatar_id: str
    metric: str
    period: str = "daily"
    points: list[TimeseriesPoint] = Field(default_factory=list)


class DriftAlertResponse(BaseModel):
    """Single drift alert."""

    metric: str
    baseline_value: float = 0.0
    current_value: float = 0.0
    change_percent: float = 0.0
    severity: str = "warning"


class DriftAlertListResponse(BaseModel):
    """List of drift alerts."""

    avatar_id: str
    alerts: list[DriftAlertResponse] = Field(default_factory=list)
    count: int = 0


class DashboardDataResponse(BaseModel):
    """Complete analytics dashboard data."""

    avatar_id: str
    kpis: KPISnapshotResponse
    trends: dict[str, list[TimeseriesPoint]] = Field(default_factory=dict)
    top_skills: list[dict[str, Any]] = Field(default_factory=list)
    bottom_skills: list[dict[str, Any]] = Field(default_factory=list)


class ReportExportResponse(BaseModel):
    """Exported analytics report."""

    avatar_id: str
    period_days: int = 30
    generated_at: float = 0.0
    kpis: KPISnapshotResponse
    skill_breakdown: list[dict[str, Any]] = Field(default_factory=list)
    daily_trends: dict[str, list[TimeseriesPoint]] = Field(default_factory=dict)


# =============================================================================
# VRM Avatar
# =============================================================================


class VRMUploadResponse(BaseModel):
    """Response after uploading a VRM model file."""

    avatar_id: str
    vrm_url: str
    file_size_bytes: int
    message: str = "VRM model uploaded successfully"


class VRMInfoResponse(BaseModel):
    """VRM avatar metadata."""

    avatar_id: str
    avatar_type: str
    vrm_url: Optional[str] = None
    has_vrm: bool = False


class AvatarTypeRequest(BaseModel):
    """Request to switch avatar rendering type."""

    avatar_type: str = Field(..., pattern="^(video|vrm)$", description="video or vrm")


# =============================================================================
# Phase 2: Employee Schemas
# =============================================================================


class EmployeeCreateRequest(BaseModel):
    """Request to create a digital employee."""

    name: str = Field(..., min_length=1, max_length=255, description="Employee name")
    avatar_id: Optional[str] = Field(default=None, description="Linked avatar ID")
    role_title: str = Field(default="", max_length=255, description="Job title")
    role_description: str = Field(default="", max_length=2000, description="Role description")
    personality: dict[str, Any] = Field(default_factory=dict, description="Personality traits")
    guardrails: dict[str, Any] = Field(default_factory=dict, description="Guardrail config")
    language: str = Field(default="ar", description="Primary language")


class EmployeeUpdateRequest(BaseModel):
    """Request to update employee fields."""

    name: Optional[str] = Field(default=None, max_length=255)
    avatar_id: Optional[str] = None
    role_title: Optional[str] = Field(default=None, max_length=255)
    role_description: Optional[str] = Field(default=None, max_length=2000)
    personality: Optional[dict[str, Any]] = None
    guardrails: Optional[dict[str, Any]] = None
    language: Optional[str] = None
    is_active: Optional[bool] = None


class EmployeeResponse(BaseModel):
    """Employee details."""

    id: str
    customer_id: str
    avatar_id: Optional[str] = None
    name: str
    role_title: str = ""
    role_description: str = ""
    personality: dict[str, Any] = Field(default_factory=dict)
    guardrails: dict[str, Any] = Field(default_factory=dict)
    language: str = "ar"
    is_active: bool = True
    knowledge_count: int = 0
    tool_count: int = 0
    created_at: Optional[str] = None


class EmployeeListResponse(BaseModel):
    """List of employees."""

    employees: list[EmployeeResponse] = Field(default_factory=list)
    count: int = 0


# =============================================================================
# Phase 2: Tool Schemas
# =============================================================================


class ToolCreateRequest(BaseModel):
    """Request to create a custom API tool."""

    tool_id: str = Field(..., min_length=1, max_length=128, pattern="^[a-z0-9_-]+$", description="Slug identifier")
    name: str = Field(..., min_length=1, max_length=255, description="Display name")
    description: str = Field(default="", max_length=2000, description="Tool description")
    category: str = Field(default="custom", max_length=64, description="Category")
    input_schema: dict[str, Any] = Field(default_factory=dict, description="JSON Schema for tool parameters")
    api_url: str = Field(default="", description="API endpoint URL")
    api_method: str = Field(default="POST", description="HTTP method")
    api_headers: dict[str, str] = Field(default_factory=dict, description="HTTP headers")
    api_body_template: dict[str, Any] = Field(default_factory=dict, description="Body template with {{param}} placeholders")
    response_mapping: dict[str, Any] = Field(default_factory=dict, description="Response field mapping")
    timeout_ms: int = Field(default=10000, ge=1000, le=60000, description="Timeout in ms")
    requires_confirmation: bool = Field(default=False, description="Require visitor confirmation before execution")

    @field_validator("api_method")
    @classmethod
    def _validate_method(cls, v: str) -> str:
        v = v.upper()
        if v not in {"GET", "POST", "PUT", "DELETE", "PATCH"}:
            raise ValueError("api_method must be GET, POST, PUT, DELETE, or PATCH")
        return v


class ToolUpdateRequest(BaseModel):
    """Request to update a custom tool."""

    name: Optional[str] = Field(default=None, max_length=255)
    description: Optional[str] = Field(default=None, max_length=2000)
    category: Optional[str] = Field(default=None, max_length=64)
    input_schema: Optional[dict[str, Any]] = None
    api_url: Optional[str] = None
    api_method: Optional[str] = None
    api_headers: Optional[dict[str, str]] = None
    api_body_template: Optional[dict[str, Any]] = None
    response_mapping: Optional[dict[str, Any]] = None
    timeout_ms: Optional[int] = Field(default=None, ge=1000, le=60000)
    requires_confirmation: Optional[bool] = None
    is_active: Optional[bool] = None


class ToolResponse(BaseModel):
    """Custom tool details."""

    id: str
    tool_id: str
    customer_id: str
    name: str
    description: str = ""
    category: str = "custom"
    input_schema: dict[str, Any] = Field(default_factory=dict)
    api_url: str = ""
    api_method: str = "POST"
    api_headers: dict[str, str] = Field(default_factory=dict)
    api_body_template: dict[str, Any] = Field(default_factory=dict)
    response_mapping: dict[str, Any] = Field(default_factory=dict)
    timeout_ms: int = 10000
    requires_confirmation: bool = False
    is_active: bool = True
    created_at: Optional[str] = None


class ToolListResponse(BaseModel):
    """List of tools."""

    tools: list[ToolResponse] = Field(default_factory=list)
    count: int = 0


class ToolTestRequest(BaseModel):
    """Request to test-execute a tool."""

    parameters: dict[str, Any] = Field(default_factory=dict, description="Test input parameters")


class ToolTestResponse(BaseModel):
    """Result of a tool test execution."""

    status: str  # success, error, timeout
    output: dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    execution_time_ms: int = 0


class ToolLogResponse(BaseModel):
    """Single tool execution log entry."""

    id: str
    tool_id: str
    employee_id: str = ""
    session_id: str = ""
    visitor_id: str = ""
    input_data: dict[str, Any] = Field(default_factory=dict)
    output_data: dict[str, Any] = Field(default_factory=dict)
    status: str = "success"
    error_message: str = ""
    execution_time_ms: int = 0
    created_at: Optional[str] = None


class ToolLogListResponse(BaseModel):
    """List of tool execution logs."""

    logs: list[ToolLogResponse] = Field(default_factory=list)
    count: int = 0


# =============================================================================
# Phase 2: Workflow Schemas
# =============================================================================


class WorkflowStepSchema(BaseModel):
    """A single step in a workflow definition."""

    type: str = Field(..., description="Step type: ask_visitor, call_tool, ai_decision, condition, set_variable, send_notification, wait, escalate")
    config: dict[str, Any] = Field(default_factory=dict, description="Step-specific configuration")


class WorkflowCreateRequest(BaseModel):
    """Request to create a workflow."""

    name: str = Field(..., min_length=1, max_length=255, description="Workflow name")
    description: str = Field(default="", max_length=2000, description="Description")
    employee_id: str = Field(default="", description="Target employee")
    trigger_type: str = Field(default="manual", description="Trigger type: manual, keyword, intent, tool_result")
    trigger_config: dict[str, Any] = Field(default_factory=dict, description="Trigger configuration")
    steps: list[WorkflowStepSchema] = Field(default_factory=list, description="Ordered step list")

    @field_validator("trigger_type")
    @classmethod
    def _validate_trigger(cls, v: str) -> str:
        if v not in {"manual", "keyword", "intent", "tool_result"}:
            raise ValueError("trigger_type must be manual, keyword, intent, or tool_result")
        return v


class WorkflowUpdateRequest(BaseModel):
    """Request to update a workflow."""

    name: Optional[str] = Field(default=None, max_length=255)
    description: Optional[str] = Field(default=None, max_length=2000)
    employee_id: Optional[str] = None
    trigger_type: Optional[str] = None
    trigger_config: Optional[dict[str, Any]] = None
    steps: Optional[list[WorkflowStepSchema]] = None


class WorkflowResponse(BaseModel):
    """Workflow details."""

    id: str
    customer_id: str
    employee_id: str = ""
    name: str
    description: str = ""
    trigger_type: str = "manual"
    trigger_config: dict[str, Any] = Field(default_factory=dict)
    steps: list[WorkflowStepSchema] = Field(default_factory=list)
    is_active: bool = False
    template_id: str = ""
    created_at: Optional[str] = None


class WorkflowListResponse(BaseModel):
    """List of workflows."""

    workflows: list[WorkflowResponse] = Field(default_factory=list)
    count: int = 0


class WorkflowTemplateResponse(BaseModel):
    """Built-in workflow template."""

    template_id: str
    name: str
    description: str = ""
    trigger_type: str = "manual"
    steps: list[WorkflowStepSchema] = Field(default_factory=list)


class WorkflowTemplateListResponse(BaseModel):
    """List of workflow templates."""

    templates: list[WorkflowTemplateResponse] = Field(default_factory=list)
    count: int = 0


class WorkflowExecutionResponse(BaseModel):
    """Workflow execution instance."""

    id: str
    workflow_id: str
    session_id: str = ""
    visitor_id: str = ""
    status: str = "running"
    current_step: int = 0
    context: dict[str, Any] = Field(default_factory=dict)
    error: str = ""
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class WorkflowExecutionListResponse(BaseModel):
    """List of workflow executions."""

    executions: list[WorkflowExecutionResponse] = Field(default_factory=list)
    count: int = 0


# =============================================================================
# Phase 2: Learning Queue Schemas
# =============================================================================


class LearningQueueItem(BaseModel):
    """Single learning queue entry."""

    id: str
    employee_id: str
    customer_id: str = ""
    session_id: str = ""
    learning_type: str  # qa_pair, preference, correction
    old_value: str = ""
    new_value: str = ""
    confidence: float = 0.0
    status: str = "pending"
    source: str = "auto"
    created_at: Optional[str] = None


class LearningQueueResponse(BaseModel):
    """Learning queue list."""

    items: list[LearningQueueItem] = Field(default_factory=list)
    count: int = 0


class LearningReviewRequest(BaseModel):
    """Request to approve/reject a learning entry."""

    edited_value: Optional[str] = Field(default=None, max_length=5000, description="Edited value before approving")


class LearningStatsResponse(BaseModel):
    """Learning queue statistics."""

    pending_count: int = 0
    approved_count: int = 0
    rejected_count: int = 0
    auto_approved_count: int = 0
    avg_confidence: float = 0.0


# =============================================================================
# Phase 2: Visitor Schemas
# =============================================================================


class VisitorProfileResponse(BaseModel):
    """Visitor profile details."""

    id: str
    visitor_id: str
    employee_id: str = ""
    customer_id: str = ""
    display_name: str = ""
    email: str = ""
    phone: str = ""
    language: str = ""
    tags: list[str] = Field(default_factory=list)
    interaction_count: int = 0
    last_seen: Optional[str] = None


class VisitorMemoryResponse(BaseModel):
    """Single visitor memory entry."""

    id: str
    visitor_id: str
    employee_id: str = ""
    memory_type: str
    content: str
    importance: float = 0.5
    source_session: str = ""
    created_at: Optional[str] = None


class VisitorMemoryListResponse(BaseModel):
    """List of visitor memories."""

    memories: list[VisitorMemoryResponse] = Field(default_factory=list)
    count: int = 0


# =============================================================================
# Phase 2: Cost Tracking Schemas
# =============================================================================


class CostRecordResponse(BaseModel):
    """Single API cost record."""

    id: str
    service: str
    customer_id: str = ""
    session_id: str = ""
    cost_usd: float = 0.0
    tokens_used: int = 0
    duration_ms: int = 0
    details: dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None


class CostTotalResponse(BaseModel):
    """Total platform costs."""

    total_cost_usd: float = 0.0
    period_start: str = ""
    period_end: str = ""
    record_count: int = 0


class CostBreakdownItem(BaseModel):
    """Cost breakdown by service."""

    service: str
    total_cost_usd: float = 0.0
    record_count: int = 0
    avg_cost_usd: float = 0.0


class CostBreakdownResponse(BaseModel):
    """Cost breakdown by service."""

    breakdown: list[CostBreakdownItem] = Field(default_factory=list)
    total_cost_usd: float = 0.0


class CostByCustomerItem(BaseModel):
    """Cost per customer."""

    customer_id: str
    customer_name: str = ""
    total_cost_usd: float = 0.0
    record_count: int = 0


class CostByCustomerResponse(BaseModel):
    """Costs grouped by customer."""

    customers: list[CostByCustomerItem] = Field(default_factory=list)
    total_cost_usd: float = 0.0


class CostMarginResponse(BaseModel):
    """Revenue vs. cost margin analysis."""

    total_revenue_usd: float = 0.0
    total_cost_usd: float = 0.0
    margin_usd: float = 0.0
    margin_percent: float = 0.0
    period_start: str = ""
    period_end: str = ""


class RunPodCostResponse(BaseModel):
    """RunPod-specific cost metrics."""

    total_cost_usd: float = 0.0
    total_jobs: int = 0
    avg_execution_ms: int = 0
    total_execution_ms: int = 0
    preprocess_jobs: int = 0
    render_jobs: int = 0


# =============================================================================
# KB Extended Schemas
# =============================================================================


class KBIngestTextRequest(BaseModel):
    """Manual text ingestion request."""

    title: str = Field(..., min_length=1, max_length=500, description="Entry title")
    content: str = Field(..., min_length=1, description="Text content to ingest")
    tags: list[str] = Field(default_factory=list, description="Optional tags")


class KBScrapeRequest(BaseModel):
    """Web scraping request."""

    url: str = Field(..., min_length=1, description="URL to scrape")


class KBKnowledgeItem(BaseModel):
    """Single knowledge item from EmployeeKnowledge."""

    id: str
    employee_id: str
    category: str = "general"
    question: str
    answer: str
    approved: bool = False
    times_used: int = 0
    success_rate: float = 0.0
    created_at: str = ""


class KBKnowledgeListResponse(BaseModel):
    """Paginated list of knowledge items."""

    items: list[KBKnowledgeItem] = Field(default_factory=list)
    count: int = 0


class KBKnowledgeUpdateRequest(BaseModel):
    """Update a knowledge item."""

    question: str | None = None
    answer: str | None = None
    category: str | None = None


class KBAnalyticsGrowth(BaseModel):
    """Single day in KB growth chart."""

    date: str
    documents: int = 0
    knowledge: int = 0


class KBAnalyticsResponse(BaseModel):
    """KB performance analytics."""

    total_documents: int = 0
    total_chunks: int = 0
    total_knowledge_items: int = 0
    avg_confidence: float = 0.0
    unanswered_count: int = 0
    growth: list[KBAnalyticsGrowth] = Field(default_factory=list)


# =============================================================================
# Admin Overview Schemas
# =============================================================================


class AdminTrainingCustomerRow(BaseModel):
    """Per-customer training stats row."""

    customer_id: str
    customer_name: str = ""
    pending: int = 0
    approved: int = 0
    rejected: int = 0
    avg_confidence: float = 0.0


class AdminTrainingRecentItem(BaseModel):
    """Recent pending learning entry for admin view."""

    id: str
    customer_id: str
    customer_name: str = ""
    employee_id: str = ""
    learning_type: str = ""
    old_value: str = ""
    new_value: str = ""
    confidence: float = 0.0
    created_at: str = ""


class AdminTrainingOverview(BaseModel):
    """Aggregated training stats across all customers."""

    total_pending: int = 0
    total_approved: int = 0
    total_rejected: int = 0
    avg_confidence: float = 0.0
    customers: list[AdminTrainingCustomerRow] = Field(default_factory=list)
    recent_pending: list[AdminTrainingRecentItem] = Field(default_factory=list)


class AdminKnowledgeCustomerRow(BaseModel):
    """Per-customer knowledge stats row."""

    customer_id: str
    customer_name: str = ""
    knowledge_count: int = 0
    avg_success_rate: float = 0.0


class AdminKnowledgeCategoryItem(BaseModel):
    """Category distribution item."""

    category: str
    count: int = 0


class AdminKnowledgeOverview(BaseModel):
    """Aggregated KB stats across all customers."""

    total_knowledge_items: int = 0
    avg_success_rate: float = 0.0
    total_categories: int = 0
    customers: list[AdminKnowledgeCustomerRow] = Field(default_factory=list)
    categories: list[AdminKnowledgeCategoryItem] = Field(default_factory=list)
