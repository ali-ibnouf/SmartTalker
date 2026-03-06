"""API route definitions for SmartTalker.

Endpoints: text-to-speech, audio-chat, voice-clone,
list voices, health check, avatar management, and WebSocket chat.
"""

from __future__ import annotations

import asyncio
import re as _re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiofiles

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Query, Request, Response, UploadFile

from src.api.schemas import (
    ActiveSessionListResponse,
    ActiveSessionSummary,
    ActivityTimelineEntry,
    ActivityTimelineResponse,
    CustomerCreate,
    CustomerResponse,
    AdminSubscriptionRequest,
    AdminSubscriptionResponse,
    AddTopupRequest,
    AuditTrailResponse,
    BillingBalanceResponse,
    BillingHistoryItem,
    BillingHistoryResponse,
    BillingQuotaResponse,
    BillingUsageListResponse,
    BillingUsageResponse,
    ConsolidateResponse,
    DashboardDataResponse,
    DecisionReviewResponse,
    DriftAlertListResponse,
    DriftAlertResponse,
    ErrorResponse,
    EscalationListResponse,
    EscalationResponse,
    ExportResponse,
    HealthResponse,
    ImprovementTimelinePoint,
    ImprovementTimelineResponse,
    KBDocumentResponse,
    KBListResponse,
    KBSearchRequest,
    KBSearchResponse,
    KBUploadResponse,
    KPISnapshotResponse,
    LearnRequest,
    LearnResponse,
    OperatorMetricsListResponse,
    OperatorMetricsResponse,
    PersonaApplyRequest,
    PersonaApplyResponse,
    PersonaExtractRequest,
    PersonaListResponse,
    PersonaMatchListResponse,
    PersonaMatchRequest,
    PersonaMatchResponse,
    PersonaResponse,
    PipelineResponse,
    PolicyRequest,
    PolicyResponse,
    QualityStatsResponse,
    RenderNodeListResponse,
    RenderNodeResponse,
    ReportExportResponse,
    ReviewQueueResponse,
    ReviewSubmitRequest,
    SessionTakeoverRequest,
    SessionTakeoverResponse,
    SessionReturnResponse,
    SkillCreateRequest,
    SkillListResponse,
    SkillResponse,
    SuspendRequest,
    SuspendResponse,
    TextRequest,
    TimeseriesPoint,
    TopupPackageListResponse,
    TopupPackageResponse,
    TimeseriesResponse,
    TrainingCorrectionItem,
    TrainingSubmitRequest,
    TrainingSubmitResponse,
    TrainingStatusResponse,
    ViolationListResponse,
    ViolationResponse,
    VoiceCloneResponse,
    VoiceInfoResponse,
    VoiceListResponse,
    VRMInfoResponse,
    VRMUploadResponse,
    AvatarTypeRequest,
    WeakAreaListResponse,
    WeakAreaResponse,
)
from src.utils.exceptions import BillingError, GuardrailsError, KnowledgeBaseError, SmartTalkerError, TrainingError
from src.utils.logger import setup_logger

logger = setup_logger("api.routes")

router = APIRouter(prefix="/api/v1", tags=["SmartTalker API v1"])


# =============================================================================
# Text-to-Speech
# =============================================================================


@router.post(
    "/text-to-speech",
    response_model=PipelineResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Text to Speech",
    description="Send text, receive AI-generated audio response.",
)
async def text_to_speech(
    body: TextRequest,
    request: Request,
) -> PipelineResponse:
    """Process text input through LLM -> TTS pipeline.

    Args:
        body: TextRequest with text, avatar_id, emotion, language, voice_id.
        request: FastAPI request object (for pipeline access).

    Returns:
        PipelineResponse with audio_url, response_text, and latency.
    """
    pipeline = request.app.state.pipeline
    config = request.app.state.config
    request_id = getattr(request.state, "request_id", None)

    try:
        result = await asyncio.wait_for(
            pipeline.process_text(
                text=body.text,
                avatar_id=body.avatar_id,
                voice_id=body.voice_id,
                emotion=body.emotion,
                language=body.language,
            ),
            timeout=config.pipeline_timeout,
        )

        audio_url = _to_file_url(result.audio_path, request)

        return PipelineResponse(
            audio_url=audio_url,
            response_text=result.response_text,
            total_latency_ms=result.total_latency_ms,
            breakdown=result.breakdown,
            request_id=request_id,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Pipeline processing timed out")
    except SmartTalkerError as exc:
        raise HTTPException(status_code=500, detail=exc.to_dict()) from exc


# =============================================================================
# Audio Chat
# =============================================================================


@router.post(
    "/audio-chat",
    response_model=PipelineResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Audio Chat",
    description="Send audio, receive AI-generated audio response.",
)
async def audio_chat(
    request: Request,
    audio: UploadFile = File(..., description="Audio file (WAV, MP3, OGG, WEBM, M4A)"),
    avatar_id: str = Form(default="default"),
    emotion: str = Form(default="neutral"),
    language: str = Form(default="ar"),
    voice_id: Optional[str] = Form(default=None),
) -> PipelineResponse:
    """Process audio input through ASR -> LLM -> TTS pipeline.

    Args:
        request: FastAPI request object.
        audio: Uploaded audio file.
        avatar_id: Avatar identifier.
        emotion: Emotion label.
        language: Target response language.
        voice_id: Optional voice clone ID.

    Returns:
        PipelineResponse with audio_url, response_text, and latency.
    """
    pipeline = request.app.state.pipeline
    config = request.app.state.config
    request_id = getattr(request.state, "request_id", None)

    # Save uploaded audio to temp file
    temp_path = _save_upload(audio, request)

    try:
        result = await asyncio.wait_for(
            pipeline.process_audio(
                audio_path=str(temp_path),
                avatar_id=avatar_id,
                voice_id=voice_id,
                emotion=emotion,
                language=language,
            ),
            timeout=config.pipeline_timeout,
        )

        audio_url = _to_file_url(result.audio_path, request)

        return PipelineResponse(
            audio_url=audio_url,
            response_text=result.response_text,
            total_latency_ms=result.total_latency_ms,
            breakdown=result.breakdown,
            request_id=request_id,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Pipeline processing timed out")
    except SmartTalkerError as exc:
        raise HTTPException(status_code=500, detail=exc.to_dict()) from exc
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass


# =============================================================================
# Voice Clone
# =============================================================================


@router.post(
    "/voice-clone",
    response_model=VoiceCloneResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Clone a Voice",
    description="Upload 3-10 second reference audio to create a cloned voice.",
)
async def voice_clone(
    request: Request,
    audio: UploadFile = File(..., description="Reference audio (3-10 seconds, WAV)"),
    voice_name: str = Form(..., min_length=1, max_length=100),
) -> VoiceCloneResponse:
    """Clone a voice from reference audio.

    Args:
        request: FastAPI request object.
        audio: Uploaded reference audio file.
        voice_name: Name for the new voice.

    Returns:
        VoiceCloneResponse with the new voice_id.
    """
    pipeline = request.app.state.pipeline

    temp_path = _save_upload(audio, request)

    try:
        voice_id = pipeline.clone_voice(
            reference_audio=str(temp_path),
            voice_name=voice_name,
        )

        return VoiceCloneResponse(
            voice_id=voice_id,
            name=voice_name,
        )
    except SmartTalkerError as exc:
        raise HTTPException(status_code=400, detail=exc.to_dict()) from exc


# =============================================================================
# List Voices
# =============================================================================


@router.get(
    "/voices",
    response_model=VoiceListResponse,
    summary="List Voices",
    description="Get all available voices for TTS synthesis.",
)
async def list_voices(request: Request) -> VoiceListResponse:
    """List all registered voices.

    Args:
        request: FastAPI request object.

    Returns:
        VoiceListResponse with list of voices and count.
    """
    pipeline = request.app.state.pipeline
    voices = pipeline.list_voices()

    voice_list = [
        VoiceInfoResponse(
            voice_id=v.voice_id,
            name=v.name,
            language=v.language,
            description=v.description,
        )
        for v in voices
    ]

    return VoiceListResponse(voices=voice_list, count=len(voice_list))


# =============================================================================
# Health Check
# =============================================================================


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check system health, model availability, and uptime.",
)
async def health_check(request: Request) -> HealthResponse:
    """Return system health information.

    Args:
        request: FastAPI request object.

    Returns:
        HealthResponse with status, uptime, and loaded models.
    """
    pipeline = request.app.state.pipeline
    health = await pipeline.health_check()

    return HealthResponse(**health)


@router.get("/health/db", summary="Database Health", tags=["health"])
async def health_db(request: Request):
    """Check PostgreSQL connectivity."""
    db = getattr(request.app.state, "db", None)
    if db is None:
        return {"status": "unavailable", "detail": "No database configured"}
    try:
        from sqlalchemy import text
        async with db.session() as session:
            await session.execute(text("SELECT 1"))
        return {"status": "healthy"}
    except Exception as exc:
        return {"status": "unhealthy", "detail": str(exc)}


@router.get("/health/redis", summary="Redis Health", tags=["health"])
async def health_redis(request: Request):
    """Check Redis connectivity."""
    redis = getattr(request.app.state, "redis", None)
    if redis is None:
        return {"status": "unavailable", "detail": "No Redis configured"}
    try:
        await redis.ping()
        return {"status": "healthy"}
    except Exception as exc:
        return {"status": "unhealthy", "detail": str(exc)}


@router.get("/health/dashscope", summary="DashScope API Health", tags=["health"])
async def health_dashscope(request: Request):
    """Check DashScope API connectivity (lightweight model list call)."""
    config = getattr(request.app.state, "config", None)
    if not config or not getattr(config, "dashscope_api_key", ""):
        return {"status": "unavailable", "detail": "DashScope not configured"}
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                "https://dashscope.aliyuncs.com/compatible-mode/v1/models",
                headers={"Authorization": f"Bearer {config.dashscope_api_key}"},
            )
            return {
                "status": "healthy" if resp.status_code == 200 else "degraded",
                "status_code": resp.status_code,
            }
    except Exception as exc:
        return {"status": "unhealthy", "detail": str(exc)}


@router.get("/health/runpod", summary="RunPod Health", tags=["health"])
async def health_runpod(request: Request):
    """Check RunPod endpoint availability."""
    config = getattr(request.app.state, "config", None)
    if not config or not getattr(config, "runpod_api_key", ""):
        return {"status": "unavailable", "detail": "RunPod not configured"}
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                f"{config.runpod_render_endpoint}/health",
                headers={"Authorization": f"Bearer {config.runpod_api_key}"},
            )
            return {
                "status": "healthy" if resp.status_code in (200, 401) else "degraded",
                "status_code": resp.status_code,
            }
    except Exception as exc:
        return {"status": "unhealthy", "detail": str(exc)}


@router.get("/health/r2", summary="R2 Storage Health", tags=["health"])
async def health_r2(request: Request):
    """Check Cloudflare R2 connectivity."""
    config = getattr(request.app.state, "config", None)
    if not config or not getattr(config, "r2_access_key", ""):
        return {"status": "unavailable", "detail": "R2 not configured"}
    try:
        import boto3
        from botocore.config import Config as BotoConfig
        client = boto3.client(
            "s3",
            endpoint_url=config.r2_endpoint_url,
            aws_access_key_id=config.r2_access_key,
            aws_secret_access_key=config.r2_secret_key,
            config=BotoConfig(region_name="auto"),
        )
        client.head_bucket(Bucket=config.r2_bucket_name)
        return {"status": "healthy"}
    except Exception as exc:
        return {"status": "unhealthy", "detail": str(exc)}


# =============================================================================
# Languages (public)
# =============================================================================


@router.get(
    "/languages",
    summary="Supported Languages",
    description="Return the list of 32 supported languages with metadata.",
)
async def list_languages() -> dict:
    """Return supported languages with code, name, native name, and RTL flag."""
    from src.config import SUPPORTED_LANGUAGES

    return {"languages": SUPPORTED_LANGUAGES, "count": len(SUPPORTED_LANGUAGES)}


# =============================================================================
# Knowledge Base
# =============================================================================


@router.post(
    "/kb/upload",
    response_model=KBUploadResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Upload KB Document",
    description="Upload and ingest a document into the knowledge base.",
)
async def kb_upload(
    request: Request,
    file: UploadFile = File(..., description="Document file (PDF, DOCX, CSV, TXT, JSON)"),
) -> KBUploadResponse:
    """Upload a document to the knowledge base."""
    pipeline = request.app.state.pipeline
    kb = getattr(pipeline, "_kb", None)
    if kb is None or not kb.is_loaded:
        raise HTTPException(status_code=503, detail="Knowledge base is not available")

    # Validate file extension
    filename = file.filename or "document.txt"
    suffix = Path(filename).suffix.lower()
    allowed = {".pdf", ".docx", ".csv", ".txt", ".json"}
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(sorted(allowed))}",
        )

    # Save to temp file
    temp_path = _save_upload(file, request)
    try:
        doc_type = suffix.lstrip(".")
        result = await kb.ingest_document(str(temp_path), doc_type=doc_type)
        return KBUploadResponse(
            doc_id=result.doc_id,
            filename=filename,
            doc_type=doc_type,
            chunk_count=result.chunk_count,
        )
    except KnowledgeBaseError as exc:
        raise HTTPException(status_code=500, detail=exc.to_dict()) from exc
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass


@router.get(
    "/kb/documents",
    response_model=KBListResponse,
    summary="List KB Documents",
    description="List all documents in the knowledge base.",
)
async def kb_list_documents(request: Request) -> KBListResponse:
    """List all knowledge base documents."""
    pipeline = request.app.state.pipeline
    kb = getattr(pipeline, "_kb", None)
    if kb is None or not kb.is_loaded:
        raise HTTPException(status_code=503, detail="Knowledge base is not available")

    docs = kb.list_documents()
    doc_list = [
        KBDocumentResponse(
            doc_id=d.doc_id,
            filename=d.filename,
            doc_type=d.doc_type,
            chunk_count=d.chunk_count,
            created_at=d.created_at,
            metadata={"file_hash": d.file_hash} if d.file_hash else {},
        )
        for d in docs
    ]
    return KBListResponse(documents=doc_list, count=len(doc_list))


@router.delete(
    "/kb/documents/{doc_id}",
    summary="Delete KB Document",
    description="Delete a document from the knowledge base.",
)
async def kb_delete_document(doc_id: str, request: Request):
    """Delete a knowledge base document by ID."""
    pipeline = request.app.state.pipeline
    kb = getattr(pipeline, "_kb", None)
    if kb is None or not kb.is_loaded:
        raise HTTPException(status_code=503, detail="Knowledge base is not available")

    deleted = kb.delete_document(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
    return {"status": "ok", "doc_id": doc_id, "message": "Document deleted"}


@router.post(
    "/kb/search",
    response_model=KBSearchResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Search KB",
    description="Search the knowledge base with a text query.",
)
async def kb_search(body: KBSearchRequest, request: Request) -> KBSearchResponse:
    """Search the knowledge base."""
    pipeline = request.app.state.pipeline
    kb = getattr(pipeline, "_kb", None)
    if kb is None or not kb.is_loaded:
        raise HTTPException(status_code=503, detail="Knowledge base is not available")

    try:
        result = await kb.search(body.query, top_k=body.top_k)
        chunks = [
            {"text": c["text"], "similarity": c.get("similarity", 0.0), "metadata": c.get("metadata", {})}
            for c in result.chunks
        ]
        return KBSearchResponse(
            chunks=chunks,
            query=result.query,
            top_similarity=result.top_similarity,
            latency_ms=result.latency_ms,
        )
    except KnowledgeBaseError as exc:
        raise HTTPException(status_code=500, detail=exc.to_dict()) from exc


# =============================================================================
# Training
# =============================================================================


@router.get(
    "/training/{avatar_id}/status",
    response_model=TrainingStatusResponse,
    summary="Training Status",
    description="Get training status for an avatar.",
)
async def training_status(avatar_id: str, request: Request) -> TrainingStatusResponse:
    """Get training status for an avatar."""
    pipeline = request.app.state.pipeline
    training = getattr(pipeline, "_training", None)
    if training is None or not training.is_loaded:
        raise HTTPException(status_code=503, detail="Training engine is not available")

    try:
        status = await training.get_status(avatar_id)
        skills = [
            SkillResponse(
                skill_id=s.skill_id,
                avatar_id=s.avatar_id,
                name=s.name,
                description=s.description,
                target_threshold=s.target_threshold,
                progress=s.progress,
                qa_count=s.qa_count,
            )
            for s in status.skills
        ]
        return TrainingStatusResponse(
            avatar_id=status.avatar_id,
            skills=skills,
            overall_progress=status.overall_progress,
            is_live=status.is_live,
            total_qa_pairs=status.total_qa_pairs,
            total_escalations=status.total_escalations,
            unresolved_escalations=status.unresolved_escalations,
        )
    except TrainingError as exc:
        raise HTTPException(status_code=500, detail=exc.to_dict()) from exc


@router.post(
    "/training/skills",
    response_model=SkillResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Create Skill",
    description="Define a new training skill.",
)
async def training_create_skill(body: SkillCreateRequest, request: Request) -> SkillResponse:
    """Define a new training skill."""
    pipeline = request.app.state.pipeline
    training = getattr(pipeline, "_training", None)
    if training is None or not training.is_loaded:
        raise HTTPException(status_code=503, detail="Training engine is not available")

    try:
        skill = await training.define_skill(
            avatar_id=body.avatar_id,
            name=body.name,
            description=body.description,
            target_threshold=body.target_threshold,
        )
        return SkillResponse(
            skill_id=skill.skill_id,
            avatar_id=skill.avatar_id,
            name=skill.name,
            description=skill.description,
            target_threshold=skill.target_threshold,
            progress=skill.progress,
            qa_count=skill.qa_count,
        )
    except TrainingError as exc:
        raise HTTPException(status_code=500, detail=exc.to_dict()) from exc


@router.get(
    "/training/{avatar_id}/skills",
    response_model=SkillListResponse,
    summary="List Skills",
    description="List all skills for an avatar.",
)
async def training_list_skills(avatar_id: str, request: Request) -> SkillListResponse:
    """List training skills for an avatar."""
    pipeline = request.app.state.pipeline
    training = getattr(pipeline, "_training", None)
    if training is None or not training.is_loaded:
        raise HTTPException(status_code=503, detail="Training engine is not available")

    try:
        skills = await training.list_skills(avatar_id)
        skill_list = [
            SkillResponse(
                skill_id=s.skill_id,
                avatar_id=s.avatar_id,
                name=s.name,
                description=s.description,
                target_threshold=s.target_threshold,
                progress=s.progress,
                qa_count=s.qa_count,
            )
            for s in skills
        ]
        return SkillListResponse(skills=skill_list, count=len(skill_list))
    except TrainingError as exc:
        raise HTTPException(status_code=500, detail=exc.to_dict()) from exc


@router.delete(
    "/training/skills/{skill_id}",
    summary="Delete Skill",
    description="Delete a training skill.",
)
async def training_delete_skill(skill_id: str, request: Request):
    """Delete a training skill by ID."""
    pipeline = request.app.state.pipeline
    training = getattr(pipeline, "_training", None)
    if training is None or not training.is_loaded:
        raise HTTPException(status_code=503, detail="Training engine is not available")

    try:
        deleted = await training.delete_skill(skill_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Skill '{skill_id}' not found")
        return {"status": "ok", "skill_id": skill_id, "message": "Skill deleted"}
    except TrainingError as exc:
        raise HTTPException(status_code=500, detail=exc.to_dict()) from exc


@router.post(
    "/training/learn",
    response_model=LearnResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Learn from Human",
    description="Record a Q&A pair from a human operator to train the avatar.",
)
async def training_learn(body: LearnRequest, request: Request) -> LearnResponse:
    """Learn from a human operator response."""
    pipeline = request.app.state.pipeline
    training = getattr(pipeline, "_training", None)
    if training is None or not training.is_loaded:
        raise HTTPException(status_code=503, detail="Training engine is not available")

    try:
        qa = await training.learn_from_human(
            avatar_id=body.avatar_id,
            skill_id=body.skill_id,
            question=body.question,
            human_answer=body.human_answer,
            ai_answer=body.ai_answer,
            quality=body.quality,
        )
        # Get updated skill progress
        skills = await training.list_skills(body.avatar_id)
        progress = 0.0
        for s in skills:
            if s.skill_id == body.skill_id:
                progress = s.progress
                break

        return LearnResponse(
            qa_id=qa.qa_id,
            skill_id=qa.skill_id,
            updated_progress=progress,
        )
    except TrainingError as exc:
        raise HTTPException(status_code=500, detail=exc.to_dict()) from exc


@router.get(
    "/training/{avatar_id}/escalations",
    response_model=EscalationListResponse,
    summary="List Escalations",
    description="List escalation events for an avatar.",
)
async def training_list_escalations(
    avatar_id: str,
    request: Request,
    unresolved_only: bool = Query(default=False, description="Only show unresolved"),
) -> EscalationListResponse:
    """List escalation events."""
    pipeline = request.app.state.pipeline
    training = getattr(pipeline, "_training", None)
    if training is None or not training.is_loaded:
        raise HTTPException(status_code=503, detail="Training engine is not available")

    try:
        escalations = await training.list_escalations(avatar_id, unresolved_only=unresolved_only)
        esc_list = [
            EscalationResponse(
                event_id=e.event_id,
                session_id=e.session_id,
                avatar_id=e.avatar_id,
                skill_id=e.skill_id,
                question=e.question,
                confidence=e.confidence,
                resolved=e.resolved,
                resolution=e.resolution,
                created_at=e.created_at,
            )
            for e in escalations
        ]
        return EscalationListResponse(escalations=esc_list, count=len(esc_list))
    except TrainingError as exc:
        raise HTTPException(status_code=500, detail=exc.to_dict()) from exc


@router.post(
    "/training/escalations/{event_id}/resolve",
    response_model=EscalationResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Resolve Escalation",
    description="Resolve an escalation event with an operator's answer.",
)
async def training_resolve_escalation(
    event_id: str,
    request: Request,
    resolution: str = Form(..., min_length=1, max_length=5000, description="Resolution text"),
) -> EscalationResponse:
    """Resolve an escalation event."""
    pipeline = request.app.state.pipeline
    training = getattr(pipeline, "_training", None)
    if training is None or not training.is_loaded:
        raise HTTPException(status_code=503, detail="Training engine is not available")

    try:
        esc = await training.resolve_escalation(event_id, resolution)
        if esc is None:
            raise HTTPException(status_code=404, detail=f"Escalation '{event_id}' not found")
        return EscalationResponse(
            event_id=esc.event_id,
            session_id=esc.session_id,
            avatar_id=esc.avatar_id,
            skill_id=esc.skill_id,
            question=esc.question,
            confidence=esc.confidence,
            resolved=esc.resolved,
            resolution=esc.resolution,
            created_at=esc.created_at,
        )
    except TrainingError as exc:
        raise HTTPException(status_code=500, detail=exc.to_dict()) from exc


# =============================================================================
# Billing
# =============================================================================


@router.get(
    "/billing/{customer_id}/usage",
    response_model=BillingUsageListResponse,
    summary="Billing Usage",
    description="Get usage records for a customer.",
)
async def billing_usage(
    customer_id: str,
    request: Request,
    days: int = Query(default=30, ge=1, le=365, description="Number of days"),
) -> BillingUsageListResponse:
    """Get billing usage records for a customer."""
    billing = getattr(request.app.state, "billing", None)
    if billing is None:
        raise HTTPException(status_code=503, detail="Billing engine is not available")

    try:
        records = await billing.get_usage(customer_id, days=days)
        return BillingUsageListResponse(
            records=[BillingUsageResponse(**r) for r in records],
            count=len(records),
        )
    except BillingError as exc:
        raise HTTPException(status_code=500, detail=exc.to_dict()) from exc


@router.get(
    "/billing/{customer_id}/quota",
    response_model=BillingQuotaResponse,
    summary="Billing Quota",
    description="Check remaining seconds for a customer.",
)
async def billing_quota(customer_id: str, request: Request) -> BillingQuotaResponse:
    """Check remaining billing quota."""
    billing = getattr(request.app.state, "billing", None)
    if billing is None:
        raise HTTPException(status_code=503, detail="Billing engine is not available")

    remaining = await billing.check_quota(customer_id)
    return BillingQuotaResponse(
        customer_id=customer_id,
        remaining_seconds=remaining,
    )


@router.get(
    "/billing/{customer_id}/balance",
    response_model=BillingBalanceResponse,
    summary="Billing Balance",
    description="Get dual balance (plan + extra seconds) for a customer.",
)
async def billing_balance(customer_id: str, request: Request) -> BillingBalanceResponse:
    """Get dual balance breakdown."""
    billing = getattr(request.app.state, "billing", None)
    if billing is None:
        raise HTTPException(status_code=503, detail="Billing engine is not available")

    balance = await billing.get_balance(customer_id)
    return BillingBalanceResponse(**balance)


@router.get(
    "/billing/topup-packages",
    response_model=TopupPackageListResponse,
    summary="Top-up Packages",
    description="List available top-up packages.",
)
async def topup_packages() -> TopupPackageListResponse:
    """List available top-up packages."""
    from src.config import TOPUP_PACKAGES
    packages = [
        TopupPackageResponse(id=pkg_id, seconds=pkg["seconds"], price=pkg["price"])
        for pkg_id, pkg in TOPUP_PACKAGES.items()
    ]
    return TopupPackageListResponse(packages=packages)


@router.post(
    "/billing/{customer_id}/add-topup",
    summary="Add Top-up (Admin)",
    description="Add extra seconds from a completed top-up purchase.",
)
async def add_topup(customer_id: str, body: AddTopupRequest, request: Request) -> dict:
    """Admin endpoint called by webhook when a top-up payment completes."""
    from src.config import TOPUP_PACKAGES

    billing = getattr(request.app.state, "billing", None)
    if billing is None:
        raise HTTPException(status_code=503, detail="Billing engine is not available")

    package = TOPUP_PACKAGES.get(body.package_id)
    if package is None:
        raise HTTPException(status_code=400, detail=f"Unknown package: {body.package_id}")

    new_total = await billing.add_topup(customer_id, package["seconds"])
    return {
        "customer_id": customer_id,
        "package_id": body.package_id,
        "seconds_added": package["seconds"],
        "extra_seconds_remaining": new_total,
    }


@router.get(
    "/billing/{customer_id}/history",
    response_model=BillingHistoryResponse,
    summary="Billing History",
    description="Get billing history (subscriptions + top-ups).",
)
async def billing_history(
    customer_id: str,
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
) -> BillingHistoryResponse:
    """Get billing transaction history for a customer."""
    billing = getattr(request.app.state, "billing", None)
    if billing is None:
        raise HTTPException(status_code=503, detail="Billing engine is not available")

    records = await billing.get_usage(customer_id, days=365)
    items = [
        BillingHistoryItem(
            date=r.get("started_at", ""),
            type="subscription",
            amount=r.get("cost", 0),
            seconds=round(r.get("duration_s", 0)),
        )
        for r in records[:limit]
    ]
    return BillingHistoryResponse(items=items, count=len(items))


# =============================================================================
# Personas
# =============================================================================


@router.get(
    "/personas",
    response_model=PersonaListResponse,
    summary="List Personas",
    description="List available job personas.",
)
async def list_personas(
    request: Request,
    industry: Optional[str] = Query(default=None, description="Filter by industry"),
) -> PersonaListResponse:
    """List job personas."""
    persona_engine = getattr(request.app.state, "persona_engine", None)
    if persona_engine is None:
        raise HTTPException(status_code=503, detail="Persona engine is not available")

    personas = await persona_engine.list_personas(industry=industry)
    return PersonaListResponse(
        personas=[
            PersonaResponse(
                persona_id=p.persona_id,
                name=p.name,
                industry=p.industry,
                description=p.description,
                skill_count=p.skill_count,
                is_public=p.is_public,
                source_avatar_id=p.source_avatar_id,
            )
            for p in personas
        ],
        count=len(personas),
    )


@router.post(
    "/personas/extract",
    response_model=PersonaResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Extract Persona",
    description="Extract a persona from a trained avatar's skills.",
)
async def extract_persona(body: PersonaExtractRequest, request: Request) -> PersonaResponse:
    """Extract a persona from a trained avatar."""
    persona_engine = getattr(request.app.state, "persona_engine", None)
    if persona_engine is None:
        raise HTTPException(status_code=503, detail="Persona engine is not available")

    try:
        persona = await persona_engine.extract_persona(
            avatar_id=body.avatar_id,
            persona_name=body.name,
            industry=body.industry,
        )
        return PersonaResponse(
            persona_id=persona.persona_id,
            name=persona.name,
            industry=persona.industry,
            description=persona.description,
            skill_count=persona.skill_count,
            is_public=persona.is_public,
            source_avatar_id=persona.source_avatar_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post(
    "/personas/match",
    response_model=PersonaMatchListResponse,
    summary="Match Personas",
    description="Find matching personas for an industry and skill set.",
)
async def match_personas(body: PersonaMatchRequest, request: Request) -> PersonaMatchListResponse:
    """Find matching personas."""
    persona_engine = getattr(request.app.state, "persona_engine", None)
    if persona_engine is None:
        raise HTTPException(status_code=503, detail="Persona engine is not available")

    matches = await persona_engine.match_persona(
        industry=body.industry,
        skills_needed=body.skills_needed,
    )
    return PersonaMatchListResponse(
        matches=[
            PersonaMatchResponse(
                persona_id=m.persona_id,
                name=m.name,
                industry=m.industry,
                match_score=m.match_score,
                pre_populated_skills=m.pre_populated_skills,
            )
            for m in matches
        ],
        count=len(matches),
    )


@router.post(
    "/personas/apply",
    response_model=PersonaApplyResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Apply Persona",
    description="Apply a persona's skills to an avatar at 70% progress.",
)
async def apply_persona(body: PersonaApplyRequest, request: Request) -> PersonaApplyResponse:
    """Apply a persona to an avatar."""
    persona_engine = getattr(request.app.state, "persona_engine", None)
    if persona_engine is None:
        raise HTTPException(status_code=503, detail="Persona engine is not available")

    try:
        count = await persona_engine.apply_persona(
            avatar_id=body.avatar_id,
            persona_id=body.persona_id,
        )
        return PersonaApplyResponse(
            avatar_id=body.avatar_id,
            persona_id=body.persona_id,
            skills_applied=count,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# =============================================================================
# Render Nodes
# =============================================================================


@router.get(
    "/nodes",
    response_model=RenderNodeListResponse,
    summary="List GPU Nodes",
    description="List GPU rendering status. GPU rendering now uses RunPod Serverless.",
)
async def list_nodes(request: Request) -> RenderNodeListResponse:
    """List GPU nodes — returns empty (RunPod Serverless has replaced persistent nodes)."""
    return RenderNodeListResponse(nodes=[], count=0)


@router.get(
    "/nodes/{node_id}",
    response_model=RenderNodeResponse,
    summary="Get GPU Node",
    description="Get a specific GPU node — deprecated, rendering is now via RunPod Serverless.",
)
async def get_node(node_id: str, request: Request) -> RenderNodeResponse:
    """Get a specific GPU node — deprecated."""
    raise HTTPException(
        status_code=410,
        detail="GPU render nodes replaced by RunPod Serverless. Use /api/v1/health for status.",
    )


# =============================================================================
# Learning Analytics
# =============================================================================


@router.get(
    "/training/{avatar_id}/quality-stats",
    response_model=QualityStatsResponse,
    summary="Skill Quality Stats",
    description="Get quality statistics for an avatar's skill.",
)
async def training_quality_stats(
    avatar_id: str,
    request: Request,
    skill_id: str = Query(..., description="Skill ID"),
) -> QualityStatsResponse:
    """Get quality stats for a specific skill."""
    analytics = getattr(request.app.state, "learning_analytics", None)
    if analytics is None or not analytics.is_loaded:
        raise HTTPException(status_code=503, detail="Learning analytics is not available")

    try:
        stats = await analytics.get_skill_quality_stats(avatar_id, skill_id)
        return QualityStatsResponse(
            skill_id=skill_id,
            total=stats.total_qa,
            good=stats.good_count,
            bad=stats.bad_count,
            none_count=stats.none_count,
            correction_count=stats.correction_count,
            bad_ratio=stats.bad_ratio,
            improvement_rate=stats.improvement_rate,
            effective_threshold=stats.effective_threshold,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get(
    "/training/{avatar_id}/improvement-timeline",
    response_model=ImprovementTimelineResponse,
    summary="Improvement Timeline",
    description="Get learning improvement timeline for an avatar.",
)
async def training_improvement_timeline(
    avatar_id: str,
    request: Request,
    skill_id: Optional[str] = Query(default=None, description="Optional skill filter"),
    days: int = Query(default=30, ge=1, le=365, description="Number of days"),
) -> ImprovementTimelineResponse:
    """Get improvement timeline."""
    analytics = getattr(request.app.state, "learning_analytics", None)
    if analytics is None or not analytics.is_loaded:
        raise HTTPException(status_code=503, detail="Learning analytics is not available")

    try:
        points = await analytics.get_improvement_timeline(avatar_id, skill_id=skill_id, days=days)
        return ImprovementTimelineResponse(
            avatar_id=avatar_id,
            skill_id=skill_id,
            days=days,
            timeline=[
                ImprovementTimelinePoint(
                    date=p.date,
                    qa_added=p.qa_added,
                    good_count=p.good_count,
                    bad_count=p.bad_count,
                    avg_confidence=p.avg_confidence,
                )
                for p in points
            ],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post(
    "/training/{avatar_id}/consolidate",
    response_model=ConsolidateResponse,
    summary="Consolidate Daily",
    description="Run daily consolidation of learning metrics.",
)
async def training_consolidate(avatar_id: str, request: Request) -> ConsolidateResponse:
    """Run daily consolidation."""
    analytics = getattr(request.app.state, "learning_analytics", None)
    if analytics is None or not analytics.is_loaded:
        raise HTTPException(status_code=503, detail="Learning analytics is not available")

    try:
        result = await analytics.consolidate_daily(avatar_id)
        return ConsolidateResponse(
            avatar_id=avatar_id,
            date=result.date,
            skills_consolidated=result.skills_updated,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get(
    "/training/{avatar_id}/export",
    response_model=ExportResponse,
    summary="Export Q&A Pairs",
    description="Export Q&A pairs in JSONL format for fine-tuning.",
)
async def training_export(
    avatar_id: str,
    request: Request,
    skill_id: Optional[str] = Query(default=None, description="Optional skill filter"),
    format: str = Query(default="jsonl", description="Export format"),
) -> ExportResponse:
    """Export Q&A pairs."""
    analytics = getattr(request.app.state, "learning_analytics", None)
    if analytics is None or not analytics.is_loaded:
        raise HTTPException(status_code=503, detail="Learning analytics is not available")

    try:
        content = await analytics.export_qa_pairs(avatar_id, skill_id=skill_id, fmt=format)
        record_count = content.count("\n") if content else 0
        return ExportResponse(
            avatar_id=avatar_id,
            skill_id=skill_id,
            format=format,
            content=content,
            record_count=record_count,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get(
    "/training/{avatar_id}/skills/{skill_id}/weak-areas",
    response_model=WeakAreaListResponse,
    summary="Weak Areas",
    description="Get weak areas for a specific skill.",
)
async def training_weak_areas(
    avatar_id: str,
    skill_id: str,
    request: Request,
) -> WeakAreaListResponse:
    """Get weak areas."""
    analytics = getattr(request.app.state, "learning_analytics", None)
    if analytics is None or not analytics.is_loaded:
        raise HTTPException(status_code=503, detail="Learning analytics is not available")

    try:
        areas = await analytics.get_weak_areas(avatar_id)
        # Filter to the specified skill if found
        filtered = [a for a in areas if a.get("skill_id") == skill_id] if skill_id != "all" else areas
        return WeakAreaListResponse(
            avatar_id=avatar_id,
            weak_areas=[
                WeakAreaResponse(
                    skill_id=a["skill_id"],
                    skill_name=a.get("skill_name", ""),
                    bad_ratio=a.get("bad_ratio", 0.0),
                    correction_count=a.get("correction_count", 0),
                    effective_threshold=a.get("effective_threshold", 0.7),
                )
                for a in filtered
            ],
            count=len(filtered),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# =============================================================================
# Guardrails
# =============================================================================


@router.get(
    "/guardrails/{avatar_id}/policy",
    response_model=PolicyResponse,
    summary="Get Guardrail Policy",
    description="Get the guardrail policy for an avatar.",
)
async def guardrails_get_policy(avatar_id: str, request: Request) -> PolicyResponse:
    """Get guardrail policy."""
    guardrails = getattr(request.app.state, "guardrails", None)
    if guardrails is None or not guardrails.is_loaded:
        raise HTTPException(status_code=503, detail="Guardrails engine is not available")

    try:
        policy = await guardrails.get_policy(avatar_id)
        return PolicyResponse(
            avatar_id=avatar_id,
            blocked_topics=policy.blocked_topics,
            required_disclaimers=policy.required_disclaimers,
            max_response_length=policy.max_response_length,
            escalation_keywords=policy.escalation_keywords,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.put(
    "/guardrails/{avatar_id}/policy",
    response_model=PolicyResponse,
    summary="Set Guardrail Policy",
    description="Set or update the guardrail policy for an avatar.",
)
async def guardrails_set_policy(
    avatar_id: str,
    body: PolicyRequest,
    request: Request,
) -> PolicyResponse:
    """Set guardrail policy."""
    guardrails = getattr(request.app.state, "guardrails", None)
    if guardrails is None or not guardrails.is_loaded:
        raise HTTPException(status_code=503, detail="Guardrails engine is not available")

    try:
        from src.pipeline.guardrails import PolicyConfig
        policy = PolicyConfig(
            blocked_topics=body.blocked_topics,
            required_disclaimers=body.required_disclaimers,
            max_response_length=body.max_response_length,
            escalation_keywords=body.escalation_keywords,
        )
        await guardrails.set_policy(avatar_id, policy)
        return PolicyResponse(
            avatar_id=avatar_id,
            blocked_topics=policy.blocked_topics,
            required_disclaimers=policy.required_disclaimers,
            max_response_length=policy.max_response_length,
            escalation_keywords=policy.escalation_keywords,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.delete(
    "/guardrails/{avatar_id}/policy",
    summary="Delete Guardrail Policy",
    description="Delete the guardrail policy for an avatar (resets to defaults).",
)
async def guardrails_delete_policy(avatar_id: str, request: Request):
    """Delete guardrail policy."""
    guardrails = getattr(request.app.state, "guardrails", None)
    if guardrails is None or not guardrails.is_loaded:
        raise HTTPException(status_code=503, detail="Guardrails engine is not available")

    try:
        deleted = await guardrails.delete_policy(avatar_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"No policy found for avatar '{avatar_id}'")
        return {"status": "ok", "avatar_id": avatar_id, "message": "Policy deleted"}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get(
    "/guardrails/{avatar_id}/violations",
    response_model=ViolationListResponse,
    summary="List Violations",
    description="List guardrail violations for an avatar.",
)
async def guardrails_list_violations(
    avatar_id: str,
    request: Request,
    violation_type: Optional[str] = Query(default=None, description="Filter by type"),
    limit: int = Query(default=100, ge=1, le=500, description="Max results"),
) -> ViolationListResponse:
    """List guardrail violations."""
    guardrails = getattr(request.app.state, "guardrails", None)
    if guardrails is None or not guardrails.is_loaded:
        raise HTTPException(status_code=503, detail="Guardrails engine is not available")

    try:
        violations = await guardrails.list_violations(
            avatar_id, violation_type=violation_type, limit=limit,
        )
        return ViolationListResponse(
            violations=[
                ViolationResponse(
                    id=v.id,
                    avatar_id=v.avatar_id,
                    session_id=v.session_id,
                    violation_type=v.violation_type,
                    original_text=v.original_response,
                    sanitized_text=v.sanitized_response,
                    severity=v.severity,
                    created_at=v.created_at,
                )
                for v in violations
            ],
            count=len(violations),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get(
    "/guardrails/{avatar_id}/audit",
    response_model=AuditTrailResponse,
    summary="Guardrails Audit Trail",
    description="Get guardrails audit trail summary for an avatar.",
)
async def guardrails_audit(avatar_id: str, request: Request) -> AuditTrailResponse:
    """Get guardrails audit trail."""
    guardrails = getattr(request.app.state, "guardrails", None)
    if guardrails is None or not guardrails.is_loaded:
        raise HTTPException(status_code=503, detail="Guardrails engine is not available")

    try:
        violations = await guardrails.list_violations(avatar_id, limit=20)
        type_counts: dict[str, int] = {}
        for v in violations:
            type_counts[v.violation_type] = type_counts.get(v.violation_type, 0) + 1

        all_violations = await guardrails.list_violations(avatar_id, limit=500)
        return AuditTrailResponse(
            avatar_id=avatar_id,
            total_checks=0,  # Would need separate counter
            total_violations=len(all_violations),
            violation_types=type_counts,
            recent_violations=[
                ViolationResponse(
                    id=v.id,
                    avatar_id=v.avatar_id,
                    session_id=v.session_id,
                    violation_type=v.violation_type,
                    original_text=v.original_response,
                    sanitized_text=v.sanitized_response,
                    severity=v.severity,
                    created_at=v.created_at,
                )
                for v in violations[:10]
            ],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# =============================================================================
# Supervisor
# =============================================================================


@router.get(
    "/supervisor/operators",
    response_model=OperatorMetricsListResponse,
    summary="List Operator Metrics",
    description="List performance metrics for all operators.",
)
async def supervisor_list_operators(
    request: Request,
    days: int = Query(default=30, ge=1, le=365, description="Days to aggregate"),
) -> OperatorMetricsListResponse:
    """List all operator metrics."""
    supervisor = getattr(request.app.state, "supervisor", None)
    if supervisor is None or not supervisor.is_loaded:
        raise HTTPException(status_code=503, detail="Supervisor engine is not available")

    try:
        metrics = await supervisor.list_operator_metrics(days=days)
        return OperatorMetricsListResponse(
            operators=[
                OperatorMetricsResponse(
                    operator_id=m.operator_id,
                    total_responses=m.total_responses,
                    avg_response_time_ms=m.avg_response_time_ms,
                    escalations_resolved=m.escalations_resolved,
                    corrections_made=m.corrections_made,
                    sessions_handled=m.sessions_handled,
                    quality_score=m.quality_score,
                )
                for m in metrics
            ],
            count=len(metrics),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get(
    "/supervisor/operators/{operator_id}",
    response_model=OperatorMetricsResponse,
    summary="Get Operator Metrics",
    description="Get performance metrics for a specific operator.",
)
async def supervisor_get_operator(
    operator_id: str,
    request: Request,
    days: int = Query(default=30, ge=1, le=365, description="Days to aggregate"),
) -> OperatorMetricsResponse:
    """Get single operator metrics."""
    supervisor = getattr(request.app.state, "supervisor", None)
    if supervisor is None or not supervisor.is_loaded:
        raise HTTPException(status_code=503, detail="Supervisor engine is not available")

    try:
        m = await supervisor.get_operator_metrics(operator_id, days=days)
        return OperatorMetricsResponse(
            operator_id=m.operator_id,
            total_responses=m.total_responses,
            avg_response_time_ms=m.avg_response_time_ms,
            escalations_resolved=m.escalations_resolved,
            corrections_made=m.corrections_made,
            sessions_handled=m.sessions_handled,
            quality_score=m.quality_score,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get(
    "/supervisor/sessions/active",
    response_model=ActiveSessionListResponse,
    summary="Active Sessions",
    description="Get summary of active customer sessions.",
)
async def supervisor_active_sessions(request: Request) -> ActiveSessionListResponse:
    """Get active sessions summary."""
    supervisor = getattr(request.app.state, "supervisor", None)
    if supervisor is None or not supervisor.is_loaded:
        raise HTTPException(status_code=503, detail="Supervisor engine is not available")

    try:
        sessions = await supervisor.get_active_sessions_summary()
        return ActiveSessionListResponse(
            sessions=[
                ActiveSessionSummary(
                    session_id=s.get("session_id", ""),
                    avatar_id=s.get("avatar_id", ""),
                    operator_id=s.get("operator_id"),
                    started_at=s.get("started_at", 0.0),
                    message_count=s.get("message_count", 0),
                )
                for s in sessions
            ],
            count=len(sessions),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get(
    "/supervisor/review-queue",
    response_model=ReviewQueueResponse,
    summary="Review Queue",
    description="Get the decision review queue.",
)
async def supervisor_review_queue(
    request: Request,
    reviewed: Optional[bool] = Query(default=None, description="Filter by reviewed status"),
    limit: int = Query(default=50, ge=1, le=200, description="Max results"),
) -> ReviewQueueResponse:
    """Get review queue."""
    supervisor = getattr(request.app.state, "supervisor", None)
    if supervisor is None or not supervisor.is_loaded:
        raise HTTPException(status_code=503, detail="Supervisor engine is not available")

    try:
        reviews = await supervisor.list_review_queue(reviewed=reviewed, limit=limit)
        return ReviewQueueResponse(
            reviews=[
                DecisionReviewResponse(
                    review_id=r.id,
                    session_id=r.session_id,
                    avatar_id=r.avatar_id,
                    question=r.question,
                    ai_response=r.ai_response,
                    confidence=r.confidence,
                    flagged_reason=r.flagged_reason,
                    reviewed=r.reviewed,
                    reviewer_id=r.reviewer_id,
                    verdict=r.review_verdict,
                    corrected_response=r.corrected_response,
                    created_at=r.created_at,
                )
                for r in reviews
            ],
            count=len(reviews),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post(
    "/supervisor/review-queue/{review_id}",
    response_model=DecisionReviewResponse,
    summary="Submit Review",
    description="Submit a verdict for a flagged decision.",
)
async def supervisor_submit_review(
    review_id: str,
    body: ReviewSubmitRequest,
    request: Request,
) -> DecisionReviewResponse:
    """Submit a review."""
    supervisor = getattr(request.app.state, "supervisor", None)
    if supervisor is None or not supervisor.is_loaded:
        raise HTTPException(status_code=503, detail="Supervisor engine is not available")

    try:
        r = await supervisor.submit_review(
            review_id=review_id,
            reviewer_id=body.reviewer_id,
            verdict=body.verdict,
            corrected_response=body.corrected_response,
        )
        return DecisionReviewResponse(
            review_id=r.id,
            session_id=r.session_id,
            avatar_id=r.avatar_id,
            question=r.question,
            ai_response=r.ai_response,
            confidence=r.confidence,
            flagged_reason=r.flagged_reason,
            reviewed=r.reviewed,
            reviewer_id=r.reviewer_id,
            verdict=r.review_verdict,
            corrected_response=r.corrected_response,
            created_at=r.created_at,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get(
    "/supervisor/activity-timeline",
    response_model=ActivityTimelineResponse,
    summary="Activity Timeline",
    description="Get recent operator activity timeline.",
)
async def supervisor_activity_timeline(
    request: Request,
    days: int = Query(default=7, ge=1, le=90, description="Days of history"),
    limit: int = Query(default=100, ge=1, le=500, description="Max entries"),
) -> ActivityTimelineResponse:
    """Get activity timeline."""
    supervisor = getattr(request.app.state, "supervisor", None)
    if supervisor is None or not supervisor.is_loaded:
        raise HTTPException(status_code=503, detail="Supervisor engine is not available")

    try:
        entries = await supervisor.get_activity_timeline(days=days, limit=limit)
        return ActivityTimelineResponse(
            entries=[
                ActivityTimelineEntry(
                    action_id=f"act_{i}",
                    operator_id=e.operator_id,
                    action_type=e.action_type,
                    session_id=e.session_id,
                    details=e.details if isinstance(e.details, dict) else {},
                    created_at=e.timestamp,
                )
                for i, e in enumerate(entries)
            ],
            count=len(entries),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post(
    "/supervisor/sessions/{session_id}/takeover",
    response_model=SessionTakeoverResponse,
    summary="Takeover Session",
    description="Operator takes over a session — pauses AI responses.",
)
async def supervisor_takeover_session(
    session_id: str,
    body: SessionTakeoverRequest,
    request: Request,
) -> SessionTakeoverResponse:
    """Operator takes over a session, pausing AI responses."""
    ws_mgr = getattr(request.app.state, "ws_manager", None)
    if ws_mgr is None:
        raise HTTPException(status_code=503, detail="WebSocket manager not available")

    session_map: dict = getattr(ws_mgr, '_sessions', {})
    session = session_map.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Pause AI and assign operator
    session.ai_paused = True
    session.operator_id = body.operator_id
    session.takeover_at = datetime.now(timezone.utc).isoformat()

    # Notify customer widget
    try:
        from starlette.websockets import WebSocketState
        ws = getattr(session, 'websocket', None)
        if ws and ws.client_state == WebSocketState.CONNECTED:
            await ws.send_json({
                "type": "operator_joined",
                "message": "You're now connected with a live agent",
            })
    except Exception:
        pass

    # Record action in supervisor
    supervisor = getattr(request.app.state, "supervisor", None)
    if supervisor and supervisor.is_loaded:
        try:
            await supervisor.record_operator_action(
                operator_id=body.operator_id,
                action_type="takeover",
                session_id=session_id,
                avatar_id=getattr(session.config, 'avatar_id', ''),
            )
        except Exception:
            pass

    logger.info(
        "Session taken over",
        extra={"session_id": session_id, "operator_id": body.operator_id},
    )

    return SessionTakeoverResponse(
        status="taken_over",
        session_id=session_id,
        operator_id=body.operator_id,
    )


@router.post(
    "/supervisor/sessions/{session_id}/return",
    response_model=SessionReturnResponse,
    summary="Return Session to AI",
    description="Return session to AI — resumes AI responses.",
)
async def supervisor_return_session(
    session_id: str,
    request: Request,
) -> SessionReturnResponse:
    """Return a session to AI, resuming automated responses."""
    ws_mgr = getattr(request.app.state, "ws_manager", None)
    if ws_mgr is None:
        raise HTTPException(status_code=503, detail="WebSocket manager not available")

    session_map: dict = getattr(ws_mgr, '_sessions', {})
    session = session_map.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    prev_operator = session.operator_id
    session.ai_paused = False
    session.operator_id = None
    session.takeover_at = None

    # Notify customer widget
    try:
        from starlette.websockets import WebSocketState
        ws = getattr(session, 'websocket', None)
        if ws and ws.client_state == WebSocketState.CONNECTED:
            await ws.send_json({
                "type": "operator_left",
                "message": "You're now back with your AI assistant",
            })
    except Exception:
        pass

    # Record action in supervisor
    supervisor = getattr(request.app.state, "supervisor", None)
    if supervisor and supervisor.is_loaded:
        try:
            await supervisor.record_operator_action(
                operator_id=prev_operator or "unknown",
                action_type="return_to_ai",
                session_id=session_id,
                avatar_id=getattr(session.config, 'avatar_id', ''),
            )
        except Exception:
            pass

    logger.info(
        "Session returned to AI",
        extra={"session_id": session_id, "prev_operator": prev_operator},
    )

    return SessionReturnResponse(
        status="returned_to_ai",
        session_id=session_id,
    )


@router.post(
    "/supervisor/sessions/{session_id}/train",
    response_model=TrainingSubmitResponse,
    summary="Submit Training Data",
    description="Submit training data from operator review of a session.",
)
async def supervisor_submit_training(
    session_id: str,
    body: TrainingSubmitRequest,
    request: Request,
) -> TrainingSubmitResponse:
    """Submit training data from operator review."""
    # Store training submission via training engine if available
    training = getattr(request.app.state, 'pipeline', None)
    training_engine = getattr(training, '_training', None) if training else None

    if training_engine and hasattr(training_engine, 'learn_from_human'):
        for correction in body.corrections:
            if correction.corrected_response:
                try:
                    await training_engine.learn_from_human(
                        avatar_id="",  # Resolved from session context if needed
                        skill_id="operator_correction",
                        question=correction.message_id,
                        answer=correction.corrected_response,
                        operator_id=body.operator_id,
                    )
                except Exception:
                    pass

    # Record action in supervisor
    supervisor = getattr(request.app.state, "supervisor", None)
    if supervisor and supervisor.is_loaded:
        try:
            await supervisor.record_operator_action(
                operator_id=body.operator_id,
                action_type="training_submit",
                session_id=session_id,
                details={
                    "approved_count": len(body.approved_messages),
                    "correction_count": len(body.corrections),
                    "notes": body.notes,
                },
            )
        except Exception:
            pass

    logger.info(
        "Training data submitted",
        extra={
            "session_id": session_id,
            "operator_id": body.operator_id,
            "approved": len(body.approved_messages),
            "corrections": len(body.corrections),
        },
    )

    return TrainingSubmitResponse(
        status="submitted",
        session_id=session_id,
        approved_count=len(body.approved_messages),
        correction_count=len(body.corrections),
    )


# =============================================================================
# Analytics
# =============================================================================


@router.get(
    "/analytics/{avatar_id}/kpis",
    response_model=KPISnapshotResponse,
    summary="KPI Snapshot",
    description="Get KPI snapshot for an avatar.",
)
async def analytics_kpis(
    avatar_id: str,
    request: Request,
    period: str = Query(default="daily", description="Period: daily, weekly, monthly"),
) -> KPISnapshotResponse:
    """Get KPI snapshot."""
    analytics_engine = getattr(request.app.state, "analytics", None)
    if analytics_engine is None or not analytics_engine.is_loaded:
        raise HTTPException(status_code=503, detail="Analytics engine is not available")

    try:
        kpis = await analytics_engine.compute_kpis(avatar_id, period=period)
        return KPISnapshotResponse(
            avatar_id=avatar_id,
            period=period,
            total_conversations=kpis.total_conversations,
            total_messages=kpis.total_messages,
            avg_response_time_ms=kpis.avg_response_time_ms,
            avg_kb_confidence=kpis.avg_kb_confidence,
            escalation_rate=kpis.escalation_rate,
            autonomy_percent=kpis.autonomy_percent,
            resolution_time_avg_s=kpis.resolution_time_avg_s,
            accuracy_score=kpis.accuracy_score,
            total_cost=kpis.total_cost,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get(
    "/analytics/{avatar_id}/timeseries",
    response_model=TimeseriesResponse,
    summary="Timeseries Data",
    description="Get timeseries data for a metric.",
)
async def analytics_timeseries(
    avatar_id: str,
    request: Request,
    metric: str = Query(..., description="Metric: conversations, autonomy, escalation_rate, confidence"),
    period: str = Query(default="daily", description="Period: daily, weekly"),
    days: int = Query(default=30, ge=1, le=365, description="Number of days"),
) -> TimeseriesResponse:
    """Get timeseries data."""
    analytics_engine = getattr(request.app.state, "analytics", None)
    if analytics_engine is None or not analytics_engine.is_loaded:
        raise HTTPException(status_code=503, detail="Analytics engine is not available")

    try:
        points = await analytics_engine.get_timeseries(avatar_id, metric=metric, period=period, days=days)
        return TimeseriesResponse(
            avatar_id=avatar_id,
            metric=metric,
            period=period,
            points=[
                TimeseriesPoint(date=p.date, value=p.value)
                for p in points
            ],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get(
    "/analytics/{avatar_id}/dashboard",
    response_model=DashboardDataResponse,
    summary="Dashboard Data",
    description="Get complete analytics dashboard data.",
)
async def analytics_dashboard(
    avatar_id: str,
    request: Request,
    days: int = Query(default=30, ge=1, le=365, description="Number of days"),
) -> DashboardDataResponse:
    """Get dashboard data."""
    analytics_engine = getattr(request.app.state, "analytics", None)
    if analytics_engine is None or not analytics_engine.is_loaded:
        raise HTTPException(status_code=503, detail="Analytics engine is not available")

    try:
        data = await analytics_engine.get_dashboard_data(avatar_id, days=days)
        kpis = data["kpis"]
        return DashboardDataResponse(
            avatar_id=avatar_id,
            kpis=KPISnapshotResponse(
                avatar_id=avatar_id,
                total_conversations=kpis.total_conversations,
                total_messages=kpis.total_messages,
                avg_response_time_ms=kpis.avg_response_time_ms,
                avg_kb_confidence=kpis.avg_kb_confidence,
                escalation_rate=kpis.escalation_rate,
                autonomy_percent=kpis.autonomy_percent,
                resolution_time_avg_s=kpis.resolution_time_avg_s,
                accuracy_score=kpis.accuracy_score,
                total_cost=kpis.total_cost,
            ),
            trends={
                k: [TimeseriesPoint(date=p.date, value=p.value) for p in v]
                for k, v in data.get("trends", {}).items()
            },
            top_skills=data.get("top_skills", []),
            bottom_skills=data.get("bottom_skills", []),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get(
    "/analytics/{avatar_id}/drift",
    response_model=DriftAlertListResponse,
    summary="Drift Detection",
    description="Check for performance drift.",
)
async def analytics_drift(
    avatar_id: str,
    request: Request,
    baseline_days: int = Query(default=30, ge=7, le=365, description="Baseline period"),
    recent_days: int = Query(default=7, ge=1, le=90, description="Recent period"),
) -> DriftAlertListResponse:
    """Check for drift."""
    analytics_engine = getattr(request.app.state, "analytics", None)
    if analytics_engine is None or not analytics_engine.is_loaded:
        raise HTTPException(status_code=503, detail="Analytics engine is not available")

    try:
        alerts = await analytics_engine.check_drift(
            avatar_id, baseline_days=baseline_days, recent_days=recent_days,
        )
        return DriftAlertListResponse(
            avatar_id=avatar_id,
            alerts=[
                DriftAlertResponse(
                    metric=a.metric,
                    baseline_value=a.baseline_value,
                    current_value=a.current_value,
                    change_percent=a.change_percent,
                    severity=a.severity,
                )
                for a in alerts
            ],
            count=len(alerts),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get(
    "/analytics/{avatar_id}/report",
    response_model=ReportExportResponse,
    summary="Export Report",
    description="Export analytics report.",
)
async def analytics_report(
    avatar_id: str,
    request: Request,
    days: int = Query(default=30, ge=1, le=365, description="Report period"),
) -> ReportExportResponse:
    """Export analytics report."""
    analytics_engine = getattr(request.app.state, "analytics", None)
    if analytics_engine is None or not analytics_engine.is_loaded:
        raise HTTPException(status_code=503, detail="Analytics engine is not available")

    try:
        import time as _time
        report = await analytics_engine.export_report(avatar_id, days=days)
        kpis = report["kpis"]
        return ReportExportResponse(
            avatar_id=avatar_id,
            period_days=days,
            generated_at=_time.time(),
            kpis=KPISnapshotResponse(
                avatar_id=avatar_id,
                total_conversations=kpis.total_conversations,
                total_messages=kpis.total_messages,
                avg_response_time_ms=kpis.avg_response_time_ms,
                avg_kb_confidence=kpis.avg_kb_confidence,
                escalation_rate=kpis.escalation_rate,
                autonomy_percent=kpis.autonomy_percent,
                resolution_time_avg_s=kpis.resolution_time_avg_s,
                accuracy_score=kpis.accuracy_score,
                total_cost=kpis.total_cost,
            ),
            skill_breakdown=report.get("skill_breakdown", []),
            daily_trends={
                k: [TimeseriesPoint(date=p.date, value=p.value) for p in v]
                for k, v in report.get("daily_trends", {}).items()
            },
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# =============================================================================
# Admin — Kill Switch
# =============================================================================


@router.post(
    "/admin/suspend/{customer_id}",
    response_model=SuspendResponse,
    summary="Suspend Customer",
    description="Immediately suspend a customer account.",
)
async def suspend_customer(
    customer_id: str,
    body: SuspendRequest,
    request: Request,
) -> SuspendResponse:
    """Suspend a customer account."""
    kill_switch = getattr(request.app.state, "kill_switch", None)
    if kill_switch is None:
        raise HTTPException(status_code=503, detail="Kill switch is not available")

    await kill_switch.suspend(customer_id, reason=body.reason)
    return SuspendResponse(
        customer_id=customer_id,
        suspended=True,
        reason=body.reason,
    )


@router.post(
    "/admin/resume/{customer_id}",
    response_model=SuspendResponse,
    summary="Resume Customer",
    description="Resume a suspended customer account.",
)
async def resume_customer(customer_id: str, request: Request) -> SuspendResponse:
    """Resume a suspended customer."""
    kill_switch = getattr(request.app.state, "kill_switch", None)
    if kill_switch is None:
        raise HTTPException(status_code=503, detail="Kill switch is not available")

    await kill_switch.resume(customer_id)
    return SuspendResponse(
        customer_id=customer_id,
        suspended=False,
    )


# =============================================================================
# Admin — Customer & Subscription Management
# =============================================================================


@router.post(
    "/admin/customers",
    response_model=CustomerResponse,
    summary="Create Customer",
    description="Create a new customer account. Returns the customer ID and API key.",
)
async def admin_create_customer(
    body: CustomerCreate,
    request: Request,
) -> CustomerResponse:
    """Create a new customer account with a unique API key."""
    from src.config import PLAN_TIERS
    from src.db.models import Customer

    db = getattr(request.app.state, "db", None)
    if db is None:
        raise HTTPException(status_code=503, detail="Database is not available")

    customer_id = uuid.uuid4().hex
    api_key = uuid.uuid4().hex + uuid.uuid4().hex[:8]  # 40 char hex key

    async with db.session() as session:
        # Check for duplicate email
        from sqlalchemy import select

        existing = await session.execute(
            select(Customer).where(Customer.email == body.email)
        )
        if existing.scalars().first():
            raise HTTPException(status_code=409, detail="Customer with this email already exists")

        customer = Customer(
            id=customer_id,
            name=body.name,
            email=body.email,
            company=body.company or "",
            api_key=api_key,
            operator_language=body.operator_language,
            data_language=body.data_language,
        )
        session.add(customer)
        await session.commit()

    logger.info("Admin created customer %s (%s)", customer_id, body.email)
    return CustomerResponse(
        customer_id=customer_id,
        api_key=api_key,
        email=body.email,
        name=body.name,
        operator_language=body.operator_language,
        data_language=body.data_language,
    )


@router.post(
    "/admin/customers/{customer_id}/subscription",
    response_model=AdminSubscriptionResponse,
    summary="Create Subscription",
    description="Create a subscription for a customer.",
)
async def admin_create_subscription(
    customer_id: str,
    body: AdminSubscriptionRequest,
    request: Request,
) -> AdminSubscriptionResponse:
    """Create or replace a customer's active subscription."""
    from src.config import PLAN_TIERS
    from src.db.models import Customer, Subscription

    db = getattr(request.app.state, "db", None)
    if db is None:
        raise HTTPException(status_code=503, detail="Database is not available")

    if body.plan not in PLAN_TIERS:
        raise HTTPException(status_code=400, detail=f"Invalid plan: {body.plan}")

    tier = PLAN_TIERS[body.plan]

    async with db.session() as session:
        from sqlalchemy import select, update

        # Verify customer exists
        customer = await session.execute(
            select(Customer).where(Customer.id == customer_id)
        )
        if not customer.scalars().first():
            raise HTTPException(status_code=404, detail="Customer not found")

        # Deactivate any existing active subscription
        await session.execute(
            update(Subscription)
            .where(Subscription.customer_id == customer_id, Subscription.is_active.is_(True))
            .values(is_active=False)
        )

        # Create new subscription
        sub_id = uuid.uuid4().hex
        sub = Subscription(
            id=sub_id,
            customer_id=customer_id,
            plan=body.plan,
            monthly_seconds=tier["monthly_seconds"],
            rate_per_second=0.001,
            max_avatars=tier["max_avatars"],
            max_concurrent_sessions=tier["max_concurrent"],
            price_monthly=tier["price_monthly"],
            is_active=True,
        )
        session.add(sub)
        await session.commit()

    logger.info("Admin created %s subscription for customer %s", body.plan, customer_id)
    return AdminSubscriptionResponse(
        subscription_id=sub_id,
        customer_id=customer_id,
        plan=body.plan,
        monthly_seconds=tier["monthly_seconds"],
        max_avatars=tier["max_avatars"],
        max_concurrent_sessions=tier["max_concurrent"],
        price_monthly=tier["price_monthly"],
    )


@router.put(
    "/admin/customers/{customer_id}/subscription",
    response_model=AdminSubscriptionResponse,
    summary="Update Subscription",
    description="Update a customer's subscription plan tier.",
)
async def admin_update_subscription(
    customer_id: str,
    body: AdminSubscriptionRequest,
    request: Request,
) -> AdminSubscriptionResponse:
    """Update the active subscription's plan tier (upgrade/downgrade)."""
    from src.config import PLAN_TIERS
    from src.db.models import Subscription

    db = getattr(request.app.state, "db", None)
    if db is None:
        raise HTTPException(status_code=503, detail="Database is not available")

    if body.plan not in PLAN_TIERS:
        raise HTTPException(status_code=400, detail=f"Invalid plan: {body.plan}")

    tier = PLAN_TIERS[body.plan]

    async with db.session() as session:
        from sqlalchemy import select

        result = await session.execute(
            select(Subscription).where(
                Subscription.customer_id == customer_id,
                Subscription.is_active.is_(True),
            )
        )
        sub = result.scalars().first()
        if not sub:
            raise HTTPException(status_code=404, detail="No active subscription found")

        sub.plan = body.plan
        sub.monthly_seconds = tier["monthly_seconds"]
        sub.max_avatars = tier["max_avatars"]
        sub.max_concurrent_sessions = tier["max_concurrent"]
        sub.price_monthly = tier["price_monthly"]
        await session.commit()

    logger.info("Admin updated subscription to %s for customer %s", body.plan, customer_id)
    return AdminSubscriptionResponse(
        subscription_id=sub.id,
        customer_id=customer_id,
        plan=body.plan,
        monthly_seconds=tier["monthly_seconds"],
        max_avatars=tier["max_avatars"],
        max_concurrent_sessions=tier["max_concurrent"],
        price_monthly=tier["price_monthly"],
    )


@router.delete(
    "/admin/customers/{customer_id}/subscription",
    summary="Delete Subscription",
    description="Deactivate a customer's subscription.",
)
async def admin_delete_subscription(
    customer_id: str,
    request: Request,
) -> dict:
    """Deactivate the customer's active subscription."""
    from src.db.models import Subscription

    db = getattr(request.app.state, "db", None)
    if db is None:
        raise HTTPException(status_code=503, detail="Database is not available")

    async with db.session() as session:
        from sqlalchemy import update

        result = await session.execute(
            update(Subscription)
            .where(Subscription.customer_id == customer_id, Subscription.is_active.is_(True))
            .values(is_active=False)
        )
        await session.commit()

    logger.info("Admin deactivated subscription for customer %s", customer_id)
    return {"customer_id": customer_id, "subscription_deactivated": True}


# =============================================================================
# WhatsApp Webhook
# =============================================================================


@router.get("/whatsapp/webhook", include_in_schema=False)
async def verify_webhook(
    request: Request,
    mode: str = Query(alias="hub.mode"),
    token: str = Query(alias="hub.verify_token"),
    challenge: str = Query(alias="hub.challenge"),
):
    """Verify WhatsApp webhook subscription."""
    whatsapp = request.app.state.whatsapp
    result = whatsapp.verify_webhook(mode, token, challenge)
    if result:
        # Return plain text response required by Meta
        return Response(content=result, media_type="text/plain")
    raise HTTPException(status_code=403, detail="Verification failed")


@router.post("/whatsapp/webhook", include_in_schema=False)
async def receive_message(request: Request, background_tasks: BackgroundTasks):
    """Receive and process WhatsApp messages."""
    whatsapp = request.app.state.whatsapp

    # Verify signature
    body = await request.body()
    signature = request.headers.get("X-Hub-Signature-256", "")
    if not whatsapp.verify_signature(body, signature):
        raise HTTPException(status_code=401, detail="Invalid signature")

    # Process in background
    try:
        payload = await request.json()
        background_tasks.add_task(_process_whatsapp_payload, payload, request.app)
    except Exception as exc:
        logger.error(f"Failed to process webhook: {exc}")

    return {"status": "ok"}


async def _process_whatsapp_payload(payload: dict, app):
    """Background task to process incoming WhatsApp messages."""
    whatsapp = app.state.whatsapp
    pipeline = app.state.pipeline

    try:
        messages = whatsapp.parse_incoming(payload)
        for msg in messages:
            msg_id = msg.get("message_id")
            if not msg_id or whatsapp.is_duplicate(msg_id):
                continue

            # Mark read
            await whatsapp.mark_read(msg_id)
            sender = msg.get("from_number")

            if msg["type"] == "text":
                text = msg.get("text", "")
                if text:
                    result = await pipeline.process_text(text=text, session_id=sender)
                    if result.response_text:
                        await whatsapp.send_text(sender, result.response_text)
                    
                    # Send audio if available and valid
                    if result.audio_path:
                        try:
                            audio_url = _build_public_audio_url(result.audio_path, app)
                            await whatsapp.send_audio(sender, audio_url)
                        except Exception as exc:
                            logger.warning(f"Failed to send WhatsApp audio: {exc}")

            elif msg["type"] == "audio":
                media_id = msg.get("media_id")
                mime_type = msg.get("mime_type", "")
                if media_id:
                    audio_path = await whatsapp.download_media(media_id, mime_type)
                    result = await pipeline.process_audio(audio_path=str(audio_path), session_id=sender)
                    if result.response_text:
                        await whatsapp.send_text(sender, result.response_text)
                    
                    # Send audio if available and valid
                    if result.audio_path:
                        try:
                            audio_url = _build_public_audio_url(result.audio_path, app)
                            await whatsapp.send_audio(sender, audio_url)
                        except Exception as exc:
                            logger.warning(f"Failed to send WhatsApp audio: {exc}")

    except Exception as exc:
        logger.error(f"Error processing WhatsApp payload: {exc}")


# =============================================================================
# VRM Avatar
# =============================================================================

_MAX_VRM_SIZE = 50 * 1024 * 1024  # 50 MB


@router.post(
    "/avatars/{avatar_id}/vrm",
    response_model=VRMUploadResponse,
    description="Upload a VRM model file for an avatar.",
)
async def upload_vrm(
    avatar_id: str,
    request: Request,
    file: UploadFile = File(...),
) -> VRMUploadResponse:
    """Upload a .vrm model file and set the avatar to VRM mode."""
    filename = file.filename or ""
    if not filename.lower().endswith(".vrm"):
        raise HTTPException(status_code=400, detail="File must have .vrm extension")

    config = request.app.state.config
    vrm_dir = config.static_files_dir / "vrm" / avatar_id
    vrm_dir.mkdir(parents=True, exist_ok=True)
    dest = vrm_dir / "model.vrm"

    total = 0
    size_exceeded = False
    async with aiofiles.open(dest, "wb") as f:
        while True:
            chunk = await file.read(64 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > _MAX_VRM_SIZE:
                size_exceeded = True
                break
            await f.write(chunk)
    if size_exceeded:
        dest.unlink(missing_ok=True)
        raise HTTPException(
            status_code=413,
            detail=f"VRM file exceeds {_MAX_VRM_SIZE // (1024 * 1024)}MB limit",
        )

    vrm_url = f"/files/vrm/{avatar_id}/model.vrm"
    logger.info("VRM uploaded", extra={"avatar_id": avatar_id, "size": total})
    return VRMUploadResponse(
        avatar_id=avatar_id,
        vrm_url=vrm_url,
        file_size_bytes=total,
    )


@router.get(
    "/avatars/{avatar_id}/vrm-info",
    response_model=VRMInfoResponse,
    description="Get VRM metadata for an avatar.",
)
async def get_vrm_info(avatar_id: str, request: Request) -> VRMInfoResponse:
    """Get VRM file info and avatar type."""
    config = request.app.state.config
    vrm_path = config.static_files_dir / "vrm" / avatar_id / "model.vrm"
    has_vrm = vrm_path.exists()
    vrm_url = f"/files/vrm/{avatar_id}/model.vrm" if has_vrm else None
    return VRMInfoResponse(
        avatar_id=avatar_id,
        avatar_type="vrm" if has_vrm else "video",
        vrm_url=vrm_url,
        has_vrm=has_vrm,
    )


@router.put(
    "/avatars/{avatar_id}/type",
    description="Switch avatar rendering type between video and vrm.",
)
async def set_avatar_type(
    avatar_id: str, body: AvatarTypeRequest, request: Request,
) -> dict:
    """Set the avatar rendering type."""
    if body.avatar_type == "vrm":
        config = request.app.state.config
        vrm_path = config.static_files_dir / "vrm" / avatar_id / "model.vrm"
        if not vrm_path.exists():
            raise HTTPException(
                status_code=400,
                detail="No VRM file uploaded for this avatar. Upload one first via POST /avatars/{id}/vrm",
            )
    logger.info("Avatar type changed", extra={"avatar_id": avatar_id, "type": body.avatar_type})
    return {"avatar_id": avatar_id, "avatar_type": body.avatar_type, "message": "Avatar type updated"}


# =============================================================================
# Photo Upload & Face Preprocessing Pipeline
# =============================================================================

_MAX_PHOTO_SIZE = 10 * 1024 * 1024  # 10 MB


@router.post(
    "/avatars/{avatar_id}/photo",
    summary="Upload Avatar Photo",
    description="Upload a photo for video avatar rendering. Triggers RunPod face preprocessing.",
)
async def upload_avatar_photo(
    avatar_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> dict:
    """Upload an avatar photo to R2 and trigger face preprocessing.

    1. Upload photo to R2
    2. Update avatar.photo_url
    3. Trigger RunPod face preprocessing (async background task)
    4. On completion: set face_data_url + photo_preprocessed = True + avatar_mode = "video"
    5. On failure: avatar stays in VRM mode
    """
    filename = file.filename or ""
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=400, detail="Photo must be .jpg, .jpeg, or .png")

    # Read photo bytes with size limit
    photo_bytes = b""
    while True:
        chunk = await file.read(64 * 1024)
        if not chunk:
            break
        photo_bytes += chunk
        if len(photo_bytes) > _MAX_PHOTO_SIZE:
            raise HTTPException(status_code=413, detail="Photo exceeds 10MB limit")

    if not photo_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    config = request.app.state.config
    db = getattr(request.app.state, "db", None)

    # Upload to R2
    from src.services.r2_storage import R2Storage
    r2 = R2Storage(config)
    photo_url = r2.upload_employee_photo(avatar_id, photo_bytes)

    # Update avatar photo_url in DB
    if db is not None:
        from sqlalchemy import update as sql_update
        from src.db.models import Avatar
        async with db.session() as session:
            await session.execute(
                sql_update(Avatar)
                .where(Avatar.id == avatar_id)
                .values(photo_url=photo_url)
            )
            await session.commit()

    # Trigger face preprocessing in background
    background_tasks.add_task(
        _preprocess_face_background, config, db, avatar_id, photo_url,
    )

    logger.info("Photo uploaded, preprocessing triggered", extra={"avatar_id": avatar_id})
    return {
        "avatar_id": avatar_id,
        "photo_url": photo_url,
        "status": "preprocessing",
        "message": "Photo uploaded. Face preprocessing started in background.",
    }


async def _preprocess_face_background(
    config, db, avatar_id: str, photo_url: str,
) -> None:
    """Background task: run RunPod face preprocessing and update avatar."""
    try:
        from src.services.runpod_client import RunPodServerless

        runpod = RunPodServerless(config)
        result = await runpod.preprocess_face(
            photo_url=photo_url,
            employee_id=avatar_id,
        )
        await runpod.close()

        # Update avatar with face data
        if db is not None:
            from sqlalchemy import update as sql_update
            from src.db.models import Avatar
            async with db.session() as session:
                await session.execute(
                    sql_update(Avatar)
                    .where(Avatar.id == avatar_id)
                    .values(
                        face_data_url=result.face_data_url,
                        photo_preprocessed=True,
                        avatar_type="video",
                    )
                )
                await session.commit()

        logger.info(
            "Face preprocessing complete",
            extra={
                "avatar_id": avatar_id,
                "face_data_url": result.face_data_url,
                "cost_usd": f"{result.cost_usd:.6f}",
            },
        )
    except Exception as exc:
        logger.error(f"Face preprocessing failed for {avatar_id}: {exc}")
        # Avatar stays in VRM mode — no DB update needed


# =============================================================================
# Voice Cloning Pipeline
# =============================================================================


@router.post(
    "/avatars/{avatar_id}/voice-enroll",
    summary="Voice Enrollment",
    description="Upload an audio sample to clone the avatar's voice via DashScope.",
)
async def voice_enroll(
    avatar_id: str,
    request: Request,
    file: UploadFile = File(...),
) -> dict:
    """Upload a voice sample, clone via DashScope, save voice_id to avatar.

    1. Upload audio sample to R2
    2. Call DashScope voice-enrollment API with R2 URL
    3. Save voice_id to avatar record
    4. Log cost ($0.20)
    """
    filename = file.filename or ""
    if not filename.lower().endswith((".wav", ".mp3", ".m4a", ".ogg")):
        raise HTTPException(status_code=400, detail="Audio must be .wav, .mp3, .m4a, or .ogg")

    audio_bytes = b""
    while True:
        chunk = await file.read(64 * 1024)
        if not chunk:
            break
        audio_bytes += chunk
        if len(audio_bytes) > _MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=413, detail="Audio exceeds 25MB limit")

    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    config = request.app.state.config
    db = getattr(request.app.state, "db", None)

    # Upload voice sample to R2
    from src.services.r2_storage import R2Storage
    r2 = R2Storage(config)
    sample_url = r2.upload_voice_sample(avatar_id, audio_bytes)

    # Clone voice via DashScope TTS
    from src.pipeline.tts import TTSEngine
    tts = TTSEngine(config)
    voice_info = await tts.clone_voice(
        audio_url=sample_url,
        prefix=f"avatar_{avatar_id}",
        language="ar",
    )

    # Update avatar voice_id in DB
    if db is not None:
        from sqlalchemy import update as sql_update
        from src.db.models import Avatar
        async with db.session() as session:
            await session.execute(
                sql_update(Avatar)
                .where(Avatar.id == avatar_id)
                .values(voice_id=voice_info.voice_id)
            )
            await session.commit()

    logger.info(
        "Voice enrolled",
        extra={
            "avatar_id": avatar_id,
            "voice_id": voice_info.voice_id,
            "cost_usd": f"{voice_info.cost_usd:.4f}",
        },
    )
    return {
        "avatar_id": avatar_id,
        "voice_id": voice_info.voice_id,
        "cost_usd": voice_info.cost_usd,
        "message": "Voice cloned successfully.",
    }


# =============================================================================
# Helpers
# =============================================================================


_MAX_UPLOAD_SIZE = 25 * 1024 * 1024  # 25 MB


def _save_upload(upload: UploadFile, request: Request) -> Path:
    """Save an uploaded file to the temp directory.

    Reads the file in chunks to enforce a 25 MB size limit without
    buffering the entire file in memory.

    Args:
        upload: Uploaded file from the request.
        request: FastAPI request for config access.

    Returns:
        Path to the saved temporary file.

    Raises:
        HTTPException: If the file cannot be saved or exceeds size limit.
    """
    try:
        config = request.app.state.config
        temp_dir = config.storage_base_dir / "uploads"
        temp_dir.mkdir(parents=True, exist_ok=True)

        suffix = Path(upload.filename or "audio.wav").suffix
        temp_path = temp_dir / f"upload_{uuid.uuid4().hex[:12]}{suffix}"

        total = 0
        with open(temp_path, "wb") as f:
            while True:
                chunk = upload.file.read(64 * 1024)  # 64 KB chunks
                if not chunk:
                    break
                total += len(chunk)
                if total > _MAX_UPLOAD_SIZE:
                    f.close()
                    temp_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"Upload exceeds {_MAX_UPLOAD_SIZE // (1024 * 1024)}MB limit",
                    )
                f.write(chunk)

        logger.info("Upload saved", extra={"path": str(temp_path), "size_bytes": total})
        return temp_path

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to save uploaded file: {exc}",
        ) from exc


def _to_file_url(file_path: str, request: Request) -> str:
    """Convert a local file path to a serveable URL.

    Args:
        file_path: Absolute or relative path to the file.
        request: FastAPI request for base URL construction.

    Returns:
        URL string pointing to the file via the static mount.
    """
    path = Path(file_path)
    # Serve files relative to the static files directory
    try:
        config = request.app.state.config
        static_dir = config.static_files_dir
        static_dir.mkdir(parents=True, exist_ok=True)

        # Copy/link to static dir if not already there
        if not str(path).startswith(str(static_dir)):
            import shutil
            dest = static_dir / path.name
            shutil.copy2(str(path), str(dest))
            relative = path.name
        else:
            relative = path.name

        base_url = str(request.base_url).rstrip("/")
        return f"{base_url}/files/{relative}"

    except Exception:
        return f"/files/{path.name}"
def _build_public_audio_url(file_path: str, app) -> str:
    """Build a public URL for an audio file using config settings.

    Args:
        file_path: Absolute path to the audio file.
        app: FastAPI application instance for config access.

    Returns:
        Publicly accessible URL string.
    """
    path = Path(file_path)
    config = app.state.config
    
    # Ensure file is in static directory
    static_dir = config.static_files_dir
    static_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy if not already in static dir
    if not str(path).startswith(str(static_dir)):
        import shutil
        dest = static_dir / path.name
        try:
            shutil.copy2(str(path), str(dest))
        except shutil.SameFileError:
            pass
        relative = path.name
    else:
        relative = path.name

    # Build base URL from webhook URL or fallback
    base_url = ""
    if config.whatsapp_webhook_url:
        # Strip /api/v1/whatsapp/webhook suffix to get root
        # Example: https://example.com/api/v1/whatsapp/webhook -> https://example.com
        import re
        base_url = re.sub(r"/api/v1/whatsapp/webhook/?$", "", config.whatsapp_webhook_url)
    
    # Fallback to empty string (resulting in relative URL) if no public URL configured
    return f"{base_url}/files/{relative}"


# =============================================================================
# Subscription Lifecycle: Freeze / Cancel / Reactivate
# =============================================================================


@router.post(
    "/admin/subscriptions/{customer_id}/freeze",
    summary="Freeze Subscription",
    description="Temporarily suspend a customer's service.",
    tags=["admin"],
)
async def freeze_subscription(
    customer_id: str,
    request: Request,
    reason: str = Query(default="", description="Reason for freezing"),
):
    from src.services.subscription import SubscriptionLifecycle

    db = getattr(request.app.state, "db", None)
    lifecycle = SubscriptionLifecycle(db=db)
    result = await lifecycle.freeze(customer_id, reason=reason)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.post(
    "/admin/subscriptions/{customer_id}/cancel",
    summary="Cancel Subscription",
    description="Permanently cancel a customer's subscription.",
    tags=["admin"],
)
async def cancel_subscription(
    customer_id: str,
    request: Request,
    reason: str = Query(default="", description="Cancellation reason"),
    purge_media: bool = Query(default=False, description="Immediately purge R2 media"),
):
    from src.services.subscription import SubscriptionLifecycle

    db = getattr(request.app.state, "db", None)
    r2 = getattr(request.app.state, "storage", None)
    lifecycle = SubscriptionLifecycle(db=db, r2_storage=r2)
    result = await lifecycle.cancel(customer_id, reason=reason, purge_media=purge_media)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.post(
    "/admin/subscriptions/{customer_id}/reactivate",
    summary="Reactivate Subscription",
    description="Reactivate a frozen or cancelled subscription.",
    tags=["admin"],
)
async def reactivate_subscription(
    customer_id: str,
    request: Request,
    plan_id: str = Query(default="", description="New plan ID (optional)"),
):
    from src.services.subscription import SubscriptionLifecycle

    db = getattr(request.app.state, "db", None)
    lifecycle = SubscriptionLifecycle(db=db)
    result = await lifecycle.reactivate(customer_id, plan_id=plan_id)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result
