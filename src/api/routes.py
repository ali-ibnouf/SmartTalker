"""API route definitions for SmartTalker.

Endpoints: text-to-speech, text-to-video, audio-chat, voice-clone,
list voices, health check, and WebSocket chat.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Query, Request, Response, UploadFile

from src.api.schemas import (
    ErrorResponse,
    HealthResponse,
    PipelineResponse,
    TextRequest,
    VoiceCloneResponse,
    VoiceInfoResponse,
    VoiceListResponse,
)
from src.utils.exceptions import SmartTalkerError
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
# Text-to-Video
# =============================================================================


@router.post(
    "/text-to-video",
    response_model=PipelineResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Text to Video",
    description="Send text, receive AI-generated video with talking avatar.",
)
async def text_to_video(
    body: TextRequest,
    request: Request,
) -> PipelineResponse:
    """Process text input through LLM -> TTS -> Video -> Upscale pipeline.

    Generates a talking-head video from text. Requires VIDEO_ENABLED=true
    and a valid avatar reference image in avatars/<avatar_id>/.

    Args:
        body: TextRequest with text, avatar_id, emotion, language, voice_id.
        request: FastAPI request object (for pipeline access).

    Returns:
        PipelineResponse with audio_url, video_url, response_text, and latency.
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
                enable_video=True,
            ),
            timeout=config.pipeline_timeout,
        )

        audio_url = _to_file_url(result.audio_path, request)
        video_url = _to_file_url(result.video_path, request) if result.video_path else None

        return PipelineResponse(
            audio_url=audio_url,
            video_url=video_url,
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
    description="Check system health, GPU status, and model availability.",
)
async def health_check(request: Request) -> HealthResponse:
    """Return system health information.

    Args:
        request: FastAPI request object.

    Returns:
        HealthResponse with status, GPU info, and loaded models.
    """
    pipeline = request.app.state.pipeline
    health = await pipeline.health_check()

    return HealthResponse(**health)


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
