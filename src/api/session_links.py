"""Session link API endpoints.

POST /api/v1/session-links/     — Create link (admin auth required)
GET  /api/v1/session-links/{t}  — Get session data (public, token = auth)
DELETE /api/v1/session-links/{t} — Invalidate link (admin auth required)
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from src.config import get_settings
from src.services.session_link_service import SessionLinkService
from src.utils.logger import setup_logger

logger = setup_logger("api.session_links")

router = APIRouter(prefix="/api/v1/session-links", tags=["session-links"])


# ── Request / Response schemas ─────────────────────────────────────────────


class CreateSessionLinkRequest(BaseModel):
    customer_id: str
    avatar_id: str
    visitor_phone: Optional[str] = None
    visitor_name: Optional[str] = None
    language: str = "ar"
    service_type: Optional[str] = None
    collected_docs: Optional[dict[str, Any]] = None
    context: Optional[dict[str, Any]] = None
    expires_minutes: int = Field(default=30, ge=1, le=1440)
    channel_source: str = "whatsapp"


class SessionLinkResponse(BaseModel):
    token: str
    url: str
    expires_at: str
    expires_minutes: int


# ── Helper ─────────────────────────────────────────────────────────────────


def _get_service(request: Request) -> SessionLinkService:
    """Get SessionLinkService from app state (Redis + config)."""
    redis = getattr(request.app.state, "redis", None)
    if redis is None:
        raise HTTPException(status_code=503, detail="Redis not available")
    config = get_settings()
    return SessionLinkService(redis, config.session_link_base_url)


# ── Endpoints ──────────────────────────────────────────────────────────────


@router.post("/", response_model=SessionLinkResponse)
async def create_session_link(body: CreateSessionLinkRequest, request: Request):
    """Create a shareable session link.

    Called by WhatsApp channel when transferring to video.
    Requires admin API key (enforced by middleware on /api/v1/* paths).
    """
    service = _get_service(request)
    config = get_settings()
    result = await service.create_link(
        customer_id=body.customer_id,
        avatar_id=body.avatar_id,
        visitor_phone=body.visitor_phone,
        visitor_name=body.visitor_name,
        language=body.language,
        service_type=body.service_type,
        collected_docs=body.collected_docs,
        context=body.context,
        expires_minutes=body.expires_minutes,
        channel_source=body.channel_source,
    )
    return SessionLinkResponse(**result)


@router.get("/{token}")
async def get_session_link(token: str, request: Request):
    """Get session data by token.

    Called by the public session page (/s/{token}) on load.
    NO auth required — token IS the auth (path excluded in middleware).
    """
    service = _get_service(request)
    session = await service.get_session(token)

    if not session:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "SESSION_NOT_FOUND",
                "message": "رابط الجلسة غير موجود أو انتهت صلاحيته",
                "message_en": "Session link not found or expired",
            },
        )

    # Activate on first access
    if session.get("status") == "pending":
        await service.activate_session(token)
        session["status"] = "active"

    # Return safe subset — no internal secrets exposed
    return {
        "token": token,
        "avatar_id": session["avatar_id"],
        "customer_id": session["customer_id"],
        "language": session["language"],
        "visitor_name": session.get("visitor_name"),
        "service_type": session.get("service_type"),
        "context": session.get("context", {}),
        "allow_camera": session.get("allow_camera", True),
        "allow_microphone": session.get("allow_microphone", True),
        "expires_at": session["expires_at"],
        "status": session["status"],
    }


@router.delete("/{token}")
async def invalidate_session_link(token: str, request: Request):
    """Manually invalidate a session link.

    Requires admin API key (enforced by middleware on /api/v1/* paths).
    """
    service = _get_service(request)
    success = await service.invalidate_link(token)
    if not success:
        raise HTTPException(status_code=404, detail="Session link not found")
    return {"success": True, "token": token}
