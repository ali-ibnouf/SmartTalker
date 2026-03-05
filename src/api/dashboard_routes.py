"""Dashboard API — customer-facing dashboard endpoints.

REST endpoints for the customer dashboard that wrap database queries.
Provides overview stats, avatar management, conversation history,
billing info, and server health for the authenticated customer.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/api/v1/dashboard", tags=["dashboard"])


@router.get("/overview")
async def dashboard_overview(request: Request) -> JSONResponse:
    """Aggregated dashboard stats for the customer."""
    db = getattr(request.app.state, "db", None)
    if db is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Database not available"},
        )

    from sqlalchemy import select, func as sa_func
    from src.db.models import Avatar, Conversation, UsageRecord

    # For now, return aggregate counts (customer filtering via auth middleware later)
    async with db.session() as session:
        avatar_count = (await session.execute(
            select(sa_func.count(Avatar.id))
        )).scalar() or 0

        conversation_count = (await session.execute(
            select(sa_func.count(Conversation.id))
        )).scalar() or 0

        total_duration = (await session.execute(
            select(sa_func.coalesce(sa_func.sum(UsageRecord.duration_s), 0.0))
        )).scalar() or 0.0

        total_cost = (await session.execute(
            select(sa_func.coalesce(sa_func.sum(UsageRecord.cost), 0.0))
        )).scalar() or 0.0

    return JSONResponse(content={
        "avatars": avatar_count,
        "conversations": conversation_count,
        "total_duration_s": round(total_duration, 2),
        "total_cost": round(total_cost, 6),
    })


@router.get("/avatars")
async def list_avatars(request: Request) -> JSONResponse:
    """List customer's avatars."""
    db = getattr(request.app.state, "db", None)
    if db is None:
        return JSONResponse(status_code=503, content={"error": "Database not available"})

    from sqlalchemy import select
    from src.db.models import Avatar

    async with db.session() as session:
        result = await session.execute(
            select(Avatar).order_by(Avatar.created_at.desc())
        )
        avatars = result.scalars().all()

    return JSONResponse(content={
        "avatars": [
            {
                "id": a.id,
                "name": a.name,
                "photo_url": a.photo_url,
                "voice_id": a.voice_id,
                "language": a.language,
                "is_live": a.is_live,
                "training_progress": a.training_progress,
                "created_at": a.created_at.isoformat() if a.created_at else None,
            }
            for a in avatars
        ]
    })


@router.get("/conversations")
async def list_conversations(
    request: Request,
    page: int = 1,
    limit: int = 20,
    avatar_id: Optional[str] = None,
) -> JSONResponse:
    """Paginated conversation history."""
    db = getattr(request.app.state, "db", None)
    if db is None:
        return JSONResponse(status_code=503, content={"error": "Database not available"})

    from sqlalchemy import select, func as sa_func
    from src.db.models import Conversation

    offset = (page - 1) * limit

    async with db.session() as session:
        stmt = select(Conversation)
        count_stmt = select(sa_func.count(Conversation.id))

        if avatar_id:
            stmt = stmt.where(Conversation.avatar_id == avatar_id)
            count_stmt = count_stmt.where(Conversation.avatar_id == avatar_id)

        total = (await session.execute(count_stmt)).scalar() or 0

        result = await session.execute(
            stmt.order_by(Conversation.started_at.desc())
            .offset(offset)
            .limit(limit)
        )
        conversations = result.scalars().all()

    return JSONResponse(content={
        "conversations": [
            {
                "id": c.id,
                "avatar_id": c.avatar_id,
                "channel": c.channel,
                "caller_id": c.caller_id,
                "started_at": c.started_at.isoformat() if c.started_at else None,
                "ended_at": c.ended_at.isoformat() if c.ended_at else None,
                "duration_s": c.duration_s,
                "message_count": c.message_count,
                "total_cost": c.total_cost,
            }
            for c in conversations
        ],
        "total": total,
        "page": page,
        "limit": limit,
    })


@router.get("/conversations/{conversation_id}")
async def get_conversation(request: Request, conversation_id: str) -> JSONResponse:
    """Conversation detail with messages."""
    db = getattr(request.app.state, "db", None)
    if db is None:
        return JSONResponse(status_code=503, content={"error": "Database not available"})

    from sqlalchemy import select
    from src.db.models import Conversation, ConversationMessage

    async with db.session() as session:
        conv_result = await session.execute(
            select(Conversation).where(Conversation.id == conversation_id)
        )
        conv = conv_result.scalar_one_or_none()

        if conv is None:
            return JSONResponse(status_code=404, content={"error": "Conversation not found"})

        msg_result = await session.execute(
            select(ConversationMessage)
            .where(ConversationMessage.conversation_id == conversation_id)
            .order_by(ConversationMessage.created_at)
        )
        messages = msg_result.scalars().all()

    return JSONResponse(content={
        "conversation": {
            "id": conv.id,
            "avatar_id": conv.avatar_id,
            "channel": conv.channel,
            "started_at": conv.started_at.isoformat() if conv.started_at else None,
            "duration_s": conv.duration_s,
            "message_count": conv.message_count,
        },
        "messages": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "emotion": m.emotion,
                "kb_confidence": m.kb_confidence,
                "escalated": m.escalated,
                "latency_ms": m.latency_ms,
                "created_at": m.created_at.isoformat() if m.created_at else None,
            }
            for m in messages
        ],
    })


@router.get("/server")
async def server_health(request: Request) -> JSONResponse:
    """Server health + render node count."""
    pipeline = request.app.state.pipeline
    health = await pipeline.health_check()

    return JSONResponse(content={
        **health,
        "render_nodes": 0,
    })


@router.get("/billing")
async def billing_overview(request: Request) -> JSONResponse:
    """Billing plan + usage summary."""
    billing = getattr(request.app.state, "billing", None)
    if billing is None:
        return JSONResponse(content={"billing_enabled": False})

    # For now return general info (customer-specific after auth)
    active_sessions = await billing.get_active_sessions()
    return JSONResponse(content={
        "billing_enabled": True,
        "rate_per_second": billing._rate,
        "active_sessions": len(active_sessions),
    })
