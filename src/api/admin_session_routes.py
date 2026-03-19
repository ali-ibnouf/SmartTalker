"""Admin API routes for live sessions and conversation history.

Provides endpoints for the admin dashboard:
- GET /api/v1/admin/sessions         — list active sessions
- GET /api/v1/admin/conversations    — search/filter conversation history
- GET /api/v1/admin/conversations/{id}/messages — get messages for a conversation

All endpoints require admin API key auth (via middleware).
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Request, Query
from sqlalchemy import select, func, desc, and_

from src.db.models import Conversation, ConversationMessage, Customer, Avatar
from src.utils.logger import setup_logger

logger = setup_logger("api.admin_sessions")

router = APIRouter(prefix="/api/v1/admin", tags=["admin-sessions"])


@router.get("/sessions")
async def list_active_sessions(request: Request):
    """List currently active sessions (no ended_at)."""
    db = getattr(request.app.state, "db", None)
    if db is None:
        return {"sessions": [], "count": 0}

    async with db.session() as session:
        result = await session.execute(
            select(
                Conversation.id,
                Conversation.avatar_id,
                Conversation.channel,
                Conversation.caller_id,
                Conversation.language,
                Conversation.started_at,
                Conversation.message_count,
                Conversation.total_cost,
                Avatar.name.label("avatar_name"),
                Customer.name.label("customer_name"),
                Customer.id.label("customer_id"),
            )
            .outerjoin(Avatar, Conversation.avatar_id == Avatar.id)
            .outerjoin(Customer, Avatar.customer_id == Customer.id)
            .where(Conversation.ended_at.is_(None))
            .order_by(Conversation.started_at.desc())
            .limit(100)
        )
        rows = result.all()

    sessions = []
    now = datetime.now(timezone.utc)
    for row in rows:
        started = row.started_at
        duration_s = (now - started).total_seconds() if started else 0
        sessions.append({
            "id": row.id,
            "avatar_id": row.avatar_id,
            "avatar_name": row.avatar_name or "",
            "customer_id": row.customer_id or "",
            "customer_name": row.customer_name or "",
            "channel": row.channel,
            "caller_id": row.caller_id,
            "language": row.language,
            "started_at": started.isoformat() if started else None,
            "duration_s": round(duration_s, 1),
            "message_count": row.message_count,
            "total_cost": float(row.total_cost or 0),
        })

    return {"sessions": sessions, "count": len(sessions)}


@router.get("/conversations")
async def list_conversations(
    request: Request,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    customer_id: str = Query("", description="Filter by customer"),
    channel: str = Query("", description="Filter by channel"),
    search: str = Query("", description="Search caller_id"),
):
    """List conversation history with optional filters."""
    db = getattr(request.app.state, "db", None)
    if db is None:
        return {"conversations": [], "count": 0, "total": 0}

    conditions = []
    if customer_id:
        conditions.append(Customer.id == customer_id)
    if channel:
        conditions.append(Conversation.channel == channel)
    if search:
        conditions.append(Conversation.caller_id.ilike(f"%{search}%"))

    async with db.session() as session:
        # Build base query
        base = (
            select(
                Conversation.id,
                Conversation.avatar_id,
                Conversation.channel,
                Conversation.caller_id,
                Conversation.language,
                Conversation.started_at,
                Conversation.ended_at,
                Conversation.duration_s,
                Conversation.message_count,
                Conversation.total_cost,
                Conversation.gpu_cost,
                Avatar.name.label("avatar_name"),
                Customer.name.label("customer_name"),
                Customer.id.label("customer_id"),
            )
            .outerjoin(Avatar, Conversation.avatar_id == Avatar.id)
            .outerjoin(Customer, Avatar.customer_id == Customer.id)
        )

        if conditions:
            base = base.where(and_(*conditions))

        # Count total
        count_q = select(func.count()).select_from(
            base.subquery()
        )
        total_result = await session.execute(count_q)
        total = total_result.scalar() or 0

        # Fetch page
        result = await session.execute(
            base.order_by(desc(Conversation.started_at))
            .offset(offset)
            .limit(limit)
        )
        rows = result.all()

    conversations = []
    for row in rows:
        conversations.append({
            "id": row.id,
            "avatar_id": row.avatar_id,
            "avatar_name": row.avatar_name or "",
            "customer_id": row.customer_id or "",
            "customer_name": row.customer_name or "",
            "channel": row.channel,
            "caller_id": row.caller_id,
            "language": row.language,
            "started_at": row.started_at.isoformat() if row.started_at else None,
            "ended_at": row.ended_at.isoformat() if row.ended_at else None,
            "duration_s": float(row.duration_s or 0),
            "message_count": row.message_count,
            "total_cost": float(row.total_cost or 0),
            "gpu_cost": float(row.gpu_cost or 0),
        })

    return {"conversations": conversations, "count": len(conversations), "total": total}


@router.get("/conversations/{conversation_id}/messages")
async def get_conversation_messages(
    conversation_id: str,
    request: Request,
):
    """Get all messages for a specific conversation."""
    db = getattr(request.app.state, "db", None)
    if db is None:
        return {"messages": [], "count": 0}

    async with db.session() as session:
        result = await session.execute(
            select(ConversationMessage)
            .where(ConversationMessage.conversation_id == conversation_id)
            .order_by(ConversationMessage.created_at.asc())
        )
        msgs = result.scalars().all()

    messages = []
    for msg in msgs:
        messages.append({
            "id": msg.id,
            "role": msg.role,
            "content": msg.content,
            "created_at": msg.created_at.isoformat() if msg.created_at else None,
        })

    return {"messages": messages, "count": len(messages)}
