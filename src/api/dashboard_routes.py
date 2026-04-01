"""Dashboard API — admin + customer dashboard endpoints.

REST endpoints that wrap database queries for both the admin
and customer dashboards. Provides overview stats, avatar management,
conversation history, billing info, and server health.
"""

from __future__ import annotations

import os
import shutil
import time
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from src.utils.logger import setup_logger

logger = setup_logger("api.dashboard")

router = APIRouter(prefix="/api/v1/dashboard", tags=["dashboard"])


@router.get("/overview")
async def dashboard_overview(request: Request) -> JSONResponse:
    """Aggregated dashboard stats (admin overview).

    Returns fields expected by the admin dashboard Command Center:
    total_customers, active_sessions, total_avatars, total_conversations,
    total_duration_hours, total_cost, monthly_revenue.
    """
    db = getattr(request.app.state, "db", None)
    if db is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Database not available"},
        )

    from sqlalchemy import select, func as sa_func
    from src.db.models import Avatar, Conversation, Customer, Subscription, UsageRecord

    async with db.session() as session:
        total_customers = (await session.execute(
            select(sa_func.count(Customer.id)).where(Customer.is_active == True)  # noqa: E712
        )).scalar() or 0

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

        monthly_revenue = (await session.execute(
            select(sa_func.coalesce(sa_func.sum(Subscription.price_monthly), 0.0))
            .where(Subscription.is_active == True)  # noqa: E712
        )).scalar() or 0.0

        # Revenue breakdown by plan (for Revenue page chart)
        plan_rows = (await session.execute(
            select(
                Subscription.plan,
                sa_func.count(Subscription.id).label("count"),
                sa_func.coalesce(sa_func.sum(Subscription.price_monthly), 0.0).label("total"),
            )
            .where(Subscription.is_active == True)  # noqa: E712
            .group_by(Subscription.plan)
            .order_by(sa_func.sum(Subscription.price_monthly).desc())
        )).all()
        revenue_by_plan = [
            {"plan": row.plan.title(), "count": row.count, "total": round(row.total, 2)}
            for row in plan_rows
        ]

    # Active sessions from billing engine
    billing = getattr(request.app.state, "billing", None)
    active_sessions = 0
    if billing:
        try:
            sessions = await billing.get_active_sessions()
            active_sessions = len(sessions)
        except Exception:
            pass

    return JSONResponse(content={
        "total_customers": total_customers,
        "active_sessions": active_sessions,
        "total_avatars": avatar_count,
        "total_conversations": conversation_count,
        "total_duration_hours": round(total_duration / 3600, 2),
        "total_cost": round(total_cost, 6),
        "monthly_revenue": round(monthly_revenue, 2),
        "revenue_by_plan": revenue_by_plan,
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
    """Server health + infrastructure metrics for admin Engineering page.

    Returns: status, models_loaded, uptime_s, cpu_percent, memory_used_mb,
    memory_total_mb, disk_used_gb, disk_total_gb, dashscope_latency_ms,
    runpod_active_workers.
    """
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        return JSONResponse(status_code=503, content={"status": "unavailable", "detail": "Pipeline not initialized"})
    health = await pipeline.health_check()

    # Infrastructure metrics (stdlib, no psutil needed)
    infra = _collect_infra_metrics()

    # DashScope latency — time the LLM health ping
    dashscope_latency_ms = 0.0
    try:
        t0 = time.perf_counter()
        client = await pipeline._llm._get_client()
        resp = await client.post(
            "/chat/completions",
            json={"model": pipeline._llm._model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1},
        )
        if resp.status_code == 200:
            dashscope_latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    except Exception:
        pass

    # RunPod active workers — count active billing sessions that have render jobs
    billing = getattr(request.app.state, "billing", None)
    runpod_workers = 0
    if billing:
        try:
            sessions = await billing.get_active_sessions()
            runpod_workers = len(sessions)
        except Exception:
            pass

    return JSONResponse(content={
        "status": health.get("status", "unknown"),
        "models_loaded": health.get("models_loaded", {}),
        "uptime_s": health.get("uptime_s", 0),
        "cpu_percent": infra["cpu_percent"],
        "memory_used_mb": infra["memory_used_mb"],
        "memory_total_mb": infra["memory_total_mb"],
        "disk_used_gb": infra["disk_used_gb"],
        "disk_total_gb": infra["disk_total_gb"],
        "dashscope_latency_ms": dashscope_latency_ms,
        "runpod_active_workers": runpod_workers,
    })


def _collect_infra_metrics() -> dict[str, Any]:
    """Collect CPU, memory, and disk metrics using stdlib (Linux /proc)."""
    result: dict[str, Any] = {
        "cpu_percent": 0.0,
        "memory_used_mb": 0,
        "memory_total_mb": 0,
        "disk_used_gb": 0.0,
        "disk_total_gb": 0.0,
    }

    # CPU — load average as rough percentage (1-min avg / num CPUs * 100)
    try:
        load_1m = os.getloadavg()[0]
        cpus = os.cpu_count() or 1
        result["cpu_percent"] = round(min(100.0, (load_1m / cpus) * 100), 1)
    except (OSError, AttributeError):
        pass

    # Memory — read /proc/meminfo (Linux)
    try:
        with open("/proc/meminfo") as f:
            meminfo: dict[str, int] = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    meminfo[parts[0].rstrip(":")] = int(parts[1])  # kB
            total_kb = meminfo.get("MemTotal", 0)
            available_kb = meminfo.get("MemAvailable", 0)
            result["memory_total_mb"] = round(total_kb / 1024)
            result["memory_used_mb"] = round((total_kb - available_kb) / 1024)
    except (OSError, ValueError):
        pass

    # Disk
    try:
        usage = shutil.disk_usage("/")
        result["disk_total_gb"] = round(usage.total / (1024 ** 3), 1)
        result["disk_used_gb"] = round(usage.used / (1024 ** 3), 1)
    except OSError:
        pass

    return result


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
