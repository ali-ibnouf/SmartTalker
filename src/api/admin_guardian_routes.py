"""Admin Cost Guardian API routes.

Provides endpoints to monitor and control the Cost Guardian:
- Guardian status + active pauses
- Recent alerts from cost_guardian_log
- Manual unpause service / customer / emergency
- Trigger daily report
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Request, Query
from pydantic import BaseModel

from sqlalchemy import select, func, and_

from src.db.models import CostGuardianLog
from src.utils.logger import setup_logger

logger = setup_logger("api.admin_guardian")

router = APIRouter(prefix="/api/v1/admin/cost-guardian", tags=["admin-cost-guardian"])


# ── Response Schemas ────────────────────────────────────────────────────


class GuardianStatusResponse(BaseModel):
    running: bool = False
    emergency_paused: bool = False
    paused_services: list[str] = []
    paused_customers: list[str] = []


class GuardianAlertItem(BaseModel):
    id: str = ""
    alert_level: str = ""
    service: str = ""
    message: str = ""
    current_value: float = 0.0
    threshold: float = 0.0
    action_taken: str = ""
    result: str = ""
    customer_id: str = ""
    created_at: str = ""


class GuardianAlertsResponse(BaseModel):
    alerts: list[GuardianAlertItem] = []
    total: int = 0


class ActionResponse(BaseModel):
    status: str = ""
    detail: str = ""


# ── Endpoints ───────────────────────────────────────────────────────────


@router.get("/status", response_model=GuardianStatusResponse)
async def guardian_status(request: Request):
    """Current guardian status + any active pauses."""
    guardian = getattr(request.app.state, "cost_guardian", None)
    redis = getattr(request.app.state, "redis", None)

    running = guardian.running if guardian else False
    emergency_paused = False
    paused_services: list[str] = []
    paused_customers: list[str] = []

    if redis:
        emergency_paused = bool(await redis.exists("cost_guardian:emergency_pause"))

        # Scan for paused services
        from src.services.cost_guardian.config import BUDGETS
        for service_key in BUDGETS:
            svc_name = BUDGETS[service_key].service
            if await redis.exists(f"cost_guardian:paused:{svc_name}"):
                paused_services.append(svc_name)

        # Scan for paused customers (scan pattern)
        try:
            cursor = "0"
            while True:
                cursor, keys = await redis.scan(
                    cursor=cursor, match="cost_guardian:customer_paused:*", count=100,
                )
                for key in keys:
                    cid = key.split(":")[-1]
                    paused_customers.append(cid)
                if cursor == "0" or cursor == 0:
                    break
        except Exception:
            pass

    return GuardianStatusResponse(
        running=running,
        emergency_paused=emergency_paused,
        paused_services=paused_services,
        paused_customers=paused_customers,
    )


@router.get("/alerts", response_model=GuardianAlertsResponse)
async def recent_alerts(
    request: Request,
    hours: int = Query(24, ge=1, le=168),
    limit: int = Query(50, ge=1, le=200),
):
    """Get recent alerts from cost_guardian_log."""
    db = getattr(request.app.state, "db", None)
    if db is None:
        return GuardianAlertsResponse()

    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    async with db.session() as session:
        result = await session.execute(
            select(CostGuardianLog)
            .where(CostGuardianLog.created_at >= cutoff)
            .order_by(CostGuardianLog.created_at.desc())
            .limit(limit)
        )
        rows = result.scalars().all()

        count_result = await session.execute(
            select(func.count(CostGuardianLog.id))
            .where(CostGuardianLog.created_at >= cutoff)
        )
        total = int(count_result.scalar_one())

    alerts = [
        GuardianAlertItem(
            id=row.id,
            alert_level=row.alert_level,
            service=row.service,
            message=row.message,
            current_value=row.current_value,
            threshold=row.threshold,
            action_taken=row.action_taken,
            result=row.result,
            customer_id=row.customer_id,
            created_at=row.created_at.isoformat() if row.created_at else "",
        )
        for row in rows
    ]

    return GuardianAlertsResponse(alerts=alerts, total=total)


@router.post("/unpause/{service}", response_model=ActionResponse)
async def unpause_service(request: Request, service: str):
    """Manually unpause a service."""
    redis = getattr(request.app.state, "redis", None)
    if redis:
        await redis.delete(f"cost_guardian:paused:{service}")
    return ActionResponse(status="unpaused", detail=f"Service '{service}' unpaused")


@router.post("/unpause-customer/{customer_id}", response_model=ActionResponse)
async def unpause_customer(request: Request, customer_id: str):
    """Manually unpause a customer."""
    redis = getattr(request.app.state, "redis", None)
    if redis:
        await redis.delete(f"cost_guardian:customer_paused:{customer_id}")
    return ActionResponse(status="unpaused", detail=f"Customer '{customer_id}' unpaused")


@router.post("/emergency-unpause", response_model=ActionResponse)
async def emergency_unpause(request: Request):
    """Remove platform-wide emergency pause."""
    redis = getattr(request.app.state, "redis", None)
    if redis:
        await redis.delete("cost_guardian:emergency_pause")
    return ActionResponse(status="emergency_unpaused", detail="Emergency pause removed")


@router.post("/daily-report", response_model=ActionResponse)
async def trigger_daily_report(request: Request):
    """Manually trigger daily cost report email."""
    guardian = getattr(request.app.state, "cost_guardian", None)
    if guardian:
        await guardian.reporter.send_daily_report()
        return ActionResponse(status="sent", detail="Daily report email triggered")
    return ActionResponse(status="error", detail="Cost Guardian not running")
