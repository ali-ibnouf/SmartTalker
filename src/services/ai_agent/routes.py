"""FastAPI routes for the AI Optimization Agent."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from src.services.ai_agent.schemas import (
    AgentStatsResponse,
    ApprovalActionResponse,
    ApprovalItem,
    ApprovalListResponse,
    DetectionItem,
    IncidentActionResponse,
    IncidentItem,
    IncidentListResponse,
    PredictionItem,
    PredictionListResponse,
    ScanResponse,
)

router = APIRouter(prefix="/api/v1/agent", tags=["agent"])


def _get_agent(request: Request):
    """Retrieve AIAgent from app state."""
    agent = getattr(request.app.state, "ai_agent", None)
    if agent is None:
        from fastapi.responses import JSONResponse
        return None
    return agent


@router.get("/stats", response_model=AgentStatsResponse)
async def get_agent_stats(request: Request):
    """Get agent summary statistics."""
    agent = _get_agent(request)
    if agent is None:
        return AgentStatsResponse()
    raw = await agent.get_stats()
    return AgentStatsResponse(**raw)


@router.get("/incidents", response_model=IncidentListResponse)
async def get_incidents(
    request: Request,
    status: Optional[str] = None,
    limit: int = 20,
):
    """List agent incidents, optionally filtered by status."""
    agent = _get_agent(request)
    if agent is None:
        return IncidentListResponse()
    raw = await agent.get_incidents(status=status, limit=limit)
    return IncidentListResponse(
        incidents=[IncidentItem(**i) for i in raw["incidents"]],
        total=raw["total"],
    )


@router.get("/predictions", response_model=PredictionListResponse)
async def get_predictions(request: Request):
    """Get recurrence predictions."""
    agent = _get_agent(request)
    if agent is None:
        return PredictionListResponse()
    raw = await agent.get_predictions()
    return PredictionListResponse(
        predictions=[PredictionItem(**p) for p in raw],
    )


@router.post("/scan", response_model=ScanResponse)
async def run_scan(request: Request):
    """Trigger a manual detection scan."""
    agent = _get_agent(request)
    if agent is None:
        return ScanResponse()
    raw = await agent.run_manual_scan()
    return ScanResponse(
        detections=[DetectionItem(**d) for d in raw],
        count=len(raw),
    )


@router.post("/incidents/{incident_id}/acknowledge", response_model=IncidentActionResponse)
async def acknowledge_incident(request: Request, incident_id: str):
    """Acknowledge an incident."""
    agent = _get_agent(request)
    if agent is None:
        return IncidentActionResponse(incident_id=incident_id, status="error")
    result = await agent.acknowledge_incident(incident_id)
    return IncidentActionResponse(**result)


@router.post("/incidents/{incident_id}/resolve", response_model=IncidentActionResponse)
async def resolve_incident(request: Request, incident_id: str):
    """Resolve an incident."""
    agent = _get_agent(request)
    if agent is None:
        return IncidentActionResponse(incident_id=incident_id, status="error")
    result = await agent.resolve_incident(incident_id)
    return IncidentActionResponse(**result)


@router.get("/auto-fixes")
async def get_auto_fixes(request: Request):
    """List all auto-fix actions taken."""
    db = getattr(request.app.state, "db", None)
    if db is None:
        return JSONResponse({"actions": [], "count": 0})

    from sqlalchemy import select

    from src.db.models import AgentAction

    async with db.session() as session:
        result = await session.execute(
            select(AgentAction)
            .where(AgentAction.auto == True)  # noqa: E712
            .order_by(AgentAction.created_at.desc())
            .limit(100)
        )
        actions = result.scalars().all()

    return JSONResponse({
        "actions": [
            {
                "id": a.id,
                "incident_id": a.incident_id,
                "action_type": a.action_type,
                "description": a.description,
                "result": a.result,
                "created_at": a.created_at.isoformat() if a.created_at else None,
            }
            for a in actions
        ],
        "count": len(actions),
    })


@router.get("/customer-health")
async def get_customer_health(request: Request):
    """Get health scores for all active customers."""
    from src.services.ai_agent.health import CustomerHealthScorer

    db = getattr(request.app.state, "db", None)
    scorer = CustomerHealthScorer(db)
    scores = await scorer.score_all()

    return JSONResponse({
        "scores": [
            {
                "customer_id": s.customer_id,
                "total_score": s.total_score,
                "usage": s.usage,
                "satisfaction": s.satisfaction,
                "payment": s.payment,
                "engagement": s.engagement,
                "risk_level": s.risk_level,
                "scored_at": s.scored_at,
            }
            for s in scores
        ],
        "count": len(scores),
    })


@router.get("/customer-health/{customer_id}")
async def get_customer_health_detail(customer_id: str, request: Request):
    """Get detailed health report for one customer."""
    from src.services.ai_agent.health import CustomerHealthScorer

    db = getattr(request.app.state, "db", None)
    scorer = CustomerHealthScorer(db)
    score = await scorer.score(customer_id)

    return JSONResponse({
        "customer_id": score.customer_id,
        "total_score": score.total_score,
        "usage": score.usage,
        "satisfaction": score.satisfaction,
        "payment": score.payment,
        "engagement": score.engagement,
        "risk_level": score.risk_level,
        "scored_at": score.scored_at,
    })


# ── Approval Queue ────────────────────────────────────────────────────────


def _get_approval_queue(request: Request):
    """Retrieve ApprovalQueue from app state."""
    return getattr(request.app.state, "approval_queue", None)


@router.get("/approvals", response_model=ApprovalListResponse)
async def list_approvals(request: Request, limit: int = 50):
    """List pending approval requests."""
    queue = _get_approval_queue(request)
    if queue is None:
        return ApprovalListResponse()
    pending = await queue.list_pending(limit=limit)
    return ApprovalListResponse(
        approvals=[ApprovalItem(**a) for a in pending],
        count=len(pending),
    )


@router.post("/approvals/{approval_id}/approve", response_model=ApprovalActionResponse)
async def approve_action(request: Request, approval_id: str):
    """Approve and execute a pending action."""
    queue = _get_approval_queue(request)
    if queue is None:
        return ApprovalActionResponse(approval_id=approval_id, status="error")
    result = await queue.approve(approval_id, reviewed_by="admin")
    if "error" in result:
        return JSONResponse(status_code=400, content=result)
    return ApprovalActionResponse(**result)


@router.post("/approvals/{approval_id}/reject", response_model=ApprovalActionResponse)
async def reject_action(request: Request, approval_id: str):
    """Reject a pending action."""
    queue = _get_approval_queue(request)
    if queue is None:
        return ApprovalActionResponse(approval_id=approval_id, status="error")
    result = await queue.reject(approval_id, reviewed_by="admin")
    if "error" in result:
        return JSONResponse(status_code=400, content=result)
    return ApprovalActionResponse(
        approval_id=result["approval_id"],
        status=result["status"],
    )
