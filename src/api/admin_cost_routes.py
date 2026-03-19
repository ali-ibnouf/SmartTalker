"""Admin cost tracking API routes.

Provides endpoints for platform-level API cost analysis:
- Total costs across all services
- Breakdown by service (ASR, LLM, TTS, RunPod)
- Costs grouped by customer
- Revenue vs cost margin
- RunPod-specific metrics

All endpoints require admin API key auth (via middleware).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Request, Query

from sqlalchemy import select, func, case, and_

from src.api.schemas import (
    CostTotalResponse,
    CostBreakdownItem,
    CostBreakdownResponse,
    CostByCustomerItem,
    CostByCustomerResponse,
    CostMarginResponse,
    RunPodCostResponse,
)
from src.db.models import APICostRecord, Customer
from src.utils.logger import setup_logger

logger = setup_logger("api.admin_costs")

router = APIRouter(prefix="/api/v1/admin/costs", tags=["admin-costs"])


def _period_bounds(days: int) -> tuple[datetime, datetime]:
    """Return (start, end) datetime bounds for the given period."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    return start, end


@router.get("/total", response_model=CostTotalResponse)
async def get_total_costs(
    request: Request,
    days: int = Query(30, ge=1, le=365),
):
    """Get total platform API costs for the given period."""
    db = getattr(request.app.state, "db", None)
    if db is None:
        return CostTotalResponse()

    start, end = _period_bounds(days)

    async with db.session() as session:
        result = await session.execute(
            select(
                func.coalesce(func.sum(APICostRecord.cost_usd), 0.0).label("total"),
                func.count(APICostRecord.id).label("cnt"),
            ).where(APICostRecord.created_at >= start)
        )
        row = result.one()

    return CostTotalResponse(
        total_cost_usd=float(row.total),
        period_start=start.isoformat(),
        period_end=end.isoformat(),
        record_count=int(row.cnt),
    )


@router.get("/breakdown", response_model=CostBreakdownResponse)
async def get_cost_breakdown(
    request: Request,
    days: int = Query(30, ge=1, le=365),
):
    """Get costs broken down by service (asr, llm, tts, runpod)."""
    db = getattr(request.app.state, "db", None)
    if db is None:
        return CostBreakdownResponse()

    start, _ = _period_bounds(days)

    async with db.session() as session:
        result = await session.execute(
            select(
                APICostRecord.service,
                func.sum(APICostRecord.cost_usd).label("total"),
                func.count(APICostRecord.id).label("cnt"),
                func.avg(APICostRecord.cost_usd).label("avg"),
            )
            .where(APICostRecord.created_at >= start)
            .group_by(APICostRecord.service)
            .order_by(func.sum(APICostRecord.cost_usd).desc())
        )
        rows = result.all()

    items = [
        CostBreakdownItem(
            service=row.service,
            total_cost_usd=float(row.total or 0),
            record_count=int(row.cnt),
            avg_cost_usd=float(row.avg or 0),
        )
        for row in rows
    ]
    grand_total = sum(i.total_cost_usd for i in items)

    return CostBreakdownResponse(breakdown=items, total_cost_usd=grand_total)


@router.get("/by-customer", response_model=CostByCustomerResponse)
async def get_costs_by_customer(
    request: Request,
    days: int = Query(30, ge=1, le=365),
):
    """Get costs grouped by customer."""
    db = getattr(request.app.state, "db", None)
    if db is None:
        return CostByCustomerResponse()

    start, _ = _period_bounds(days)

    async with db.session() as session:
        result = await session.execute(
            select(
                APICostRecord.customer_id,
                func.sum(APICostRecord.cost_usd).label("total"),
                func.count(APICostRecord.id).label("cnt"),
            )
            .where(
                and_(
                    APICostRecord.created_at >= start,
                    APICostRecord.customer_id != "",
                )
            )
            .group_by(APICostRecord.customer_id)
            .order_by(func.sum(APICostRecord.cost_usd).desc())
        )
        rows = result.all()

        # Resolve customer names
        cust_ids = [r.customer_id for r in rows]
        names: dict[str, str] = {}
        if cust_ids:
            name_result = await session.execute(
                select(Customer.id, Customer.name).where(Customer.id.in_(cust_ids))
            )
            names = {r.id: r.name for r in name_result.all()}

    items = [
        CostByCustomerItem(
            customer_id=row.customer_id,
            customer_name=names.get(row.customer_id, ""),
            total_cost_usd=float(row.total or 0),
            record_count=int(row.cnt),
        )
        for row in rows
    ]
    grand_total = sum(i.total_cost_usd for i in items)

    return CostByCustomerResponse(customers=items, total_cost_usd=grand_total)


@router.get("/margin", response_model=CostMarginResponse)
async def get_cost_margin(
    request: Request,
    days: int = Query(30, ge=1, le=365),
):
    """Get revenue vs. cost margin for the period."""
    db = getattr(request.app.state, "db", None)
    billing = getattr(request.app.state, "billing", None)
    if db is None:
        return CostMarginResponse()

    start, end = _period_bounds(days)

    # Get total costs
    async with db.session() as session:
        cost_result = await session.execute(
            select(
                func.coalesce(func.sum(APICostRecord.cost_usd), 0.0).label("total"),
            ).where(APICostRecord.created_at >= start)
        )
        total_cost = float(cost_result.scalar_one())

    # Get revenue from billing engine
    total_revenue = 0.0
    if billing:
        try:
            total_revenue = await billing.get_period_revenue(
                start=start, end=end
            )
        except Exception:
            pass

    margin = total_revenue - total_cost
    margin_pct = (margin / total_revenue * 100) if total_revenue > 0 else 0.0

    return CostMarginResponse(
        total_revenue_usd=total_revenue,
        total_cost_usd=total_cost,
        margin_usd=margin,
        margin_percent=round(margin_pct, 2),
        period_start=start.isoformat(),
        period_end=end.isoformat(),
    )


@router.get("/runpod", response_model=RunPodCostResponse)
async def get_runpod_costs(
    request: Request,
    days: int = Query(30, ge=1, le=365),
):
    """Get RunPod-specific cost and job metrics."""
    db = getattr(request.app.state, "db", None)
    if db is None:
        return RunPodCostResponse()

    start, _ = _period_bounds(days)

    async with db.session() as session:
        result = await session.execute(
            select(
                func.coalesce(func.sum(APICostRecord.cost_usd), 0.0).label("total"),
                func.count(APICostRecord.id).label("jobs"),
                func.coalesce(func.avg(APICostRecord.duration_ms), 0).label("avg_ms"),
                func.coalesce(func.sum(APICostRecord.duration_ms), 0).label("total_ms"),
                func.sum(
                    case(
                        (APICostRecord.details.like("%preprocess%"), 1),
                        else_=0,
                    )
                ).label("preprocess_cnt"),
                func.sum(
                    case(
                        (APICostRecord.details.like("%render%"), 1),
                        else_=0,
                    )
                ).label("render_cnt"),
            ).where(
                and_(
                    APICostRecord.service == "runpod",
                    APICostRecord.created_at >= start,
                )
            )
        )
        row = result.one()

    return RunPodCostResponse(
        total_cost_usd=float(row.total),
        total_jobs=int(row.jobs),
        avg_execution_ms=int(row.avg_ms),
        total_execution_ms=int(row.total_ms),
        preprocess_jobs=int(row.preprocess_cnt or 0),
        render_jobs=int(row.render_cnt or 0),
    )
