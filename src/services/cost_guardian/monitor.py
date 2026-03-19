"""Cost Monitor — collects cost data from api_cost_records via SQLAlchemy."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import select, func, and_, distinct

from src.db.models import APICostRecord, Customer, Subscription
from src.utils.logger import setup_logger

logger = setup_logger("cost_guardian.monitor")


class CostMonitor:
    """Collects cost data from the api_cost_records table."""

    def __init__(self, db: Any) -> None:
        self.db = db

    async def get_hourly_spend(self, service: str, hours: int = 1) -> float:
        """Get total spend for a service in the last N hours."""
        if self.db is None:
            return 0.0
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        async with self.db.session() as session:
            result = await session.execute(
                select(func.coalesce(func.sum(APICostRecord.cost_usd), 0.0))
                .where(and_(
                    APICostRecord.service == service,
                    APICostRecord.created_at >= cutoff,
                ))
            )
            return float(result.scalar_one())

    async def get_daily_spend(self, service: str) -> float:
        """Get today's total spend for a service."""
        if self.db is None:
            return 0.0
        today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        async with self.db.session() as session:
            result = await session.execute(
                select(func.coalesce(func.sum(APICostRecord.cost_usd), 0.0))
                .where(and_(
                    APICostRecord.service == service,
                    APICostRecord.created_at >= today,
                ))
            )
            return float(result.scalar_one())

    async def get_monthly_spend(self, service: str) -> float:
        """Get this month's total spend for a service."""
        if self.db is None:
            return 0.0
        month_start = datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        async with self.db.session() as session:
            result = await session.execute(
                select(func.coalesce(func.sum(APICostRecord.cost_usd), 0.0))
                .where(and_(
                    APICostRecord.service == service,
                    APICostRecord.created_at >= month_start,
                ))
            )
            return float(result.scalar_one())

    async def get_total_monthly_spend(self) -> dict[str, dict[str, Any]]:
        """Get full breakdown of this month's spend by service."""
        if self.db is None:
            return {}
        month_start = datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        async with self.db.session() as session:
            result = await session.execute(
                select(
                    APICostRecord.service,
                    func.coalesce(func.sum(APICostRecord.cost_usd), 0.0).label("total"),
                    func.count(APICostRecord.id).label("call_count"),
                )
                .where(APICostRecord.created_at >= month_start)
                .group_by(APICostRecord.service)
            )
            rows = result.all()
        return {
            row.service: {"cost": float(row.total), "calls": int(row.call_count)}
            for row in rows
        }

    async def get_customer_spend(self, customer_id: str) -> float:
        """Get this month's total API spend for a specific customer."""
        if self.db is None:
            return 0.0
        month_start = datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        async with self.db.session() as session:
            result = await session.execute(
                select(func.coalesce(func.sum(APICostRecord.cost_usd), 0.0))
                .where(and_(
                    APICostRecord.customer_id == customer_id,
                    APICostRecord.created_at >= month_start,
                ))
            )
            return float(result.scalar_one())

    async def get_request_rate(self, service: str, minutes: int = 1) -> int:
        """Get number of API calls in last N minutes."""
        if self.db is None:
            return 0
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        async with self.db.session() as session:
            result = await session.execute(
                select(func.count(APICostRecord.id))
                .where(and_(
                    APICostRecord.service == service,
                    APICostRecord.created_at >= cutoff,
                ))
            )
            return int(result.scalar_one())

    async def get_customer_request_rate(self, customer_id: str, minutes: int = 1) -> int:
        """Get number of API calls from a customer in last N minutes."""
        if self.db is None:
            return 0
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        async with self.db.session() as session:
            result = await session.execute(
                select(func.count(APICostRecord.id))
                .where(and_(
                    APICostRecord.customer_id == customer_id,
                    APICostRecord.created_at >= cutoff,
                ))
            )
            return int(result.scalar_one())

    async def get_average_hourly_spend(self, service: str, lookback_hours: int = 24) -> float:
        """Get average hourly spend over the lookback period."""
        if self.db is None:
            return 0.0
        cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
        async with self.db.session() as session:
            result = await session.execute(
                select(func.coalesce(func.sum(APICostRecord.cost_usd), 0.0))
                .where(and_(
                    APICostRecord.service == service,
                    APICostRecord.created_at >= cutoff,
                ))
            )
            total = float(result.scalar_one())
        return total / max(lookback_hours, 1)

    async def get_last_request_cost(self, service: str) -> Optional[float]:
        """Get the cost of the most recent API call for a service."""
        if self.db is None:
            return None
        async with self.db.session() as session:
            result = await session.execute(
                select(APICostRecord.cost_usd)
                .where(APICostRecord.service == service)
                .order_by(APICostRecord.created_at.desc())
                .limit(1)
            )
            row = result.scalar_one_or_none()
        return float(row) if row is not None else None

    async def get_runpod_active_jobs(self) -> list[dict[str, Any]]:
        """Get RunPod jobs that started but logged $0 cost (still running or stuck)."""
        if self.db is None:
            return []
        cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
        async with self.db.session() as session:
            result = await session.execute(
                select(
                    APICostRecord.id,
                    APICostRecord.customer_id,
                    APICostRecord.session_id,
                    APICostRecord.created_at,
                    APICostRecord.details,
                )
                .where(and_(
                    APICostRecord.service.in_(["gpu_render", "gpu_preprocess", "runpod"]),
                    APICostRecord.cost_usd == 0,
                    APICostRecord.created_at >= cutoff,
                ))
            )
            rows = result.all()
        return [
            {
                "id": row.id,
                "customer_id": row.customer_id,
                "session_id": row.session_id,
                "created_at": row.created_at,
                "details": row.details,
            }
            for row in rows
        ]

    async def get_top_spending_customers(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get customers with highest spend this month."""
        if self.db is None:
            return []
        month_start = datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        async with self.db.session() as session:
            result = await session.execute(
                select(
                    APICostRecord.customer_id,
                    Customer.company.label("company_name"),
                    Subscription.plan.label("plan_tier"),
                    func.coalesce(func.sum(APICostRecord.cost_usd), 0.0).label("total_cost"),
                    func.count(APICostRecord.id).label("total_calls"),
                )
                .join(Customer, Customer.id == APICostRecord.customer_id)
                .outerjoin(
                    Subscription,
                    and_(
                        Subscription.customer_id == APICostRecord.customer_id,
                        Subscription.is_active == True,  # noqa: E712
                    ),
                )
                .where(and_(
                    APICostRecord.created_at >= month_start,
                    APICostRecord.customer_id != "",
                ))
                .group_by(
                    APICostRecord.customer_id,
                    Customer.company,
                    Subscription.plan,
                )
                .order_by(func.sum(APICostRecord.cost_usd).desc())
                .limit(limit)
            )
            rows = result.all()
        return [
            {
                "customer_id": row.customer_id,
                "company_name": row.company_name or "",
                "plan_tier": row.plan_tier or "starter",
                "total_cost": float(row.total_cost),
                "total_calls": int(row.total_calls),
            }
            for row in rows
        ]

    async def get_recent_active_customer_ids(self, minutes: int = 5) -> list[str]:
        """Get distinct customer IDs active in the last N minutes."""
        if self.db is None:
            return []
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        async with self.db.session() as session:
            result = await session.execute(
                select(distinct(APICostRecord.customer_id))
                .where(and_(
                    APICostRecord.created_at >= cutoff,
                    APICostRecord.customer_id != "",
                ))
            )
            return [row[0] for row in result.all()]

    async def get_zero_cost_counts(self) -> list[dict[str, Any]]:
        """Get services with many $0-cost calls in the last hour (billing broken?)."""
        if self.db is None:
            return []
        cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
        async with self.db.session() as session:
            result = await session.execute(
                select(
                    APICostRecord.service,
                    func.count(APICostRecord.id).label("zero_count"),
                )
                .where(and_(
                    APICostRecord.cost_usd == 0,
                    APICostRecord.service.notin_(["voice_clone"]),
                    APICostRecord.created_at >= cutoff,
                ))
                .group_by(APICostRecord.service)
                .having(func.count(APICostRecord.id) > 5)
            )
            rows = result.all()
        return [
            {"service": row.service, "zero_count": int(row.zero_count)}
            for row in rows
        ]
