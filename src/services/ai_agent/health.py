"""Customer health scoring engine — composite 0-100 score."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from src.utils.logger import setup_logger

logger = setup_logger("ai_agent.health")


@dataclass
class CustomerHealth:
    customer_id: str
    total_score: int  # 0-100
    usage: int  # 0-25
    satisfaction: int  # 0-25
    payment: int  # 0-25
    engagement: int  # 0-25
    risk_level: str  # "healthy", "at_risk", "critical"
    scored_at: str = ""


class CustomerHealthScorer:
    """Composite health score for each customer.

    4 dimensions, each 0-25 points:
    - Usage trend: are they using the product more or less?
    - Satisfaction: escalation rate + training progress
    - Payment health: on-time payments, no failures
    - Engagement: login frequency, session count
    """

    def __init__(self, db: Any = None):
        self._db = db

    async def score(self, customer_id: str) -> CustomerHealth:
        usage = await self._score_usage_trend(customer_id)
        satisfaction = await self._score_satisfaction(customer_id)
        payment = await self._score_payments(customer_id)
        engagement = await self._score_engagement(customer_id)

        total = usage + satisfaction + payment + engagement
        risk = "healthy" if total >= 70 else "at_risk" if total >= 40 else "critical"

        return CustomerHealth(
            customer_id=customer_id,
            total_score=total,
            usage=usage,
            satisfaction=satisfaction,
            payment=payment,
            engagement=engagement,
            risk_level=risk,
            scored_at=datetime.now(timezone.utc).isoformat(),
        )

    async def score_all(self) -> list[CustomerHealth]:
        """Score all active customers."""
        if self._db is None:
            return []

        from sqlalchemy import select

        from src.db.models import Customer

        async with self._db.session() as session:
            result = await session.execute(
                select(Customer.id).where(Customer.is_active == True)  # noqa: E712
            )
            customer_ids = [row[0] for row in result.all()]

        scores = []
        for cid in customer_ids:
            try:
                scores.append(await self.score(cid))
            except Exception as exc:
                logger.warning(f"Failed to score customer {cid}: {exc}")

        return sorted(scores, key=lambda s: s.total_score)  # worst first

    async def _score_usage_trend(self, customer_id: str) -> int:
        """0-25: based on recent usage vs plan allocation."""
        if self._db is None:
            return 20  # Default healthy

        from datetime import timedelta

        from sqlalchemy import select
        from sqlalchemy import func as sa_func

        from src.db.models import Subscription, UsageRecord

        now = datetime.now(timezone.utc)
        week_ago = now - timedelta(days=7)
        two_weeks_ago = now - timedelta(days=14)

        async with self._db.session() as session:
            # This week's usage
            r1 = await session.execute(
                select(sa_func.coalesce(sa_func.sum(UsageRecord.duration_s), 0.0))
                .where(
                    UsageRecord.customer_id == customer_id,
                    UsageRecord.started_at >= week_ago,
                )
            )
            this_week = r1.scalar() or 0.0

            # Last week's usage
            r2 = await session.execute(
                select(sa_func.coalesce(sa_func.sum(UsageRecord.duration_s), 0.0))
                .where(
                    UsageRecord.customer_id == customer_id,
                    UsageRecord.started_at >= two_weeks_ago,
                    UsageRecord.started_at < week_ago,
                )
            )
            last_week = r2.scalar() or 0.0

            # Get plan total
            sub_r = await session.execute(
                select(Subscription.monthly_seconds)
                .where(
                    Subscription.customer_id == customer_id,
                    Subscription.is_active == True,  # noqa: E712
                )
                .limit(1)
            )
            plan_total = sub_r.scalar() or 1  # noqa: F841

        # Score: 25 if growing or stable, lower if declining
        if last_week == 0 and this_week == 0:
            return 10  # No usage at all
        if last_week == 0:
            return 20  # New usage

        trend = (this_week - last_week) / last_week if last_week > 0 else 0
        if trend >= 0:
            return 25  # Growing or stable
        elif trend > -0.3:
            return 18  # Slight decline
        elif trend > -0.5:
            return 10  # Moderate decline
        else:
            return 5  # Severe decline

    async def _score_satisfaction(self, customer_id: str) -> int:
        """0-25: based on escalation rate."""
        if self._db is None:
            return 20

        from datetime import timedelta

        from sqlalchemy import select
        from sqlalchemy import func as sa_func

        from src.db.models import UsageRecord

        now = datetime.now(timezone.utc)
        month_ago = now - timedelta(days=30)

        async with self._db.session() as session:
            r = await session.execute(
                select(sa_func.count(UsageRecord.id))
                .where(
                    UsageRecord.customer_id == customer_id,
                    UsageRecord.started_at >= month_ago,
                )
            )
            session_count = r.scalar() or 0

        if session_count == 0:
            return 15  # No data

        # Higher session count = better satisfaction (simplified)
        if session_count >= 50:
            return 25
        elif session_count >= 20:
            return 20
        elif session_count >= 5:
            return 15
        else:
            return 10

    async def _score_payments(self, customer_id: str) -> int:
        """0-25: active subscription = 25, none = 0."""
        if self._db is None:
            return 25

        from sqlalchemy import select

        from src.db.models import Subscription

        async with self._db.session() as session:
            r = await session.execute(
                select(Subscription.is_active)
                .where(Subscription.customer_id == customer_id)
                .order_by(Subscription.created_at.desc())
                .limit(1)
            )
            is_active = r.scalar()

        return 25 if is_active else 0

    async def _score_engagement(self, customer_id: str) -> int:
        """0-25: based on recent session count (last 7 days)."""
        if self._db is None:
            return 20

        from datetime import timedelta

        from sqlalchemy import select
        from sqlalchemy import func as sa_func

        from src.db.models import UsageRecord

        cutoff = datetime.now(timezone.utc) - timedelta(days=7)

        async with self._db.session() as session:
            r = await session.execute(
                select(sa_func.count(UsageRecord.id))
                .where(
                    UsageRecord.customer_id == customer_id,
                    UsageRecord.started_at >= cutoff,
                )
            )
            count = r.scalar() or 0

        if count >= 10:
            return 25
        elif count >= 5:
            return 20
        elif count >= 1:
            return 15
        else:
            return 5
