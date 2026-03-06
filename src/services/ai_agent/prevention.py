"""Pattern tracker for issue recurrence prediction."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from src.utils.logger import setup_logger

logger = setup_logger("ai_agent.prevention")


class PatternTracker:
    """Tracks recurring detections and predicts future occurrences.

    Uses a simple heuristic: if an issue occurred N times with average
    interval T between occurrences, predict next at last_seen + T.
    """

    def __init__(self, db: Any) -> None:
        self._db = db

    async def record_occurrence(self, rule_id: str, pattern_key: str) -> None:
        """Record that a detection occurred. Upserts agent_patterns row."""
        if self._db is None:
            return

        try:
            from sqlalchemy import select
            from src.db.models import AgentPattern

            now = datetime.utcnow()

            async with self._db.session_ctx() as session:
                stmt = select(AgentPattern).where(
                    AgentPattern.rule_id == rule_id,
                    AgentPattern.pattern_key == pattern_key,
                )
                result = await session.execute(stmt)
                existing = result.scalar_one_or_none()

                if existing:
                    # Update interval prediction
                    interval = now - existing.last_seen
                    count = existing.occurrence_count + 1
                    # Weighted average interval: mix old avg with new observation
                    if count > 2:
                        old_avg_s = (
                            (existing.last_seen - existing.first_seen).total_seconds()
                            / (existing.occurrence_count - 1)
                        ) if existing.occurrence_count > 1 else interval.total_seconds()
                        avg_interval_s = (old_avg_s * 0.7) + (interval.total_seconds() * 0.3)
                    else:
                        avg_interval_s = interval.total_seconds()

                    existing.occurrence_count = count
                    existing.last_seen = now
                    if avg_interval_s > 0:
                        existing.predicted_next = now + timedelta(seconds=avg_interval_s)
                else:
                    pattern = AgentPattern(
                        rule_id=rule_id,
                        pattern_key=pattern_key,
                        occurrence_count=1,
                        first_seen=now,
                        last_seen=now,
                        predicted_next=None,
                    )
                    session.add(pattern)
        except Exception as exc:
            logger.error(f"Failed to record pattern: {exc}")

    async def get_predictions(self) -> list[dict[str, Any]]:
        """Return patterns with predictions for future recurrence."""
        if self._db is None:
            return []

        try:
            from sqlalchemy import select
            from src.db.models import AgentPattern

            async with self._db.session_ctx() as session:
                stmt = (
                    select(AgentPattern)
                    .where(AgentPattern.predicted_next.isnot(None))
                    .where(AgentPattern.occurrence_count >= 2)
                    .order_by(AgentPattern.predicted_next.asc())
                    .limit(50)
                )
                result = await session.execute(stmt)
                patterns = result.scalars().all()
                return [
                    {
                        "rule_id": p.rule_id,
                        "pattern_key": p.pattern_key,
                        "occurrences": p.occurrence_count,
                        "first_seen": p.first_seen.isoformat() if p.first_seen else None,
                        "last_seen": p.last_seen.isoformat() if p.last_seen else None,
                        "predicted_next": p.predicted_next.isoformat() if p.predicted_next else None,
                    }
                    for p in patterns
                ]
        except Exception as exc:
            logger.error(f"Failed to get predictions: {exc}")
            return []

    async def analyze_trends(self) -> list[dict[str, Any]]:
        """Analyze incident patterns and return actionable trend insights.

        Identifies chronic issues (3+ occurrences) and whether they're
        accelerating (getting more frequent) or decelerating.
        """
        if self._db is None:
            return []

        try:
            from sqlalchemy import select
            from src.db.models import AgentPattern

            async with self._db.session_ctx() as session:
                stmt = (
                    select(AgentPattern)
                    .where(AgentPattern.occurrence_count >= 3)
                    .order_by(AgentPattern.occurrence_count.desc())
                    .limit(20)
                )
                result = await session.execute(stmt)
                patterns = result.scalars().all()

            trends: list[dict[str, Any]] = []
            for p in patterns:
                if not p.first_seen or not p.last_seen:
                    continue

                span_s = (p.last_seen - p.first_seen).total_seconds()
                if span_s <= 0 or p.occurrence_count < 2:
                    continue

                avg_interval_h = (span_s / (p.occurrence_count - 1)) / 3600

                # Determine trend direction:
                # If predicted_next is sooner than avg interval from last_seen,
                # the issue is accelerating.
                trend = "stable"
                if p.predicted_next and p.last_seen:
                    next_interval_h = (p.predicted_next - p.last_seen).total_seconds() / 3600
                    if next_interval_h < avg_interval_h * 0.8:
                        trend = "accelerating"
                    elif next_interval_h > avg_interval_h * 1.2:
                        trend = "decelerating"

                insight = (
                    f"{p.rule_id} has occurred {p.occurrence_count} times, "
                    f"avg every {avg_interval_h:.1f}h. Trend: {trend}."
                )

                trends.append({
                    "rule_id": p.rule_id,
                    "pattern_key": p.pattern_key,
                    "occurrences": p.occurrence_count,
                    "avg_interval_hours": round(avg_interval_h, 1),
                    "trend": trend,
                    "insight": insight,
                })

            return trends

        except Exception as exc:
            logger.error(f"Failed to analyze trends: {exc}")
            return []

    async def predict_quota_exhaustion(self, customer_id: str) -> dict[str, Any] | None:
        """Predict days until quota runs out based on recent usage trend.

        Returns dict with days_remaining, daily_burn_rate, and confidence,
        or None if not enough data.
        """
        if self._db is None:
            return None

        try:
            from sqlalchemy import func, select
            from src.db.models import Subscription, UsageRecord

            now = datetime.utcnow()
            week_ago = now - timedelta(days=7)

            async with self._db.session_ctx() as session:
                # Get current subscription quota
                sub_r = await session.execute(
                    select(Subscription.monthly_seconds, Subscription.plan)
                    .where(
                        Subscription.customer_id == customer_id,
                        Subscription.is_active == True,  # noqa: E712
                    )
                )
                sub_row = sub_r.first()
                if not sub_row or sub_row[0] == 0:
                    return None

                monthly_limit = sub_row[0]

                # Usage in current month
                month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                used_r = await session.execute(
                    select(func.coalesce(func.sum(UsageRecord.duration_s), 0))
                    .where(
                        UsageRecord.customer_id == customer_id,
                        UsageRecord.started_at >= month_start,
                    )
                )
                used_s = used_r.scalar() or 0

                # Daily burn rate over last 7 days
                daily_r = await session.execute(
                    select(func.coalesce(func.sum(UsageRecord.duration_s), 0))
                    .where(
                        UsageRecord.customer_id == customer_id,
                        UsageRecord.started_at >= week_ago,
                    )
                )
                week_total = daily_r.scalar() or 0
                daily_burn = week_total / 7

            remaining = max(0, monthly_limit - used_s)

            if daily_burn <= 0:
                return {
                    "customer_id": customer_id,
                    "days_remaining": None,
                    "daily_burn_rate_s": 0,
                    "used_s": round(used_s),
                    "remaining_s": round(remaining),
                    "confidence": "low",
                }

            days_remaining = remaining / daily_burn

            # Confidence based on data consistency
            confidence = "high" if week_total > 3600 else "medium" if week_total > 600 else "low"

            return {
                "customer_id": customer_id,
                "days_remaining": round(days_remaining, 1),
                "daily_burn_rate_s": round(daily_burn),
                "used_s": round(used_s),
                "remaining_s": round(remaining),
                "monthly_limit_s": monthly_limit,
                "confidence": confidence,
            }

        except Exception as exc:
            logger.error(f"Quota prediction failed for {customer_id}: {exc}")
            return None

    async def get_customer_churn_probability(self, customer_id: str) -> float:
        """Calculate 0-1 churn probability based on multiple signals.

        Weighted formula:
        - usage_trend_declining * 0.30
        - inactive_days / 30 * 0.25
        - training_progress_stalled * 0.20
        - payment_failures * 0.15
        - low_engagement * 0.10
        """
        if self._db is None:
            return 0.0

        try:
            from sqlalchemy import func, select
            from src.db.models import Avatar, Conversation, Customer, Subscription, UsageRecord

            now = datetime.utcnow()
            score = 0.0

            async with self._db.session_ctx() as session:
                # 1. Usage trend declining (30% weight)
                week_ago = now - timedelta(days=7)
                two_weeks_ago = now - timedelta(days=14)
                this_week_r = await session.execute(
                    select(func.coalesce(func.sum(UsageRecord.duration_s), 0))
                    .where(
                        UsageRecord.customer_id == customer_id,
                        UsageRecord.started_at >= week_ago,
                    )
                )
                last_week_r = await session.execute(
                    select(func.coalesce(func.sum(UsageRecord.duration_s), 0))
                    .where(
                        UsageRecord.customer_id == customer_id,
                        UsageRecord.started_at >= two_weeks_ago,
                        UsageRecord.started_at < week_ago,
                    )
                )
                this_week = this_week_r.scalar() or 0
                last_week = last_week_r.scalar() or 0
                if last_week > 0 and this_week < last_week * 0.5:
                    score += 0.30  # Usage dropped by >50%
                elif last_week > 0 and this_week < last_week * 0.8:
                    score += 0.15  # Moderate decline

                # 2. Inactive days (25% weight)
                last_convo_r = await session.execute(
                    select(func.max(Conversation.started_at))
                    .where(Conversation.customer_id == customer_id)
                )
                last_convo = last_convo_r.scalar()
                if last_convo:
                    inactive_days = (now - last_convo).days
                    score += min(0.25, (inactive_days / 30) * 0.25)
                else:
                    score += 0.25  # No conversations at all

                # 3. Training progress stalled (20% weight)
                stall_cutoff = now - timedelta(days=7)
                stalled_r = await session.execute(
                    select(func.count(Avatar.id))
                    .where(
                        Avatar.customer_id == customer_id,
                        Avatar.training_progress < 0.5,
                        Avatar.training_progress > 0.0,
                        Avatar.updated_at <= stall_cutoff,
                    )
                )
                stalled = stalled_r.scalar() or 0
                if stalled > 0:
                    score += 0.20

                # 4. Payment failures (15% weight)
                pay_r = await session.execute(
                    select(func.coalesce(func.max(Subscription.payment_failures), 0))
                    .where(
                        Subscription.customer_id == customer_id,
                        Subscription.is_active == True,  # noqa: E712
                    )
                )
                failures = pay_r.scalar() or 0
                if failures >= 3:
                    score += 0.15
                elif failures >= 1:
                    score += 0.07

                # 5. Low engagement — no conversations in 48h (10% weight)
                recent_cutoff = now - timedelta(hours=48)
                recent_r = await session.execute(
                    select(func.count(Conversation.id))
                    .where(
                        Conversation.customer_id == customer_id,
                        Conversation.started_at >= recent_cutoff,
                    )
                )
                recent = recent_r.scalar() or 0
                if recent == 0:
                    score += 0.10

            return min(1.0, max(0.0, score))

        except Exception as exc:
            logger.error(f"Churn prediction failed for {customer_id}: {exc}")
            return 0.0
