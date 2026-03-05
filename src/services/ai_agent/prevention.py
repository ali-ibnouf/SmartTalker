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
