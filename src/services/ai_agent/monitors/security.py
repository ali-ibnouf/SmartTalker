"""Security monitoring rules."""

from __future__ import annotations

from datetime import datetime, timedelta

from src.services.ai_agent.rules import AgentContext, Detection
from src.utils.logger import setup_logger

logger = setup_logger("ai_agent.monitors.security")


class PolicyViolationSpikeRule:
    """Detects spikes in content policy violations."""

    rule_id = "security.violation_spike"

    async def evaluate(self, ctx: AgentContext) -> list[Detection]:
        if ctx.db is None:
            return []

        detections: list[Detection] = []
        threshold = ctx.agent_config.violation_spike_24h

        try:
            from sqlalchemy import func, select
            from src.db.models import PolicyViolation

            cutoff = datetime.utcnow() - timedelta(hours=24)

            async with ctx.db.session_ctx() as session:
                stmt = (
                    select(func.count(PolicyViolation.id))
                    .where(PolicyViolation.created_at >= cutoff)
                )
                result = await session.execute(stmt)
                count = result.scalar() or 0

                if count >= threshold:
                    # Get breakdown by type
                    type_stmt = (
                        select(
                            PolicyViolation.violation_type,
                            func.count(PolicyViolation.id).label("cnt"),
                        )
                        .where(PolicyViolation.created_at >= cutoff)
                        .group_by(PolicyViolation.violation_type)
                        .order_by(func.count(PolicyViolation.id).desc())
                    )
                    type_result = await session.execute(type_stmt)
                    breakdown = {row[0]: row[1] for row in type_result.all()}

                    detections.append(Detection(
                        rule_id=self.rule_id,
                        severity="critical" if count >= threshold * 2 else "warning",
                        title=f"Policy violations spike: {count} in 24h",
                        description=(
                            f"{count} guardrail violations in the last 24 hours "
                            f"(threshold: {threshold}). Breakdown: "
                            + ", ".join(f"{t}: {c}" for t, c in breakdown.items())
                        ),
                        details={
                            "total_violations": count,
                            "threshold": threshold,
                            "breakdown": breakdown,
                        },
                        recommendation=(
                            "Review guardrail policies. Check if LLM is generating "
                            "inappropriate content or if policies need updating."
                        ),
                    ))
        except Exception as exc:
            logger.error(f"PolicyViolationSpikeRule failed: {exc}")

        return detections


class SuspiciousActivityRule:
    """Detects abnormal session creation patterns (potential abuse)."""

    rule_id = "security.suspicious_activity"

    async def evaluate(self, ctx: AgentContext) -> list[Detection]:
        if ctx.db is None:
            return []

        detections: list[Detection] = []
        threshold = ctx.agent_config.rapid_session_threshold

        try:
            from sqlalchemy import func, select
            from src.db.models import Conversation

            # Check for unusually high session creation in the last 5 minutes
            cutoff = datetime.utcnow() - timedelta(minutes=5)

            async with ctx.db.session_ctx() as session:
                stmt = (
                    select(
                        Conversation.channel,
                        func.count(Conversation.id).label("cnt"),
                    )
                    .where(Conversation.started_at >= cutoff)
                    .group_by(Conversation.channel)
                )
                result = await session.execute(stmt)
                for row in result.all():
                    channel, count = row
                    # Normalize to per-minute rate
                    rate = count / 5
                    if rate >= threshold:
                        detections.append(Detection(
                            rule_id=self.rule_id,
                            severity="critical",
                            title=f"Suspicious activity: {rate:.0f} sessions/min ({channel})",
                            description=(
                                f"Channel '{channel}' created {count} sessions in 5 minutes "
                                f"({rate:.0f}/min). Threshold: {threshold}/min. "
                                f"Possible abuse or misconfigured client."
                            ),
                            details={
                                "channel": channel,
                                "sessions_5min": count,
                                "rate_per_min": round(rate, 1),
                                "threshold": threshold,
                            },
                            recommendation=(
                                "Investigate source. Consider rate-limiting the channel "
                                "or blocking the caller."
                            ),
                        ))
        except Exception as exc:
            logger.error(f"SuspiciousActivityRule failed: {exc}")

        return detections
