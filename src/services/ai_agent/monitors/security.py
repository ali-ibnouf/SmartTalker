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


class FailedAuthRule:
    """Detects brute-force login attempts by scanning Redis auth-failure keys."""

    rule_id = "security.failed_auth"

    async def evaluate(self, ctx: AgentContext) -> list[Detection]:
        if ctx.redis is None:
            return []

        detections: list[Detection] = []
        threshold = ctx.agent_config.failed_auth_threshold
        window_min = ctx.agent_config.failed_auth_window_min

        try:
            # Scan for rate-limiter keys tracking failed auth attempts
            cursor = b"0"
            ip_failures: dict[str, int] = {}

            while True:
                cursor, keys = await ctx.redis.scan(
                    cursor=cursor, match="rate:auth_fail:*", count=100
                )
                for key in keys:
                    # Key format: rate:auth_fail:{ip}
                    ip = key.decode().split("rate:auth_fail:")[-1] if isinstance(key, bytes) else key.split("rate:auth_fail:")[-1]
                    ttl = await ctx.redis.ttl(key)
                    # Only count keys within our window
                    if ttl > 0 and ttl <= window_min * 60:
                        count = await ctx.redis.get(key)
                        if count:
                            try:
                                ip_failures[ip] = int(count) if isinstance(count, (bytes, str)) else count
                            except (ValueError, TypeError):
                                continue

                if cursor == 0 or cursor == b"0":
                    break

            for ip, count in ip_failures.items():
                if count >= threshold:
                    severity = "critical" if count >= threshold * 3 else "warning"
                    detections.append(Detection(
                        rule_id=self.rule_id,
                        severity=severity,
                        title=f"Brute-force attempt: {ip} ({count} failures)",
                        description=(
                            f"IP {ip} has {count} failed auth attempts in the "
                            f"last {window_min} minutes (threshold: {threshold})."
                        ),
                        details={
                            "ip": ip,
                            "failed_attempts": count,
                            "window_minutes": window_min,
                        },
                        recommendation=(
                            "Block IP via firewall or rate limiter. "
                            "Check if it's a misconfigured client."
                        ),
                        auto_fixable=True,
                    ))
        except Exception as exc:
            logger.debug(f"FailedAuthRule skipped: {exc}")

        return detections


class APIUsageSpikeRule:
    """Detects abnormal API usage spikes per customer."""

    rule_id = "security.api_spike"

    async def evaluate(self, ctx: AgentContext) -> list[Detection]:
        if ctx.db is None:
            return []

        detections: list[Detection] = []
        multiplier = ctx.agent_config.api_spike_multiplier

        try:
            from sqlalchemy import func, select
            from src.db.models import APICostRecord

            now = datetime.utcnow()
            last_hour = now - timedelta(hours=1)
            week_ago = now - timedelta(days=7)

            async with ctx.db.session_ctx() as session:
                # Last hour request count per customer
                recent_stmt = (
                    select(
                        APICostRecord.customer_id,
                        func.count(APICostRecord.id).label("cnt"),
                    )
                    .where(APICostRecord.created_at >= last_hour)
                    .group_by(APICostRecord.customer_id)
                )
                recent_result = await session.execute(recent_stmt)
                recent_counts = {row[0]: row[1] for row in recent_result.all()}

                if not recent_counts:
                    return []

                # 7-day hourly average per customer
                avg_stmt = (
                    select(
                        APICostRecord.customer_id,
                        (func.count(APICostRecord.id) / (7 * 24.0)).label("hourly_avg"),
                    )
                    .where(APICostRecord.created_at >= week_ago)
                    .where(APICostRecord.created_at < last_hour)
                    .group_by(APICostRecord.customer_id)
                )
                avg_result = await session.execute(avg_stmt)
                avg_counts = {row[0]: row[1] for row in avg_result.all()}

                for cid, count in recent_counts.items():
                    avg = avg_counts.get(cid, 0)
                    if avg < 5:  # Skip low-volume customers
                        continue
                    ratio = count / avg if avg > 0 else 0
                    if ratio >= multiplier:
                        detections.append(Detection(
                            rule_id=self.rule_id,
                            severity="critical" if ratio >= multiplier * 2 else "warning",
                            title=f"API spike: customer {cid} ({ratio:.1f}x normal)",
                            description=(
                                f"Customer {cid} made {count} API calls in the last hour, "
                                f"which is {ratio:.1f}x their 7-day hourly average of "
                                f"{avg:.0f}. Possible abuse or runaway client."
                            ),
                            details={
                                "customer_id": cid,
                                "last_hour_count": count,
                                "hourly_average": round(avg, 1),
                                "ratio": round(ratio, 1),
                            },
                            recommendation=(
                                "Investigate usage source. Consider temporary "
                                "rate-limit throttle for this customer."
                            ),
                            auto_fixable=True,
                        ))
        except Exception as exc:
            logger.debug(f"APIUsageSpikeRule skipped: {exc}")

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
