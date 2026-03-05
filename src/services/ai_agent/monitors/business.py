"""Business intelligence monitoring rules."""

from __future__ import annotations

from datetime import datetime, timedelta

from src.services.ai_agent.rules import AgentContext, Detection
from src.utils.logger import setup_logger

logger = setup_logger("ai_agent.monitors.business")


class ChurnRiskRule:
    """Detects customers at risk of churning based on activity drop."""

    rule_id = "business.churn_risk"

    async def evaluate(self, ctx: AgentContext) -> list[Detection]:
        if ctx.db is None:
            return []

        detections: list[Detection] = []
        days = ctx.agent_config.churn_days_inactive
        cutoff = datetime.utcnow() - timedelta(days=days)

        try:
            from sqlalchemy import func, select
            from src.db.models import Avatar, Conversation, Customer

            async with ctx.db.session_ctx() as session:
                # Find customers with no conversations in the last N days
                recent_active = (
                    select(Conversation.avatar_id)
                    .where(Conversation.started_at >= cutoff)
                    .distinct()
                    .subquery()
                )
                stmt = (
                    select(Customer, Avatar)
                    .join(Avatar, Avatar.customer_id == Customer.id)
                    .where(Customer.is_active == True)
                    .where(Customer.suspended == False)
                    .where(Avatar.id.notin_(select(recent_active.c.avatar_id)))
                    .where(Avatar.training_progress < 0.5)
                )
                result = await session.execute(stmt)
                rows = result.all()

                for customer, avatar in rows:
                    detections.append(Detection(
                        rule_id=self.rule_id,
                        severity="warning",
                        title=f'Churn risk: "{customer.name}"',
                        description=(
                            f"No conversations in {days} days. "
                            f"Training at {avatar.training_progress * 100:.0f}%. "
                            f"Customer may be frustrated with slow progress."
                        ),
                        details={
                            "customer_id": customer.id,
                            "customer_name": customer.name,
                            "avatar_id": avatar.id,
                            "training_progress": avatar.training_progress,
                            "inactive_days": days,
                        },
                        recommendation=(
                            "Offer onboarding support, extend trial, or assign "
                            "a Job Persona to jumpstart training."
                        ),
                    ))
        except Exception as exc:
            logger.error(f"ChurnRiskRule failed: {exc}")

        return detections


class QuotaExhaustionRule:
    """Detects customers nearing their monthly usage quota."""

    rule_id = "business.quota_exhaustion"

    async def evaluate(self, ctx: AgentContext) -> list[Detection]:
        if ctx.db is None:
            return []

        detections: list[Detection] = []
        warn_pct = ctx.agent_config.quota_warn_pct

        try:
            from sqlalchemy import func, select
            from src.db.models import Customer, Subscription, UsageRecord

            now = datetime.utcnow()
            month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

            async with ctx.db.session_ctx() as session:
                # Get active subscriptions with monthly usage
                stmt = (
                    select(
                        Customer.id,
                        Customer.name,
                        Subscription.plan,
                        Subscription.monthly_seconds,
                        func.coalesce(func.sum(UsageRecord.duration_s), 0).label("used_s"),
                    )
                    .join(Subscription, Subscription.customer_id == Customer.id)
                    .outerjoin(
                        UsageRecord,
                        (UsageRecord.customer_id == Customer.id)
                        & (UsageRecord.started_at >= month_start),
                    )
                    .where(Customer.is_active == True)
                    .where(Subscription.is_active == True)
                    .group_by(Customer.id, Customer.name, Subscription.plan, Subscription.monthly_seconds)
                )
                result = await session.execute(stmt)
                for row in result.all():
                    cid, cname, plan, limit_s, used_s = row
                    if limit_s == 0:
                        continue
                    usage_pct = (used_s / limit_s) * 100
                    if usage_pct >= warn_pct:
                        severity = "critical" if usage_pct >= 100 else "warning"
                        detections.append(Detection(
                            rule_id=self.rule_id,
                            severity=severity,
                            title=f"Quota {usage_pct:.0f}%: {cname}",
                            description=(
                                f"{cname} ({plan} plan) has used {used_s:.0f}s of "
                                f"{limit_s}s quota ({usage_pct:.0f}%). "
                                f"{'Quota exceeded!' if usage_pct >= 100 else 'Nearing limit.'}"
                            ),
                            details={
                                "customer_id": cid,
                                "plan": plan,
                                "used_s": round(used_s),
                                "limit_s": limit_s,
                                "usage_pct": round(usage_pct, 1),
                            },
                            recommendation=(
                                "Send upgrade offer for next tier, or warn customer about usage."
                            ),
                            auto_fixable=True,
                        ))
        except Exception as exc:
            logger.error(f"QuotaExhaustionRule failed: {exc}")

        return detections


class EscalationSpikeRule:
    """Detects avatars with abnormally high escalation rates."""

    rule_id = "business.escalation_spike"

    async def evaluate(self, ctx: AgentContext) -> list[Detection]:
        if ctx.db is None:
            return []

        detections: list[Detection] = []
        threshold = ctx.agent_config.escalation_rate_warn

        try:
            from sqlalchemy import Integer, func, select
            from src.db.models import Avatar, Conversation, ConversationMessage, Customer

            cutoff = datetime.utcnow() - timedelta(days=7)

            async with ctx.db.session_ctx() as session:
                # Per avatar: total messages vs escalated messages in last 7 days
                stmt = (
                    select(
                        Avatar.id,
                        Avatar.name,
                        Customer.name.label("customer_name"),
                        func.count(ConversationMessage.id).label("total_msgs"),
                        func.sum(
                            func.cast(ConversationMessage.escalated, Integer)
                        ).label("escalated_msgs"),
                    )
                    .join(Conversation, Conversation.avatar_id == Avatar.id)
                    .join(ConversationMessage, ConversationMessage.conversation_id == Conversation.id)
                    .join(Customer, Customer.id == Avatar.customer_id)
                    .where(ConversationMessage.created_at >= cutoff)
                    .group_by(Avatar.id, Avatar.name, Customer.name)
                    .having(func.count(ConversationMessage.id) >= 10)  # min sample
                )
                result = await session.execute(stmt)
                for row in result.all():
                    aid, aname, cname, total, escalated = row
                    esc_count = escalated or 0
                    rate = esc_count / total if total > 0 else 0
                    if rate >= threshold:
                        detections.append(Detection(
                            rule_id=self.rule_id,
                            severity="warning",
                            title=f"High escalation: {aname} ({rate*100:.0f}%)",
                            description=(
                                f"Avatar '{aname}' for {cname} has {rate*100:.0f}% "
                                f"escalation rate ({esc_count}/{total} messages). "
                                f"Average is ~8%."
                            ),
                            details={
                                "avatar_id": aid,
                                "avatar_name": aname,
                                "customer_name": cname,
                                "escalation_rate": round(rate, 3),
                                "escalated_count": esc_count,
                                "total_messages": total,
                            },
                            recommendation=(
                                "Review KB coverage for this avatar. Upload missing documents "
                                "to reduce escalations."
                            ),
                        ))
        except Exception as exc:
            logger.error(f"EscalationSpikeRule failed: {exc}")

        return detections
