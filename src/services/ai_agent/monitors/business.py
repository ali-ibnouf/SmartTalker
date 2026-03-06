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


class FailedPaymentRule:
    """Detects customers with multiple failed payment attempts."""

    rule_id = "business.failed_payment"

    async def evaluate(self, ctx: AgentContext) -> list[Detection]:
        if ctx.db is None:
            return []

        detections: list[Detection] = []
        threshold = ctx.agent_config.failed_payment_threshold

        try:
            from sqlalchemy import select
            from src.db.models import Customer, Subscription

            async with ctx.db.session_ctx() as session:
                stmt = (
                    select(
                        Customer.id, Customer.name,
                        Subscription.plan, Subscription.payment_failures,
                    )
                    .join(Subscription, Subscription.customer_id == Customer.id)
                    .where(Customer.is_active == True)  # noqa: E712
                    .where(Subscription.is_active == True)  # noqa: E712
                    .where(Subscription.payment_failures >= threshold)
                )
                result = await session.execute(stmt)
                for row in result.all():
                    cid, cname, plan, failures = row
                    severity = "critical" if failures >= 3 else "warning"
                    detections.append(Detection(
                        rule_id=self.rule_id,
                        severity=severity,
                        title=f"Payment failures: {cname} ({failures}x)",
                        description=(
                            f"{cname} ({plan} plan) has {failures} failed payment "
                            f"attempts. Risk of service disruption."
                        ),
                        details={
                            "customer_id": cid,
                            "customer_name": cname,
                            "plan": plan,
                            "failures": failures,
                        },
                        recommendation=(
                            "Send payment reminder. If 3+ failures, "
                            "consider account suspension via approval queue."
                        ),
                        auto_fixable=failures < 3,
                    ))
        except Exception as exc:
            logger.error(f"FailedPaymentRule failed: {exc}")

        return detections


class TrainingStallRule:
    """Detects avatars with stalled training progress."""

    rule_id = "business.training_stall"

    async def evaluate(self, ctx: AgentContext) -> list[Detection]:
        if ctx.db is None:
            return []

        detections: list[Detection] = []
        stall_days = ctx.agent_config.training_stall_days

        try:
            from sqlalchemy import select
            from src.db.models import Avatar, Customer

            cutoff = datetime.utcnow() - timedelta(days=stall_days)

            async with ctx.db.session_ctx() as session:
                stmt = (
                    select(Avatar, Customer.name)
                    .join(Customer, Customer.id == Avatar.customer_id)
                    .where(Customer.is_active == True)  # noqa: E712
                    .where(Avatar.training_progress < 0.8)
                    .where(Avatar.training_progress > 0.0)
                    .where(Avatar.updated_at <= cutoff)
                )
                result = await session.execute(stmt)
                for avatar, customer_name in result.all():
                    detections.append(Detection(
                        rule_id=self.rule_id,
                        severity="warning",
                        title=f"Training stalled: {avatar.name} ({avatar.training_progress*100:.0f}%)",
                        description=(
                            f"Avatar '{avatar.name}' for {customer_name} at "
                            f"{avatar.training_progress*100:.0f}% training for "
                            f">{stall_days} days."
                        ),
                        details={
                            "avatar_id": avatar.id,
                            "avatar_name": avatar.name,
                            "customer_name": customer_name,
                            "training_progress": avatar.training_progress,
                            "stall_days": stall_days,
                        },
                        recommendation=(
                            "Reach out with training tips. Consider assigning "
                            "industry knowledge to jumpstart progress."
                        ),
                    ))
        except Exception as exc:
            logger.error(f"TrainingStallRule failed: {exc}")

        return detections


class OnboardingStuckRule:
    """Detects new customers who haven't started using the platform."""

    rule_id = "business.onboarding_stuck"

    async def evaluate(self, ctx: AgentContext) -> list[Detection]:
        if ctx.db is None:
            return []

        detections: list[Detection] = []
        stuck_hours = ctx.agent_config.onboarding_stuck_hours

        try:
            from sqlalchemy import func, select
            from src.db.models import Avatar, Conversation, Customer

            cutoff = datetime.utcnow() - timedelta(hours=stuck_hours)

            async with ctx.db.session_ctx() as session:
                # Conversation -> Avatar -> Customer to find who has conversations
                has_convos = (
                    select(Avatar.customer_id)
                    .join(Conversation, Conversation.avatar_id == Avatar.id)
                    .distinct()
                    .subquery()
                )
                stmt = (
                    select(Customer)
                    .where(Customer.is_active == True)  # noqa: E712
                    .where(Customer.created_at <= cutoff)
                    .where(Customer.id.notin_(select(has_convos.c.customer_id)))
                )
                result = await session.execute(stmt)
                for customer in result.scalars().all():
                    age_hours = (datetime.utcnow() - customer.created_at).total_seconds() / 3600
                    detections.append(Detection(
                        rule_id=self.rule_id,
                        severity="warning",
                        title=f"Onboarding stuck: {customer.name}",
                        description=(
                            f"{customer.name} signed up {age_hours:.0f}h ago "
                            f"with zero conversations."
                        ),
                        details={
                            "customer_id": customer.id,
                            "customer_name": customer.name,
                            "age_hours": round(age_hours),
                        },
                        recommendation=(
                            "Send onboarding email with getting-started guide. "
                            "Consider assigning a demo avatar."
                        ),
                    ))
        except Exception as exc:
            logger.error(f"OnboardingStuckRule failed: {exc}")

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
