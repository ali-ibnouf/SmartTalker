"""Predictive monitoring rules — forecast issues before they happen.

Rules:
- DashScope quota exhaustion prediction (days until budget runs out)
- RunPod cost projection (monthly spend based on usage trend)
- Customer margin prediction (will a customer become unprofitable?)
- VRM vs Video ratio (optimize RunPod spend by analyzing fallback frequency)
"""

from __future__ import annotations

from datetime import datetime, timedelta

from src.services.ai_agent.rules import AgentContext, Detection
from src.utils.logger import setup_logger

logger = setup_logger("ai_agent.monitors.predictions")


class DashScopeQuotaExhaustionRule:
    """Predicts days until DashScope monthly budget is exhausted.

    Uses the current month's spending trend to forecast when the budget
    will run out. Fires a warning when < 7 days remain.
    """

    rule_id = "prediction.dashscope_quota_exhaustion"

    async def evaluate(self, ctx: AgentContext) -> list[Detection]:
        if ctx.db is None:
            return []

        try:
            from sqlalchemy import func, select
            from src.db.models import APICostRecord

            now = datetime.utcnow()
            month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            days_elapsed = max(1, (now - month_start).days or 1)

            async with ctx.db.session_ctx() as session:
                # Sum DashScope costs this month (ASR + LLM + TTS)
                stmt = (
                    select(func.coalesce(func.sum(APICostRecord.cost_usd), 0))
                    .where(
                        APICostRecord.created_at >= month_start,
                        APICostRecord.service.in_(["asr", "llm", "tts"]),
                    )
                )
                result = await session.execute(stmt)
                total_cost = result.scalar() or 0.0

            budget = ctx.agent_config.dashscope_monthly_budget_usd
            if budget <= 0 or total_cost <= 0:
                return []

            daily_burn = total_cost / days_elapsed
            remaining = max(0, budget - total_cost)
            days_remaining = remaining / daily_burn if daily_burn > 0 else 999

            # Also calculate end-of-month projected total
            import calendar
            days_in_month = calendar.monthrange(now.year, now.month)[1]
            projected_total = daily_burn * days_in_month

            if days_remaining <= 7:
                severity = "critical" if days_remaining <= 3 else "warning"
                return [Detection(
                    rule_id=self.rule_id,
                    severity=severity,
                    title=f"DashScope budget: {days_remaining:.0f} days remaining",
                    description=(
                        f"At current burn rate (${daily_burn:.2f}/day), DashScope budget "
                        f"of ${budget:.0f} will be exhausted in {days_remaining:.0f} days. "
                        f"Spent: ${total_cost:.2f}, remaining: ${remaining:.2f}. "
                        f"Projected month-end total: ${projected_total:.2f}."
                    ),
                    details={
                        "days_remaining": round(days_remaining, 1),
                        "daily_burn_usd": round(daily_burn, 2),
                        "total_spent_usd": round(total_cost, 2),
                        "remaining_usd": round(remaining, 2),
                        "budget_usd": budget,
                        "projected_total_usd": round(projected_total, 2),
                    },
                    recommendation=(
                        "Reduce DashScope usage by limiting concurrent sessions or "
                        "upgrading budget allocation. Consider rate-limiting heavy customers."
                    ),
                )]
        except Exception as exc:
            logger.debug(f"DashScopeQuotaExhaustionRule skipped: {exc}")

        return []


class RunPodCostProjectionRule:
    """Projects monthly RunPod costs based on recent usage trend.

    Fires a warning if projected monthly spend exceeds the configured
    RunPod budget threshold.
    """

    rule_id = "prediction.runpod_cost_projection"

    async def evaluate(self, ctx: AgentContext) -> list[Detection]:
        if ctx.db is None:
            return []

        try:
            from sqlalchemy import func, select
            from src.db.models import APICostRecord

            now = datetime.utcnow()
            week_ago = now - timedelta(days=7)

            async with ctx.db.session_ctx() as session:
                # RunPod costs over the last 7 days
                stmt = (
                    select(func.coalesce(func.sum(APICostRecord.cost_usd), 0))
                    .where(
                        APICostRecord.service == "runpod",
                        APICostRecord.created_at >= week_ago,
                    )
                )
                result = await session.execute(stmt)
                week_cost = result.scalar() or 0.0

                # Also get job count
                count_stmt = (
                    select(func.count(APICostRecord.id))
                    .where(
                        APICostRecord.service == "runpod",
                        APICostRecord.created_at >= week_ago,
                    )
                )
                count_result = await session.execute(count_stmt)
                job_count = count_result.scalar() or 0

            if week_cost <= 0:
                return []

            daily_rate = week_cost / 7
            import calendar
            days_in_month = calendar.monthrange(now.year, now.month)[1]
            projected_monthly = daily_rate * days_in_month

            # Get RunPod budget from DashScope budget (RunPod typically 40% of total)
            runpod_budget = ctx.agent_config.dashscope_monthly_budget_usd * 0.4

            if runpod_budget <= 0:
                return []

            usage_pct = (projected_monthly / runpod_budget) * 100

            if usage_pct >= 80:
                severity = "critical" if usage_pct >= 100 else "warning"
                return [Detection(
                    rule_id=self.rule_id,
                    severity=severity,
                    title=f"RunPod projection: ${projected_monthly:.2f}/mo ({usage_pct:.0f}% of budget)",
                    description=(
                        f"Based on 7-day trend (${week_cost:.2f}/wk, {job_count} jobs), "
                        f"RunPod is projected to cost ${projected_monthly:.2f}/mo. "
                        f"Budget: ${runpod_budget:.0f} ({usage_pct:.0f}% utilization)."
                    ),
                    details={
                        "projected_monthly_usd": round(projected_monthly, 2),
                        "daily_rate_usd": round(daily_rate, 2),
                        "week_cost_usd": round(week_cost, 2),
                        "week_jobs": job_count,
                        "budget_usd": round(runpod_budget, 2),
                        "usage_pct": round(usage_pct, 1),
                    },
                    recommendation=(
                        "Optimize RunPod costs: increase min workers to reduce cold starts, "
                        "or use VRM fallback more aggressively for low-value sessions."
                    ),
                )]
        except Exception as exc:
            logger.debug(f"RunPodCostProjectionRule skipped: {exc}")

        return []


class CustomerMarginPredictionRule:
    """Predicts which customers will become unprofitable this month.

    Examines usage trend over the past week and projects whether
    API cost will exceed revenue by month-end.
    """

    rule_id = "prediction.customer_margin"

    async def evaluate(self, ctx: AgentContext) -> list[Detection]:
        if ctx.db is None:
            return []

        detections: list[Detection] = []

        try:
            from sqlalchemy import func, select
            from src.db.models import APICostRecord, Customer, Subscription

            now = datetime.utcnow()
            month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            days_elapsed = max(1, (now - month_start).days or 1)

            import calendar
            days_in_month = calendar.monthrange(now.year, now.month)[1]

            async with ctx.db.session_ctx() as session:
                # Active customers with subscriptions
                stmt = (
                    select(
                        Customer.id,
                        Customer.company,
                        Subscription.plan,
                        Subscription.price_monthly,
                    )
                    .join(Subscription, Subscription.customer_id == Customer.id)
                    .where(
                        Customer.is_active == True,  # noqa: E712
                        Subscription.is_active == True,  # noqa: E712
                        Subscription.price_monthly > 0,
                    )
                )
                customers = (await session.execute(stmt)).all()

                for cust_id, company, plan, revenue in customers:
                    # Current month API costs
                    cost_stmt = (
                        select(func.coalesce(func.sum(APICostRecord.cost_usd), 0))
                        .where(
                            APICostRecord.customer_id == cust_id,
                            APICostRecord.created_at >= month_start,
                        )
                    )
                    cost_result = await session.execute(cost_stmt)
                    current_cost = cost_result.scalar() or 0.0

                    if current_cost <= 0:
                        continue

                    # Project end-of-month cost
                    daily_rate = current_cost / days_elapsed
                    projected_cost = daily_rate * days_in_month
                    projected_margin_pct = ((revenue - projected_cost) / revenue) * 100

                    # Warn if projected to exceed 70% of revenue
                    if projected_cost >= revenue * 0.7:
                        severity = "critical" if projected_cost >= revenue else "warning"
                        detections.append(Detection(
                            rule_id=self.rule_id,
                            severity=severity,
                            title=f"Margin risk: {company} projected ${projected_cost:.2f} vs ${revenue:.0f} revenue",
                            description=(
                                f"Customer '{company}' ({plan}) is projected to cost "
                                f"${projected_cost:.2f}/mo against ${revenue:.2f}/mo revenue "
                                f"(margin: {projected_margin_pct:.0f}%). "
                                f"Current spend: ${current_cost:.2f} over {days_elapsed} days."
                            ),
                            details={
                                "customer_id": cust_id,
                                "company": company,
                                "plan": plan,
                                "current_cost_usd": round(current_cost, 2),
                                "projected_cost_usd": round(projected_cost, 2),
                                "revenue_usd": round(revenue, 2),
                                "projected_margin_pct": round(projected_margin_pct, 1),
                                "daily_rate_usd": round(daily_rate, 2),
                            },
                            recommendation=(
                                f"Suggest plan upgrade for '{company}'. "
                                f"Or review their usage for optimization opportunities."
                            ),
                        ))
        except Exception as exc:
            logger.debug(f"CustomerMarginPredictionRule skipped: {exc}")

        return detections


class VRMVideoRatioRule:
    """Monitors the VRM fallback vs video rendering ratio.

    If VRM fallbacks exceed a threshold, it indicates RunPod reliability
    issues or an opportunity to save costs by defaulting to VRM for
    low-value sessions.
    """

    rule_id = "prediction.vrm_video_ratio"

    async def evaluate(self, ctx: AgentContext) -> list[Detection]:
        if ctx.db is None:
            return []

        try:
            from sqlalchemy import func, select
            from src.db.models import APICostRecord, Conversation

            now = datetime.utcnow()
            day_ago = now - timedelta(hours=24)

            async with ctx.db.session_ctx() as session:
                # Count successful RunPod render jobs in last 24h
                video_stmt = (
                    select(func.count(APICostRecord.id))
                    .where(
                        APICostRecord.service == "runpod",
                        APICostRecord.created_at >= day_ago,
                    )
                )
                video_result = await session.execute(video_stmt)
                video_count = video_result.scalar() or 0

                # Count conversations with GPU cost > 0 (video was attempted)
                # plus conversations that would have tried video (gpu_cost == 0 but had avatar)
                total_stmt = (
                    select(func.count(Conversation.id))
                    .where(Conversation.started_at >= day_ago)
                )
                total_result = await session.execute(total_stmt)
                total_video_sessions = total_result.scalar() or 0

            if total_video_sessions < 5:
                return []  # Not enough data

            # Estimate fallbacks: sessions that expected video but didn't get RunPod jobs
            # (rough heuristic — a turn can have multiple renders)
            vrm_fallbacks = max(0, total_video_sessions - video_count)
            fallback_pct = (vrm_fallbacks / total_video_sessions) * 100

            runpod_cost_saved = vrm_fallbacks * 0.01  # ~$0.01 per render saved

            detections: list[Detection] = []

            if fallback_pct >= 30:
                severity = "warning" if fallback_pct < 50 else "critical"
                detections.append(Detection(
                    rule_id=self.rule_id,
                    severity=severity,
                    title=f"VRM fallback rate: {fallback_pct:.0f}% ({vrm_fallbacks}/{total_video_sessions})",
                    description=(
                        f"{fallback_pct:.0f}% of video sessions fell back to VRM in the last 24h "
                        f"({vrm_fallbacks} fallbacks out of {total_video_sessions} video sessions). "
                        f"This indicates RunPod reliability issues."
                    ),
                    details={
                        "fallback_pct": round(fallback_pct, 1),
                        "vrm_fallbacks": vrm_fallbacks,
                        "video_renders": video_count,
                        "total_video_sessions": total_video_sessions,
                    },
                    recommendation=(
                        "Investigate RunPod failure causes. If GPU reliability is low, "
                        "consider defaulting low-value sessions to VRM mode."
                    ),
                ))

            # Optimization opportunity: if RunPod cost per render is high
            if video_count > 0 and fallback_pct < 10:
                # Everything works but we could save by routing some to VRM
                detections.append(Detection(
                    rule_id=self.rule_id,
                    severity="info",
                    title=f"RunPod optimization: {video_count} renders, {fallback_pct:.0f}% fallback",
                    description=(
                        f"RunPod is stable ({video_count} renders, {fallback_pct:.0f}% fallback). "
                        f"Consider routing brief/simple interactions to VRM to reduce costs."
                    ),
                    details={
                        "video_renders": video_count,
                        "fallback_pct": round(fallback_pct, 1),
                        "potential_savings_usd": round(runpod_cost_saved, 2),
                    },
                    recommendation=(
                        "Route short conversations (< 30s) or FAQ replies to VRM mode "
                        "to save ~$0.01/render on RunPod costs."
                    ),
                ))

            return detections

        except Exception as exc:
            logger.debug(f"VRMVideoRatioRule skipped: {exc}")

        return []
