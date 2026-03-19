"""Channel integration monitoring rules.

Monitors WhatsApp, Telegram, and Widget channel health:
- Webhook delivery failures
- Channel routing errors
- Disconnected/misconfigured channels
- Visitor resolution failures
- Cross-channel response latency
"""

from __future__ import annotations

from datetime import datetime, timedelta

from src.services.ai_agent.rules import AgentContext, Detection
from src.utils.logger import setup_logger

logger = setup_logger("ai_agent.monitors.channels")


class WebhookDeliveryFailureRule:
    """Detects channels with high webhook delivery failure rates.

    Checks Redis counters for failed webhook deliveries per employee/channel.
    Key pattern: channel_fail:{channel_type}:{employee_id}
    """

    rule_id = "channels.webhook_failures"

    async def evaluate(self, ctx: AgentContext) -> list[Detection]:
        if ctx.redis is None:
            return []

        detections: list[Detection] = []
        threshold = ctx.agent_config.webhook_failure_threshold

        try:
            cursor = b"0"
            while True:
                cursor, keys = await ctx.redis.scan(
                    cursor=cursor, match="channel_fail:*", count=100
                )
                for key in keys:
                    count = await ctx.redis.get(key)
                    if count is None:
                        continue
                    fail_count = int(count)
                    if fail_count >= threshold:
                        # Parse key: channel_fail:{type}:{employee_id}:{hour}
                        # (_record_metric appends :{hour} automatically)
                        parts = key if isinstance(key, str) else key.decode()
                        segments = parts.split(":")
                        ch_type = segments[1] if len(segments) > 1 else "unknown"
                        # employee_id is segments[2], segments[-1] is the hour
                        emp_id = segments[2] if len(segments) > 2 else "unknown"

                        severity = "critical" if fail_count >= threshold * 2 else "warning"
                        detections.append(Detection(
                            rule_id=self.rule_id,
                            severity=severity,
                            title=f"Webhook failures: {ch_type} ({fail_count}x)",
                            description=(
                                f"{ch_type.capitalize()} webhook for employee {emp_id} "
                                f"has {fail_count} delivery failures in the current window. "
                                f"Messages may not be reaching visitors."
                            ),
                            details={
                                "channel_type": ch_type,
                                "employee_id": emp_id,
                                "failure_count": fail_count,
                            },
                            recommendation=(
                                "Check channel credentials and webhook URL. "
                                "For WhatsApp: verify access token hasn't expired. "
                                "For Telegram: re-register webhook via dashboard."
                            ),
                            auto_fixable=ch_type == "telegram",
                        ))
                if cursor == b"0" or cursor == 0:
                    break
        except Exception as exc:
            logger.error(f"WebhookDeliveryFailureRule failed: {exc}")

        return detections


class ChannelDisconnectedRule:
    """Detects enabled channels with missing or invalid credentials.

    Queries employee_channels table for enabled channels that have
    empty credentials (no token, no access key, etc.).
    """

    rule_id = "channels.disconnected"

    async def evaluate(self, ctx: AgentContext) -> list[Detection]:
        if ctx.db is None:
            return []

        detections: list[Detection] = []

        try:
            from sqlalchemy import select
            from src.db.models import Employee, EmployeeChannel

            async with ctx.db.session_ctx() as session:
                stmt = (
                    select(EmployeeChannel, Employee.name)
                    .join(Employee, Employee.id == EmployeeChannel.employee_id)
                    .where(EmployeeChannel.enabled == True)  # noqa: E712
                )
                result = await session.execute(stmt)

                for channel, emp_name in result.all():
                    missing = self._check_credentials(channel)
                    if missing:
                        detections.append(Detection(
                            rule_id=self.rule_id,
                            severity="warning",
                            title=f"Channel misconfigured: {emp_name} / {channel.channel_type}",
                            description=(
                                f"{channel.channel_type.capitalize()} channel for "
                                f"employee '{emp_name}' is enabled but missing: "
                                f"{', '.join(missing)}. Messages will fail."
                            ),
                            details={
                                "employee_id": channel.employee_id,
                                "employee_name": emp_name,
                                "channel_type": channel.channel_type,
                                "missing_fields": missing,
                            },
                            recommendation=(
                                f"Go to Dashboard -> {emp_name} -> Channels -> "
                                f"{channel.channel_type.capitalize()} and fill in "
                                f"the missing credentials, or disable the channel."
                            ),
                        ))
        except Exception as exc:
            logger.error(f"ChannelDisconnectedRule failed: {exc}")

        return detections

    @staticmethod
    def _check_credentials(channel) -> list[str]:
        """Return list of missing credential field names.

        Widget channels have no external credentials (they work via
        WebSocket at /session), so they are not checked here.
        """
        missing = []
        ch_type = channel.channel_type

        if ch_type == "whatsapp":
            if not getattr(channel, "wa_phone_number_id", ""):
                missing.append("phone_number_id")
            if not getattr(channel, "wa_access_token", ""):
                missing.append("access_token")
        elif ch_type == "telegram":
            if not getattr(channel, "tg_bot_token", ""):
                missing.append("bot_token")
        # widget: no external credentials needed (WebSocket-based)

        return missing


class ChannelRoutingErrorRule:
    """Detects high error rates in channel message routing.

    Checks Redis counters for routing errors (ASR failures, agent
    errors, TTS errors) per channel type in the last hour.
    Key pattern: route_err:{channel_type}:{hour}
    """

    rule_id = "channels.routing_errors"

    async def evaluate(self, ctx: AgentContext) -> list[Detection]:
        if ctx.redis is None:
            return []

        detections: list[Detection] = []
        threshold = ctx.agent_config.channel_routing_error_threshold
        current_hour = datetime.utcnow().strftime("%Y%m%d%H")

        try:
            for ch_type in ("whatsapp", "telegram", "widget"):
                key = f"route_err:{ch_type}:{current_hour}"
                count = await ctx.redis.get(key)
                if count is None:
                    continue
                err_count = int(count)
                if err_count >= threshold:
                    # Also get success count for error rate
                    ok_key = f"route_ok:{ch_type}:{current_hour}"
                    ok_count = int(await ctx.redis.get(ok_key) or 0)
                    total = err_count + ok_count
                    error_rate = (err_count / total * 100) if total > 0 else 100

                    severity = "critical" if error_rate >= 50 else "warning"
                    detections.append(Detection(
                        rule_id=self.rule_id,
                        severity=severity,
                        title=f"Routing errors: {ch_type} ({error_rate:.0f}%)",
                        description=(
                            f"{ch_type.capitalize()} channel has {err_count} routing "
                            f"errors out of {total} messages ({error_rate:.0f}%) "
                            f"in the current hour. Visitors are not receiving responses."
                        ),
                        details={
                            "channel_type": ch_type,
                            "error_count": err_count,
                            "success_count": ok_count,
                            "error_rate_pct": round(error_rate, 1),
                        },
                        recommendation=(
                            "Check Agent Engine, ASR, and TTS health. "
                            "Review logs at 'channels.router' for specific errors."
                        ),
                        auto_fixable=True,
                    ))
        except Exception as exc:
            logger.error(f"ChannelRoutingErrorRule failed: {exc}")

        return detections


class InactiveChannelRule:
    """Detects channels that haven't received messages for an extended period.

    An enabled channel with no messages for N days may indicate a
    broken webhook or misconfiguration the customer doesn't know about.
    """

    rule_id = "channels.inactive"

    async def evaluate(self, ctx: AgentContext) -> list[Detection]:
        if ctx.db is None:
            return []

        detections: list[Detection] = []
        inactive_days = ctx.agent_config.channel_inactive_days

        try:
            from sqlalchemy import func, select
            from src.db.models import (
                Avatar, Conversation, Employee, EmployeeChannel,
            )

            cutoff = datetime.utcnow() - timedelta(days=inactive_days)

            async with ctx.db.session_ctx() as session:
                # Get enabled channels
                ch_stmt = (
                    select(EmployeeChannel, Employee.name)
                    .join(Employee, Employee.id == EmployeeChannel.employee_id)
                    .where(EmployeeChannel.enabled == True)  # noqa: E712
                    .where(EmployeeChannel.channel_type != "widget")  # widget always works
                )
                ch_result = await session.execute(ch_stmt)

                for channel, emp_name in ch_result.all():
                    # Check for recent conversations on this channel
                    conv_stmt = (
                        select(func.count(Conversation.id))
                        .join(Avatar, Avatar.id == Conversation.avatar_id)
                        .join(Employee, Employee.id == Avatar.employee_id)
                        .where(Employee.id == channel.employee_id)
                        .where(Conversation.channel == channel.channel_type)
                        .where(Conversation.started_at >= cutoff)
                    )
                    conv_result = await session.execute(conv_stmt)
                    count = conv_result.scalar() or 0

                    if count == 0:
                        # Check if channel was created more than inactive_days ago
                        if channel.created_at and channel.created_at < cutoff:
                            detections.append(Detection(
                                rule_id=self.rule_id,
                                severity="info",
                                title=f"Inactive: {emp_name} / {channel.channel_type}",
                                description=(
                                    f"{channel.channel_type.capitalize()} channel for "
                                    f"'{emp_name}' has zero messages in the last "
                                    f"{inactive_days} days. The webhook may be broken "
                                    f"or the channel is not being used."
                                ),
                                details={
                                    "employee_id": channel.employee_id,
                                    "employee_name": emp_name,
                                    "channel_type": channel.channel_type,
                                    "inactive_days": inactive_days,
                                },
                                recommendation=(
                                    "Verify the webhook is registered and working. "
                                    "Send a test message to confirm. If unused, "
                                    "consider disabling the channel."
                                ),
                            ))
        except Exception as exc:
            logger.error(f"InactiveChannelRule failed: {exc}")

        return detections


class VisitorResolutionFailureRule:
    """Detects high visitor resolution failure rates.

    When visitor identity resolution fails (DB errors, timeouts),
    visitors get random IDs and lose cross-channel memory continuity.
    Checks Redis counter: visitor_resolve_fail:{hour}
    """

    rule_id = "channels.visitor_resolution_failures"

    async def evaluate(self, ctx: AgentContext) -> list[Detection]:
        if ctx.redis is None:
            return []

        detections: list[Detection] = []
        threshold = ctx.agent_config.visitor_resolve_fail_threshold
        current_hour = datetime.utcnow().strftime("%Y%m%d%H")

        try:
            fail_key = f"visitor_resolve_fail:{current_hour}"
            ok_key = f"visitor_resolve_ok:{current_hour}"
            fail_count = int(await ctx.redis.get(fail_key) or 0)
            ok_count = int(await ctx.redis.get(ok_key) or 0)

            if fail_count >= threshold:
                total = fail_count + ok_count
                fail_rate = (fail_count / total * 100) if total > 0 else 100

                detections.append(Detection(
                    rule_id=self.rule_id,
                    severity="warning" if fail_rate < 50 else "critical",
                    title=f"Visitor resolution failures: {fail_count} ({fail_rate:.0f}%)",
                    description=(
                        f"{fail_count} visitor identity resolutions failed in the "
                        f"current hour ({fail_rate:.0f}% failure rate). "
                        f"Cross-channel memory continuity is degraded."
                    ),
                    details={
                        "failure_count": fail_count,
                        "success_count": ok_count,
                        "failure_rate_pct": round(fail_rate, 1),
                    },
                    recommendation=(
                        "Check PostgreSQL connection health and "
                        "visitor_channel_map table for issues."
                    ),
                ))
        except Exception as exc:
            logger.error(f"VisitorResolutionFailureRule failed: {exc}")

        return detections
