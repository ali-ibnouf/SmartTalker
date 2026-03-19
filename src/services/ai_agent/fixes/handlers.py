"""Auto-fix handlers for the AI Optimization Agent.

Each handler targets a specific rule_id and can automatically
remediate the detected issue.
"""

from __future__ import annotations

import time
from typing import Any, Protocol, runtime_checkable

from src.services.ai_agent.rules import AgentContext, Detection
from src.utils.logger import setup_logger

logger = setup_logger("ai_agent.fixes")


@runtime_checkable
class AutoFixHandler(Protocol):
    """Protocol for auto-fix handlers."""

    fix_id: str

    async def can_fix(self, detection: Detection, ctx: AgentContext) -> bool:
        """Check if this handler can fix the given detection."""
        ...

    async def apply(self, detection: Detection, ctx: AgentContext) -> dict[str, Any]:
        """Apply the fix. Returns a result dict describing what was done."""
        ...


class FixRegistry:
    """Maps rule_ids to AutoFixHandler instances with cooldown enforcement."""

    def __init__(self) -> None:
        self._handlers: dict[str, AutoFixHandler] = {}
        self._last_fix_time: dict[str, float] = {}

    def register(self, rule_id: str, handler: AutoFixHandler) -> None:
        self._handlers[rule_id] = handler

    async def try_fix(
        self, detection: Detection, ctx: AgentContext
    ) -> dict[str, Any] | None:
        """Attempt to auto-fix a detection. Returns result or None."""
        handler = self._handlers.get(detection.rule_id)
        if handler is None:
            return None

        # Cooldown check
        cooldown_s = ctx.agent_config.fix_cooldown_s
        last_time = self._last_fix_time.get(handler.fix_id, 0)
        if time.time() - last_time < cooldown_s:
            logger.debug(
                f"Fix {handler.fix_id} skipped (cooldown)",
                extra={"rule_id": detection.rule_id},
            )
            return None

        if not await handler.can_fix(detection, ctx):
            logger.info(
                f"Fix {handler.fix_id} cannot handle detection",
                extra={"rule_id": detection.rule_id},
            )
            return None

        try:
            result = await handler.apply(detection, ctx)
            self._last_fix_time[handler.fix_id] = time.time()
            logger.info(
                f"Auto-fix applied: {handler.fix_id}",
                extra={"rule_id": detection.rule_id, "result": result},
            )
            return result
        except Exception as exc:
            logger.error(
                f"Auto-fix failed: {handler.fix_id}: {exc}",
                extra={"rule_id": detection.rule_id},
            )
            return {"action": "error", "error": str(exc), "fix_id": handler.fix_id}


# ── Concrete Handlers ─────────────────────────────────────────────────────


class SessionLimitFix:
    """Reduces max concurrent sessions on an overloaded GPU node."""

    fix_id = "fix.session_limit"

    async def can_fix(self, detection: Detection, ctx: AgentContext) -> bool:
        return (
            detection.rule_id == "gpu.fps_drop"
            and "node_id" in detection.details
            and detection.details.get("active_sessions", 0) > 1
        )

    async def apply(self, detection: Detection, ctx: AgentContext) -> dict[str, Any]:
        node_id = detection.details["node_id"]
        nm = ctx.node_manager
        node = nm.get_node(node_id)
        if node is None:
            return {"action": "skip", "reason": "node not found"}

        old_max = node.max_concurrent
        new_max = max(1, node.active_sessions - 1)
        node.max_concurrent = new_max
        return {
            "action": "reduced_max_concurrent",
            "node_id": node_id,
            "old_max": old_max,
            "new_max": new_max,
        }


class CacheClearFix:
    """Clears Redis cache to free memory when system memory is high."""

    fix_id = "fix.cache_clear"

    async def can_fix(self, detection: Detection, ctx: AgentContext) -> bool:
        return detection.rule_id == "system.high_memory" and ctx.redis is not None

    async def apply(self, detection: Detection, ctx: AgentContext) -> dict[str, Any]:
        redis = ctx.redis
        if redis is None:
            return {"action": "skip", "reason": "no redis"}

        # Clear cache keys (session:*, plan:*, rate:*)
        cleared = 0
        for pattern in ["session:*", "plan:*", "rate:*"]:
            try:
                keys = []
                async for key in redis.scan_iter(match=pattern, count=100):
                    keys.append(key)
                if keys:
                    await redis.delete(*keys)
                    cleared += len(keys)
            except Exception:
                pass

        return {"action": "cache_cleared", "keys_deleted": cleared}



class QuotaWarningFix:
    """Logs a notification for near-quota customers."""

    fix_id = "fix.quota_warning"

    async def can_fix(self, detection: Detection, ctx: AgentContext) -> bool:
        return detection.rule_id == "business.quota_exhaustion"

    async def apply(self, detection: Detection, ctx: AgentContext) -> dict[str, Any]:
        customer_id = detection.details.get("customer_id", "")
        usage_pct = detection.details.get("usage_pct", 0)
        plan = detection.details.get("plan", "")
        logger.info(
            f"Quota warning notification for customer {customer_id}",
            extra={
                "customer_id": customer_id,
                "usage_pct": usage_pct,
                "plan": plan,
            },
        )
        return {
            "action": "notification_logged",
            "customer_id": customer_id,
            "usage_pct": usage_pct,
            "message": f"Customer at {usage_pct:.0f}% quota — upgrade offer recommended.",
        }


class MemoryCleanupFix:
    """Clears expired Redis keys and closes idle WS sessions to free memory."""

    fix_id = "fix.memory_cleanup"

    async def can_fix(self, detection: Detection, ctx: AgentContext) -> bool:
        return detection.rule_id == "system.high_memory"

    async def apply(self, detection: Detection, ctx: AgentContext) -> dict[str, Any]:
        cleared_keys = 0

        # Clear Redis cache keys
        if ctx.redis is not None:
            for pattern in ["session:*", "plan:*", "rate:*"]:
                try:
                    keys: list[Any] = []
                    async for key in ctx.redis.scan_iter(match=pattern, count=100):
                        keys.append(key)
                    if keys:
                        await ctx.redis.delete(*keys)
                        cleared_keys += len(keys)
                except Exception:
                    pass

        # Close idle WS sessions older than threshold (capped)
        closed_sessions = 0
        max_close = getattr(ctx.agent_config, "safety_max_session_close_per_cycle", 5)
        if ctx.operator_manager is not None:
            try:
                timeout_min = ctx.agent_config.stale_session_timeout_min
                sessions = getattr(ctx.operator_manager, "get_active_sessions", lambda: [])()
                now = time.time()
                for s in sessions:
                    if closed_sessions >= max_close:
                        break
                    last_activity = getattr(s, "last_activity", now)
                    if isinstance(last_activity, (int, float)) and (now - last_activity) > timeout_min * 60:
                        close_fn = getattr(ctx.operator_manager, "close_session", None)
                        if close_fn:
                            await close_fn(s.session_id)
                            closed_sessions += 1
            except Exception as exc:
                logger.debug(f"Session cleanup failed: {exc}")

        return {
            "action": "memory_cleanup",
            "redis_keys_cleared": cleared_keys,
            "sessions_closed": closed_sessions,
        }


class DBConnectionFix:
    """Terminates idle PostgreSQL connections to free up connection slots."""

    fix_id = "fix.db_connection_cleanup"

    async def can_fix(self, detection: Detection, ctx: AgentContext) -> bool:
        return detection.rule_id == "infra.pg_connections" and ctx.db is not None

    async def apply(self, detection: Detection, ctx: AgentContext) -> dict[str, Any]:
        if ctx.db is None:
            return {"action": "skip", "reason": "no database"}

        max_kill = getattr(ctx.agent_config, "safety_db_kill_max", 5)

        try:
            from sqlalchemy import text

            async with ctx.db.session_ctx() as session:
                # Kill idle connections older than 5 minutes, excluding:
                # - Our own connection (pg_backend_pid())
                # - System/replication processes (backend_type filtering)
                # Limited to max_kill to prevent mass termination
                result = await session.execute(text(
                    "SELECT pg_terminate_backend(pid) "
                    "FROM pg_stat_activity "
                    "WHERE state = 'idle' "
                    "AND query_start < now() - interval '300 seconds' "
                    "AND pid <> pg_backend_pid() "
                    "AND backend_type = 'client backend' "
                    f"LIMIT {int(max_kill)}"
                ))
                killed = sum(1 for row in result if row[0])

            # Re-check usage
            usage_pct = detection.details.get("usage_pct", 0)
            if usage_pct >= 85 and killed == 0:
                return {
                    "action": "escalate",
                    "reason": "Still above 85% after cleanup, no idle connections to kill",
                    "killed": 0,
                }

            return {
                "action": "db_connection_cleanup",
                "connections_killed": killed,
                "max_kill_limit": max_kill,
            }
        except Exception as exc:
            return {"action": "error", "error": str(exc)}


class StaleSessionFix:
    """Closes visitor sessions inactive longer than the configured timeout.

    This fix runs on a schedule (not triggered by rule detection).
    The AIAgent._stale_session_loop() calls apply() directly.
    """

    fix_id = "fix.stale_session_cleanup"

    async def can_fix(self, detection: Detection, ctx: AgentContext) -> bool:
        return False  # Only called via scheduled loop, not rule-based

    async def apply_scheduled(self, ctx: AgentContext) -> dict[str, Any]:
        """Scheduled cleanup — not triggered by a Detection.

        Respects safety_max_session_close_per_cycle to prevent mass disconnection.
        """
        closed = 0
        max_close = getattr(ctx.agent_config, "safety_max_session_close_per_cycle", 5)

        if ctx.operator_manager is None:
            return {"action": "skip", "reason": "no operator_manager"}

        try:
            timeout_min = ctx.agent_config.stale_session_timeout_min
            sessions = getattr(ctx.operator_manager, "get_active_sessions", lambda: [])()
            now = time.time()

            for s in sessions:
                if closed >= max_close:
                    logger.info(
                        f"Stale session cleanup capped at {max_close} — "
                        f"remaining stale sessions deferred to next cycle"
                    )
                    break
                last_activity = getattr(s, "last_activity", now)
                if isinstance(last_activity, (int, float)) and (now - last_activity) > timeout_min * 60:
                    close_fn = getattr(ctx.operator_manager, "close_session", None)
                    if close_fn:
                        await close_fn(s.session_id)
                        closed += 1
        except Exception as exc:
            logger.warning(f"Stale session cleanup failed: {exc}")

        if closed:
            logger.info(f"Stale session cleanup: closed {closed} sessions")

        return {"action": "stale_session_cleanup", "sessions_closed": closed, "max_per_cycle": max_close}


class QuotaGraceFix:
    """Grants a grace period when a customer exceeds their quota.

    When usage >= 100%: sets a 2-hour grace period on the subscription.
    After grace period expires, the rate limiter enforces throttling.
    """

    fix_id = "fix.quota_grace"

    async def can_fix(self, detection: Detection, ctx: AgentContext) -> bool:
        return (
            detection.rule_id == "business.quota_exhaustion"
            and detection.details.get("usage_pct", 0) >= 100
            and ctx.db is not None
        )

    async def apply(self, detection: Detection, ctx: AgentContext) -> dict[str, Any]:
        if ctx.db is None:
            return {"action": "skip", "reason": "no database"}

        customer_id = detection.details.get("customer_id", "")
        if not customer_id:
            return {"action": "error", "error": "missing customer_id in detection details"}

        grace_hours = ctx.agent_config.quota_grace_hours

        try:
            from datetime import datetime, timedelta

            from sqlalchemy import update
            from src.db.models import Subscription

            grace_until = datetime.utcnow() + timedelta(hours=grace_hours)

            async with ctx.db.session_ctx() as session:
                await session.execute(
                    update(Subscription)
                    .where(
                        Subscription.customer_id == customer_id,
                        Subscription.is_active == True,  # noqa: E712
                    )
                    .values(grace_period_until=grace_until)
                )
                await session.commit()

            logger.info(
                f"Quota grace period set for customer {customer_id}",
                extra={"grace_until": grace_until.isoformat()},
            )

            return {
                "action": "quota_grace_set",
                "customer_id": customer_id,
                "grace_hours": grace_hours,
                "grace_until": grace_until.isoformat(),
            }
        except Exception as exc:
            return {"action": "error", "error": str(exc)}


class RateLimitThrottleFix:
    """Temporarily throttles a customer's API rate limit via Redis."""

    fix_id = "fix.rate_throttle"

    async def can_fix(self, detection: Detection, ctx: AgentContext) -> bool:
        return (
            detection.rule_id == "security.api_spike"
            and "customer_id" in detection.details
            and ctx.redis is not None
        )

    async def apply(self, detection: Detection, ctx: AgentContext) -> dict[str, Any]:
        if ctx.redis is None:
            return {"action": "skip", "reason": "no redis"}

        customer_id = detection.details.get("customer_id", "")
        if not customer_id:
            return {"action": "error", "error": "missing customer_id in detection details"}
        ttl = ctx.agent_config.throttle_duration_s
        limit = ctx.agent_config.throttle_rate_limit

        try:
            key = f"throttle:{customer_id}"
            await ctx.redis.set(key, limit, ex=ttl)

            logger.info(
                f"Rate throttle applied for customer {customer_id}",
                extra={"limit": limit, "ttl_s": ttl},
            )

            return {
                "action": "rate_throttle_applied",
                "customer_id": customer_id,
                "throttle_limit": limit,
                "ttl_seconds": ttl,
            }
        except Exception as exc:
            return {"action": "error", "error": str(exc)}


class WarmupJobFix:
    """Sends a lightweight warmup job to RunPod when no idle workers are available.

    Prevents cold start latency by proactively spinning up a worker.
    Has its own 10-minute cooldown to avoid spamming warmup jobs.
    """

    fix_id = "fix.runpod_warmup"

    def __init__(self) -> None:
        self._last_applied: float = 0.0
        self._cooldown_s: float = 600.0  # 10 minutes

    async def can_fix(self, detection: Detection, ctx: AgentContext) -> bool:
        if time.time() - self._last_applied < self._cooldown_s:
            return False
        return (
            detection.rule_id == "infra.runpod_workers"
            and detection.details.get("idle", 0) == 0
            and ctx.config is not None
        )

    async def apply(self, detection: Detection, ctx: AgentContext) -> dict[str, Any]:
        config = ctx.config
        endpoint_url = getattr(config, "runpod_endpoint_musetalk", "") or ""
        api_key = getattr(config, "runpod_api_key", "") or ""

        if not endpoint_url or not api_key:
            return {"action": "skip", "reason": "RunPod endpoint not configured"}

        try:
            import httpx

            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{endpoint_url}/run",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={"input": {"task": "warmup"}},
                )
                resp.raise_for_status()
                data = resp.json()
                job_id = data.get("id", "") if isinstance(data, dict) else ""

            self._last_applied = time.time()
            logger.info(
                "RunPod warmup job sent",
                extra={"job_id": job_id, "endpoint": endpoint_url},
            )

            return {
                "action": "runpod_warmup_sent",
                "job_id": job_id,
                "active_customers": detection.details.get("active_customers", 0),
            }
        except Exception as exc:
            return {"action": "error", "error": str(exc)}


class VideoDisableFix:
    """Disables video rendering on the RunPod client after consecutive failures.

    Sets runpod.video_disabled = True so the visitor WS handler
    sends fallback_vrm instead of attempting GPU renders.
    """

    fix_id = "fix.video_disable"

    async def can_fix(self, detection: Detection, ctx: AgentContext) -> bool:
        if detection.rule_id != "resilience.runpod_consecutive_failures":
            return False
        if ctx.pipeline is None:
            return False
        runpod = getattr(ctx.pipeline, "_runpod", None) or getattr(ctx.pipeline, "_runpod_client", None)
        if runpod is None:
            return False
        return not getattr(runpod, "video_disabled", False)

    async def apply(self, detection: Detection, ctx: AgentContext) -> dict[str, Any]:
        runpod = getattr(ctx.pipeline, "_runpod", None) or getattr(ctx.pipeline, "_runpod_client", None)
        if runpod is None:
            return {"action": "skip", "reason": "no runpod client"}

        runpod.video_disabled = True
        failures = detection.details.get("consecutive_failures", 0)
        logger.info(
            f"Video rendering disabled after {failures} consecutive failures",
            extra={"consecutive_failures": failures},
        )
        return {
            "action": "video_disabled",
            "consecutive_failures": failures,
            "message": "Video rendering disabled — visitors will receive VRM fallback.",
        }


class TextOnlyModeFix:
    """Enables text-only mode on the LLM engine after consecutive timeouts.

    Sets llm.text_only_mode = True so the pipeline skips TTS/rendering
    and returns text responses only.
    """

    fix_id = "fix.text_only_mode"

    async def can_fix(self, detection: Detection, ctx: AgentContext) -> bool:
        if detection.rule_id != "resilience.dashscope_consecutive_timeouts":
            return False
        if ctx.pipeline is None:
            return False
        llm = getattr(ctx.pipeline, "_llm", None)
        if llm is None:
            return False
        return not getattr(llm, "text_only_mode", False)

    async def apply(self, detection: Detection, ctx: AgentContext) -> dict[str, Any]:
        llm = getattr(ctx.pipeline, "_llm", None)
        if llm is None:
            return {"action": "skip", "reason": "no llm engine"}

        llm.text_only_mode = True
        timeouts = detection.details.get("consecutive_timeouts", 0)
        logger.info(
            f"Text-only mode enabled after {timeouts} consecutive DashScope timeouts",
            extra={"consecutive_timeouts": timeouts},
        )
        return {
            "action": "text_only_mode_enabled",
            "consecutive_timeouts": timeouts,
            "message": "Text-only mode enabled — TTS and rendering skipped.",
        }


class RedisMemoryFix:
    """Frees Redis memory by purging and clearing safe-to-evict cache keys."""

    fix_id = "fix.redis_memory"

    async def can_fix(self, detection: Detection, ctx: AgentContext) -> bool:
        return detection.rule_id == "infra.redis_memory" and ctx.redis is not None

    async def apply(self, detection: Detection, ctx: AgentContext) -> dict[str, Any]:
        if ctx.redis is None:
            return {"action": "skip", "reason": "no redis"}

        cleared = 0
        try:
            # memory purge (defrag)
            try:
                await ctx.redis.execute_command("MEMORY", "PURGE")
            except Exception:
                pass  # Not supported on all Redis versions

            # Clear safe-to-evict cache keys (plan cache refetches on miss)
            for pattern in ["plan:*", "session:*"]:
                keys: list[Any] = []
                async for key in ctx.redis.scan_iter(match=pattern, count=200):
                    keys.append(key)
                if keys:
                    await ctx.redis.delete(*keys)
                    cleared += len(keys)

        except Exception as exc:
            return {"action": "error", "error": str(exc)}

        return {
            "action": "redis_memory_cleanup",
            "keys_cleared": cleared,
        }


# ── Channel Integration Fixes ────────────────────────────────────────────────


class TelegramWebhookReregisterFix:
    """Re-registers Telegram webhook when delivery failures are detected.

    When the AI Agent detects webhook failures for a Telegram channel,
    this fix re-registers the webhook URL with the Telegram Bot API.
    """

    fix_id = "fix.telegram_webhook_reregister"

    async def can_fix(self, detection: Detection, ctx: AgentContext) -> bool:
        return (
            detection.rule_id == "channels.webhook_failures"
            and detection.details.get("channel_type") == "telegram"
            and ctx.db is not None
        )

    async def apply(self, detection: Detection, ctx: AgentContext) -> dict[str, Any]:
        if ctx.db is None:
            return {"action": "skip", "reason": "no database"}

        employee_id = detection.details.get("employee_id", "")
        if not employee_id:
            return {"action": "error", "error": "missing employee_id in detection details"}

        try:
            from sqlalchemy import select
            from src.db.models import EmployeeChannel

            async with ctx.db.session_ctx() as session:
                stmt = select(EmployeeChannel).where(
                    EmployeeChannel.employee_id == employee_id,
                    EmployeeChannel.channel_type == "telegram",
                    EmployeeChannel.enabled == True,  # noqa: E712
                )
                result = await session.execute(stmt)
                channel = result.scalar()

            if channel is None or not channel.tg_bot_token:
                return {"action": "skip", "reason": "no telegram config"}

            from src.channels.telegram import TelegramAdapter

            # Use config webhook URL if available, else default
            if ctx.config and getattr(ctx.config, "telegram_webhook_url", None):
                base = ctx.config.telegram_webhook_url.rstrip("/")
                webhook_url = f"{base}/{employee_id}" if not base.endswith(employee_id) else base
            else:
                webhook_url = f"https://api.maskki.com/webhooks/telegram/{employee_id}"

            tg_result = await TelegramAdapter.register_webhook(
                channel.tg_bot_token, webhook_url
            )

            # Clear failure counter (key includes hour suffix from _record_metric)
            if ctx.redis is not None:
                # Scan and delete all matching keys for this employee
                cursor = b"0"
                while True:
                    cursor, keys = await ctx.redis.scan(
                        cursor=cursor,
                        match=f"channel_fail:telegram:{employee_id}:*",
                        count=100,
                    )
                    if keys:
                        await ctx.redis.delete(*keys)
                    if cursor == b"0" or cursor == 0:
                        break

            telegram_ok = tg_result.get("ok", False) if isinstance(tg_result, dict) else False
            logger.info(
                f"Telegram webhook re-registered for {employee_id}",
                extra={"ok": telegram_ok},
            )

            return {
                "action": "telegram_webhook_reregistered",
                "employee_id": employee_id,
                "webhook_url": webhook_url,
                "telegram_ok": telegram_ok,
            }
        except Exception as exc:
            return {"action": "error", "error": str(exc)}


class ChannelRoutingFallbackFix:
    """Clears routing error counters and logs an alert.

    When a channel's routing error rate is too high, this fix:
    1. Clears the error counter to prevent stale alerts
    2. Logs a notification for the operator to investigate

    The actual fallback (routing to widget) is handled by ChannelRouter itself.
    """

    fix_id = "fix.channel_routing_fallback"

    async def can_fix(self, detection: Detection, ctx: AgentContext) -> bool:
        return (
            detection.rule_id == "channels.routing_errors"
            and ctx.redis is not None
        )

    async def apply(self, detection: Detection, ctx: AgentContext) -> dict[str, Any]:
        if ctx.redis is None:
            return {"action": "skip", "reason": "no redis"}

        ch_type = detection.details.get("channel_type", "")
        if not ch_type:
            return {"action": "error", "error": "missing channel_type in detection details"}
        error_rate = detection.details.get("error_rate_pct", 0)

        # If error rate is extreme (>80%), escalate — do NOT auto-disable
        # channels across all customers. Mass-disabling is a destructive
        # action that requires manual approval.
        if error_rate > 80:
            logger.warning(
                f"Channel {ch_type} has {error_rate:.0f}% error rate — escalating for manual review"
            )
            return {
                "action": "routing_alert_escalated",
                "channel_type": ch_type,
                "error_rate_pct": error_rate,
                "reason": (
                    "Error rate exceeded 80%. Channel disable requires "
                    "manual approval to avoid cross-customer impact."
                ),
            }

        return {
            "action": "routing_alert_logged",
            "channel_type": ch_type,
            "error_rate_pct": error_rate,
        }
