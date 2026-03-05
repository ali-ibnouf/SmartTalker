"""Auto-fix handlers for the AI Optimization Agent.

Each handler targets a specific rule_id and can automatically
remediate the detected issue.
"""

from __future__ import annotations

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
    """Maps rule_ids to AutoFixHandler instances."""

    def __init__(self) -> None:
        self._handlers: dict[str, AutoFixHandler] = {}

    def register(self, rule_id: str, handler: AutoFixHandler) -> None:
        self._handlers[rule_id] = handler

    async def try_fix(
        self, detection: Detection, ctx: AgentContext
    ) -> dict[str, Any] | None:
        """Attempt to auto-fix a detection. Returns result or None."""
        handler = self._handlers.get(detection.rule_id)
        if handler is None:
            return None

        if not await handler.can_fix(detection, ctx):
            logger.info(
                f"Fix {handler.fix_id} cannot handle detection",
                extra={"rule_id": detection.rule_id},
            )
            return None

        try:
            result = await handler.apply(detection, ctx)
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
            return {"error": str(exc), "fix_id": handler.fix_id}


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


class HeartbeatReconnectFix:
    """Marks a timed-out node for reconnection."""

    fix_id = "fix.heartbeat_reconnect"

    async def can_fix(self, detection: Detection, ctx: AgentContext) -> bool:
        return (
            detection.rule_id == "gpu.heartbeat_timeout"
            and "node_id" in detection.details
        )

    async def apply(self, detection: Detection, ctx: AgentContext) -> dict[str, Any]:
        node_id = detection.details["node_id"]
        nm = ctx.node_manager
        node = nm.get_node(node_id)
        if node is None:
            return {"action": "skip", "reason": "node already deregistered"}

        # Mark node offline so it won't receive new sessions
        node.status = "offline"
        return {
            "action": "marked_offline",
            "node_id": node_id,
            "hostname": detection.details.get("hostname", ""),
        }


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
