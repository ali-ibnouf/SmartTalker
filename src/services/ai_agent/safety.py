"""Safety guard for the AI Optimization Agent.

Prevents the agent from destabilizing the main application through:
- Runtime kill switch (Redis-based, instant disable without restart)
- Circuit breaker (disables rules that fail consecutively)
- Blast radius limits (caps fixes per cycle and per hour)
- High-impact fix gating (routes dangerous fixes through approval queue)
"""

from __future__ import annotations

import time
from typing import Any

from src.utils.logger import setup_logger

logger = setup_logger("ai_agent.safety")

# Fixes that modify global pipeline state or kill external resources.
# These MUST go through the approval queue instead of auto-executing.
HIGH_IMPACT_FIXES = frozenset({
    "fix.video_disable",        # Disables video for ALL customers
    "fix.text_only_mode",       # Switches ALL customers to text-only
    "fix.db_connection_cleanup", # Kills PostgreSQL connections
})


class CircuitBreaker:
    """Disables a rule temporarily after consecutive failures."""

    def __init__(self, threshold: int = 5, cooldown_s: int = 600) -> None:
        self._threshold = threshold
        self._cooldown_s = cooldown_s
        self._failure_counts: dict[str, int] = {}
        self._tripped_at: dict[str, float] = {}

    def record_success(self, rule_id: str) -> None:
        self._failure_counts.pop(rule_id, None)

    def record_failure(self, rule_id: str) -> None:
        count = self._failure_counts.get(rule_id, 0) + 1
        self._failure_counts[rule_id] = count
        if count >= self._threshold:
            self._tripped_at[rule_id] = time.time()
            logger.warning(
                f"Circuit breaker tripped for rule {rule_id} "
                f"after {count} consecutive failures — disabled for {self._cooldown_s}s"
            )

    def is_open(self, rule_id: str) -> bool:
        """Return True if the rule should be skipped (circuit is open)."""
        tripped = self._tripped_at.get(rule_id)
        if tripped is None:
            return False
        if time.time() - tripped >= self._cooldown_s:
            # Cooldown expired — reset and allow retry
            self._tripped_at.pop(rule_id, None)
            self._failure_counts.pop(rule_id, None)
            logger.info(f"Circuit breaker reset for rule {rule_id}")
            return False
        return True

    def get_open_circuits(self) -> list[str]:
        """Return rule_ids with open (tripped) circuit breakers."""
        now = time.time()
        return [
            rid for rid, t in self._tripped_at.items()
            if now - t < self._cooldown_s
        ]


class FixRateLimiter:
    """Limits the number of auto-fixes per cycle and per hour."""

    def __init__(self, max_per_cycle: int = 3, max_per_hour: int = 10) -> None:
        self._max_per_cycle = max_per_cycle
        self._max_per_hour = max_per_hour
        self._cycle_count = 0
        self._hour_timestamps: list[float] = []

    def reset_cycle(self) -> None:
        """Call at the start of each scan cycle."""
        self._cycle_count = 0

    def can_fix(self) -> bool:
        """Check if another fix is allowed in this cycle and hour."""
        if self._cycle_count >= self._max_per_cycle:
            return False

        now = time.time()
        cutoff = now - 3600
        self._hour_timestamps = [t for t in self._hour_timestamps if t > cutoff]
        if len(self._hour_timestamps) >= self._max_per_hour:
            return False

        return True

    def record_fix(self) -> None:
        """Record that a fix was applied."""
        self._cycle_count += 1
        self._hour_timestamps.append(time.time())

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    @property
    def hour_count(self) -> int:
        now = time.time()
        cutoff = now - 3600
        return sum(1 for t in self._hour_timestamps if t > cutoff)


class SafetyGuard:
    """Central safety coordinator for the AI Agent."""

    def __init__(
        self,
        redis: Any = None,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_cooldown_s: int = 600,
        max_fixes_per_cycle: int = 3,
        max_fixes_per_hour: int = 10,
    ) -> None:
        self._redis = redis
        self.circuit_breaker = CircuitBreaker(
            threshold=circuit_breaker_threshold,
            cooldown_s=circuit_breaker_cooldown_s,
        )
        self.fix_limiter = FixRateLimiter(
            max_per_cycle=max_fixes_per_cycle,
            max_per_hour=max_fixes_per_hour,
        )

    async def is_kill_switch_active(self) -> bool:
        """Check if the runtime kill switch is engaged via Redis."""
        if self._redis is None:
            return False
        try:
            return bool(await self._redis.exists("agent:kill_switch"))
        except Exception:
            # If Redis is unreachable, do NOT disable the agent —
            # the kill switch is opt-in safety, not a default-deny gate.
            return False

    async def activate_kill_switch(self, reason: str = "") -> bool:
        """Engage the runtime kill switch (stops all agent activity)."""
        if self._redis is None:
            return False
        try:
            await self._redis.set("agent:kill_switch", reason or "activated", ex=86400)
            logger.critical(f"Agent kill switch ACTIVATED: {reason}")
            return True
        except Exception as exc:
            logger.error(f"Failed to activate kill switch: {exc}")
            return False

    async def deactivate_kill_switch(self) -> bool:
        """Release the runtime kill switch."""
        if self._redis is None:
            return False
        try:
            await self._redis.delete("agent:kill_switch")
            logger.info("Agent kill switch deactivated")
            return True
        except Exception as exc:
            logger.error(f"Failed to deactivate kill switch: {exc}")
            return False

    def should_skip_rule(self, rule_id: str) -> bool:
        """Return True if circuit breaker says this rule should be skipped."""
        return self.circuit_breaker.is_open(rule_id)

    def can_auto_fix(self, fix_id: str) -> bool:
        """Check if an auto-fix is allowed (rate limits + impact check)."""
        if fix_id in HIGH_IMPACT_FIXES:
            return False  # Must go through approval queue
        return self.fix_limiter.can_fix()

    def record_fix_applied(self) -> None:
        """Record that an auto-fix was applied."""
        self.fix_limiter.record_fix()

    def start_cycle(self) -> None:
        """Reset per-cycle counters at the beginning of each scan."""
        self.fix_limiter.reset_cycle()

    async def get_status(self) -> dict[str, Any]:
        """Return safety system status for admin API."""
        kill_switch = await self.is_kill_switch_active()
        return {
            "kill_switch_active": kill_switch,
            "open_circuit_breakers": self.circuit_breaker.get_open_circuits(),
            "fixes_this_cycle": self.fix_limiter.cycle_count,
            "fixes_this_hour": self.fix_limiter.hour_count,
            "max_fixes_per_cycle": self.fix_limiter._max_per_cycle,
            "max_fixes_per_hour": self.fix_limiter._max_per_hour,
            "high_impact_fixes_requiring_approval": sorted(HIGH_IMPACT_FIXES),
        }
