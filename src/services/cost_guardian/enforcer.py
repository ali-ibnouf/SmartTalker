"""Cost Enforcer — executes automatic cost-control actions."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any, Optional

from src.services.cost_guardian.analyzer import CostAlert
from src.utils.logger import setup_logger

logger = setup_logger("cost_guardian.enforcer")


class CostEnforcer:
    """Executes automatic cost-control actions via Redis flags + DB updates.

    Rate-limits destructive actions to prevent runaway enforcement loops
    when the analyzer repeatedly triggers alerts every cycle.
    """

    def __init__(
        self,
        db: Any,
        redis: Any,
        runpod_client: Any = None,
        max_actions_per_hour: int = 5,
    ) -> None:
        self.db = db
        self.redis = redis
        self.runpod = runpod_client
        self._max_actions_per_hour = max_actions_per_hour
        self._action_timestamps: list[float] = []

    async def execute_action(self, alert: CostAlert) -> dict[str, Any]:
        """Execute the appropriate action for an alert."""
        action = alert.action
        if not action:
            return {"action": "none", "reason": "no action specified"}

        # Rate-limit destructive actions
        if not self._can_execute():
            logger.warning(
                f"Cost Guardian action rate limit reached ({self._max_actions_per_hour}/hour) "
                f"— skipping {action}"
            )
            return {"action": action, "status": "rate_limited"}

        result: dict[str, Any] = {
            "action": action,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        try:
            if action == "pause_service":
                result.update(await self._pause_service(alert.service))
            elif action == "pause_all_sessions":
                result.update(await self._pause_all_sessions())
            elif action == "pause_customer":
                result.update(await self._pause_customer(alert.customer_id or ""))
            elif action == "throttle_customer":
                result.update(await self._throttle_customer(alert.customer_id or ""))
            elif action == "cancel_runpod_job":
                result.update(await self._cancel_stuck_jobs())
            elif action == "investigate":
                result.update({"status": "logged_for_investigation"})

            self._action_timestamps.append(time.time())
            await self._log_action(alert, result)

        except Exception as e:
            logger.error(f"Enforcer action failed: {action} — {e}")
            result["status"] = "error"
            result["error"] = str(e)

        return result

    def _can_execute(self) -> bool:
        """Check if another action is allowed within the hourly cap."""
        now = time.time()
        cutoff = now - 3600
        self._action_timestamps = [t for t in self._action_timestamps if t > cutoff]
        return len(self._action_timestamps) < self._max_actions_per_hour

    # ── Pause / Throttle Mechanisms ─────────────────────────────────────

    async def _pause_service(self, service_name: str) -> dict[str, Any]:
        """Temporarily pause a specific AI service by setting a Redis flag."""
        if self.redis:
            await self.redis.set(
                f"cost_guardian:paused:{service_name}", "1", ex=3600,
            )
        logger.critical(f"SERVICE PAUSED: {service_name} for 1 hour due to cost overrun")
        return {"status": "paused", "duration": "1 hour", "service": service_name}

    async def _pause_all_sessions(self) -> dict[str, Any]:
        """Emergency: pause all new sessions platform-wide."""
        if self.redis:
            await self.redis.set("cost_guardian:emergency_pause", "1", ex=3600)
        logger.critical("EMERGENCY PAUSE: All new sessions blocked for 1 hour")
        return {"status": "emergency_pause", "duration": "1 hour"}

    async def _pause_customer(self, customer_id: str) -> dict[str, Any]:
        """Pause a specific customer's sessions (margin protection)."""
        if self.redis:
            await self.redis.set(
                f"cost_guardian:customer_paused:{customer_id}", "1", ex=86400,
            )
        # Close active conversations
        if self.db:
            from src.db.models import Conversation
            from sqlalchemy import update, and_
            async with self.db.session() as session:
                await session.execute(
                    update(Conversation)
                    .where(and_(
                        Conversation.customer_id == customer_id,
                        Conversation.ended_at.is_(None),
                    ))
                    .values(ended_at=datetime.now(timezone.utc))
                )
                await session.commit()
        logger.critical(f"CUSTOMER PAUSED: {customer_id} for 24h due to margin overrun")
        return {"status": "customer_paused", "customer_id": customer_id, "duration": "24 hours"}

    async def _throttle_customer(self, customer_id: str) -> dict[str, Any]:
        """Reduce a customer's request rate (rapid fire protection)."""
        if self.redis:
            await self.redis.set(
                f"cost_guardian:throttle:{customer_id}", "10", ex=600,
            )
        logger.warning(f"CUSTOMER THROTTLED: {customer_id} to 10 req/min for 10 minutes")
        return {
            "status": "throttled", "customer_id": customer_id,
            "limit": "10 req/min", "duration": "10 minutes",
        }

    async def _cancel_stuck_jobs(self) -> dict[str, Any]:
        """Cancel RunPod jobs that have been running too long."""
        cancelled = 0
        if self.runpod and hasattr(self.runpod, "cancel_job"):
            if self.db:
                from src.db.models import APICostRecord
                from sqlalchemy import select, and_
                cutoff = datetime.now(timezone.utc) - __import__("datetime").timedelta(minutes=10)
                async with self.db.session() as session:
                    result = await session.execute(
                        select(APICostRecord.details)
                        .where(and_(
                            APICostRecord.service.in_(["gpu_render", "gpu_preprocess", "runpod"]),
                            APICostRecord.cost_usd == 0,
                            APICostRecord.created_at <= cutoff,
                        ))
                    )
                    for row in result.all():
                        try:
                            details = json.loads(row[0]) if row[0] else {}
                            job_id = details.get("job_id", "")
                            if job_id:
                                await self.runpod.cancel_job(job_id)
                                cancelled += 1
                        except Exception as e:
                            logger.error(f"Failed to cancel RunPod job: {e}")

        logger.warning(f"CANCELLED {cancelled} stuck RunPod jobs")
        return {"status": "jobs_cancelled", "count": cancelled}

    # ── Query Flags ─────────────────────────────────────────────────────

    async def is_service_paused(self, service: str) -> bool:
        """Check if a service is paused by cost guardian."""
        if not self.redis:
            return False
        return bool(await self.redis.exists(f"cost_guardian:paused:{service}"))

    async def is_emergency_paused(self) -> bool:
        """Check if platform is in emergency pause."""
        if not self.redis:
            return False
        return bool(await self.redis.exists("cost_guardian:emergency_pause"))

    async def is_customer_paused(self, customer_id: str) -> bool:
        """Check if customer is paused due to cost overrun."""
        if not self.redis:
            return False
        return bool(await self.redis.exists(f"cost_guardian:customer_paused:{customer_id}"))

    async def is_customer_throttled(self, customer_id: str) -> Optional[int]:
        """Return throttle limit if customer is throttled, else None."""
        if not self.redis:
            return None
        val = await self.redis.get(f"cost_guardian:throttle:{customer_id}")
        return int(val) if val else None

    # ── Logging ─────────────────────────────────────────────────────────

    async def _log_action(self, alert: CostAlert, result: dict[str, Any]) -> None:
        """Log enforcement action to cost_guardian_log table."""
        if not self.db:
            return
        try:
            from src.db.models import CostGuardianLog
            log_entry = CostGuardianLog(
                alert_level=alert.level.value,
                service=alert.service,
                message=alert.message,
                current_value=alert.current_value,
                threshold=alert.threshold,
                action_taken=alert.action or "",
                result=json.dumps(result),
                customer_id=alert.customer_id or "",
            )
            async with self.db.session() as session:
                session.add(log_entry)
                await session.commit()
        except Exception as e:
            logger.debug(f"Failed to log guardian action: {e}")
