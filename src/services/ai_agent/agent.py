"""AI Optimization Agent — main orchestrator.

Runs a background loop that periodically evaluates all registered
monitor rules, persists incidents, applies auto-fixes, and tracks
patterns for prevention.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any, Optional

from src.services.ai_agent.config import AgentSettings
from src.services.ai_agent.fixes.handlers import (
    ChannelRoutingFallbackFix,
    DBConnectionFix,
    FixRegistry,
    MemoryCleanupFix,
    QuotaGraceFix,
    QuotaWarningFix,
    RateLimitThrottleFix,
    RedisMemoryFix,
    StaleSessionFix,
    TelegramWebhookReregisterFix,
    TextOnlyModeFix,
    VideoDisableFix,
    WarmupJobFix,
)
from src.services.ai_agent.monitors.business import (
    ChurnRiskRule,
    EscalationSpikeRule,
    FailedPaymentRule,
    OnboardingStuckRule,
    QuotaExhaustionRule,
    TrainingStallRule,
)
from src.services.ai_agent.monitors.infrastructure import (
    ASRConnectionRule,
    ASRLatencyRule,
    DashScopeConsecutiveTimeoutRule,
    DashScopeLatencyRule,
    DashScopeQueueDepthRule,
    DashScopeQuotaRule,
    MarginSqueezeRule,
    PostgreSQLConnectionsRule,
    R2ConnectivityRule,
    R2DowntimeRule,
    RedisMemoryRule as InfraRedisMemoryRule,
    RunPodColdStartRule,
    RunPodConsecutiveFailureRule,
    RunPodHealthRule,
    RunPodQueueDepthRule,
    RunPodR2LatencyRule,
    RunPodRenderTimeRule,
    RunPodTaskTimeRule,
    RunPodTimeoutRule,
    RunPodWorkersRule,
    TTSConnectionRule,
    TTSLatencyRule,
)
from src.services.ai_agent.monitors.predictions import (
    CustomerMarginPredictionRule,
    DashScopeQuotaExhaustionRule,
    RunPodCostProjectionRule,
    VRMVideoRatioRule,
)
from src.services.ai_agent.monitors.channels import (
    ChannelDisconnectedRule,
    ChannelRoutingErrorRule,
    InactiveChannelRule,
    VisitorResolutionFailureRule,
    WebhookDeliveryFailureRule,
)
from src.services.ai_agent.monitors.security import (
    APIUsageSpikeRule,
    FailedAuthRule,
    PolicyViolationSpikeRule,
    SuspiciousActivityRule,
)
from src.services.ai_agent.monitors.system import (
    DiskSpaceRule,
    HighCPURule,
    HighMemoryRule,
)
from src.services.ai_agent.approval import ApprovalQueue
from src.services.ai_agent.notifications import NotificationDispatcher, NotificationSettings
from src.services.ai_agent.prevention import PatternTracker
from src.services.ai_agent.rules import AgentContext, Detection, RuleRegistry
from src.services.ai_agent.safety import HIGH_IMPACT_FIXES, SafetyGuard
from src.utils.logger import setup_logger

logger = setup_logger("ai_agent")


class AIAgent:
    """Central orchestrator for the AI Optimization Agent."""

    def __init__(self, ctx: AgentContext) -> None:
        self._ctx = ctx
        self._running = False
        self._task: Optional[asyncio.Task[None]] = None
        self._last_scan: Optional[datetime] = None
        self._scan_count = 0

        # Rule engine
        self._rule_registry = RuleRegistry()
        self._register_rules()

        # Auto-fix engine
        self._fix_registry = FixRegistry()
        self._register_fixes()

        # Prevention engine
        self._pattern_tracker = PatternTracker(ctx.db)

        # Notification engine
        self._notifier = NotificationDispatcher(
            operator_manager=ctx.operator_manager,
            agent_config=ctx.agent_config,
            smtp_config=NotificationSettings(),
        )

        # Approval queue
        self._approval_queue = ApprovalQueue(db=ctx.db, config=ctx.agent_config)

        # Safety guard
        self._safety = SafetyGuard(
            redis=ctx.redis,
            circuit_breaker_threshold=ctx.agent_config.safety_circuit_breaker_threshold,
            circuit_breaker_cooldown_s=ctx.agent_config.safety_circuit_breaker_cooldown_s,
            max_fixes_per_cycle=ctx.agent_config.safety_max_fixes_per_cycle,
            max_fixes_per_hour=ctx.agent_config.safety_max_fixes_per_hour,
        )

    def _register_rules(self) -> None:
        """Register all monitor rules."""
        # System
        self._rule_registry.register(HighCPURule())
        self._rule_registry.register(HighMemoryRule())
        self._rule_registry.register(DiskSpaceRule())
        # Infrastructure
        self._rule_registry.register(PostgreSQLConnectionsRule())
        self._rule_registry.register(InfraRedisMemoryRule())
        self._rule_registry.register(DashScopeLatencyRule())
        self._rule_registry.register(ASRLatencyRule())
        self._rule_registry.register(TTSLatencyRule())
        self._rule_registry.register(ASRConnectionRule())
        self._rule_registry.register(TTSConnectionRule())
        self._rule_registry.register(DashScopeQuotaRule())
        self._rule_registry.register(RunPodHealthRule())
        self._rule_registry.register(RunPodQueueDepthRule())
        self._rule_registry.register(RunPodRenderTimeRule())
        self._rule_registry.register(RunPodWorkersRule())
        self._rule_registry.register(RunPodColdStartRule())
        self._rule_registry.register(RunPodTaskTimeRule())
        self._rule_registry.register(RunPodR2LatencyRule())
        self._rule_registry.register(RunPodTimeoutRule())
        self._rule_registry.register(R2ConnectivityRule())
        # Resilience
        self._rule_registry.register(RunPodConsecutiveFailureRule())
        self._rule_registry.register(DashScopeConsecutiveTimeoutRule())
        self._rule_registry.register(DashScopeQueueDepthRule())
        self._rule_registry.register(R2DowntimeRule())
        self._rule_registry.register(MarginSqueezeRule())
        # Predictions
        self._rule_registry.register(DashScopeQuotaExhaustionRule())
        self._rule_registry.register(RunPodCostProjectionRule())
        self._rule_registry.register(CustomerMarginPredictionRule())
        self._rule_registry.register(VRMVideoRatioRule())
        # Business
        self._rule_registry.register(ChurnRiskRule())
        self._rule_registry.register(QuotaExhaustionRule())
        self._rule_registry.register(EscalationSpikeRule())
        self._rule_registry.register(FailedPaymentRule())
        self._rule_registry.register(TrainingStallRule())
        self._rule_registry.register(OnboardingStuckRule())
        # Security
        self._rule_registry.register(PolicyViolationSpikeRule())
        self._rule_registry.register(SuspiciousActivityRule())
        self._rule_registry.register(FailedAuthRule())
        self._rule_registry.register(APIUsageSpikeRule())
        # Channels
        self._rule_registry.register(WebhookDeliveryFailureRule())
        self._rule_registry.register(ChannelDisconnectedRule())
        self._rule_registry.register(ChannelRoutingErrorRule())
        self._rule_registry.register(InactiveChannelRule())
        self._rule_registry.register(VisitorResolutionFailureRule())

        # Stale session cleanup (scheduled)
        self._stale_session_fix = StaleSessionFix()
        self._stale_session_task: Optional[asyncio.Task[None]] = None

    def _register_fixes(self) -> None:
        """Register auto-fix handlers."""
        self._fix_registry.register("system.high_memory", MemoryCleanupFix())
        self._fix_registry.register("infra.pg_connections", DBConnectionFix())
        self._fix_registry.register("infra.redis_memory", RedisMemoryFix())
        self._fix_registry.register("infra.runpod_workers", WarmupJobFix())
        self._fix_registry.register("business.quota_exhaustion", QuotaGraceFix())
        self._fix_registry.register("security.api_spike", RateLimitThrottleFix())
        self._fix_registry.register("resilience.runpod_consecutive_failures", VideoDisableFix())
        self._fix_registry.register("resilience.dashscope_consecutive_timeouts", TextOnlyModeFix())
        # Channels
        self._fix_registry.register("channels.webhook_failures", TelegramWebhookReregisterFix())
        self._fix_registry.register("channels.routing_errors", ChannelRoutingFallbackFix())

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the background monitoring loop."""
        if self._running:
            return
        self._running = True
        await self._notifier.start()
        self._task = asyncio.create_task(self._loop())
        self._stale_session_task = asyncio.create_task(self._stale_session_loop())
        logger.info("AI Agent started", extra={
            "scan_interval_s": self._ctx.agent_config.scan_interval_s,
            "rules_count": len(self._rule_registry.rules),
        })

    async def stop(self) -> None:
        """Stop the background loop."""
        self._running = False
        for task in (self._task, self._stale_session_task):
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._task = None
        self._stale_session_task = None
        await self._notifier.stop()
        logger.info("AI Agent stopped")

    async def _loop(self) -> None:
        """Main scan loop — runs every scan_interval_s.

        Safety features:
        - Kill switch check at the top of each cycle (Redis-based, instant disable)
        - Circuit breaker skips rules that fail consecutively
        - Fix rate limiter caps auto-fixes per cycle and per hour
        - Entire cycle wrapped in broad exception handler to prevent crash
        """
        while self._running:
            try:
                # Kill switch — skip entire cycle if engaged
                if await self._safety.is_kill_switch_active():
                    logger.info("Agent kill switch active — skipping scan cycle")
                    await asyncio.sleep(self._ctx.agent_config.scan_interval_s)
                    continue

                # Reset per-cycle fix counter
                self._safety.start_cycle()

                detections = await self._safe_run_rules()
                for d in detections:
                    await self._handle_detection(d)

                # Expire stale approval requests
                try:
                    await self._approval_queue.expire_stale()
                except Exception as exc:
                    logger.debug(f"Approval expiry failed: {exc}")

                self._last_scan = datetime.utcnow()
                self._scan_count += 1
                if detections:
                    logger.info(
                        f"Scan complete: {len(detections)} detections",
                        extra={"scan_number": self._scan_count},
                    )
            except asyncio.CancelledError:
                raise  # Allow clean shutdown
            except Exception as exc:
                # Broad catch — agent must NEVER crash the FastAPI process
                logger.error(
                    f"Agent scan failed (isolated): {exc}",
                    extra={"scan_number": self._scan_count},
                )
            await asyncio.sleep(self._ctx.agent_config.scan_interval_s)

    async def _safe_run_rules(self) -> list[Detection]:
        """Run all rules with circuit breaker integration.

        Skips rules whose circuit breaker is open, and records
        successes/failures to update breaker state.
        """
        all_detections: list[Detection] = []

        async def _run_one_safe(rule: Any) -> list[Detection]:
            rule_id = rule.rule_id
            if self._safety.should_skip_rule(rule_id):
                return []
            try:
                result = await rule.evaluate(self._ctx)
                self._safety.circuit_breaker.record_success(rule_id)
                return result
            except Exception as exc:
                self._safety.circuit_breaker.record_failure(rule_id)
                logger.error(
                    f"Rule {rule_id} failed: {exc}",
                    extra={"rule_id": rule_id},
                )
                return []

        results = await asyncio.gather(
            *[_run_one_safe(r) for r in self._rule_registry.rules]
        )
        for batch in results:
            all_detections.extend(batch)
        return all_detections

    async def _stale_session_loop(self) -> None:
        """Periodically clean up stale visitor sessions.

        Respects the kill switch and limits sessions closed per cycle.
        """
        interval = self._ctx.agent_config.stale_session_check_interval_s
        while self._running:
            try:
                # Skip if kill switch is active
                if await self._safety.is_kill_switch_active():
                    await asyncio.sleep(interval)
                    continue
                result = await self._stale_session_fix.apply_scheduled(self._ctx)
                closed = result.get("sessions_closed", 0)
                if closed:
                    logger.info(f"Stale session cleanup: {closed} sessions closed")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug(f"Stale session loop error: {exc}")
            await asyncio.sleep(interval)

    # ── Detection Handling ────────────────────────────────────────────────

    async def _handle_detection(self, d: Detection) -> None:
        """Persist incident, record pattern, notify, try auto-fix.

        Safety checks:
        - High-impact fixes are routed to the approval queue
        - Fix rate limiter caps the number of fixes per cycle/hour
        """
        # 1. Persist incident
        incident_id = await self._persist_incident(d)

        # 2. Record pattern for prevention
        pattern_key = self._make_pattern_key(d)
        await self._pattern_tracker.record_occurrence(d.rule_id, pattern_key)

        # 3. Dispatch notification (WebSocket / email / audit log)
        await self._notifier.dispatch(d, incident_id)

        # 4. Auto-fix if enabled and applicable
        if not (d.auto_fixable and self._ctx.agent_config.auto_fix_enabled and incident_id):
            return

        # Check if this fix requires approval (high-impact)
        handler = self._fix_registry._handlers.get(d.rule_id)
        if handler is not None and getattr(handler, "fix_id", "") in HIGH_IMPACT_FIXES:
            await self._route_to_approval(d, handler, incident_id)
            return

        # Check rate limits
        if not self._safety.can_auto_fix(getattr(handler, "fix_id", "") if handler else ""):
            logger.info(
                f"Fix rate limit reached — skipping auto-fix for {d.rule_id}",
                extra={"rule_id": d.rule_id, "cycle_count": self._safety.fix_limiter.cycle_count},
            )
            return

        result = await self._fix_registry.try_fix(d, self._ctx)
        if result is not None:
            self._safety.record_fix_applied()
            await self._persist_action(
                incident_id=incident_id,
                action_type=result.get("action", "unknown"),
                description=f"Auto-fix applied for {d.rule_id}",
                result=result,
                auto=True,
            )
            if result.get("action") != "error":
                await self._update_incident_status(incident_id, "auto_fixed")
                await self._notifier.dispatch_status_change(incident_id, "auto_fixed")

    async def _route_to_approval(
        self, d: Detection, handler: Any, incident_id: str
    ) -> None:
        """Route a high-impact fix to the approval queue instead of auto-executing."""
        fix_id = getattr(handler, "fix_id", "unknown")
        logger.info(
            f"High-impact fix {fix_id} requires approval — routing to queue",
            extra={"rule_id": d.rule_id, "fix_id": fix_id},
        )
        approval_id = await self._approval_queue.request_approval(
            action_type=fix_id.replace("fix.", ""),
            target_id=d.rule_id,
            description=(
                f"Auto-fix '{fix_id}' triggered by rule '{d.rule_id}': {d.title}. "
                f"This fix affects global pipeline state and requires admin approval."
            ),
            details={
                "incident_id": incident_id,
                "detection_details": d.details,
                "severity": d.severity,
                "recommendation": d.recommendation,
            },
        )
        if approval_id:
            await self._persist_action(
                incident_id=incident_id,
                action_type="approval_requested",
                description=f"High-impact fix {fix_id} sent to approval queue",
                result={"approval_id": approval_id, "fix_id": fix_id},
                auto=True,
            )

    def _make_pattern_key(self, d: Detection) -> str:
        """Create a stable pattern key from detection details."""
        parts = [d.rule_id]
        for key in ("node_id", "customer_id", "avatar_id", "channel"):
            if key in d.details:
                parts.append(f"{key}={d.details[key]}")
        return "|".join(parts)

    # ── Database Persistence ──────────────────────────────────────────────

    async def _persist_incident(self, d: Detection) -> Optional[str]:
        """Save a detection as an agent_incidents row. Returns incident ID."""
        if self._ctx.db is None:
            return None

        try:
            from src.db.models import AgentIncident

            incident = AgentIncident(
                rule_id=d.rule_id,
                severity=d.severity,
                title=d.title,
                description=d.description,
                details=json.dumps(d.details),
                recommendation=d.recommendation,
                status="open",
            )
            async with self._ctx.db.session_ctx() as session:
                session.add(incident)
                await session.flush()
                return incident.id
        except Exception as exc:
            logger.error(f"Failed to persist incident: {exc}")
            return None

    async def _persist_action(
        self,
        incident_id: str,
        action_type: str,
        description: str,
        result: dict[str, Any],
        auto: bool = True,
    ) -> None:
        """Save an action taken on an incident."""
        if self._ctx.db is None:
            return

        try:
            from src.db.models import AgentAction

            action = AgentAction(
                incident_id=incident_id,
                action_type=action_type,
                description=description,
                result=json.dumps(result),
                auto=auto,
            )
            async with self._ctx.db.session_ctx() as session:
                session.add(action)
        except Exception as exc:
            logger.error(f"Failed to persist action: {exc}")

    async def _update_incident_status(self, incident_id: str, status: str) -> None:
        """Update an incident's status."""
        if self._ctx.db is None:
            return

        try:
            from sqlalchemy import update
            from src.db.models import AgentIncident

            async with self._ctx.db.session_ctx() as session:
                stmt = (
                    update(AgentIncident)
                    .where(AgentIncident.id == incident_id)
                    .values(
                        status=status,
                        resolved_at=datetime.utcnow() if status in ("resolved", "auto_fixed") else None,
                    )
                )
                await session.execute(stmt)
        except Exception as exc:
            logger.error(f"Failed to update incident status: {exc}")

    # ── Public Query Methods (for API routes) ─────────────────────────────

    @property
    def approval_queue(self) -> ApprovalQueue:
        return self._approval_queue

    @property
    def safety(self) -> SafetyGuard:
        return self._safety

    async def get_stats(self) -> dict[str, Any]:
        """Return agent summary statistics."""
        stats: dict[str, Any] = {
            "scan_count": self._scan_count,
            "last_scan_at": self._last_scan.isoformat() if self._last_scan else None,
            "rules_count": len(self._rule_registry.rules),
            "running": self._running,
            "incidents_total": 0,
            "open_incidents": 0,
            "auto_fixes_applied": 0,
            "patterns_tracked": 0,
        }

        if self._ctx.db is None:
            return stats

        try:
            from sqlalchemy import func, select
            from src.db.models import AgentAction, AgentIncident, AgentPattern

            async with self._ctx.db.session_ctx() as session:
                # Total incidents
                r = await session.execute(select(func.count(AgentIncident.id)))
                stats["incidents_total"] = r.scalar() or 0

                # Open incidents
                r = await session.execute(
                    select(func.count(AgentIncident.id))
                    .where(AgentIncident.status == "open")
                )
                stats["open_incidents"] = r.scalar() or 0

                # Auto-fixes
                r = await session.execute(
                    select(func.count(AgentAction.id))
                    .where(AgentAction.auto == True)
                )
                stats["auto_fixes_applied"] = r.scalar() or 0

                # Patterns
                r = await session.execute(select(func.count(AgentPattern.id)))
                stats["patterns_tracked"] = r.scalar() or 0
        except Exception as exc:
            logger.error(f"Failed to get stats: {exc}")

        return stats

    async def get_incidents(
        self, status: Optional[str] = None, limit: int = 20
    ) -> dict[str, Any]:
        """Return incidents, optionally filtered by status."""
        if self._ctx.db is None:
            return {"incidents": [], "total": 0}

        try:
            from sqlalchemy import func, select
            from src.db.models import AgentIncident

            async with self._ctx.db.session_ctx() as session:
                base = select(AgentIncident)
                count_base = select(func.count(AgentIncident.id))
                if status:
                    base = base.where(AgentIncident.status == status)
                    count_base = count_base.where(AgentIncident.status == status)

                # Total
                r = await session.execute(count_base)
                total = r.scalar() or 0

                # Items
                stmt = base.order_by(AgentIncident.created_at.desc()).limit(limit)
                r = await session.execute(stmt)
                incidents = r.scalars().all()

                return {
                    "incidents": [
                        {
                            "id": i.id,
                            "rule_id": i.rule_id,
                            "severity": i.severity,
                            "title": i.title,
                            "description": i.description,
                            "details": json.loads(i.details) if i.details else {},
                            "recommendation": i.recommendation,
                            "status": i.status,
                            "created_at": i.created_at.isoformat() if i.created_at else None,
                            "resolved_at": i.resolved_at.isoformat() if i.resolved_at else None,
                        }
                        for i in incidents
                    ],
                    "total": total,
                }
        except Exception as exc:
            logger.error(f"Failed to get incidents: {exc}")
            return {"incidents": [], "total": 0}

    async def get_predictions(self) -> list[dict[str, Any]]:
        """Return recurrence predictions from the pattern tracker."""
        return await self._pattern_tracker.get_predictions()

    async def acknowledge_incident(self, incident_id: str) -> dict[str, Any]:
        """Mark an incident as acknowledged."""
        if not incident_id:
            return {"error": "incident_id is required"}
        if not await self._incident_exists(incident_id):
            return {"error": "incident not found", "incident_id": incident_id}
        await self._update_incident_status(incident_id, "acknowledged")
        return {"incident_id": incident_id, "status": "acknowledged"}

    async def resolve_incident(self, incident_id: str) -> dict[str, Any]:
        """Mark an incident as resolved."""
        if not incident_id:
            return {"error": "incident_id is required"}
        if not await self._incident_exists(incident_id):
            return {"error": "incident not found", "incident_id": incident_id}
        await self._update_incident_status(incident_id, "resolved")
        return {"incident_id": incident_id, "status": "resolved"}

    async def _incident_exists(self, incident_id: str) -> bool:
        """Check if an incident exists in the database."""
        if self._ctx.db is None:
            return False
        try:
            from sqlalchemy import select
            from src.db.models import AgentIncident

            async with self._ctx.db.session_ctx() as session:
                r = await session.execute(
                    select(AgentIncident.id).where(AgentIncident.id == incident_id)
                )
                return r.scalar() is not None
        except Exception:
            return False

    async def run_manual_scan(self) -> list[dict[str, Any]]:
        """Run a single scan and return detections (for on-demand API).

        Respects the same safety guards as the automated loop:
        - Kill switch blocks the entire scan
        - Circuit breakers skip failing rules
        - Fix rate limiter caps auto-fixes
        """
        # Safety: honour the kill switch even for manual scans
        if await self._safety.is_kill_switch_active():
            logger.info("Manual scan blocked — kill switch active")
            return []

        # Reset per-cycle fix counter so rate limits apply cleanly
        self._safety.start_cycle()

        # Use _safe_run_rules (circuit-breaker aware) instead of raw run_all
        detections = await self._safe_run_rules()
        for d in detections:
            await self._handle_detection(d)
        self._last_scan = datetime.utcnow()
        self._scan_count += 1
        return [
            {
                "rule_id": d.rule_id,
                "severity": d.severity,
                "title": d.title,
                "description": d.description,
                "recommendation": d.recommendation,
                "auto_fixable": d.auto_fixable,
            }
            for d in detections
        ]
