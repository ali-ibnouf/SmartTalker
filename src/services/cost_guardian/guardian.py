"""Cost Guardian — main loop that runs every 60 seconds."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Optional

from src.services.cost_guardian.analyzer import AlertLevel, CostAnalyzer
from src.services.cost_guardian.enforcer import CostEnforcer
from src.services.cost_guardian.monitor import CostMonitor
from src.services.cost_guardian.reporter import CostReporter
from src.utils.logger import setup_logger

logger = setup_logger("cost_guardian")


class CostGuardian:
    """Main guardian loop — runs every 60 seconds, analyzes costs, takes action."""

    def __init__(
        self,
        db: Any,
        redis: Any,
        resend_api_key: Optional[str] = None,
        runpod_client: Any = None,
    ) -> None:
        self.monitor = CostMonitor(db)
        self.analyzer = CostAnalyzer(self.monitor)
        self.enforcer = CostEnforcer(db, redis, runpod_client)
        self.reporter = CostReporter(resend_api_key, self.monitor)
        self.running = False
        self._last_daily_report: Optional[datetime] = None

    async def start(self) -> None:
        """Start the guardian loop."""
        self.running = True
        logger.info("Cost Guardian started — monitoring every 60 seconds")

        while self.running:
            try:
                await self._check_cycle()
            except Exception as e:
                logger.error(f"Cost Guardian cycle error: {e}")

            # Daily report at 23:55 UTC
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            if now.hour == 23 and now.minute >= 55:
                if self._last_daily_report is None or self._last_daily_report.date() != now.date():
                    try:
                        await self.reporter.send_daily_report()
                        self._last_daily_report = now
                        logger.info("Daily cost report sent")
                    except Exception as e:
                        logger.error(f"Daily report failed: {e}")

            await asyncio.sleep(60)

    async def stop(self) -> None:
        """Signal the guardian loop to stop."""
        self.running = False

    async def _check_cycle(self) -> None:
        """Single monitoring cycle."""
        alerts = await self.analyzer.run_full_analysis()

        if not alerts:
            return

        # Separate by action needed
        action_alerts = [
            a for a in alerts
            if a.action and a.level == AlertLevel.EMERGENCY
        ]
        notification_alerts = [
            a for a in alerts
            if a.level in (AlertLevel.WARNING, AlertLevel.CRITICAL)
        ]

        # Execute emergency actions
        for alert in action_alerts:
            result = await self.enforcer.execute_action(alert)
            await self.reporter.send_emergency_report(alert, result)
            logger.critical(f"EMERGENCY ACTION: {alert.message} -> {result}")

        # Send warning/critical email (batched, max 1 per 15 min)
        if notification_alerts:
            should_send = await self._should_send_notification()
            if should_send:
                await self.reporter.send_alert_email(notification_alerts)

    async def _should_send_notification(self) -> bool:
        """Rate limit notification emails to max 1 per 15 minutes."""
        if not self.enforcer.redis:
            return True
        key = "cost_guardian:last_notification"
        last = await self.enforcer.redis.get(key)
        if last:
            return False
        await self.enforcer.redis.set(key, "1", ex=900)  # 15 min cooldown
        return True
