"""Billing Engine for per-second session metering.

Tracks active sessions, calculates costs at $0.002/sec (configurable via
billing_rate_per_second), enforces quotas, and writes UsageRecords to the database.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from src.config import Settings
from src.utils.exceptions import BillingError
from src.utils.logger import setup_logger

logger = setup_logger("pipeline.billing")


@dataclass
class BillingSession:
    """Active billing session."""

    session_id: str
    customer_id: str
    avatar_id: str = ""
    channel: str = "web"
    started_at: float = 0.0
    stopped_at: float = 0.0
    total_seconds: float = 0.0
    total_cost: float = 0.0


class BillingEngine:
    """Per-second billing metering engine.

    Tracks active sessions, calculates costs, enforces quotas,
    and persists usage records to the database.

    Args:
        config: Application settings.
        db: Database instance for usage record persistence.
    """

    # Alert thresholds: (percentage_remaining, level_name)
    ALERT_THRESHOLDS = [
        (20, "warning"),
        (5, "urgent"),
        (0, "critical"),
    ]

    def __init__(self, config: Settings, db: Any = None) -> None:
        self._config = config
        self._db = db
        self._rate = config.billing_rate_per_second
        self._grace = config.billing_grace_period_s
        self._active_sessions: dict[str, BillingSession] = {}
        self._alerted: dict[str, str] = {}  # customer_id → last alert level sent
        self._loaded = False

        logger.info(
            "BillingEngine initialized",
            extra={"rate": self._rate, "grace_s": self._grace},
        )

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    async def load(self) -> None:
        """Mark billing engine as ready."""
        self._loaded = True
        logger.info("BillingEngine loaded")

    async def unload(self) -> None:
        """Stop all active sessions and clean up."""
        for session_id in list(self._active_sessions.keys()):
            try:
                await self.stop_session(session_id)
            except Exception as exc:
                logger.warning(f"Error stopping session {session_id}: {exc}")
        self._loaded = False
        logger.info("BillingEngine unloaded")

    async def start_session(
        self,
        session_id: str,
        customer_id: str,
        avatar_id: str = "",
        channel: str = "web",
    ) -> BillingSession:
        """Start billing for a session.

        Checks quota and concurrent session limit before allowing start.
        """
        if not self._config.billing_enabled:
            return BillingSession(session_id=session_id, customer_id=customer_id)

        # Check quota
        remaining = await self.check_quota(customer_id)
        if remaining <= 0:
            raise BillingError(
                message="Quota exceeded",
                detail=f"Customer {customer_id} has no remaining seconds",
            )

        # Check concurrent session limit
        if not await self.check_concurrent_limit(customer_id):
            plan_info = await self.get_plan_info(customer_id)
            max_conc = plan_info.get("max_concurrent", 1)
            raise BillingError(
                message="Concurrent session limit reached",
                detail=f"Customer {customer_id} max concurrent sessions: {max_conc}",
            )

        session = BillingSession(
            session_id=session_id,
            customer_id=customer_id,
            avatar_id=avatar_id,
            channel=channel,
            started_at=time.time(),
        )
        self._active_sessions[session_id] = session

        logger.info(
            "Billing session started",
            extra={"session_id": session_id, "customer_id": customer_id},
        )
        return session

    async def stop_session(self, session_id: str) -> Optional[BillingSession]:
        """Stop billing for a session and write the usage record."""
        session = self._active_sessions.pop(session_id, None)
        if session is None:
            return None

        session.stopped_at = time.time()
        raw_seconds = session.stopped_at - session.started_at

        # Apply grace period
        billable_seconds = max(0, raw_seconds - self._grace)
        session.total_seconds = round(billable_seconds, 2)
        session.total_cost = round(billable_seconds * self._rate, 6)

        # Write usage record to DB
        if self._db is not None and session.total_seconds > 0:
            try:
                await self._write_usage_record(session)
            except Exception as exc:
                logger.warning(f"Failed to write usage record: {exc}")

        logger.info(
            "Billing session stopped",
            extra={
                "session_id": session_id,
                "seconds": session.total_seconds,
                "cost": session.total_cost,
            },
        )

        # Check balance alerts after usage is recorded
        if self._db is not None and self._config.billing_enabled:
            try:
                await self.check_balance_and_alert(session.customer_id)
            except Exception as exc:
                logger.warning(f"Balance alert check failed: {exc}")

        return session

    async def check_quota(self, customer_id: str) -> float:
        """Check remaining seconds for a customer (plan + extra).

        Returns total remaining seconds (positive) or 0 if over quota.
        """
        if self._db is None:
            return float("inf")  # No DB = unlimited

        balance = await self.get_balance(customer_id)
        return float(balance["total_remaining"])

    async def get_balance(self, customer_id: str) -> dict[str, Any]:
        """Get dual balance breakdown for a customer.

        Returns plan_seconds_remaining, plan_seconds_total, extra_seconds_remaining,
        total_remaining, plan_renewal_date, and usage_pct.
        """
        if self._db is None:
            return {
                "plan_seconds_remaining": float("inf"),
                "plan_seconds_total": float("inf"),
                "extra_seconds_remaining": 0,
                "total_remaining": float("inf"),
                "plan_renewal_date": None,
                "usage_pct": 0.0,
            }

        from sqlalchemy import select, func as sa_func
        from src.db.models import Subscription, UsageRecord, Customer

        async with self._db.session() as session:
            sub_result = await session.execute(
                select(Subscription)
                .where(Subscription.customer_id == customer_id, Subscription.is_active == True)  # noqa: E712
                .order_by(Subscription.created_at.desc())
                .limit(1)
            )
            subscription = sub_result.scalar_one_or_none()
            if subscription is None:
                return {
                    "plan_seconds_remaining": 0, "plan_seconds_total": 0,
                    "extra_seconds_remaining": 0, "total_remaining": 0,
                    "plan_renewal_date": None, "usage_pct": 100.0,
                }

            plan_total = subscription.monthly_seconds
            now = datetime.now(timezone.utc)
            month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

            usage_result = await session.execute(
                select(sa_func.coalesce(sa_func.sum(UsageRecord.duration_s), 0.0))
                .where(UsageRecord.customer_id == customer_id, UsageRecord.started_at >= month_start)
            )
            used_seconds = usage_result.scalar() or 0.0
            plan_remaining = max(0, plan_total - used_seconds)

            cust_result = await session.execute(
                select(Customer.extra_seconds_remaining).where(Customer.id == customer_id)
            )
            extra = cust_result.scalar() or 0

            if now.month == 12:
                renewal = now.replace(year=now.year + 1, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
            else:
                renewal = now.replace(month=now.month + 1, day=1, hour=0, minute=0, second=0, microsecond=0)

            usage_pct = round((used_seconds / plan_total) * 100, 1) if plan_total > 0 else 100.0

            return {
                "plan_seconds_remaining": round(plan_remaining),
                "plan_seconds_total": plan_total,
                "extra_seconds_remaining": extra,
                "total_remaining": round(plan_remaining) + extra,
                "plan_renewal_date": renewal.isoformat(),
                "usage_pct": min(100.0, usage_pct),
            }

    async def add_topup(self, customer_id: str, seconds: int) -> int:
        """Add top-up seconds to extra_seconds_remaining. Returns new total."""
        if self._db is None:
            return seconds

        from sqlalchemy import select, update
        from src.db.models import Customer

        async with self._db.session() as session:
            await session.execute(
                update(Customer).where(Customer.id == customer_id)
                .values(extra_seconds_remaining=Customer.extra_seconds_remaining + seconds)
            )
            await session.commit()
            result = await session.execute(
                select(Customer.extra_seconds_remaining).where(Customer.id == customer_id)
            )
            return result.scalar() or seconds

    async def check_balance_and_alert(
        self, customer_id: str, balance: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        """Check balance thresholds and return an alert if a new threshold is crossed.

        Thresholds: 20% remaining (warning), 5% (urgent), 0% with no extra (critical).
        Only fires once per threshold — resets when balance improves.

        Returns alert dict with level/message or None if no new alert.
        """
        if balance is None:
            balance = await self.get_balance(customer_id)

        plan_total = balance.get("plan_seconds_total", 0)
        plan_remaining = balance.get("plan_seconds_remaining", 0)
        extra = balance.get("extra_seconds_remaining", 0)

        if plan_total <= 0:
            return None

        pct_remaining = (plan_remaining / plan_total) * 100
        alert_level: Optional[str] = None
        message = ""

        if plan_remaining == 0 and extra == 0:
            alert_level = "critical"
            message = "Balance exhausted — service paused"
        elif pct_remaining <= 5:
            alert_level = "urgent"
            message = f"Only {pct_remaining:.1f}% plan seconds remaining"
        elif pct_remaining <= 20:
            alert_level = "warning"
            message = f"{pct_remaining:.1f}% plan seconds remaining"

        if alert_level is None:
            # Balance healthy — clear any previous alert state
            self._alerted.pop(customer_id, None)
            return None

        # Only fire if this is a new or escalated alert
        prev = self._alerted.get(customer_id)
        severity_order = {"warning": 0, "urgent": 1, "critical": 2}
        if prev and severity_order.get(prev, -1) >= severity_order.get(alert_level, -1):
            return None  # Already alerted at this level or higher

        self._alerted[customer_id] = alert_level

        alert = {
            "customer_id": customer_id,
            "level": alert_level,
            "alert_message": message,
            "pct_remaining": round(pct_remaining, 1),
            "plan_remaining": plan_remaining,
            "extra_remaining": extra,
        }
        logger.warning(
            f"Balance alert: {alert_level} — {message}",
            extra={
                "customer_id": customer_id,
                "alert_level": alert_level,
                "pct_remaining": round(pct_remaining, 1),
            },
        )
        return alert

    async def get_usage(
        self,
        customer_id: str,
        days: int = 30,
    ) -> list[dict[str, Any]]:
        """Get usage records for a customer within a date range."""
        if self._db is None:
            return []

        from sqlalchemy import select
        from src.db.models import UsageRecord
        from datetime import timedelta

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        async with self._db.session() as session:
            result = await session.execute(
                select(UsageRecord)
                .where(
                    UsageRecord.customer_id == customer_id,
                    UsageRecord.started_at >= cutoff,
                )
                .order_by(UsageRecord.started_at.desc())
            )
            records = result.scalars().all()
            return [
                {
                    "id": r.id,
                    "session_id": r.session_id,
                    "avatar_id": r.avatar_id,
                    "channel": r.channel,
                    "duration_s": r.duration_s,
                    "cost": r.cost,
                    "started_at": r.started_at.isoformat() if r.started_at else None,
                    "ended_at": r.ended_at.isoformat() if r.ended_at else None,
                }
                for r in records
            ]

    async def get_active_sessions(self) -> list[BillingSession]:
        """List all active billing sessions."""
        return list(self._active_sessions.values())

    async def check_concurrent_limit(self, customer_id: str) -> bool:
        """Check if the customer can start another concurrent session.

        Returns True if under the limit, False if at or over.
        """
        plan_info = await self.get_plan_info(customer_id)
        max_concurrent = plan_info.get("max_concurrent", 1)

        active_count = sum(
            1
            for s in self._active_sessions.values()
            if s.customer_id == customer_id
        )

        return active_count < max_concurrent

    async def get_plan_info(self, customer_id: str) -> dict[str, Any]:
        """Get the customer's active subscription plan info.

        Returns dict with plan name, monthly_seconds, max_avatars,
        max_concurrent, and price_monthly.
        """
        from src.config import PLAN_TIERS

        if self._db is None:
            # No DB — return starter defaults
            return PLAN_TIERS.get("starter", {})

        from sqlalchemy import select
        from src.db.models import Subscription

        async with self._db.session() as session:
            result = await session.execute(
                select(Subscription)
                .where(
                    Subscription.customer_id == customer_id,
                    Subscription.is_active == True,  # noqa: E712
                )
                .order_by(Subscription.created_at.desc())
                .limit(1)
            )
            subscription = result.scalar_one_or_none()

            if subscription is None:
                return {"plan": "none", "monthly_seconds": 0, "max_avatars": 0, "max_concurrent": 0, "price_monthly": 0}

            return {
                "plan": subscription.plan,
                "monthly_seconds": subscription.monthly_seconds,
                "max_avatars": subscription.max_avatars,
                "max_concurrent": subscription.max_concurrent_sessions,
                "price_monthly": subscription.price_monthly,
            }

    async def _write_usage_record(self, billing_session: BillingSession) -> None:
        """Persist a usage record to the database."""
        from src.db.models import UsageRecord

        async with self._db.session() as session:
            record = UsageRecord(
                customer_id=billing_session.customer_id,
                session_id=billing_session.session_id,
                avatar_id=billing_session.avatar_id,
                channel=billing_session.channel,
                duration_s=billing_session.total_seconds,
                cost=billing_session.total_cost,
                started_at=datetime.fromtimestamp(billing_session.started_at, tz=timezone.utc),
                ended_at=datetime.fromtimestamp(billing_session.stopped_at, tz=timezone.utc),
            )
            session.add(record)
            await session.commit()
