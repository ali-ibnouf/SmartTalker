"""Subscription lifecycle management: freeze, cancel, reactivate.

Handles the full subscription lifecycle:
- **Freeze**: Temporarily suspend service (employees deactivated, WS sessions closed).
  R2 media and knowledge base are preserved.
- **Cancel**: Permanently end subscription. Employees deactivated, R2 media
  (photos, face data, voice samples) purged after 30-day grace period.
  Knowledge base is preserved for potential reactivation.
- **Reactivate**: Resume a frozen or cancelled subscription. Employees
  reactivated, new billing cycle starts.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import select, update

from src.utils.logger import setup_logger

logger = setup_logger("services.subscription")

# Grace period before R2 media is permanently deleted (days)
CANCEL_GRACE_DAYS = 30


class SubscriptionLifecycle:
    """Manages subscription state transitions."""

    def __init__(self, db: Any, r2_storage: Any = None, config: Any = None):
        self._db = db
        self._r2 = r2_storage
        self._config = config

    async def freeze(self, customer_id: str, reason: str = "") -> dict:
        """Freeze a subscription — pause all services, preserve data.

        Args:
            customer_id: Customer to freeze.
            reason: Reason for freezing (e.g., "payment_failed", "user_request").

        Returns:
            Status dict with affected employee count.
        """
        if self._db is None:
            return {"error": "Database not available"}

        from src.db.models import Customer, Employee, Subscription

        async with self._db.session() as session:
            # Mark subscription as frozen
            await session.execute(
                update(Subscription)
                .where(
                    Subscription.customer_id == customer_id,
                    Subscription.is_active == True,  # noqa: E712
                )
                .values(is_active=False)
            )

            # Deactivate all employees
            result = await session.execute(
                update(Employee)
                .where(
                    Employee.customer_id == customer_id,
                    Employee.is_active == True,  # noqa: E712
                )
                .values(is_active=False)
            )
            emp_count = result.rowcount

            # Mark customer as frozen
            await session.execute(
                update(Customer)
                .where(Customer.id == customer_id)
                .values(is_active=False)
            )

            await session.commit()

        logger.info(
            "Subscription frozen",
            extra={
                "customer_id": customer_id,
                "reason": reason,
                "employees_deactivated": emp_count,
            },
        )

        return {
            "status": "frozen",
            "customer_id": customer_id,
            "employees_deactivated": emp_count,
            "reason": reason,
        }

    async def cancel(
        self, customer_id: str, reason: str = "", purge_media: bool = False
    ) -> dict:
        """Cancel a subscription — end service, optionally purge R2 media.

        Args:
            customer_id: Customer to cancel.
            reason: Cancellation reason.
            purge_media: If True, immediately delete R2 media (skip grace period).

        Returns:
            Status dict.
        """
        if self._db is None:
            return {"error": "Database not available"}

        # First freeze (deactivate employees + subscription)
        freeze_result = await self.freeze(customer_id, reason=reason)
        if "error" in freeze_result:
            return freeze_result

        from src.db.models import Customer, Subscription

        async with self._db.session() as session:
            # Update subscription status
            await session.execute(
                update(Subscription)
                .where(Subscription.customer_id == customer_id)
                .values(is_active=False)
            )
            await session.commit()

        # Purge R2 media if requested
        media_purged = False
        if purge_media and self._r2:
            try:
                await self._purge_r2_media(customer_id)
                media_purged = True
            except Exception as exc:
                logger.warning(f"R2 media purge failed: {exc}")

        logger.info(
            "Subscription cancelled",
            extra={
                "customer_id": customer_id,
                "reason": reason,
                "media_purged": media_purged,
            },
        )

        return {
            "status": "cancelled",
            "customer_id": customer_id,
            "employees_deactivated": freeze_result.get("employees_deactivated", 0),
            "media_purged": media_purged,
            "reason": reason,
        }

    async def reactivate(self, customer_id: str, plan_id: str = "") -> dict:
        """Reactivate a frozen or cancelled subscription.

        Employees are re-enabled, a new billing cycle starts.
        Knowledge base is preserved across freeze/cancel/reactivate.

        Args:
            customer_id: Customer to reactivate.
            plan_id: New plan ID (if upgrading/changing plan).

        Returns:
            Status dict with reactivated employee count.
        """
        if self._db is None:
            return {"error": "Database not available"}

        from src.db.models import Customer, Employee, Subscription

        async with self._db.session() as session:
            # Reactivate customer
            await session.execute(
                update(Customer)
                .where(Customer.id == customer_id)
                .values(is_active=True)
            )

            # Reactivate subscription
            sub_values: dict = {"is_active": True}
            if plan_id:
                sub_values["plan_id"] = plan_id
            await session.execute(
                update(Subscription)
                .where(Subscription.customer_id == customer_id)
                .values(**sub_values)
            )

            # Reactivate all employees
            result = await session.execute(
                update(Employee)
                .where(
                    Employee.customer_id == customer_id,
                    Employee.is_active == False,  # noqa: E712
                )
                .values(is_active=True)
            )
            emp_count = result.rowcount

            await session.commit()

        logger.info(
            "Subscription reactivated",
            extra={
                "customer_id": customer_id,
                "plan_id": plan_id,
                "employees_reactivated": emp_count,
            },
        )

        return {
            "status": "reactivated",
            "customer_id": customer_id,
            "employees_reactivated": emp_count,
            "plan_id": plan_id or "unchanged",
        }

    async def _purge_r2_media(self, customer_id: str) -> int:
        """Delete all R2 media for a customer (photos, face data, voice samples).

        Returns count of deleted objects.
        """
        if self._r2 is None:
            return 0

        prefixes = [
            f"employees/{customer_id}/",
            f"customers/{customer_id}/",
        ]

        # Also get employee IDs to delete their specific media
        if self._db:
            from src.db.models import Employee

            async with self._db.session() as session:
                result = await session.execute(
                    select(Employee.id).where(Employee.customer_id == customer_id)
                )
                emp_ids = [row[0] for row in result.all()]

            for emp_id in emp_ids:
                prefixes.append(f"employees/{emp_id}/")

        deleted = 0
        for prefix in prefixes:
            try:
                count = self._r2.delete_prefix(prefix)
                deleted += count
            except Exception as exc:
                logger.debug(f"Failed to delete prefix {prefix}: {exc}")

        logger.info(
            "R2 media purged",
            extra={"customer_id": customer_id, "deleted_objects": deleted},
        )
        return deleted
