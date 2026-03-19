"""Admin approval queue for destructive agent actions.

Actions like suspending a customer, killing service, or deleting data
must be approved by a human admin before execution.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any, Optional

from src.utils.logger import setup_logger

logger = setup_logger("ai_agent.approval")

# Action types that require admin approval before execution
APPROVAL_ACTIONS = {
    "suspend_customer",
    "kill_switch",
    "plan_downgrade",
    "data_deletion",
}

# Auto-fix IDs that affect global pipeline state or external resources.
# These bypass the normal auto-fix path and go to the approval queue instead.
HIGH_IMPACT_FIX_ACTIONS = {
    "video_disable",            # Disables GPU rendering for ALL customers
    "text_only_mode",           # Forces text-only mode for ALL customers
    "db_connection_cleanup",    # Kills PostgreSQL connections
}


class ApprovalQueue:
    """Manages the approval workflow for destructive agent actions."""

    def __init__(self, db: Any, config: Any = None) -> None:
        self._db = db
        self._config = config

    async def request_approval(
        self,
        action_type: str,
        target_id: str,
        description: str,
        details: dict[str, Any] | None = None,
        requested_by: str = "agent",
        expires_hours: int | None = None,
    ) -> Optional[str]:
        """Create a pending approval request. Returns approval ID."""
        if self._db is None:
            return None

        if expires_hours is None:
            expires_hours = 24
            if self._config:
                expires_hours = getattr(self._config, "approval_expiry_hours", 24)

        try:
            from src.db.models import AgentApproval

            # Check for duplicate pending requests
            from sqlalchemy import select

            async with self._db.session_ctx() as session:
                existing = await session.execute(
                    select(AgentApproval.id)
                    .where(
                        AgentApproval.action_type == action_type,
                        AgentApproval.target_id == target_id,
                        AgentApproval.status == "pending",
                    )
                )
                if existing.scalar():
                    logger.debug(
                        f"Duplicate approval request skipped: {action_type} for {target_id}"
                    )
                    return None

            async with self._db.session_ctx() as session:
                approval = AgentApproval(
                    action_type=action_type,
                    target_id=target_id,
                    description=description,
                    details=json.dumps(details or {}),
                    status="pending",
                    requested_by=requested_by,
                    expires_at=datetime.utcnow() + timedelta(hours=expires_hours),
                )
                session.add(approval)
                await session.flush()
                approval_id = approval.id

            logger.info(
                "Approval requested",
                extra={
                    "approval_id": approval_id,
                    "action_type": action_type,
                    "target_id": target_id,
                },
            )
            return approval_id

        except Exception as exc:
            logger.error(f"Failed to create approval request: {exc}")
            return None

    async def approve(self, approval_id: str, reviewed_by: str) -> dict[str, Any]:
        """Approve a request and execute the associated action."""
        if self._db is None:
            return {"error": "Database not available"}

        try:
            from sqlalchemy import select, update
            from src.db.models import AgentApproval

            async with self._db.session_ctx() as session:
                r = await session.execute(
                    select(AgentApproval).where(AgentApproval.id == approval_id)
                )
                approval = r.scalar_one_or_none()

                if approval is None:
                    return {"error": "Approval not found"}
                if approval.status != "pending":
                    return {"error": f"Approval already {approval.status}"}

                # Mark approved
                await session.execute(
                    update(AgentApproval)
                    .where(AgentApproval.id == approval_id)
                    .values(
                        status="approved",
                        reviewed_by=reviewed_by,
                        reviewed_at=datetime.utcnow(),
                    )
                )

                action_type = approval.action_type
                target_id = approval.target_id
                details = json.loads(approval.details) if approval.details else {}

            # Execute the approved action
            result = await self._execute_action(action_type, target_id, details)

            logger.info(
                "Approval approved and executed",
                extra={
                    "approval_id": approval_id,
                    "action_type": action_type,
                    "reviewed_by": reviewed_by,
                },
            )

            return {
                "approval_id": approval_id,
                "status": "approved",
                "action_type": action_type,
                "target_id": target_id,
                "execution_result": result,
            }

        except Exception as exc:
            logger.error(f"Failed to approve: {exc}")
            return {"error": str(exc)}

    async def reject(self, approval_id: str, reviewed_by: str) -> dict[str, Any]:
        """Reject a pending approval request."""
        if self._db is None:
            return {"error": "Database not available"}

        try:
            from sqlalchemy import select, update
            from src.db.models import AgentApproval

            async with self._db.session_ctx() as session:
                r = await session.execute(
                    select(AgentApproval.status).where(AgentApproval.id == approval_id)
                )
                status = r.scalar_one_or_none()
                if status is None:
                    return {"error": "Approval not found"}
                if status != "pending":
                    return {"error": f"Approval already {status}"}

                await session.execute(
                    update(AgentApproval)
                    .where(AgentApproval.id == approval_id)
                    .values(
                        status="rejected",
                        reviewed_by=reviewed_by,
                        reviewed_at=datetime.utcnow(),
                    )
                )

            logger.info(
                "Approval rejected",
                extra={"approval_id": approval_id, "reviewed_by": reviewed_by},
            )
            return {"approval_id": approval_id, "status": "rejected"}

        except Exception as exc:
            logger.error(f"Failed to reject: {exc}")
            return {"error": str(exc)}

    async def list_pending(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return all pending approval requests."""
        if self._db is None:
            return []

        try:
            from sqlalchemy import select
            from src.db.models import AgentApproval

            async with self._db.session_ctx() as session:
                r = await session.execute(
                    select(AgentApproval)
                    .where(AgentApproval.status == "pending")
                    .order_by(AgentApproval.created_at.desc())
                    .limit(limit)
                )
                approvals = r.scalars().all()
                return [
                    {
                        "id": a.id,
                        "action_type": a.action_type,
                        "target_id": a.target_id,
                        "description": a.description,
                        "details": json.loads(a.details) if a.details else {},
                        "status": a.status,
                        "requested_by": a.requested_by,
                        "created_at": a.created_at.isoformat() if a.created_at else None,
                        "expires_at": a.expires_at.isoformat() if a.expires_at else None,
                    }
                    for a in approvals
                ]

        except Exception as exc:
            logger.error(f"Failed to list approvals: {exc}")
            return []

    async def expire_stale(self) -> int:
        """Mark expired pending approvals. Called by agent loop."""
        if self._db is None:
            return 0

        try:
            from sqlalchemy import update
            from src.db.models import AgentApproval

            now = datetime.utcnow()

            async with self._db.session_ctx() as session:
                result = await session.execute(
                    update(AgentApproval)
                    .where(
                        AgentApproval.status == "pending",
                        AgentApproval.expires_at <= now,
                    )
                    .values(status="expired")
                )
                expired = result.rowcount

            if expired:
                logger.info(f"Expired {expired} stale approval requests")
            return expired

        except Exception as exc:
            logger.error(f"Failed to expire approvals: {exc}")
            return 0

    async def _execute_action(
        self, action_type: str, target_id: str, details: dict
    ) -> dict[str, Any]:
        """Execute an approved action."""
        try:
            if action_type == "suspend_customer":
                from src.services.subscription import SubscriptionLifecycle

                lifecycle = SubscriptionLifecycle(db=self._db)
                return await lifecycle.freeze(
                    target_id, reason=details.get("reason", "agent_approved")
                )

            elif action_type == "kill_switch":
                from sqlalchemy import update
                from src.db.models import Customer

                async with self._db.session_ctx() as session:
                    await session.execute(
                        update(Customer)
                        .where(Customer.id == target_id)
                        .values(suspended=True)
                    )
                return {"action": "kill_switch", "customer_id": target_id, "suspended": True}

            elif action_type == "plan_downgrade":
                from sqlalchemy import update
                from src.db.models import Subscription

                new_plan = details.get("new_plan", "starter")
                async with self._db.session_ctx() as session:
                    await session.execute(
                        update(Subscription)
                        .where(
                            Subscription.customer_id == target_id,
                            Subscription.is_active == True,  # noqa: E712
                        )
                        .values(plan=new_plan)
                    )
                return {"action": "plan_downgrade", "customer_id": target_id, "new_plan": new_plan}

            elif action_type == "data_deletion":
                from src.services.subscription import SubscriptionLifecycle

                lifecycle = SubscriptionLifecycle(db=self._db)
                count = await lifecycle._purge_r2_media(target_id)
                return {"action": "data_deletion", "customer_id": target_id, "deleted_objects": count}

            else:
                return {"error": f"Unknown action type: {action_type}"}

        except Exception as exc:
            logger.error(f"Action execution failed: {action_type}: {exc}")
            return {"error": str(exc)}
