"""Onboarding status and progression API routes.

Routes:
    GET  /api/v1/onboarding/status   Get onboarding step for the customer
    POST /api/v1/onboarding/advance  Advance to the next onboarding step
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy import func, select

from src.db.models import Avatar, Conversation, Employee, EmployeeKnowledge, Subscription
from src.utils.logger import setup_logger

logger = setup_logger("api.onboarding")

router = APIRouter(prefix="/api/v1/onboarding", tags=["onboarding"])

# Onboarding steps in order.  The status endpoint returns the first
# incomplete step so the frontend knows what to show.
ONBOARDING_STEPS = [
    "subscription",      # 1. Customer has an active subscription
    "create_employee",   # 2. At least one employee created
    "upload_avatar",     # 3. At least one avatar with a photo
    "add_knowledge",     # 4. At least one KB entry added
    "first_conversation",  # 5. At least one conversation completed
]


def _get_db(request: Request):
    db = getattr(request.app.state, "db", None)
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    return db


def _get_customer_id(request: Request) -> str:
    return getattr(request.state, "customer_id", "")


@router.get("/status")
async def get_onboarding_status(request: Request):
    """Return the current onboarding step for the authenticated customer.

    Checks each step in order and returns the first one that is not yet
    completed, along with the full checklist of completed/pending steps.
    """
    db = _get_db(request)
    customer_id = _get_customer_id(request)
    if not customer_id:
        raise HTTPException(status_code=401, detail="Customer not identified")

    try:
        async with db.session_ctx() as session:
            # Step 1: active subscription
            sub_result = await session.execute(
                select(func.count()).select_from(Subscription).where(
                    Subscription.customer_id == customer_id,
                    Subscription.is_active == True,  # noqa: E712
                )
            )
            has_subscription = (sub_result.scalar() or 0) > 0

            # Step 2: at least one employee
            emp_result = await session.execute(
                select(func.count()).select_from(Employee).where(
                    Employee.customer_id == customer_id,
                    Employee.is_active == True,  # noqa: E712
                )
            )
            has_employee = (emp_result.scalar() or 0) > 0

            # Step 3: at least one avatar with photo
            av_result = await session.execute(
                select(func.count()).select_from(Avatar).where(
                    Avatar.customer_id == customer_id,
                    Avatar.photo_url != "",
                )
            )
            has_avatar = (av_result.scalar() or 0) > 0

            # Step 4: at least one knowledge entry
            if has_employee:
                kb_result = await session.execute(
                    select(func.count())
                    .select_from(EmployeeKnowledge)
                    .join(Employee, Employee.id == EmployeeKnowledge.employee_id)
                    .where(Employee.customer_id == customer_id)
                )
                has_knowledge = (kb_result.scalar() or 0) > 0
            else:
                has_knowledge = False

            # Step 5: at least one conversation
            conv_result = await session.execute(
                select(func.count())
                .select_from(Conversation)
                .join(Avatar, Avatar.id == Conversation.avatar_id)
                .where(Avatar.customer_id == customer_id)
            )
            has_conversation = (conv_result.scalar() or 0) > 0

        step_status = {
            "subscription": has_subscription,
            "create_employee": has_employee,
            "upload_avatar": has_avatar,
            "add_knowledge": has_knowledge,
            "first_conversation": has_conversation,
        }

        # Find first incomplete step
        current_step = "complete"
        for step in ONBOARDING_STEPS:
            if not step_status[step]:
                current_step = step
                break

        completed_count = sum(1 for v in step_status.values() if v)

        return {
            "customer_id": customer_id,
            "current_step": current_step,
            "is_complete": current_step == "complete",
            "completed_count": completed_count,
            "total_steps": len(ONBOARDING_STEPS),
            "steps": step_status,
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Onboarding status failed: {exc}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to fetch onboarding status", "detail": str(exc)},
        )


@router.post("/advance")
async def advance_onboarding(request: Request):
    """Advance the onboarding state for the customer.

    This is an idempotent signal — the frontend calls it after completing
    a step.  The endpoint re-evaluates onboarding progress and returns
    the updated status (same shape as GET /status).

    The body is intentionally empty; progression is derived from actual
    data presence rather than a client-declared step.
    """
    db = _get_db(request)
    customer_id = _get_customer_id(request)
    if not customer_id:
        raise HTTPException(status_code=401, detail="Customer not identified")

    # Re-use the status logic — advance is data-driven
    try:
        # Build a fresh request-like call to the status endpoint
        # by just duplicating the same logic inline (keeps it simple).
        async with db.session_ctx() as session:
            sub_result = await session.execute(
                select(func.count()).select_from(Subscription).where(
                    Subscription.customer_id == customer_id,
                    Subscription.is_active == True,  # noqa: E712
                )
            )
            has_subscription = (sub_result.scalar() or 0) > 0

            emp_result = await session.execute(
                select(func.count()).select_from(Employee).where(
                    Employee.customer_id == customer_id,
                    Employee.is_active == True,  # noqa: E712
                )
            )
            has_employee = (emp_result.scalar() or 0) > 0

            av_result = await session.execute(
                select(func.count()).select_from(Avatar).where(
                    Avatar.customer_id == customer_id,
                    Avatar.photo_url != "",
                )
            )
            has_avatar = (av_result.scalar() or 0) > 0

            if has_employee:
                kb_result = await session.execute(
                    select(func.count())
                    .select_from(EmployeeKnowledge)
                    .join(Employee, Employee.id == EmployeeKnowledge.employee_id)
                    .where(Employee.customer_id == customer_id)
                )
                has_knowledge = (kb_result.scalar() or 0) > 0
            else:
                has_knowledge = False

            conv_result = await session.execute(
                select(func.count())
                .select_from(Conversation)
                .join(Avatar, Avatar.id == Conversation.avatar_id)
                .where(Avatar.customer_id == customer_id)
            )
            has_conversation = (conv_result.scalar() or 0) > 0

        step_status = {
            "subscription": has_subscription,
            "create_employee": has_employee,
            "upload_avatar": has_avatar,
            "add_knowledge": has_knowledge,
            "first_conversation": has_conversation,
        }

        current_step = "complete"
        for step in ONBOARDING_STEPS:
            if not step_status[step]:
                current_step = step
                break

        completed_count = sum(1 for v in step_status.values() if v)

        logger.info(
            "Onboarding advance",
            extra={
                "customer_id": customer_id,
                "current_step": current_step,
                "completed": completed_count,
            },
        )

        return {
            "customer_id": customer_id,
            "current_step": current_step,
            "is_complete": current_step == "complete",
            "completed_count": completed_count,
            "total_steps": len(ONBOARDING_STEPS),
            "steps": step_status,
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Onboarding advance failed: {exc}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to advance onboarding", "detail": str(exc)},
        )
