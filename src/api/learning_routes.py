"""CRUD API for learning queue management.

Routes:
    GET    /api/v1/learning/queue              List pending learning entries
    POST   /api/v1/learning/queue/{id}/approve  Approve a learning entry
    POST   /api/v1/learning/queue/{id}/reject   Reject a learning entry
    PUT    /api/v1/learning/queue/{id}          Edit a learning entry
    GET    /api/v1/learning/stats               Learning queue statistics
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request
from sqlalchemy import func, select, update

from src.api.schemas import (
    LearningQueueItem,
    LearningQueueResponse,
    LearningReviewRequest,
    LearningStatsResponse,
)
from src.db.models import EmployeeKnowledge, EmployeeLearning
from src.utils.logger import setup_logger

logger = setup_logger("api.learning")
router = APIRouter(prefix="/api/v1", tags=["learning"])


def _get_db(request: Request):
    db = getattr(request.app.state, "db", None)
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    return db


def _get_customer_id(request: Request) -> str:
    return getattr(request.state, "customer_id", "")


def _entry_to_item(entry: EmployeeLearning) -> LearningQueueItem:
    """Convert an EmployeeLearning ORM object to a response schema."""
    return LearningQueueItem(
        id=entry.id,
        employee_id=entry.employee_id,
        customer_id=entry.customer_id,
        session_id=entry.session_id,
        learning_type=entry.learning_type,
        old_value=entry.old_value,
        new_value=entry.new_value,
        confidence=entry.confidence,
        status=entry.status,
        source=entry.source,
        created_at=entry.created_at.isoformat() if entry.created_at else None,
    )


# -- List Learning Queue ---------------------------------------------------


@router.get("/learning/queue", response_model=LearningQueueResponse)
async def list_learning_queue(
    request: Request,
    employee_id: Optional[str] = Query(default=None, description="Filter by employee ID"),
    limit: int = Query(default=50, ge=1, le=200, description="Max items to return"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
):
    """List pending learning entries for the authenticated customer."""
    db = _get_db(request)
    customer_id = _get_customer_id(request)

    async with db.session() as session:
        query = (
            select(EmployeeLearning)
            .where(
                EmployeeLearning.customer_id == customer_id,
                EmployeeLearning.status == "pending",
            )
        )

        if employee_id is not None:
            query = query.where(EmployeeLearning.employee_id == employee_id)

        query = (
            query
            .order_by(EmployeeLearning.created_at.desc())
            .offset(offset)
            .limit(limit)
        )

        result = await session.execute(query)
        entries = result.scalars().all()

        # Total count for pagination
        count_query = (
            select(func.count(EmployeeLearning.id))
            .where(
                EmployeeLearning.customer_id == customer_id,
                EmployeeLearning.status == "pending",
            )
        )
        if employee_id is not None:
            count_query = count_query.where(EmployeeLearning.employee_id == employee_id)

        count_result = await session.execute(count_query)
        total = count_result.scalar() or 0

    return LearningQueueResponse(
        items=[_entry_to_item(e) for e in entries],
        count=total,
    )


# -- Approve Learning Entry ------------------------------------------------


@router.post("/learning/queue/{entry_id}/approve", response_model=LearningQueueItem)
async def approve_learning_entry(
    entry_id: str,
    request: Request,
    body: Optional[LearningReviewRequest] = None,
):
    """Approve a learning queue entry.

    If the entry is a qa_pair, a new EmployeeKnowledge record is created
    using old_value as the question and new_value (or edited_value from
    the request body) as the answer.
    """
    db = _get_db(request)
    customer_id = _get_customer_id(request)

    async with db.session() as session:
        result = await session.execute(
            select(EmployeeLearning).where(
                EmployeeLearning.id == entry_id,
                EmployeeLearning.customer_id == customer_id,
            )
        )
        entry = result.scalar_one_or_none()
        if not entry:
            raise HTTPException(status_code=404, detail="Learning entry not found")

        if entry.status != "pending":
            raise HTTPException(
                status_code=409,
                detail=f"Entry already {entry.status}",
            )

        # Apply edited_value if provided
        if body and body.edited_value is not None:
            entry.new_value = body.edited_value

        entry.status = "approved"
        entry.reviewed_at = datetime.utcnow()

        # If qa_pair, create an EmployeeKnowledge entry
        if entry.learning_type == "qa_pair":
            question = entry.old_value
            answer = entry.new_value

            # Attempt JSON parse in case values are stored as JSON strings
            try:
                parsed = json.loads(question)
                if isinstance(parsed, str):
                    question = parsed
            except (json.JSONDecodeError, TypeError):
                pass

            try:
                parsed = json.loads(answer)
                if isinstance(parsed, str):
                    answer = parsed
            except (json.JSONDecodeError, TypeError):
                pass

            knowledge = EmployeeKnowledge(
                employee_id=entry.employee_id,
                category="learned",
                question=question,
                answer=answer,
                approved=True,
            )
            session.add(knowledge)

        await session.commit()
        await session.refresh(entry)

    logger.info(
        "Learning entry approved",
        extra={"entry_id": entry_id, "learning_type": entry.learning_type},
    )
    return _entry_to_item(entry)


# -- Reject Learning Entry -------------------------------------------------


@router.post("/learning/queue/{entry_id}/reject", response_model=LearningQueueItem)
async def reject_learning_entry(entry_id: str, request: Request):
    """Reject a learning queue entry."""
    db = _get_db(request)
    customer_id = _get_customer_id(request)

    async with db.session() as session:
        result = await session.execute(
            select(EmployeeLearning).where(
                EmployeeLearning.id == entry_id,
                EmployeeLearning.customer_id == customer_id,
            )
        )
        entry = result.scalar_one_or_none()
        if not entry:
            raise HTTPException(status_code=404, detail="Learning entry not found")

        if entry.status != "pending":
            raise HTTPException(
                status_code=409,
                detail=f"Entry already {entry.status}",
            )

        entry.status = "rejected"
        entry.reviewed_at = datetime.utcnow()

        await session.commit()
        await session.refresh(entry)

    logger.info("Learning entry rejected", extra={"entry_id": entry_id})
    return _entry_to_item(entry)


# -- Edit Learning Entry ---------------------------------------------------


@router.put("/learning/queue/{entry_id}", response_model=LearningQueueItem)
async def edit_learning_entry(
    entry_id: str, body: LearningReviewRequest, request: Request
):
    """Edit a learning entry's new_value before approving."""
    db = _get_db(request)
    customer_id = _get_customer_id(request)

    async with db.session() as session:
        result = await session.execute(
            select(EmployeeLearning).where(
                EmployeeLearning.id == entry_id,
                EmployeeLearning.customer_id == customer_id,
            )
        )
        entry = result.scalar_one_or_none()
        if not entry:
            raise HTTPException(status_code=404, detail="Learning entry not found")

        if entry.status != "pending":
            raise HTTPException(
                status_code=409,
                detail=f"Entry already {entry.status}",
            )

        if body.edited_value is not None:
            entry.new_value = body.edited_value

        await session.commit()
        await session.refresh(entry)

    logger.info("Learning entry edited", extra={"entry_id": entry_id})
    return _entry_to_item(entry)


# -- Learning Stats --------------------------------------------------------


@router.get("/learning/stats", response_model=LearningStatsResponse)
async def get_learning_stats(request: Request):
    """Return learning queue statistics for the authenticated customer."""
    db = _get_db(request)
    customer_id = _get_customer_id(request)

    async with db.session() as session:
        # Count by status
        status_counts = await session.execute(
            select(
                EmployeeLearning.status,
                func.count(EmployeeLearning.id),
            )
            .where(EmployeeLearning.customer_id == customer_id)
            .group_by(EmployeeLearning.status)
        )
        counts: dict[str, int] = {}
        for status, count in status_counts:
            counts[status] = count

        # Count auto-approved (approved entries with source="auto")
        auto_approved_result = await session.execute(
            select(func.count(EmployeeLearning.id)).where(
                EmployeeLearning.customer_id == customer_id,
                EmployeeLearning.status == "approved",
                EmployeeLearning.source == "auto",
            )
        )
        auto_approved_count = auto_approved_result.scalar() or 0

        # Average confidence of pending items
        avg_conf_result = await session.execute(
            select(func.avg(EmployeeLearning.confidence)).where(
                EmployeeLearning.customer_id == customer_id,
                EmployeeLearning.status == "pending",
            )
        )
        avg_confidence = avg_conf_result.scalar() or 0.0

    return LearningStatsResponse(
        pending_count=counts.get("pending", 0),
        approved_count=counts.get("approved", 0),
        rejected_count=counts.get("rejected", 0),
        auto_approved_count=auto_approved_count,
        avg_confidence=round(avg_confidence, 4),
    )
