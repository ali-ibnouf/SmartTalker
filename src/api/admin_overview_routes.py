"""Admin overview routes — cross-customer training & knowledge stats.

Routes:
    GET /api/v1/admin/training/overview    Aggregated training stats
    GET /api/v1/admin/knowledge/overview   Aggregated KB stats
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from sqlalchemy import case, func, select

from src.api.schemas import (
    AdminKnowledgeCategoryItem,
    AdminKnowledgeCustomerRow,
    AdminKnowledgeOverview,
    AdminTrainingCustomerRow,
    AdminTrainingOverview,
    AdminTrainingRecentItem,
)
from src.db.models import Customer, Employee, EmployeeKnowledge, EmployeeLearning
from src.utils.logger import setup_logger

logger = setup_logger("api.admin_overview")
router = APIRouter(prefix="/api/v1", tags=["admin-overview"])


def _get_db(request: Request):
    db = getattr(request.app.state, "db", None)
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    return db


# ---------------------------------------------------------------------------
# GET /admin/training/overview
# ---------------------------------------------------------------------------


@router.get("/admin/training/overview", response_model=AdminTrainingOverview)
async def admin_training_overview(request: Request) -> AdminTrainingOverview:
    """Aggregated training stats across ALL customers."""
    db = _get_db(request)

    async with db.session() as session:
        # Global totals by status
        status_q = await session.execute(
            select(
                EmployeeLearning.status,
                func.count(EmployeeLearning.id),
            ).group_by(EmployeeLearning.status)
        )
        totals: dict[str, int] = {}
        for status, count in status_q:
            totals[status] = count

        # Global avg confidence of pending
        avg_q = await session.execute(
            select(func.avg(EmployeeLearning.confidence)).where(
                EmployeeLearning.status == "pending"
            )
        )
        avg_confidence = round(float(avg_q.scalar() or 0), 4)

        # Per-customer breakdown
        customer_q = await session.execute(
            select(
                EmployeeLearning.customer_id,
                Customer.name,
                func.count(case((EmployeeLearning.status == "pending", 1))).label("pending"),
                func.count(case((EmployeeLearning.status == "approved", 1))).label("approved"),
                func.count(case((EmployeeLearning.status == "rejected", 1))).label("rejected"),
                func.avg(
                    case((EmployeeLearning.status == "pending", EmployeeLearning.confidence))
                ).label("avg_conf"),
            )
            .join(Customer, EmployeeLearning.customer_id == Customer.id)
            .group_by(EmployeeLearning.customer_id, Customer.name)
            .order_by(func.count(case((EmployeeLearning.status == "pending", 1))).desc())
        )
        customers = [
            AdminTrainingCustomerRow(
                customer_id=row.customer_id,
                customer_name=row.name,
                pending=row.pending,
                approved=row.approved,
                rejected=row.rejected,
                avg_confidence=round(float(row.avg_conf or 0), 4),
            )
            for row in customer_q
        ]

        # Recent 10 pending entries
        recent_q = await session.execute(
            select(EmployeeLearning, Customer.name)
            .join(Customer, EmployeeLearning.customer_id == Customer.id)
            .where(EmployeeLearning.status == "pending")
            .order_by(EmployeeLearning.created_at.desc())
            .limit(10)
        )
        recent = [
            AdminTrainingRecentItem(
                id=entry.id,
                customer_id=entry.customer_id,
                customer_name=cname,
                employee_id=entry.employee_id,
                learning_type=entry.learning_type,
                old_value=entry.old_value[:200] if entry.old_value else "",
                new_value=entry.new_value[:200] if entry.new_value else "",
                confidence=entry.confidence,
                created_at=entry.created_at.isoformat() if entry.created_at else "",
            )
            for entry, cname in recent_q
        ]

    return AdminTrainingOverview(
        total_pending=totals.get("pending", 0),
        total_approved=totals.get("approved", 0),
        total_rejected=totals.get("rejected", 0),
        avg_confidence=avg_confidence,
        customers=customers,
        recent_pending=recent,
    )


# ---------------------------------------------------------------------------
# GET /admin/knowledge/overview
# ---------------------------------------------------------------------------


@router.get("/admin/knowledge/overview", response_model=AdminKnowledgeOverview)
async def admin_knowledge_overview(request: Request) -> AdminKnowledgeOverview:
    """Aggregated KB stats across ALL customers."""
    db = _get_db(request)

    async with db.session() as session:
        # Total knowledge items & avg success rate
        totals_q = await session.execute(
            select(
                func.count(EmployeeKnowledge.id),
                func.avg(EmployeeKnowledge.success_rate),
            )
        )
        total_items, avg_rate = totals_q.one()
        total_items = total_items or 0
        avg_rate = round(float(avg_rate or 0), 2)

        # Per-customer breakdown
        customer_q = await session.execute(
            select(
                Employee.customer_id,
                Customer.name,
                func.count(EmployeeKnowledge.id).label("count"),
                func.avg(EmployeeKnowledge.success_rate).label("avg_rate"),
            )
            .join(Employee, EmployeeKnowledge.employee_id == Employee.id)
            .join(Customer, Employee.customer_id == Customer.id)
            .group_by(Employee.customer_id, Customer.name)
            .order_by(func.count(EmployeeKnowledge.id).desc())
        )
        customers = [
            AdminKnowledgeCustomerRow(
                customer_id=row.customer_id,
                customer_name=row.name,
                knowledge_count=row.count,
                avg_success_rate=round(float(row.avg_rate or 0), 2),
            )
            for row in customer_q
        ]

        # Category distribution
        cat_q = await session.execute(
            select(
                EmployeeKnowledge.category,
                func.count(EmployeeKnowledge.id),
            ).group_by(EmployeeKnowledge.category)
            .order_by(func.count(EmployeeKnowledge.id).desc())
        )
        categories = [
            AdminKnowledgeCategoryItem(category=cat, count=cnt)
            for cat, cnt in cat_q
        ]

    return AdminKnowledgeOverview(
        total_knowledge_items=total_items,
        avg_success_rate=avg_rate,
        total_categories=len(categories),
        customers=customers,
        categories=categories,
    )
