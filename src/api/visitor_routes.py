"""Visitor profile and memory API routes.

Routes:
    GET /api/v1/visitors/{visitor_id}/profile  Get visitor profile
    GET /api/v1/visitors/{visitor_id}/memory   Get visitor memory entries
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy import select

from src.db.models import VisitorMemory, VisitorProfile
from src.utils.logger import setup_logger

logger = setup_logger("api.visitors")

router = APIRouter(prefix="/api/v1", tags=["visitors"])


def _get_db(request: Request):
    db = getattr(request.app.state, "db", None)
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    return db


def _get_customer_id(request: Request) -> str:
    return getattr(request.state, "customer_id", "")


def _profile_to_dict(profile: VisitorProfile) -> dict:
    """Convert a VisitorProfile ORM object to a JSON-serializable dict."""
    return {
        "id": profile.id,
        "visitor_id": profile.visitor_id,
        "employee_id": profile.employee_id,
        "customer_id": profile.customer_id,
        "display_name": profile.display_name,
        "email": profile.email,
        "phone": profile.phone,
        "language": profile.language,
        "tags": profile.tags,
        "interaction_count": profile.interaction_count,
        "last_seen": profile.last_seen.isoformat() if profile.last_seen else None,
        "created_at": profile.created_at.isoformat() if profile.created_at else None,
        "updated_at": profile.updated_at.isoformat() if profile.updated_at else None,
    }


def _memory_to_dict(memory: VisitorMemory) -> dict:
    """Convert a VisitorMemory ORM object to a JSON-serializable dict."""
    return {
        "id": memory.id,
        "visitor_id": memory.visitor_id,
        "profile_id": memory.profile_id,
        "employee_id": memory.employee_id,
        "memory_type": memory.memory_type,
        "content": memory.content,
        "source_session": memory.source_session,
        "importance": memory.importance,
        "expires_at": memory.expires_at.isoformat() if memory.expires_at else None,
        "created_at": memory.created_at.isoformat() if memory.created_at else None,
    }


@router.get("/visitors/{visitor_id}/profile")
async def get_visitor_profile(visitor_id: str, request: Request):
    """Return the profile for a specific visitor.

    Scoped to the authenticated customer — only returns profiles where
    customer_id matches the caller.
    """
    db = _get_db(request)
    customer_id = _get_customer_id(request)

    try:
        async with db.session_ctx() as session:
            stmt = select(VisitorProfile).where(
                VisitorProfile.visitor_id == visitor_id,
            )
            # Scope to customer if authenticated
            if customer_id:
                stmt = stmt.where(VisitorProfile.customer_id == customer_id)

            result = await session.execute(stmt)
            profile = result.scalar()

            if profile is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Visitor profile not found for visitor_id={visitor_id}",
                )

            return _profile_to_dict(profile)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Get visitor profile failed: {exc}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to fetch visitor profile", "detail": str(exc)},
        )


@router.get("/visitors/{visitor_id}/memory")
async def get_visitor_memory(visitor_id: str, request: Request):
    """Return all memory entries for a specific visitor.

    Scoped to the authenticated customer via the visitor's profile.
    Returns memories ordered by creation date (newest first).
    """
    db = _get_db(request)
    customer_id = _get_customer_id(request)

    try:
        async with db.session_ctx() as session:
            # First verify the visitor belongs to this customer
            profile_stmt = select(VisitorProfile).where(
                VisitorProfile.visitor_id == visitor_id,
            )
            if customer_id:
                profile_stmt = profile_stmt.where(
                    VisitorProfile.customer_id == customer_id,
                )

            profile_result = await session.execute(profile_stmt)
            profile = profile_result.scalar()

            if profile is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Visitor profile not found for visitor_id={visitor_id}",
                )

            # Fetch memories
            mem_stmt = (
                select(VisitorMemory)
                .where(VisitorMemory.visitor_id == visitor_id)
                .order_by(VisitorMemory.created_at.desc())
            )
            mem_result = await session.execute(mem_stmt)
            memories = mem_result.scalars().all()

            return {
                "visitor_id": visitor_id,
                "profile_id": profile.id,
                "count": len(memories),
                "memories": [_memory_to_dict(m) for m in memories],
            }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Get visitor memory failed: {exc}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to fetch visitor memory", "detail": str(exc)},
        )
