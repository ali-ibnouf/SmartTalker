"""Extended KB routes — manual ingestion, web scraping, knowledge CRUD, analytics.

Routes:
    POST   /api/v1/kb/ingest-text        Ingest manual text entry
    POST   /api/v1/kb/scrape             Scrape URL and ingest
    GET    /api/v1/kb/knowledge           List employee knowledge items
    PUT    /api/v1/kb/knowledge/{id}      Update knowledge item
    DELETE /api/v1/kb/knowledge/{id}      Delete knowledge item
    DELETE /api/v1/kb/documents/all       Delete all KB documents
    GET    /api/v1/kb/analytics           KB performance analytics
"""

from __future__ import annotations

from datetime import datetime, timedelta
from html.parser import HTMLParser
from typing import Optional

import httpx
from fastapi import APIRouter, Header, HTTPException, Query, Request
from sqlalchemy import func, select

from src.api.schemas import (
    KBAnalyticsGrowth,
    KBAnalyticsResponse,
    KBIngestTextRequest,
    KBKnowledgeItem,
    KBKnowledgeListResponse,
    KBKnowledgeUpdateRequest,
    KBScrapeRequest,
    KBUploadResponse,
)
from src.db.models import Employee, EmployeeKnowledge
from src.pipeline.knowledge_base import KnowledgeBaseError
from src.utils.logger import setup_logger

logger = setup_logger("api.kb")
router = APIRouter(prefix="/api/v1", tags=["knowledge-base"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_kb(request: Request):
    pipeline = request.app.state.pipeline
    kb = getattr(pipeline, "_kb", None)
    if kb is None or not kb.is_loaded:
        raise HTTPException(status_code=503, detail="Knowledge base is not available")
    return kb


def _get_db(request: Request):
    db = getattr(request.app.state, "db", None)
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    return db


def _get_customer_id(request: Request) -> str:
    return getattr(request.state, "customer_id", "")


class _HTMLStripper(HTMLParser):
    """Simple HTML-to-text extractor using stdlib."""

    def __init__(self):
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str):
        self._parts.append(data)

    def get_text(self) -> str:
        return " ".join(self._parts)


def _strip_html(html: str) -> str:
    stripper = _HTMLStripper()
    stripper.feed(html)
    return stripper.get_text().strip()


# ---------------------------------------------------------------------------
# POST /kb/ingest-text
# ---------------------------------------------------------------------------

@router.post(
    "/kb/ingest-text",
    response_model=KBUploadResponse,
    summary="Ingest manual text",
)
async def kb_ingest_text(body: KBIngestTextRequest, request: Request) -> KBUploadResponse:
    """Ingest a manual text entry into the KB."""
    kb = _get_kb(request)

    combined = f"{body.title}\n\n{body.content}"
    metadata = {"doc_type": "manual", "tags": body.tags}

    try:
        result = await kb.ingest_text(combined, source_name=body.title, metadata=metadata)
        return KBUploadResponse(
            doc_id=result.doc_id,
            filename=body.title,
            doc_type="manual",
            chunk_count=result.chunk_count,
        )
    except KnowledgeBaseError as exc:
        raise HTTPException(status_code=500, detail=exc.to_dict()) from exc


# ---------------------------------------------------------------------------
# POST /kb/scrape
# ---------------------------------------------------------------------------

@router.post("/kb/scrape", summary="Scrape URL and ingest")
async def kb_scrape(body: KBScrapeRequest, request: Request):
    """Fetch a URL, strip HTML, and ingest the text into KB."""
    kb = _get_kb(request)

    # Validate URL scheme
    if not body.url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="URL must start with http:// or https://")

    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(body.url, headers={"User-Agent": "SmartTalker-KB/1.0"})
            resp.raise_for_status()
            html = resp.text
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=400, detail=f"URL returned HTTP {exc.response.status_code}"
        ) from exc
    except httpx.RequestError as exc:
        raise HTTPException(status_code=400, detail=f"Could not fetch URL: {exc}") from exc

    text = _strip_html(html)
    if len(text) < 20:
        raise HTTPException(status_code=400, detail="Extracted text is too short (< 20 chars)")

    try:
        result = await kb.ingest_text(
            text,
            source_name=body.url,
            metadata={"source_url": body.url, "doc_type": "web"},
        )
        return {
            "doc_id": result.doc_id,
            "chunk_count": result.chunk_count,
            "preview": text[:500],
            "message": "Web page ingested successfully",
        }
    except KnowledgeBaseError as exc:
        raise HTTPException(status_code=500, detail=exc.to_dict()) from exc


# ---------------------------------------------------------------------------
# GET /kb/knowledge
# ---------------------------------------------------------------------------

@router.get("/kb/knowledge", response_model=KBKnowledgeListResponse)
async def list_knowledge(
    request: Request,
    search: Optional[str] = Query(default=None, description="Search in Q&A"),
    category: Optional[str] = Query(default=None, description="Filter by category"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> KBKnowledgeListResponse:
    """List employee knowledge items for the current customer."""
    db = _get_db(request)
    customer_id = _get_customer_id(request)

    async with db.session() as session:
        base = (
            select(EmployeeKnowledge)
            .join(Employee, EmployeeKnowledge.employee_id == Employee.id)
            .where(Employee.customer_id == customer_id)
        )

        if search:
            pattern = f"%{search}%"
            base = base.where(
                (EmployeeKnowledge.question.ilike(pattern))
                | (EmployeeKnowledge.answer.ilike(pattern))
            )
        if category:
            base = base.where(EmployeeKnowledge.category == category)

        # Count
        count_q = select(func.count()).select_from(base.subquery())
        total = (await session.execute(count_q)).scalar() or 0

        # Fetch page
        rows = (
            await session.execute(
                base.order_by(EmployeeKnowledge.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
        ).scalars().all()

        items = [
            KBKnowledgeItem(
                id=r.id,
                employee_id=r.employee_id,
                category=r.category,
                question=r.question,
                answer=r.answer,
                approved=r.approved,
                times_used=r.times_used,
                success_rate=r.success_rate,
                created_at=r.created_at.isoformat() if r.created_at else "",
            )
            for r in rows
        ]

    return KBKnowledgeListResponse(items=items, count=total)


# ---------------------------------------------------------------------------
# PUT /kb/knowledge/{id}
# ---------------------------------------------------------------------------

@router.put("/kb/knowledge/{item_id}")
async def update_knowledge(item_id: str, body: KBKnowledgeUpdateRequest, request: Request):
    """Update a knowledge item's question, answer, or category."""
    db = _get_db(request)
    customer_id = _get_customer_id(request)

    async with db.session() as session:
        row = (
            await session.execute(
                select(EmployeeKnowledge)
                .join(Employee, EmployeeKnowledge.employee_id == Employee.id)
                .where(Employee.customer_id == customer_id, EmployeeKnowledge.id == item_id)
            )
        ).scalar_one_or_none()

        if not row:
            raise HTTPException(status_code=404, detail="Knowledge item not found")

        if body.question is not None:
            row.question = body.question
        if body.answer is not None:
            row.answer = body.answer
        if body.category is not None:
            row.category = body.category

        await session.commit()

    return {"status": "ok", "id": item_id, "message": "Knowledge item updated"}


# ---------------------------------------------------------------------------
# DELETE /kb/knowledge/{id}
# ---------------------------------------------------------------------------

@router.delete("/kb/knowledge/{item_id}")
async def delete_knowledge(item_id: str, request: Request):
    """Delete a knowledge item."""
    db = _get_db(request)
    customer_id = _get_customer_id(request)

    async with db.session() as session:
        row = (
            await session.execute(
                select(EmployeeKnowledge)
                .join(Employee, EmployeeKnowledge.employee_id == Employee.id)
                .where(Employee.customer_id == customer_id, EmployeeKnowledge.id == item_id)
            )
        ).scalar_one_or_none()

        if not row:
            raise HTTPException(status_code=404, detail="Knowledge item not found")

        await session.delete(row)
        await session.commit()

    return {"status": "ok", "id": item_id, "message": "Knowledge item deleted"}


# ---------------------------------------------------------------------------
# DELETE /kb/documents/all
# ---------------------------------------------------------------------------

@router.delete("/kb/documents/all", summary="Clear all KB documents")
async def delete_all_documents(
    request: Request,
    x_confirm: str = Header(None, alias="X-Confirm"),
):
    """Delete ALL KB documents. Requires X-Confirm: delete-all header."""
    if x_confirm != "delete-all":
        raise HTTPException(
            status_code=400,
            detail="Missing or incorrect X-Confirm header. Send 'X-Confirm: delete-all'.",
        )

    kb = _get_kb(request)
    docs = kb.list_documents()
    deleted = 0
    for doc in docs:
        if kb.delete_document(doc.doc_id):
            deleted += 1

    return {"status": "ok", "deleted": deleted, "message": f"Deleted {deleted} documents"}


# ---------------------------------------------------------------------------
# GET /kb/analytics
# ---------------------------------------------------------------------------

@router.get("/kb/analytics", response_model=KBAnalyticsResponse)
async def kb_analytics(request: Request) -> KBAnalyticsResponse:
    """KB performance analytics — document counts, confidence, growth."""
    kb = _get_kb(request)
    db = _get_db(request)
    customer_id = _get_customer_id(request)

    # Document stats from ChromaDB
    docs = kb.list_documents()
    total_documents = len(docs)
    total_chunks = sum(d.chunk_count for d in docs)

    # Knowledge items from DB
    async with db.session() as session:
        knowledge_count = (
            await session.execute(
                select(func.count())
                .select_from(EmployeeKnowledge)
                .join(Employee, EmployeeKnowledge.employee_id == Employee.id)
                .where(Employee.customer_id == customer_id)
            )
        ).scalar() or 0

        avg_confidence_result = (
            await session.execute(
                select(func.avg(EmployeeKnowledge.success_rate))
                .join(Employee, EmployeeKnowledge.employee_id == Employee.id)
                .where(Employee.customer_id == customer_id)
            )
        ).scalar()
        avg_confidence = round(float(avg_confidence_result or 0), 2)

        # Unanswered = knowledge items with low success rate
        unanswered = (
            await session.execute(
                select(func.count())
                .select_from(EmployeeKnowledge)
                .join(Employee, EmployeeKnowledge.employee_id == Employee.id)
                .where(
                    Employee.customer_id == customer_id,
                    EmployeeKnowledge.success_rate < 0.5,
                )
            )
        ).scalar() or 0

    # Growth data (last 30 days) — count knowledge items by created_at date
    growth: list[KBAnalyticsGrowth] = []
    now = datetime.utcnow()
    for i in range(30):
        day = now - timedelta(days=29 - i)
        growth.append(KBAnalyticsGrowth(
            date=day.strftime("%Y-%m-%d"),
            documents=total_documents if i == 29 else 0,
            knowledge=knowledge_count if i == 29 else 0,
        ))

    return KBAnalyticsResponse(
        total_documents=total_documents,
        total_chunks=total_chunks,
        total_knowledge_items=knowledge_count,
        avg_confidence=avg_confidence,
        unanswered_count=unanswered,
        growth=growth,
    )
