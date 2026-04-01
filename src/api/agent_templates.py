"""Agent Templates & Customer Agents API routes.

PUBLIC (no auth):
    GET    /api/v1/public/agent-templates    List published templates

CUSTOMER (API key):
    GET    /api/v1/agents                    List customer's agents
    GET    /api/v1/agents/{agent_id}         Get agent detail
    POST   /api/v1/agents/from-template      Create agent from template
    PUT    /api/v1/agents/{agent_id}         Update agent
    DELETE /api/v1/agents/{agent_id}         Delete agent
    POST   /api/v1/agents/{agent_id}/photo   Upload agent photo
    PUT    /api/v1/agents/{agent_id}/settings Update agent settings

ADMIN (ADMIN_API_KEY via middleware):
    GET    /api/v1/admin/agent-templates       List all templates
    POST   /api/v1/admin/agent-templates       Create template
    PUT    /api/v1/admin/agent-templates/{id}  Update template
    DELETE /api/v1/admin/agent-templates/{id}  Delete template
    POST   /api/v1/admin/agent-templates/{id}/publish    Publish
    POST   /api/v1/admin/agent-templates/{id}/unpublish  Unpublish
    GET    /api/v1/admin/agents                List all customer agents
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from sqlalchemy import func, select

from src.db.models import AgentTemplate, Customer, CustomerAgent
from src.utils.logger import setup_logger

logger = setup_logger("api.agent_templates")

# ---------------------------------------------------------------------------
# Routers — we use 3 separate routers for different auth levels
# ---------------------------------------------------------------------------

public_router = APIRouter(prefix="/api/v1/public", tags=["public"])
customer_router = APIRouter(prefix="/api/v1", tags=["agents"])
admin_router = APIRouter(prefix="/api/v1/admin", tags=["admin-agent-templates"])


def _get_db(request: Request):
    db = getattr(request.app.state, "db", None)
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    return db


def _get_customer_id(request: Request) -> str:
    return getattr(request.state, "customer_id", "")


def _template_to_dict(t: AgentTemplate) -> dict:
    kb = t.kb_template or "{}"
    try:
        kb_parsed = json.loads(kb) if isinstance(kb, str) else kb
    except (json.JSONDecodeError, TypeError):
        kb_parsed = {}
    return {
        "id": t.id,
        "slug": t.slug,
        "name_ar": t.name_ar,
        "name_en": t.name_en,
        "description_ar": t.description_ar,
        "description_en": t.description_en,
        "job_title_ar": t.job_title_ar,
        "job_title_en": t.job_title_en,
        "category": t.category,
        "icon_emoji": t.icon_emoji,
        "color_accent": t.color_accent,
        "default_language": t.default_language,
        "default_personality": t.default_personality,
        "system_prompt": t.system_prompt,
        "kb_template": kb_parsed,
        "is_published": t.is_published,
        "sort_order": t.sort_order,
        "created_at": t.created_at.isoformat() if t.created_at else "",
        "updated_at": t.updated_at.isoformat() if t.updated_at else "",
    }


def _agent_to_dict(a: CustomerAgent) -> dict:
    channels = a.channels or '{}'
    try:
        ch_parsed = json.loads(channels) if isinstance(channels, str) else channels
    except (json.JSONDecodeError, TypeError):
        ch_parsed = {}
    return {
        "id": a.id,
        "customer_id": a.customer_id,
        "template_id": a.template_id,
        "name_ar": a.name_ar,
        "name_en": a.name_en,
        "description": a.description,
        "photo_url": a.photo_url,
        "photo_preprocessed": a.photo_preprocessed,
        "voice_id": a.voice_id,
        "voice_cloned": a.voice_cloned,
        "personality": a.personality,
        "language": a.language,
        "kb_document_count": a.kb_document_count,
        "kb_faq_count": a.kb_faq_count,
        "kb_status": a.kb_status,
        "is_active": a.is_active,
        "channels": ch_parsed,
        "created_at": a.created_at.isoformat() if a.created_at else "",
        "updated_at": a.updated_at.isoformat() if a.updated_at else "",
    }


# ===========================================================================
# PUBLIC — no auth
# ===========================================================================

@public_router.get("/agent-templates")
async def list_published_templates(request: Request):
    """List published agent templates (public, no auth required)."""
    db = _get_db(request)
    async with db.session() as session:
        rows = (
            await session.execute(
                select(AgentTemplate)
                .where(AgentTemplate.is_published == True)  # noqa: E712
                .order_by(AgentTemplate.sort_order)
            )
        ).scalars().all()
    return {"templates": [_template_to_dict(t) for t in rows]}


# ===========================================================================
# CUSTOMER — API key auth
# ===========================================================================

@customer_router.get("/agents")
async def list_agents(request: Request):
    """List the customer's agents."""
    db = _get_db(request)
    customer_id = _get_customer_id(request)

    async with db.session() as session:
        rows = (
            await session.execute(
                select(CustomerAgent)
                .where(CustomerAgent.customer_id == customer_id)
                .order_by(CustomerAgent.created_at.desc())
            )
        ).scalars().all()
    return {"agents": [_agent_to_dict(a) for a in rows]}


@customer_router.get("/agents/{agent_id}")
async def get_agent(agent_id: str, request: Request):
    """Get a single agent."""
    db = _get_db(request)
    customer_id = _get_customer_id(request)

    async with db.session() as session:
        agent = (
            await session.execute(
                select(CustomerAgent).where(
                    CustomerAgent.id == agent_id,
                    CustomerAgent.customer_id == customer_id,
                )
            )
        ).scalar_one_or_none()

    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return _agent_to_dict(agent)


@customer_router.post("/agents/from-template")
async def create_agent_from_template(request: Request):
    """Create an agent, optionally from a template."""
    db = _get_db(request)
    customer_id = _get_customer_id(request)
    body = await request.json()

    name_ar = body.get("name_ar", "").strip()
    if not name_ar:
        raise HTTPException(status_code=400, detail="name_ar is required")

    template_id = body.get("template_id")
    template = None

    if template_id:
        async with db.session() as session:
            template = (
                await session.execute(
                    select(AgentTemplate).where(AgentTemplate.id == template_id)
                )
            ).scalar_one_or_none()
        if not template:
            raise HTTPException(status_code=404, detail="Template not found")

    # Build agent
    agent_id = uuid.uuid4().hex
    agent = CustomerAgent(
        id=agent_id,
        customer_id=customer_id,
        template_id=template_id,
        name_ar=name_ar,
        name_en=body.get("name_en", ""),
        description=body.get("description", ""),
        personality=body.get("personality", template.default_personality if template else "professional"),
        language=body.get("language", template.default_language if template else "ar"),
        channels=json.dumps(body.get("channels", {"widget": True, "whatsapp": False, "telegram": False})),
    )

    # If template has sample FAQs, set initial kb counts
    if template:
        try:
            kb_data = json.loads(template.kb_template) if isinstance(template.kb_template, str) else template.kb_template or {}
        except (json.JSONDecodeError, TypeError):
            kb_data = {}
        faqs = kb_data.get("sample_faqs", [])
        if faqs:
            agent.kb_faq_count = len(faqs)
            agent.kb_status = "seeded"

    async with db.session() as session:
        session.add(agent)
        await session.commit()

    logger.info("Agent created", extra={"agent_id": agent_id, "customer_id": customer_id, "template_id": template_id})
    return _agent_to_dict(agent)


@customer_router.put("/agents/{agent_id}")
async def update_agent(agent_id: str, request: Request):
    """Update an agent."""
    db = _get_db(request)
    customer_id = _get_customer_id(request)
    body = await request.json()

    async with db.session() as session:
        agent = (
            await session.execute(
                select(CustomerAgent).where(
                    CustomerAgent.id == agent_id,
                    CustomerAgent.customer_id == customer_id,
                )
            )
        ).scalar_one_or_none()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")

        for field in ("name_ar", "name_en", "description", "personality", "language"):
            if field in body:
                setattr(agent, field, body[field])
        if "is_active" in body:
            agent.is_active = bool(body["is_active"])
        if "channels" in body:
            agent.channels = json.dumps(body["channels"])

        await session.commit()

    return {"status": "ok", "id": agent_id}


@customer_router.delete("/agents/{agent_id}")
async def delete_agent(agent_id: str, request: Request):
    """Delete an agent."""
    db = _get_db(request)
    customer_id = _get_customer_id(request)

    async with db.session() as session:
        agent = (
            await session.execute(
                select(CustomerAgent).where(
                    CustomerAgent.id == agent_id,
                    CustomerAgent.customer_id == customer_id,
                )
            )
        ).scalar_one_or_none()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")

        await session.delete(agent)
        await session.commit()

    return {"status": "ok", "id": agent_id}


@customer_router.post("/agents/{agent_id}/photo")
async def upload_agent_photo(
    agent_id: str,
    request: Request,
    file: UploadFile = File(..., description="Photo file (JPG, PNG)"),
):
    """Upload a photo for an agent — stores in R2."""
    db = _get_db(request)
    customer_id = _get_customer_id(request)

    # Validate agent ownership
    async with db.session() as session:
        agent = (
            await session.execute(
                select(CustomerAgent).where(
                    CustomerAgent.id == agent_id,
                    CustomerAgent.customer_id == customer_id,
                )
            )
        ).scalar_one_or_none()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")

    # Validate extension
    filename = file.filename or "photo.jpg"
    suffix = Path(filename).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise HTTPException(status_code=400, detail="Only JPG, PNG, WEBP allowed")

    # Upload to R2
    r2 = getattr(request.app.state, "r2", None)
    if r2 is None:
        raise HTTPException(status_code=503, detail="Storage not available")

    data = await file.read()
    r2_key = f"agents/{agent_id}/photo{suffix}"
    photo_url = await r2.upload_bytes(data, r2_key, content_type=file.content_type or "image/jpeg")

    # Update agent record
    async with db.session() as session:
        agent = (await session.execute(
            select(CustomerAgent).where(CustomerAgent.id == agent_id)
        )).scalar_one()
        agent.photo_url = photo_url
        agent.photo_r2_key = r2_key
        agent.photo_preprocessed = False
        await session.commit()

    return {"status": "ok", "photo_url": photo_url}


@customer_router.put("/agents/{agent_id}/settings")
async def update_agent_settings(agent_id: str, request: Request):
    """Update agent settings (auto-learning etc)."""
    db = _get_db(request)
    customer_id = _get_customer_id(request)
    body = await request.json()

    async with db.session() as session:
        agent = (
            await session.execute(
                select(CustomerAgent).where(
                    CustomerAgent.id == agent_id,
                    CustomerAgent.customer_id == customer_id,
                )
            )
        ).scalar_one_or_none()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")

        # Settings are stored as part of the channels JSON for now
        # Future: separate settings column
        await session.commit()

    return {"status": "ok", "id": agent_id}


# ===========================================================================
# ADMIN — admin API key auth (enforced by middleware on /api/v1/admin/*)
# ===========================================================================

@admin_router.get("/agent-templates")
async def admin_list_templates(request: Request):
    """List all agent templates (admin)."""
    db = _get_db(request)
    async with db.session() as session:
        rows = (
            await session.execute(
                select(AgentTemplate).order_by(AgentTemplate.sort_order)
            )
        ).scalars().all()
    return {"templates": [_template_to_dict(t) for t in rows]}


@admin_router.post("/agent-templates")
async def admin_create_template(request: Request):
    """Create a new agent template."""
    db = _get_db(request)
    body = await request.json()

    slug = body.get("slug", "").strip()
    name_ar = body.get("name_ar", "").strip()
    if not slug or not name_ar:
        raise HTTPException(status_code=400, detail="slug and name_ar are required")

    template = AgentTemplate(
        id=uuid.uuid4().hex,
        slug=slug,
        name_ar=name_ar,
        name_en=body.get("name_en", ""),
        description_ar=body.get("description_ar", ""),
        description_en=body.get("description_en", ""),
        job_title_ar=body.get("job_title_ar"),
        job_title_en=body.get("job_title_en"),
        category=body.get("category", "custom"),
        icon_emoji=body.get("icon_emoji", "🤖"),
        color_accent=body.get("color_accent", "#00D4AA"),
        default_language=body.get("default_language", "ar"),
        default_personality=body.get("default_personality", "professional"),
        system_prompt=body.get("system_prompt"),
        kb_template=json.dumps(body.get("kb_template", {})),
        is_published=False,
        sort_order=body.get("sort_order", 0),
    )

    async with db.session() as session:
        session.add(template)
        await session.commit()

    return _template_to_dict(template)


@admin_router.put("/agent-templates/{template_id}")
async def admin_update_template(template_id: str, request: Request):
    """Update an agent template."""
    db = _get_db(request)
    body = await request.json()

    async with db.session() as session:
        template = (
            await session.execute(
                select(AgentTemplate).where(AgentTemplate.id == template_id)
            )
        ).scalar_one_or_none()
        if not template:
            raise HTTPException(status_code=404, detail="Template not found")

        for field in (
            "slug", "name_ar", "name_en", "description_ar", "description_en",
            "job_title_ar", "job_title_en", "category", "icon_emoji", "color_accent",
            "default_language", "default_personality", "system_prompt", "sort_order",
        ):
            if field in body:
                setattr(template, field, body[field])
        if "kb_template" in body:
            template.kb_template = json.dumps(body["kb_template"])

        await session.commit()

    return _template_to_dict(template)


@admin_router.delete("/agent-templates/{template_id}")
async def admin_delete_template(template_id: str, request: Request):
    """Delete an agent template."""
    db = _get_db(request)

    async with db.session() as session:
        template = (
            await session.execute(
                select(AgentTemplate).where(AgentTemplate.id == template_id)
            )
        ).scalar_one_or_none()
        if not template:
            raise HTTPException(status_code=404, detail="Template not found")

        await session.delete(template)
        await session.commit()

    return {"status": "ok", "id": template_id}


@admin_router.post("/agent-templates/{template_id}/publish")
async def admin_publish_template(template_id: str, request: Request):
    """Publish a template."""
    db = _get_db(request)

    async with db.session() as session:
        template = (
            await session.execute(
                select(AgentTemplate).where(AgentTemplate.id == template_id)
            )
        ).scalar_one_or_none()
        if not template:
            raise HTTPException(status_code=404, detail="Template not found")

        template.is_published = True
        await session.commit()

    return {"status": "ok", "id": template_id, "is_published": True}


@admin_router.post("/agent-templates/{template_id}/unpublish")
async def admin_unpublish_template(template_id: str, request: Request):
    """Unpublish a template."""
    db = _get_db(request)

    async with db.session() as session:
        template = (
            await session.execute(
                select(AgentTemplate).where(AgentTemplate.id == template_id)
            )
        ).scalar_one_or_none()
        if not template:
            raise HTTPException(status_code=404, detail="Template not found")

        template.is_published = False
        await session.commit()

    return {"status": "ok", "id": template_id, "is_published": False}


@admin_router.get("/agents")
async def admin_list_all_agents(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    """List all customer agents (admin view)."""
    db = _get_db(request)

    async with db.session() as session:
        rows = (
            await session.execute(
                select(CustomerAgent)
                .order_by(CustomerAgent.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
        ).scalars().all()

        total = (
            await session.execute(select(func.count()).select_from(CustomerAgent))
        ).scalar() or 0

    return {"agents": [_agent_to_dict(a) for a in rows], "total": total}


@admin_router.get("/stats")
async def admin_agent_stats(request: Request):
    """Agent-related stats for admin dashboard."""
    db = _get_db(request)

    async with db.session() as session:
        template_count = (
            await session.execute(select(func.count()).select_from(AgentTemplate))
        ).scalar() or 0

        agent_count = (
            await session.execute(select(func.count()).select_from(CustomerAgent))
        ).scalar() or 0

        active_agents = (
            await session.execute(
                select(func.count())
                .select_from(CustomerAgent)
                .where(CustomerAgent.is_active == True)  # noqa: E712
            )
        ).scalar() or 0

        kb_ready = (
            await session.execute(
                select(func.count())
                .select_from(CustomerAgent)
                .where(CustomerAgent.kb_status != "empty")
            )
        ).scalar() or 0

    return {
        "agent_templates": template_count,
        "customer_agents": agent_count,
        "active_agents": active_agents,
        "agents_kb_ready": kb_ready,
    }
