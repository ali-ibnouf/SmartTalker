"""CRUD API for workflow management.

Routes:
    GET    /api/v1/workflows                          List workflows for customer
    POST   /api/v1/workflows                          Create a new workflow
    GET    /api/v1/workflows/templates                 List built-in workflow templates
    POST   /api/v1/workflows/from-template/{tid}       Create workflow from template
    GET    /api/v1/workflows/{id}                      Get workflow details
    PUT    /api/v1/workflows/{id}                      Update a workflow
    DELETE /api/v1/workflows/{id}                      Delete a workflow
    POST   /api/v1/workflows/{id}/activate             Activate a workflow
    POST   /api/v1/workflows/{id}/deactivate           Deactivate a workflow
    GET    /api/v1/workflows/{id}/executions            List executions for a workflow
"""

from __future__ import annotations

import json
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from sqlalchemy import delete, select, func

from src.api.schemas import (
    WorkflowCreateRequest,
    WorkflowExecutionListResponse,
    WorkflowExecutionResponse,
    WorkflowListResponse,
    WorkflowResponse,
    WorkflowStepSchema,
    WorkflowTemplateListResponse,
    WorkflowTemplateResponse,
    WorkflowUpdateRequest,
)
from src.db.models import Workflow, WorkflowExecution
from src.utils.logger import setup_logger

logger = setup_logger("api.workflows")
router = APIRouter(prefix="/api/v1", tags=["workflows"])


def _get_db(request: Request):
    db = getattr(request.app.state, "db", None)
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    return db


def _get_customer_id(request: Request) -> str:
    return getattr(request.state, "customer_id", "")


def _parse_json(val: str) -> Any:
    """Safely parse a JSON text column."""
    try:
        return json.loads(val) if val else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _parse_json_list(val: str) -> list:
    """Safely parse a JSON text column expected to be a list."""
    try:
        result = json.loads(val) if val else []
        return result if isinstance(result, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def _workflow_to_response(wf: Workflow) -> WorkflowResponse:
    """Convert a Workflow ORM object to a response schema."""
    raw_steps = _parse_json_list(wf.steps)
    steps = []
    for s in raw_steps:
        if isinstance(s, dict):
            steps.append(WorkflowStepSchema(
                type=s.get("type", ""),
                config=s.get("config", {}),
            ))

    return WorkflowResponse(
        id=wf.id,
        customer_id=wf.customer_id,
        employee_id=wf.employee_id or "",
        name=wf.name,
        description=wf.description or "",
        trigger_type=wf.trigger_type or "manual",
        trigger_config=_parse_json(wf.trigger_config),
        steps=steps,
        is_active=wf.is_active,
        template_id=wf.template_id or "",
        created_at=wf.created_at.isoformat() if wf.created_at else None,
    )


def _execution_to_response(ex: WorkflowExecution) -> WorkflowExecutionResponse:
    """Convert a WorkflowExecution ORM object to a response schema."""
    return WorkflowExecutionResponse(
        id=ex.id,
        workflow_id=ex.workflow_id,
        session_id=ex.session_id or "",
        visitor_id=ex.visitor_id or "",
        status=ex.status or "running",
        current_step=ex.current_step,
        context=_parse_json(ex.context),
        error=ex.error or "",
        started_at=ex.started_at.isoformat() if ex.started_at else None,
        completed_at=ex.completed_at.isoformat() if ex.completed_at else None,
    )


# ── List Workflows ──────────────────────────────────────────────────────────


@router.get("/workflows", response_model=WorkflowListResponse)
async def list_workflows(request: Request):
    """List all workflows for the authenticated customer."""
    db = _get_db(request)
    customer_id = _get_customer_id(request)

    async with db.session() as session:
        result = await session.execute(
            select(Workflow)
            .where(Workflow.customer_id == customer_id)
            .order_by(Workflow.created_at.desc())
        )
        workflows = result.scalars().all()

    return WorkflowListResponse(
        workflows=[_workflow_to_response(wf) for wf in workflows],
        count=len(workflows),
    )


# ── Create Workflow ─────────────────────────────────────────────────────────


@router.post("/workflows", response_model=WorkflowResponse, status_code=201)
async def create_workflow(body: WorkflowCreateRequest, request: Request):
    """Create a new workflow."""
    db = _get_db(request)
    customer_id = _get_customer_id(request)

    steps_json = json.dumps([s.model_dump() for s in body.steps])
    trigger_json = json.dumps(body.trigger_config)

    async with db.session() as session:
        wf = Workflow(
            customer_id=customer_id,
            employee_id=body.employee_id,
            name=body.name,
            description=body.description,
            trigger_type=body.trigger_type,
            trigger_config=trigger_json,
            steps=steps_json,
        )
        session.add(wf)
        await session.commit()
        await session.refresh(wf)

    logger.info("workflow_created", extra={"workflow_id": wf.id, "customer_id": customer_id})
    return _workflow_to_response(wf)


# ── List Templates ──────────────────────────────────────────────────────────
# IMPORTANT: This must be defined BEFORE /workflows/{workflow_id} to avoid
# FastAPI matching "templates" as a workflow_id path parameter.


@router.get("/workflows/templates", response_model=WorkflowTemplateListResponse)
async def list_templates(request: Request):
    """Return built-in workflow templates."""
    try:
        from src.pipeline.workflow_engine import WORKFLOW_TEMPLATES
    except ImportError:
        WORKFLOW_TEMPLATES = {}

    templates = []
    for tid, tpl in WORKFLOW_TEMPLATES.items():
        raw_steps = tpl.get("steps", [])
        steps = []
        for s in raw_steps:
            if isinstance(s, dict):
                steps.append(WorkflowStepSchema(
                    type=s.get("type", ""),
                    config=s.get("config", {}),
                ))
        templates.append(WorkflowTemplateResponse(
            template_id=tid,
            name=tpl.get("name", tid),
            description=tpl.get("description", ""),
            trigger_type=tpl.get("trigger_type", "manual"),
            steps=steps,
        ))

    return WorkflowTemplateListResponse(
        templates=templates,
        count=len(templates),
    )


# ── Create from Template ────────────────────────────────────────────────────


@router.post("/workflows/from-template/{template_id}", response_model=WorkflowResponse, status_code=201)
async def create_from_template(template_id: str, request: Request):
    """Create a new workflow from a built-in template."""
    try:
        from src.pipeline.workflow_engine import WORKFLOW_TEMPLATES
    except ImportError:
        WORKFLOW_TEMPLATES = {}

    tpl = WORKFLOW_TEMPLATES.get(template_id)
    if not tpl:
        raise HTTPException(status_code=404, detail=f"Template '{template_id}' not found")

    db = _get_db(request)
    customer_id = _get_customer_id(request)

    steps_json = json.dumps(tpl.get("steps", []))
    trigger_config_json = json.dumps(tpl.get("trigger_config", {}))

    async with db.session() as session:
        wf = Workflow(
            customer_id=customer_id,
            name=tpl.get("name", template_id),
            description=tpl.get("description", ""),
            trigger_type=tpl.get("trigger_type", "manual"),
            trigger_config=trigger_config_json,
            steps=steps_json,
            template_id=template_id,
        )
        session.add(wf)
        await session.commit()
        await session.refresh(wf)

    logger.info(
        "workflow_created_from_template",
        extra={"workflow_id": wf.id, "template_id": template_id, "customer_id": customer_id},
    )
    return _workflow_to_response(wf)


# ── Get Workflow ────────────────────────────────────────────────────────────


@router.get("/workflows/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow(workflow_id: str, request: Request):
    """Get workflow details by ID."""
    db = _get_db(request)
    customer_id = _get_customer_id(request)

    async with db.session() as session:
        result = await session.execute(
            select(Workflow).where(
                Workflow.id == workflow_id,
                Workflow.customer_id == customer_id,
            )
        )
        wf = result.scalar_one_or_none()

    if not wf:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return _workflow_to_response(wf)


# ── Update Workflow ─────────────────────────────────────────────────────────


@router.put("/workflows/{workflow_id}", response_model=WorkflowResponse)
async def update_workflow(workflow_id: str, body: WorkflowUpdateRequest, request: Request):
    """Update a workflow."""
    db = _get_db(request)
    customer_id = _get_customer_id(request)

    async with db.session() as session:
        result = await session.execute(
            select(Workflow).where(
                Workflow.id == workflow_id,
                Workflow.customer_id == customer_id,
            )
        )
        wf = result.scalar_one_or_none()
        if not wf:
            raise HTTPException(status_code=404, detail="Workflow not found")

        if body.name is not None:
            wf.name = body.name
        if body.description is not None:
            wf.description = body.description
        if body.employee_id is not None:
            wf.employee_id = body.employee_id
        if body.trigger_type is not None:
            wf.trigger_type = body.trigger_type
        if body.trigger_config is not None:
            wf.trigger_config = json.dumps(body.trigger_config)
        if body.steps is not None:
            wf.steps = json.dumps([s.model_dump() for s in body.steps])

        await session.commit()
        await session.refresh(wf)

    logger.info("workflow_updated", extra={"workflow_id": workflow_id, "customer_id": customer_id})
    return _workflow_to_response(wf)


# ── Delete Workflow ─────────────────────────────────────────────────────────


@router.delete("/workflows/{workflow_id}", status_code=204)
async def delete_workflow(workflow_id: str, request: Request):
    """Delete a workflow and its executions."""
    db = _get_db(request)
    customer_id = _get_customer_id(request)

    async with db.session() as session:
        result = await session.execute(
            select(Workflow).where(
                Workflow.id == workflow_id,
                Workflow.customer_id == customer_id,
            )
        )
        wf = result.scalar_one_or_none()
        if not wf:
            raise HTTPException(status_code=404, detail="Workflow not found")

        # Remove executions first
        await session.execute(
            delete(WorkflowExecution).where(WorkflowExecution.workflow_id == wf.id)
        )
        await session.delete(wf)
        await session.commit()

    logger.info("workflow_deleted", extra={"workflow_id": workflow_id, "customer_id": customer_id})


# ── Activate Workflow ───────────────────────────────────────────────────────


@router.post("/workflows/{workflow_id}/activate", response_model=WorkflowResponse)
async def activate_workflow(workflow_id: str, request: Request):
    """Set a workflow to active."""
    db = _get_db(request)
    customer_id = _get_customer_id(request)

    async with db.session() as session:
        result = await session.execute(
            select(Workflow).where(
                Workflow.id == workflow_id,
                Workflow.customer_id == customer_id,
            )
        )
        wf = result.scalar_one_or_none()
        if not wf:
            raise HTTPException(status_code=404, detail="Workflow not found")

        wf.is_active = True
        await session.commit()
        await session.refresh(wf)

    logger.info("workflow_activated", extra={"workflow_id": workflow_id, "customer_id": customer_id})
    return _workflow_to_response(wf)


# ── Deactivate Workflow ─────────────────────────────────────────────────────


@router.post("/workflows/{workflow_id}/deactivate", response_model=WorkflowResponse)
async def deactivate_workflow(workflow_id: str, request: Request):
    """Set a workflow to inactive."""
    db = _get_db(request)
    customer_id = _get_customer_id(request)

    async with db.session() as session:
        result = await session.execute(
            select(Workflow).where(
                Workflow.id == workflow_id,
                Workflow.customer_id == customer_id,
            )
        )
        wf = result.scalar_one_or_none()
        if not wf:
            raise HTTPException(status_code=404, detail="Workflow not found")

        wf.is_active = False
        await session.commit()
        await session.refresh(wf)

    logger.info("workflow_deactivated", extra={"workflow_id": workflow_id, "customer_id": customer_id})
    return _workflow_to_response(wf)


# ── List Executions ─────────────────────────────────────────────────────────


@router.get("/workflows/{workflow_id}/executions", response_model=WorkflowExecutionListResponse)
async def list_executions(
    workflow_id: str, request: Request, limit: int = 50, offset: int = 0
):
    """List executions for a specific workflow."""
    db = _get_db(request)
    customer_id = _get_customer_id(request)

    async with db.session() as session:
        # Verify workflow belongs to customer
        wf_result = await session.execute(
            select(Workflow).where(
                Workflow.id == workflow_id,
                Workflow.customer_id == customer_id,
            )
        )
        wf = wf_result.scalar_one_or_none()
        if not wf:
            raise HTTPException(status_code=404, detail="Workflow not found")

        # Fetch executions
        result = await session.execute(
            select(WorkflowExecution)
            .where(WorkflowExecution.workflow_id == workflow_id)
            .order_by(WorkflowExecution.started_at.desc())
            .offset(offset)
            .limit(limit)
        )
        executions = result.scalars().all()

        count_result = await session.execute(
            select(func.count(WorkflowExecution.id))
            .where(WorkflowExecution.workflow_id == workflow_id)
        )
        total = count_result.scalar() or 0

    return WorkflowExecutionListResponse(
        executions=[_execution_to_response(ex) for ex in executions],
        count=total,
    )
