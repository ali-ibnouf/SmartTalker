"""CRUD API for custom tool management.

Routes:
    GET    /api/v1/tools               List all tools for the customer
    POST   /api/v1/tools               Create a new custom tool
    GET    /api/v1/tools/{id}           Get tool details
    PUT    /api/v1/tools/{id}           Update a tool
    DELETE /api/v1/tools/{id}           Delete a tool
    POST   /api/v1/tools/{id}/test      Test-execute a tool
    GET    /api/v1/tools/{id}/logs      Get tool execution logs
    POST   /api/v1/employees/{eid}/tools/{tid}   Assign tool to employee
    DELETE /api/v1/employees/{eid}/tools/{tid}   Unassign tool from employee
"""

from __future__ import annotations

import json
import time
from typing import Any, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import delete, select, func

from src.api.schemas import (
    ToolCreateRequest,
    ToolListResponse,
    ToolLogListResponse,
    ToolLogResponse,
    ToolResponse,
    ToolTestRequest,
    ToolTestResponse,
    ToolUpdateRequest,
)
from src.db.models import EmployeeTools, ToolExecutionLog, ToolRegistry
from src.utils.logger import setup_logger

logger = setup_logger("api.tools")
router = APIRouter(prefix="/api/v1", tags=["tools"])


def _get_db(request: Request):
    db = getattr(request.app.state, "db", None)
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    return db


def _get_customer_id(request: Request) -> str:
    return getattr(request.state, "customer_id", "")


def _tool_to_response(tool: ToolRegistry) -> ToolResponse:
    """Convert a ToolRegistry ORM object to a response schema."""
    def _parse(val: str) -> dict:
        try:
            return json.loads(val) if val else {}
        except (json.JSONDecodeError, TypeError):
            return {}

    return ToolResponse(
        id=tool.id,
        tool_id=tool.tool_id,
        customer_id=tool.customer_id,
        name=tool.name,
        description=tool.description,
        category=tool.category,
        input_schema=_parse(tool.input_schema),
        api_url=tool.api_url,
        api_method=tool.api_method,
        api_headers=_parse(tool.api_headers),
        api_body_template=_parse(tool.api_body_template),
        response_mapping=_parse(tool.response_mapping),
        timeout_ms=tool.timeout_ms,
        requires_confirmation=tool.requires_confirmation,
        is_active=tool.is_active,
        created_at=tool.created_at.isoformat() if tool.created_at else None,
    )


# ── List Tools ──────────────────────────────────────────────────────────────

@router.get("/tools", response_model=ToolListResponse)
async def list_tools(request: Request):
    """List all custom tools for the authenticated customer."""
    db = _get_db(request)
    customer_id = _get_customer_id(request)

    async with db.session() as session:
        result = await session.execute(
            select(ToolRegistry)
            .where(ToolRegistry.customer_id == customer_id)
            .order_by(ToolRegistry.created_at.desc())
        )
        tools = result.scalars().all()

    return ToolListResponse(
        tools=[_tool_to_response(t) for t in tools],
        count=len(tools),
    )


# ── Create Tool ─────────────────────────────────────────────────────────────

@router.post("/tools", response_model=ToolResponse, status_code=201)
async def create_tool(body: ToolCreateRequest, request: Request):
    """Create a new custom API tool."""
    db = _get_db(request)
    customer_id = _get_customer_id(request)

    # SSRF protection: validate API URL before saving
    if body.api_url:
        from src.agent.security import validate_tool_url
        if not validate_tool_url(body.api_url):
            raise HTTPException(
                status_code=400,
                detail="URL not allowed: points to internal/private network",
            )

    async with db.session() as session:
        # Check uniqueness of tool_id
        existing = await session.execute(
            select(ToolRegistry).where(ToolRegistry.tool_id == body.tool_id)
        )
        if existing.scalar_one_or_none():
            raise HTTPException(status_code=409, detail=f"Tool ID '{body.tool_id}' already exists")

        tool = ToolRegistry(
            customer_id=customer_id,
            tool_id=body.tool_id,
            name=body.name,
            description=body.description,
            category=body.category,
            input_schema=json.dumps(body.input_schema),
            api_url=body.api_url,
            api_method=body.api_method,
            api_headers=json.dumps(body.api_headers),
            api_body_template=json.dumps(body.api_body_template),
            response_mapping=json.dumps(body.response_mapping),
            timeout_ms=body.timeout_ms,
            requires_confirmation=body.requires_confirmation,
        )
        session.add(tool)
        await session.commit()
        await session.refresh(tool)

    return _tool_to_response(tool)


# ── Get Tool ────────────────────────────────────────────────────────────────

@router.get("/tools/{tool_db_id}", response_model=ToolResponse)
async def get_tool(tool_db_id: str, request: Request):
    """Get tool details by database ID."""
    db = _get_db(request)
    customer_id = _get_customer_id(request)

    async with db.session() as session:
        result = await session.execute(
            select(ToolRegistry).where(
                ToolRegistry.id == tool_db_id,
                ToolRegistry.customer_id == customer_id,
            )
        )
        tool = result.scalar_one_or_none()

    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")
    return _tool_to_response(tool)


# ── Update Tool ─────────────────────────────────────────────────────────────

@router.put("/tools/{tool_db_id}", response_model=ToolResponse)
async def update_tool(tool_db_id: str, body: ToolUpdateRequest, request: Request):
    """Update a custom tool."""
    db = _get_db(request)
    customer_id = _get_customer_id(request)

    async with db.session() as session:
        result = await session.execute(
            select(ToolRegistry).where(
                ToolRegistry.id == tool_db_id,
                ToolRegistry.customer_id == customer_id,
            )
        )
        tool = result.scalar_one_or_none()
        if not tool:
            raise HTTPException(status_code=404, detail="Tool not found")

        # Apply updates
        if body.name is not None:
            tool.name = body.name
        if body.description is not None:
            tool.description = body.description
        if body.category is not None:
            tool.category = body.category
        if body.input_schema is not None:
            tool.input_schema = json.dumps(body.input_schema)
        if body.api_url is not None:
            from src.agent.security import validate_tool_url
            if not validate_tool_url(body.api_url):
                raise HTTPException(status_code=400, detail="URL blocked by SSRF policy")
            tool.api_url = body.api_url
        if body.api_method is not None:
            tool.api_method = body.api_method.upper()
        if body.api_headers is not None:
            tool.api_headers = json.dumps(body.api_headers)
        if body.api_body_template is not None:
            tool.api_body_template = json.dumps(body.api_body_template)
        if body.response_mapping is not None:
            tool.response_mapping = json.dumps(body.response_mapping)
        if body.timeout_ms is not None:
            tool.timeout_ms = body.timeout_ms
        if body.requires_confirmation is not None:
            tool.requires_confirmation = body.requires_confirmation
        if body.is_active is not None:
            tool.is_active = body.is_active

        await session.commit()
        await session.refresh(tool)

    return _tool_to_response(tool)


# ── Delete Tool ─────────────────────────────────────────────────────────────

@router.delete("/tools/{tool_db_id}", status_code=204)
async def delete_tool(tool_db_id: str, request: Request):
    """Delete a custom tool and its employee assignments."""
    db = _get_db(request)
    customer_id = _get_customer_id(request)

    async with db.session() as session:
        result = await session.execute(
            select(ToolRegistry).where(
                ToolRegistry.id == tool_db_id,
                ToolRegistry.customer_id == customer_id,
            )
        )
        tool = result.scalar_one_or_none()
        if not tool:
            raise HTTPException(status_code=404, detail="Tool not found")

        # Remove assignments first
        await session.execute(
            delete(EmployeeTools).where(EmployeeTools.tool_id == tool.id)
        )
        await session.delete(tool)
        await session.commit()


# ── Test Tool ───────────────────────────────────────────────────────────────

@router.post("/tools/{tool_db_id}/test", response_model=ToolTestResponse)
async def test_tool(tool_db_id: str, body: ToolTestRequest, request: Request):
    """Test-execute a custom API tool with sample parameters."""
    db = _get_db(request)
    customer_id = _get_customer_id(request)

    async with db.session() as session:
        result = await session.execute(
            select(ToolRegistry).where(
                ToolRegistry.id == tool_db_id,
                ToolRegistry.customer_id == customer_id,
            )
        )
        tool = result.scalar_one_or_none()

    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")

    if not tool.api_url:
        return ToolTestResponse(status="error", error="No API URL configured")

    # Build request from template
    try:
        headers_dict = json.loads(tool.api_headers) if tool.api_headers else {}
    except (json.JSONDecodeError, TypeError):
        headers_dict = {}

    try:
        body_template = json.loads(tool.api_body_template) if tool.api_body_template else {}
    except (json.JSONDecodeError, TypeError):
        body_template = {}

    # Replace {{param}} placeholders
    body_str = json.dumps(body_template)
    for key, value in body.parameters.items():
        body_str = body_str.replace(f"{{{{{key}}}}}", str(value))

    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=tool.timeout_ms / 1000.0) as client:
            if tool.api_method.upper() == "GET":
                resp = await client.get(tool.api_url, headers=headers_dict, params=body.parameters)
            else:
                resp = await client.request(
                    method=tool.api_method.upper(),
                    url=tool.api_url,
                    headers={**headers_dict, "Content-Type": "application/json"},
                    content=body_str,
                )

        elapsed_ms = int((time.perf_counter() - start) * 1000)

        try:
            output = resp.json()
        except Exception:
            output = {"raw": resp.text[:2000]}

        if resp.is_success:
            return ToolTestResponse(status="success", output=output, execution_time_ms=elapsed_ms)
        else:
            return ToolTestResponse(
                status="error",
                output=output,
                error=f"HTTP {resp.status_code}",
                execution_time_ms=elapsed_ms,
            )
    except httpx.TimeoutException:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return ToolTestResponse(status="timeout", error="Request timed out", execution_time_ms=elapsed_ms)
    except Exception as exc:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return ToolTestResponse(status="error", error=str(exc), execution_time_ms=elapsed_ms)


# ── Tool Execution Logs ─────────────────────────────────────────────────────

@router.get("/tools/{tool_db_id}/logs", response_model=ToolLogListResponse)
async def get_tool_logs(
    tool_db_id: str, request: Request, limit: int = 50, offset: int = 0
):
    """Get execution logs for a specific tool."""
    db = _get_db(request)
    customer_id = _get_customer_id(request)

    async with db.session() as session:
        # Verify tool belongs to customer
        tool_result = await session.execute(
            select(ToolRegistry).where(
                ToolRegistry.id == tool_db_id,
                ToolRegistry.customer_id == customer_id,
            )
        )
        tool = tool_result.scalar_one_or_none()
        if not tool:
            raise HTTPException(status_code=404, detail="Tool not found")

        # Fetch logs
        result = await session.execute(
            select(ToolExecutionLog)
            .where(ToolExecutionLog.tool_id == tool.tool_id)
            .order_by(ToolExecutionLog.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        logs = result.scalars().all()

        count_result = await session.execute(
            select(func.count(ToolExecutionLog.id))
            .where(ToolExecutionLog.tool_id == tool.tool_id)
        )
        total = count_result.scalar() or 0

    def _parse(val: str) -> dict:
        try:
            return json.loads(val) if val else {}
        except (json.JSONDecodeError, TypeError):
            return {}

    return ToolLogListResponse(
        logs=[
            ToolLogResponse(
                id=log.id,
                tool_id=log.tool_id,
                employee_id=log.employee_id,
                session_id=log.session_id,
                visitor_id=log.visitor_id,
                input_data=_parse(log.input_data),
                output_data=_parse(log.output_data),
                status=log.status,
                error_message=log.error_message,
                execution_time_ms=log.execution_time_ms,
                created_at=log.created_at.isoformat() if log.created_at else None,
            )
            for log in logs
        ],
        count=total,
    )


# ── Employee Tool Assignment ────────────────────────────────────────────────

@router.post("/employees/{employee_id}/tools/{tool_db_id}", status_code=201)
async def assign_tool(employee_id: str, tool_db_id: str, request: Request):
    """Assign a tool to an employee."""
    db = _get_db(request)

    async with db.session() as session:
        # Check for existing assignment
        existing = await session.execute(
            select(EmployeeTools).where(
                EmployeeTools.employee_id == employee_id,
                EmployeeTools.tool_id == tool_db_id,
            )
        )
        if existing.scalar_one_or_none():
            raise HTTPException(status_code=409, detail="Tool already assigned")

        assignment = EmployeeTools(
            employee_id=employee_id,
            tool_id=tool_db_id,
            enabled=True,
        )
        session.add(assignment)
        await session.commit()

    return {"status": "assigned", "employee_id": employee_id, "tool_id": tool_db_id}


@router.delete("/employees/{employee_id}/tools/{tool_db_id}", status_code=204)
async def unassign_tool(employee_id: str, tool_db_id: str, request: Request):
    """Remove a tool assignment from an employee."""
    db = _get_db(request)

    async with db.session() as session:
        result = await session.execute(
            delete(EmployeeTools).where(
                EmployeeTools.employee_id == employee_id,
                EmployeeTools.tool_id == tool_db_id,
            )
        )
        await session.commit()

        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="Assignment not found")
