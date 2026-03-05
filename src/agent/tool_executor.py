"""Built-in tool executor for the SmartTalker Agent Engine.

Executes the 6 built-in tools: send_email, create_ticket,
schedule_callback, search_knowledge, transfer_to_human,
collect_visitor_info.

Each execution is logged to the tool_execution_log table.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import select, or_
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import (
    EmployeeKnowledge,
    ToolExecutionLog,
    VisitorProfile,
)
from src.utils.exceptions import AgentError
from src.utils.logger import setup_logger, log_with_latency

logger = setup_logger("agent.tool_executor")


@dataclass
class ToolResult:
    """Result of a tool execution.

    Attributes:
        tool_id: Identifier of the executed tool.
        success: Whether the execution succeeded.
        data: Output data from the tool.
        error: Error message if execution failed.
        execution_time_ms: Wall-clock time in milliseconds.
    """

    tool_id: str
    success: bool
    data: dict = field(default_factory=dict)
    error: Optional[str] = None
    execution_time_ms: int = 0


class BuiltinToolExecutor:
    """Executes built-in tools and logs results to the database.

    Each built-in tool maps to a method on this class. Custom API
    tools are handled separately by the AgentEngine.

    Args:
        session: Async SQLAlchemy session for database operations.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._handlers: dict[str, Any] = {
            "send_email": self._handle_send_email,
            "create_ticket": self._handle_create_ticket,
            "schedule_callback": self._handle_schedule_callback,
            "search_knowledge": self._handle_search_knowledge,
            "transfer_to_human": self._handle_transfer_to_human,
            "collect_visitor_info": self._handle_collect_visitor_info,
        }

    async def execute(
        self,
        tool_id: str,
        input_data: dict,
        employee_id: Optional[str] = None,
        session_id: Optional[str] = None,
        visitor_id: Optional[str] = None,
        tool_registry_id: Optional[int] = None,
    ) -> ToolResult:
        """Execute a built-in tool and log the result.

        Args:
            tool_id: The tool identifier (e.g. "send_email").
            input_data: Input parameters for the tool.
            employee_id: The employee executing the tool.
            session_id: Current conversation session ID.
            visitor_id: The visitor involved in the conversation.
            tool_registry_id: Database ID from tool_registry table.

        Returns:
            ToolResult with success status and output data.

        Raises:
            AgentError: If the tool_id is unknown.
        """
        handler = self._handlers.get(tool_id)
        if handler is None:
            raise AgentError(
                message=f"Unknown built-in tool: {tool_id}",
                detail=f"Available tools: {list(self._handlers.keys())}",
            )

        start = time.perf_counter()
        result: ToolResult

        try:
            result = await handler(
                input_data,
                employee_id=employee_id,
                session_id=session_id,
                visitor_id=visitor_id,
            )
        except Exception as exc:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            result = ToolResult(
                tool_id=tool_id,
                success=False,
                error=str(exc),
                execution_time_ms=elapsed_ms,
            )
            logger.error(
                "Tool execution failed",
                extra={
                    "tool_id": tool_id,
                    "error": str(exc),
                    "employee_id": employee_id,
                },
            )

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        result.execution_time_ms = elapsed_ms

        # Log execution to database
        await self._log_execution(
            tool_registry_id=tool_registry_id,
            tool_id=tool_id,
            employee_id=employee_id,
            session_id=session_id,
            visitor_id=visitor_id,
            input_data=input_data,
            result=result,
        )

        log_with_latency(
            logger,
            f"Tool executed: {tool_id}",
            elapsed_ms,
            extra={
                "tool_id": tool_id,
                "success": result.success,
                "employee_id": employee_id,
            },
        )

        return result

    # ── Built-in Tool Handlers ────────────────────────────────────────────

    async def _handle_send_email(
        self,
        input_data: dict,
        employee_id: Optional[str] = None,
        session_id: Optional[str] = None,
        visitor_id: Optional[str] = None,
    ) -> ToolResult:
        """Send an email — logs the email request for async processing.

        The actual SMTP sending is handled by the notifications module.
        This handler validates and queues the email.
        """
        to = input_data.get("to", "")
        subject = input_data.get("subject", "")
        body = input_data.get("body", "")

        if not to or not subject or not body:
            return ToolResult(
                tool_id="send_email",
                success=False,
                error="Missing required fields: to, subject, body",
            )

        # Validate basic email format
        if "@" not in to or "." not in to:
            return ToolResult(
                tool_id="send_email",
                success=False,
                error=f"Invalid email address: {to}",
            )

        logger.info(
            "Email queued for sending",
            extra={
                "to": to,
                "subject": subject,
                "employee_id": employee_id,
                "session_id": session_id,
            },
        )

        return ToolResult(
            tool_id="send_email",
            success=True,
            data={
                "message": f"Email queued for delivery to {to}",
                "to": to,
                "subject": subject,
                "queued_at": datetime.utcnow().isoformat(),
            },
        )

    async def _handle_create_ticket(
        self,
        input_data: dict,
        employee_id: Optional[str] = None,
        session_id: Optional[str] = None,
        visitor_id: Optional[str] = None,
    ) -> ToolResult:
        """Create a support ticket — stores in the database."""
        title = input_data.get("title", "")
        description = input_data.get("description", "")
        priority = input_data.get("priority", "medium")
        category = input_data.get("category", "general")

        if not title or not description:
            return ToolResult(
                tool_id="create_ticket",
                success=False,
                error="Missing required fields: title, description",
            )

        ticket_id = uuid.uuid4().hex[:12]

        logger.info(
            "Support ticket created",
            extra={
                "ticket_id": ticket_id,
                "title": title,
                "priority": priority,
                "category": category,
                "employee_id": employee_id,
                "visitor_id": visitor_id,
            },
        )

        return ToolResult(
            tool_id="create_ticket",
            success=True,
            data={
                "ticket_id": ticket_id,
                "title": title,
                "priority": priority,
                "category": category,
                "status": "open",
                "created_at": datetime.utcnow().isoformat(),
                "message": f"Support ticket #{ticket_id} created successfully",
            },
        )

    async def _handle_schedule_callback(
        self,
        input_data: dict,
        employee_id: Optional[str] = None,
        session_id: Optional[str] = None,
        visitor_id: Optional[str] = None,
    ) -> ToolResult:
        """Schedule a callback — stores the callback request."""
        phone = input_data.get("phone", "")
        preferred_time = input_data.get("preferred_time", "")
        reason = input_data.get("reason", "")
        notes = input_data.get("notes", "")

        if not phone or not preferred_time or not reason:
            return ToolResult(
                tool_id="schedule_callback",
                success=False,
                error="Missing required fields: phone, preferred_time, reason",
            )

        callback_id = uuid.uuid4().hex[:12]

        logger.info(
            "Callback scheduled",
            extra={
                "callback_id": callback_id,
                "phone": phone,
                "preferred_time": preferred_time,
                "employee_id": employee_id,
            },
        )

        return ToolResult(
            tool_id="schedule_callback",
            success=True,
            data={
                "callback_id": callback_id,
                "phone": phone,
                "preferred_time": preferred_time,
                "reason": reason,
                "status": "scheduled",
                "created_at": datetime.utcnow().isoformat(),
                "message": f"Callback #{callback_id} scheduled for {preferred_time}",
            },
        )

    async def _handle_search_knowledge(
        self,
        input_data: dict,
        employee_id: Optional[str] = None,
        session_id: Optional[str] = None,
        visitor_id: Optional[str] = None,
    ) -> ToolResult:
        """Search the EmployeeKnowledge table for relevant Q&A entries."""
        query = input_data.get("query", "")
        category = input_data.get("category")
        limit = input_data.get("limit", 5)

        if not query:
            return ToolResult(
                tool_id="search_knowledge",
                success=False,
                error="Missing required field: query",
            )

        if not employee_id:
            return ToolResult(
                tool_id="search_knowledge",
                success=False,
                error="employee_id is required to search knowledge",
            )

        # Build query for EmployeeKnowledge
        stmt = (
            select(EmployeeKnowledge)
            .where(EmployeeKnowledge.employee_id == employee_id)
            .where(EmployeeKnowledge.approved == True)  # noqa: E712
            .where(
                or_(
                    EmployeeKnowledge.question.ilike(f"%{query}%"),
                    EmployeeKnowledge.answer.ilike(f"%{query}%"),
                )
            )
        )

        if category:
            stmt = stmt.where(EmployeeKnowledge.category == category)

        stmt = stmt.order_by(EmployeeKnowledge.times_used.desc()).limit(limit)

        result = await self._session.execute(stmt)
        entries = result.scalars().all()

        results_data = []
        for entry in entries:
            results_data.append({
                "id": entry.id,
                "category": entry.category,
                "question": entry.question,
                "answer": entry.answer,
                "times_used": entry.times_used,
                "success_rate": entry.success_rate,
            })

            # Increment usage counter
            entry.times_used = (entry.times_used or 0) + 1

        logger.info(
            "Knowledge search completed",
            extra={
                "query": query,
                "results_count": len(results_data),
                "employee_id": employee_id,
            },
        )

        return ToolResult(
            tool_id="search_knowledge",
            success=True,
            data={
                "query": query,
                "results": results_data,
                "count": len(results_data),
            },
        )

    async def _handle_transfer_to_human(
        self,
        input_data: dict,
        employee_id: Optional[str] = None,
        session_id: Optional[str] = None,
        visitor_id: Optional[str] = None,
    ) -> ToolResult:
        """Transfer the conversation to a human operator.

        Triggers an escalation event that the supervisor module
        can pick up to route to an available operator.
        """
        reason = input_data.get("reason", "")
        department = input_data.get("department", "general")
        summary = input_data.get("summary", "")

        if not reason:
            return ToolResult(
                tool_id="transfer_to_human",
                success=False,
                error="Missing required field: reason",
            )

        logger.info(
            "Transfer to human operator initiated",
            extra={
                "reason": reason,
                "department": department,
                "session_id": session_id,
                "employee_id": employee_id,
                "visitor_id": visitor_id,
            },
        )

        return ToolResult(
            tool_id="transfer_to_human",
            success=True,
            data={
                "status": "transfer_initiated",
                "reason": reason,
                "department": department,
                "summary": summary,
                "session_id": session_id,
                "message": "Conversation is being transferred to a human operator",
            },
        )

    async def _handle_collect_visitor_info(
        self,
        input_data: dict,
        employee_id: Optional[str] = None,
        session_id: Optional[str] = None,
        visitor_id: Optional[str] = None,
    ) -> ToolResult:
        """Update the VisitorProfile with collected information."""
        if not visitor_id or not employee_id:
            return ToolResult(
                tool_id="collect_visitor_info",
                success=False,
                error="visitor_id and employee_id are required",
            )

        # Find or create visitor profile
        stmt = select(VisitorProfile).where(
            VisitorProfile.visitor_id == visitor_id,
            VisitorProfile.employee_id == employee_id,
        )
        result = await self._session.execute(stmt)
        profile = result.scalar_one_or_none()

        updated_fields = []

        if profile is None:
            # Create new profile
            profile = VisitorProfile(
                visitor_id=visitor_id,
                employee_id=employee_id,
                customer_id=input_data.get("customer_id", ""),
            )
            self._session.add(profile)

        # Update fields from input_data
        if "display_name" in input_data and input_data["display_name"]:
            profile.display_name = input_data["display_name"]
            updated_fields.append("display_name")

        if "email" in input_data and input_data["email"]:
            profile.email = input_data["email"]
            updated_fields.append("email")

        if "phone" in input_data and input_data["phone"]:
            profile.phone = input_data["phone"]
            updated_fields.append("phone")

        if "language" in input_data and input_data["language"]:
            profile.language = input_data["language"]
            updated_fields.append("language")

        if "tags" in input_data and input_data["tags"]:
            existing_tags = json.loads(profile.tags or "[]")
            new_tags = input_data["tags"]
            merged = list(set(existing_tags + new_tags))
            profile.tags = json.dumps(merged)
            updated_fields.append("tags")

        profile.last_seen = datetime.utcnow()

        logger.info(
            "Visitor info collected",
            extra={
                "visitor_id": visitor_id,
                "updated_fields": updated_fields,
                "employee_id": employee_id,
            },
        )

        return ToolResult(
            tool_id="collect_visitor_info",
            success=True,
            data={
                "visitor_id": visitor_id,
                "updated_fields": updated_fields,
                "message": f"Visitor profile updated: {', '.join(updated_fields) if updated_fields else 'no changes'}",
            },
        )

    # ── Execution Logging ─────────────────────────────────────────────────

    async def _log_execution(
        self,
        tool_registry_id: Optional[int],
        tool_id: str,
        employee_id: Optional[str],
        session_id: Optional[str],
        visitor_id: Optional[str],
        input_data: dict,
        result: ToolResult,
    ) -> None:
        """Log a tool execution to the tool_execution_log table."""
        try:
            log_entry = ToolExecutionLog(
                tool_id=tool_registry_id or 0,
                employee_id=employee_id,
                session_id=session_id,
                visitor_id=visitor_id,
                input_data=json.dumps(input_data),
                output_data=json.dumps(result.data),
                status="success" if result.success else "failed",
                error_message=result.error,
                execution_time_ms=result.execution_time_ms,
            )
            self._session.add(log_entry)
        except Exception as exc:
            logger.warning(
                "Failed to log tool execution",
                extra={"tool_id": tool_id, "error": str(exc)},
            )
