"""Core Agent Engine for the SmartTalker AI Employee system.

The AgentEngine orchestrates the full agent loop:
1. Receive a visitor message
2. Build context (persona, KB, memory, tools)
3. Call the LLM with function-calling support
4. Execute any tool calls the LLM requests
5. Return the final response to the visitor
6. Update visitor memory and queue learning

Uses Qwen via DashScope (OpenAI-compatible API) with function calling,
the existing LLMEngine for raw API calls, and the Database for persistence.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Callable, Awaitable

from src.agent.builtin_tools import BUILTIN_TOOLS, get_openai_tool_definitions
from src.agent.tool_executor import BuiltinToolExecutor, ToolResult
from src.config import Settings, get_settings
from src.db.engine import Database
from src.db.models import (
    Employee,
    EmployeeKnowledge,
    EmployeeLearning,
    EmployeeTools,
    ToolRegistry,
    VisitorMemory,
    VisitorProfile,
)
from src.utils.exceptions import AgentError
from src.utils.logger import setup_logger, log_with_latency

# Type alias for the optional confirmation callback.
# Called with (tool_id, tool_name, description, parameters) when a tool
# has requires_confirmation=True. Returns True if visitor approves.
ConfirmationCallback = Callable[[str, str, str, dict], Awaitable[bool]]

logger = setup_logger("agent.engine")

# Maximum tool call iterations to prevent infinite loops
_MAX_TOOL_ITERATIONS = 5

# Maximum memory entries to include in context
_MAX_MEMORY_ENTRIES = 20

# Maximum KB entries to include in context
_MAX_KB_ENTRIES = 10


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class AgentContext:
    """Assembled context for a single agent turn.

    Attributes:
        employee: The AI employee handling this conversation.
        system_prompt: Fully assembled system prompt with persona + guardrails.
        knowledge_entries: Relevant KB Q&A entries for context injection.
        memories: Visitor-specific memories for personalization.
        tools: OpenAI-format tool definitions available for this employee.
        tool_registry_map: Mapping of tool_id string to ToolRegistry DB object.
        visitor_profile: The visitor's profile if available.
        conversation_history: Recent message history for this session.
    """

    employee: Employee
    system_prompt: str
    knowledge_entries: list[dict] = field(default_factory=list)
    memories: list[dict] = field(default_factory=list)
    tools: list[dict] = field(default_factory=list)
    tool_registry_map: dict[str, ToolRegistry] = field(default_factory=dict)
    visitor_profile: Optional[VisitorProfile] = None
    conversation_history: list[dict[str, str]] = field(default_factory=list)


@dataclass
class AIDecision:
    """The LLM's decision for a single turn.

    Attributes:
        response_text: The text response (if no tool call).
        tool_calls: List of tool calls the LLM wants to make.
        finish_reason: Why the LLM stopped (stop, tool_calls, length).
        tokens_used: Total tokens consumed.
        latency_ms: LLM call latency in milliseconds.
    """

    response_text: Optional[str] = None
    tool_calls: list[dict] = field(default_factory=list)
    finish_reason: str = "stop"
    tokens_used: int = 0
    latency_ms: int = 0


# =============================================================================
# Session State (in-memory per conversation)
# =============================================================================


@dataclass
class _SessionState:
    """Internal per-session state for the agent loop."""

    employee_id: str
    visitor_id: str
    customer_id: str
    history: list[dict[str, str]] = field(default_factory=list)
    tool_call_count: int = 0
    last_access: float = 0.0

    # Per-session cache (avoid DB queries every turn)
    cached_employee: Optional[Any] = None
    cached_tools: Optional[list[dict]] = None
    cached_tool_registry_map: Optional[dict] = None


# =============================================================================
# Agent Engine
# =============================================================================


class AgentEngine:
    """Core agent engine for AI employee conversations.

    Orchestrates the full agent loop: context assembly, LLM calls
    with function calling, tool execution, memory updates, and
    learning queue.

    Args:
        db: Database instance for persistence.
        config: Application settings. Uses singleton if not provided.
    """

    def __init__(
        self,
        db: Database,
        config: Optional[Settings] = None,
    ) -> None:
        self._db = db
        self._config = config or get_settings()

        # LLM API configuration (direct HTTP, not via LLMEngine,
        # to support function calling parameters)
        self._llm_base_url = self._config.llm_base_url
        self._llm_model = self._config.llm_model_name
        self._llm_api_key = self._config.llm_api_key
        self._llm_timeout = self._config.llm_timeout
        self._llm_max_tokens = self._config.llm_max_tokens
        self._llm_temperature = self._config.llm_temperature

        # HTTP client for LLM API calls
        self._client: Optional[httpx.AsyncClient] = None

        # In-memory session state
        self._sessions: dict[str, _SessionState] = {}

        logger.info(
            "AgentEngine initialized",
            extra={"model": self._llm_model, "base_url": self._llm_base_url},
        )

    # ── HTTP Client ───────────────────────────────────────────────────────

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the async HTTP client for the LLM API."""
        if self._client is None or self._client.is_closed:
            headers = {"Content-Type": "application/json"}
            if self._llm_api_key:
                headers["Authorization"] = f"Bearer {self._llm_api_key}"
            self._client = httpx.AsyncClient(
                base_url=self._llm_base_url,
                timeout=httpx.Timeout(self._llm_timeout, connect=10.0),
                headers=headers,
            )
        return self._client

    # ── Main Entry Point ──────────────────────────────────────────────────

    async def handle_message(
        self,
        session_id: str,
        visitor_message: str,
        employee_id: Optional[str] = None,
        visitor_id: Optional[str] = None,
        customer_id: Optional[str] = None,
        confirmation_callback: Optional[ConfirmationCallback] = None,
    ) -> str:
        """Handle an incoming visitor message through the full agent loop.

        This is the main entry point. It:
        1. Loads or creates the session state
        2. Builds the full agent context
        3. Calls the LLM (with tool-calling loop)
        4. Updates visitor memory
        5. Queues learning entries
        6. Returns the final response text

        Args:
            session_id: Unique conversation session identifier.
            visitor_message: The visitor's text message.
            employee_id: The AI employee to use. Required on first call.
            visitor_id: The visitor identifier. Defaults to session_id.
            customer_id: The customer (tenant) identifier.

        Returns:
            The AI employee's text response.

        Raises:
            AgentError: If employee not found or LLM call fails.
        """
        if not visitor_message.strip():
            raise AgentError(message="Visitor message cannot be empty")

        start = time.perf_counter()

        # Resolve session state
        session = self._get_or_create_session(
            session_id, employee_id, visitor_id, customer_id
        )

        async with self._db.session() as db_session:
            # Load employee (cached after first load per session)
            if session.cached_employee is not None:
                employee = session.cached_employee
            else:
                employee = await self._load_employee(db_session, session.employee_id)
                if employee is None:
                    raise AgentError(
                        message=f"Employee not found: {session.employee_id}",
                        detail="Ensure the employee_id exists in the employees table",
                    )
                session.cached_employee = employee

            # Build context
            context = await self.build_context(
                db_session, employee, session, visitor_message
            )

            # Add visitor message to history
            session.history.append({"role": "user", "content": visitor_message})

            # Run the agent loop (LLM + tool calls)
            final_response = await self._agent_loop(
                db_session, context, session, visitor_message,
                confirmation_callback=confirmation_callback,
            )

            # Add response to history
            session.history.append({"role": "assistant", "content": final_response})

            # Trim history to prevent unbounded growth
            max_history = self._config.llm_max_history * 2
            if len(session.history) > max_history:
                session.history = session.history[-max_history:]

            # Post-processing: memory + learning (non-blocking errors)
            try:
                await self.update_visitor_memory(
                    db_session, session, visitor_message, final_response
                )
            except Exception as exc:
                logger.warning(
                    "Failed to update visitor memory",
                    extra={"error": str(exc), "session_id": session_id},
                )

            try:
                await self.queue_learning(
                    db_session, session, visitor_message, final_response
                )
            except Exception as exc:
                logger.warning(
                    "Failed to queue learning",
                    extra={"error": str(exc), "session_id": session_id},
                )

            await db_session.commit()

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        log_with_latency(
            logger,
            "Agent message handled",
            elapsed_ms,
            extra={
                "session_id": session_id,
                "employee_id": session.employee_id,
                "response_length": len(final_response),
            },
        )

        return final_response

    # ── Session Management ────────────────────────────────────────────────

    def _get_or_create_session(
        self,
        session_id: str,
        employee_id: Optional[str],
        visitor_id: Optional[str],
        customer_id: Optional[str],
    ) -> _SessionState:
        """Get existing session or create a new one."""
        if session_id in self._sessions:
            session = self._sessions[session_id]
            session.last_access = time.time()
            return session

        if not employee_id:
            raise AgentError(
                message="employee_id is required for new sessions",
                detail=f"session_id={session_id}",
            )

        session = _SessionState(
            employee_id=employee_id,
            visitor_id=visitor_id or session_id,
            customer_id=customer_id or "",
            last_access=time.time(),
        )
        self._sessions[session_id] = session

        # Evict expired sessions (idle > 30 min)
        now = time.time()
        expired = [
            sid for sid, s in self._sessions.items()
            if now - s.last_access > 1800
        ]
        for sid in expired:
            del self._sessions[sid]

        return session

    def clear_session(self, session_id: str) -> None:
        """Clear the session state for a given session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info("Agent session cleared", extra={"session_id": session_id})

    # ── Context Building ──────────────────────────────────────────────────

    async def build_context(
        self,
        db_session: AsyncSession,
        employee: Employee,
        session: _SessionState,
        visitor_message: str,
    ) -> AgentContext:
        """Build the full agent context for an LLM call.

        Assembles: system prompt (persona + guardrails), relevant KB
        entries, visitor memories, available tools, and visitor profile.

        Args:
            db_session: Active database session.
            employee: The AI employee model.
            session: Current session state.
            visitor_message: The visitor's current message.

        Returns:
            AgentContext with all assembled information.
        """
        # 1. Build system prompt from employee persona
        system_prompt = self._build_system_prompt(employee)

        # 2. Load relevant knowledge entries
        knowledge_entries = await self._load_knowledge(
            db_session, employee.id, visitor_message
        )

        # 3. Load visitor memories
        memories = await self._load_visitor_memories(
            db_session, session.visitor_id, employee.id
        )

        # 4. Load available tools (cached per session)
        if session.cached_tools is not None and session.cached_tool_registry_map is not None:
            tools = session.cached_tools
            tool_registry_map = session.cached_tool_registry_map
        else:
            tools, tool_registry_map = await self._load_tools(
                db_session, employee.id
            )
            session.cached_tools = tools
            session.cached_tool_registry_map = tool_registry_map

        # 5. Load visitor profile
        visitor_profile = await self._load_visitor_profile(
            db_session, session.visitor_id, employee.id
        )

        # 6. Inject KB context into system prompt
        if knowledge_entries:
            kb_text = "\n\n".join(
                f"Q: {e['question']}\nA: {e['answer']}"
                for e in knowledge_entries
            )
            system_prompt += (
                "\n\n--- Knowledge Base ---\n"
                "Use the following Q&A pairs to help answer the visitor's question. "
                "If a match is relevant, base your answer on it.\n\n"
                f"{kb_text}"
            )

        # 7. Inject memories into system prompt
        if memories:
            mem_text = "\n".join(
                f"- [{m['type']}] {m['content']}" for m in memories
            )
            system_prompt += (
                "\n\n--- Visitor Memory ---\n"
                "Previous information about this visitor:\n"
                f"{mem_text}"
            )

        # 8. Inject visitor profile info
        if visitor_profile and visitor_profile.display_name:
            system_prompt += (
                f"\n\nThe visitor's name is {visitor_profile.display_name}. "
                "Address them by name when appropriate."
            )

        return AgentContext(
            employee=employee,
            system_prompt=system_prompt,
            knowledge_entries=knowledge_entries,
            memories=memories,
            tools=tools,
            tool_registry_map=tool_registry_map,
            visitor_profile=visitor_profile,
            conversation_history=list(session.history),
        )

    def _build_system_prompt(self, employee: Employee) -> str:
        """Build the base system prompt from employee persona."""
        parts = []

        # Core identity
        parts.append(f"You are {employee.name}")
        if employee.role_title:
            parts.append(f", a {employee.role_title}")
        parts.append(".")

        # Role description
        if employee.role_description:
            parts.append(f" {employee.role_description}")

        # Language instruction
        lang = employee.language or "en"
        if lang == "ar":
            parts.append(" Respond in Arabic (العربية).")
        elif lang == "en":
            parts.append(" Respond in English.")
        elif lang == "fr":
            parts.append(" Respond in French (Fran\u00e7ais).")
        elif lang == "tr":
            parts.append(" Respond in Turkish (T\u00fcrk\u00e7e).")
        else:
            parts.append(f" Respond in the language code: {lang}.")

        # Personality traits
        personality = self._parse_json_field(employee.personality)
        if personality:
            tone = personality.get("tone", "")
            style = personality.get("style", "")
            if tone:
                parts.append(f" Your tone is {tone}.")
            if style:
                parts.append(f" Your communication style is {style}.")

        # Guardrails
        guardrails = self._parse_json_field(employee.guardrails)
        if guardrails:
            blocked = guardrails.get("blocked_topics", [])
            if blocked:
                topics_str = ", ".join(blocked)
                parts.append(
                    f" Never discuss the following topics: {topics_str}."
                )
            max_length = guardrails.get("max_response_length")
            if max_length:
                parts.append(
                    f" Keep responses under {max_length} characters."
                )

        # General instructions
        parts.append(
            "\n\nBe helpful, professional, and concise. "
            "Use the tools available to you when they can help the visitor. "
            "If you do not know the answer, say so honestly rather than guessing."
        )

        return "".join(parts)

    @staticmethod
    def _parse_json_field(value: Optional[str]) -> dict:
        """Safely parse a JSON string field, returning empty dict on failure."""
        if not value:
            return {}
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}

    # ── Data Loading ──────────────────────────────────────────────────────

    async def _load_employee(
        self, db_session: AsyncSession, employee_id: str
    ) -> Optional[Employee]:
        """Load an employee by ID."""
        stmt = select(Employee).where(Employee.id == employee_id)
        result = await db_session.execute(stmt)
        return result.scalar_one_or_none()

    async def _load_knowledge(
        self,
        db_session: AsyncSession,
        employee_id: str,
        query: str,
    ) -> list[dict]:
        """Load relevant knowledge entries via keyword matching.

        Uses SQL LIKE matching on question and answer fields.
        For production, this would be replaced by vector similarity search.
        """
        # Extract keywords (simple word tokenization)
        keywords = [w.strip() for w in query.split() if len(w.strip()) > 2]

        if not keywords:
            return []

        # Build OR conditions for keyword matching
        from sqlalchemy import or_

        conditions = []
        for kw in keywords[:5]:  # Limit to first 5 keywords
            conditions.append(EmployeeKnowledge.question.ilike(f"%{kw}%"))
            conditions.append(EmployeeKnowledge.answer.ilike(f"%{kw}%"))

        stmt = (
            select(EmployeeKnowledge)
            .where(EmployeeKnowledge.employee_id == employee_id)
            .where(EmployeeKnowledge.approved == True)  # noqa: E712
            .where(or_(*conditions))
            .order_by(EmployeeKnowledge.times_used.desc())
            .limit(_MAX_KB_ENTRIES)
        )

        result = await db_session.execute(stmt)
        entries = result.scalars().all()

        return [
            {
                "id": e.id,
                "category": e.category,
                "question": e.question,
                "answer": e.answer,
            }
            for e in entries
        ]

    async def _load_visitor_memories(
        self,
        db_session: AsyncSession,
        visitor_id: str,
        employee_id: str,
    ) -> list[dict]:
        """Load visitor memories, filtering expired entries."""
        now = datetime.utcnow()

        stmt = (
            select(VisitorMemory)
            .where(VisitorMemory.visitor_id == visitor_id)
            .where(VisitorMemory.employee_id == employee_id)
            .where(
                (VisitorMemory.expires_at == None)  # noqa: E711
                | (VisitorMemory.expires_at > now)
            )
            .order_by(VisitorMemory.importance.desc())
            .limit(_MAX_MEMORY_ENTRIES)
        )

        result = await db_session.execute(stmt)
        entries = result.scalars().all()

        return [
            {
                "type": e.memory_type,
                "content": e.content,
                "importance": e.importance,
            }
            for e in entries
        ]

    async def _load_visitor_profile(
        self,
        db_session: AsyncSession,
        visitor_id: str,
        employee_id: str,
    ) -> Optional[VisitorProfile]:
        """Load the visitor profile if it exists."""
        stmt = select(VisitorProfile).where(
            VisitorProfile.visitor_id == visitor_id,
            VisitorProfile.employee_id == employee_id,
        )
        result = await db_session.execute(stmt)
        return result.scalar_one_or_none()

    async def _load_tools(
        self,
        db_session: AsyncSession,
        employee_id: str,
    ) -> tuple[list[dict], dict[str, ToolRegistry]]:
        """Load tools available to this employee.

        Returns both the OpenAI-format tool definitions and a mapping
        from tool_id string to ToolRegistry model.

        Returns:
            Tuple of (openai_tools_list, tool_registry_map).
        """
        # Load employee's assigned tools
        stmt = (
            select(ToolRegistry)
            .join(EmployeeTools, EmployeeTools.tool_id == ToolRegistry.id)
            .where(EmployeeTools.employee_id == employee_id)
            .where(EmployeeTools.enabled == True)  # noqa: E712
            .where(ToolRegistry.is_active == True)  # noqa: E712
        )
        result = await db_session.execute(stmt)
        db_tools = result.scalars().all()

        # Build OpenAI format + registry map
        openai_tools = []
        registry_map: dict[str, ToolRegistry] = {}

        for tool in db_tools:
            try:
                schema = json.loads(tool.input_schema)
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    "Invalid input_schema for tool",
                    extra={"tool_id": tool.tool_id},
                )
                continue

            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.tool_id,
                    "description": tool.description,
                    "parameters": schema,
                },
            })
            registry_map[tool.tool_id] = tool

        # Always include built-in tools
        for bt in BUILTIN_TOOLS:
            if bt["tool_id"] not in registry_map:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": bt["tool_id"],
                        "description": bt["description"],
                        "parameters": json.loads(bt["input_schema"])
                        if isinstance(bt["input_schema"], str)
                        else bt["input_schema"],
                    },
                })

        return openai_tools, registry_map

    # ── Agent Loop (LLM + Tool Calls) ─────────────────────────────────────

    async def _agent_loop(
        self,
        db_session: AsyncSession,
        context: AgentContext,
        session: _SessionState,
        visitor_message: str,
        confirmation_callback: Optional[ConfirmationCallback] = None,
    ) -> str:
        """Run the agent loop: call LLM, execute tools, repeat.

        Loops up to _MAX_TOOL_ITERATIONS times if the LLM keeps
        requesting tool calls. Returns the final text response.
        """
        messages = [{"role": "system", "content": context.system_prompt}]
        messages.extend(context.conversation_history)
        messages.append({"role": "user", "content": visitor_message})

        for iteration in range(_MAX_TOOL_ITERATIONS):
            decision = await self._call_llm(messages, context.tools)

            if not decision.tool_calls:
                # LLM gave a direct text response
                return decision.response_text or ""

            # Process tool calls
            for tool_call in decision.tool_calls:
                tool_name = tool_call.get("function", {}).get("name", "")
                tool_args_str = tool_call.get("function", {}).get("arguments", "{}")
                tool_call_id = tool_call.get("id", "")

                try:
                    tool_args = json.loads(tool_args_str)
                except (json.JSONDecodeError, TypeError):
                    tool_args = {}

                # Execute the tool
                tool_result = await self.execute_tool(
                    db_session=db_session,
                    tool_id=tool_name,
                    input_data=tool_args,
                    session=session,
                    context=context,
                    confirmation_callback=confirmation_callback,
                )

                # Append the assistant's tool call message
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call],
                })

                # Append the tool result as a tool message
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": json.dumps(tool_result.data if tool_result.success
                                          else {"error": tool_result.error}),
                })

                session.tool_call_count += 1

        # If we hit max iterations, make one final call without tools
        logger.warning(
            "Max tool iterations reached, forcing final response",
            extra={"session_id": session.visitor_id, "iterations": _MAX_TOOL_ITERATIONS},
        )
        decision = await self._call_llm(messages, tools=None)
        return decision.response_text or ""

    async def _call_llm(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
    ) -> AIDecision:
        """Call the LLM API with optional function-calling tools.

        Args:
            messages: The conversation messages array.
            tools: Optional OpenAI-format tool definitions.

        Returns:
            AIDecision with response text and/or tool calls.

        Raises:
            AgentError: If the API call fails.
        """
        start = time.perf_counter()

        payload: dict[str, Any] = {
            "model": self._llm_model,
            "messages": messages,
            "temperature": self._llm_temperature,
            "max_tokens": self._llm_max_tokens,
        }

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        try:
            client = await self._get_client()
            response = await client.post("/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
        except httpx.ConnectError as exc:
            raise AgentError(
                message="Cannot connect to LLM API",
                detail=f"Check that LLM API is reachable at: {self._llm_base_url}",
                original_exception=exc,
            ) from exc
        except httpx.TimeoutException as exc:
            raise AgentError(
                message=f"LLM request timed out after {self._llm_timeout}s",
                original_exception=exc,
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise AgentError(
                message=f"LLM API error: {exc.response.status_code}",
                detail=exc.response.text,
                original_exception=exc,
            ) from exc
        except Exception as exc:
            raise AgentError(
                message="LLM call failed",
                detail=str(exc),
                original_exception=exc,
            ) from exc

        elapsed_ms = int((time.perf_counter() - start) * 1000)

        # Parse response
        choices = data.get("choices", [])
        if not choices:
            raise AgentError(
                message="Empty response from LLM",
                detail=f"Raw response: {data}",
            )

        choice = choices[0]
        message = choice.get("message", {})
        finish_reason = choice.get("finish_reason", "stop")
        tokens_used = data.get("usage", {}).get("total_tokens", 0)

        # Check for tool calls
        tool_calls = message.get("tool_calls", [])

        decision = AIDecision(
            response_text=message.get("content"),
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            tokens_used=tokens_used,
            latency_ms=elapsed_ms,
        )

        log_with_latency(
            logger,
            "LLM call complete",
            elapsed_ms,
            extra={
                "finish_reason": finish_reason,
                "tokens": tokens_used,
                "tool_calls_count": len(tool_calls),
            },
        )

        return decision

    # ── Tool Execution ────────────────────────────────────────────────────

    async def execute_tool(
        self,
        db_session: AsyncSession,
        tool_id: str,
        input_data: dict,
        session: _SessionState,
        context: AgentContext,
        confirmation_callback: Optional[ConfirmationCallback] = None,
    ) -> ToolResult:
        """Execute a tool (built-in or custom API) and log the result.

        Args:
            db_session: Active database session.
            tool_id: The tool identifier string.
            input_data: Input parameters for the tool.
            session: Current session state.
            context: Current agent context.
            confirmation_callback: Optional async callback for visitor confirmation.

        Returns:
            ToolResult with success status and output data.
        """
        # If tool requires confirmation and a callback is available, ask visitor
        if tool_id in context.tool_registry_map:
            tool_reg = context.tool_registry_map[tool_id]
            if getattr(tool_reg, "requires_confirmation", False) and confirmation_callback:
                approved = await confirmation_callback(
                    tool_id,
                    tool_reg.name,
                    tool_reg.description,
                    input_data,
                )
                if not approved:
                    return ToolResult(
                        tool_id=tool_id,
                        success=False,
                        error="Visitor declined tool execution",
                    )

        # Check if it's a built-in tool
        builtin_ids = {bt["tool_id"] for bt in BUILTIN_TOOLS}

        if tool_id in builtin_ids:
            executor = BuiltinToolExecutor(db_session)
            registry_id = None
            if tool_id in context.tool_registry_map:
                registry_id = context.tool_registry_map[tool_id].id

            return await executor.execute(
                tool_id=tool_id,
                input_data=input_data,
                employee_id=session.employee_id,
                session_id=session.visitor_id,
                visitor_id=session.visitor_id,
                tool_registry_id=registry_id,
            )

        # Check for custom API tool in registry
        if tool_id in context.tool_registry_map:
            tool = context.tool_registry_map[tool_id]
            return await self.call_custom_api(tool, input_data, session)

        # Unknown tool
        logger.warning(
            "Unknown tool requested by LLM",
            extra={"tool_id": tool_id, "employee_id": session.employee_id},
        )
        return ToolResult(
            tool_id=tool_id,
            success=False,
            error=f"Unknown tool: {tool_id}",
        )

    async def call_custom_api(
        self,
        tool: ToolRegistry,
        input_data: dict,
        session: _SessionState,
    ) -> ToolResult:
        """Call a customer's custom API tool.

        Constructs an HTTP request from the tool's configuration,
        applies input data via the body template, and maps the response.

        Args:
            tool: The ToolRegistry model with API configuration.
            input_data: Input parameters from the LLM.
            session: Current session state.

        Returns:
            ToolResult with the API response data.
        """
        start = time.perf_counter()

        if not tool.api_url:
            return ToolResult(
                tool_id=tool.tool_id,
                success=False,
                error="Tool has no api_url configured",
            )

        # SSRF protection — block internal/private URLs
        from src.agent.security import validate_tool_url, sanitize_tool_input
        if not validate_tool_url(tool.api_url):
            logger.warning(
                "SSRF blocked: custom tool URL points to internal network",
                extra={"tool_id": tool.tool_id, "url": tool.api_url},
            )
            return ToolResult(
                tool_id=tool.tool_id,
                success=False,
                error=f"URL blocked by SSRF protection: {tool.api_url}",
            )
        input_data = sanitize_tool_input(input_data)

        method = (tool.api_method or "POST").upper()
        headers = self._parse_json_field(tool.api_headers)
        timeout_s = (tool.timeout_ms or 5000) / 1000.0

        # Build request body from template
        body = None
        if tool.api_body_template:
            try:
                template = json.loads(tool.api_body_template)
                body = self._apply_template(template, input_data)
            except (json.JSONDecodeError, TypeError):
                body = input_data
        else:
            body = input_data

        try:
            async with httpx.AsyncClient(timeout=timeout_s) as client:
                response = await client.request(
                    method=method,
                    url=tool.api_url,
                    headers=headers,
                    json=body if method in ("POST", "PUT", "PATCH") else None,
                    params=input_data if method == "GET" else None,
                )
                response.raise_for_status()
                response_data = response.json()
        except httpx.TimeoutException:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            return ToolResult(
                tool_id=tool.tool_id,
                success=False,
                error=f"Custom API timed out after {timeout_s}s",
                execution_time_ms=elapsed_ms,
            )
        except httpx.HTTPStatusError as exc:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            return ToolResult(
                tool_id=tool.tool_id,
                success=False,
                error=f"Custom API error: {exc.response.status_code} — {exc.response.text[:200]}",
                execution_time_ms=elapsed_ms,
            )
        except Exception as exc:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            return ToolResult(
                tool_id=tool.tool_id,
                success=False,
                error=f"Custom API call failed: {str(exc)}",
                execution_time_ms=elapsed_ms,
            )

        elapsed_ms = int((time.perf_counter() - start) * 1000)

        # Apply response mapping if configured
        mapped_data = response_data
        if tool.response_mapping:
            try:
                mapping = json.loads(tool.response_mapping)
                mapped_data = self._apply_response_mapping(response_data, mapping)
            except (json.JSONDecodeError, TypeError):
                pass

        log_with_latency(
            logger,
            f"Custom API call: {tool.tool_id}",
            elapsed_ms,
            extra={
                "api_url": tool.api_url,
                "method": method,
                "status": "success",
            },
        )

        return ToolResult(
            tool_id=tool.tool_id,
            success=True,
            data=mapped_data,
            execution_time_ms=elapsed_ms,
        )

    @staticmethod
    def _apply_template(template: dict, data: dict) -> dict:
        """Apply input data to a body template using {{key}} substitution."""
        result = {}
        for key, value in template.items():
            if isinstance(value, str) and "{{" in value:
                for data_key, data_val in data.items():
                    value = value.replace(f"{{{{{data_key}}}}}", str(data_val))
                result[key] = value
            elif isinstance(value, dict):
                result[key] = AgentEngine._apply_template(value, data)
            else:
                result[key] = value
        return result

    @staticmethod
    def _apply_response_mapping(response: dict, mapping: dict) -> dict:
        """Extract fields from API response using a mapping dict.

        Mapping format: {"output_key": "response.nested.key"}
        """
        result = {}
        for output_key, path in mapping.items():
            parts = path.split(".") if isinstance(path, str) else [path]
            value: Any = response
            for part in parts:
                if isinstance(value, dict):
                    value = value.get(part)
                else:
                    value = None
                    break
            result[output_key] = value
        return result

    # ── Visitor Memory ────────────────────────────────────────────────────

    async def update_visitor_memory(
        self,
        db_session: AsyncSession,
        session: _SessionState,
        visitor_msg: str,
        ai_response: str,
    ) -> None:
        """Extract and store relevant memories from the conversation turn.

        Identifies visitor preferences, facts, and important details
        from the exchange and persists them for future personalization.

        Args:
            db_session: Active database session.
            session: Current session state.
            visitor_msg: The visitor's message.
            ai_response: The AI employee's response.
        """
        # Simple heuristic-based memory extraction
        # In production, this would use the LLM for entity extraction
        memories_to_add = []

        # Detect self-introduction patterns
        lower_msg = visitor_msg.lower()

        # Name detection
        name_prefixes = ["my name is", "i'm", "i am", "call me", "this is"]
        for prefix in name_prefixes:
            if prefix in lower_msg:
                idx = lower_msg.index(prefix) + len(prefix)
                remaining = visitor_msg[idx:].strip().split(".")[0].split(",")[0]
                name = remaining.strip()
                if name and len(name) < 50:
                    memories_to_add.append({
                        "type": "identity",
                        "content": f"Visitor's name: {name}",
                        "importance": 0.9,
                    })
                break

        # Preference detection
        pref_prefixes = ["i prefer", "i like", "i want", "i need", "i'd like"]
        for prefix in pref_prefixes:
            if prefix in lower_msg:
                idx = lower_msg.index(prefix)
                pref_text = visitor_msg[idx:].strip().split(".")[0]
                if pref_text and len(pref_text) < 200:
                    memories_to_add.append({
                        "type": "preference",
                        "content": pref_text,
                        "importance": 0.6,
                    })
                break

        # Contact info detection
        if "@" in visitor_msg and "." in visitor_msg:
            # Likely contains an email
            words = visitor_msg.split()
            for word in words:
                if "@" in word and "." in word:
                    memories_to_add.append({
                        "type": "contact",
                        "content": f"Email: {word.strip('.,;:!?')}",
                        "importance": 0.8,
                    })
                    break

        # Store memories
        for mem in memories_to_add:
            memory = VisitorMemory(
                visitor_id=session.visitor_id,
                employee_id=session.employee_id,
                memory_type=mem["type"],
                content=mem["content"],
                source_session=session.visitor_id,
                importance=mem["importance"],
            )
            db_session.add(memory)

        # Update visitor profile interaction count
        stmt = select(VisitorProfile).where(
            VisitorProfile.visitor_id == session.visitor_id,
            VisitorProfile.employee_id == session.employee_id,
        )
        result = await db_session.execute(stmt)
        profile = result.scalar_one_or_none()

        if profile:
            profile.interaction_count = (profile.interaction_count or 0) + 1
            profile.last_seen = datetime.utcnow()

    # ── Learning Queue ────────────────────────────────────────────────────

    async def queue_learning(
        self,
        db_session: AsyncSession,
        session: _SessionState,
        visitor_msg: str,
        response: str,
    ) -> None:
        """Queue a learning entry for review.

        Records the conversation exchange so it can be reviewed and
        potentially added to the employee's knowledge base.

        Args:
            db_session: Active database session.
            session: Current session state.
            visitor_msg: The visitor's message.
            response: The AI employee's response.
        """
        # Only queue if the response is substantive
        if len(response) < 10 or len(visitor_msg) < 5:
            return

        learning = EmployeeLearning(
            employee_id=session.employee_id,
            customer_id=session.customer_id or None,
            learning_type="conversation_qa",
            old_value=json.dumps({"question": visitor_msg}),
            new_value=json.dumps({"answer": response}),
            confidence=0.0,
            status="pending",
            source="conversation",
        )
        db_session.add(learning)

    # ── Cleanup ───────────────────────────────────────────────────────────

    async def close(self) -> None:
        """Close the HTTP client and clean up resources."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
        self._sessions.clear()
        logger.info("AgentEngine closed")
