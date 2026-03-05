"""Workflow execution engine for multi-step visitor interactions.

Executes workflow definitions stored in the database, supporting
sequential step evaluation with pause/resume for visitor input.
Step types: ask_visitor, call_tool, ai_decision, condition,
set_variable, send_notification, wait, escalate.

Uses qwen3-max via DashScope for AI decision steps.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
from sqlalchemy import select, update

from src.config import Settings, get_settings
from src.db.engine import Database
from src.db.models import Workflow, WorkflowExecution, _uuid
from src.utils.logger import setup_logger, log_with_latency

logger = setup_logger("pipeline.workflow")


# =============================================================================
# Built-in Workflow Templates
# =============================================================================

WORKFLOW_TEMPLATES: dict[str, dict[str, Any]] = {
    "lead_capture": {
        "name": "Lead Capture",
        "description": "Collect visitor name and email, then notify the team.",
        "trigger_type": "manual",
        "trigger_config": {},
        "steps": [
            {
                "type": "ask_visitor",
                "name": "Ask Name",
                "config": {"question": "What is your name?"},
            },
            {
                "type": "ask_visitor",
                "name": "Ask Email",
                "config": {"question": "What is your email address?"},
            },
            {
                "type": "call_tool",
                "name": "Collect Visitor Info",
                "config": {
                    "url": "/api/v1/visitors/collect",
                    "method": "POST",
                    "headers": {},
                    "body": {
                        "name": "{{step_0_response}}",
                        "email": "{{step_1_response}}",
                    },
                },
            },
            {
                "type": "send_notification",
                "name": "Notify Team",
                "config": {
                    "message": "New lead captured: {{step_0_response}} ({{step_1_response}})",
                },
            },
        ],
    },
    "complaint_handler": {
        "name": "Complaint Handler",
        "description": "Capture complaint, assess severity via AI, escalate if high.",
        "trigger_type": "keyword",
        "trigger_config": {"keywords": ["complaint", "problem", "issue"]},
        "steps": [
            {
                "type": "ask_visitor",
                "name": "Ask Issue",
                "config": {"question": "Please describe your issue in detail."},
            },
            {
                "type": "ai_decision",
                "name": "Assess Severity",
                "config": {
                    "prompt": (
                        "Analyze the following customer complaint and classify its "
                        "severity as 'low', 'medium', or 'high'. Respond with ONLY "
                        "the severity level word.\n\nComplaint: {{step_0_response}}"
                    ),
                    "branches": {
                        "high": 3,
                        "medium": 4,
                        "low": 4,
                    },
                },
            },
            {
                "type": "condition",
                "name": "Check High Severity",
                "config": {
                    "condition": "step_1_decision == 'high'",
                    "true_step": 3,
                    "false_step": 4,
                },
            },
            {
                "type": "escalate",
                "name": "Escalate to Manager",
                "config": {
                    "reason": "High severity complaint requires human attention",
                    "priority": "high",
                },
            },
            {
                "type": "set_variable",
                "name": "Mark Acknowledged",
                "config": {
                    "key": "complaint_status",
                    "value": "acknowledged",
                },
            },
        ],
    },
    "appointment_scheduler": {
        "name": "Appointment Scheduler",
        "description": "Collect preferred date and time, schedule a callback.",
        "trigger_type": "intent",
        "trigger_config": {"intents": ["schedule", "appointment", "booking"]},
        "steps": [
            {
                "type": "ask_visitor",
                "name": "Ask Preferred Date",
                "config": {"question": "What date would you prefer for your appointment?"},
            },
            {
                "type": "ask_visitor",
                "name": "Ask Preferred Time",
                "config": {"question": "What time works best for you?"},
            },
            {
                "type": "call_tool",
                "name": "Schedule Callback",
                "config": {
                    "url": "/api/v1/appointments/schedule",
                    "method": "POST",
                    "headers": {},
                    "body": {
                        "date": "{{step_0_response}}",
                        "time": "{{step_1_response}}",
                    },
                },
            },
            {
                "type": "send_notification",
                "name": "Confirm Appointment",
                "config": {
                    "message": (
                        "Appointment scheduled for {{step_0_response}} "
                        "at {{step_1_response}}."
                    ),
                },
            },
        ],
    },
}

# Valid step types for validation
_VALID_STEP_TYPES = frozenset({
    "ask_visitor",
    "call_tool",
    "ai_decision",
    "condition",
    "set_variable",
    "send_notification",
    "wait",
    "escalate",
})


# =============================================================================
# WorkflowEngine
# =============================================================================


class WorkflowEngine:
    """Execute and manage multi-step workflow definitions.

    Handles sequential step evaluation with support for pausing
    (ask_visitor, wait) and resuming after visitor input.

    Args:
        db: Database instance for persistence.
        config: Application settings. Uses singleton if not provided.
    """

    def __init__(self, db: Database, config: Settings | None = None) -> None:
        self._db = db
        self._config = config or get_settings()

        # LLM settings for ai_decision steps
        self._llm_base_url = self._config.llm_base_url
        self._llm_model = self._config.llm_model_name
        self._llm_api_key = self._config.llm_api_key or self._config.dashscope_api_key
        self._llm_timeout = self._config.llm_timeout

        # Shared httpx client for tool calls and LLM requests
        self._client: Optional[httpx.AsyncClient] = None

        logger.info(
            "WorkflowEngine initialized",
            extra={"llm_model": self._llm_model},
        )

    # ── HTTP client management ───────────────────────────────────────────

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the shared async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._llm_timeout, connect=10.0),
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client and release resources."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
        logger.info("WorkflowEngine closed")

    # ── Template variable substitution ───────────────────────────────────

    @staticmethod
    def _substitute_vars(text: str, context: dict[str, Any]) -> str:
        """Replace {{variable}} placeholders with context values.

        Args:
            text: String potentially containing {{key}} placeholders.
            context: Dictionary of key-value pairs for substitution.

        Returns:
            String with placeholders replaced by their context values.
        """
        result = text
        for key, value in context.items():
            placeholder = "{{" + str(key) + "}}"
            result = result.replace(placeholder, str(value))
        return result

    def _substitute_config(
        self, config: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        """Recursively substitute template variables in a step config dict.

        Args:
            config: Step configuration dictionary.
            context: Current workflow execution context.

        Returns:
            New config dict with all string values substituted.
        """
        resolved: dict[str, Any] = {}
        for key, value in config.items():
            if isinstance(value, str):
                resolved[key] = self._substitute_vars(value, context)
            elif isinstance(value, dict):
                resolved[key] = self._substitute_config(value, context)
            else:
                resolved[key] = value
        return resolved

    # ── Workflow execution ───────────────────────────────────────────────

    async def execute_workflow(
        self,
        workflow_id: str,
        session_id: str,
        visitor_id: str,
        initial_context: dict[str, Any] | None = None,
    ) -> WorkflowExecution:
        """Start executing a workflow from its first step.

        Creates a WorkflowExecution record and iterates through the
        workflow steps sequentially. Pauses if a step requires visitor
        input (ask_visitor) or encounters an error.

        Args:
            workflow_id: ID of the Workflow definition to execute.
            session_id: Current conversation session ID.
            visitor_id: Visitor identifier for the session.
            initial_context: Optional seed data for the execution context.

        Returns:
            The WorkflowExecution record (may be running, waiting,
            completed, or failed).

        Raises:
            ValueError: If the workflow_id is not found.
        """
        start_time = time.perf_counter()

        # Load workflow definition
        async with self._db.session() as session:
            result = await session.execute(
                select(Workflow).where(Workflow.id == workflow_id)
            )
            workflow = result.scalar_one_or_none()

        if workflow is None:
            raise ValueError(f"Workflow not found: {workflow_id}")

        steps = json.loads(workflow.steps) if isinstance(workflow.steps, str) else workflow.steps
        if not steps:
            raise ValueError(f"Workflow has no steps: {workflow_id}")

        # Create execution record
        context = dict(initial_context) if initial_context else {}
        execution = WorkflowExecution(
            id=_uuid(),
            workflow_id=workflow_id,
            session_id=session_id,
            visitor_id=visitor_id,
            status="running",
            current_step=0,
            context=json.dumps(context),
            error="",
        )

        async with self._db.session() as session:
            session.add(execution)
            await session.commit()
            await session.refresh(execution)

        logger.info(
            "Workflow execution started",
            extra={
                "execution_id": execution.id,
                "workflow_id": workflow_id,
                "workflow_name": workflow.name,
                "total_steps": len(steps),
            },
        )

        # Run the step loop
        execution = await self._run_step_loop(execution, steps, context)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        log_with_latency(
            logger,
            f"Workflow execution paused/finished with status={execution.status}",
            elapsed_ms,
            extra={
                "execution_id": execution.id,
                "current_step": execution.current_step,
                "status": execution.status,
            },
        )

        return execution

    async def resume_workflow(
        self,
        execution_id: str,
        visitor_response: str | None = None,
    ) -> WorkflowExecution:
        """Resume a paused workflow execution after receiving visitor input.

        Loads the execution, injects the visitor's response into context,
        advances to the next step, and continues the step loop.

        Args:
            execution_id: ID of the WorkflowExecution to resume.
            visitor_response: The visitor's answer to the pending question.

        Returns:
            Updated WorkflowExecution record.

        Raises:
            ValueError: If the execution is not found or not in waiting state.
        """
        start_time = time.perf_counter()

        # Load execution and its workflow
        async with self._db.session() as session:
            result = await session.execute(
                select(WorkflowExecution).where(WorkflowExecution.id == execution_id)
            )
            execution = result.scalar_one_or_none()

        if execution is None:
            raise ValueError(f"Workflow execution not found: {execution_id}")
        if execution.status != "waiting":
            raise ValueError(
                f"Execution {execution_id} is not waiting (status={execution.status})"
            )

        # Load workflow steps
        async with self._db.session() as session:
            result = await session.execute(
                select(Workflow).where(Workflow.id == execution.workflow_id)
            )
            workflow = result.scalar_one_or_none()

        if workflow is None:
            raise ValueError(f"Workflow not found: {execution.workflow_id}")

        steps = json.loads(workflow.steps) if isinstance(workflow.steps, str) else workflow.steps
        context = json.loads(execution.context) if isinstance(execution.context, str) else execution.context

        # Store visitor response in context keyed by current step index
        current = execution.current_step
        if visitor_response is not None:
            context[f"step_{current}_response"] = visitor_response

        # Advance to the next step
        execution.current_step = current + 1
        execution.status = "running"
        execution.context = json.dumps(context)

        async with self._db.session() as session:
            await session.execute(
                update(WorkflowExecution)
                .where(WorkflowExecution.id == execution_id)
                .values(
                    current_step=execution.current_step,
                    status=execution.status,
                    context=execution.context,
                )
            )
            await session.commit()

        logger.info(
            "Workflow execution resumed",
            extra={
                "execution_id": execution_id,
                "resumed_at_step": execution.current_step,
                "has_response": visitor_response is not None,
            },
        )

        # Continue the step loop
        execution = await self._run_step_loop(execution, steps, context)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        log_with_latency(
            logger,
            f"Workflow resumed and paused/finished with status={execution.status}",
            elapsed_ms,
            extra={
                "execution_id": execution_id,
                "current_step": execution.current_step,
                "status": execution.status,
            },
        )

        return execution

    # ── Step loop ────────────────────────────────────────────────────────

    async def _run_step_loop(
        self,
        execution: WorkflowExecution,
        steps: list[dict[str, Any]],
        context: dict[str, Any],
    ) -> WorkflowExecution:
        """Iterate through steps starting from execution.current_step.

        Evaluates each step, updates the execution record after each one,
        and stops when the workflow completes, pauses for input, or fails.

        Args:
            execution: The current WorkflowExecution record.
            steps: List of step definition dicts from the Workflow.
            context: Mutable context dict accumulated across steps.

        Returns:
            Updated WorkflowExecution (persisted to DB).
        """
        step_index = execution.current_step

        while step_index < len(steps):
            step = steps[step_index]
            step_type = step.get("type", "")
            step_name = step.get("name", f"step_{step_index}")

            if step_type not in _VALID_STEP_TYPES:
                logger.error(
                    "Unknown step type",
                    extra={
                        "execution_id": execution.id,
                        "step_index": step_index,
                        "step_type": step_type,
                    },
                )
                execution = await self._update_execution(
                    execution,
                    status="failed",
                    error=f"Unknown step type: {step_type}",
                    current_step=step_index,
                    context=context,
                )
                return execution

            logger.info(
                "Evaluating step",
                extra={
                    "execution_id": execution.id,
                    "step_index": step_index,
                    "step_type": step_type,
                    "step_name": step_name,
                },
            )

            try:
                result = await self.evaluate_step(execution, step, context)
            except Exception as exc:
                logger.exception(
                    "Step evaluation failed",
                    extra={
                        "execution_id": execution.id,
                        "step_index": step_index,
                        "step_type": step_type,
                    },
                )
                execution = await self._update_execution(
                    execution,
                    status="failed",
                    error=f"Step {step_index} ({step_type}) failed: {exc}",
                    current_step=step_index,
                    context=context,
                )
                return execution

            # Store step result in context
            context[f"step_{step_index}_result"] = result

            # Handle step-specific flow control
            action = result.get("action")

            if action == "ask":
                # Pause execution — waiting for visitor response
                execution = await self._update_execution(
                    execution,
                    status="waiting",
                    current_step=step_index,
                    context=context,
                )
                return execution

            if action == "escalate":
                # Escalation ends the workflow
                execution = await self._update_execution(
                    execution,
                    status="failed",
                    error=result.get("reason", "Escalated"),
                    current_step=step_index,
                    context=context,
                    completed=True,
                )
                return execution

            if action == "jump":
                # Condition or branch jump to a specific step index
                next_step = result.get("next_step")
                if isinstance(next_step, int) and 0 <= next_step < len(steps):
                    step_index = next_step
                    continue
                else:
                    # Invalid jump target — fall through to next step
                    logger.warning(
                        "Invalid jump target, advancing sequentially",
                        extra={
                            "execution_id": execution.id,
                            "step_index": step_index,
                            "target": next_step,
                        },
                    )

            # Default: advance to the next step
            step_index += 1

        # All steps completed successfully
        execution = await self._update_execution(
            execution,
            status="completed",
            current_step=step_index,
            context=context,
            completed=True,
        )

        logger.info(
            "Workflow completed successfully",
            extra={
                "execution_id": execution.id,
                "workflow_id": execution.workflow_id,
            },
        )

        return execution

    # ── Step evaluation ──────────────────────────────────────────────────

    async def evaluate_step(
        self,
        execution: WorkflowExecution,
        step: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Evaluate a single workflow step and return its result.

        Dispatches to the appropriate handler based on step["type"].

        Args:
            execution: The current WorkflowExecution record.
            step: Step definition dict with "type", "name", "config".
            context: Current accumulated workflow context.

        Returns:
            Dict with step-specific result fields. Common keys:
            - "action": flow-control signal ("ask", "escalate", "jump", or None)
            - Additional keys depend on the step type.
        """
        step_type = step.get("type", "")
        raw_config = step.get("config", {})
        config = self._substitute_config(raw_config, context)

        if step_type == "ask_visitor":
            return self._eval_ask_visitor(config)

        if step_type == "call_tool":
            return await self._eval_call_tool(config, execution)

        if step_type == "ai_decision":
            return await self._eval_ai_decision(config, context, execution)

        if step_type == "condition":
            return self._eval_condition(config, context)

        if step_type == "set_variable":
            return self._eval_set_variable(config, context)

        if step_type == "send_notification":
            return self._eval_send_notification(config, execution)

        if step_type == "wait":
            return await self._eval_wait(config)

        if step_type == "escalate":
            return self._eval_escalate(config, execution)

        return {"action": None, "error": f"Unhandled step type: {step_type}"}

    # ── Individual step type handlers ────────────────────────────────────

    def _eval_ask_visitor(self, config: dict[str, Any]) -> dict[str, Any]:
        """Ask the visitor a question, pausing the workflow.

        Returns:
            Dict with action="ask" and the question text.
        """
        question = config.get("question", "")
        logger.info("Asking visitor", extra={"question": question})
        return {"action": "ask", "question": question}

    async def _eval_call_tool(
        self, config: dict[str, Any], execution: WorkflowExecution
    ) -> dict[str, Any]:
        """Make an HTTP call to an external tool/API.

        Args:
            config: Must contain "url" and "method". Optional: "headers", "body".
            execution: Current execution for logging context.

        Returns:
            Dict with tool response data or error information.
        """
        url = config.get("url", "")
        method = config.get("method", "POST").upper()
        headers = config.get("headers", {})
        body = config.get("body", {})

        logger.info(
            "Calling tool",
            extra={
                "execution_id": execution.id,
                "url": url,
                "method": method,
            },
        )

        try:
            client = await self._get_client()
            response = await client.request(
                method=method,
                url=url,
                headers=headers,
                json=body if method in ("POST", "PUT", "PATCH") else None,
                params=body if method == "GET" else None,
            )
            response.raise_for_status()

            try:
                output = response.json()
            except (json.JSONDecodeError, ValueError):
                output = {"raw": response.text}

            logger.info(
                "Tool call succeeded",
                extra={
                    "execution_id": execution.id,
                    "status_code": response.status_code,
                },
            )

            return {
                "action": None,
                "status_code": response.status_code,
                "output": output,
            }

        except httpx.HTTPStatusError as exc:
            logger.warning(
                "Tool call returned error status",
                extra={
                    "execution_id": execution.id,
                    "status_code": exc.response.status_code,
                    "url": url,
                },
            )
            return {
                "action": None,
                "status_code": exc.response.status_code,
                "error": exc.response.text,
            }

        except Exception as exc:
            logger.error(
                "Tool call failed",
                extra={
                    "execution_id": execution.id,
                    "url": url,
                    "error": str(exc),
                },
            )
            return {"action": None, "error": str(exc)}

    async def _eval_ai_decision(
        self,
        config: dict[str, Any],
        context: dict[str, Any],
        execution: WorkflowExecution,
    ) -> dict[str, Any]:
        """Call the LLM to make a decision based on context.

        Sends the prompt (with context substituted) to qwen3-max and
        maps the response to a branch if branches are configured.

        Args:
            config: Must contain "prompt". Optional: "branches" mapping.
            context: Current workflow context for prompt substitution.
            execution: Current execution for logging.

        Returns:
            Dict with "decision" (LLM output) and optional "branch"/"next_step".
        """
        prompt = config.get("prompt", "")

        # Build context summary for the LLM
        context_summary = json.dumps(context, ensure_ascii=False, default=str)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a workflow decision engine. Analyze the provided "
                    "information and respond concisely with your decision. "
                    "Do not include explanations unless asked."
                ),
            },
            {
                "role": "user",
                "content": f"{prompt}\n\nWorkflow context:\n{context_summary}",
            },
        ]

        logger.info(
            "AI decision step — calling LLM",
            extra={"execution_id": execution.id, "prompt_length": len(prompt)},
        )

        start = time.perf_counter()

        try:
            client = await self._get_client()
            headers = {"Content-Type": "application/json"}
            if self._llm_api_key:
                headers["Authorization"] = f"Bearer {self._llm_api_key}"

            response = await client.post(
                f"{self._llm_base_url}/chat/completions",
                headers=headers,
                json={
                    "model": self._llm_model,
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": 256,
                },
            )
            response.raise_for_status()
            data = response.json()

            choices = data.get("choices", [])
            if not choices:
                raise ValueError("Empty response from LLM")

            decision = (
                choices[0].get("message", {}).get("content", "").strip().lower()
            )

        except Exception as exc:
            logger.error(
                "AI decision LLM call failed",
                extra={"execution_id": execution.id, "error": str(exc)},
            )
            return {"action": None, "decision": "", "error": str(exc)}

        elapsed_ms = (time.perf_counter() - start) * 1000
        log_with_latency(
            logger,
            "AI decision complete",
            elapsed_ms,
            extra={
                "execution_id": execution.id,
                "decision": decision,
            },
        )

        # Map decision to branch if branches are configured
        branches = config.get("branches", {})
        matched_branch = None
        next_step = None

        for branch_key, branch_target in branches.items():
            if branch_key.lower() in decision:
                matched_branch = branch_key
                next_step = branch_target
                break

        result: dict[str, Any] = {
            "action": "jump" if next_step is not None else None,
            "decision": decision,
            "branch": matched_branch,
        }
        if next_step is not None:
            result["next_step"] = next_step

        # Store the decision in context for downstream steps
        context[f"step_{execution.current_step}_decision"] = decision

        return result

    def _eval_condition(
        self, config: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        """Evaluate a condition expression against the workflow context.

        The condition string is checked by simple key-value comparison.
        Supports "key == 'value'" syntax evaluated against the context dict.

        Args:
            config: Must contain "condition". Optional: "true_step", "false_step".
            context: Current workflow context for variable lookup.

        Returns:
            Dict with "result" (bool) and "action"/"next_step" for branching.
        """
        condition_str = config.get("condition", "")
        true_step = config.get("true_step")
        false_step = config.get("false_step")

        # Simple safe evaluation: support "key == 'value'" comparisons
        condition_result = False
        try:
            if "==" in condition_str:
                parts = condition_str.split("==", 1)
                left_key = parts[0].strip()
                right_value = parts[1].strip().strip("'\"")
                left_value = str(context.get(left_key, "")).strip().lower()
                condition_result = left_value == right_value.lower()
            elif "!=" in condition_str:
                parts = condition_str.split("!=", 1)
                left_key = parts[0].strip()
                right_value = parts[1].strip().strip("'\"")
                left_value = str(context.get(left_key, "")).strip().lower()
                condition_result = left_value != right_value.lower()
            else:
                # Check truthiness of a single context key
                condition_result = bool(context.get(condition_str, False))
        except Exception as exc:
            logger.warning(
                "Condition evaluation error",
                extra={"condition": condition_str, "error": str(exc)},
            )
            condition_result = False

        next_step = true_step if condition_result else false_step

        logger.info(
            "Condition evaluated",
            extra={
                "condition": condition_str,
                "result": condition_result,
                "next_step": next_step,
            },
        )

        result: dict[str, Any] = {
            "result": condition_result,
        }

        if next_step is not None:
            result["action"] = "jump"
            result["next_step"] = next_step
        else:
            result["action"] = None

        return result

    def _eval_set_variable(
        self, config: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        """Set a variable in the workflow context.

        Args:
            config: Must contain "key" and "value".
            context: Mutable workflow context to update.

        Returns:
            Dict confirming the variable was set.
        """
        key = config.get("key", "")
        value = config.get("value", "")

        context[key] = value

        logger.info(
            "Variable set",
            extra={"key": key, "value": value},
        )

        return {"action": None, "key": key, "value": value}

    def _eval_send_notification(
        self, config: dict[str, Any], execution: WorkflowExecution
    ) -> dict[str, Any]:
        """Send a notification (logged for now — future: push/email).

        Args:
            config: Must contain "message".
            execution: Current execution for logging context.

        Returns:
            Dict confirming the notification was sent.
        """
        message = config.get("message", "")

        logger.info(
            "Notification sent",
            extra={
                "execution_id": execution.id,
                "workflow_id": execution.workflow_id,
                "notification_message": message,
            },
        )

        return {"action": None, "notification": message, "delivered": True}

    async def _eval_wait(self, config: dict[str, Any]) -> dict[str, Any]:
        """Pause execution for a specified number of seconds.

        Args:
            config: Optional "seconds" (default 0).

        Returns:
            Dict confirming the wait completed.
        """
        seconds = config.get("seconds", 0)

        if seconds > 0:
            logger.info("Wait step — sleeping", extra={"seconds": seconds})
            await asyncio.sleep(seconds)

        return {"action": None, "waited_seconds": seconds}

    def _eval_escalate(
        self, config: dict[str, Any], execution: WorkflowExecution
    ) -> dict[str, Any]:
        """Escalate the workflow to a human operator.

        Marks the execution as requiring human intervention and stops
        further automated processing.

        Args:
            config: Must contain "reason". Optional: "priority".
            execution: Current execution for logging.

        Returns:
            Dict with action="escalate" to signal the step loop to stop.
        """
        reason = config.get("reason", "Escalation requested")
        priority = config.get("priority", "normal")

        logger.warning(
            "Workflow escalated",
            extra={
                "execution_id": execution.id,
                "workflow_id": execution.workflow_id,
                "reason": reason,
                "priority": priority,
            },
        )

        return {
            "action": "escalate",
            "reason": reason,
            "priority": priority,
        }

    # ── Persistence helpers ──────────────────────────────────────────────

    async def _update_execution(
        self,
        execution: WorkflowExecution,
        *,
        status: str,
        current_step: int,
        context: dict[str, Any],
        error: str = "",
        completed: bool = False,
    ) -> WorkflowExecution:
        """Persist updated execution state to the database.

        Args:
            execution: The execution record to update.
            status: New status value.
            current_step: Current step index.
            context: Accumulated context dict to serialize.
            error: Error message (empty string if no error).
            completed: Whether to set completed_at timestamp.

        Returns:
            The updated WorkflowExecution with refreshed field values.
        """
        execution.status = status
        execution.current_step = current_step
        execution.context = json.dumps(context, ensure_ascii=False, default=str)
        execution.error = error

        values: dict[str, Any] = {
            "status": status,
            "current_step": current_step,
            "context": execution.context,
            "error": error,
        }

        if completed:
            now = datetime.now(timezone.utc)
            execution.completed_at = now
            values["completed_at"] = now

        async with self._db.session() as session:
            await session.execute(
                update(WorkflowExecution)
                .where(WorkflowExecution.id == execution.id)
                .values(**values)
            )
            await session.commit()

        return execution
