"""Rule engine framework for the AI Optimization Agent.

Defines Detection (a single finding), MonitorRule (Protocol for rules),
AgentContext (shared state), and RuleRegistry (collection + runner).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

from src.utils.logger import setup_logger

logger = setup_logger("ai_agent.rules")


@dataclass
class Detection:
    """A single finding from a monitor rule."""

    rule_id: str
    severity: str  # "info" | "warning" | "critical"
    title: str
    description: str
    details: dict[str, Any] = field(default_factory=dict)
    recommendation: str = ""
    auto_fixable: bool = False


@dataclass
class AgentContext:
    """Shared context passed to every rule during evaluation."""

    db: Any  # Database | None
    redis: Any  # Redis client | None
    pipeline: Any  # SmartTalkerPipeline
    config: Any  # Settings
    agent_config: Any  # AgentSettings
    operator_manager: Any = None  # OperatorWebSocketManager | None
    node_manager: Any = None  # Deprecated — GPU nodes replaced by RunPod Serverless


@runtime_checkable
class MonitorRule(Protocol):
    """Protocol that all detection rules must implement."""

    rule_id: str

    async def evaluate(self, ctx: AgentContext) -> list[Detection]:
        """Run this rule and return zero or more detections."""
        ...


class RuleRegistry:
    """Collects MonitorRule instances and runs them all."""

    def __init__(self) -> None:
        self._rules: list[MonitorRule] = []

    def register(self, rule: MonitorRule) -> None:
        self._rules.append(rule)

    @property
    def rules(self) -> list[MonitorRule]:
        return list(self._rules)

    async def run_all(self, ctx: AgentContext) -> list[Detection]:
        """Evaluate every registered rule, collecting all detections.

        Rules run concurrently. If a single rule raises, it is logged
        and skipped — other rules still produce results.
        """
        detections: list[Detection] = []

        async def _run_one(rule: MonitorRule) -> list[Detection]:
            try:
                return await rule.evaluate(ctx)
            except Exception as exc:
                logger.error(
                    f"Rule {rule.rule_id} failed: {exc}",
                    extra={"rule_id": rule.rule_id},
                )
                return []

        results = await asyncio.gather(*[_run_one(r) for r in self._rules])
        for batch in results:
            detections.extend(batch)
        return detections
