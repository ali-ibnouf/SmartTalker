"""Guardrails engine — last line of defense before TTS synthesis.

Checks agent responses against:
1. Global blocked patterns (PII, harmful content)
2. Employee-specific guardrails (blocked topics, custom rules)
3. Response length limits (to control TTS/GPU costs)

Integrated in ws_visitor.py before TTS synthesis.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional

from src.utils.logger import setup_logger

logger = setup_logger("agent.guardrails")

# Global patterns that must NEVER appear in agent responses
GLOBAL_BLOCKS = [
    r"(?i)(password|كلمة.*سر|رقم.*بطاقة|credit.*card|ssn|social.*security)",
    r"(?i)(kill\s+yourself|suicide|self[\s\-]?harm|إيذاء.*نفس)",
    r"(?i)(api[\s_]?key|secret[\s_]?key|private[\s_]?key|bearer\s+[a-zA-Z0-9\-_.]+)",
]

# Default fallback messages per language
DEFAULT_FALLBACK = {
    "ar": "أعتذر، لا أستطيع المساعدة في هذا الطلب. دعني أحولك لموظف بشري.",
    "en": "I apologize, but I'm unable to help with that request. Let me connect you with a human agent.",
    "fr": "Je m'excuse, mais je ne peux pas vous aider avec cette demande. Permettez-moi de vous mettre en contact avec un agent humain.",
    "tr": "Özür dilerim, bu konuda size yardımcı olamıyorum. Sizi bir insan temsilciye bağlayayım.",
}

# Default max response length for TTS (chars)
DEFAULT_MAX_CHARS = 500


@dataclass
class GuardrailResult:
    """Result of a guardrail check.

    Attributes:
        approved: Whether the response passed all checks.
        text: The (possibly modified) response text.
        reason: Reason for blocking (if blocked).
        trimmed: Whether the response was trimmed for length.
    """
    approved: bool
    text: str
    reason: str = ""
    trimmed: bool = False


class GuardrailsEngine:
    """Pre-TTS guardrail checks on agent responses."""

    def __init__(self) -> None:
        self._compiled_patterns = [re.compile(p) for p in GLOBAL_BLOCKS]
        logger.info("GuardrailsEngine initialized", extra={"pattern_count": len(GLOBAL_BLOCKS)})

    def check_response(
        self,
        response_text: str,
        employee_guardrails: dict[str, Any] | None = None,
        language: str = "en",
    ) -> GuardrailResult:
        """Check an agent response against guardrails.

        Args:
            response_text: The LLM-generated response to check.
            employee_guardrails: Employee-specific guardrail config (JSON parsed).
            language: Language code for fallback messages.

        Returns:
            GuardrailResult with approval status and possibly modified text.
        """
        if not response_text:
            return GuardrailResult(approved=True, text=response_text)

        guardrails = employee_guardrails or {}
        fallback = guardrails.get(
            "fallback_message",
            DEFAULT_FALLBACK.get(language, DEFAULT_FALLBACK["en"]),
        )

        # 1. Global blocked patterns
        for pattern in self._compiled_patterns:
            if pattern.search(response_text):
                logger.warning(
                    "Guardrail: global block triggered",
                    extra={"pattern": pattern.pattern[:50]},
                )
                return GuardrailResult(
                    approved=False,
                    text=fallback,
                    reason="global_block",
                )

        # 2. Employee-specific blocked topics
        blocked_topics: list[str] = guardrails.get("blocked_topics", [])
        for topic in blocked_topics:
            if topic.lower() in response_text.lower():
                logger.warning(
                    "Guardrail: employee topic block",
                    extra={"topic": topic},
                )
                return GuardrailResult(
                    approved=False,
                    text=guardrails.get("topic_fallback", fallback),
                    reason=f"employee_guardrail:{topic}",
                )

        # 3. Response length limit (for TTS cost control)
        max_chars = guardrails.get("max_response_chars", DEFAULT_MAX_CHARS)
        if len(response_text) > max_chars:
            trimmed_text = response_text[:max_chars].rsplit(" ", 1)[0] + "..."
            logger.info(
                "Guardrail: response trimmed",
                extra={"original_len": len(response_text), "max_chars": max_chars},
            )
            return GuardrailResult(
                approved=True,
                text=trimmed_text,
                trimmed=True,
            )

        return GuardrailResult(approved=True, text=response_text)
