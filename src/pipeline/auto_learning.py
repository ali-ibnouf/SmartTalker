"""Automatic learning engine for extracting knowledge from conversations.

After a visitor session ends, this engine analyses the conversation transcript
using Qwen3-max to extract:
1. Q&A pairs — reusable knowledge for the employee's knowledge base
2. Visitor memories — personal facts/preferences for returning-visitor context

Q&A confidence thresholds:
- >= 0.85  -> auto-create EmployeeKnowledge (approved=True)
- 0.5-0.84 -> create EmployeeLearning (status="pending", learning_type="qa_pair")
- < 0.5   -> discard
"""

from __future__ import annotations

import json
import time
from typing import Any, Optional

import httpx
from sqlalchemy import select

from src.config import Settings, get_settings
from src.db.engine import Database
from src.db.models import (
    ConversationMessage,
    EmployeeKnowledge,
    EmployeeLearning,
    VisitorMemory,
    _uuid,
)
from src.utils.logger import log_with_latency, setup_logger

logger = setup_logger("pipeline.auto_learning")

# Confidence thresholds for auto-learning decisions
_HIGH_CONFIDENCE = 0.85
_MEDIUM_CONFIDENCE = 0.50

# System prompt for Q&A extraction
_EXTRACTION_SYSTEM_PROMPT = (
    "You are an expert at analysing customer-service conversation transcripts. "
    "Your task is to extract useful question-answer pairs that can be added to "
    "a knowledge base for training a digital employee.\n\n"
    "Rules:\n"
    "1. Only extract pairs where the assistant (or operator) provided a clear, "
    "   factual, and helpful answer to a visitor question.\n"
    "2. Ignore greetings, small talk, and purely procedural exchanges.\n"
    "3. Generalise the question so it can match future visitors.\n"
    "4. Assign a confidence score (0.0-1.0) reflecting how useful and accurate "
    "   the pair is for knowledge-base training.\n"
    "5. Assign a short category label (e.g. 'pricing', 'product', 'support', "
    "   'shipping', 'returns', 'general').\n\n"
    "Return a JSON object with a single key \"pairs\" whose value is an array "
    "of objects. Each object must have exactly these keys:\n"
    '  - "question": string\n'
    '  - "answer": string\n'
    '  - "confidence": number (0.0 to 1.0)\n'
    '  - "category": string\n\n'
    "If no useful pairs can be extracted, return {\"pairs\": []}.\n"
    "Do NOT include any text outside the JSON object."
)

# System prompt for visitor memory extraction
_MEMORY_SYSTEM_PROMPT = (
    "You are an expert at analysing customer-service conversation transcripts. "
    "Your task is to extract notable personal information about the visitor that "
    "would be useful to remember for future conversations.\n\n"
    "Rules:\n"
    "1. Only extract factual, specific information the visitor shared.\n"
    "2. Ignore generic or trivial details.\n"
    "3. Assign a type: preference, fact, issue, purchase, complaint, or note.\n"
    "4. Assign an importance score (0.0-1.0) reflecting how useful it would be "
    "   to recall this in a future conversation.\n\n"
    "Return a JSON object with a single key \"memories\" whose value is an array "
    "of objects. Each object must have exactly these keys:\n"
    '  - "type": string (one of: preference, fact, issue, purchase, complaint, note)\n'
    '  - "content": string (concise description of the memory)\n'
    '  - "importance": number (0.0 to 1.0)\n\n'
    "If nothing notable was shared, return {\"memories\": []}.\n"
    "Do NOT include any text outside the JSON object."
)


class AutoLearningEngine:
    """Extracts Q&A knowledge from completed conversation sessions.

    Called as a background task after a visitor session ends.  Uses
    Qwen3-max to analyse the transcript and produce structured Q&A pairs,
    then persists them based on confidence thresholds.

    Args:
        db: Async database connection manager.
        config: Application settings.  Falls back to ``get_settings()``.
    """

    def __init__(self, db: Database, config: Optional[Settings] = None) -> None:
        self._db = db
        self._config = config or get_settings()

        api_key = self._config.llm_api_key or self._config.dashscope_api_key
        self._model = self._config.llm_model_name
        self._client = httpx.AsyncClient(
            base_url=self._config.llm_base_url,
            timeout=httpx.Timeout(self._config.llm_timeout, connect=10.0),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )

        logger.info(
            "AutoLearningEngine initialized",
            extra={"model": self._model, "base_url": self._config.llm_base_url},
        )

    # ── Public API ────────────────────────────────────────────────────────

    async def process_session(
        self,
        session_id: str,
        employee_id: str,
        customer_id: str,
        visitor_id: str = "",
    ) -> None:
        """Analyse a completed session and extract knowledge + visitor memories.

        Loads conversation messages, builds a transcript, calls the LLM to
        extract Q&A pairs and visitor memories, then persists results.

        Args:
            session_id: Conversation ID to process.
            employee_id: Employee who handled the session.
            customer_id: Customer that owns the employee.
            visitor_id: Visitor who participated (for memory storage).
        """
        start = time.perf_counter()
        logger.info(
            "Processing session for auto-learning",
            extra={"session_id": session_id, "employee_id": employee_id},
        )

        # 1. Load conversation messages
        transcript = await self._load_transcript(session_id)
        if not transcript:
            logger.warning(
                "No messages found for session — skipping",
                extra={"session_id": session_id},
            )
            return

        # 2. Extract Q&A pairs via LLM
        pairs: list[dict[str, Any]] = []
        try:
            pairs = await self._extract_qa_pairs(transcript)
        except Exception as exc:
            logger.error(
                "Q&A extraction failed",
                extra={"session_id": session_id, "error": str(exc)},
            )

        # 3. Extract visitor memories via LLM (independent of Q&A success)
        memories: list[dict[str, Any]] = []
        if visitor_id:
            try:
                memories = await self._extract_visitor_memories(transcript)
            except Exception as exc:
                logger.error(
                    "Visitor memory extraction failed",
                    extra={"session_id": session_id, "error": str(exc)},
                )

        if not pairs and not memories:
            logger.info(
                "No Q&A pairs or memories extracted from session",
                extra={"session_id": session_id},
            )
            return

        # 4. Persist pairs according to confidence thresholds
        auto_approved = 0
        pending_review = 0
        discarded = 0
        memories_saved = 0

        async with self._db.session() as session:
            for pair in pairs:
                confidence = float(pair.get("confidence", 0.0))
                question = str(pair.get("question", "")).strip()
                answer = str(pair.get("answer", "")).strip()
                category = str(pair.get("category", "general")).strip()

                if not question or not answer:
                    discarded += 1
                    continue

                if confidence >= _HIGH_CONFIDENCE:
                    # Auto-approve into knowledge base
                    knowledge = EmployeeKnowledge(
                        id=_uuid(),
                        employee_id=employee_id,
                        category=category,
                        question=question,
                        answer=answer,
                        approved=True,
                        times_used=0,
                        success_rate=0.0,
                    )
                    session.add(knowledge)
                    auto_approved += 1

                elif confidence >= _MEDIUM_CONFIDENCE:
                    # Queue for human review
                    new_value = json.dumps(
                        {"question": question, "answer": answer, "category": category},
                        ensure_ascii=False,
                    )
                    learning = EmployeeLearning(
                        id=_uuid(),
                        employee_id=employee_id,
                        customer_id=customer_id,
                        session_id=session_id,
                        learning_type="qa_pair",
                        old_value="",
                        new_value=new_value,
                        confidence=confidence,
                        status="pending",
                        source="auto",
                    )
                    session.add(learning)
                    pending_review += 1

                else:
                    discarded += 1

            # 5. Persist visitor memories
            for mem in memories:
                mem_type = str(mem.get("type", "note")).strip()
                content = str(mem.get("content", "")).strip()
                importance = float(mem.get("importance", 0.5))

                if not content:
                    continue
                if mem_type not in ("preference", "fact", "issue", "purchase", "complaint", "note"):
                    mem_type = "note"

                visitor_mem = VisitorMemory(
                    id=_uuid(),
                    visitor_id=visitor_id,
                    employee_id=employee_id,
                    memory_type=mem_type,
                    content=content,
                    source_session=session_id,
                    importance=importance,
                )
                session.add(visitor_mem)
                memories_saved += 1

            await session.commit()

        elapsed_ms = (time.perf_counter() - start) * 1000
        log_with_latency(
            logger,
            "Auto-learning processing complete",
            elapsed_ms,
            extra={
                "session_id": session_id,
                "total_pairs": len(pairs),
                "auto_approved": auto_approved,
                "pending_review": pending_review,
                "discarded": discarded,
                "memories_saved": memories_saved,
            },
        )

    # ── Internal helpers ──────────────────────────────────────────────────

    async def _load_transcript(self, session_id: str) -> str:
        """Load conversation messages and build a plain-text transcript.

        Args:
            session_id: The conversation ID to look up.

        Returns:
            Formatted transcript string, or empty string if no messages.
        """
        async with self._db.session() as session:
            stmt = (
                select(ConversationMessage)
                .where(ConversationMessage.conversation_id == session_id)
                .order_by(ConversationMessage.created_at.asc())
            )
            result = await session.execute(stmt)
            messages = result.scalars().all()

        if not messages:
            return ""

        lines: list[str] = []
        for msg in messages:
            lines.append(f"{msg.role}: {msg.content}")

        return "\n".join(lines)

    async def _extract_qa_pairs(self, transcript: str) -> list[dict[str, Any]]:
        """Call Qwen3-max to extract Q&A pairs from a transcript.

        Args:
            transcript: Plain-text conversation transcript.

        Returns:
            List of dicts with keys: question, answer, confidence, category.

        Raises:
            httpx.HTTPStatusError: If the LLM API returns a non-2xx status.
            ValueError: If the LLM response cannot be parsed as valid JSON.
        """
        messages = [
            {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Extract Q&A pairs from the following conversation transcript:\n\n"
                    f"{transcript}"
                ),
            },
        ]

        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": 0.3,
            "response_format": {"type": "json_object"},
        }

        response = await self._client.post("/chat/completions", json=payload)
        response.raise_for_status()

        data = response.json()
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )

        if not content:
            logger.warning("LLM returned empty content for Q&A extraction")
            return []

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            logger.error(
                "Failed to parse LLM Q&A extraction response as JSON",
                extra={"raw_content": content[:500], "error": str(exc)},
            )
            raise ValueError(
                f"LLM returned invalid JSON for Q&A extraction: {exc}"
            ) from exc

        pairs = parsed.get("pairs", [])
        if not isinstance(pairs, list):
            logger.warning(
                "LLM response 'pairs' field is not a list",
                extra={"type": type(pairs).__name__},
            )
            return []

        return pairs

    async def _extract_visitor_memories(self, transcript: str) -> list[dict[str, Any]]:
        """Call Qwen3-max to extract visitor memories from a transcript.

        Args:
            transcript: Plain-text conversation transcript.

        Returns:
            List of dicts with keys: type, content, importance.
        """
        messages = [
            {"role": "system", "content": _MEMORY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Extract notable visitor information from this conversation:\n\n"
                    f"{transcript}"
                ),
            },
        ]

        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 500,
            "response_format": {"type": "json_object"},
        }

        response = await self._client.post("/chat/completions", json=payload)
        response.raise_for_status()

        data = response.json()
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )

        if not content:
            return []

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            logger.error(
                "Failed to parse LLM memory extraction response",
                extra={"raw_content": content[:500], "error": str(exc)},
            )
            return []

        memories = parsed.get("memories", [])
        if not isinstance(memories, list):
            return []

        return memories

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None  # type: ignore[assignment]
        logger.info("AutoLearningEngine closed")
