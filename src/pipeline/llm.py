"""Large Language Model engine using Qwen 2.5 via Ollama.

Manages per-session conversation history, Arabic system prompts,
and streaming/non-streaming text generation.

Audit fixes applied:
- Per-user session isolation via session_id keyed dict
- Session auto-expiry for idle sessions (30 min)
- Circuit breaker pattern for transient Ollama failures
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import httpx

from src.config import Settings
from src.utils.exceptions import LLMError
from src.utils.logger import setup_logger, log_with_latency

logger = setup_logger("pipeline.llm")

# Arabic system prompt — instructs the LLM to respond concisely in Arabic
ARABIC_SYSTEM_PROMPT = (
    "\u0623\u0646\u062a \u0645\u0633\u0627\u0639\u062f \u0630\u0643\u064a \u064a\u062a\u062d\u062f\u062b \u0627\u0644\u0639\u0631\u0628\u064a\u0629 \u0628\u0637\u0644\u0627\u0642\u0629. "
    "\u0623\u062c\u0628 \u0628\u0625\u064a\u062c\u0627\u0632 \u0648\u0648\u0636\u0648\u062d. "
    "\u0627\u0633\u062a\u062e\u062f\u0645 1-3 \u062c\u0645\u0644 \u0641\u0642\u0637. "
    "\u0643\u0646 \u0648\u062f\u0648\u062f\u0627\u064b \u0648\u0645\u0647\u0646\u064a\u0627\u064b."
)

# English fallback system prompt
ENGLISH_SYSTEM_PROMPT = (
    "You are a helpful AI assistant. "
    "Respond concisely and clearly. "
    "Use 1-3 sentences only. "
    "Be friendly and professional."
)

# Emotion-aware prompt suffixes appended to system prompt
EMOTION_PROMPTS: dict[str, str] = {
    "neutral": "",
    "happy": " \u0623\u0638\u0647\u0631 \u062d\u0645\u0627\u0633\u0627\u064b \u0648\u0625\u064a\u062c\u0627\u0628\u064a\u0629 \u0641\u064a \u0631\u062f\u0643.",
    "sad": " \u0643\u0646 \u0645\u062a\u0639\u0627\u0637\u0641\u0627\u064b \u0648\u0644\u0637\u064a\u0641\u0627\u064b \u0641\u064a \u0631\u062f\u0643.",
    "angry": " \u0643\u0646 \u0647\u0627\u062f\u0626\u0627\u064b \u0648\u0645\u0637\u0645\u0626\u0646\u0627\u064b \u0641\u064a \u0631\u062f\u0643.",
    "surprised": " \u0623\u0638\u0647\u0631 \u0627\u0647\u062a\u0645\u0627\u0645\u0627\u064b \u0648\u0641\u0636\u0648\u0644\u0627\u064b \u0641\u064a \u0631\u062f\u0643.",
    "fearful": " \u0643\u0646 \u0645\u0637\u0645\u0626\u0646\u0627\u064b \u0648\u062f\u0627\u0639\u0645\u0627\u064b \u0641\u064a \u0631\u062f\u0643.",
    "disgusted": " \u0643\u0646 \u0645\u062d\u0627\u064a\u062f\u0627\u064b \u0648\u0645\u0647\u0646\u064a\u0627\u064b \u0641\u064a \u0631\u062f\u0643.",
    "contempt": " \u0643\u0646 \u0645\u062d\u062a\u0631\u0645\u0627\u064b \u0648\u0645\u0647\u0646\u064a\u0627\u064b \u0641\u064a \u0631\u062f\u0643.",
}

# Session idle timeout (seconds)
_SESSION_TTL = 1800  # 30 minutes

# Circuit breaker: max consecutive failures before opening
_CB_THRESHOLD = 3
_CB_COOLDOWN = 30.0  # seconds


@dataclass
class LLMResult:
    """Result of an LLM text generation.

    Attributes:
        text: Generated response text.
        emotion: Detected or requested emotion label.
        latency_ms: Processing time in milliseconds.
        tokens_used: Total token count (prompt + completion).
    """

    text: str
    emotion: str = "neutral"
    latency_ms: int = 0
    tokens_used: int = 0


@dataclass
class _SessionState:
    """Internal per-session conversation state."""
    history: deque
    last_access: float


class LLMEngine:
    """Qwen 2.5 LLM engine via Ollama HTTP API.

    Manages per-session conversation history, supports
    Arabic/English prompts, and provides streaming output.
    Includes a circuit breaker for transient Ollama failures.

    Args:
        config: Application settings with LLM configuration.
    """

    def __init__(self, config: Settings) -> None:
        """Initialize the LLM engine.

        Args:
            config: Application settings instance.
        """
        self._config = config
        self._base_url = config.llm_base_url
        self._model = config.llm_model_name
        self._timeout = config.llm_timeout
        self._max_tokens = config.llm_max_tokens
        self._temperature = config.llm_temperature
        self._max_history = config.llm_max_history

        # Per-session conversation histories
        self._sessions: dict[str, _SessionState] = {}
        self._session_lock = asyncio.Lock()

        # Reusable async HTTP client
        self._client: Optional[httpx.AsyncClient] = None

        # Circuit breaker state
        self._cb_failures = 0
        self._cb_last_failure = 0.0

        logger.info(
            "LLMEngine initialized",
            extra={"model": self._model, "base_url": self._base_url},
        )

    # ── Session management ───────────────────────────────────────────────

    async def _get_session(self, session_id: str) -> _SessionState:
        """Get or create a session state.

        Also prunes expired sessions periodically.
        Uses an asyncio.Lock to prevent concurrent dict mutation.
        """
        now = time.time()

        async with self._session_lock:
            # Prune expired sessions when dict grows large
            if len(self._sessions) > 50:
                expired = [
                    sid for sid, state in self._sessions.items()
                    if now - state.last_access > _SESSION_TTL
                ]
                for sid in expired:
                    del self._sessions[sid]

            if session_id not in self._sessions:
                self._sessions[session_id] = _SessionState(
                    history=deque(maxlen=self._max_history * 2),
                    last_access=now,
                )

            session = self._sessions[session_id]
            session.last_access = now
            return session

    def clear_session(self, session_id: str) -> None:
        """Clear conversation history for a specific session.

        Args:
            session_id: Session identifier to clear.
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info("Session cleared", extra={"session_id": session_id})

    def clear_history(self) -> None:
        """Clear all session histories."""
        self._sessions.clear()
        logger.info("All conversation histories cleared")

    # ── Client ───────────────────────────────────────────────────────────

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the async HTTP client.

        Returns:
            Configured httpx.AsyncClient instance.
        """
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(self._timeout, connect=10.0),
            )
        return self._client

    # ── Circuit breaker ──────────────────────────────────────────────────

    def _check_circuit_breaker(self) -> None:
        """Check if the circuit breaker is open.

        Raises:
            LLMError: If too many consecutive failures have occurred.
        """
        if self._cb_failures >= _CB_THRESHOLD:
            elapsed = time.time() - self._cb_last_failure
            if elapsed < _CB_COOLDOWN:
                raise LLMError(
                    message=f"Circuit breaker open — {self._cb_failures} consecutive Ollama failures",
                    detail=f"Will retry in {_CB_COOLDOWN - elapsed:.0f}s",
                )
            # Cooldown expired — allow one retry (half-open)
            self._cb_failures = _CB_THRESHOLD - 1

    def _record_success(self) -> None:
        """Record a successful call — reset circuit breaker."""
        self._cb_failures = 0

    def _record_failure(self) -> None:
        """Record a failed call — increment circuit breaker."""
        self._cb_failures += 1
        self._cb_last_failure = time.time()

    # ── Message building ─────────────────────────────────────────────────

    def _build_messages(
        self,
        user_text: str,
        emotion: str = "neutral",
        language: str = "ar",
        conversation_history: Optional[list[dict[str, str]]] = None,
        session_history: Optional[deque] = None,
    ) -> list[dict[str, str]]:
        """Build the message array for the Ollama API.

        Args:
            user_text: The user's input text.
            emotion: Emotion label for prompt adjustment.
            language: Target language ("ar" or "en").
            conversation_history: Optional external history override.
            session_history: Pre-fetched session history deque.

        Returns:
            List of message dicts with role and content.
        """
        # Select system prompt by language
        system_prompt = ARABIC_SYSTEM_PROMPT if language == "ar" else ENGLISH_SYSTEM_PROMPT

        # Append emotion modifier
        emotion_suffix = EMOTION_PROMPTS.get(emotion, "")
        if emotion_suffix:
            system_prompt += emotion_suffix

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
        ]

        # Add conversation history (session-specific)
        if conversation_history is not None:
            messages.extend(conversation_history)
        elif session_history is not None:
            messages.extend(list(session_history))

        # Add current user message
        messages.append({"role": "user", "content": user_text})

        return messages

    # ── Non-streaming generation ─────────────────────────────────────────

    async def generate(
        self,
        user_text: str,
        emotion: str = "neutral",
        conversation_history: Optional[list[dict[str, str]]] = None,
        language: str = "ar",
        session_id: str = "default",
    ) -> LLMResult:
        """Generate a response from the LLM (non-streaming).

        Args:
            user_text: The user's input text.
            emotion: Emotion context for response adjustment.
            conversation_history: Optional external history to use.
            language: Target response language ("ar" or "en").
            session_id: Session identifier for conversation isolation.

        Returns:
            LLMResult with generated text, emotion, latency, and token count.

        Raises:
            LLMError: If the API call fails, times out, or returns invalid data.
        """
        if not user_text.strip():
            raise LLMError(message="User text cannot be empty")

        self._check_circuit_breaker()

        # Fetch session history before building messages (async lock)
        session = await self._get_session(session_id)
        messages = self._build_messages(
            user_text, emotion, language,
            conversation_history=conversation_history,
            session_history=session.history,
        )
        start = time.perf_counter()

        try:
            client = await self._get_client()
            response = await client.post(
                "/api/chat",
                json={
                    "model": self._model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": self._temperature,
                        "num_predict": self._max_tokens,
                    },
                },
            )
            response.raise_for_status()
            data = response.json()

        except httpx.ConnectError as exc:
            self._record_failure()
            raise LLMError(
                message="Cannot connect to Ollama",
                detail=f"Is Ollama running at {self._base_url}?",
                original_exception=exc,
            ) from exc
        except httpx.TimeoutException as exc:
            self._record_failure()
            raise LLMError(
                message=f"LLM request timed out after {self._timeout}s",
                original_exception=exc,
            ) from exc
        except httpx.HTTPStatusError as exc:
            self._record_failure()
            raise LLMError(
                message=f"Ollama API error: {exc.response.status_code}",
                detail=exc.response.text,
                original_exception=exc,
            ) from exc
        except Exception as exc:
            self._record_failure()
            raise LLMError(
                message="LLM generation failed",
                detail=str(exc),
                original_exception=exc,
            ) from exc

        self._record_success()
        elapsed_ms = int((time.perf_counter() - start) * 1000)

        # Parse response
        response_text = self._extract_text(data)
        tokens_used = self._extract_tokens(data)

        # Update session-specific history (session already fetched above)
        session.history.append({"role": "user", "content": user_text})
        session.history.append({"role": "assistant", "content": response_text})

        result = LLMResult(
            text=response_text,
            emotion=emotion,
            latency_ms=elapsed_ms,
            tokens_used=tokens_used,
        )

        log_with_latency(
            logger,
            "LLM generation complete",
            elapsed_ms,
            extra={
                "input_length": len(user_text),
                "output_length": len(response_text),
                "tokens": tokens_used,
                "session_id": session_id,
            },
        )
        return result

    # ── Response parsing ─────────────────────────────────────────────────

    @staticmethod
    def _extract_text(data: dict) -> str:
        """Extract the assistant's response text from Ollama API data.

        Args:
            data: Parsed JSON response from Ollama.

        Returns:
            The assistant's response text.

        Raises:
            LLMError: If the response format is unexpected.
        """
        try:
            message = data.get("message", {})
            text = message.get("content", "").strip()
            if not text:
                raise LLMError(
                    message="Empty response from LLM",
                    detail=f"Raw response: {data}",
                )
            return text
        except AttributeError as exc:
            raise LLMError(
                message="Unexpected LLM response format",
                detail=str(data),
                original_exception=exc,
            ) from exc

    @staticmethod
    def _extract_tokens(data: dict) -> int:
        """Extract total token usage from Ollama API data.

        Args:
            data: Parsed JSON response from Ollama.

        Returns:
            Total tokens used (prompt + completion).
        """
        prompt_tokens = data.get("prompt_eval_count", 0) or 0
        completion_tokens = data.get("eval_count", 0) or 0
        return prompt_tokens + completion_tokens

    async def close(self) -> None:
        """Close the HTTP client and clean up resources."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
        self._sessions.clear()
        logger.info("LLM client closed")
