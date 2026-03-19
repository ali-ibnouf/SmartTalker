"""Large Language Model engine using Qwen3 via DashScope.

Uses the OpenAI-compatible endpoint provided by DashScope API.
Model: qwen3-max ($1.20/1M input, $6.00/1M output tokens).
Manages per-session conversation history, multilingual system prompts,
and non-streaming text generation.

Audit fixes applied:
- Per-user session isolation via session_id keyed dict
- Session auto-expiry for idle sessions (30 min)
- Circuit breaker pattern for transient API failures
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

# French system prompt
FRENCH_SYSTEM_PROMPT = (
    "Vous \u00eates un assistant IA serviable. "
    "R\u00e9pondez de mani\u00e8re concise et claire. "
    "Utilisez 1 \u00e0 3 phrases uniquement. "
    "Soyez amical et professionnel."
)

# Turkish system prompt
TURKISH_SYSTEM_PROMPT = (
    "Sen yard\u0131mc\u0131 bir yapay zeka asistan\u0131s\u0131n. "
    "K\u0131sa ve net yan\u0131tla. "
    "Sadece 1-3 c\u00fcmle kullan. "
    "Samimi ve profesyonel ol."
)

# ── Per-language system prompt lookup ─────────────────────────────────────
_SYSTEM_PROMPTS: dict[str, str] = {
    "ar": ARABIC_SYSTEM_PROMPT,
    "en": ENGLISH_SYSTEM_PROMPT,
    "fr": FRENCH_SYSTEM_PROMPT,
    "tr": TURKISH_SYSTEM_PROMPT,
}

# ── Emotion-aware prompt suffixes (Arabic — original) ────────────────────
EMOTION_PROMPTS_AR: dict[str, str] = {
    "neutral": "",
    "happy": " \u0623\u0638\u0647\u0631 \u062d\u0645\u0627\u0633\u0627\u064b \u0648\u0625\u064a\u062c\u0627\u0628\u064a\u0629 \u0641\u064a \u0631\u062f\u0643.",
    "sad": " \u0643\u0646 \u0645\u062a\u0639\u0627\u0637\u0641\u0627\u064b \u0648\u0644\u0637\u064a\u0641\u0627\u064b \u0641\u064a \u0631\u062f\u0643.",
    "angry": " \u0643\u0646 \u0647\u0627\u062f\u0626\u0627\u064b \u0648\u0645\u0637\u0645\u0626\u0646\u0627\u064b \u0641\u064a \u0631\u062f\u0643.",
    "surprised": " \u0623\u0638\u0647\u0631 \u0627\u0647\u062a\u0645\u0627\u0645\u0627\u064b \u0648\u0641\u0636\u0648\u0644\u0627\u064b \u0641\u064a \u0631\u062f\u0643.",
    "fearful": " \u0643\u0646 \u0645\u0637\u0645\u0626\u0646\u0627\u064b \u0648\u062f\u0627\u0639\u0645\u0627\u064b \u0641\u064a \u0631\u062f\u0643.",
    "disgusted": " \u0643\u0646 \u0645\u062d\u0627\u064a\u062f\u0627\u064b \u0648\u0645\u0647\u0646\u064a\u0627\u064b \u0641\u064a \u0631\u062f\u0643.",
    "contempt": " \u0643\u0646 \u0645\u062d\u062a\u0631\u0645\u0627\u064b \u0648\u0645\u0647\u0646\u064a\u0627\u064b \u0641\u064a \u0631\u062f\u0643.",
}

# Keep backwards-compatible alias
EMOTION_PROMPTS = EMOTION_PROMPTS_AR

EMOTION_PROMPTS_EN: dict[str, str] = {
    "neutral": "",
    "happy": " Show enthusiasm and positivity in your response.",
    "sad": " Be empathetic and gentle in your response.",
    "angry": " Be calm and reassuring in your response.",
    "surprised": " Show interest and curiosity in your response.",
    "fearful": " Be reassuring and supportive in your response.",
    "disgusted": " Be neutral and professional in your response.",
    "contempt": " Be respectful and professional in your response.",
}

EMOTION_PROMPTS_FR: dict[str, str] = {
    "neutral": "",
    "happy": " Montrez de l'enthousiasme et de la positivit\u00e9 dans votre r\u00e9ponse.",
    "sad": " Soyez empathique et doux dans votre r\u00e9ponse.",
    "angry": " Soyez calme et rassurant dans votre r\u00e9ponse.",
    "surprised": " Montrez de l'int\u00e9r\u00eat et de la curiosit\u00e9 dans votre r\u00e9ponse.",
    "fearful": " Soyez rassurant et encourageant dans votre r\u00e9ponse.",
    "disgusted": " Soyez neutre et professionnel dans votre r\u00e9ponse.",
    "contempt": " Soyez respectueux et professionnel dans votre r\u00e9ponse.",
}

EMOTION_PROMPTS_TR: dict[str, str] = {
    "neutral": "",
    "happy": " Yan\u0131t\u0131n\u0131zda co\u015fku ve pozitiflik g\u00f6sterin.",
    "sad": " Yan\u0131t\u0131n\u0131zda empatik ve nazik olun.",
    "angry": " Yan\u0131t\u0131n\u0131zda sakin ve g\u00fcven verici olun.",
    "surprised": " Yan\u0131t\u0131n\u0131zda ilgi ve merak g\u00f6sterin.",
    "fearful": " Yan\u0131t\u0131n\u0131zda g\u00fcven verici ve destekleyici olun.",
    "disgusted": " Yan\u0131t\u0131n\u0131zda tarafs\u0131z ve profesyonel olun.",
    "contempt": " Yan\u0131t\u0131n\u0131zda sayg\u0131l\u0131 ve profesyonel olun.",
}

# Per-language emotion prompt lookup
_EMOTION_PROMPT_MAP: dict[str, dict[str, str]] = {
    "ar": EMOTION_PROMPTS_AR,
    "en": EMOTION_PROMPTS_EN,
    "fr": EMOTION_PROMPTS_FR,
    "tr": EMOTION_PROMPTS_TR,
}

# Per-language Knowledge Base context injection templates
_KB_CONTEXT_TEMPLATES: dict[str, str] = {
    "ar": (
        "\n\n\u0627\u0633\u062a\u062e\u062f\u0645 \u0627\u0644\u0633\u064a\u0627\u0642 \u0627\u0644\u062a\u0627\u0644\u064a \u0645\u0646 \u0642\u0627\u0639\u062f\u0629 \u0627\u0644\u0645\u0639\u0631\u0641\u0629 \u0644\u0644\u0625\u062c\u0627\u0628\u0629 \u0639\u0644\u0649 \u0633\u0624\u0627\u0644 \u0627\u0644\u0645\u0633\u062a\u062e\u062f\u0645. "
        "\u0625\u0630\u0627 \u0643\u0627\u0646 \u0627\u0644\u0633\u064a\u0627\u0642 \u0630\u0627 \u0635\u0644\u0629\u060c \u0627\u0633\u062a\u0646\u062f \u0625\u0644\u064a\u0647 \u0641\u064a \u0625\u062c\u0627\u0628\u062a\u0643.\n\n"
        "\u0627\u0644\u0633\u064a\u0627\u0642:\n{context}"
    ),
    "en": (
        "\n\nUse the following knowledge base context to answer the user's question. "
        "If the context is relevant, base your answer on it.\n\n"
        "Context:\n{context}"
    ),
    "fr": (
        "\n\nUtilisez le contexte suivant de la base de connaissances pour r\u00e9pondre \u00e0 la question. "
        "Si le contexte est pertinent, basez votre r\u00e9ponse dessus.\n\n"
        "Contexte :\n{context}"
    ),
    "tr": (
        "\n\nKullan\u0131c\u0131n\u0131n sorusunu yan\u0131tlamak i\u00e7in a\u015fa\u011f\u0131daki bilgi taban\u0131 ba\u011flam\u0131n\u0131 kullan\u0131n. "
        "Ba\u011flam ilgiliyse, yan\u0131t\u0131n\u0131z\u0131 buna dayand\u0131r\u0131n.\n\n"
        "Ba\u011flam:\n{context}"
    ),
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
    cost_usd: float = 0.0


@dataclass
class _SessionState:
    """Internal per-session conversation state."""
    history: deque
    last_access: float


class LLMEngine:
    """Qwen LLM engine via DashScope (OpenAI-compatible).

    Manages per-session conversation history, supports
    Arabic/English prompts, and provides text generation.
    Includes a circuit breaker for transient API failures.

    Args:
        config: Application settings with LLM configuration.
    """

    # Qwen3-max pricing per 1M tokens
    INPUT_COST_PER_M = 1.20
    OUTPUT_COST_PER_M = 6.00

    def __init__(self, config: Settings) -> None:
        self._config = config
        self._base_url = config.llm_base_url
        self._model = config.llm_model_name
        self._api_key = config.llm_api_key or config.dashscope_api_key
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

        # Rolling latency buffer for monitoring (last 50 calls)
        from collections import deque
        self._recent_latencies: deque[float] = deque(maxlen=50)

        # Consecutive timeout/error tracking (for agent escalation)
        self._consecutive_timeouts: int = 0
        self._rate_limit_429_count: int = 0
        self._pending_requests: int = 0
        self.text_only_mode: bool = False  # Set by agent auto-fix after 5+ consecutive timeouts

        logger.info(
            "LLMEngine initialized (DashScope API)",
            extra={"model": self._model, "base_url": self._base_url},
        )

    # ── Session management ───────────────────────────────────────────────

    async def _get_session(self, session_id: str) -> _SessionState:
        """Get or create a session state."""
        now = time.time()

        async with self._session_lock:
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

    async def clear_session(self, session_id: str) -> None:
        """Clear conversation history for a specific session."""
        async with self._session_lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.info("Session cleared", extra={"session_id": session_id})

    async def clear_history(self) -> None:
        """Clear all session histories."""
        async with self._session_lock:
            self._sessions.clear()
            logger.info("All conversation histories cleared")

    @property
    def session_count(self) -> int:
        """Return number of active sessions."""
        return len(self._sessions)

    # ── Client ───────────────────────────────────────────────────────────

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the async HTTP client for DashScope API."""
        if self._client is None or self._client.is_closed:
            headers = {"Content-Type": "application/json"}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(self._timeout, connect=10.0),
                headers=headers,
            )
        return self._client

    # ── Circuit breaker ──────────────────────────────────────────────────

    def _check_circuit_breaker(self) -> None:
        """Check if the circuit breaker is open."""
        if self._cb_failures >= _CB_THRESHOLD:
            elapsed = time.time() - self._cb_last_failure
            if elapsed < _CB_COOLDOWN:
                raise LLMError(
                    message=f"Circuit breaker open — {self._cb_failures} consecutive API failures",
                    detail=f"Will retry in {_CB_COOLDOWN - elapsed:.0f}s",
                )
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
        kb_context: Optional[str] = None,
    ) -> list[dict[str, str]]:
        """Build the message array for the OpenAI-compatible API."""
        system_prompt = _SYSTEM_PROMPTS.get(language, ENGLISH_SYSTEM_PROMPT)

        emotion_dict = _EMOTION_PROMPT_MAP.get(language, EMOTION_PROMPTS_EN)
        emotion_suffix = emotion_dict.get(emotion, "")
        if emotion_suffix:
            system_prompt += emotion_suffix

        # Inject Knowledge Base context if available
        if kb_context:
            template = _KB_CONTEXT_TEMPLATES.get(language, _KB_CONTEXT_TEMPLATES["en"])
            system_prompt += template.format(context=kb_context)

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
        ]

        if conversation_history is not None:
            messages.extend(conversation_history)
        elif session_history is not None:
            messages.extend(list(session_history))

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
        kb_context: Optional[str] = None,
    ) -> LLMResult:
        """Generate a response from the LLM via DashScope API (non-streaming).

        Args:
            user_text: The user's input text.
            emotion: Emotion context for response adjustment.
            conversation_history: Optional external history to use.
            language: Target response language ("ar" or "en").
            session_id: Session identifier for conversation isolation.
            kb_context: Optional Knowledge Base context for RAG.

        Returns:
            LLMResult with generated text, emotion, latency, and token count.

        Raises:
            LLMError: If the API call fails, times out, or returns invalid data.
        """
        if not user_text.strip():
            raise LLMError(message="User text cannot be empty")

        self._check_circuit_breaker()
        self._pending_requests += 1

        try:
            session = await self._get_session(session_id)
            messages = self._build_messages(
                user_text, emotion, language,
                conversation_history=conversation_history,
                session_history=session.history,
                kb_context=kb_context,
            )
            start = time.perf_counter()

            # Retry with exponential backoff for transient errors (1s, 2s, 4s)
            _backoff = [1.0, 2.0, 4.0]
            _max_attempts = len(_backoff) + 1  # 4 total
            data: Optional[dict] = None

            for attempt in range(_max_attempts):
                try:
                    client = await self._get_client()
                    response = await client.post(
                        "/chat/completions",
                        json={
                            "model": self._model,
                            "messages": messages,
                            "temperature": self._temperature,
                            "max_tokens": self._max_tokens,
                        },
                    )

                    # Handle 429 rate limit with Retry-After or backoff
                    if response.status_code == 429:
                        self._rate_limit_429_count += 1
                        if attempt < _max_attempts - 1:
                            try:
                                retry_after = float(
                                    response.headers.get("Retry-After", _backoff[attempt])
                                )
                            except (ValueError, TypeError):
                                retry_after = _backoff[attempt]
                            logger.warning(
                                "DashScope rate limited (429), retrying",
                                extra={"attempt": attempt + 1, "retry_after": retry_after},
                            )
                            await asyncio.sleep(retry_after)
                            continue
                        # Exhausted retries on 429
                        response.raise_for_status()

                    response.raise_for_status()
                    data = response.json()
                    break  # Success

                except httpx.TimeoutException as exc:
                    if attempt < _max_attempts - 1:
                        delay = _backoff[attempt]
                        logger.warning(
                            "DashScope timeout, retrying with backoff",
                            extra={"attempt": attempt + 1, "delay": delay, "session_id": session_id},
                        )
                        await asyncio.sleep(delay)
                        continue
                    self._record_failure()
                    self._consecutive_timeouts += 1
                    raise LLMError(
                        message=f"LLM request timed out after {_max_attempts} attempts",
                        original_exception=exc,
                    ) from exc

                except httpx.ConnectError as exc:
                    if attempt < _max_attempts - 1:
                        delay = _backoff[attempt]
                        logger.warning(
                            "DashScope connection error, retrying with backoff",
                            extra={"attempt": attempt + 1, "delay": delay, "session_id": session_id},
                        )
                        await asyncio.sleep(delay)
                        continue
                    self._record_failure()
                    self._consecutive_timeouts += 1
                    raise LLMError(
                        message="Cannot connect to LLM API after retries",
                        detail=f"Check that LLM API is reachable at: {self._base_url}",
                        original_exception=exc,
                    ) from exc

                except httpx.HTTPStatusError as exc:
                    # Non-429 HTTP errors — don't retry
                    self._record_failure()
                    raise LLMError(
                        message=f"LLM API error: {exc.response.status_code}",
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

            if data is None:
                self._record_failure()
                self._consecutive_timeouts += 1
                raise LLMError(message="LLM request failed after all retry attempts")

            # Success — reset counters
            self._record_success()
            self._consecutive_timeouts = 0
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            self._recent_latencies.append(elapsed_ms / 1000.0)

            # Parse OpenAI-compatible response
            response_text = self._extract_text(data)
            tokens_used = self._extract_tokens(data)

            # Calculate cost (qwen3-max pricing)
            usage = data.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            cost_usd = (input_tokens / 1_000_000 * self.INPUT_COST_PER_M) + (
                output_tokens / 1_000_000 * self.OUTPUT_COST_PER_M
            )

            # Update session history (under lock to prevent interleaving)
            async with self._session_lock:
                session.history.append({"role": "user", "content": user_text})
                session.history.append({"role": "assistant", "content": response_text})

            result = LLMResult(
                text=response_text,
                emotion=emotion,
                latency_ms=elapsed_ms,
                tokens_used=tokens_used,
                cost_usd=cost_usd,
            )

            log_with_latency(
                logger,
                "LLM generation complete",
                elapsed_ms,
                extra={
                    "input_length": len(user_text),
                    "output_length": len(response_text),
                    "tokens": tokens_used,
                    "cost_usd": f"{cost_usd:.6f}",
                    "session_id": session_id,
                },
            )
            return result

        finally:
            self._pending_requests -= 1

    # ── Response parsing (OpenAI-compatible format) ───────────────────────

    @staticmethod
    def _extract_text(data: dict) -> str:
        """Extract the assistant's response text from OpenAI-compatible response."""
        try:
            choices = data.get("choices", [])
            if not choices:
                raise LLMError(
                    message="Empty response from LLM",
                    detail=f"Raw response: {data}",
                )
            text = choices[0].get("message", {}).get("content", "").strip()
            if not text:
                raise LLMError(
                    message="Empty response from LLM",
                    detail=f"Raw response: {data}",
                )
            return text
        except (AttributeError, IndexError, KeyError) as exc:
            raise LLMError(
                message="Unexpected LLM response format",
                detail=str(data),
                original_exception=exc,
            ) from exc

    @staticmethod
    def _extract_tokens(data: dict) -> int:
        """Extract total token usage from OpenAI-compatible response."""
        usage = data.get("usage", {})
        return usage.get("total_tokens", 0)

    async def close(self) -> None:
        """Close the HTTP client and clean up resources."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
        self._sessions.clear()
        logger.info("LLM client closed")
