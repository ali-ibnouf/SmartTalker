"""Pipeline orchestrator — coordinates ASR -> LLM -> TTS.

Manages model lifecycle and end-to-end processing
for text-to-speech and audio-chat pipelines.
Avatar clips are pre-generated and served as static files.
All processing runs on CPU — no GPU required.

Audit fixes applied:
- Per-request session_id for LLM history isolation
- Public clone_voice() and list_voices() methods (no private access from routes)
"""

from __future__ import annotations

import asyncio
import base64
import functools
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from src.config import Settings
from src.pipeline.asr import ASREngine
from src.pipeline.emotions import EmotionEngine
from src.pipeline.llm import LLMEngine
from src.pipeline.tts import TTSEngine, VoiceInfo
from src.pipeline.visemes import VisemeExtractor
from src.utils.exceptions import PipelineError
from src.utils.logger import setup_logger, log_with_latency
from src.utils.metrics import (
    ACTIVE_SESSIONS,
    INFERENCE_LATENCY,
    GUARDRAIL_VIOLATIONS,
)
logger = setup_logger("pipeline.orchestrator")

# Default avatar reference image path
DEFAULT_AVATAR_DIR = Path("avatars")

# Emotion-to-body-state mapping for avatar clip selection
EMOTION_TO_BODY_STATE: dict[str, str] = {
    "neutral": "talking_happy",
    "happy": "talking_happy",
    "sad": "talking_sad",
    "angry": "talking_sad",
    "surprised": "talking_happy",
    "fearful": "talking_sad",
    "disgusted": "talking_sad",
    "contempt": "talking_sad",
}


@dataclass
class PipelineResult:
    """Result of a full pipeline execution.

    Attributes:
        audio_path: Path to the generated audio file.
        response_text: LLM-generated response text.
        detected_emotion: Detected emotion from the input.
        body_state: Avatar clip state to play (idle/thinking/talking_happy/talking_sad).
        lip_sync: Word-level timing data for lip synchronization.
        total_latency_ms: Total end-to-end processing time.
        breakdown: Per-layer latency breakdown.
        kb_confidence: Knowledge Base search confidence score.
        kb_sources: Source document filenames from KB retrieval.
        escalated: Whether this response was escalated to a human.
        escalation_id: Escalation event ID if escalated.
    """

    audio_path: str = ""
    response_text: str = ""
    detected_emotion: str = "neutral"
    body_state: str = "idle"
    lip_sync: dict = field(default_factory=dict)
    total_latency_ms: int = 0
    breakdown: dict[str, int] = field(default_factory=dict)
    kb_confidence: float = 0.0
    kb_sources: list[str] = field(default_factory=list)
    escalated: bool = False
    escalation_id: Optional[str] = None
    detected_language: str = ""


@dataclass
class StreamChunk:
    """A single chunk in the streaming conversation protocol.

    Sent from Central to GPU Render Node via WebSocket.

    Types:
        thinking       — avatar enters thinking state with motion params
        response_start — full response text + emotion + initial motion
        audio_chunk    — base64 PCM audio + lip params + micro motion
        response_end   — session cost + duration + idle motion
    """

    type: str
    session_id: str = ""
    text: str = ""
    emotion: str = ""
    motion: dict = field(default_factory=dict)
    audio_b64: str = ""
    seq: int = 0
    duration_ms: int = 0
    lip_params: dict = field(default_factory=dict)
    cost: str = ""
    duration_sec: float = 0.0

    def to_dict(self) -> dict:
        """Serialize to JSON-safe dict for WebSocket transport."""
        d: dict[str, Any] = {"type": self.type, "session_id": self.session_id}
        if self.text:
            d["text"] = self.text
        if self.emotion:
            d["emotion"] = self.emotion
        if self.motion:
            d["motion"] = self.motion
        if self.audio_b64:
            d["audio"] = self.audio_b64
        if self.seq > 0 or self.type == "audio_chunk":
            d["seq"] = self.seq
        if self.duration_ms:
            d["duration_ms"] = self.duration_ms
        if self.lip_params:
            d["lip_params"] = self.lip_params
        if self.cost:
            d["cost"] = self.cost
        if self.duration_sec > 0:
            d["duration_sec"] = self.duration_sec
        return d


class SmartTalkerPipeline:
    """Main pipeline orchestrator for SmartTalker.

    Coordinates the full processing flow:
    - Text input:  Emotion -> LLM -> TTS
    - Audio input: ASR -> Emotion -> LLM -> TTS

    All processing runs on CPU. Avatar clips are pre-generated
    and served as static files — no GPU required at runtime.

    Args:
        config: Application settings.
    """

    def __init__(self, config: Settings, db: Any = None) -> None:
        self._config = config
        self._db = db
        self._asr = ASREngine(config)
        self._llm = LLMEngine(config)
        self._tts = TTSEngine(config)
        self._emotion = EmotionEngine(config)

        # Knowledge Base and Training engines (conditional on config)
        self._kb: Any = None
        self._training: Any = None
        self._guardrails: Any = None

        if config.kb_enabled:
            from src.pipeline.knowledge_base import KnowledgeBaseEngine
            self._kb = KnowledgeBaseEngine(config)

        if config.training_enabled:
            from src.pipeline.training import TrainingEngine
            self._training = TrainingEngine(config, kb_engine=self._kb, db=db)

        if config.guardrails_enabled:
            from src.pipeline.guardrails import GuardrailsEngine
            self._guardrails = GuardrailsEngine(config, db=db)

        self._start_time = time.time()

        logger.info("SmartTalkerPipeline initialized (CPU mode)")

    @property
    def uptime_seconds(self) -> float:
        """Seconds since pipeline was initialized."""
        return round(time.time() - self._start_time, 1)

    def load_all(self) -> None:
        """Load all pipeline models.

        Non-critical models are loaded with graceful fallbacks.
        """
        logger.info("Loading pipeline models...")
        start = time.perf_counter()

        # Core models (warn but don't fail)
        for name, loader in [
            ("ASR", self._asr.load),
            ("TTS", self._tts.load),
        ]:
            try:
                loader()
            except PipelineError as exc:
                logger.warning(f"{name} model failed to load: {exc.message}")

        # Optional: Emotion model (lightweight)
        try:
            self._emotion.load()
        except Exception as exc:
            logger.warning(f"Emotion model failed to load: {exc}")

        # Knowledge Base (sync ChromaDB init)
        if self._kb is not None:
            try:
                self._kb.load()
            except Exception as exc:
                logger.warning(f"KB engine failed to load: {exc}")

        elapsed = int((time.perf_counter() - start) * 1000)
        log_with_latency(logger, "Pipeline models loaded", elapsed)

    # ═════════════════════════════════════════════════════════════════════
    # Body State Selection
    # ═════════════════════════════════════════════════════════════════════

    @staticmethod
    def _select_body_state(text: str, emotion: str, turn_count: int = 0) -> str:
        """Select avatar body state based on emotion and context.

        Args:
            text: The response text.
            emotion: Detected emotion label.
            turn_count: Number of conversation turns so far.

        Returns:
            Body state string: idle, thinking, talking_happy, or talking_sad.
        """
        return EMOTION_TO_BODY_STATE.get(emotion, "talking_happy")

    # ═════════════════════════════════════════════════════════════════════
    # Text Pipeline
    # ═════════════════════════════════════════════════════════════════════

    async def process_text(
        self,
        text: str,
        avatar_id: str = "default",
        voice_id: Optional[str] = None,
        emotion: str = "neutral",
        language: str = "ar",
        session_id: str = "default",
    ) -> PipelineResult:
        """Process text input through Emotion -> LLM -> TTS.

        Args:
            text: User's text input.
            avatar_id: Avatar identifier.
            voice_id: Optional voice ID for TTS cloning.
            emotion: Emotion label (auto-detected if "neutral").
            language: Target response language.
            session_id: Unique session identifier for conversation isolation.

        Returns:
            PipelineResult with audio path, text, body state, lip sync, and latency.
        """
        start = time.perf_counter()
        breakdown: dict[str, int] = {}
        kb_confidence = 0.0
        kb_sources: list[str] = []
        escalated = False
        escalation_id: Optional[str] = None

        # ── Emotion Detection (lightweight — OK on event loop) ────────────
        if emotion == "neutral":
            with INFERENCE_LATENCY.labels(model_type="emotion").time():
                emo_result = self._emotion.detect_from_text(text)
            emotion = emo_result.primary_emotion
            breakdown["emotion_ms"] = emo_result.latency_ms

        # ── KB Query (RAG context retrieval) ──────────────────────────────
        kb_context: Optional[str] = None
        if self._kb is not None and self._kb.is_loaded:
            try:
                kb_result = await self._kb.query(text)
                breakdown["kb_ms"] = kb_result.latency_ms
                kb_confidence = kb_result.confidence
                kb_sources = [c.get("filename", "") for c in kb_result.source_chunks]
                if kb_result.has_answer:
                    kb_context = kb_result.context
            except Exception as exc:
                logger.warning(f"KB query failed: {exc}")

        # ── LLM (already async via httpx) ─────────────────────────────────
        with INFERENCE_LATENCY.labels(model_type="llm").time():
            llm_result = await self._llm.generate(
                user_text=text,
                emotion=emotion,
                language=language,
                session_id=session_id,
                kb_context=kb_context,
            )
        breakdown["llm_ms"] = llm_result.latency_ms

        # Update session metrics
        ACTIVE_SESSIONS.set(self._llm.session_count)

        # ── Guardrails Check (after LLM, before TTS) ─────────────────────
        response_text = llm_result.text
        if self._guardrails is not None:
            try:
                if not self._guardrails.is_loaded:
                    await self._guardrails.load()
                gr_result = await self._guardrails.check_response(
                    avatar_id=avatar_id,
                    session_id=session_id,
                    response_text=response_text,
                    user_question=text,
                )
                if not gr_result.passed:
                    response_text = gr_result.sanitized_text
                    for v in gr_result.violations:
                        GUARDRAIL_VIOLATIONS.labels(
                            avatar_id=avatar_id,
                            violation_type=v.get("type", "unknown"),
                        ).inc()
                if gr_result.escalation_triggered:
                    escalated = True
                breakdown["guardrails_ms"] = int(
                    (time.perf_counter() - start) * 1000
                ) - sum(breakdown.values())
            except Exception as exc:
                logger.warning(f"Guardrails check failed: {exc}")

        # ── Escalation Check ──────────────────────────────────────────────
        if self._training is not None and self._training.is_loaded:
            try:
                should_esc, matched_skill = await self._training.should_escalate(
                    avatar_id=avatar_id,
                    question=text,
                    ai_confidence=kb_confidence,
                )
                if should_esc:
                    escalated = True
                    esc_event = await self._training.create_escalation(
                        session_id=session_id,
                        avatar_id=avatar_id,
                        skill_id=matched_skill,
                        question=text,
                        confidence=kb_confidence,
                    )
                    escalation_id = esc_event.event_id
            except Exception as exc:
                logger.warning(f"Escalation check failed: {exc}")

        # ── Body State Selection ──────────────────────────────────────────
        body_state = self._select_body_state(response_text, emotion)

        # ── TTS (runs on CPU in thread pool to avoid blocking) ────────────
        voice_ref = self._resolve_voice_ref(voice_id)
        loop = asyncio.get_running_loop()
        with INFERENCE_LATENCY.labels(model_type="tts").time():
            tts_result = await loop.run_in_executor(
                None,
                functools.partial(
                    self._tts.synthesize,
                    text=response_text,
                    voice_ref=voice_ref,
                    emotion=emotion,
                    language=language,
                ),
            )
        breakdown["tts_ms"] = tts_result.latency_ms

        total_ms = int((time.perf_counter() - start) * 1000)

        result = PipelineResult(
            audio_path=tts_result.audio_path,
            response_text=response_text,
            detected_emotion=emotion,
            body_state=body_state,
            lip_sync=tts_result.lip_sync,
            total_latency_ms=total_ms,
            breakdown=breakdown,
            kb_confidence=kb_confidence,
            kb_sources=kb_sources,
            escalated=escalated,
            escalation_id=escalation_id,
        )

        log_with_latency(
            logger, "Text pipeline complete", total_ms,
            extra={"input_length": len(text), "emotion": emotion, "body_state": body_state,
                   "kb_confidence": kb_confidence, "escalated": escalated},
        )
        return result

    # ═════════════════════════════════════════════════════════════════════
    # Audio Pipeline
    # ═════════════════════════════════════════════════════════════════════

    async def process_audio(
        self,
        audio_path: str,
        avatar_id: str = "default",
        voice_id: Optional[str] = None,
        emotion: str = "neutral",
        language: str = "ar",
        session_id: str = "default",
    ) -> PipelineResult:
        """Process audio input through ASR -> Emotion -> LLM -> TTS.

        Args:
            audio_path: Path to the user's audio file.
            avatar_id: Avatar identifier.
            voice_id: Optional voice ID for TTS cloning.
            emotion: Emotion label (auto-detected if "neutral").
            language: Target response language.
            session_id: Unique session identifier for conversation isolation.

        Returns:
            PipelineResult with audio path, text, body state, lip sync, and latency.
        """
        start = time.perf_counter()
        breakdown: dict[str, int] = {}
        kb_confidence = 0.0
        kb_sources: list[str] = []
        escalated = False
        escalation_id: Optional[str] = None

        # ── ASR (runs on CPU in thread pool) ──────────────────────────────
        loop = asyncio.get_running_loop()
        with INFERENCE_LATENCY.labels(model_type="asr").time():
            asr_result = await loop.run_in_executor(
                None, functools.partial(self._asr.transcribe, audio_path)
            )
        breakdown["asr_ms"] = asr_result.latency_ms
        detected_lang = asr_result.language if asr_result.language != "unknown" else language

        # ── Emotion Detection ────────────────────────────────────────────
        if emotion == "neutral":
            emo_result = self._emotion.detect_from_text(asr_result.text)
            emotion = emo_result.primary_emotion
            breakdown["emotion_ms"] = emo_result.latency_ms

        # ── KB Query (RAG context retrieval) ──────────────────────────────
        kb_context: Optional[str] = None
        if self._kb is not None and self._kb.is_loaded:
            try:
                kb_result = await self._kb.query(asr_result.text)
                breakdown["kb_ms"] = kb_result.latency_ms
                kb_confidence = kb_result.confidence
                kb_sources = [c.get("filename", "") for c in kb_result.source_chunks]
                if kb_result.has_answer:
                    kb_context = kb_result.context
            except Exception as exc:
                logger.warning(f"KB query failed: {exc}")

        # ── LLM (already async) ──────────────────────────────────────────
        llm_result = await self._llm.generate(
            user_text=asr_result.text,
            emotion=emotion,
            language=detected_lang,
            session_id=session_id,
            kb_context=kb_context,
        )
        breakdown["llm_ms"] = llm_result.latency_ms

        # ── Guardrails Check (after LLM, before TTS) ─────────────────────
        response_text = llm_result.text
        if self._guardrails is not None:
            try:
                if not self._guardrails.is_loaded:
                    await self._guardrails.load()
                gr_result = await self._guardrails.check_response(
                    avatar_id=avatar_id,
                    session_id=session_id,
                    response_text=response_text,
                    user_question=asr_result.text,
                )
                if not gr_result.passed:
                    response_text = gr_result.sanitized_text
                    for v in gr_result.violations:
                        GUARDRAIL_VIOLATIONS.labels(
                            avatar_id=avatar_id,
                            violation_type=v.get("type", "unknown"),
                        ).inc()
                if gr_result.escalation_triggered:
                    escalated = True
            except Exception as exc:
                logger.warning(f"Guardrails check failed: {exc}")

        # ── Escalation Check ──────────────────────────────────────────────
        if self._training is not None and self._training.is_loaded:
            try:
                should_esc, matched_skill = await self._training.should_escalate(
                    avatar_id=avatar_id,
                    question=asr_result.text,
                    ai_confidence=kb_confidence,
                )
                if should_esc:
                    escalated = True
                    esc_event = await self._training.create_escalation(
                        session_id=session_id,
                        avatar_id=avatar_id,
                        skill_id=matched_skill,
                        question=asr_result.text,
                        confidence=kb_confidence,
                    )
                    escalation_id = esc_event.event_id
            except Exception as exc:
                logger.warning(f"Escalation check failed: {exc}")

        # ── Body State Selection ──────────────────────────────────────────
        body_state = self._select_body_state(response_text, emotion)

        # ── TTS (runs on CPU in thread pool) ──────────────────────────────
        voice_ref = self._resolve_voice_ref(voice_id)
        tts_result = await loop.run_in_executor(
            None,
            functools.partial(
                self._tts.synthesize,
                text=response_text,
                voice_ref=voice_ref,
                emotion=emotion,
                language=detected_lang,
            ),
        )
        breakdown["tts_ms"] = tts_result.latency_ms

        total_ms = int((time.perf_counter() - start) * 1000)

        result = PipelineResult(
            audio_path=tts_result.audio_path,
            response_text=response_text,
            detected_emotion=emotion,
            body_state=body_state,
            lip_sync=tts_result.lip_sync,
            total_latency_ms=total_ms,
            breakdown=breakdown,
            kb_confidence=kb_confidence,
            kb_sources=kb_sources,
            escalated=escalated,
            escalation_id=escalation_id,
        )

        log_with_latency(
            logger, "Audio pipeline complete", total_ms,
            extra={"transcribed": asr_result.text[:80], "emotion": emotion, "body_state": body_state,
                   "kb_confidence": kb_confidence, "escalated": escalated},
        )
        return result

    # ═════════════════════════════════════════════════════════════════════
    # Streaming Pipeline (for GPU Render Nodes)
    # ═════════════════════════════════════════════════════════════════════

    async def process_text_stream(
        self,
        text: str,
        avatar_id: str = "default",
        voice_id: Optional[str] = None,
        emotion: str = "neutral",
        language: str = "ar",
        session_id: str = "default",
    ) -> AsyncIterator[StreamChunk]:
        """Process text through the streaming pipeline.

        Yields StreamChunk objects for real-time relay to GPU Render Nodes:
        1. thinking      — avatar enters contemplative state
        2. response_start — full text + emotion + motion params
        3. audio_chunk    — repeated: base64 audio + lip + micro motion
        4. response_end   — final cost + duration + idle motion

        Args:
            text: User's text input.
            avatar_id: Avatar identifier.
            voice_id: Optional voice ID for TTS cloning.
            emotion: Emotion label (auto-detected if "neutral").
            language: Target response language.
            session_id: Unique session identifier.

        Yields:
            StreamChunk objects in protocol order.
        """
        start = time.perf_counter()

        # 1. Yield "thinking" state
        thinking_motion = EmotionEngine.get_thinking_motion()
        yield StreamChunk(
            type="thinking",
            session_id=session_id,
            motion=thinking_motion.to_dict(),
        )

        # ── Emotion Detection ────────────────────────────────────────
        if emotion == "neutral":
            emo_result = self._emotion.detect_from_text(text)
            emotion = emo_result.primary_emotion

        # ── KB Query (RAG context retrieval) ─────────────────────────
        kb_context: Optional[str] = None
        if self._kb is not None and self._kb.is_loaded:
            try:
                kb_result = await self._kb.query(text)
                if kb_result.has_answer:
                    kb_context = kb_result.context
            except Exception as exc:
                logger.warning(f"KB query failed in streaming: {exc}")

        # ── LLM ──────────────────────────────────────────────────────
        llm_result = await self._llm.generate(
            user_text=text,
            emotion=emotion,
            language=language,
            session_id=session_id,
            kb_context=kb_context,
        )
        ACTIVE_SESSIONS.set(self._llm.session_count)

        # 2. Yield "response_start" with motion params
        motion_params = EmotionEngine.get_motion_params(emotion)
        yield StreamChunk(
            type="response_start",
            session_id=session_id,
            text=llm_result.text,
            emotion=emotion,
            motion=motion_params.to_dict(),
        )

        # 3. Stream TTS chunks
        voice_ref = self._resolve_voice_ref(voice_id)
        loop = asyncio.get_running_loop()
        total_audio_ms = 0

        # Run TTS streaming in thread pool (CPU-bound)
        chunks = await loop.run_in_executor(
            None,
            lambda: list(self._tts.synthesize_stream(
                text=llm_result.text,
                voice_ref=voice_ref,
                emotion=emotion,
                language=language,
            )),
        )

        # Split response text roughly across chunks for viseme extraction
        words = llm_result.text.split()
        words_per_chunk = max(1, len(words) // max(len(chunks), 1))

        for tts_chunk in chunks:
            # Viseme extraction for this chunk's text portion
            chunk_start_word = tts_chunk.seq * words_per_chunk
            chunk_words = words[chunk_start_word:chunk_start_word + words_per_chunk]
            chunk_text = " ".join(chunk_words) if chunk_words else ""

            lip_params = VisemeExtractor.extract_from_text(
                chunk_text, tts_chunk.duration_ms,
            )
            micro_motion = EmotionEngine.get_micro_motion(emotion, tts_chunk.seq)

            audio_b64 = base64.b64encode(tts_chunk.audio_bytes).decode("ascii")
            total_audio_ms += tts_chunk.duration_ms

            yield StreamChunk(
                type="audio_chunk",
                session_id=session_id,
                seq=tts_chunk.seq,
                audio_b64=audio_b64,
                duration_ms=tts_chunk.duration_ms,
                lip_params=lip_params.to_dict(),
                motion=micro_motion,
            )

        # 4. Yield "response_end"
        total_sec = round(total_audio_ms / 1000, 2)
        cost = round(total_sec * 0.001, 6)  # $0.001/sec
        idle_motion = EmotionEngine.get_idle_motion()

        yield StreamChunk(
            type="response_end",
            session_id=session_id,
            duration_sec=total_sec,
            cost=f"${cost}",
            motion=idle_motion.to_dict(),
        )

        total_ms = int((time.perf_counter() - start) * 1000)
        log_with_latency(
            logger, "Streaming text pipeline complete", total_ms,
            extra={"input_length": len(text), "emotion": emotion, "chunks": len(chunks)},
        )

    async def process_audio_stream(
        self,
        audio_path: str,
        avatar_id: str = "default",
        voice_id: Optional[str] = None,
        emotion: str = "neutral",
        language: str = "ar",
        session_id: str = "default",
    ) -> AsyncIterator[StreamChunk]:
        """Process audio input through the streaming pipeline (ASR -> LLM -> streaming TTS).

        Same protocol as process_text_stream but with ASR transcription first.

        Yields:
            StreamChunk objects in protocol order.
        """
        # ASR transcription first
        loop = asyncio.get_running_loop()
        asr_result = await loop.run_in_executor(
            None, functools.partial(self._asr.transcribe, audio_path)
        )
        detected_lang = asr_result.language if asr_result.language != "unknown" else language

        # Delegate to text streaming pipeline
        async for chunk in self.process_text_stream(
            text=asr_result.text,
            avatar_id=avatar_id,
            voice_id=voice_id,
            emotion=emotion,
            language=detected_lang,
            session_id=session_id,
        ):
            yield chunk

    # ═════════════════════════════════════════════════════════════════════
    # Public API (used by routes — no private member access)
    # ═════════════════════════════════════════════════════════════════════

    def clone_voice(self, reference_audio: str, voice_name: str) -> str:
        """Register a new cloned voice from reference audio."""
        return self._tts.clone_voice(
            reference_audio=reference_audio,
            voice_name=voice_name,
        )

    def list_voices(self) -> list[VoiceInfo]:
        """List all registered voices."""
        return self._tts.list_voices()

    # ═════════════════════════════════════════════════════════════════════
    # Helpers
    # ═════════════════════════════════════════════════════════════════════

    def _resolve_voice_ref(self, voice_id: Optional[str]) -> Optional[str]:
        """Resolve a voice ID to a reference audio path."""
        if not voice_id:
            return None

        voices = {v.voice_id: v for v in self._tts.list_voices()}
        voice_info = voices.get(voice_id)
        if voice_info and voice_info.reference_audio:
            return voice_info.reference_audio
        return None

    @staticmethod
    def _resolve_avatar_image(avatar_id: str) -> Optional[str]:
        """Resolve an avatar ID to a reference image path."""
        if not avatar_id or "/" in avatar_id or "\\" in avatar_id or ".." in avatar_id:
            logger.warning("Invalid avatar_id rejected", extra={"avatar_id": avatar_id})
            return None

        avatar_dir = DEFAULT_AVATAR_DIR / avatar_id

        try:
            avatar_dir.resolve().relative_to(DEFAULT_AVATAR_DIR.resolve())
        except ValueError:
            logger.warning("Avatar path traversal blocked", extra={"avatar_id": avatar_id})
            return None

        if not avatar_dir.exists():
            return None

        for ext in [".png", ".jpg", ".jpeg", ".webp"]:
            for pattern in [f"reference{ext}", f"avatar{ext}", f"{avatar_id}{ext}"]:
                candidate = avatar_dir / pattern
                if candidate.exists():
                    return str(candidate)

        for f in avatar_dir.iterdir():
            if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                return str(f)

        return None

    async def health_check(self) -> dict[str, Any]:
        """Check system health, model status, and LLM API connectivity.

        Returns:
            Dictionary with status, loaded models, and LLM reachability.
        """
        # Check LLM API connectivity
        llm_reachable = False
        try:
            client = await self._llm._get_client()
            resp = await client.post(
                "/chat/completions",
                json={"model": self._llm._model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1},
            )
            llm_reachable = resp.status_code == 200
        except Exception:
            llm_reachable = False

        models_loaded = {
            "asr": self._asr.is_loaded,
            "tts": self._tts.is_loaded,
            "emotion": self._emotion.is_loaded,
            "llm": llm_reachable,
            "kb": self._kb.is_loaded if self._kb is not None else False,
            "training": self._training.is_loaded if self._training is not None else False,
            "guardrails": self._guardrails.is_loaded if self._guardrails is not None else False,
        }

        all_ok = any(models_loaded.values()) and llm_reachable
        status = "healthy" if all_ok else "degraded"

        return {
            "status": status,
            "models_loaded": models_loaded,
            "uptime_s": self.uptime_seconds,
        }

    async def unload_all(self) -> None:
        """Unload all models and free resources."""
        logger.info("Unloading all pipeline models...")

        self._asr.unload()
        self._tts.unload()
        self._emotion.unload()
        await self._llm.close()

        if self._kb is not None:
            self._kb.unload()
        if self._training is not None:
            await self._training.unload()
        if self._guardrails is not None:
            await self._guardrails.unload()

        logger.info("All pipeline models unloaded")
