"""Visitor WebSocket handler for direct voice sessions.

Handles end-user conversations on wss://ws.maskki.com/session.
The full pipeline runs here: ASR → AgentEngine → TTS (→ optional RunPod render).
Audio is streamed back to the browser; video URL is sent if avatar_mode == "video".

Protocol:
  → auth:            {type: "auth", token: "jwt...", employee_id: "..."}
  ← auth_ok:         {type: "auth_ok", avatar_id, avatar_mode, session_id}
  → audio_chunk:     {type: "audio_chunk", audio: "<base64 PCM 16kHz>"}
  → audio_end:       {type: "audio_end"}
  ← text_response:   {type: "text_response", text, emotion, session_id}
  ← audio_response:  {type: "audio_response", audio: "<base64 PCM 48kHz>", duration_ms}
  ← video_url:       {type: "video_url", url, session_id}
  ← fallback_vrm:    {type: "fallback_vrm", session_id, reason}
  → text_message:    {type: "text_message", text: "..."}
  ← action_required: {type: "action_required", tool_id, tool_name, description, parameters}
  → action_response: {type: "action_response", approved: bool}
  → disconnect:      connection close
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
import uuid
from typing import Any, Optional

from fastapi import WebSocket, WebSocketDisconnect

from src.config import get_settings
from src.db.models import APICostRecord
from src.utils.exceptions import SmartTalkerError
from src.utils.logger import setup_logger, log_with_latency

logger = setup_logger("api.ws_visitor")

_MAX_MSG_SIZE = 512 * 1024  # 512 KB per message
_ACTION_RESPONSE_TIMEOUT = 60.0  # seconds to wait for visitor confirmation


async def visitor_session_handler(websocket: WebSocket) -> None:
    """Handle a single visitor voice/text session.

    Mounted at /session on the FastAPI app.
    """
    app = websocket.app
    config = get_settings()
    pipeline = app.state.pipeline
    billing = getattr(app.state, "billing", None)
    agent_engine = getattr(app.state, "agent_engine", None)

    await websocket.accept()
    session_id = uuid.uuid4().hex[:16]
    employee_id: Optional[str] = None
    customer_id: Optional[str] = None
    avatar = None
    asr_session = None
    session_start = time.perf_counter()

    try:
        # ── Step 1: Auth ──────────────────────────────────────────────
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=15.0)
        msg = json.loads(raw)

        if msg.get("type") != "auth":
            await _send(websocket, {"type": "error", "error": "Expected auth message"})
            await websocket.close(code=4001)
            return

        token = msg.get("token", "")
        employee_id = msg.get("employee_id", "")

        # Validate JWT via billing / DB
        customer_id, avatar_rec = await _authenticate(
            app, token, employee_id
        )
        if customer_id is None:
            await _send(websocket, {
                "type": "error", "error": "Authentication failed",
            })
            await websocket.close(code=4003)
            return

        avatar = avatar_rec
        avatar_mode = getattr(avatar, "avatar_type", "vrm") if avatar else "vrm"

        await _send(websocket, {
            "type": "auth_ok",
            "session_id": session_id,
            "avatar_id": employee_id,
            "avatar_mode": avatar_mode,
        })

        logger.info(
            "Visitor session authenticated",
            extra={
                "session_id": session_id,
                "customer_id": customer_id,
                "employee_id": employee_id,
                "avatar_mode": avatar_mode,
            },
        )

        # ── Step 2: Message loop ──────────────────────────────────────
        while True:
            raw = await websocket.receive_text()
            if len(raw) > _MAX_MSG_SIZE:
                await _send(websocket, {"type": "error", "error": "Message too large"})
                continue

            msg = json.loads(raw)
            msg_type = msg.get("type", "")

            # ── Audio chunk → accumulate in ASR session ───────────
            if msg_type == "audio_chunk":
                audio_b64 = msg.get("audio", "")
                if not audio_b64:
                    continue

                pcm_bytes = base64.b64decode(audio_b64)

                if asr_session is None:
                    language = msg.get("language", "ar")
                    asr_session = await pipeline._asr.create_session(language)

                await asr_session.send_audio(pcm_bytes)

            # ── Audio end → process full pipeline ─────────────────
            elif msg_type == "audio_end":
                if asr_session is None:
                    continue

                turn_start = time.perf_counter()

                # Finish ASR
                asr_result = await asr_session.finish()
                asr_session = None

                transcript = asr_result.text.strip()
                if not transcript:
                    continue

                # Record ASR cost
                await _record_cost(
                    app, "asr", customer_id, session_id,
                    cost_usd=asr_result.cost_usd,
                    duration_ms=asr_result.latency_ms,
                    details={"text_length": len(transcript), "language": asr_result.language},
                )

                # Process through pipeline
                await _process_and_respond(
                    websocket=websocket,
                    pipeline=pipeline,
                    app=app,
                    agent_engine=agent_engine,
                    text=transcript,
                    session_id=session_id,
                    employee_id=employee_id or "",
                    customer_id=customer_id,
                    avatar=avatar,
                    avatar_mode=avatar_mode,
                    turn_start=turn_start,
                )

            # ── Text message → AgentEngine + TTS ─────────────────
            elif msg_type == "text_message":
                text = msg.get("text", "").strip()
                if not text:
                    continue

                turn_start = time.perf_counter()

                await _process_and_respond(
                    websocket=websocket,
                    pipeline=pipeline,
                    app=app,
                    agent_engine=agent_engine,
                    text=text,
                    session_id=session_id,
                    employee_id=employee_id or "",
                    customer_id=customer_id,
                    avatar=avatar,
                    avatar_mode=avatar_mode,
                    turn_start=turn_start,
                )

            # ── Ping/keepalive ────────────────────────────────────
            elif msg_type == "ping":
                await _send(websocket, {"type": "pong"})

    except WebSocketDisconnect:
        logger.info("Visitor disconnected", extra={"session_id": session_id})
    except asyncio.TimeoutError:
        logger.warning("Visitor auth timeout", extra={"session_id": session_id})
        await websocket.close(code=4008, reason="Auth timeout")
    except Exception as exc:
        logger.error(f"Visitor session error: {exc}", extra={"session_id": session_id})
    finally:
        # Clean up any open ASR session
        if asr_session is not None:
            try:
                await asr_session.finish()
            except Exception:
                pass

        duration_s = time.perf_counter() - session_start

        # Deduct billing seconds
        if billing and customer_id:
            try:
                await billing.deduct_seconds(
                    customer_id=customer_id,
                    seconds=duration_s,
                    session_id=session_id,
                )
            except Exception as exc:
                logger.warning(f"Billing deduction failed: {exc}")

        log_with_latency(
            logger,
            "Visitor session ended",
            int(duration_s * 1000),
            extra={"session_id": session_id, "duration_s": f"{duration_s:.1f}"},
        )


# ── Internal Helpers ───────────────────────────────────────────────────────


async def _send(ws: WebSocket, data: dict) -> None:
    """Send a JSON message over the WebSocket."""
    await ws.send_text(json.dumps(data))


async def _record_cost(
    app: Any,
    service: str,
    customer_id: str,
    session_id: str,
    cost_usd: float,
    tokens_used: int = 0,
    duration_ms: int = 0,
    details: dict | None = None,
) -> None:
    """Record an API cost entry in the database (fire-and-forget)."""
    db = getattr(app.state, "db", None)
    if db is None or cost_usd <= 0:
        return
    try:
        record = APICostRecord(
            service=service,
            customer_id=customer_id,
            session_id=session_id,
            cost_usd=cost_usd,
            tokens_used=tokens_used,
            duration_ms=duration_ms,
            details=json.dumps(details) if details else "{}",
        )
        async with db.session() as session:
            session.add(record)
            await session.commit()
    except Exception as exc:
        logger.debug(f"Cost recording failed: {exc}")


async def _authenticate(
    app: Any, token: str, employee_id: str
) -> tuple[Optional[str], Any]:
    """Validate JWT and resolve employee/avatar.

    Returns (customer_id, avatar_record) or (None, None) on failure.
    """
    db = getattr(app.state, "db", None)
    if db is None or not token:
        return None, None

    try:
        # Decode JWT to extract customer_id
        import jwt as pyjwt

        config = get_settings()
        secret = config.api_key or "dev-secret"
        payload = pyjwt.decode(token, secret, algorithms=["HS256"])
        customer_id = payload.get("customer_id", payload.get("sub", ""))

        if not customer_id:
            return None, None

        # Check suspension
        kill_switch = getattr(app.state, "kill_switch", None)
        if kill_switch and await kill_switch.is_suspended(customer_id):
            logger.warning("Suspended customer attempted session", extra={"customer_id": customer_id})
            return None, None

        # Load avatar record
        from sqlalchemy import select
        from src.db.models import Avatar

        async with db.session() as session:
            result = await session.execute(
                select(Avatar).where(
                    Avatar.id == employee_id,
                    Avatar.customer_id == customer_id,
                )
            )
            avatar = result.scalar_one_or_none()

        return customer_id, avatar

    except Exception as exc:
        logger.warning(f"Auth failed: {exc}")
        return None, None


def _make_confirmation_callback(websocket: WebSocket, session_id: str):
    """Create an async confirmation callback for the AgentEngine.

    When a tool has requires_confirmation=True, this callback sends an
    action_required message to the visitor and waits for action_response.
    """

    async def callback(
        tool_id: str,
        tool_name: str,
        description: str,
        parameters: dict,
    ) -> bool:
        # Send action_required to visitor
        await _send(websocket, {
            "type": "action_required",
            "tool_id": tool_id,
            "tool_name": tool_name,
            "description": description,
            "parameters": parameters,
            "session_id": session_id,
        })

        # Wait for action_response from visitor
        try:
            raw = await asyncio.wait_for(
                websocket.receive_text(),
                timeout=_ACTION_RESPONSE_TIMEOUT,
            )
            msg = json.loads(raw)
            if msg.get("type") == "action_response":
                approved = msg.get("approved", False)
                logger.info(
                    "Action response received",
                    extra={
                        "tool_id": tool_id,
                        "approved": approved,
                        "session_id": session_id,
                    },
                )
                return bool(approved)
        except asyncio.TimeoutError:
            logger.warning(
                "Action response timeout",
                extra={"tool_id": tool_id, "session_id": session_id},
            )
        except Exception as exc:
            logger.warning(f"Action response error: {exc}")

        return False

    return callback


async def _process_and_respond(
    *,
    websocket: WebSocket,
    pipeline: Any,
    app: Any,
    agent_engine: Any,
    text: str,
    session_id: str,
    employee_id: str,
    customer_id: str,
    avatar: Any,
    avatar_mode: str,
    turn_start: float,
) -> None:
    """Run AgentEngine (or fallback LLM) → TTS → optional RunPod render."""

    language = getattr(avatar, "language", "ar") if avatar else "ar"
    voice_id = getattr(avatar, "voice_id", "default") if avatar else "default"

    # ── Generate response via AgentEngine or fallback LLM ─────────────
    llm_cost_usd = 0.0
    llm_tokens = 0
    llm_latency = 0
    if agent_engine is not None:
        try:
            confirmation_cb = _make_confirmation_callback(websocket, session_id)
            response_text = await agent_engine.handle_message(
                session_id=session_id,
                visitor_message=text,
                employee_id=employee_id,
                visitor_id=session_id,
                customer_id=customer_id,
                confirmation_callback=confirmation_cb,
            )
            emotion = "neutral"
        except Exception as exc:
            logger.warning(
                f"AgentEngine failed, falling back to direct LLM: {exc}",
                extra={"session_id": session_id},
            )
            # Fallback to direct LLM
            llm_result = await pipeline._llm.generate(
                text=text,
                language=language,
                session_id=session_id,
                avatar_id=employee_id,
            )
            response_text = llm_result.text
            emotion = llm_result.emotion
            llm_cost_usd = llm_result.cost_usd
            llm_tokens = llm_result.tokens_used
            llm_latency = llm_result.latency_ms
    else:
        # No agent engine — use direct LLM
        llm_result = await pipeline._llm.generate(
            text=text,
            language=language,
            session_id=session_id,
            avatar_id=employee_id,
        )
        response_text = llm_result.text
        emotion = llm_result.emotion
        llm_cost_usd = llm_result.cost_usd
        llm_tokens = llm_result.tokens_used
        llm_latency = llm_result.latency_ms

    # Record LLM cost
    if llm_cost_usd > 0:
        await _record_cost(
            app, "llm", customer_id, session_id,
            cost_usd=llm_cost_usd,
            tokens_used=llm_tokens,
            duration_ms=llm_latency,
            details={"text_length": len(text), "response_length": len(response_text)},
        )

    # Guardrails check before sending response / TTS
    guardrails = getattr(app.state, "guardrails", None)
    if guardrails is not None:
        employee_guardrails = {}
        if avatar:
            try:
                import json as _json
                raw = getattr(avatar, "guardrails", "{}")
                employee_guardrails = _json.loads(raw) if raw else {}
            except Exception:
                pass
        guard_result = guardrails.check_response(response_text, employee_guardrails, language)
        if not guard_result.approved:
            logger.warning(
                "Guardrail blocked response",
                extra={"session_id": session_id, "reason": guard_result.reason},
            )
        response_text = guard_result.text

    # Send text response immediately (low latency)
    await _send(websocket, {
        "type": "text_response",
        "text": response_text,
        "emotion": emotion,
        "session_id": session_id,
    })

    # ── TTS ───────────────────────────────────────────────────────────
    try:
        tts_stream = await pipeline._tts.synthesize_stream(
            text=response_text,
            voice_id=voice_id,
            language=language,
            emotion=emotion,
        )

        audio_bytes = await tts_stream.collect_all()
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
        duration_ms = int(tts_stream.duration_seconds * 1000)

        await _send(websocket, {
            "type": "audio_response",
            "audio": audio_b64,
            "duration_ms": duration_ms,
            "session_id": session_id,
        })

        # Record TTS cost
        await _record_cost(
            app, "tts", customer_id, session_id,
            cost_usd=tts_stream.cost_usd,
            duration_ms=duration_ms,
            details={"text_length": len(response_text)},
        )
    except Exception as exc:
        logger.warning(f"TTS failed: {exc}", extra={"session_id": session_id})
        audio_bytes = b""

    # ── RunPod video render (only for video avatars with face data) ────
    if (
        avatar_mode == "video"
        and avatar
        and getattr(avatar, "photo_preprocessed", False)
        and getattr(avatar, "face_data_url", "")
        and audio_bytes
    ):
        try:
            from src.services.runpod_client import RunPodServerless
            from src.services.r2_storage import R2Storage

            config = get_settings()
            r2 = R2Storage(config)
            runpod = RunPodServerless(config)

            # Skip rendering if agent has disabled video mode due to consecutive failures
            # Check the pipeline's shared RunPod client (the one the agent modifies)
            pipeline_runpod = getattr(pipeline, '_runpod', None) or getattr(pipeline, '_runpod_client', None)
            if pipeline_runpod is not None and getattr(pipeline_runpod, 'video_disabled', False) is True:
                logger.info("Video rendering disabled by agent — sending fallback_vrm")
                await _send(websocket, {
                    "type": "fallback_vrm",
                    "session_id": session_id,
                    "reason": "Video rendering temporarily disabled due to repeated failures",
                })
                raise Exception("video_disabled")  # Skip to finally block

            # Upload audio to R2
            audio_url = r2.upload_audio(session_id, audio_bytes)

            # Submit render job
            render_result = await runpod.render_lipsync(
                audio_url=audio_url,
                face_data_url=avatar.face_data_url,
                employee_id=employee_id,
                session_id=session_id,
            )

            await _send(websocket, {
                "type": "video_url",
                "url": render_result.video_url,
                "session_id": session_id,
            })

            # Record RunPod cost
            await _record_cost(
                app, "runpod", customer_id, session_id,
                cost_usd=render_result.cost_usd,
                duration_ms=render_result.execution_time_ms,
                details={"task": "render_lipsync", "job_id": render_result.job_id},
            )

            await runpod.close()

        except Exception as exc:
            logger.warning(f"RunPod render failed, falling back to VRM: {exc}")
            await _send(websocket, {
                "type": "fallback_vrm",
                "session_id": session_id,
                "reason": str(exc),
            })

    turn_ms = int((time.perf_counter() - turn_start) * 1000)
    log_with_latency(
        logger,
        "Turn processed",
        turn_ms,
        extra={"session_id": session_id, "text_len": len(response_text)},
    )
