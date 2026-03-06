"""Operator WebSocket handler for real-time customer session monitoring.

Allows business operators to observe customer avatar sessions,
view live camera feeds, read chat transcripts, and send override
responses to train the Avatar gradually.

Protocol (JSON frames):

Client -> Server:
    {"type": "auth", "api_key": "..."}
    {"type": "subscribe", "session_id": "..."}
    {"type": "unsubscribe"}
    {"type": "operator_message", "text": "...", "learn_skill_id": "...(optional)"}
    {"type": "training_feedback", "message_id": "...", "quality": "good"|"bad", "note": "..."}
    {"type": "ping"}

Server -> Client:
    {"type": "authenticated", "operator_id": "..."}
    {"type": "session_list", "sessions": [...]}
    {"type": "customer_video_frame", "frame": "base64..."}
    {"type": "customer_audio_chunk", "chunk": "base64..."}
    {"type": "customer_message", "role": "user"|"bot", "text": "...", ...}
    {"type": "document_received", "image_url": "...", "filename": "..."}
    {"type": "escalation_alert", "session_id": "...", "question": "...", ...}
    {"type": "subscribed", "session_id": "..."}
    {"type": "error", "error": "...", "detail": "..."}
    {"type": "pong"}
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from src.utils.logger import setup_logger

logger = setup_logger("api.operator_ws")


@dataclass
class OperatorSession:
    """Represents one connected operator.

    Attributes:
        operator_id: Unique identifier for this operator.
        websocket: The FastAPI WebSocket connection.
        subscribed_session_id: Customer session ID the operator is watching.
        connected_at: Timestamp of connection establishment.
        authenticated: Whether the operator has authenticated.
    """

    operator_id: str
    websocket: WebSocket
    subscribed_session_id: Optional[str] = None
    connected_at: float = field(default_factory=time.time)
    authenticated: bool = False


class OperatorWebSocketManager:
    """Manages operator WebSocket connections and customer session relay.

    Bridges the operator dashboard with active customer sessions,
    relaying video frames, audio chunks, chat messages, and
    document scans from customers to operators.

    Args:
        customer_ws_manager: Reference to the customer WebSocketManager.
        storage_dir: Directory for saved documents.
        api_key: API key for operator authentication.
    """

    def __init__(
        self,
        customer_ws_manager: Any,
        storage_dir: Path,
        api_key: Optional[str] = None,
        pipeline: Any = None,
        supervisor: Any = None,
    ) -> None:
        self._customer_manager = customer_ws_manager
        self._storage_dir = storage_dir
        self._api_key = api_key
        self._pipeline = pipeline
        self._supervisor = supervisor
        self._operators: dict[str, OperatorSession] = {}
        self._lock = asyncio.Lock()
        self._training_log: list[dict] = []
        self._escalation_times: dict[str, float] = {}  # session_id -> timestamp

        # Ensure documents directory exists
        self._docs_dir = storage_dir / "documents"
        self._docs_dir.mkdir(parents=True, exist_ok=True)

    @property
    def active_count(self) -> int:
        """Number of connected operators."""
        return len(self._operators)

    def get_customer_sessions(self) -> list[dict]:
        """Return a summary of active customer sessions."""
        sessions = []
        for sid, session in self._customer_manager._sessions.items():
            sessions.append({
                "session_id": sid,
                "client_ip": session.client_ip,
                "connected_at": session.connected_at,
                "language": session.config.language,
                "avatar_id": session.config.avatar_id,
                "is_recording": session.is_recording,
            })
        return sessions

    async def connect(self, websocket: WebSocket) -> OperatorSession:
        """Accept an operator WebSocket connection."""
        await websocket.accept()

        operator_id = f"op_{uuid.uuid4().hex[:12]}"
        operator = OperatorSession(
            operator_id=operator_id,
            websocket=websocket,
        )

        async with self._lock:
            self._operators[operator_id] = operator

        logger.info(
            "Operator connected",
            extra={"operator_id": operator_id},
        )
        return operator

    async def disconnect(self, operator: OperatorSession) -> None:
        """Clean up a disconnected operator."""
        async with self._lock:
            self._operators.pop(operator.operator_id, None)

        logger.info(
            "Operator disconnected",
            extra={"operator_id": operator.operator_id},
        )

    async def handle_session(self, operator: OperatorSession) -> None:
        """Main receive loop for an operator WebSocket."""
        try:
            # Require auth first
            if self._api_key:
                await self._send_json(operator.websocket, {
                    "type": "auth_required",
                })
            else:
                operator.authenticated = True
                await self._send_json(operator.websocket, {
                    "type": "authenticated",
                    "operator_id": operator.operator_id,
                })
                await self._send_session_list(operator)

            while True:
                message = await operator.websocket.receive()

                if message["type"] == "websocket.disconnect":
                    break

                if "text" in message and message["text"]:
                    await self._handle_message(operator, message["text"])

        except WebSocketDisconnect:
            pass
        except Exception as exc:
            logger.error(
                f"Operator session error: {exc}",
                extra={"operator_id": operator.operator_id},
            )

    async def _handle_message(self, operator: OperatorSession, raw: str) -> None:
        """Parse and dispatch operator messages."""
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            await self._send_error(operator.websocket, "Invalid JSON")
            return

        msg_type = data.get("type", "")

        # Auth must come first
        if not operator.authenticated:
            if msg_type == "auth":
                await self._handle_auth(operator, data)
            else:
                await self._send_error(
                    operator.websocket,
                    "Not authenticated",
                    "Send auth message first",
                )
            return

        handlers = {
            "subscribe": self._handle_subscribe,
            "unsubscribe": self._handle_unsubscribe,
            "operator_message": self._handle_operator_message,
            "training_feedback": self._handle_training_feedback,
            "refresh_sessions": self._handle_refresh_sessions,
            "ping": self._handle_ping,
        }

        handler = handlers.get(msg_type)
        if handler:
            await handler(operator, data)
        else:
            await self._send_error(
                operator.websocket,
                "Unknown message type",
                f"Unsupported: '{msg_type}'",
            )

    async def _handle_auth(self, operator: OperatorSession, data: dict) -> None:
        """Authenticate an operator."""
        import hmac

        client_key = data.get("api_key", "")
        if self._api_key and client_key and hmac.compare_digest(client_key, self._api_key):
            operator.authenticated = True
            await self._send_json(operator.websocket, {
                "type": "authenticated",
                "operator_id": operator.operator_id,
            })
            await self._send_session_list(operator)
            logger.info("Operator authenticated", extra={"operator_id": operator.operator_id})
        else:
            await self._send_error(operator.websocket, "Authentication failed")

    async def _handle_subscribe(self, operator: OperatorSession, data: dict) -> None:
        """Subscribe to a customer session's feed."""
        session_id = data.get("session_id", "")
        if not session_id:
            await self._send_error(operator.websocket, "Missing session_id")
            return

        if session_id not in self._customer_manager._sessions:
            await self._send_error(
                operator.websocket,
                "Session not found",
                f"No active customer session: {session_id}",
            )
            return

        operator.subscribed_session_id = session_id
        await self._send_json(operator.websocket, {
            "type": "subscribed",
            "session_id": session_id,
        })
        logger.info(
            "Operator subscribed to session",
            extra={
                "operator_id": operator.operator_id,
                "session_id": session_id,
            },
        )

    async def _handle_unsubscribe(self, operator: OperatorSession, data: dict) -> None:
        """Unsubscribe from current customer session."""
        old_id = operator.subscribed_session_id
        operator.subscribed_session_id = None
        await self._send_json(operator.websocket, {
            "type": "unsubscribed",
            "previous_session_id": old_id,
        })

    async def _handle_operator_message(self, operator: OperatorSession, data: dict) -> None:
        """Send operator override message to a customer session.

        If learn_skill_id is provided, also triggers learn_from_human
        on the training engine to record the Q&A pair.
        """
        text = data.get("text", "").strip()
        if not text:
            await self._send_error(operator.websocket, "Empty message")
            return

        session_id = operator.subscribed_session_id
        if not session_id or session_id not in self._customer_manager._sessions:
            await self._send_error(
                operator.websocket,
                "Not subscribed to an active session",
            )
            return

        customer_session = self._customer_manager._sessions[session_id]

        # Send as operator_override to the customer
        try:
            if customer_session.websocket.client_state == WebSocketState.CONNECTED:
                await customer_session.websocket.send_json({
                    "type": "operator_override",
                    "text": text,
                    "operator_id": operator.operator_id,
                    "timestamp": time.time(),
                })
                await self._send_json(operator.websocket, {
                    "type": "message_sent",
                    "text": text,
                    "session_id": session_id,
                })
                logger.info(
                    "Operator message sent to customer",
                    extra={
                        "operator_id": operator.operator_id,
                        "session_id": session_id,
                        "text_length": len(text),
                    },
                )

                # Record supervisor action
                if self._supervisor is not None:
                    try:
                        response_time_ms = None
                        if session_id in self._escalation_times:
                            response_time_ms = int(
                                (time.time() - self._escalation_times.pop(session_id)) * 1000
                            )
                        await self._supervisor.record_operator_action(
                            operator_id=operator.operator_id,
                            action_type="response",
                            session_id=session_id,
                            details={
                                "text_length": len(text),
                                "response_time_ms": response_time_ms,
                            },
                        )
                    except Exception as sup_exc:
                        logger.warning(f"Failed to record supervisor action: {sup_exc}")

                # Synthesize operator text through cloned voice (fire-and-forget)
                if data.get("use_voice", True) and customer_session.ai_paused:
                    try:
                        from src.agent.operator_takeover import synthesize_operator_message
                        import asyncio

                        avatar = getattr(customer_session.config, "avatar", None)
                        asyncio.create_task(synthesize_operator_message(
                            text=text,
                            pipeline=self._pipeline,
                            avatar=avatar,
                            session_id=session_id,
                            employee_id=getattr(customer_session.config, "avatar_id", ""),
                            customer_ws=customer_session.websocket,
                            app=None,
                        ))
                    except Exception as voice_exc:
                        logger.debug(f"Voice synthesis skipped: {voice_exc}")

                # If learn_skill_id is provided, record as training Q&A
                learn_skill_id = data.get("learn_skill_id", "")
                question = data.get("original_question", "")
                if learn_skill_id and question and self._pipeline:
                    training = getattr(self._pipeline, "_training", None)
                    if training and training.is_loaded:
                        try:
                            await training.learn_from_human(
                                avatar_id=customer_session.config.avatar_id,
                                skill_id=learn_skill_id,
                                question=question,
                                human_answer=text,
                                ai_answer=data.get("ai_answer", ""),
                                quality=data.get("quality", "good"),
                            )
                            await self._send_json(operator.websocket, {
                                "type": "learn_recorded",
                                "skill_id": learn_skill_id,
                                "question": question,
                            })
                        except Exception as learn_exc:
                            logger.warning(f"Failed to record training: {learn_exc}")

        except Exception as exc:
            await self._send_error(
                operator.websocket,
                "Failed to send message",
                str(exc),
            )

    async def _handle_training_feedback(self, operator: OperatorSession, data: dict) -> None:
        """Record training feedback for an AI response."""
        feedback = {
            "message_id": data.get("message_id", ""),
            "quality": data.get("quality", ""),
            "note": data.get("note", ""),
            "operator_id": operator.operator_id,
            "session_id": operator.subscribed_session_id,
            "timestamp": time.time(),
        }
        self._training_log.append(feedback)

        # Record supervisor action for training feedback
        if self._supervisor is not None:
            try:
                await self._supervisor.record_operator_action(
                    operator_id=operator.operator_id,
                    action_type="correction" if feedback["quality"] == "bad" else "feedback",
                    session_id=operator.subscribed_session_id or "",
                    details={
                        "message_id": feedback["message_id"],
                        "quality": feedback["quality"],
                        "note": feedback["note"],
                    },
                )
            except Exception as sup_exc:
                logger.warning(f"Failed to record supervisor feedback action: {sup_exc}")

        # Also save to a file for persistence
        log_path = self._storage_dir / "training_feedback.jsonl"
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(feedback, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning(f"Failed to save training feedback: {exc}")

        await self._send_json(operator.websocket, {
            "type": "feedback_recorded",
            "message_id": feedback["message_id"],
        })

        logger.info(
            "Training feedback recorded",
            extra={
                "operator_id": operator.operator_id,
                "quality": feedback["quality"],
            },
        )

    async def _handle_refresh_sessions(self, operator: OperatorSession, data: dict) -> None:
        """Refresh the session list for the operator."""
        await self._send_session_list(operator)

    async def _handle_ping(self, operator: OperatorSession, data: dict) -> None:
        """Respond with pong."""
        await self._send_json(operator.websocket, {
            "type": "pong",
            "timestamp": time.time(),
        })

    async def _send_session_list(self, operator: OperatorSession) -> None:
        """Send the list of active customer sessions."""
        sessions = self.get_customer_sessions()
        await self._send_json(operator.websocket, {
            "type": "session_list",
            "sessions": sessions,
            "count": len(sessions),
        })

    # ═══════════════════════════════════════════════════════════════════
    # Relay Methods (called by customer WebSocket handlers)
    # ═══════════════════════════════════════════════════════════════════

    async def relay_video_frame(self, session_id: str, frame_b64: str) -> None:
        """Relay a customer video frame to subscribed operators."""
        for operator in self._operators.values():
            if operator.subscribed_session_id == session_id and operator.authenticated:
                try:
                    await self._send_json(operator.websocket, {
                        "type": "customer_video_frame",
                        "session_id": session_id,
                        "frame": frame_b64,
                        "timestamp": time.time(),
                    })
                except Exception:
                    pass

    async def relay_chat_message(
        self,
        session_id: str,
        role: str,
        text: str,
        extras: Optional[dict] = None,
    ) -> None:
        """Relay a customer chat message to subscribed operators."""
        msg = {
            "type": "customer_message",
            "session_id": session_id,
            "role": role,
            "text": text,
            "timestamp": time.time(),
        }
        if extras:
            msg.update(extras)

        for operator in self._operators.values():
            if operator.subscribed_session_id == session_id and operator.authenticated:
                try:
                    await self._send_json(operator.websocket, msg)
                except Exception:
                    pass

    async def relay_escalation(
        self,
        session_id: str,
        question: str,
        reason: str = "low_confidence",
        escalation_id: Optional[str] = None,
        confidence: float = 0.0,
    ) -> None:
        """Relay an escalation alert to subscribed operators.

        Args:
            session_id: Customer session ID.
            question: The customer's question that triggered escalation.
            reason: Why the escalation was triggered.
            escalation_id: Optional escalation event ID.
            confidence: KB confidence score.
        """
        now = time.time()
        self._escalation_times[session_id] = now

        msg = {
            "type": "escalation_alert",
            "session_id": session_id,
            "question": question,
            "reason": reason,
            "escalation_id": escalation_id,
            "confidence": confidence,
            "timestamp": now,
        }

        for operator in self._operators.values():
            if operator.authenticated:
                # Send to all authenticated operators (not just subscribed)
                try:
                    await self._send_json(operator.websocket, msg)
                except Exception:
                    pass

        logger.info(
            "Escalation relayed to operators",
            extra={
                "session_id": session_id,
                "reason": reason,
                "escalation_id": escalation_id,
            },
        )

    async def relay_document(
        self,
        session_id: str,
        image_data: bytes,
        filename: str,
    ) -> str:
        """Save a scanned document and relay to subscribed operators.

        Args:
            session_id: Customer session ID.
            image_data: Raw image bytes (JPEG/PNG).
            filename: Original filename.

        Returns:
            URL path to the saved document.
        """
        # Save to documents directory
        safe_name = f"{session_id}_{uuid.uuid4().hex[:8]}_{filename}"
        doc_path = self._docs_dir / safe_name
        doc_path.write_bytes(image_data)

        url = f"/files/documents/{safe_name}"

        logger.info(
            "Document saved",
            extra={
                "session_id": session_id,
                "filename": safe_name,
                "size_bytes": len(image_data),
            },
        )

        # Relay to subscribed operators
        for operator in self._operators.values():
            if operator.subscribed_session_id == session_id and operator.authenticated:
                try:
                    await self._send_json(operator.websocket, {
                        "type": "document_received",
                        "session_id": session_id,
                        "image_url": url,
                        "filename": filename,
                        "size_bytes": len(image_data),
                        "timestamp": time.time(),
                    })
                except Exception:
                    pass

        return url

    async def notify_session_ended(self, session_id: str) -> None:
        """Notify operators that a customer session has ended."""
        for operator in self._operators.values():
            if operator.subscribed_session_id == session_id and operator.authenticated:
                try:
                    await self._send_json(operator.websocket, {
                        "type": "session_ended",
                        "session_id": session_id,
                        "timestamp": time.time(),
                    })
                    operator.subscribed_session_id = None
                except Exception:
                    pass

    # ═══════════════════════════════════════════════════════════════════
    # Send Helpers
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    async def _send_json(websocket: WebSocket, data: dict[str, Any]) -> None:
        """Send a JSON message to a WebSocket client."""
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_json(data)
        except Exception as exc:
            logger.debug(f"Operator WebSocket send failed: {exc}")

    @staticmethod
    async def _send_error(
        websocket: WebSocket,
        error: str,
        detail: Optional[str] = None,
    ) -> None:
        """Send an error message to a WebSocket client."""
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                msg: dict[str, Any] = {"type": "error", "error": error}
                if detail:
                    msg["detail"] = detail
                await websocket.send_json(msg)
        except Exception as exc:
            logger.debug(f"Operator error send failed: {exc}")


# ═══════════════════════════════════════════════════════════════════════
# WebSocket Endpoint
# ═══════════════════════════════════════════════════════════════════════


async def operator_websocket_endpoint(
    websocket: WebSocket,
    manager: OperatorWebSocketManager,
) -> None:
    """FastAPI WebSocket endpoint for operator dashboard.

    Args:
        websocket: The incoming WebSocket connection.
        manager: The OperatorWebSocketManager instance.
    """
    operator: Optional[OperatorSession] = None
    try:
        operator = await manager.connect(websocket)
        await manager.handle_session(operator)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.error(f"Operator WebSocket error: {exc}")
    finally:
        if operator:
            await manager.disconnect(operator)
