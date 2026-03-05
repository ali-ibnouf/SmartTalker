"""Node Manager for GPU Render Node registration and monitoring.

Handles:
- License key validation on node registration
- WebSocket-based heartbeat monitoring
- Session tracking per node with concurrent limit enforcement
- Message relay to connected GPU nodes
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from src.utils.logger import setup_logger

logger = setup_logger("pipeline.node_manager")


@dataclass
class RenderNodeInfo:
    """Status of a registered GPU Render Node."""

    node_id: str
    hostname: str
    gpu_type: str = ""
    vram_mb: int = 0
    status: str = "offline"  # online, offline, busy
    current_fps: float = 0.0
    last_heartbeat: float = 0.0
    registered_at: float = 0.0
    license_key: str = ""
    customer_id: str = ""
    active_sessions: int = 0
    max_concurrent: int = 1
    gpu_usage: int = 0
    vram_used: int = 0
    uptime_hours: float = 0.0
    avatars_loaded: list[str] = field(default_factory=list)


@dataclass
class NodeSession:
    """Tracks an active session on a render node."""

    session_id: str
    node_id: str
    avatar_id: str
    started_at: float = 0.0


class NodeManager:
    """Manages GPU Render Node registration and health monitoring.

    Nodes connect via WebSocket to /ws/node and send periodic heartbeats.
    Central tracks node state, validates licenses, and can assign/relay
    sessions to available nodes.
    """

    def __init__(self, db: Any = None) -> None:
        self._nodes: dict[str, RenderNodeInfo] = {}
        self._websockets: dict[str, Any] = {}  # node_id -> WebSocket
        self._sessions: dict[str, NodeSession] = {}  # session_id -> NodeSession
        self._node_sessions: dict[str, set[str]] = {}  # node_id -> set of session_ids
        self._db = db
        logger.info("NodeManager initialized")

    async def register_node(
        self,
        websocket: Any,
        node_id: str,
        hostname: str,
        gpu_type: str = "",
        vram_mb: int = 0,
        license_key: str = "",
        customer_id: str = "",
        max_concurrent: int = 1,
        avatars_loaded: Optional[list[str]] = None,
    ) -> RenderNodeInfo:
        """Register a new render node via WebSocket."""
        now = time.time()
        node = RenderNodeInfo(
            node_id=node_id,
            hostname=hostname,
            gpu_type=gpu_type,
            vram_mb=vram_mb,
            status="online",
            last_heartbeat=now,
            registered_at=now,
            license_key=license_key,
            customer_id=customer_id,
            max_concurrent=max_concurrent,
            avatars_loaded=avatars_loaded or [],
        )
        self._nodes[node_id] = node
        self._websockets[node_id] = websocket
        self._node_sessions.setdefault(node_id, set())

        logger.info(
            "Render node registered",
            extra={
                "node_id": node_id,
                "hostname": hostname,
                "gpu": gpu_type,
                "customer_id": customer_id,
            },
        )
        return node

    async def deregister_node(self, node_id: str) -> None:
        """Remove a render node on disconnect."""
        # Clean up sessions on this node
        session_ids = list(self._node_sessions.get(node_id, set()))
        for sid in session_ids:
            self._sessions.pop(sid, None)
        self._node_sessions.pop(node_id, None)

        self._nodes.pop(node_id, None)
        self._websockets.pop(node_id, None)
        logger.info("Render node deregistered", extra={"node_id": node_id})

    async def heartbeat(
        self,
        node_id: str,
        fps: float = 0.0,
        status: str = "online",
        gpu_usage: int = 0,
        vram_used: int = 0,
        active_sessions: int = 0,
        uptime_hours: float = 0.0,
    ) -> Optional[RenderNodeInfo]:
        """Update node health from heartbeat."""
        node = self._nodes.get(node_id)
        if node is None:
            return None

        node.last_heartbeat = time.time()
        node.current_fps = fps
        node.status = status
        node.gpu_usage = gpu_usage
        node.vram_used = vram_used
        node.active_sessions = active_sessions
        node.uptime_hours = uptime_hours
        return node

    def list_nodes(self) -> list[RenderNodeInfo]:
        """List all registered render nodes."""
        return list(self._nodes.values())

    def get_node(self, node_id: str) -> Optional[RenderNodeInfo]:
        """Get a specific node's info."""
        return self._nodes.get(node_id)

    def get_available_node(self) -> Optional[RenderNodeInfo]:
        """Find the best available node for a new session.

        Prefers nodes with highest FPS and 'online' status.
        """
        available = [n for n in self._nodes.values() if n.status == "online"]
        if not available:
            return None
        return max(available, key=lambda n: n.current_fps)

    def get_nodes_for_customer(self, customer_id: str) -> list[RenderNodeInfo]:
        """Get all nodes belonging to a specific customer."""
        return [n for n in self._nodes.values() if n.customer_id == customer_id]

    # ═════════════════════════════════════════════════════════════════════
    # License Validation
    # ═════════════════════════════════════════════════════════════════════

    async def validate_license(self, license_key: str) -> tuple[bool, dict]:
        """Validate a GPU Node's license key against the database.

        Args:
            license_key: License key from the node_register message.

        Returns:
            Tuple of (is_valid, plan_info). plan_info contains:
            plan, seconds_remaining, max_concurrent, customer_id.
        """
        if self._db is None:
            # No DB = accept all (dev mode)
            return True, {
                "plan": "professional",
                "seconds_remaining": 100_000,
                "max_concurrent": 3,
                "customer_id": "dev",
            }

        from sqlalchemy import select
        from src.db.models import Customer, Subscription

        async with self._db.session() as session:
            # Find customer by API key (license_key maps to api_key)
            result = await session.execute(
                select(Customer).where(
                    Customer.api_key == license_key,
                    Customer.is_active == True,  # noqa: E712
                    Customer.suspended == False,  # noqa: E712
                )
            )
            customer = result.scalar_one_or_none()
            if customer is None:
                return False, {}

            # Get active subscription
            sub_result = await session.execute(
                select(Subscription)
                .where(
                    Subscription.customer_id == customer.id,
                    Subscription.is_active == True,  # noqa: E712
                )
                .order_by(Subscription.created_at.desc())
                .limit(1)
            )
            subscription = sub_result.scalar_one_or_none()
            if subscription is None:
                return False, {}

            return True, {
                "plan": subscription.plan,
                "seconds_remaining": subscription.monthly_seconds,
                "max_concurrent": subscription.max_concurrent_sessions,
                "customer_id": customer.id,
            }

    # ═════════════════════════════════════════════════════════════════════
    # Session Tracking
    # ═════════════════════════════════════════════════════════════════════

    def start_session(
        self,
        node_id: str,
        session_id: str,
        avatar_id: str = "",
    ) -> Optional[NodeSession]:
        """Track a new session on a node, enforcing concurrent limits.

        Returns:
            NodeSession if allowed, None if concurrent limit exceeded.
        """
        node = self._nodes.get(node_id)
        if node is None:
            return None

        # Check concurrent limit
        current = len(self._node_sessions.get(node_id, set()))
        if current >= node.max_concurrent:
            logger.warning(
                "Concurrent session limit reached",
                extra={"node_id": node_id, "limit": node.max_concurrent, "current": current},
            )
            return None

        ns = NodeSession(
            session_id=session_id,
            node_id=node_id,
            avatar_id=avatar_id,
            started_at=time.time(),
        )
        self._sessions[session_id] = ns
        self._node_sessions.setdefault(node_id, set()).add(session_id)
        node.active_sessions = len(self._node_sessions[node_id])

        return ns

    def end_session(self, session_id: str) -> None:
        """Remove a session from tracking."""
        ns = self._sessions.pop(session_id, None)
        if ns is None:
            return

        node_sessions = self._node_sessions.get(ns.node_id, set())
        node_sessions.discard(session_id)

        node = self._nodes.get(ns.node_id)
        if node is not None:
            node.active_sessions = len(node_sessions)

    def get_session(self, session_id: str) -> Optional[NodeSession]:
        """Get a tracked session."""
        return self._sessions.get(session_id)

    # ═════════════════════════════════════════════════════════════════════
    # Message Relay
    # ═════════════════════════════════════════════════════════════════════

    def get_node_websocket(self, node_id: str) -> Any:
        """Get the WebSocket for a specific node."""
        return self._websockets.get(node_id)

    async def send_to_node(self, node_id: str, message: dict) -> bool:
        """Send a JSON message to a connected GPU node.

        Args:
            node_id: Target node ID.
            message: Dict to serialize and send.

        Returns:
            True if sent successfully, False otherwise.
        """
        ws = self._websockets.get(node_id)
        if ws is None:
            logger.warning("Cannot send to node — not connected", extra={"node_id": node_id})
            return False

        try:
            await ws.send_text(json.dumps(message))
            return True
        except Exception as exc:
            logger.warning(f"Failed to send to node {node_id}: {exc}")
            return False
