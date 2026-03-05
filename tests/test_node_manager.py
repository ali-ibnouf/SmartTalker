"""Tests for NodeManager."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.pipeline.node_manager import NodeManager, NodeSession, RenderNodeInfo


class TestNodeManagerInit:
    """Test NodeManager initialization."""

    def test_init(self):
        nm = NodeManager()
        assert nm.list_nodes() == []
        assert nm.get_node("unknown") is None


class TestNodeRegistration:
    """Test node registration and deregistration."""

    @pytest.mark.asyncio
    async def test_register_node(self):
        nm = NodeManager()
        ws = MagicMock()
        node = await nm.register_node(
            websocket=ws,
            node_id="node-1",
            hostname="gpu-server-1",
            gpu_type="RTX 4090",
            vram_mb=24576,
        )
        assert isinstance(node, RenderNodeInfo)
        assert node.node_id == "node-1"
        assert node.hostname == "gpu-server-1"
        assert node.gpu_type == "RTX 4090"
        assert node.vram_mb == 24576
        assert node.status == "online"
        assert node.registered_at > 0

    @pytest.mark.asyncio
    async def test_register_multiple_nodes(self):
        nm = NodeManager()
        await nm.register_node(MagicMock(), "n1", "host1")
        await nm.register_node(MagicMock(), "n2", "host2")
        nodes = nm.list_nodes()
        assert len(nodes) == 2

    @pytest.mark.asyncio
    async def test_deregister_node(self):
        nm = NodeManager()
        await nm.register_node(MagicMock(), "n1", "host1")
        assert len(nm.list_nodes()) == 1

        await nm.deregister_node("n1")
        assert len(nm.list_nodes()) == 0

    @pytest.mark.asyncio
    async def test_deregister_nonexistent(self):
        nm = NodeManager()
        # Should not raise
        await nm.deregister_node("nope")


class TestNodeHeartbeat:
    """Test heartbeat updates."""

    @pytest.mark.asyncio
    async def test_heartbeat_updates(self):
        nm = NodeManager()
        await nm.register_node(MagicMock(), "n1", "host1")

        result = await nm.heartbeat("n1", fps=30.0, status="busy")
        assert result is not None
        assert result.current_fps == 30.0
        assert result.status == "busy"
        assert result.last_heartbeat > 0

    @pytest.mark.asyncio
    async def test_heartbeat_unknown_node(self):
        nm = NodeManager()
        result = await nm.heartbeat("unknown", fps=10.0)
        assert result is None


class TestNodeLookup:
    """Test node lookup operations."""

    @pytest.mark.asyncio
    async def test_get_node(self):
        nm = NodeManager()
        await nm.register_node(MagicMock(), "n1", "host1", gpu_type="A100")
        node = nm.get_node("n1")
        assert node is not None
        assert node.gpu_type == "A100"

    @pytest.mark.asyncio
    async def test_get_node_not_found(self):
        nm = NodeManager()
        assert nm.get_node("nope") is None

    @pytest.mark.asyncio
    async def test_get_available_node_none(self):
        nm = NodeManager()
        assert nm.get_available_node() is None

    @pytest.mark.asyncio
    async def test_get_available_node_prefers_highest_fps(self):
        nm = NodeManager()
        await nm.register_node(MagicMock(), "n1", "host1")
        await nm.register_node(MagicMock(), "n2", "host2")
        await nm.heartbeat("n1", fps=15.0)
        await nm.heartbeat("n2", fps=30.0)

        best = nm.get_available_node()
        assert best is not None
        assert best.node_id == "n2"

    @pytest.mark.asyncio
    async def test_get_available_node_skips_busy(self):
        nm = NodeManager()
        await nm.register_node(MagicMock(), "n1", "host1")
        await nm.register_node(MagicMock(), "n2", "host2")
        await nm.heartbeat("n1", fps=30.0, status="busy")
        await nm.heartbeat("n2", fps=10.0, status="online")

        best = nm.get_available_node()
        assert best is not None
        assert best.node_id == "n2"


class TestRenderNodeInfo:
    """Test RenderNodeInfo dataclass."""

    def test_defaults(self):
        node = RenderNodeInfo(node_id="n1", hostname="host1")
        assert node.gpu_type == ""
        assert node.vram_mb == 0
        assert node.status == "offline"
        assert node.current_fps == 0.0
        assert node.last_heartbeat == 0.0
        assert node.registered_at == 0.0

    def test_custom_values(self):
        node = RenderNodeInfo(
            node_id="n1",
            hostname="gpu-server",
            gpu_type="RTX 4090",
            vram_mb=24576,
            status="online",
            current_fps=60.0,
        )
        assert node.gpu_type == "RTX 4090"
        assert node.vram_mb == 24576
        assert node.status == "online"
        assert node.current_fps == 60.0

    def test_v3_fields(self):
        node = RenderNodeInfo(
            node_id="n1",
            hostname="host1",
            license_key="lic-123",
            customer_id="cust-1",
            max_concurrent=5,
            avatars_loaded=["av1", "av2"],
        )
        assert node.license_key == "lic-123"
        assert node.customer_id == "cust-1"
        assert node.max_concurrent == 5
        assert node.avatars_loaded == ["av1", "av2"]


class TestLicenseValidation:
    """Test license key validation."""

    @pytest.mark.asyncio
    async def test_validate_no_db_accepts_all(self):
        """No DB = dev mode, accepts any license."""
        nm = NodeManager(db=None)
        valid, info = await nm.validate_license("any-key")
        assert valid is True
        assert info["plan"] == "professional"
        assert "customer_id" in info

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("sqlalchemy"),
        reason="sqlalchemy not installed",
    )
    async def test_validate_with_db_not_found(self):
        """Invalid license key returns False."""
        mock_db = MagicMock()
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_db.session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_db.session.return_value.__aexit__ = AsyncMock(return_value=False)

        nm = NodeManager(db=mock_db)
        valid, info = await nm.validate_license("bad-key")
        assert valid is False
        assert info == {}


class TestNodeSessionTracking:
    """Test session tracking on nodes."""

    @pytest.mark.asyncio
    async def test_start_session(self):
        nm = NodeManager()
        await nm.register_node(MagicMock(), "n1", "host1", max_concurrent=3)
        ns = nm.start_session("n1", "s1", "avatar1")
        assert isinstance(ns, NodeSession)
        assert ns.session_id == "s1"
        assert ns.node_id == "n1"
        assert ns.avatar_id == "avatar1"
        assert ns.started_at > 0

    @pytest.mark.asyncio
    async def test_start_session_unknown_node(self):
        nm = NodeManager()
        ns = nm.start_session("unknown", "s1")
        assert ns is None

    @pytest.mark.asyncio
    async def test_concurrent_limit_enforced(self):
        nm = NodeManager()
        await nm.register_node(MagicMock(), "n1", "host1", max_concurrent=2)
        assert nm.start_session("n1", "s1") is not None
        assert nm.start_session("n1", "s2") is not None
        # Third should be rejected
        assert nm.start_session("n1", "s3") is None

    @pytest.mark.asyncio
    async def test_end_session_frees_slot(self):
        nm = NodeManager()
        await nm.register_node(MagicMock(), "n1", "host1", max_concurrent=1)
        nm.start_session("n1", "s1")
        # At limit
        assert nm.start_session("n1", "s2") is None
        # Free the slot
        nm.end_session("s1")
        # Now allowed
        assert nm.start_session("n1", "s2") is not None

    @pytest.mark.asyncio
    async def test_get_session(self):
        nm = NodeManager()
        await nm.register_node(MagicMock(), "n1", "host1")
        nm.start_session("n1", "s1", "av1")
        ns = nm.get_session("s1")
        assert ns is not None
        assert ns.avatar_id == "av1"

    @pytest.mark.asyncio
    async def test_get_session_not_found(self):
        nm = NodeManager()
        assert nm.get_session("nope") is None

    @pytest.mark.asyncio
    async def test_active_session_count_updates(self):
        nm = NodeManager()
        await nm.register_node(MagicMock(), "n1", "host1", max_concurrent=5)
        nm.start_session("n1", "s1")
        nm.start_session("n1", "s2")
        node = nm.get_node("n1")
        assert node.active_sessions == 2
        nm.end_session("s1")
        assert node.active_sessions == 1

    @pytest.mark.asyncio
    async def test_deregister_cleans_sessions(self):
        nm = NodeManager()
        await nm.register_node(MagicMock(), "n1", "host1", max_concurrent=5)
        nm.start_session("n1", "s1")
        nm.start_session("n1", "s2")
        await nm.deregister_node("n1")
        assert nm.get_session("s1") is None
        assert nm.get_session("s2") is None


class TestGetNodesForCustomer:
    """Test customer-filtered node listing."""

    @pytest.mark.asyncio
    async def test_returns_matching_nodes(self):
        nm = NodeManager()
        await nm.register_node(MagicMock(), "n1", "host1", customer_id="c1")
        await nm.register_node(MagicMock(), "n2", "host2", customer_id="c1")
        await nm.register_node(MagicMock(), "n3", "host3", customer_id="c2")
        nodes = nm.get_nodes_for_customer("c1")
        assert len(nodes) == 2
        assert all(n.customer_id == "c1" for n in nodes)

    @pytest.mark.asyncio
    async def test_returns_empty_for_unknown(self):
        nm = NodeManager()
        assert nm.get_nodes_for_customer("unknown") == []


class TestSendToNode:
    """Test message relay to nodes."""

    @pytest.mark.asyncio
    async def test_send_to_connected_node(self):
        nm = NodeManager()
        ws = AsyncMock()
        await nm.register_node(ws, "n1", "host1")
        result = await nm.send_to_node("n1", {"type": "test"})
        assert result is True
        ws.send_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_to_disconnected_node(self):
        nm = NodeManager()
        result = await nm.send_to_node("unknown", {"type": "test"})
        assert result is False

    @pytest.mark.asyncio
    async def test_send_failure_returns_false(self):
        nm = NodeManager()
        ws = AsyncMock()
        ws.send_text.side_effect = Exception("connection closed")
        await nm.register_node(ws, "n1", "host1")
        result = await nm.send_to_node("n1", {"type": "test"})
        assert result is False
