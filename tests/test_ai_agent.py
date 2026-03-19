"""Tests for AI Optimization Agent service."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from src.services.ai_agent.config import AgentSettings
from src.services.ai_agent.rules import AgentContext, Detection, RuleRegistry
from src.services.ai_agent.schemas import (
    AgentStatsResponse,
    DetectionItem,
    IncidentActionResponse,
    IncidentItem,
    IncidentListResponse,
    PredictionItem,
    PredictionListResponse,
    ScanResponse,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def agent_config():
    return AgentSettings(
        agent_enabled=True,
        auto_fix_enabled=True,
        scan_interval_s=5,
        cpu_warn_pct=80.0,
        memory_warn_pct=80.0,
        disk_warn_pct=90.0,
        fps_min=20.0,
        vram_warn_pct=90.0,
        churn_days_inactive=7,
        quota_warn_pct=90.0,
        escalation_rate_warn=0.2,
        violation_spike_24h=10,
        rapid_session_threshold=50,
    )


@pytest.fixture
def agent_ctx(config, agent_config, mock_pipeline):
    return AgentContext(
        db=None,
        redis=None,
        pipeline=mock_pipeline,
        config=config,
        agent_config=agent_config,
    )


@pytest.fixture
def mock_db():
    """Mock Database with session_ctx() async context manager."""
    db = MagicMock()
    session = AsyncMock()

    @asynccontextmanager
    async def _session_ctx():
        yield session

    db.session_ctx = _session_ctx
    return db, session


@pytest.fixture
def db_agent_ctx(config, agent_config, mock_pipeline, mock_db):
    """AgentContext with a mock database for DB-dependent rule tests."""
    db, _ = mock_db
    return AgentContext(
        db=db,
        redis=None,
        pipeline=mock_pipeline,
        config=config,
        agent_config=agent_config,
    )


@pytest.fixture
def mock_operator_manager():
    """Mock OperatorWebSocketManager for notification tests."""
    mgr = MagicMock()
    mgr._operators = {}
    mgr._send_json = AsyncMock()
    return mgr


# ── Detection Tests ───────────────────────────────────────────────────────


class TestDetection:
    """Tests for Detection dataclass."""

    def test_detection_creation(self):
        d = Detection(
            rule_id="test.rule",
            severity="warning",
            title="Test Issue",
            description="Something happened",
        )
        assert d.rule_id == "test.rule"
        assert d.severity == "warning"
        assert d.auto_fixable is False
        assert d.details == {}

    def test_detection_with_details(self):
        d = Detection(
            rule_id="test.rule",
            severity="critical",
            title="Critical Issue",
            description="Big problem",
            details={"node_id": "n1", "fps": 5.0},
            recommendation="Fix it",
            auto_fixable=True,
        )
        assert d.details["node_id"] == "n1"
        assert d.auto_fixable is True

    def test_severity_values(self):
        for sev in ("info", "warning", "critical"):
            d = Detection(
                rule_id="test",
                severity=sev,
                title="t",
                description="d",
            )
            assert d.severity == sev


# ── RuleRegistry Tests ────────────────────────────────────────────────────


class TestRuleRegistry:
    """Tests for RuleRegistry."""

    def test_register_and_list(self, agent_ctx):
        registry = RuleRegistry()
        mock_rule = MagicMock()
        mock_rule.rule_id = "test.mock"
        registry.register(mock_rule)
        assert len(registry.rules) == 1
        assert registry.rules[0].rule_id == "test.mock"

    @pytest.mark.asyncio
    async def test_run_all_empty(self, agent_ctx):
        registry = RuleRegistry()
        detections = await registry.run_all(agent_ctx)
        assert detections == []

    @pytest.mark.asyncio
    async def test_run_all_with_detections(self, agent_ctx):
        registry = RuleRegistry()

        mock_rule = MagicMock()
        mock_rule.rule_id = "test.rule"
        mock_rule.evaluate = AsyncMock(return_value=[
            Detection(rule_id="test.rule", severity="info", title="Found", description="x"),
        ])
        registry.register(mock_rule)

        detections = await registry.run_all(agent_ctx)
        assert len(detections) == 1
        assert detections[0].title == "Found"

    @pytest.mark.asyncio
    async def test_run_all_skips_failing_rule(self, agent_ctx):
        registry = RuleRegistry()

        # Failing rule
        bad_rule = MagicMock()
        bad_rule.rule_id = "test.bad"
        bad_rule.evaluate = AsyncMock(side_effect=RuntimeError("boom"))
        registry.register(bad_rule)

        # Good rule
        good_rule = MagicMock()
        good_rule.rule_id = "test.good"
        good_rule.evaluate = AsyncMock(return_value=[
            Detection(rule_id="test.good", severity="info", title="OK", description=""),
        ])
        registry.register(good_rule)

        detections = await registry.run_all(agent_ctx)
        assert len(detections) == 1
        assert detections[0].rule_id == "test.good"

    @pytest.mark.asyncio
    async def test_run_all_concurrent(self, agent_ctx):
        """Multiple rules run concurrently."""
        registry = RuleRegistry()
        for i in range(5):
            rule = MagicMock()
            rule.rule_id = f"test.rule_{i}"
            rule.evaluate = AsyncMock(return_value=[
                Detection(rule_id=f"test.rule_{i}", severity="info", title=f"R{i}", description=""),
            ])
            registry.register(rule)

        detections = await registry.run_all(agent_ctx)
        assert len(detections) == 5


# ── AgentSettings Tests ──────────────────────────────────────────────────


class TestAgentSettings:
    """Tests for AgentSettings defaults."""

    def test_defaults(self):
        s = AgentSettings()
        assert s.agent_enabled is True
        assert s.auto_fix_enabled is True
        assert s.scan_interval_s == 60
        assert s.cpu_warn_pct == 85.0
        assert s.fps_min == 20.0

    def test_custom_values(self):
        s = AgentSettings(scan_interval_s=10, cpu_warn_pct=50.0)
        assert s.scan_interval_s == 10
        assert s.cpu_warn_pct == 50.0


# ── System Monitor Rules ─────────────────────────────────────────────────


class TestSystemRules:
    """Tests for system monitoring rules.

    psutil is imported locally inside evaluate(), so we inject a mock
    via sys.modules instead of patching a module-level attribute.
    """

    def _make_psutil_mock(self, **overrides):
        """Create a mock psutil module with configurable returns."""
        mock_psutil = MagicMock()
        mock_psutil.cpu_percent.return_value = overrides.get("cpu_pct", 20.0)
        mem = MagicMock()
        mem.percent = overrides.get("mem_pct", 50.0)
        mem.used = overrides.get("mem_used", 8 * 1024**3)
        mem.total = overrides.get("mem_total", 16 * 1024**3)
        mock_psutil.virtual_memory.return_value = mem
        disk = MagicMock()
        disk.percent = overrides.get("disk_pct", 50.0)
        disk.free = overrides.get("disk_free", 100 * 1024**3)
        disk.total = overrides.get("disk_total", 200 * 1024**3)
        mock_psutil.disk_usage.return_value = disk
        return mock_psutil

    @pytest.mark.asyncio
    async def test_high_cpu_triggers(self, agent_ctx):
        from src.services.ai_agent.monitors.system import HighCPURule

        rule = HighCPURule()
        mock_psutil = self._make_psutil_mock(cpu_pct=92.0)
        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            detections = await rule.evaluate(agent_ctx)
            assert len(detections) == 1
            assert detections[0].severity == "warning"
            assert "92%" in detections[0].title

    @pytest.mark.asyncio
    async def test_high_cpu_critical(self, agent_ctx):
        from src.services.ai_agent.monitors.system import HighCPURule

        rule = HighCPURule()
        mock_psutil = self._make_psutil_mock(cpu_pct=97.0)
        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            detections = await rule.evaluate(agent_ctx)
            assert len(detections) == 1
            assert detections[0].severity == "critical"

    @pytest.mark.asyncio
    async def test_high_cpu_ok(self, agent_ctx):
        from src.services.ai_agent.monitors.system import HighCPURule

        rule = HighCPURule()
        mock_psutil = self._make_psutil_mock(cpu_pct=40.0)
        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            detections = await rule.evaluate(agent_ctx)
            assert len(detections) == 0

    @pytest.mark.asyncio
    async def test_high_cpu_no_psutil(self, agent_ctx):
        from src.services.ai_agent.monitors.system import HighCPURule

        rule = HighCPURule()
        with patch.dict("sys.modules", {"psutil": None}):
            detections = await rule.evaluate(agent_ctx)
            assert len(detections) == 0

    @pytest.mark.asyncio
    async def test_high_memory_triggers(self, agent_ctx):
        from src.services.ai_agent.monitors.system import HighMemoryRule

        rule = HighMemoryRule()
        mock_psutil = self._make_psutil_mock(
            mem_pct=88.0, mem_used=14 * 1024**3, mem_total=16 * 1024**3,
        )
        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            detections = await rule.evaluate(agent_ctx)
            assert len(detections) == 1
            assert detections[0].auto_fixable is True

    @pytest.mark.asyncio
    async def test_disk_space_ok(self, agent_ctx):
        from src.services.ai_agent.monitors.system import DiskSpaceRule

        rule = DiskSpaceRule()
        mock_psutil = self._make_psutil_mock(
            disk_pct=50.0, disk_free=100 * 1024**3, disk_total=200 * 1024**3,
        )
        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            detections = await rule.evaluate(agent_ctx)
            assert len(detections) == 0


# ── GPU Monitor Rules (removed — GPU rendering now via RunPod Serverless) ─


class TestGPURules:
    """Tests for GPU render node monitoring rules (deprecated — RunPod Serverless).

    These rules still exist in code but are no longer registered in the agent.
    Tests provide a mock node_manager since the default is None.
    """

    @dataclass
    class FakeNode:
        node_id: str = "node-1"
        hostname: str = "gpu-node-001"
        gpu_type: str = "RTX 4090"
        vram_mb: int = 24576
        vram_used: int = 0
        status: str = "online"
        current_fps: float = 30.0
        last_heartbeat: float = 0.0
        active_sessions: int = 2
        max_concurrent: int = 4
        customer_id: str = "cust-1"

    @pytest.mark.asyncio
    async def test_fps_drop_triggers(self, agent_ctx):
        from src.services.ai_agent.monitors.gpu import NodeFPSDropRule
        import time

        rule = NodeFPSDropRule()
        node = self.FakeNode(current_fps=12.0)
        agent_ctx.node_manager = MagicMock()
        agent_ctx.node_manager.list_nodes.return_value = [node]

        detections = await rule.evaluate(agent_ctx)
        assert len(detections) == 1
        assert "12" in detections[0].title
        assert detections[0].auto_fixable is True

    @pytest.mark.asyncio
    async def test_fps_ok(self, agent_ctx):
        from src.services.ai_agent.monitors.gpu import NodeFPSDropRule

        rule = NodeFPSDropRule()
        node = self.FakeNode(current_fps=30.0)
        agent_ctx.node_manager = MagicMock()
        agent_ctx.node_manager.list_nodes.return_value = [node]

        detections = await rule.evaluate(agent_ctx)
        assert len(detections) == 0

    @pytest.mark.asyncio
    async def test_vram_high(self, agent_ctx):
        from src.services.ai_agent.monitors.gpu import NodeVRAMRule

        rule = NodeVRAMRule()
        node = self.FakeNode(vram_mb=24576, vram_used=23000)
        agent_ctx.node_manager = MagicMock()
        agent_ctx.node_manager.list_nodes.return_value = [node]

        detections = await rule.evaluate(agent_ctx)
        assert len(detections) == 1
        assert "VRAM" in detections[0].title

    @pytest.mark.asyncio
    async def test_vram_ok(self, agent_ctx):
        from src.services.ai_agent.monitors.gpu import NodeVRAMRule

        rule = NodeVRAMRule()
        node = self.FakeNode(vram_mb=24576, vram_used=5000)
        agent_ctx.node_manager = MagicMock()
        agent_ctx.node_manager.list_nodes.return_value = [node]

        detections = await rule.evaluate(agent_ctx)
        assert len(detections) == 0

    @pytest.mark.asyncio
    async def test_offline_nodes_skipped(self, agent_ctx):
        from src.services.ai_agent.monitors.gpu import NodeFPSDropRule

        rule = NodeFPSDropRule()
        node = self.FakeNode(status="offline", current_fps=0.0)
        agent_ctx.node_manager = MagicMock()
        agent_ctx.node_manager.list_nodes.return_value = [node]

        detections = await rule.evaluate(agent_ctx)
        assert len(detections) == 0


# ── Fix Handlers ──────────────────────────────────────────────────────────


class TestFixHandlers:
    """Tests for auto-fix handlers."""

    @pytest.mark.asyncio
    async def test_session_limit_fix(self, agent_ctx):
        from src.services.ai_agent.fixes.handlers import SessionLimitFix

        fix = SessionLimitFix()
        node = MagicMock()
        node.max_concurrent = 8
        node.active_sessions = 6
        agent_ctx.node_manager = MagicMock()
        agent_ctx.node_manager.get_node.return_value = node

        d = Detection(
            rule_id="gpu.fps_drop",
            severity="warning",
            title="Low FPS",
            description="",
            details={"node_id": "n1", "active_sessions": 6},
            auto_fixable=True,
        )

        assert await fix.can_fix(d, agent_ctx) is True
        result = await fix.apply(d, agent_ctx)
        assert result["action"] == "reduced_max_concurrent"
        assert result["new_max"] == 5

    @pytest.mark.asyncio
    async def test_session_limit_cannot_fix_other_rules(self, agent_ctx):
        from src.services.ai_agent.fixes.handlers import SessionLimitFix

        fix = SessionLimitFix()
        d = Detection(
            rule_id="system.high_cpu",
            severity="warning",
            title="CPU",
            description="",
            details={},
        )
        assert await fix.can_fix(d, agent_ctx) is False

    @pytest.mark.asyncio
    async def test_cache_clear_fix(self, agent_ctx):
        from src.services.ai_agent.fixes.handlers import CacheClearFix

        fix = CacheClearFix()
        mock_redis = AsyncMock()
        mock_redis.scan_iter = MagicMock(return_value=AsyncIteratorMock([]))
        agent_ctx.redis = mock_redis

        d = Detection(
            rule_id="system.high_memory",
            severity="warning",
            title="Memory",
            description="",
        )
        assert await fix.can_fix(d, agent_ctx) is True
        result = await fix.apply(d, agent_ctx)
        assert result["action"] == "cache_cleared"

    @pytest.mark.asyncio
    async def test_quota_warning_fix(self, agent_ctx):
        from src.services.ai_agent.fixes.handlers import QuotaWarningFix

        fix = QuotaWarningFix()
        d = Detection(
            rule_id="business.quota_exhaustion",
            severity="warning",
            title="Quota",
            description="",
            details={"customer_id": "c1", "usage_pct": 95.0, "plan": "starter"},
            auto_fixable=True,
        )
        assert await fix.can_fix(d, agent_ctx) is True
        result = await fix.apply(d, agent_ctx)
        assert result["action"] == "notification_logged"


# ── FixRegistry Tests ─────────────────────────────────────────────────────


class TestFixRegistry:
    """Tests for FixRegistry."""

    @pytest.mark.asyncio
    async def test_try_fix_no_handler(self, agent_ctx):
        from src.services.ai_agent.fixes.handlers import FixRegistry

        registry = FixRegistry()
        d = Detection(
            rule_id="unknown.rule",
            severity="info",
            title="Test",
            description="",
        )
        result = await registry.try_fix(d, agent_ctx)
        assert result is None

    @pytest.mark.asyncio
    async def test_try_fix_with_handler(self, agent_ctx):
        from src.services.ai_agent.fixes.handlers import FixRegistry

        handler = MagicMock()
        handler.fix_id = "test.fix"
        handler.can_fix = AsyncMock(return_value=True)
        handler.apply = AsyncMock(return_value={"action": "test_fixed"})

        registry = FixRegistry()
        registry.register("test.rule", handler)

        d = Detection(
            rule_id="test.rule",
            severity="info",
            title="Test",
            description="",
        )
        result = await registry.try_fix(d, agent_ctx)
        assert result == {"action": "test_fixed"}


# ── Pydantic Schema Tests ────────────────────────────────────────────────


class TestSchemas:
    """Tests for Pydantic response models."""

    def test_agent_stats_response(self):
        s = AgentStatsResponse(
            scan_count=5,
            incidents_total=10,
            open_incidents=3,
            auto_fixes_applied=7,
            running=True,
        )
        assert s.scan_count == 5
        assert s.running is True

    def test_incident_item(self):
        i = IncidentItem(
            id="abc",
            rule_id="gpu.fps_drop",
            severity="warning",
            title="Low FPS",
        )
        assert i.status == "open"
        assert i.details == {}

    def test_incident_list_response(self):
        r = IncidentListResponse(incidents=[], total=0)
        assert r.total == 0

    def test_prediction_item(self):
        p = PredictionItem(
            rule_id="gpu.fps_drop",
            pattern_key="node-1",
            occurrences=5,
        )
        assert p.predicted_next is None

    def test_scan_response(self):
        s = ScanResponse(
            detections=[
                DetectionItem(
                    rule_id="test",
                    severity="info",
                    title="Found",
                ),
            ],
            count=1,
        )
        assert s.count == 1

    def test_incident_action_response(self):
        r = IncidentActionResponse(incident_id="x", status="acknowledged")
        assert r.status == "acknowledged"


# ── AIAgent Integration Tests ─────────────────────────────────────────────


class TestAIAgent:
    """Integration tests for the AIAgent orchestrator."""

    @pytest.mark.asyncio
    async def test_agent_start_stop(self, agent_ctx):
        from src.services.ai_agent.agent import AIAgent

        agent = AIAgent(agent_ctx)
        assert agent._running is False

        await agent.start()
        assert agent._running is True
        assert agent._task is not None

        await agent.stop()
        assert agent._running is False

    @pytest.mark.asyncio
    async def test_agent_get_stats_no_db(self, agent_ctx):
        from src.services.ai_agent.agent import AIAgent

        agent = AIAgent(agent_ctx)
        stats = await agent.get_stats()
        assert stats["scan_count"] == 0
        assert stats["running"] is False
        assert stats["rules_count"] > 0

    @pytest.mark.asyncio
    async def test_agent_get_incidents_no_db(self, agent_ctx):
        from src.services.ai_agent.agent import AIAgent

        agent = AIAgent(agent_ctx)
        result = await agent.get_incidents()
        assert result == {"incidents": [], "total": 0}

    @pytest.mark.asyncio
    async def test_agent_manual_scan(self, agent_ctx):
        from src.services.ai_agent.agent import AIAgent

        agent = AIAgent(agent_ctx)
        # With no real system data (mocked), should get zero or just system detections
        results = await agent.run_manual_scan()
        assert isinstance(results, list)
        assert agent._scan_count == 1
        assert agent._last_scan is not None

    @pytest.mark.asyncio
    async def test_agent_rules_registered(self, agent_ctx):
        from src.services.ai_agent.agent import AIAgent

        agent = AIAgent(agent_ctx)
        rule_ids = [r.rule_id for r in agent._rule_registry.rules]
        assert "system.high_cpu" in rule_ids
        assert "business.churn_risk" in rule_ids
        assert "security.violation_spike" in rule_ids
        # GPU rules removed in Phase 1 migration (RunPod Serverless)
        assert "gpu.fps_drop" not in rule_ids

    @pytest.mark.asyncio
    async def test_agent_fixes_registered(self, agent_ctx):
        from src.services.ai_agent.agent import AIAgent

        agent = AIAgent(agent_ctx)
        assert "system.high_memory" in agent._fix_registry._handlers
        assert "business.quota_exhaustion" in agent._fix_registry._handlers
        assert "infra.runpod_workers" in agent._fix_registry._handlers
        assert "resilience.runpod_consecutive_failures" in agent._fix_registry._handlers
        assert "resilience.dashscope_consecutive_timeouts" in agent._fix_registry._handlers
        # GPU fixes removed in Phase 1 migration (RunPod Serverless)
        assert "gpu.fps_drop" not in agent._fix_registry._handlers

    @pytest.mark.asyncio
    async def test_agent_pattern_key_generation(self, agent_ctx):
        from src.services.ai_agent.agent import AIAgent

        agent = AIAgent(agent_ctx)
        d = Detection(
            rule_id="gpu.fps_drop",
            severity="warning",
            title="Low FPS",
            description="",
            details={"node_id": "n1", "customer_id": "c1"},
        )
        key = agent._make_pattern_key(d)
        assert "gpu.fps_drop" in key
        assert "node_id=n1" in key
        assert "customer_id=c1" in key

    @pytest.mark.asyncio
    async def test_agent_get_predictions_no_db(self, agent_ctx):
        from src.services.ai_agent.agent import AIAgent

        agent = AIAgent(agent_ctx)
        preds = await agent.get_predictions()
        assert preds == []


# ── AgentError Tests ──────────────────────────────────────────────────────


class TestAgentError:
    """Tests for the AgentError exception."""

    def test_agent_error_creation(self):
        from src.utils.exceptions import AgentError

        err = AgentError("test failure", detail={"key": "val"})
        assert err.message == "test failure"
        assert err.detail == {"key": "val"}
        assert err.original_exception is None

    def test_agent_error_to_dict(self):
        from src.utils.exceptions import AgentError

        err = AgentError("fail", original_exception=ValueError("bad"))
        d = err.to_dict()
        assert d["error"] == "fail"
        assert "bad" in d["cause"]

    def test_agent_error_is_smarttalker_error(self):
        from src.utils.exceptions import AgentError, SmartTalkerError

        err = AgentError()
        assert isinstance(err, SmartTalkerError)


# ── Helper ────────────────────────────────────────────────────────────────


class AsyncIteratorMock:
    """Mock async iterator for redis scan_iter."""

    def __init__(self, items):
        self._items = list(items)
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._index]
        self._index += 1
        return item


# ── Business Rule Tests ──────────────────────────────────────────────────


class TestBusinessRules:
    """Tests for business monitoring rules with mock DB."""

    @pytest.mark.asyncio
    async def test_churn_risk_no_db(self, agent_ctx):
        from src.services.ai_agent.monitors.business import ChurnRiskRule

        rule = ChurnRiskRule()
        detections = await rule.evaluate(agent_ctx)
        assert detections == []

    @pytest.mark.asyncio
    async def test_churn_risk_detects_inactive(self, db_agent_ctx, mock_db):
        from src.services.ai_agent.monitors.business import ChurnRiskRule

        _, session = mock_db
        rule = ChurnRiskRule()

        # Mock customer and avatar
        customer = MagicMock()
        customer.id = "cust-1"
        customer.name = "Acme Corp"
        avatar = MagicMock()
        avatar.id = "avatar-1"
        avatar.training_progress = 0.3

        mock_result = MagicMock()
        mock_result.all.return_value = [(customer, avatar)]
        session.execute.return_value = mock_result

        detections = await rule.evaluate(db_agent_ctx)
        assert len(detections) == 1
        assert detections[0].severity == "warning"
        assert "Acme Corp" in detections[0].title

    @pytest.mark.asyncio
    async def test_churn_risk_no_inactive(self, db_agent_ctx, mock_db):
        from src.services.ai_agent.monitors.business import ChurnRiskRule

        _, session = mock_db
        rule = ChurnRiskRule()

        mock_result = MagicMock()
        mock_result.all.return_value = []
        session.execute.return_value = mock_result

        detections = await rule.evaluate(db_agent_ctx)
        assert detections == []

    @pytest.mark.asyncio
    async def test_quota_exhaustion_no_db(self, agent_ctx):
        from src.services.ai_agent.monitors.business import QuotaExhaustionRule

        rule = QuotaExhaustionRule()
        detections = await rule.evaluate(agent_ctx)
        assert detections == []

    @pytest.mark.asyncio
    async def test_quota_exhaustion_warning(self, db_agent_ctx, mock_db):
        from src.services.ai_agent.monitors.business import QuotaExhaustionRule

        _, session = mock_db
        rule = QuotaExhaustionRule()

        # customer_id, name, plan, monthly_seconds, used_s
        mock_result = MagicMock()
        mock_result.all.return_value = [
            ("c1", "BigCo", "starter", 3600, 3400),  # 94.4%
        ]
        session.execute.return_value = mock_result

        detections = await rule.evaluate(db_agent_ctx)
        assert len(detections) == 1
        assert detections[0].severity == "warning"
        assert detections[0].auto_fixable is True

    @pytest.mark.asyncio
    async def test_quota_exhaustion_critical(self, db_agent_ctx, mock_db):
        from src.services.ai_agent.monitors.business import QuotaExhaustionRule

        _, session = mock_db
        rule = QuotaExhaustionRule()

        mock_result = MagicMock()
        mock_result.all.return_value = [
            ("c1", "BigCo", "starter", 3600, 3700),  # 102.8%
        ]
        session.execute.return_value = mock_result

        detections = await rule.evaluate(db_agent_ctx)
        assert len(detections) == 1
        assert detections[0].severity == "critical"

    @pytest.mark.asyncio
    async def test_escalation_spike_no_db(self, agent_ctx):
        from src.services.ai_agent.monitors.business import EscalationSpikeRule

        rule = EscalationSpikeRule()
        detections = await rule.evaluate(agent_ctx)
        assert detections == []

    @pytest.mark.asyncio
    async def test_escalation_spike_detects(self, db_agent_ctx, mock_db):
        from src.services.ai_agent.monitors.business import EscalationSpikeRule

        _, session = mock_db
        rule = EscalationSpikeRule()

        # avatar_id, avatar_name, customer_name, total_msgs, escalated_msgs
        mock_result = MagicMock()
        mock_result.all.return_value = [
            ("a1", "SalesBot", "Acme", 100, 30),  # 30% rate
        ]
        session.execute.return_value = mock_result

        detections = await rule.evaluate(db_agent_ctx)
        assert len(detections) == 1
        assert "SalesBot" in detections[0].title


# ── Security Rule Tests ──────────────────────────────────────────────────


class TestSecurityRules:
    """Tests for security monitoring rules with mock DB."""

    @pytest.mark.asyncio
    async def test_violation_spike_no_db(self, agent_ctx):
        from src.services.ai_agent.monitors.security import PolicyViolationSpikeRule

        rule = PolicyViolationSpikeRule()
        detections = await rule.evaluate(agent_ctx)
        assert detections == []

    @pytest.mark.asyncio
    async def test_violation_spike_detects(self, db_agent_ctx, mock_db):
        from src.services.ai_agent.monitors.security import PolicyViolationSpikeRule

        _, session = mock_db
        rule = PolicyViolationSpikeRule()

        # First call: count query returns 15
        # Second call: type breakdown
        count_result = MagicMock()
        count_result.scalar.return_value = 15

        type_result = MagicMock()
        type_result.all.return_value = [("profanity", 10), ("pii", 5)]

        session.execute = AsyncMock(side_effect=[count_result, type_result])

        detections = await rule.evaluate(db_agent_ctx)
        assert len(detections) == 1
        assert detections[0].severity == "warning"
        assert "15" in detections[0].title

    @pytest.mark.asyncio
    async def test_violation_spike_critical(self, db_agent_ctx, mock_db):
        from src.services.ai_agent.monitors.security import PolicyViolationSpikeRule

        _, session = mock_db
        rule = PolicyViolationSpikeRule()

        count_result = MagicMock()
        count_result.scalar.return_value = 25  # >= threshold * 2

        type_result = MagicMock()
        type_result.all.return_value = [("profanity", 25)]

        session.execute = AsyncMock(side_effect=[count_result, type_result])

        detections = await rule.evaluate(db_agent_ctx)
        assert len(detections) == 1
        assert detections[0].severity == "critical"

    @pytest.mark.asyncio
    async def test_violation_spike_ok(self, db_agent_ctx, mock_db):
        from src.services.ai_agent.monitors.security import PolicyViolationSpikeRule

        _, session = mock_db
        rule = PolicyViolationSpikeRule()

        count_result = MagicMock()
        count_result.scalar.return_value = 3  # below threshold
        session.execute.return_value = count_result

        detections = await rule.evaluate(db_agent_ctx)
        assert detections == []

    @pytest.mark.asyncio
    async def test_suspicious_activity_no_db(self, agent_ctx):
        from src.services.ai_agent.monitors.security import SuspiciousActivityRule

        rule = SuspiciousActivityRule()
        detections = await rule.evaluate(agent_ctx)
        assert detections == []

    @pytest.mark.asyncio
    async def test_suspicious_activity_detects(self, db_agent_ctx, mock_db):
        from src.services.ai_agent.monitors.security import SuspiciousActivityRule

        _, session = mock_db
        rule = SuspiciousActivityRule()

        # channel, count in 5 minutes (rate = count/5 >= 50)
        mock_result = MagicMock()
        mock_result.all.return_value = [("websocket", 300)]  # 60/min
        session.execute.return_value = mock_result

        detections = await rule.evaluate(db_agent_ctx)
        assert len(detections) == 1
        assert detections[0].severity == "critical"
        assert "websocket" in detections[0].title


# ── API Endpoint Tests ───────────────────────────────────────────────────


class TestAgentAPI:
    """Tests for agent REST API endpoints."""

    @pytest_asyncio.fixture
    async def api_client(self, agent_ctx):
        from fastapi import FastAPI
        from httpx import ASGITransport, AsyncClient
        from src.services.ai_agent.agent import AIAgent
        from src.services.ai_agent.routes import router

        app = FastAPI()
        app.include_router(router)
        agent = AIAgent(agent_ctx)
        app.state.ai_agent = agent

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            yield client

    @pytest.mark.asyncio
    async def test_get_stats(self, api_client):
        resp = await api_client.get("/api/v1/agent/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "scan_count" in data
        assert data["running"] is False
        assert data["rules_count"] == 44  # 30 + 5 resilience + 4 prediction + 5 channel rules

    @pytest.mark.asyncio
    async def test_get_incidents_empty(self, api_client):
        resp = await api_client.get("/api/v1/agent/incidents")
        assert resp.status_code == 200
        data = resp.json()
        assert data["incidents"] == []
        assert data["total"] == 0

    @pytest.mark.asyncio
    async def test_get_incidents_with_status(self, api_client):
        resp = await api_client.get("/api/v1/agent/incidents?status=open")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_get_predictions(self, api_client):
        resp = await api_client.get("/api/v1/agent/predictions")
        assert resp.status_code == 200
        data = resp.json()
        assert "predictions" in data

    @pytest.mark.asyncio
    async def test_post_scan(self, api_client):
        resp = await api_client.post("/api/v1/agent/scan")
        assert resp.status_code == 200
        data = resp.json()
        assert "detections" in data
        assert "count" in data

    @pytest.mark.asyncio
    async def test_acknowledge_nonexistent_incident(self, api_client):
        """Acknowledging a non-existent incident returns 404."""
        resp = await api_client.post("/api/v1/agent/incidents/test-id/acknowledge")
        assert resp.status_code == 404
        data = resp.json()
        assert "error" in data

    @pytest.mark.asyncio
    async def test_resolve_nonexistent_incident(self, api_client):
        """Resolving a non-existent incident returns 404."""
        resp = await api_client.post("/api/v1/agent/incidents/test-id/resolve")
        assert resp.status_code == 404
        data = resp.json()
        assert "error" in data

    @pytest.mark.asyncio
    async def test_stats_no_agent(self):
        """Endpoints gracefully handle missing agent on app.state."""
        from fastapi import FastAPI
        from httpx import ASGITransport, AsyncClient
        from src.services.ai_agent.routes import router

        app = FastAPI()
        app.include_router(router)
        # No ai_agent on app.state

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await client.get("/api/v1/agent/stats")
            assert resp.status_code == 200
            assert resp.json()["running"] is False


# ── Notification Dispatcher Tests ────────────────────────────────────────


class TestNotificationDispatcher:
    """Tests for the NotificationDispatcher."""

    @pytest.fixture
    def dispatcher(self, mock_operator_manager, agent_config, tmp_path):
        from src.services.ai_agent.notifications import (
            NotificationDispatcher,
            NotificationSettings,
        )

        agent_config.audit_log_path = str(tmp_path / "audit.jsonl")
        smtp_config = NotificationSettings(
            host="localhost", port=25, username="", password=""
        )
        return NotificationDispatcher(
            operator_manager=mock_operator_manager,
            agent_config=agent_config,
            smtp_config=smtp_config,
        )

    def _make_detection(self, severity="warning", rule_id="test.rule"):
        return Detection(
            rule_id=rule_id,
            severity=severity,
            title=f"Test {severity}",
            description="Test description",
            details={"key": "val"},
            recommendation="Do something",
        )

    @pytest.mark.asyncio
    async def test_critical_sends_ws_immediately(
        self, dispatcher, mock_operator_manager
    ):
        """Critical detections are pushed to WebSocket immediately."""
        op = MagicMock()
        op.authenticated = True
        op.websocket = MagicMock()
        mock_operator_manager._operators = {"op1": op}

        d = self._make_detection("critical")
        await dispatcher.dispatch(d, "inc-1")

        mock_operator_manager._send_json.assert_called_once()
        msg = mock_operator_manager._send_json.call_args[0][1]
        assert msg["type"] == "agent_alert"
        assert msg["severity"] == "critical"

    @pytest.mark.asyncio
    async def test_critical_skips_email_no_username(self, dispatcher):
        """Email is skipped when SMTP username is empty."""
        d = self._make_detection("critical")
        # Should not raise even though email is configured but no username
        await dispatcher.dispatch(d, "inc-1")

    @pytest.mark.asyncio
    async def test_warning_queued_not_immediate(
        self, dispatcher, mock_operator_manager
    ):
        """Warning detections are queued, not sent immediately."""
        op = MagicMock()
        op.authenticated = True
        mock_operator_manager._operators = {"op1": op}

        d = self._make_detection("warning")
        await dispatcher.dispatch(d, "inc-1")

        # No immediate WS call for warning
        mock_operator_manager._send_json.assert_not_called()
        assert dispatcher._warning_queue.qsize() == 1

    @pytest.mark.asyncio
    async def test_warning_batch_flush(self, dispatcher, mock_operator_manager):
        """Flushing warning queue sends batched alert."""
        op = MagicMock()
        op.authenticated = True
        op.websocket = MagicMock()
        mock_operator_manager._operators = {"op1": op}

        # Enqueue 3 warnings
        for i in range(3):
            d = self._make_detection("warning", f"rule.{i}")
            await dispatcher._warning_queue.put((d, f"inc-{i}"))

        await dispatcher._flush_warning_queue()

        mock_operator_manager._send_json.assert_called_once()
        msg = mock_operator_manager._send_json.call_args[0][1]
        assert msg["type"] == "agent_alert_batch"
        assert msg["count"] == 3
        assert dispatcher._warning_queue.qsize() == 0

    @pytest.mark.asyncio
    async def test_info_queued(self, dispatcher):
        """Info detections are queued for digest."""
        d = self._make_detection("info")
        await dispatcher.dispatch(d, "inc-1")
        assert dispatcher._info_queue.qsize() == 1

    @pytest.mark.asyncio
    async def test_info_digest_flush(self, dispatcher, mock_operator_manager):
        """Flushing info queue sends digest alert."""
        op = MagicMock()
        op.authenticated = True
        op.websocket = MagicMock()
        mock_operator_manager._operators = {"op1": op}

        for i in range(5):
            d = self._make_detection("info", f"rule.{i}")
            await dispatcher._info_queue.put((d, f"inc-{i}"))

        await dispatcher._flush_info_queue()

        mock_operator_manager._send_json.assert_called_once()
        msg = mock_operator_manager._send_json.call_args[0][1]
        assert msg["type"] == "agent_alert_batch"
        assert msg["severity"] == "info"
        assert msg["count"] == 5

    @pytest.mark.asyncio
    async def test_audit_log_written(self, dispatcher, tmp_path):
        """Audit log file contains entry after dispatch."""
        d = self._make_detection("warning")
        await dispatcher.dispatch(d, "inc-1")

        # Force handler flush
        for handler in dispatcher._audit.handlers:
            handler.flush()

        audit_path = tmp_path / "audit.jsonl"
        assert audit_path.exists()
        content = audit_path.read_text()
        assert "Test warning" in content

    @pytest.mark.asyncio
    async def test_audit_log_json_format(self, dispatcher, tmp_path):
        """Each line in audit log is valid JSON."""
        d = self._make_detection("critical")
        await dispatcher.dispatch(d, "inc-1")

        for handler in dispatcher._audit.handlers:
            handler.flush()

        audit_path = tmp_path / "audit.jsonl"
        for line in audit_path.read_text().strip().split("\n"):
            parsed = json.loads(line)
            assert "message" in parsed

    @pytest.mark.asyncio
    async def test_status_change_logged(self, dispatcher, tmp_path):
        """Status changes are written to audit log."""
        await dispatcher.dispatch_status_change("inc-1", "auto_fixed")

        for handler in dispatcher._audit.handlers:
            handler.flush()

        audit_path = tmp_path / "audit.jsonl"
        content = audit_path.read_text()
        assert "auto_fixed" in content

    @pytest.mark.asyncio
    async def test_no_operator_manager_graceful(self, agent_config, tmp_path):
        """Dispatcher with operator_manager=None skips WebSocket."""
        from src.services.ai_agent.notifications import (
            NotificationDispatcher,
            NotificationSettings,
        )

        agent_config.audit_log_path = str(tmp_path / "audit2.jsonl")
        smtp_config = NotificationSettings(
            host="localhost", port=25, username="", password=""
        )
        dispatcher = NotificationDispatcher(
            operator_manager=None,
            agent_config=agent_config,
            smtp_config=smtp_config,
        )
        d = self._make_detection("critical")
        # Should not raise
        await dispatcher.dispatch(d, "inc-1")

    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self, dispatcher):
        """Dispatcher starts and stops background tasks."""
        await dispatcher.start()
        assert dispatcher._warning_task is not None
        assert dispatcher._info_task is not None

        await dispatcher.stop()
        assert dispatcher._warning_task is None
        assert dispatcher._info_task is None


# ── Pattern Tracker Tests ────────────────────────────────────────────────


class TestPatternTracker:
    """Tests for pattern tracking and prediction."""

    @pytest.mark.asyncio
    async def test_record_no_db(self):
        from src.services.ai_agent.prevention import PatternTracker

        tracker = PatternTracker(db=None)
        await tracker.record_occurrence("test.rule", "key")  # should not raise

    @pytest.mark.asyncio
    async def test_get_predictions_no_db(self):
        from src.services.ai_agent.prevention import PatternTracker

        tracker = PatternTracker(db=None)
        result = await tracker.get_predictions()
        assert result == []

    @pytest.mark.asyncio
    async def test_record_new_pattern(self, mock_db):
        from src.services.ai_agent.prevention import PatternTracker

        db, session = mock_db
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        session.execute.return_value = mock_result

        tracker = PatternTracker(db=db)
        await tracker.record_occurrence("test.rule", "key")
        session.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_record_existing_pattern(self, mock_db):
        from src.services.ai_agent.prevention import PatternTracker

        db, session = mock_db
        existing = MagicMock()
        existing.occurrence_count = 3
        existing.first_seen = datetime.utcnow() - timedelta(hours=3)
        existing.last_seen = datetime.utcnow() - timedelta(hours=1)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = existing
        session.execute.return_value = mock_result

        tracker = PatternTracker(db=db)
        await tracker.record_occurrence("test.rule", "key")
        assert existing.occurrence_count == 4

    @pytest.mark.asyncio
    async def test_get_predictions_returns_list(self, mock_db):
        from src.services.ai_agent.prevention import PatternTracker

        db, session = mock_db
        pattern = MagicMock()
        pattern.rule_id = "test.rule"
        pattern.pattern_key = "key"
        pattern.occurrence_count = 5
        pattern.first_seen = datetime(2025, 1, 1)
        pattern.last_seen = datetime(2025, 1, 2)
        pattern.predicted_next = datetime(2025, 1, 3)

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [pattern]
        session.execute.return_value = mock_result

        tracker = PatternTracker(db=db)
        preds = await tracker.get_predictions()
        assert len(preds) == 1
        assert preds[0]["rule_id"] == "test.rule"
        assert preds[0]["occurrences"] == 5


# ── Incident Persistence Tests ──────────────────────────────────────────


class TestIncidentPersistence:
    """Tests for agent incident DB operations."""

    @pytest.mark.asyncio
    async def test_persist_incident_no_db(self, agent_ctx):
        from src.services.ai_agent.agent import AIAgent

        agent = AIAgent(agent_ctx)
        d = Detection(
            rule_id="test", severity="info", title="T", description="D"
        )
        result = await agent._persist_incident(d)
        assert result is None

    @pytest.mark.asyncio
    async def test_persist_action_no_db(self, agent_ctx):
        from src.services.ai_agent.agent import AIAgent

        agent = AIAgent(agent_ctx)
        # Should not raise
        await agent._persist_action(
            incident_id="x",
            action_type="test",
            description="d",
            result={"a": 1},
        )

    @pytest.mark.asyncio
    async def test_update_incident_status_no_db(self, agent_ctx):
        from src.services.ai_agent.agent import AIAgent

        agent = AIAgent(agent_ctx)
        # Should not raise
        await agent._update_incident_status("x", "resolved")

    @pytest.mark.asyncio
    async def test_persist_incident_with_db(self, db_agent_ctx, mock_db):
        from src.services.ai_agent.agent import AIAgent

        _, session = mock_db
        agent = AIAgent(db_agent_ctx)
        d = Detection(
            rule_id="test", severity="info", title="T", description="D"
        )
        await agent._persist_incident(d)
        session.add.assert_called_once()
        session.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_persist_incident_db_error(self, db_agent_ctx, mock_db):
        from src.services.ai_agent.agent import AIAgent

        _, session = mock_db
        session.add.side_effect = RuntimeError("DB down")

        agent = AIAgent(db_agent_ctx)
        d = Detection(
            rule_id="test", severity="info", title="T", description="D"
        )
        result = await agent._persist_incident(d)
        assert result is None  # Error caught, returns None


# ── Handle Detection Flow Tests ──────────────────────────────────────────


class TestHandleDetection:
    """Tests for the full detection handling pipeline."""

    @pytest.mark.asyncio
    async def test_handle_detection_calls_notifier(self, agent_ctx):
        from src.services.ai_agent.agent import AIAgent

        agent = AIAgent(agent_ctx)
        agent._notifier = MagicMock()
        agent._notifier.dispatch = AsyncMock()
        agent._notifier.dispatch_status_change = AsyncMock()

        d = Detection(
            rule_id="test.rule",
            severity="warning",
            title="Test",
            description="",
        )
        await agent._handle_detection(d)

        agent._notifier.dispatch.assert_called_once_with(d, None)

    @pytest.mark.asyncio
    async def test_handle_detection_no_autofix_when_not_fixable(self, agent_ctx):
        from src.services.ai_agent.agent import AIAgent

        agent = AIAgent(agent_ctx)
        agent._notifier = MagicMock()
        agent._notifier.dispatch = AsyncMock()
        agent._fix_registry = MagicMock()
        agent._fix_registry.try_fix = AsyncMock()

        d = Detection(
            rule_id="test",
            severity="info",
            title="T",
            description="",
            auto_fixable=False,
        )
        await agent._handle_detection(d)

        agent._fix_registry.try_fix.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_detection_no_autofix_when_disabled(self, agent_ctx):
        from src.services.ai_agent.agent import AIAgent

        agent_ctx.agent_config.auto_fix_enabled = False
        agent = AIAgent(agent_ctx)
        agent._notifier = MagicMock()
        agent._notifier.dispatch = AsyncMock()
        agent._fix_registry = MagicMock()
        agent._fix_registry.try_fix = AsyncMock()

        d = Detection(
            rule_id="test",
            severity="info",
            title="T",
            description="",
            auto_fixable=True,
        )
        await agent._handle_detection(d)

        agent._fix_registry.try_fix.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_detection_autofix_triggers_status_change(
        self, db_agent_ctx, mock_db
    ):
        from src.services.ai_agent.agent import AIAgent

        _, session = mock_db
        agent = AIAgent(db_agent_ctx)

        # Mock persist_incident to return an ID
        agent._persist_incident = AsyncMock(return_value="inc-1")
        agent._persist_action = AsyncMock()
        agent._update_incident_status = AsyncMock()
        agent._notifier = MagicMock()
        agent._notifier.dispatch = AsyncMock()
        agent._notifier.dispatch_status_change = AsyncMock()
        agent._fix_registry = MagicMock()
        agent._fix_registry.try_fix = AsyncMock(
            return_value={"action": "test_fix"}
        )

        d = Detection(
            rule_id="test",
            severity="warning",
            title="T",
            description="",
            auto_fixable=True,
        )
        await agent._handle_detection(d)

        agent._fix_registry.try_fix.assert_called_once()
        agent._update_incident_status.assert_called_once_with(
            "inc-1", "auto_fixed"
        )
        agent._notifier.dispatch_status_change.assert_called_once_with(
            "inc-1", "auto_fixed"
        )

    @pytest.mark.asyncio
    async def test_handle_detection_full_flow(self, agent_ctx):
        """Complete flow: persist → pattern → notify → (skip fix for non-fixable)."""
        from src.services.ai_agent.agent import AIAgent

        agent = AIAgent(agent_ctx)
        agent._notifier = MagicMock()
        agent._notifier.dispatch = AsyncMock()

        d = Detection(
            rule_id="gpu.fps_drop",
            severity="warning",
            title="Low FPS",
            description="",
            details={"node_id": "n1"},
        )
        # Should not raise with db=None
        await agent._handle_detection(d)
        agent._notifier.dispatch.assert_called_once()


# ── NotificationSettings Tests ───────────────────────────────────────────


class TestNotificationSettings:
    """Tests for NotificationSettings Pydantic model."""

    def test_defaults(self):
        from src.services.ai_agent.notifications import NotificationSettings

        s = NotificationSettings()
        assert s.host == "smtp.gmail.com"
        assert s.port == 587
        assert s.use_tls is True

    def test_custom(self):
        from src.services.ai_agent.notifications import NotificationSettings

        s = NotificationSettings(host="mail.example.com", port=465, use_tls=False)
        assert s.host == "mail.example.com"
        assert s.port == 465

    def test_notification_config_fields(self):
        """AgentSettings has notification-related fields."""
        s = AgentSettings()
        assert s.notification_ws_enabled is True
        assert s.notification_email_enabled is True
        assert s.notification_email_to == "contact@lsmarttech.com"
        assert s.notification_warning_batch_s == 300
        assert s.notification_info_digest_s == 3600
        assert s.audit_log_path == "./logs/agent_audit.jsonl"
