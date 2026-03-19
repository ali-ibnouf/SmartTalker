"""Tests for AI Agent safety constraints.

Covers: SafetyGuard, CircuitBreaker, FixRateLimiter, kill switch,
high-impact fix gating, DB connection kill limits, session close limits,
and agent loop error isolation.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.ai_agent.config import AgentSettings
from src.services.ai_agent.rules import AgentContext, Detection
from src.services.ai_agent.safety import (
    HIGH_IMPACT_FIXES,
    CircuitBreaker,
    FixRateLimiter,
    SafetyGuard,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def agent_config():
    return AgentSettings(
        agent_enabled=True,
        auto_fix_enabled=True,
        scan_interval_s=1,
        safety_max_fixes_per_cycle=3,
        safety_max_fixes_per_hour=10,
        safety_circuit_breaker_threshold=3,
        safety_circuit_breaker_cooldown_s=60,
        safety_db_kill_max=5,
        safety_max_session_close_per_cycle=5,
    )


@pytest.fixture
def mock_redis():
    redis = AsyncMock()
    redis.exists = AsyncMock(return_value=False)
    redis.set = AsyncMock()
    redis.delete = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    return redis


@pytest.fixture
def agent_ctx(config, agent_config, mock_pipeline, mock_redis):
    return AgentContext(
        db=None,
        redis=mock_redis,
        pipeline=mock_pipeline,
        config=config,
        agent_config=agent_config,
    )


@pytest.fixture
def safety_guard(mock_redis, agent_config):
    return SafetyGuard(
        redis=mock_redis,
        circuit_breaker_threshold=agent_config.safety_circuit_breaker_threshold,
        circuit_breaker_cooldown_s=agent_config.safety_circuit_breaker_cooldown_s,
        max_fixes_per_cycle=agent_config.safety_max_fixes_per_cycle,
        max_fixes_per_hour=agent_config.safety_max_fixes_per_hour,
    )


# ── CircuitBreaker Tests ─────────────────────────────────────────────────


class TestCircuitBreaker:

    def test_starts_closed(self):
        cb = CircuitBreaker(threshold=3, cooldown_s=60)
        assert not cb.is_open("test_rule")

    def test_opens_after_threshold_failures(self):
        cb = CircuitBreaker(threshold=3, cooldown_s=60)
        cb.record_failure("test_rule")
        cb.record_failure("test_rule")
        assert not cb.is_open("test_rule")
        cb.record_failure("test_rule")  # 3rd failure = threshold
        assert cb.is_open("test_rule")

    def test_success_resets_count(self):
        cb = CircuitBreaker(threshold=3, cooldown_s=60)
        cb.record_failure("test_rule")
        cb.record_failure("test_rule")
        cb.record_success("test_rule")
        cb.record_failure("test_rule")
        cb.record_failure("test_rule")
        # Only 2 failures since last success — should still be closed
        assert not cb.is_open("test_rule")

    def test_cooldown_resets_breaker(self):
        cb = CircuitBreaker(threshold=2, cooldown_s=3600)
        cb.record_failure("test_rule")
        cb.record_failure("test_rule")
        assert cb.is_open("test_rule")
        # Simulate cooldown expiry by backdating the trip timestamp
        cb._tripped_at["test_rule"] = time.time() - 3601
        assert not cb.is_open("test_rule")

    def test_independent_rules(self):
        cb = CircuitBreaker(threshold=2, cooldown_s=60)
        cb.record_failure("rule_a")
        cb.record_failure("rule_a")
        assert cb.is_open("rule_a")
        assert not cb.is_open("rule_b")

    def test_get_open_circuits(self):
        cb = CircuitBreaker(threshold=1, cooldown_s=3600)
        cb.record_failure("rule_a")
        cb.record_failure("rule_b")
        open_circuits = cb.get_open_circuits()
        assert "rule_a" in open_circuits
        assert "rule_b" in open_circuits

    def test_get_open_circuits_empty(self):
        cb = CircuitBreaker(threshold=3, cooldown_s=60)
        assert cb.get_open_circuits() == []


# ── FixRateLimiter Tests ─────────────────────────────────────────────────


class TestFixRateLimiter:

    def test_allows_within_limits(self):
        limiter = FixRateLimiter(max_per_cycle=3, max_per_hour=10)
        assert limiter.can_fix()

    def test_blocks_after_cycle_limit(self):
        limiter = FixRateLimiter(max_per_cycle=2, max_per_hour=10)
        limiter.record_fix()
        limiter.record_fix()
        assert not limiter.can_fix()

    def test_cycle_reset(self):
        limiter = FixRateLimiter(max_per_cycle=1, max_per_hour=10)
        limiter.record_fix()
        assert not limiter.can_fix()
        limiter.reset_cycle()
        assert limiter.can_fix()

    def test_blocks_after_hour_limit(self):
        limiter = FixRateLimiter(max_per_cycle=100, max_per_hour=3)
        limiter.record_fix()
        limiter.record_fix()
        limiter.record_fix()
        assert not limiter.can_fix()

    def test_hour_count_property(self):
        limiter = FixRateLimiter(max_per_cycle=10, max_per_hour=10)
        limiter.record_fix()
        limiter.record_fix()
        assert limiter.hour_count == 2

    def test_cycle_count_property(self):
        limiter = FixRateLimiter(max_per_cycle=10, max_per_hour=10)
        limiter.record_fix()
        assert limiter.cycle_count == 1


# ── SafetyGuard Tests ────────────────────────────────────────────────────


class TestSafetyGuard:

    @pytest.mark.asyncio
    async def test_kill_switch_inactive_by_default(self, safety_guard, mock_redis):
        mock_redis.exists.return_value = False
        assert not await safety_guard.is_kill_switch_active()

    @pytest.mark.asyncio
    async def test_kill_switch_active(self, safety_guard, mock_redis):
        mock_redis.exists.return_value = True
        assert await safety_guard.is_kill_switch_active()

    @pytest.mark.asyncio
    async def test_activate_kill_switch(self, safety_guard, mock_redis):
        result = await safety_guard.activate_kill_switch("test reason")
        assert result is True
        mock_redis.set.assert_called_once_with("agent:kill_switch", "test reason", ex=86400)

    @pytest.mark.asyncio
    async def test_deactivate_kill_switch(self, safety_guard, mock_redis):
        result = await safety_guard.deactivate_kill_switch()
        assert result is True
        mock_redis.delete.assert_called_once_with("agent:kill_switch")

    @pytest.mark.asyncio
    async def test_kill_switch_no_redis(self):
        guard = SafetyGuard(redis=None)
        assert not await guard.is_kill_switch_active()
        assert not await guard.activate_kill_switch("test")
        assert not await guard.deactivate_kill_switch()

    @pytest.mark.asyncio
    async def test_kill_switch_redis_error_returns_false(self, safety_guard, mock_redis):
        """If Redis is down, kill switch should NOT block (fail-open)."""
        mock_redis.exists.side_effect = ConnectionError("Redis down")
        assert not await safety_guard.is_kill_switch_active()

    def test_blocks_high_impact_fixes(self, safety_guard):
        assert not safety_guard.can_auto_fix("fix.video_disable")
        assert not safety_guard.can_auto_fix("fix.text_only_mode")
        assert not safety_guard.can_auto_fix("fix.db_connection_cleanup")

    def test_allows_normal_fixes(self, safety_guard):
        assert safety_guard.can_auto_fix("fix.cache_clear")
        assert safety_guard.can_auto_fix("fix.quota_warning")
        assert safety_guard.can_auto_fix("fix.rate_throttle")

    def test_respects_rate_limits(self, safety_guard):
        for _ in range(3):
            safety_guard.record_fix_applied()
        # Cycle limit of 3 reached
        assert not safety_guard.can_auto_fix("fix.cache_clear")

    def test_start_cycle_resets(self, safety_guard):
        safety_guard.record_fix_applied()
        safety_guard.record_fix_applied()
        safety_guard.record_fix_applied()
        assert not safety_guard.can_auto_fix("fix.cache_clear")
        safety_guard.start_cycle()
        assert safety_guard.can_auto_fix("fix.cache_clear")

    def test_should_skip_rule_delegates_to_circuit_breaker(self, safety_guard):
        cb = safety_guard.circuit_breaker
        cb.record_failure("rule_x")
        cb.record_failure("rule_x")
        cb.record_failure("rule_x")  # threshold = 3
        assert safety_guard.should_skip_rule("rule_x")
        assert not safety_guard.should_skip_rule("rule_y")

    @pytest.mark.asyncio
    async def test_get_status(self, safety_guard, mock_redis):
        mock_redis.exists.return_value = False
        status = await safety_guard.get_status()
        assert status["kill_switch_active"] is False
        assert isinstance(status["open_circuit_breakers"], list)
        assert isinstance(status["fixes_this_cycle"], int)
        assert isinstance(status["fixes_this_hour"], int)
        assert status["max_fixes_per_cycle"] == 3
        assert status["max_fixes_per_hour"] == 10
        assert isinstance(status["high_impact_fixes_requiring_approval"], list)
        assert len(status["high_impact_fixes_requiring_approval"]) == 3


# ── HIGH_IMPACT_FIXES Set Tests ──────────────────────────────────────────


class TestHighImpactFixes:

    def test_contains_expected_fixes(self):
        assert "fix.video_disable" in HIGH_IMPACT_FIXES
        assert "fix.text_only_mode" in HIGH_IMPACT_FIXES
        assert "fix.db_connection_cleanup" in HIGH_IMPACT_FIXES

    def test_does_not_contain_safe_fixes(self):
        assert "fix.cache_clear" not in HIGH_IMPACT_FIXES
        assert "fix.quota_warning" not in HIGH_IMPACT_FIXES
        assert "fix.rate_throttle" not in HIGH_IMPACT_FIXES

    def test_is_frozenset(self):
        assert isinstance(HIGH_IMPACT_FIXES, frozenset)


# ── Handler Safety Tests ─────────────────────────────────────────────────


class TestDBConnectionFixSafety:

    @pytest.mark.asyncio
    async def test_respects_max_kill_limit(self, agent_ctx, agent_config):
        """DBConnectionFix should LIMIT kills per the config."""
        from src.services.ai_agent.fixes.handlers import DBConnectionFix

        fix = DBConnectionFix()
        detection = Detection(
            rule_id="infra.pg_connections",
            severity="critical",
            title="High DB connections",
            description="test",
            details={"usage_pct": 90},
            auto_fixable=True,
        )

        # Create mock DB
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.__iter__ = lambda self: iter([(True,)] * 3)
        mock_session.execute = AsyncMock(return_value=mock_result)

        mock_db = MagicMock()

        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def mock_session_ctx():
            yield mock_session

        mock_db.session_ctx = mock_session_ctx
        agent_ctx.db = mock_db

        result = await fix.apply(detection, agent_ctx)
        assert result["action"] == "db_connection_cleanup"
        assert result.get("max_kill_limit") == agent_config.safety_db_kill_max

        # Verify the SQL contains LIMIT
        sql_call = mock_session.execute.call_args[0][0]
        sql_text = str(sql_call)
        assert "LIMIT" in sql_text
        assert "backend_type" in sql_text


class TestStaleSessionFixSafety:

    @pytest.mark.asyncio
    async def test_respects_max_close_limit(self, agent_ctx, agent_config):
        """StaleSessionFix should cap sessions closed per cycle."""
        from src.services.ai_agent.fixes.handlers import StaleSessionFix

        fix = StaleSessionFix()

        # Create 20 stale sessions
        sessions = []
        for i in range(20):
            s = MagicMock()
            s.session_id = f"session_{i}"
            s.last_activity = 0.0  # very old
            sessions.append(s)

        mock_om = MagicMock()
        mock_om.get_active_sessions = MagicMock(return_value=sessions)
        mock_om.close_session = AsyncMock()
        agent_ctx.operator_manager = mock_om

        result = await fix.apply_scheduled(agent_ctx)
        assert result["sessions_closed"] == agent_config.safety_max_session_close_per_cycle
        assert result["max_per_cycle"] == agent_config.safety_max_session_close_per_cycle
        # Should only have been called 5 times (the limit)
        assert mock_om.close_session.call_count == agent_config.safety_max_session_close_per_cycle


class TestMemoryCleanupFixSafety:

    @pytest.mark.asyncio
    async def test_session_close_capped(self, agent_ctx, agent_config):
        """MemoryCleanupFix should cap session closes."""
        from src.services.ai_agent.fixes.handlers import MemoryCleanupFix

        fix = MemoryCleanupFix()
        detection = Detection(
            rule_id="system.high_memory",
            severity="warning",
            title="High memory",
            description="test",
            auto_fixable=True,
        )

        # Mock Redis scan_iter to return nothing
        agent_ctx.redis.scan_iter = MagicMock(return_value=AsyncIterMock([]))

        # Create 20 stale sessions
        sessions = []
        for i in range(20):
            s = MagicMock()
            s.session_id = f"session_{i}"
            s.last_activity = 0.0
            sessions.append(s)

        mock_om = MagicMock()
        mock_om.get_active_sessions = MagicMock(return_value=sessions)
        mock_om.close_session = AsyncMock()
        agent_ctx.operator_manager = mock_om

        result = await fix.apply(detection, agent_ctx)
        assert result["sessions_closed"] <= agent_config.safety_max_session_close_per_cycle


class AsyncIterMock:
    """Helper to mock async iterators."""

    def __init__(self, items):
        self._items = items
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._index]
        self._index += 1
        return item


# ── Agent Loop Integration Tests ─────────────────────────────────────────


class TestAgentLoopSafety:

    @pytest.mark.asyncio
    async def test_kill_switch_skips_scan(self, agent_ctx, mock_redis):
        """When kill switch is active, the agent loop should skip scanning."""
        from src.services.ai_agent.agent import AIAgent

        mock_redis.exists = AsyncMock(return_value=True)  # kill switch active
        agent = AIAgent(agent_ctx)

        # Run one iteration
        agent._running = True
        original_interval = agent._ctx.agent_config.scan_interval_s
        agent._ctx.agent_config.scan_interval_s = 0  # no delay

        # Patch sleep to stop after one iteration
        call_count = 0

        async def mock_sleep(seconds):
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                agent._running = False

        with patch("asyncio.sleep", mock_sleep):
            await agent._loop()

        # The agent should NOT have run any rules
        assert agent._scan_count == 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_skips_failing_rules(self, agent_ctx, mock_redis):
        """Circuit breaker should skip rules that fail consecutively."""
        from src.services.ai_agent.agent import AIAgent

        mock_redis.exists = AsyncMock(return_value=False)  # kill switch off
        agent = AIAgent(agent_ctx)

        # Trip the circuit breaker for a specific rule
        for _ in range(agent_ctx.agent_config.safety_circuit_breaker_threshold):
            agent._safety.circuit_breaker.record_failure("system.high_cpu")

        # Verify the rule is now skipped
        assert agent._safety.should_skip_rule("system.high_cpu")

    @pytest.mark.asyncio
    async def test_high_impact_fix_routed_to_approval(self, agent_ctx, mock_redis):
        """VideoDisableFix should be routed to approval queue, not auto-executed."""
        from src.services.ai_agent.agent import AIAgent

        mock_redis.exists = AsyncMock(return_value=False)
        agent = AIAgent(agent_ctx)

        # Create a detection that would trigger VideoDisableFix
        detection = Detection(
            rule_id="resilience.runpod_consecutive_failures",
            severity="critical",
            title="RunPod consecutive failures",
            description="5 consecutive RunPod render failures",
            details={"consecutive_failures": 5},
            auto_fixable=True,
        )

        # Mock the persist methods
        agent._persist_incident = AsyncMock(return_value="inc_123")
        agent._pattern_tracker.record_occurrence = AsyncMock()
        agent._notifier.dispatch = AsyncMock()

        # Mock _route_to_approval to track if it's called
        agent._route_to_approval = AsyncMock()

        await agent._handle_detection(detection)

        # Should have been routed to approval, not auto-fixed
        agent._route_to_approval.assert_called_once()

    @pytest.mark.asyncio
    async def test_fix_rate_limit_enforced(self, agent_ctx, mock_redis):
        """Fix rate limiter should block fixes after cycle limit."""
        from src.services.ai_agent.agent import AIAgent

        mock_redis.exists = AsyncMock(return_value=False)
        agent = AIAgent(agent_ctx)
        agent._persist_incident = AsyncMock(return_value="inc_123")
        agent._pattern_tracker.record_occurrence = AsyncMock()
        agent._notifier.dispatch = AsyncMock()

        # Fill up the fix rate limit
        for _ in range(agent_ctx.agent_config.safety_max_fixes_per_cycle):
            agent._safety.record_fix_applied()

        detection = Detection(
            rule_id="system.high_memory",
            severity="warning",
            title="High memory",
            description="test",
            auto_fixable=True,
        )

        # Mock the fix registry to track calls
        agent._fix_registry.try_fix = AsyncMock(return_value={"action": "cache_cleared"})

        await agent._handle_detection(detection)

        # Fix should NOT have been attempted
        agent._fix_registry.try_fix.assert_not_called()

    @pytest.mark.asyncio
    async def test_agent_loop_survives_exception(self, agent_ctx, mock_redis):
        """Agent loop should catch exceptions without crashing."""
        from src.services.ai_agent.agent import AIAgent

        mock_redis.exists = AsyncMock(return_value=False)
        agent = AIAgent(agent_ctx)

        # Make _safe_run_rules raise an exception
        agent._safe_run_rules = AsyncMock(side_effect=RuntimeError("boom"))

        agent._running = True
        call_count = 0

        async def mock_sleep(seconds):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                agent._running = False

        with patch("asyncio.sleep", mock_sleep):
            # Should NOT raise — the loop catches everything
            await agent._loop()

        # Should have survived 2 iterations despite errors
        assert call_count >= 2

    @pytest.mark.asyncio
    async def test_stale_session_loop_respects_kill_switch(self, agent_ctx, mock_redis):
        """Stale session loop should skip when kill switch is active."""
        from src.services.ai_agent.agent import AIAgent

        mock_redis.exists = AsyncMock(return_value=True)  # kill switch ON
        agent = AIAgent(agent_ctx)
        agent._stale_session_fix.apply_scheduled = AsyncMock()

        agent._running = True
        call_count = 0

        async def mock_sleep(seconds):
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                agent._running = False

        with patch("asyncio.sleep", mock_sleep):
            await agent._stale_session_loop()

        # apply_scheduled should NOT have been called
        agent._stale_session_fix.apply_scheduled.assert_not_called()


# ── Config Tests ─────────────────────────────────────────────────────────


class TestSafetyConfig:

    def test_safety_defaults(self):
        cfg = AgentSettings()
        assert cfg.safety_max_fixes_per_cycle == 3
        assert cfg.safety_max_fixes_per_hour == 10
        assert cfg.safety_circuit_breaker_threshold == 5
        assert cfg.safety_circuit_breaker_cooldown_s == 600
        assert cfg.safety_db_kill_max == 5
        assert cfg.safety_max_session_close_per_cycle == 5

    def test_safety_overridable(self):
        cfg = AgentSettings(
            safety_max_fixes_per_cycle=1,
            safety_max_fixes_per_hour=5,
            safety_circuit_breaker_threshold=2,
            safety_db_kill_max=2,
        )
        assert cfg.safety_max_fixes_per_cycle == 1
        assert cfg.safety_max_fixes_per_hour == 5
        assert cfg.safety_circuit_breaker_threshold == 2
        assert cfg.safety_db_kill_max == 2


# ── Approval Expansion Tests ─────────────────────────────────────────────


class TestApprovalExpansion:

    def test_high_impact_fix_actions_defined(self):
        from src.services.ai_agent.approval import HIGH_IMPACT_FIX_ACTIONS

        assert "video_disable" in HIGH_IMPACT_FIX_ACTIONS
        assert "text_only_mode" in HIGH_IMPACT_FIX_ACTIONS
        assert "db_connection_cleanup" in HIGH_IMPACT_FIX_ACTIONS

    def test_original_approval_actions_unchanged(self):
        from src.services.ai_agent.approval import APPROVAL_ACTIONS

        assert "suspend_customer" in APPROVAL_ACTIONS
        assert "kill_switch" in APPROVAL_ACTIONS
        assert "plan_downgrade" in APPROVAL_ACTIONS
        assert "data_deletion" in APPROVAL_ACTIONS


# ── Manual Scan Safety Tests ─────────────────────────────────────────────


class TestManualScanSafety:

    @pytest.mark.asyncio
    async def test_manual_scan_blocked_by_kill_switch(self, agent_ctx, mock_redis):
        """Manual scan should return [] when kill switch is active."""
        from src.services.ai_agent.agent import AIAgent

        mock_redis.exists = AsyncMock(return_value=True)  # kill switch ON
        agent = AIAgent(agent_ctx)

        result = await agent.run_manual_scan()
        assert result == []
        # Scan count should NOT increment
        assert agent._scan_count == 0

    @pytest.mark.asyncio
    async def test_manual_scan_uses_circuit_breakers(self, agent_ctx, mock_redis):
        """Manual scan should skip rules with tripped circuit breakers."""
        from src.services.ai_agent.agent import AIAgent

        mock_redis.exists = AsyncMock(return_value=False)  # kill switch OFF
        agent = AIAgent(agent_ctx)

        # Trip the circuit breaker for a rule
        for _ in range(agent_ctx.agent_config.safety_circuit_breaker_threshold):
            agent._safety.circuit_breaker.record_failure("system.high_cpu")

        assert agent._safety.should_skip_rule("system.high_cpu")

        # Run manual scan — the tripped rule should be skipped (not crash)
        result = await agent.run_manual_scan()
        assert isinstance(result, list)


# ── Cost Guardian Enforcer Rate Limit Tests ─────────────────────────────


class TestCostEnforcerRateLimit:

    @pytest.mark.asyncio
    async def test_enforcer_rate_limits_actions(self):
        """CostEnforcer should block actions after reaching the hourly cap."""
        from src.services.cost_guardian.enforcer import CostEnforcer
        from src.services.cost_guardian.analyzer import AlertLevel, CostAlert

        mock_redis = AsyncMock()
        enforcer = CostEnforcer(
            db=None, redis=mock_redis, max_actions_per_hour=2,
        )

        alert = CostAlert(
            level=AlertLevel.EMERGENCY,
            service="LLM",
            message="Hourly cost exceeded",
            current_value=25.0,
            threshold=20.0,
            action="pause_service",
        )

        # First 2 actions should succeed
        r1 = await enforcer.execute_action(alert)
        assert r1.get("status") != "rate_limited"

        r2 = await enforcer.execute_action(alert)
        assert r2.get("status") != "rate_limited"

        # 3rd action should be rate-limited
        r3 = await enforcer.execute_action(alert)
        assert r3["status"] == "rate_limited"

    @pytest.mark.asyncio
    async def test_enforcer_allows_after_window(self):
        """CostEnforcer should allow actions after the 1h window expires."""
        from src.services.cost_guardian.enforcer import CostEnforcer
        from src.services.cost_guardian.analyzer import AlertLevel, CostAlert

        mock_redis = AsyncMock()
        enforcer = CostEnforcer(
            db=None, redis=mock_redis, max_actions_per_hour=1,
        )

        alert = CostAlert(
            level=AlertLevel.EMERGENCY,
            service="LLM",
            message="test",
            current_value=25.0,
            threshold=20.0,
            action="pause_service",
        )

        # Use up the limit
        await enforcer.execute_action(alert)
        r2 = await enforcer.execute_action(alert)
        assert r2["status"] == "rate_limited"

        # Backdate the timestamp to simulate 1h passing
        enforcer._action_timestamps = [time.time() - 3601]

        r3 = await enforcer.execute_action(alert)
        assert r3.get("status") != "rate_limited"
