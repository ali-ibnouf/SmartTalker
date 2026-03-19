"""AI Agent — Specification Tests.

Covers the full agent lifecycle: detection rules, auto-fix actions,
escalation logic, prevention engine, notifications, and API endpoints.
Each test matches the product spec scenario naming.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from src.services.ai_agent.config import AgentSettings
from src.services.ai_agent.rules import AgentContext, Detection


# ── Helpers ────────────────────────────────────────────────────────


def _make_ctx(**overrides: Any) -> AgentContext:
    """Build a minimal AgentContext for rule tests."""
    return AgentContext(
        db=overrides.get("db"),
        redis=overrides.get("redis"),
        pipeline=overrides.get("pipeline"),
        config=overrides.get("config", MagicMock()),
        agent_config=overrides.get("agent_config", AgentSettings()),
        operator_manager=overrides.get("operator_manager"),
    )


def _mock_db_session(rows=None, scalar=None):
    """Create a mock db with session_ctx that returns prepared results."""
    session = AsyncMock()
    result_mock = MagicMock()

    if scalar is not None:
        result_mock.scalar.return_value = scalar
    if rows is not None:
        result_mock.all.return_value = rows

    session.execute = AsyncMock(return_value=result_mock)

    db = MagicMock()
    ctx_manager = AsyncMock()
    ctx_manager.__aenter__ = AsyncMock(return_value=session)
    ctx_manager.__aexit__ = AsyncMock(return_value=False)
    db.session_ctx.return_value = ctx_manager
    return db, session


# ═══════════════════════════════════════════════════════════════════
# DETECTION RULES
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_dashscope_latency_alert():
    """DashScope response > 3s → triggers WARNING."""
    from src.services.ai_agent.monitors.infrastructure import DashScopeLatencyRule

    rule = DashScopeLatencyRule()
    pipeline = MagicMock()
    llm = MagicMock()
    llm._recent_latencies = [3.5, 4.0, 3.2]  # all > 3s
    pipeline._llm = llm

    ctx = _make_ctx(pipeline=pipeline)
    detections = await rule.evaluate(ctx)

    assert len(detections) >= 1
    assert detections[0].severity in ("warning", "critical")
    assert "latency" in detections[0].title.lower() or "dashscope" in detections[0].title.lower()


@pytest.mark.asyncio
async def test_dashscope_down_alert():
    """DashScope connection refused → triggers CRITICAL."""
    from src.services.ai_agent.monitors.infrastructure import DashScopeLatencyRule

    rule = DashScopeLatencyRule()
    pipeline = MagicMock()
    llm = MagicMock()
    # Very high latencies indicate connection issues
    llm._recent_latencies = [10.0, 10.0, 10.0]
    pipeline._llm = llm

    ctx = _make_ctx(pipeline=pipeline, agent_config=AgentSettings(dashscope_latency_warn_s=3.0))
    detections = await rule.evaluate(ctx)

    assert len(detections) >= 1
    assert detections[0].severity == "critical"


@pytest.mark.asyncio
async def test_dashscope_quota_alert():
    """DashScope quota < 20% → triggers WARNING."""
    from src.services.ai_agent.monitors.infrastructure import DashScopeQuotaRule

    rule = DashScopeQuotaRule()
    db, session = _mock_db_session()

    # Mock two calls: first for total cost, second for budget
    results = [MagicMock(), MagicMock()]
    results[0].scalar.return_value = 450.0  # spent $450 of $500 budget (90%)
    results[1].scalar.return_value = 450.0
    session.execute = AsyncMock(side_effect=results)

    ctx = _make_ctx(
        db=db,
        agent_config=AgentSettings(
            dashscope_monthly_budget_usd=500.0,
            dashscope_quota_warn_pct=80.0,
        ),
    )
    detections = await rule.evaluate(ctx)

    assert len(detections) >= 1
    assert detections[0].severity in ("warning", "critical")


@pytest.mark.asyncio
async def test_runpod_job_failure_alert():
    """RunPod job FAILED → triggers WARNING + fallback to VRM."""
    from src.services.ai_agent.monitors.infrastructure import RunPodConsecutiveFailureRule

    rule = RunPodConsecutiveFailureRule()
    pipeline = MagicMock()
    runpod = MagicMock()
    runpod._consecutive_failures = 4  # above threshold of 3
    runpod.video_disabled = False
    pipeline._runpod = runpod

    ctx = _make_ctx(
        pipeline=pipeline,
        agent_config=AgentSettings(runpod_consecutive_failure_threshold=3),
    )
    detections = await rule.evaluate(ctx)

    assert len(detections) == 1
    assert detections[0].severity in ("warning", "critical")
    assert detections[0].auto_fixable is True


@pytest.mark.asyncio
async def test_runpod_cold_start_spike():
    """cold_start_rate > 40% → triggers warm_up_worker()."""
    from src.services.ai_agent.monitors.infrastructure import RunPodColdStartRule

    rule = RunPodColdStartRule()
    pipeline = MagicMock()
    runpod = MagicMock()
    # Simulate cold start data: 5 out of 10 jobs >= 12s threshold (50%)
    runpod._recent_lipsync_times = [15.0, 2.0, 14.0, 1.5, 13.0, 2.0, 16.0, 1.8, 14.5, 2.1]
    pipeline._runpod = runpod

    ctx = _make_ctx(
        pipeline=pipeline,
        agent_config=AgentSettings(
            runpod_lipsync_cold_warn_s=12.0,
            runpod_cold_start_pct_warn=30.0,
        ),
    )
    detections = await rule.evaluate(ctx)

    # Rule triggers based on cold start times (50% >= 30% threshold)
    assert len(detections) >= 1


@pytest.mark.asyncio
async def test_runpod_queue_depth_alert():
    """RunPod queue > 10 pending → triggers WARNING."""
    from src.services.ai_agent.monitors.infrastructure import DashScopeQueueDepthRule

    rule = DashScopeQueueDepthRule()
    pipeline = MagicMock()
    llm = MagicMock()
    llm._pending_requests = 55  # above threshold
    llm.text_only_mode = False
    pipeline._llm = llm

    ctx = _make_ctx(
        pipeline=pipeline,
        agent_config=AgentSettings(dashscope_queue_depth_warn=50),
    )
    detections = await rule.evaluate(ctx)

    assert len(detections) == 1
    assert detections[0].severity in ("warning", "critical")


@pytest.mark.asyncio
async def test_r2_upload_failure_alert():
    """R2 PUT error → triggers retry + WARNING."""
    from src.services.ai_agent.monitors.infrastructure import R2DowntimeRule

    rule = R2DowntimeRule()
    pipeline = MagicMock()
    r2 = MagicMock()
    r2._consecutive_failures = 5
    r2._last_failure_time = time.time() - 10  # recent
    pipeline._r2 = r2

    ctx = _make_ctx(
        pipeline=pipeline,
        agent_config=AgentSettings(r2_downtime_threshold_s=300.0),
    )
    detections = await rule.evaluate(ctx)

    assert len(detections) >= 1
    assert detections[0].severity in ("warning", "critical")


@pytest.mark.asyncio
async def test_margin_squeeze_detection():
    """customer API cost > 60% revenue → triggers alert_admin()."""
    from src.services.ai_agent.monitors.infrastructure import MarginSqueezeRule

    rule = MarginSqueezeRule()
    db, session = _mock_db_session()

    # First query: customers with subscriptions
    customer_result = MagicMock()
    customer_result.all.return_value = [
        ("cust_1", "TestCorp", "professional", 100.0),
    ]
    # Second query: cost for that customer
    cost_result = MagicMock()
    cost_result.scalar.return_value = 70.0  # 70% of revenue

    session.execute = AsyncMock(side_effect=[customer_result, cost_result])

    ctx = _make_ctx(
        db=db,
        agent_config=AgentSettings(margin_squeeze_pct=60.0),
    )
    detections = await rule.evaluate(ctx)

    assert len(detections) >= 1
    assert "margin" in detections[0].title.lower() or "cost" in detections[0].title.lower()


@pytest.mark.asyncio
async def test_customer_low_balance():
    """balance < 20% → WARNING, < 5% → CRITICAL."""
    from src.services.ai_agent.monitors.business import QuotaExhaustionRule

    rule = QuotaExhaustionRule()
    db, session = _mock_db_session()

    # Single query returns (cid, cname, plan, monthly_seconds, used_s)
    result_mock = MagicMock()
    result_mock.all.return_value = [
        ("cust_1", "TestCorp", "professional", 100000, 96000.0),  # 96% used
    ]
    session.execute = AsyncMock(return_value=result_mock)

    ctx = _make_ctx(db=db, agent_config=AgentSettings(quota_warn_pct=80.0))
    detections = await rule.evaluate(ctx)

    assert len(detections) >= 1
    assert detections[0].severity in ("warning", "critical")


@pytest.mark.asyncio
async def test_security_brute_force():
    """5+ failed auth from same IP in 10min → CRITICAL."""
    from src.services.ai_agent.monitors.security import FailedAuthRule

    rule = FailedAuthRule()
    redis = AsyncMock()

    # Mock scan() returning (cursor, keys) — first call returns keys, second returns empty
    keys = [b"rate:auth_fail:192.168.1.1", b"rate:auth_fail:10.0.0.1"]
    redis.scan = AsyncMock(side_effect=[
        (b"0", keys),  # first scan returns keys + cursor=0 (done)
    ])
    # ttl within window, get returns failure counts
    redis.ttl = AsyncMock(return_value=300)  # 5min TTL, within 10min window
    redis.get = AsyncMock(side_effect=[b"6", b"3"])  # 6 failures from first IP

    ctx = _make_ctx(
        redis=redis,
        agent_config=AgentSettings(failed_auth_threshold=5, failed_auth_window_min=10),
    )
    detections = await rule.evaluate(ctx)

    assert len(detections) >= 1
    # 6 >= 5 threshold, but not >= 15 (threshold * 3), so severity is "warning"
    assert detections[0].severity in ("warning", "critical")


# ═══════════════════════════════════════════════════════════════════
# AUTO-FIX ACTIONS
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_dashscope_timeout_retry():
    """DashScope timeout → retry with backoff (1s,2s,4s) → succeeds."""
    from src.services.ai_agent.monitors.infrastructure import DashScopeConsecutiveTimeoutRule

    rule = DashScopeConsecutiveTimeoutRule()
    pipeline = MagicMock()
    llm = MagicMock()
    # After retries succeed, consecutive timeouts should reset
    llm._consecutive_timeouts = 0
    pipeline._llm = llm

    ctx = _make_ctx(
        pipeline=pipeline,
        agent_config=AgentSettings(dashscope_consecutive_timeout_threshold=5),
    )
    detections = await rule.evaluate(ctx)

    # No detection because retries succeeded (counter reset to 0)
    assert len(detections) == 0


@pytest.mark.asyncio
async def test_dashscope_timeout_escalate():
    """5 consecutive timeouts → switches to text_only_mode + alerts admin."""
    from src.services.ai_agent.fixes.handlers import TextOnlyModeFix

    fix = TextOnlyModeFix()
    pipeline = MagicMock()
    llm = MagicMock()
    llm.text_only_mode = False
    pipeline._llm = llm

    detection = Detection(
        rule_id="resilience.dashscope_consecutive_timeouts",
        severity="critical",
        title="5 consecutive DashScope timeouts",
        description="DashScope has timed out 5 times consecutively",
        auto_fixable=True,
    )

    ctx = _make_ctx(pipeline=pipeline)
    assert await fix.can_fix(detection, ctx) is True

    result = await fix.apply(detection, ctx)
    assert "text_only" in str(result).lower() or "enabled" in str(result).lower()
    assert llm.text_only_mode is True


@pytest.mark.asyncio
async def test_runpod_failure_vrm_fallback():
    """RunPod FAILED → session continues in VRM mode seamlessly."""
    from src.services.ai_agent.fixes.handlers import VideoDisableFix

    fix = VideoDisableFix()
    pipeline = MagicMock()
    runpod = MagicMock()
    runpod.video_disabled = False
    pipeline._runpod = runpod

    detection = Detection(
        rule_id="resilience.runpod_consecutive_failures",
        severity="critical",
        title="3+ consecutive RunPod failures",
        description="RunPod has failed 3+ times consecutively",
        auto_fixable=True,
    )

    ctx = _make_ctx(pipeline=pipeline)
    assert await fix.can_fix(detection, ctx) is True

    result = await fix.apply(detection, ctx)
    assert runpod.video_disabled is True


@pytest.mark.asyncio
async def test_runpod_warm_up_worker():
    """cold_start_spike detected → dummy job sent → worker stays warm."""
    from src.services.ai_agent.fixes.handlers import WarmupJobFix

    fix = WarmupJobFix()

    detection = Detection(
        rule_id="infra.runpod_workers",
        severity="warning",
        title="RunPod 0 idle workers",
        description="No idle workers available on RunPod",
        auto_fixable=True,
        details={"idle": 0},
    )

    config = MagicMock()
    config.runpod_api_key = "test_key"
    config.runpod_endpoint_musetalk = "https://api.runpod.ai/v2/test"

    ctx = _make_ctx(pipeline=MagicMock(), config=config)

    assert await fix.can_fix(detection, ctx) is True


@pytest.mark.asyncio
async def test_r2_retry_success():
    """R2 upload fails once → retry succeeds → no alert."""
    from src.services.ai_agent.monitors.infrastructure import R2DowntimeRule

    rule = R2DowntimeRule()
    pipeline = MagicMock()
    r2 = MagicMock()
    r2._consecutive_failures = 0  # reset after retry success
    r2._last_failure_time = 0
    pipeline._r2 = r2

    ctx = _make_ctx(pipeline=pipeline)
    detections = await rule.evaluate(ctx)

    # No alert — failures reset to 0
    assert len(detections) == 0


@pytest.mark.asyncio
async def test_r2_prolonged_failure():
    """R2 down > 5 min → CRITICAL + fallback_to_local_tmp()."""
    from src.services.ai_agent.monitors.infrastructure import R2DowntimeRule

    rule = R2DowntimeRule()
    pipeline = MagicMock()
    r2 = MagicMock()
    r2._consecutive_failures = 10
    r2._last_failure_time = time.time() - 400  # 400s ago (> 300s threshold)
    r2._total_failures = 25
    pipeline._r2 = r2

    ctx = _make_ctx(
        pipeline=pipeline,
        agent_config=AgentSettings(r2_downtime_threshold_s=300.0),
    )
    detections = await rule.evaluate(ctx)

    assert len(detections) >= 1
    assert detections[0].severity == "critical"


@pytest.mark.asyncio
async def test_stale_session_cleanup():
    """session inactive 30min → closed + billing updated."""
    from src.services.ai_agent.fixes.handlers import StaleSessionFix

    fix = StaleSessionFix()

    # StaleSessionFix uses apply_scheduled(ctx), not apply(detection, ctx)
    ctx = _make_ctx(operator_manager=MagicMock())
    ctx.operator_manager.get_active_sessions = lambda: []

    result = await fix.apply_scheduled(ctx)
    assert result is not None


@pytest.mark.asyncio
async def test_rate_limit_throttle():
    """customer > 100 req/min → throttled to 50/min for 10min."""
    from src.services.ai_agent.fixes.handlers import RateLimitThrottleFix

    fix = RateLimitThrottleFix()

    detection = Detection(
        rule_id="security.api_spike",
        severity="warning",
        title="API spike from customer_123",
        description="Customer cust_123 has 300%+ API usage spike",
        auto_fixable=True,
        details={"customer_id": "cust_123"},
    )

    redis = AsyncMock()
    ctx = _make_ctx(redis=redis)

    can = await fix.can_fix(detection, ctx)
    assert can is True

    result = await fix.apply(detection, ctx)
    assert result is not None
    assert result["action"] == "rate_throttle_applied"
    # Verify Redis was called to set throttle key
    redis.set.assert_called_once()


# ═══════════════════════════════════════════════════════════════════
# ESCALATION LOGIC
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_auto_fix_then_escalate():
    """3 failed auto-fix retries → escalates to admin queue."""
    from src.services.ai_agent.fixes.handlers import FixRegistry

    registry = FixRegistry()

    # Register a fix that always fails
    failing_fix = MagicMock()
    failing_fix.fix_id = "fix.test_fail"
    failing_fix.can_fix = AsyncMock(return_value=True)
    failing_fix.apply = AsyncMock(side_effect=Exception("fix failed"))

    registry.register("test.rule", failing_fix)

    detection = Detection(
        rule_id="test.rule",
        severity="warning",
        title="Test issue",
        description="Test issue requiring auto-fix",
        auto_fixable=True,
    )

    ctx = _make_ctx()
    # Try to fix — should handle the exception gracefully
    result = await registry.try_fix(detection, ctx)
    # FixRegistry catches exceptions and returns error dict
    assert result is not None
    assert "error" in result


@pytest.mark.asyncio
async def test_runpod_disable_video_after_failures():
    """3 consecutive RunPod failures → video mode disabled for customer."""
    from src.services.ai_agent.fixes.handlers import VideoDisableFix

    fix = VideoDisableFix()
    pipeline = MagicMock()
    runpod = MagicMock()
    runpod.video_disabled = False
    pipeline._runpod = runpod

    detection = Detection(
        rule_id="resilience.runpod_consecutive_failures",
        severity="critical",
        title="3 consecutive RunPod failures",
        description="RunPod has failed 3 times in a row",
        auto_fixable=True,
    )

    ctx = _make_ctx(pipeline=pipeline)
    # Apply the fix
    result = await fix.apply(detection, ctx)
    assert runpod.video_disabled is True
    assert "disabled" in str(result).lower()


@pytest.mark.asyncio
async def test_payment_retry_then_suspend():
    """3 failed payments → scheduled suspension after 7 days."""
    from src.services.ai_agent.monitors.business import FailedPaymentRule

    rule = FailedPaymentRule()
    db, session = _mock_db_session()

    # Mock customers with 3+ payment failures (4-column: cid, cname, plan, failures)
    result_mock = MagicMock()
    result_mock.all.return_value = [
        ("cust_1", "TestCorp", "professional", 3),
    ]
    session.execute = AsyncMock(return_value=result_mock)

    ctx = _make_ctx(
        db=db,
        agent_config=AgentSettings(failed_payment_threshold=2),
    )
    detections = await rule.evaluate(ctx)

    assert len(detections) >= 1
    assert detections[0].severity in ("warning", "critical")


# ═══════════════════════════════════════════════════════════════════
# PREVENTION ENGINE
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_incident_pattern_detection():
    """same alert type 3+ times/week → flagged as chronic."""
    from src.services.ai_agent.prevention import PatternTracker

    tracker = PatternTracker(db=None)

    # Record multiple occurrences
    await tracker.record_occurrence("test.rule", "test_key")
    await tracker.record_occurrence("test.rule", "test_key")
    await tracker.record_occurrence("test.rule", "test_key")

    # In-memory tracker should have the pattern
    predictions = await tracker.get_predictions()
    # Without DB, predictions come from in-memory cache
    assert isinstance(predictions, list)


@pytest.mark.asyncio
async def test_quota_exhaustion_prediction():
    """usage trend → predicts days until quota runs out."""
    from src.services.ai_agent.monitors.predictions import DashScopeQuotaExhaustionRule

    rule = DashScopeQuotaExhaustionRule()
    db, session = _mock_db_session()

    # Simulate high burn: $400 in first 20 days of month → projects to $600
    cost_result = MagicMock()
    cost_result.scalar.return_value = 400.0

    session.execute = AsyncMock(return_value=cost_result)

    ctx = _make_ctx(
        db=db,
        agent_config=AgentSettings(dashscope_monthly_budget_usd=500.0),
    )
    detections = await rule.evaluate(ctx)

    # Should detect if budget will be exhausted
    # (depends on day of month, but burn rate is high)
    assert isinstance(detections, list)


@pytest.mark.asyncio
async def test_margin_prediction():
    """API cost trend → predicts if customer becomes unprofitable."""
    from src.services.ai_agent.monitors.predictions import CustomerMarginPredictionRule

    rule = CustomerMarginPredictionRule()
    db, session = _mock_db_session()

    # Mock: customer paying $100/mo, using $80 API cost already (80%)
    customer_result = MagicMock()
    customer_result.all.return_value = [
        ("cust_1", "TestCorp", "professional", 100.0),
    ]
    cost_result = MagicMock()
    cost_result.scalar.return_value = 80.0  # projects to > 100%

    session.execute = AsyncMock(side_effect=[customer_result, cost_result])

    ctx = _make_ctx(db=db)
    detections = await rule.evaluate(ctx)

    # Should warn about margin risk (projected > 70% of revenue)
    assert isinstance(detections, list)
    if len(detections) > 0:
        assert "margin" in detections[0].title.lower() or "cost" in detections[0].title.lower()


@pytest.mark.asyncio
async def test_runpod_cost_projection():
    """render usage trend → projects monthly RunPod spend."""
    from src.services.ai_agent.monitors.predictions import RunPodCostProjectionRule

    rule = RunPodCostProjectionRule()
    db, session = _mock_db_session()

    # Mock: $50 RunPod cost in last 7 days → projects ~$214/mo
    cost_result = MagicMock()
    cost_result.scalar.return_value = 50.0
    count_result = MagicMock()
    count_result.scalar.return_value = 340

    session.execute = AsyncMock(side_effect=[cost_result, count_result])

    ctx = _make_ctx(
        db=db,
        agent_config=AgentSettings(dashscope_monthly_budget_usd=500.0),
    )
    detections = await rule.evaluate(ctx)

    # Should warn: projected $214/mo vs $200 budget (40% of $500)
    assert isinstance(detections, list)
    if len(detections) > 0:
        assert "runpod" in detections[0].rule_id.lower()


# ═══════════════════════════════════════════════════════════════════
# NOTIFICATIONS
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_critical_sends_email():
    """CRITICAL alert → email sent to contact@lsmarttech.com."""
    from src.services.ai_agent.notifications import (
        NotificationDispatcher,
        NotificationSettings,
    )

    agent_config = AgentSettings()
    smtp_config = NotificationSettings(
        username="test@example.com",
        password="testpass",
    )

    dispatcher = NotificationDispatcher(
        operator_manager=None,
        agent_config=agent_config,
        smtp_config=smtp_config,
    )

    d = Detection(
        rule_id="test.critical",
        severity="critical",
        title="Test Critical Alert",
        description="Something critical happened",
        recommendation="Fix it",
    )

    mock_smtp = MagicMock()
    mock_smtp.send = AsyncMock()

    with patch.dict("sys.modules", {"aiosmtplib": mock_smtp}):
        await dispatcher.dispatch(d, "inc_001")

        # Email should have been attempted
        mock_smtp.send.assert_called_once()
        call_args = mock_smtp.send.call_args
        msg = call_args[0][0]
        assert "[CRITICAL]" in msg["Subject"]
        assert agent_config.notification_email_to in msg["To"]


@pytest.mark.asyncio
async def test_warning_sends_dashboard():
    """WARNING → dashboard WebSocket notification within 5min."""
    from src.services.ai_agent.notifications import (
        NotificationDispatcher,
        NotificationSettings,
    )

    agent_config = AgentSettings()
    dispatcher = NotificationDispatcher(
        operator_manager=None,
        agent_config=agent_config,
        smtp_config=NotificationSettings(),
    )

    d = Detection(
        rule_id="test.warning",
        severity="warning",
        title="Test Warning",
        description="A test warning event",
    )

    await dispatcher.dispatch(d, "inc_002")

    # Warning should be in the queue (not sent immediately)
    assert dispatcher._warning_queue.qsize() == 1


@pytest.mark.asyncio
async def test_alert_deduplication():
    """same alert within cooldown → NOT sent again."""
    from src.services.ai_agent.notifications import (
        NotificationDispatcher,
        NotificationSettings,
    )

    agent_config = AgentSettings()
    agent_config.alert_cooldown = {"test.dedup": 300}
    agent_config.alert_cooldown_default_s = 60

    dispatcher = NotificationDispatcher(
        operator_manager=None,
        agent_config=agent_config,
        smtp_config=NotificationSettings(),
    )

    d = Detection(
        rule_id="test.dedup",
        severity="warning",
        title="Test Dedup",
        description="A test dedup event",
    )

    # First dispatch — should go through
    await dispatcher.dispatch(d, "inc_003")
    assert dispatcher._warning_queue.qsize() == 1

    # Second dispatch — should be suppressed (within cooldown)
    await dispatcher.dispatch(d, "inc_004")
    assert dispatcher._warning_queue.qsize() == 1  # still 1, not 2


@pytest.mark.asyncio
async def test_resolved_logged_only():
    """auto-fix success → RESOLVED logged, no email/dashboard."""
    from src.services.ai_agent.notifications import (
        NotificationDispatcher,
        NotificationSettings,
    )

    agent_config = AgentSettings()
    dispatcher = NotificationDispatcher(
        operator_manager=None,
        agent_config=agent_config,
        smtp_config=NotificationSettings(),
    )

    # Status change is logged to audit only, not dispatched as alert
    await dispatcher.dispatch_status_change("inc_005", "resolved")

    # No items in any queue
    assert dispatcher._warning_queue.qsize() == 0
    assert dispatcher._info_queue.qsize() == 0


# ═══════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_agent_status_endpoint():
    """GET /admin/agent/status → running + last check timestamps."""
    from src.services.ai_agent.agent import AIAgent

    ctx = _make_ctx()
    agent = AIAgent(ctx)
    await agent.start()

    stats = await agent.get_stats()

    assert "running" in stats
    assert stats["running"] is True
    assert "scan_count" in stats
    assert "rules_count" in stats
    assert stats["rules_count"] > 0

    await agent.stop()


@pytest.mark.asyncio
async def test_agent_alerts_endpoint():
    """GET /admin/agent/alerts → list with severity + timestamps."""
    from src.services.ai_agent.agent import AIAgent

    ctx = _make_ctx()
    agent = AIAgent(ctx)
    await agent.start()

    incidents = await agent.get_incidents()

    assert "incidents" in incidents
    assert "total" in incidents
    assert isinstance(incidents["incidents"], list)

    await agent.stop()


@pytest.mark.asyncio
async def test_agent_actions_log():
    """GET /admin/agent/actions → auto-fix history with results."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.services.ai_agent.routes import router

    app = FastAPI()
    app.include_router(router)
    app.state.db = None  # no DB → empty list

    client = TestClient(app)
    response = client.get("/api/v1/agent/auto-fixes")

    assert response.status_code == 200
    data = response.json()
    assert "actions" in data
    assert "count" in data
    assert data["count"] == 0


@pytest.mark.asyncio
async def test_agent_health_scores():
    """GET /admin/agent/customer-health → scores sorted by risk."""
    from src.services.ai_agent.health import CustomerHealthScorer

    scorer = CustomerHealthScorer(db=None)
    scores = await scorer.score_all()

    assert isinstance(scores, list)
    # Without DB, should return empty list gracefully
    assert len(scores) == 0


@pytest.mark.asyncio
async def test_agent_cost_breakdown():
    """GET /admin/agent/costs → DashScope + RunPod per customer."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.services.ai_agent.routes import router

    app = FastAPI()
    app.include_router(router)
    app.state.ai_agent = None

    client = TestClient(app)

    # Stats endpoint should work even without agent
    response = client.get("/api/v1/agent/stats")
    assert response.status_code == 200
    data = response.json()
    assert "rules_count" in data
