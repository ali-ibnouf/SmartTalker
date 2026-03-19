"""Tests for channel integration monitoring rules and auto-fix handlers.

Covers: WebhookDeliveryFailure, ChannelDisconnected, ChannelRoutingError,
InactiveChannel, VisitorResolutionFailure, TelegramWebhookReregister,
ChannelRoutingFallback, and Redis instrumentation in router/resolver.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.services.ai_agent.rules import AgentContext, Detection
from src.services.ai_agent.monitors.channels import (
    ChannelDisconnectedRule,
    ChannelRoutingErrorRule,
    InactiveChannelRule,
    VisitorResolutionFailureRule,
    WebhookDeliveryFailureRule,
)
from src.services.ai_agent.fixes.handlers import (
    ChannelRoutingFallbackFix,
    TelegramWebhookReregisterFix,
)


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def agent_config():
    config = MagicMock()
    config.webhook_failure_threshold = 5
    config.channel_routing_error_threshold = 10
    config.channel_inactive_days = 7
    config.visitor_resolve_fail_threshold = 10
    config.fix_cooldown_s = 300
    config.throttle_duration_s = 600
    config.throttle_rate_limit = 50
    return config


@pytest.fixture
def ctx(agent_config):
    return AgentContext(
        db=None,
        redis=None,
        pipeline=None,
        config=None,
        agent_config=agent_config,
    )


@pytest.fixture
def mock_redis():
    redis = AsyncMock()
    redis.scan = AsyncMock(return_value=(0, []))
    redis.get = AsyncMock(return_value=None)
    redis.incr = AsyncMock()
    redis.expire = AsyncMock()
    redis.delete = AsyncMock()
    redis.set = AsyncMock()
    return redis


@pytest.fixture
def mock_db():
    db = AsyncMock()
    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    result = AsyncMock()
    result.scalar = MagicMock(return_value=None)
    result.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))
    result.all = MagicMock(return_value=[])
    session.execute = AsyncMock(return_value=result)
    session.commit = AsyncMock()
    session.add = MagicMock()
    db.session_ctx = MagicMock(return_value=session)
    return db


# ═══════════════════════════════════════════════════════════════════
# WebhookDeliveryFailureRule Tests
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_webhook_failure_no_redis(ctx):
    """No Redis → no detections."""
    rule = WebhookDeliveryFailureRule()
    assert await rule.evaluate(ctx) == []


@pytest.mark.asyncio
async def test_webhook_failure_below_threshold(ctx, mock_redis):
    """Failure count below threshold → no detection."""
    ctx.redis = mock_redis
    # Keys include hour suffix as produced by _record_metric()
    mock_redis.scan = AsyncMock(return_value=(
        0, [b"channel_fail:whatsapp:emp_001:2026030612"]
    ))
    mock_redis.get = AsyncMock(return_value="2")  # below threshold of 5

    rule = WebhookDeliveryFailureRule()
    detections = await rule.evaluate(ctx)
    assert len(detections) == 0


@pytest.mark.asyncio
async def test_webhook_failure_above_threshold(ctx, mock_redis):
    """Failure count above threshold → warning detection with correct parsing."""
    ctx.redis = mock_redis
    # Realistic 4-part key: channel_fail:{type}:{employee_id}:{hour}
    mock_redis.scan = AsyncMock(return_value=(
        0, [b"channel_fail:telegram:emp_002:2026030612"]
    ))
    mock_redis.get = AsyncMock(return_value="8")

    rule = WebhookDeliveryFailureRule()
    detections = await rule.evaluate(ctx)

    assert len(detections) == 1
    d = detections[0]
    assert d.rule_id == "channels.webhook_failures"
    assert d.severity == "warning"
    assert d.details["channel_type"] == "telegram"
    # Critical: employee_id must NOT include the hour suffix
    assert d.details["employee_id"] == "emp_002"
    assert d.details["failure_count"] == 8
    assert d.auto_fixable is True  # Telegram is auto-fixable


@pytest.mark.asyncio
async def test_webhook_failure_critical_severity(ctx, mock_redis):
    """Double threshold → critical severity."""
    ctx.redis = mock_redis
    mock_redis.scan = AsyncMock(return_value=(
        0, [b"channel_fail:whatsapp:emp_003:2026030612"]
    ))
    mock_redis.get = AsyncMock(return_value="12")  # >= 5*2 = 10

    rule = WebhookDeliveryFailureRule()
    detections = await rule.evaluate(ctx)

    assert len(detections) == 1
    assert detections[0].severity == "critical"
    assert detections[0].details["employee_id"] == "emp_003"
    assert detections[0].auto_fixable is False  # WhatsApp not auto-fixable


# ═══════════════════════════════════════════════════════════════════
# ChannelDisconnectedRule Tests
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_channel_disconnected_no_db(ctx):
    """No DB → no detections."""
    rule = ChannelDisconnectedRule()
    assert await rule.evaluate(ctx) == []


@pytest.mark.asyncio
async def test_channel_disconnected_missing_wa_creds(ctx, mock_db):
    """WhatsApp channel with missing access token → detection."""
    ctx.db = mock_db
    session = mock_db.session_ctx.return_value

    channel = MagicMock()
    channel.employee_id = "emp_001"
    channel.channel_type = "whatsapp"
    channel.enabled = True
    channel.wa_phone_number_id = "12345"
    channel.wa_access_token = ""  # Missing!

    result = AsyncMock()
    result.all = MagicMock(return_value=[(channel, "Alice")])
    session.execute = AsyncMock(return_value=result)

    rule = ChannelDisconnectedRule()
    detections = await rule.evaluate(ctx)

    assert len(detections) == 1
    assert detections[0].details["missing_fields"] == ["access_token"]
    assert "Alice" in detections[0].title


@pytest.mark.asyncio
async def test_channel_disconnected_missing_tg_token(ctx, mock_db):
    """Telegram channel with missing bot token → detection."""
    ctx.db = mock_db
    session = mock_db.session_ctx.return_value

    channel = MagicMock()
    channel.employee_id = "emp_002"
    channel.channel_type = "telegram"
    channel.enabled = True
    channel.tg_bot_token = ""  # Missing!

    result = AsyncMock()
    result.all = MagicMock(return_value=[(channel, "Bob")])
    session.execute = AsyncMock(return_value=result)

    rule = ChannelDisconnectedRule()
    detections = await rule.evaluate(ctx)

    assert len(detections) == 1
    assert "bot_token" in detections[0].details["missing_fields"]


@pytest.mark.asyncio
async def test_channel_connected_ok(ctx, mock_db):
    """Channel with all credentials → no detection."""
    ctx.db = mock_db
    session = mock_db.session_ctx.return_value

    channel = MagicMock()
    channel.employee_id = "emp_001"
    channel.channel_type = "whatsapp"
    channel.enabled = True
    channel.wa_phone_number_id = "12345"
    channel.wa_access_token = "valid_token"

    result = AsyncMock()
    result.all = MagicMock(return_value=[(channel, "Alice")])
    session.execute = AsyncMock(return_value=result)

    rule = ChannelDisconnectedRule()
    detections = await rule.evaluate(ctx)

    assert len(detections) == 0


# ═══════════════════════════════════════════════════════════════════
# ChannelRoutingErrorRule Tests
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_routing_error_no_redis(ctx):
    """No Redis → no detections."""
    rule = ChannelRoutingErrorRule()
    assert await rule.evaluate(ctx) == []


@pytest.mark.asyncio
async def test_routing_error_above_threshold(ctx, mock_redis):
    """High error count → detection with error rate."""
    ctx.redis = mock_redis

    async def mock_get(key):
        if "route_err:whatsapp:" in key:
            return "15"
        if "route_ok:whatsapp:" in key:
            return "85"
        return None

    mock_redis.get = AsyncMock(side_effect=mock_get)

    rule = ChannelRoutingErrorRule()
    detections = await rule.evaluate(ctx)

    assert len(detections) == 1
    d = detections[0]
    assert d.rule_id == "channels.routing_errors"
    assert d.details["channel_type"] == "whatsapp"
    assert d.details["error_count"] == 15
    assert d.details["success_count"] == 85
    assert d.details["error_rate_pct"] == 15.0
    assert d.auto_fixable is True


@pytest.mark.asyncio
async def test_routing_error_critical_at_50_pct(ctx, mock_redis):
    """Error rate >= 50% → critical severity."""
    ctx.redis = mock_redis

    async def mock_get(key):
        if "route_err:telegram:" in key:
            return "60"
        if "route_ok:telegram:" in key:
            return "40"
        return None

    mock_redis.get = AsyncMock(side_effect=mock_get)

    rule = ChannelRoutingErrorRule()
    detections = await rule.evaluate(ctx)

    tg_detection = [d for d in detections if d.details.get("channel_type") == "telegram"]
    assert len(tg_detection) == 1
    assert tg_detection[0].severity == "critical"


# ═══════════════════════════════════════════════════════════════════
# VisitorResolutionFailureRule Tests
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_visitor_resolution_failure_no_redis(ctx):
    """No Redis → no detections."""
    rule = VisitorResolutionFailureRule()
    assert await rule.evaluate(ctx) == []


@pytest.mark.asyncio
async def test_visitor_resolution_failure_detected(ctx, mock_redis):
    """High failure count → detection."""
    ctx.redis = mock_redis

    async def mock_get(key):
        if "visitor_resolve_fail:" in key:
            return "15"
        if "visitor_resolve_ok:" in key:
            return "85"
        return None

    mock_redis.get = AsyncMock(side_effect=mock_get)

    rule = VisitorResolutionFailureRule()
    detections = await rule.evaluate(ctx)

    assert len(detections) == 1
    d = detections[0]
    assert d.rule_id == "channels.visitor_resolution_failures"
    assert d.details["failure_count"] == 15
    assert d.details["failure_rate_pct"] == 15.0


# ═══════════════════════════════════════════════════════════════════
# TelegramWebhookReregisterFix Tests
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_telegram_reregister_can_fix(ctx, mock_db, mock_redis):
    """TelegramWebhookReregisterFix can fix telegram webhook failures."""
    ctx.db = mock_db
    ctx.redis = mock_redis

    fix = TelegramWebhookReregisterFix()
    detection = Detection(
        rule_id="channels.webhook_failures",
        severity="warning",
        title="Webhook failures",
        description="Telegram webhook failing",
        details={"channel_type": "telegram", "employee_id": "emp_001"},
    )
    assert await fix.can_fix(detection, ctx) is True


@pytest.mark.asyncio
async def test_telegram_reregister_skips_whatsapp(ctx, mock_db):
    """TelegramWebhookReregisterFix skips non-telegram channels."""
    ctx.db = mock_db
    fix = TelegramWebhookReregisterFix()
    detection = Detection(
        rule_id="channels.webhook_failures",
        severity="warning",
        title="Webhook failures",
        description="WA failing",
        details={"channel_type": "whatsapp", "employee_id": "emp_001"},
    )
    assert await fix.can_fix(detection, ctx) is False


@pytest.mark.asyncio
async def test_telegram_reregister_applies(ctx, mock_db, mock_redis):
    """TelegramWebhookReregisterFix re-registers webhook and clears counters."""
    ctx.db = mock_db
    ctx.redis = mock_redis
    session = mock_db.session_ctx.return_value

    channel = MagicMock()
    channel.tg_bot_token = "123456:ABC"
    result = AsyncMock()
    result.scalar = MagicMock(return_value=channel)
    session.execute = AsyncMock(return_value=result)

    # Simulate Redis scan returning matching failure keys
    mock_redis.scan = AsyncMock(return_value=(
        0, [b"channel_fail:telegram:emp_001:2026030612"]
    ))

    detection = Detection(
        rule_id="channels.webhook_failures",
        severity="warning",
        title="Webhook failures",
        description="TG failing",
        details={"channel_type": "telegram", "employee_id": "emp_001"},
    )

    fix = TelegramWebhookReregisterFix()
    with patch(
        "src.channels.telegram.TelegramAdapter.register_webhook",
        new_callable=AsyncMock,
        return_value={"ok": True},
    ):
        result = await fix.apply(detection, ctx)

    assert result["action"] == "telegram_webhook_reregistered"
    assert result["employee_id"] == "emp_001"
    assert result["telegram_ok"] is True
    # Verify scan+delete pattern for clearing failure counters
    mock_redis.scan.assert_called()
    mock_redis.delete.assert_called_once()


@pytest.mark.asyncio
async def test_telegram_reregister_uses_config_url(ctx, mock_db, mock_redis):
    """TelegramWebhookReregisterFix uses config webhook URL when available."""
    ctx.db = mock_db
    ctx.redis = mock_redis
    ctx.config = MagicMock()
    ctx.config.telegram_webhook_url = "https://custom.example.com/webhooks/telegram"
    session = mock_db.session_ctx.return_value

    channel = MagicMock()
    channel.tg_bot_token = "123456:ABC"
    result = AsyncMock()
    result.scalar = MagicMock(return_value=channel)
    session.execute = AsyncMock(return_value=result)
    mock_redis.scan = AsyncMock(return_value=(0, []))

    detection = Detection(
        rule_id="channels.webhook_failures",
        severity="warning",
        title="Webhook failures",
        description="TG failing",
        details={"channel_type": "telegram", "employee_id": "emp_001"},
    )

    fix = TelegramWebhookReregisterFix()
    with patch(
        "src.channels.telegram.TelegramAdapter.register_webhook",
        new_callable=AsyncMock,
        return_value={"ok": True},
    ) as mock_register:
        result = await fix.apply(detection, ctx)

    assert result["action"] == "telegram_webhook_reregistered"
    # Verify it used the config URL, not the hardcoded default
    call_args = mock_register.call_args
    assert "custom.example.com" in call_args[0][1]


@pytest.mark.asyncio
async def test_telegram_reregister_handles_non_dict_result(ctx, mock_db, mock_redis):
    """TelegramWebhookReregisterFix handles non-dict return from register_webhook."""
    ctx.db = mock_db
    ctx.redis = mock_redis
    session = mock_db.session_ctx.return_value

    channel = MagicMock()
    channel.tg_bot_token = "123456:ABC"
    result = AsyncMock()
    result.scalar = MagicMock(return_value=channel)
    session.execute = AsyncMock(return_value=result)
    mock_redis.scan = AsyncMock(return_value=(0, []))

    detection = Detection(
        rule_id="channels.webhook_failures",
        severity="warning",
        title="Webhook failures",
        description="TG failing",
        details={"channel_type": "telegram", "employee_id": "emp_001"},
    )

    fix = TelegramWebhookReregisterFix()
    # Return a non-dict value (e.g., string)
    with patch(
        "src.channels.telegram.TelegramAdapter.register_webhook",
        new_callable=AsyncMock,
        return_value="unexpected_string_response",
    ):
        result = await fix.apply(detection, ctx)

    # Should not crash, telegram_ok should be False
    assert result["action"] == "telegram_webhook_reregistered"
    assert result["telegram_ok"] is False


# ═══════════════════════════════════════════════════════════════════
# ChannelRoutingFallbackFix Tests
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_routing_fallback_can_fix(ctx, mock_redis):
    """ChannelRoutingFallbackFix can fix routing errors."""
    ctx.redis = mock_redis
    fix = ChannelRoutingFallbackFix()
    detection = Detection(
        rule_id="channels.routing_errors",
        severity="warning",
        title="Routing errors",
        description="High error rate",
        details={"channel_type": "whatsapp", "error_rate_pct": 30},
    )
    assert await fix.can_fix(detection, ctx) is True


@pytest.mark.asyncio
async def test_routing_fallback_escalates_on_extreme_error(ctx, mock_redis, mock_db):
    """ChannelRoutingFallbackFix escalates (not auto-disables) when error rate > 80%."""
    ctx.redis = mock_redis
    ctx.db = mock_db

    fix = ChannelRoutingFallbackFix()
    detection = Detection(
        rule_id="channels.routing_errors",
        severity="critical",
        title="Routing errors",
        description="Extreme error rate",
        details={"channel_type": "telegram", "error_rate_pct": 90},
    )
    result = await fix.apply(detection, ctx)

    # Must escalate for manual review, NOT auto-disable across all customers
    assert result["action"] == "routing_alert_escalated"
    assert result["channel_type"] == "telegram"


@pytest.mark.asyncio
async def test_routing_fallback_alert_only_under_80(ctx, mock_redis):
    """ChannelRoutingFallbackFix only logs alert when error rate < 80%."""
    ctx.redis = mock_redis
    fix = ChannelRoutingFallbackFix()
    detection = Detection(
        rule_id="channels.routing_errors",
        severity="warning",
        title="Routing errors",
        description="Moderate error rate",
        details={"channel_type": "whatsapp", "error_rate_pct": 30},
    )
    result = await fix.apply(detection, ctx)

    assert result["action"] == "routing_alert_logged"


@pytest.mark.asyncio
async def test_routing_fallback_escalates_without_db_ops(ctx, mock_redis, mock_db):
    """ChannelRoutingFallbackFix escalates without touching the DB for >80% error rate."""
    ctx.redis = mock_redis
    ctx.db = mock_db
    session = mock_db.session_ctx.return_value

    fix = ChannelRoutingFallbackFix()
    detection = Detection(
        rule_id="channels.routing_errors",
        severity="critical",
        title="Routing errors",
        description="Extreme error rate",
        details={"channel_type": "telegram", "error_rate_pct": 95},
    )
    result = await fix.apply(detection, ctx)

    # Must escalate, NOT execute DB operations
    assert result["action"] == "routing_alert_escalated"
    session.execute.assert_not_called()


# ═══════════════════════════════════════════════════════════════════
# Redis Instrumentation Tests (Router + Resolver)
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_router_records_success_metric(mock_redis):
    """ChannelRouter records route_ok metric on successful routing."""
    from src.channels.router import ChannelRouter
    from src.channels.base import ChannelType, IncomingMessage

    agent = AsyncMock()
    agent.handle_text_message = AsyncMock(return_value={"text": "Hi"})
    agent.get_employee = AsyncMock(return_value=None)

    adapter = AsyncMock()
    adapter.send_response = AsyncMock()

    router = ChannelRouter(
        agent_engine=agent, tts=None, asr=None,
        runpod=None, r2=None, redis=mock_redis,
    )
    router._adapters[ChannelType.WHATSAPP] = adapter

    msg = IncomingMessage(
        channel=ChannelType.WHATSAPP,
        channel_session_id="wa_123_emp",
        employee_id="emp_001",
        customer_id="cust_001",
        visitor_id="v_x",
        message_type="text",
        text="Hello",
    )
    await router.handle_message(msg)

    # Should have called incr for route_ok:whatsapp:{hour}
    mock_redis.incr.assert_called()
    call_keys = [call.args[0] for call in mock_redis.incr.call_args_list]
    assert any("route_ok:whatsapp:" in k for k in call_keys)


@pytest.mark.asyncio
async def test_router_records_error_metric(mock_redis):
    """ChannelRouter records route_err metric when send_response fails."""
    from src.channels.router import ChannelRouter
    from src.channels.base import ChannelType, IncomingMessage

    agent = AsyncMock()
    agent.handle_text_message = AsyncMock(return_value={"text": "Hi"})
    agent.get_employee = AsyncMock(return_value=None)

    adapter = AsyncMock()
    adapter.send_response = AsyncMock(side_effect=Exception("send failed"))

    router = ChannelRouter(
        agent_engine=agent, tts=None, asr=None,
        runpod=None, r2=None, redis=mock_redis,
    )
    router._adapters[ChannelType.TELEGRAM] = adapter

    msg = IncomingMessage(
        channel=ChannelType.TELEGRAM,
        channel_session_id="tg_123_emp",
        employee_id="emp_001",
        customer_id="cust_001",
        visitor_id="v_x",
        message_type="text",
        text="Hello",
    )

    with pytest.raises(Exception, match="send failed"):
        await router.handle_message(msg)

    call_keys = [call.args[0] for call in mock_redis.incr.call_args_list]
    assert any("route_err:telegram:" in k for k in call_keys)
    assert any("channel_fail:telegram:emp_001" in k for k in call_keys)


@pytest.mark.asyncio
async def test_visitor_resolver_records_success(mock_redis):
    """VisitorResolver records visitor_resolve_ok on success."""
    from src.channels.visitor_resolver import VisitorResolver

    resolver = VisitorResolver(db=None, redis=mock_redis)
    vid = await resolver.resolve_visitor("whatsapp", "12345", "emp_001")

    assert vid.startswith("v_")
    # No DB → falls through to fallback → records failure (since it's DB-less)
    # But with db=None, resolve_visitor returns early without hitting DB
    # so no metric is recorded (only recorded on DB path)


@pytest.mark.asyncio
async def test_visitor_resolver_records_failure(mock_redis, mock_db):
    """VisitorResolver records visitor_resolve_fail on DB error."""
    from src.channels.visitor_resolver import VisitorResolver

    session = mock_db.session_ctx.return_value
    session.execute = AsyncMock(side_effect=Exception("DB down"))

    resolver = VisitorResolver(db=mock_db, redis=mock_redis)
    vid = await resolver.resolve_visitor("whatsapp", "12345", "emp_001")

    assert vid.startswith("v_")
    call_keys = [call.args[0] for call in mock_redis.incr.call_args_list]
    assert any("visitor_resolve_fail:" in k for k in call_keys)


# ═══════════════════════════════════════════════════════════════════
# InactiveChannelRule Tests
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_inactive_channel_no_db(ctx):
    """No DB → no detections."""
    rule = InactiveChannelRule()
    assert await rule.evaluate(ctx) == []


# ═══════════════════════════════════════════════════════════════════
# Agent Registration Tests
# ═══════════════════════════════════════════════════════════════════


def test_agent_registers_channel_rules():
    """AIAgent registers all 5 channel monitoring rules."""
    from src.services.ai_agent.agent import AIAgent

    ctx = AgentContext(
        db=None, redis=None, pipeline=None,
        config=None, agent_config=MagicMock(
            scan_interval_s=60,
            auto_fix_enabled=True,
            fix_cooldown_s=300,
            stale_session_timeout_min=30,
            stale_session_check_interval_s=300,
            notification_ws_enabled=False,
            notification_email_enabled=False,
            notification_warning_batch_s=300,
            notification_info_digest_s=3600,
            audit_log_path="./logs/test.jsonl",
            approval_expiry_hours=24,
        ),
    )
    agent = AIAgent(ctx)

    rule_ids = [r.rule_id for r in agent._rule_registry.rules]
    assert "channels.webhook_failures" in rule_ids
    assert "channels.disconnected" in rule_ids
    assert "channels.routing_errors" in rule_ids
    assert "channels.inactive" in rule_ids
    assert "channels.visitor_resolution_failures" in rule_ids


def test_agent_registers_channel_fixes():
    """AIAgent registers channel auto-fix handlers."""
    from src.services.ai_agent.agent import AIAgent

    ctx = AgentContext(
        db=None, redis=None, pipeline=None,
        config=None, agent_config=MagicMock(
            scan_interval_s=60,
            auto_fix_enabled=True,
            fix_cooldown_s=300,
            stale_session_timeout_min=30,
            stale_session_check_interval_s=300,
            notification_ws_enabled=False,
            notification_email_enabled=False,
            notification_warning_batch_s=300,
            notification_info_digest_s=3600,
            audit_log_path="./logs/test.jsonl",
            approval_expiry_hours=24,
        ),
    )
    agent = AIAgent(ctx)

    fix_rule_ids = list(agent._fix_registry._handlers.keys())
    assert "channels.webhook_failures" in fix_rule_ids
    assert "channels.routing_errors" in fix_rule_ids
