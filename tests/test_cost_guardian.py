"""Tests for Cost Guardian — budget checks, anomaly detection, enforcer, reporter, guardian loop."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.cost_guardian.config import (
    ANOMALY_CONFIG,
    BUDGETS,
    CUSTOMER_COST_RATIO,
    ServiceBudget,
)
from src.services.cost_guardian.analyzer import AlertLevel, CostAlert, CostAnalyzer
from src.services.cost_guardian.enforcer import CostEnforcer
from src.services.cost_guardian.guardian import CostGuardian
from src.services.cost_guardian.monitor import CostMonitor
from src.services.cost_guardian.reporter import CostReporter
from src.utils.exceptions import ServicePausedError


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture
def mock_monitor():
    """CostMonitor with all methods mocked."""
    m = MagicMock(spec=CostMonitor)
    m.db = MagicMock()
    # Default: everything returns 0 / empty
    m.get_hourly_spend = AsyncMock(return_value=0.0)
    m.get_daily_spend = AsyncMock(return_value=0.0)
    m.get_monthly_spend = AsyncMock(return_value=0.0)
    m.get_total_monthly_spend = AsyncMock(return_value={})
    m.get_customer_spend = AsyncMock(return_value=0.0)
    m.get_request_rate = AsyncMock(return_value=0)
    m.get_customer_request_rate = AsyncMock(return_value=0)
    m.get_average_hourly_spend = AsyncMock(return_value=0.0)
    m.get_last_request_cost = AsyncMock(return_value=None)
    m.get_runpod_active_jobs = AsyncMock(return_value=[])
    m.get_top_spending_customers = AsyncMock(return_value=[])
    m.get_recent_active_customer_ids = AsyncMock(return_value=[])
    m.get_zero_cost_counts = AsyncMock(return_value=[])
    return m


@pytest.fixture
def analyzer(mock_monitor):
    return CostAnalyzer(mock_monitor)


@pytest.fixture
def mock_redis():
    r = AsyncMock()
    r.set = AsyncMock()
    r.get = AsyncMock(return_value=None)
    r.exists = AsyncMock(return_value=False)
    r.delete = AsyncMock()
    return r


@pytest.fixture
def enforcer(mock_redis):
    return CostEnforcer(db=None, redis=mock_redis)


@pytest.fixture
def reporter(mock_monitor):
    return CostReporter(resend_api_key=None, monitor=mock_monitor)


# ═══════════════════════════════════════════════════════════════════════
# Config Tests
# ═══════════════════════════════════════════════════════════════════════


class TestConfig:
    def test_budgets_defined(self):
        assert "llm" in BUDGETS
        assert "asr" in BUDGETS
        assert "tts" in BUDGETS
        assert "gpu_render" in BUDGETS
        assert "gpu_preprocess" in BUDGETS
        assert "voice_clone" in BUDGETS

    def test_budget_fields(self):
        llm = BUDGETS["llm"]
        assert isinstance(llm, ServiceBudget)
        assert llm.hourly_limit < llm.hourly_kill
        assert llm.daily_limit < llm.daily_kill
        assert llm.monthly_limit < llm.monthly_kill

    def test_customer_cost_ratio_order(self):
        assert CUSTOMER_COST_RATIO["warning"] < CUSTOMER_COST_RATIO["critical"]
        assert CUSTOMER_COST_RATIO["critical"] < CUSTOMER_COST_RATIO["kill"]

    def test_anomaly_config(self):
        assert ANOMALY_CONFIG["spike_multiplier"] > 1
        assert ANOMALY_CONFIG["rapid_fire_threshold"] > 0


# ═══════════════════════════════════════════════════════════════════════
# Budget Check Tests
# ═══════════════════════════════════════════════════════════════════════


class TestBudgetChecks:
    @pytest.mark.asyncio
    async def test_no_alerts_when_under_budget(self, analyzer, mock_monitor):
        mock_monitor.get_hourly_spend.return_value = 1.0
        mock_monitor.get_daily_spend.return_value = 5.0
        mock_monitor.get_monthly_spend.return_value = 20.0

        alerts = await analyzer.run_full_analysis()
        budget_alerts = [a for a in alerts if "budget" in a.service.lower() or "DashScope" in a.service]
        # Should have no emergency/critical alerts for LLM at these values
        emergency = [a for a in alerts if a.level == AlertLevel.EMERGENCY]
        assert len(emergency) == 0

    @pytest.mark.asyncio
    async def test_hourly_warning_triggers(self, analyzer, mock_monitor):
        mock_monitor.get_hourly_spend.return_value = 6.0  # > llm hourly_limit=5.0
        mock_monitor.get_daily_spend.return_value = 6.0
        mock_monitor.get_monthly_spend.return_value = 6.0

        alerts = await analyzer._check_service_budget("llm", BUDGETS["llm"])
        warnings = [a for a in alerts if a.level == AlertLevel.WARNING]
        assert len(warnings) >= 1
        assert "hourly" in warnings[0].message.lower()

    @pytest.mark.asyncio
    async def test_hourly_kill_triggers_pause(self, analyzer, mock_monitor):
        mock_monitor.get_hourly_spend.return_value = 25.0  # > llm hourly_kill=20.0
        mock_monitor.get_daily_spend.return_value = 25.0
        mock_monitor.get_monthly_spend.return_value = 25.0

        alerts = await analyzer._check_service_budget("llm", BUDGETS["llm"])
        emergencies = [a for a in alerts if a.level == AlertLevel.EMERGENCY]
        assert len(emergencies) >= 1
        assert emergencies[0].action == "pause_service"

    @pytest.mark.asyncio
    async def test_daily_kill_triggers_pause(self, analyzer, mock_monitor):
        mock_monitor.get_hourly_spend.return_value = 0.0
        mock_monitor.get_daily_spend.return_value = 110.0  # > llm daily_kill=100.0
        mock_monitor.get_monthly_spend.return_value = 110.0

        alerts = await analyzer._check_service_budget("llm", BUDGETS["llm"])
        emergencies = [a for a in alerts if a.level == AlertLevel.EMERGENCY]
        assert len(emergencies) >= 1
        assert "daily" in emergencies[0].message.lower()

    @pytest.mark.asyncio
    async def test_monthly_limit_triggers_critical(self, analyzer, mock_monitor):
        mock_monitor.get_hourly_spend.return_value = 0.0
        mock_monitor.get_daily_spend.return_value = 0.0
        mock_monitor.get_monthly_spend.return_value = 150.0  # > llm monthly_limit=100.0

        alerts = await analyzer._check_service_budget("llm", BUDGETS["llm"])
        criticals = [a for a in alerts if a.level == AlertLevel.CRITICAL]
        assert len(criticals) >= 1
        assert "monthly" in criticals[0].message.lower()

    @pytest.mark.asyncio
    async def test_monthly_kill_triggers_emergency(self, analyzer, mock_monitor):
        mock_monitor.get_hourly_spend.return_value = 0.0
        mock_monitor.get_daily_spend.return_value = 0.0
        mock_monitor.get_monthly_spend.return_value = 600.0  # > llm monthly_kill=500.0

        alerts = await analyzer._check_service_budget("llm", BUDGETS["llm"])
        emergencies = [a for a in alerts if a.level == AlertLevel.EMERGENCY]
        assert len(emergencies) >= 1
        assert emergencies[0].action == "pause_all_sessions"


# ═══════════════════════════════════════════════════════════════════════
# Anomaly Detection Tests
# ═══════════════════════════════════════════════════════════════════════


class TestAnomalyDetection:
    @pytest.mark.asyncio
    async def test_spike_detection_5x_average(self, analyzer, mock_monitor):
        mock_monitor.get_hourly_spend.return_value = 10.0
        mock_monitor.get_average_hourly_spend.return_value = 1.0  # 10x average

        alert = await analyzer._check_spike("llm")
        assert alert is not None
        assert alert.level == AlertLevel.CRITICAL
        assert "SPIKE" in alert.message

    @pytest.mark.asyncio
    async def test_no_spike_when_avg_zero(self, analyzer, mock_monitor):
        mock_monitor.get_hourly_spend.return_value = 10.0
        mock_monitor.get_average_hourly_spend.return_value = 0.0

        alert = await analyzer._check_spike("llm")
        assert alert is None

    @pytest.mark.asyncio
    async def test_per_request_max_exceeded(self, analyzer, mock_monitor):
        mock_monitor.get_last_request_cost.return_value = 1.50  # > llm per_request_max=0.50

        alert = await analyzer._check_per_request("llm", BUDGETS["llm"])
        assert alert is not None
        assert alert.level == AlertLevel.WARNING
        assert "ANOMALY" in alert.message

    @pytest.mark.asyncio
    async def test_per_request_normal(self, analyzer, mock_monitor):
        mock_monitor.get_last_request_cost.return_value = 0.01

        alert = await analyzer._check_per_request("llm", BUDGETS["llm"])
        assert alert is None

    @pytest.mark.asyncio
    async def test_rapid_fire_detection(self, analyzer, mock_monitor):
        mock_monitor.get_recent_active_customer_ids.return_value = ["cust_1"]
        mock_monitor.get_customer_request_rate.return_value = 50  # > threshold=20

        alerts = await analyzer._check_rapid_fire()
        assert len(alerts) == 1
        assert alerts[0].action == "throttle_customer"
        assert alerts[0].customer_id == "cust_1"

    @pytest.mark.asyncio
    async def test_rapid_fire_normal(self, analyzer, mock_monitor):
        mock_monitor.get_recent_active_customer_ids.return_value = ["cust_1"]
        mock_monitor.get_customer_request_rate.return_value = 5

        alerts = await analyzer._check_rapid_fire()
        assert len(alerts) == 0

    @pytest.mark.asyncio
    async def test_stuck_runpod_job_detected(self, analyzer, mock_monitor):
        old_time = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=15)
        mock_monitor.get_runpod_active_jobs.return_value = [
            {"id": "job_1", "customer_id": "c1", "session_id": "s1",
             "created_at": old_time, "details": "{}"},
        ]

        alerts = await analyzer._check_stuck_jobs()
        assert len(alerts) == 1
        assert "STUCK" in alerts[0].message
        assert alerts[0].action == "cancel_runpod_job"

    @pytest.mark.asyncio
    async def test_stuck_job_not_triggered_for_new_jobs(self, analyzer, mock_monitor):
        recent_time = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=2)
        mock_monitor.get_runpod_active_jobs.return_value = [
            {"id": "job_2", "customer_id": "c1", "session_id": "s1",
             "created_at": recent_time, "details": "{}"},
        ]

        alerts = await analyzer._check_stuck_jobs()
        assert len(alerts) == 0

    @pytest.mark.asyncio
    async def test_zero_cost_billing_broken(self, analyzer, mock_monitor):
        mock_monitor.get_zero_cost_counts.return_value = [
            {"service": "llm", "zero_count": 20},
        ]

        alerts = await analyzer._check_zero_costs()
        assert len(alerts) == 1
        assert "ZERO COST" in alerts[0].message


# ═══════════════════════════════════════════════════════════════════════
# Customer Margin Tests
# ═══════════════════════════════════════════════════════════════════════


class TestCustomerMargin:
    @pytest.mark.asyncio
    async def test_customer_margin_warning_50pct(self, analyzer, mock_monitor):
        mock_monitor.get_top_spending_customers.return_value = [
            {"customer_id": "c1", "company_name": "TestCo", "plan_tier": "starter",
             "total_cost": 55.0, "total_calls": 100},  # 55% of $100
        ]

        alerts = await analyzer._check_customer_margins()
        warnings = [a for a in alerts if a.level == AlertLevel.WARNING]
        assert len(warnings) == 1
        assert warnings[0].customer_id == "c1"

    @pytest.mark.asyncio
    async def test_customer_margin_critical_70pct(self, analyzer, mock_monitor):
        mock_monitor.get_top_spending_customers.return_value = [
            {"customer_id": "c2", "company_name": "BigCo", "plan_tier": "professional",
             "total_cost": 150.0, "total_calls": 500},  # 75% of $200
        ]

        alerts = await analyzer._check_customer_margins()
        criticals = [a for a in alerts if a.level == AlertLevel.CRITICAL]
        assert len(criticals) == 1

    @pytest.mark.asyncio
    async def test_customer_margin_kill_90pct(self, analyzer, mock_monitor):
        mock_monitor.get_top_spending_customers.return_value = [
            {"customer_id": "c3", "company_name": "LossCo", "plan_tier": "starter",
             "total_cost": 95.0, "total_calls": 1000},  # 95% of $100
        ]

        alerts = await analyzer._check_customer_margins()
        emergencies = [a for a in alerts if a.level == AlertLevel.EMERGENCY]
        assert len(emergencies) == 1
        assert emergencies[0].action == "pause_customer"


# ═══════════════════════════════════════════════════════════════════════
# Enforcer Action Tests
# ═══════════════════════════════════════════════════════════════════════


class TestEnforcerActions:
    @pytest.mark.asyncio
    async def test_pause_service_sets_redis_flag(self, enforcer, mock_redis):
        alert = CostAlert(
            AlertLevel.EMERGENCY, "LLM", "test", 25.0, 20.0,
            action="pause_service",
        )
        result = await enforcer.execute_action(alert)
        assert result["action"] == "pause_service"
        mock_redis.set.assert_any_call("cost_guardian:paused:LLM", "1", ex=3600)

    @pytest.mark.asyncio
    async def test_emergency_pause_blocks_all(self, enforcer, mock_redis):
        alert = CostAlert(
            AlertLevel.EMERGENCY, "LLM", "test", 600.0, 500.0,
            action="pause_all_sessions",
        )
        result = await enforcer.execute_action(alert)
        assert result["action"] == "pause_all_sessions"
        mock_redis.set.assert_any_call("cost_guardian:emergency_pause", "1", ex=3600)

    @pytest.mark.asyncio
    async def test_pause_customer(self, enforcer, mock_redis):
        alert = CostAlert(
            AlertLevel.EMERGENCY, "customer_margin", "test", 95.0, 90.0,
            action="pause_customer", customer_id="cust_abc",
        )
        result = await enforcer.execute_action(alert)
        assert result["action"] == "pause_customer"
        mock_redis.set.assert_any_call(
            "cost_guardian:customer_paused:cust_abc", "1", ex=86400,
        )

    @pytest.mark.asyncio
    async def test_throttle_customer(self, enforcer, mock_redis):
        alert = CostAlert(
            AlertLevel.CRITICAL, "rapid_fire", "test", 50.0, 20.0,
            action="throttle_customer", customer_id="cust_xyz",
        )
        result = await enforcer.execute_action(alert)
        assert result["action"] == "throttle_customer"
        mock_redis.set.assert_any_call(
            "cost_guardian:throttle:cust_xyz", "10", ex=600,
        )

    @pytest.mark.asyncio
    async def test_cancel_stuck_runpod_jobs(self, enforcer, mock_redis):
        alert = CostAlert(
            AlertLevel.CRITICAL, "RunPod stuck job", "test", 15.0, 10.0,
            action="cancel_runpod_job",
        )
        result = await enforcer.execute_action(alert)
        assert result["action"] == "cancel_runpod_job"

    @pytest.mark.asyncio
    async def test_investigate_action(self, enforcer, mock_redis):
        alert = CostAlert(
            AlertLevel.CRITICAL, "LLM", "test", 10.0, 5.0,
            action="investigate",
        )
        result = await enforcer.execute_action(alert)
        assert result["action"] == "investigate"
        assert result["status"] == "logged_for_investigation"

    @pytest.mark.asyncio
    async def test_no_action(self, enforcer, mock_redis):
        alert = CostAlert(
            AlertLevel.WARNING, "LLM", "test", 6.0, 5.0,
            action=None,
        )
        result = await enforcer.execute_action(alert)
        assert result["action"] == "none"

    @pytest.mark.asyncio
    async def test_is_service_paused(self, enforcer, mock_redis):
        mock_redis.exists.return_value = True
        assert await enforcer.is_service_paused("LLM") is True

    @pytest.mark.asyncio
    async def test_is_service_not_paused(self, enforcer, mock_redis):
        mock_redis.exists.return_value = False
        assert await enforcer.is_service_paused("LLM") is False

    @pytest.mark.asyncio
    async def test_is_emergency_paused(self, enforcer, mock_redis):
        mock_redis.exists.return_value = True
        assert await enforcer.is_emergency_paused() is True

    @pytest.mark.asyncio
    async def test_is_customer_paused(self, enforcer, mock_redis):
        mock_redis.exists.return_value = True
        assert await enforcer.is_customer_paused("cust_1") is True

    @pytest.mark.asyncio
    async def test_is_customer_throttled(self, enforcer, mock_redis):
        mock_redis.get.return_value = "10"
        assert await enforcer.is_customer_throttled("cust_1") == 10

    @pytest.mark.asyncio
    async def test_is_customer_not_throttled(self, enforcer, mock_redis):
        mock_redis.get.return_value = None
        assert await enforcer.is_customer_throttled("cust_1") is None

    @pytest.mark.asyncio
    async def test_enforcer_no_redis(self):
        """Enforcer degrades gracefully without Redis."""
        enforcer = CostEnforcer(db=None, redis=None)
        assert await enforcer.is_service_paused("LLM") is False
        assert await enforcer.is_emergency_paused() is False
        assert await enforcer.is_customer_paused("x") is False
        assert await enforcer.is_customer_throttled("x") is None


# ═══════════════════════════════════════════════════════════════════════
# Reporter Tests
# ═══════════════════════════════════════════════════════════════════════


class TestReporter:
    def test_build_alert_email(self, reporter):
        alerts_e = [CostAlert(AlertLevel.EMERGENCY, "LLM", "Emergency msg", 25.0, 20.0, action="pause_service")]
        alerts_c = [CostAlert(AlertLevel.CRITICAL, "ASR", "Critical msg", 15.0, 10.0)]
        alerts_w = [CostAlert(AlertLevel.WARNING, "TTS", "Warning msg", 4.0, 3.0)]

        html = reporter._build_alert_email(alerts_e, alerts_c, alerts_w)
        assert "Emergency msg" in html
        assert "Critical msg" in html
        assert "Warning msg" in html
        assert "EMERGENCY" in html

    def test_build_daily_report(self, reporter):
        spend = {"llm": {"cost": 1.50, "calls": 100}, "asr": {"cost": 0.30, "calls": 50}}
        customers = [
            {"company_name": "TestCo", "plan_tier": "starter", "total_cost": 1.80, "total_calls": 150},
        ]
        html = reporter._build_daily_report(spend, customers, 1.80)
        assert "TestCo" in html
        assert "$1.80" in html

    @pytest.mark.asyncio
    async def test_send_skipped_without_api_key(self, reporter, mock_monitor):
        """No error when resend_api_key is None."""
        await reporter.send_alert_email([
            CostAlert(AlertLevel.WARNING, "LLM", "test", 6.0, 5.0),
        ])
        # Should not raise

    @pytest.mark.asyncio
    async def test_send_daily_report_no_key(self, reporter, mock_monitor):
        """Daily report is silently skipped without API key."""
        await reporter.send_daily_report()

    @pytest.mark.asyncio
    async def test_send_emergency_report_no_key(self, reporter):
        alert = CostAlert(AlertLevel.EMERGENCY, "LLM", "test", 25.0, 20.0)
        await reporter.send_emergency_report(alert, {"action": "pause_service"})


# ═══════════════════════════════════════════════════════════════════════
# Guardian Loop Tests
# ═══════════════════════════════════════════════════════════════════════


class TestGuardianLoop:
    @pytest.mark.asyncio
    async def test_check_cycle_no_alerts(self):
        guardian = CostGuardian(db=None, redis=None)
        guardian.analyzer.run_full_analysis = AsyncMock(return_value=[])
        await guardian._check_cycle()
        # No crash, no emails

    @pytest.mark.asyncio
    async def test_check_cycle_with_emergency(self):
        guardian = CostGuardian(db=None, redis=AsyncMock())
        guardian.analyzer.run_full_analysis = AsyncMock(return_value=[
            CostAlert(AlertLevel.EMERGENCY, "LLM", "Overrun", 25.0, 20.0, action="pause_service"),
        ])
        guardian.enforcer.execute_action = AsyncMock(return_value={"action": "pause_service"})
        guardian.reporter.send_emergency_report = AsyncMock()

        await guardian._check_cycle()
        guardian.enforcer.execute_action.assert_called_once()
        guardian.reporter.send_emergency_report.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_cycle_notification_rate_limited(self):
        redis_mock = AsyncMock()
        redis_mock.get = AsyncMock(return_value="1")  # Already sent recently
        redis_mock.set = AsyncMock()

        guardian = CostGuardian(db=None, redis=redis_mock)
        guardian.analyzer.run_full_analysis = AsyncMock(return_value=[
            CostAlert(AlertLevel.WARNING, "LLM", "Warning", 6.0, 5.0),
        ])
        guardian.reporter.send_alert_email = AsyncMock()

        await guardian._check_cycle()
        guardian.reporter.send_alert_email.assert_not_called()

    @pytest.mark.asyncio
    async def test_check_cycle_sends_notification_when_no_cooldown(self):
        redis_mock = AsyncMock()
        redis_mock.get = AsyncMock(return_value=None)  # No cooldown
        redis_mock.set = AsyncMock()

        guardian = CostGuardian(db=None, redis=redis_mock)
        guardian.analyzer.run_full_analysis = AsyncMock(return_value=[
            CostAlert(AlertLevel.WARNING, "LLM", "Warning", 6.0, 5.0),
        ])
        guardian.reporter.send_alert_email = AsyncMock()

        await guardian._check_cycle()
        guardian.reporter.send_alert_email.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_stop(self):
        guardian = CostGuardian(db=None, redis=None)
        guardian.analyzer.run_full_analysis = AsyncMock(return_value=[])

        # Start in background
        task = asyncio.create_task(guardian.start())
        await asyncio.sleep(0.1)
        assert guardian.running is True

        await guardian.stop()
        # Give the loop time to exit
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_notification_rate_limit_no_redis(self):
        guardian = CostGuardian(db=None, redis=None)
        assert await guardian._should_send_notification() is True


# ═══════════════════════════════════════════════════════════════════════
# ServicePausedError Tests
# ═══════════════════════════════════════════════════════════════════════


class TestServicePausedError:
    def test_default_message(self):
        err = ServicePausedError()
        assert "paused" in err.message.lower()

    def test_custom_message(self):
        err = ServicePausedError("Platform emergency pause active")
        assert err.message == "Platform emergency pause active"

    def test_to_dict(self):
        err = ServicePausedError("Paused", detail={"reason": "cost_overrun"})
        d = err.to_dict()
        assert d["error"] == "Paused"
        assert d["detail"]["reason"] == "cost_overrun"


# ═══════════════════════════════════════════════════════════════════════
# CostAlert Tests
# ═══════════════════════════════════════════════════════════════════════


class TestCostAlert:
    def test_alert_repr(self):
        alert = CostAlert(AlertLevel.WARNING, "LLM", "Test message", 6.0, 5.0)
        r = repr(alert)
        assert "warning" in r
        assert "LLM" in r

    def test_alert_fields(self):
        alert = CostAlert(
            AlertLevel.EMERGENCY, "ASR", "msg", 10.0, 5.0,
            action="pause_service", customer_id="c1",
        )
        assert alert.level == AlertLevel.EMERGENCY
        assert alert.service == "ASR"
        assert alert.current_value == 10.0
        assert alert.threshold == 5.0
        assert alert.action == "pause_service"
        assert alert.customer_id == "c1"
        assert alert.timestamp is not None


# ═══════════════════════════════════════════════════════════════════════
# Integration: Full Analysis Cycle
# ═══════════════════════════════════════════════════════════════════════


class TestFullAnalysisCycle:
    @pytest.mark.asyncio
    async def test_full_analysis_empty(self, analyzer, mock_monitor):
        """Full analysis with zero spend returns no alerts."""
        alerts = await analyzer.run_full_analysis()
        assert len(alerts) == 0

    @pytest.mark.asyncio
    async def test_full_analysis_multiple_issues(self, analyzer, mock_monitor):
        """Full analysis can return alerts from multiple checks."""
        # LLM hourly over limit
        mock_monitor.get_hourly_spend.return_value = 6.0
        mock_monitor.get_daily_spend.return_value = 0.0
        mock_monitor.get_monthly_spend.return_value = 0.0
        # Spike: avg=1, current=6 -> 6x
        mock_monitor.get_average_hourly_spend.return_value = 1.0

        alerts = await analyzer.run_full_analysis()
        # Should have at least warnings for all 6 services + spike alerts
        assert len(alerts) > 0
