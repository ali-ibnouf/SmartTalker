"""Tests for BillingEngine."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.pipeline.billing import BillingEngine, BillingSession


@pytest.fixture
def billing_config():
    """Create a billing-enabled config."""
    config = MagicMock()
    config.billing_enabled = True
    config.billing_rate_per_second = 0.001
    config.billing_grace_period_s = 5
    return config


@pytest.fixture
def billing_config_disabled():
    """Create a billing-disabled config."""
    config = MagicMock()
    config.billing_enabled = False
    config.billing_rate_per_second = 0.001
    config.billing_grace_period_s = 5
    return config


@pytest.fixture
def billing_engine(billing_config):
    """Create a BillingEngine with no DB."""
    return BillingEngine(billing_config, db=None)


class TestBillingEngineInit:
    """Test BillingEngine initialization."""

    def test_init(self, billing_config):
        engine = BillingEngine(billing_config)
        assert engine.is_loaded is False
        assert engine._rate == 0.001
        assert engine._grace == 5

    @pytest.mark.asyncio
    async def test_load_unload(self, billing_engine):
        assert billing_engine.is_loaded is False
        await billing_engine.load()
        assert billing_engine.is_loaded is True
        await billing_engine.unload()
        assert billing_engine.is_loaded is False


class TestBillingSession:
    """Test billing session lifecycle."""

    @pytest.mark.asyncio
    async def test_start_session_billing_disabled(self, billing_config_disabled):
        engine = BillingEngine(billing_config_disabled, db=None)
        session = await engine.start_session("s1", "c1")
        assert session.session_id == "s1"
        assert session.customer_id == "c1"
        # No active sessions tracked when billing disabled
        active = await engine.get_active_sessions()
        assert len(active) == 0

    @pytest.mark.asyncio
    async def test_start_session_no_db_unlimited_quota(self, billing_engine):
        """No DB = infinite quota, session should start."""
        session = await billing_engine.start_session("s1", "c1", avatar_id="a1")
        assert session.session_id == "s1"
        assert session.customer_id == "c1"
        assert session.avatar_id == "a1"
        assert session.started_at > 0

    @pytest.mark.asyncio
    async def test_stop_session(self, billing_engine):
        await billing_engine.start_session("s1", "c1")
        stopped = await billing_engine.stop_session("s1")
        assert stopped is not None
        assert stopped.stopped_at > 0
        assert stopped.total_seconds >= 0
        assert stopped.total_cost >= 0

    @pytest.mark.asyncio
    async def test_stop_nonexistent_session(self, billing_engine):
        result = await billing_engine.stop_session("nope")
        assert result is None

    @pytest.mark.asyncio
    async def test_active_sessions(self, billing_engine):
        await billing_engine.start_session("s1", "c1")
        await billing_engine.start_session("s2", "c2")
        active = await billing_engine.get_active_sessions()
        assert len(active) == 2

        await billing_engine.stop_session("s1")
        active = await billing_engine.get_active_sessions()
        assert len(active) == 1

    @pytest.mark.asyncio
    async def test_grace_period_applied(self, billing_engine):
        """Grace period should reduce billable seconds."""
        import time
        await billing_engine.start_session("s1", "c1")
        # Manually set started_at to 3 seconds ago (less than grace)
        billing_engine._active_sessions["s1"].started_at = time.time() - 3
        stopped = await billing_engine.stop_session("s1")
        # 3 seconds - 5 grace = 0 billable
        assert stopped.total_seconds == 0
        assert stopped.total_cost == 0


class TestBillingQuota:
    """Test quota checking."""

    @pytest.mark.asyncio
    async def test_check_quota_no_db(self, billing_engine):
        remaining = await billing_engine.check_quota("c1")
        assert remaining == float("inf")

    @pytest.mark.asyncio
    async def test_get_usage_no_db(self, billing_engine):
        usage = await billing_engine.get_usage("c1")
        assert usage == []


class TestBillingConcurrentLimit:
    """Test concurrent session limit enforcement."""

    @pytest.mark.asyncio
    async def test_check_concurrent_no_db_unlimited(self, billing_engine):
        """No DB = plan_info defaults, concurrent check passes."""
        allowed = await billing_engine.check_concurrent_limit("c1")
        assert allowed is True

    @pytest.mark.asyncio
    async def test_concurrent_limit_blocks_at_max(self, billing_engine):
        """When active sessions == max_concurrent, new session is rejected."""
        from src.utils.exceptions import BillingError

        # No DB = starter plan defaults (max_concurrent via PLAN_TIERS)
        # Start sessions until limit is hit
        # With no DB, get_plan_info returns starter default (max_concurrent=1)
        await billing_engine.start_session("s1", "c1")
        with pytest.raises(BillingError, match="Concurrent session limit"):
            await billing_engine.start_session("s2", "c1")

    @pytest.mark.asyncio
    async def test_concurrent_different_customers_independent(self, billing_engine):
        """Different customers have independent concurrent limits."""
        s1 = await billing_engine.start_session("s1", "c1")
        s2 = await billing_engine.start_session("s2", "c2")
        assert s1 is not None
        assert s2 is not None

    @pytest.mark.asyncio
    async def test_stop_frees_concurrent_slot(self, billing_engine):
        """Stopping a session allows a new one for the same customer."""
        await billing_engine.start_session("s1", "c1")
        await billing_engine.stop_session("s1")
        # Should succeed now
        s2 = await billing_engine.start_session("s2", "c1")
        assert s2 is not None


class TestBillingPlanInfo:
    """Test get_plan_info."""

    @pytest.mark.asyncio
    async def test_plan_info_no_db(self, billing_engine):
        """No DB returns starter plan defaults."""
        info = await billing_engine.get_plan_info("c1")
        assert isinstance(info, dict)
        assert "monthly_seconds" in info or "max_concurrent" in info


class TestBillingUnload:
    """Test unload stops all sessions."""

    @pytest.mark.asyncio
    async def test_unload_stops_sessions(self, billing_engine):
        await billing_engine.start_session("s1", "c1")
        await billing_engine.start_session("s2", "c2")
        await billing_engine.load()
        await billing_engine.unload()
        active = await billing_engine.get_active_sessions()
        assert len(active) == 0


class TestDualBalance:
    """Test dual balance model (plan seconds + extra seconds)."""

    @pytest.mark.asyncio
    async def test_get_balance_no_db(self, billing_engine):
        """No DB returns infinite plan seconds, zero extra."""
        balance = await billing_engine.get_balance("c1")
        assert balance["plan_seconds_remaining"] == float("inf")
        assert balance["plan_seconds_total"] == float("inf")
        assert balance["extra_seconds_remaining"] == 0
        assert balance["total_remaining"] == float("inf")
        assert balance["usage_pct"] == 0.0

    @pytest.mark.asyncio
    async def test_check_quota_delegates_to_balance(self, billing_engine):
        """check_quota returns total_remaining from get_balance."""
        remaining = await billing_engine.check_quota("c1")
        assert remaining == float("inf")

    @pytest.mark.asyncio
    async def test_add_topup_no_db(self, billing_engine):
        """add_topup without DB returns the seconds value."""
        result = await billing_engine.add_topup("c1", 10_000)
        assert result == 10_000

    @pytest.mark.asyncio
    async def test_add_topup_different_amounts(self, billing_engine):
        """Different top-up amounts return correctly."""
        assert await billing_engine.add_topup("c1", 25_000) == 25_000
        assert await billing_engine.add_topup("c1", 50_000) == 50_000


class TestBalanceAlerts:
    """Test balance alert threshold system."""

    @pytest.mark.asyncio
    async def test_alert_warning_at_20pct(self, billing_engine):
        """Warning alert fires when 20% of plan remaining."""
        balance = {
            "plan_seconds_remaining": 15_000,
            "plan_seconds_total": 100_000,
            "extra_seconds_remaining": 0,
        }
        alert = await billing_engine.check_balance_and_alert("c1", balance)
        assert alert is not None
        assert alert["level"] == "warning"
        assert alert["customer_id"] == "c1"

    @pytest.mark.asyncio
    async def test_alert_urgent_at_5pct(self, billing_engine):
        """Urgent alert fires when 5% of plan remaining."""
        balance = {
            "plan_seconds_remaining": 3_000,
            "plan_seconds_total": 100_000,
            "extra_seconds_remaining": 0,
        }
        alert = await billing_engine.check_balance_and_alert("c1", balance)
        assert alert is not None
        assert alert["level"] == "urgent"

    @pytest.mark.asyncio
    async def test_alert_critical_at_zero(self, billing_engine):
        """Critical alert fires when plan=0 and extra=0."""
        balance = {
            "plan_seconds_remaining": 0,
            "plan_seconds_total": 100_000,
            "extra_seconds_remaining": 0,
        }
        alert = await billing_engine.check_balance_and_alert("c1", balance)
        assert alert is not None
        assert alert["level"] == "critical"
        assert "exhausted" in alert["alert_message"].lower()

    @pytest.mark.asyncio
    async def test_no_alert_healthy_balance(self, billing_engine):
        """No alert when balance is above 20%."""
        balance = {
            "plan_seconds_remaining": 50_000,
            "plan_seconds_total": 100_000,
            "extra_seconds_remaining": 0,
        }
        alert = await billing_engine.check_balance_and_alert("c1", balance)
        assert alert is None

    @pytest.mark.asyncio
    async def test_alert_deduplication(self, billing_engine):
        """Same alert level does not fire twice for same customer."""
        balance = {
            "plan_seconds_remaining": 15_000,
            "plan_seconds_total": 100_000,
            "extra_seconds_remaining": 0,
        }
        alert1 = await billing_engine.check_balance_and_alert("c1", balance)
        assert alert1 is not None
        assert alert1["level"] == "warning"

        # Second call at same level — no alert
        alert2 = await billing_engine.check_balance_and_alert("c1", balance)
        assert alert2 is None

    @pytest.mark.asyncio
    async def test_alert_escalation(self, billing_engine):
        """Alert escalates from warning to urgent."""
        warn_balance = {
            "plan_seconds_remaining": 15_000,
            "plan_seconds_total": 100_000,
            "extra_seconds_remaining": 0,
        }
        alert1 = await billing_engine.check_balance_and_alert("c1", warn_balance)
        assert alert1["level"] == "warning"

        # Balance drops further
        urgent_balance = {
            "plan_seconds_remaining": 3_000,
            "plan_seconds_total": 100_000,
            "extra_seconds_remaining": 0,
        }
        alert2 = await billing_engine.check_balance_and_alert("c1", urgent_balance)
        assert alert2 is not None
        assert alert2["level"] == "urgent"

    @pytest.mark.asyncio
    async def test_alert_reset_on_healthy_balance(self, billing_engine):
        """Alert state resets when balance improves above threshold."""
        low_balance = {
            "plan_seconds_remaining": 15_000,
            "plan_seconds_total": 100_000,
            "extra_seconds_remaining": 0,
        }
        await billing_engine.check_balance_and_alert("c1", low_balance)

        # Balance restored (new billing cycle)
        healthy_balance = {
            "plan_seconds_remaining": 100_000,
            "plan_seconds_total": 100_000,
            "extra_seconds_remaining": 0,
        }
        alert = await billing_engine.check_balance_and_alert("c1", healthy_balance)
        assert alert is None
        # State should be cleared
        assert "c1" not in billing_engine._alerted

        # Warning should fire again after reset
        alert2 = await billing_engine.check_balance_and_alert("c1", low_balance)
        assert alert2 is not None
        assert alert2["level"] == "warning"

    @pytest.mark.asyncio
    async def test_alert_independent_customers(self, billing_engine):
        """Different customers have independent alert states."""
        low_balance = {
            "plan_seconds_remaining": 15_000,
            "plan_seconds_total": 100_000,
            "extra_seconds_remaining": 0,
        }
        alert1 = await billing_engine.check_balance_and_alert("c1", low_balance)
        assert alert1 is not None

        alert2 = await billing_engine.check_balance_and_alert("c2", low_balance)
        assert alert2 is not None  # Different customer, fires independently

    @pytest.mark.asyncio
    async def test_alert_no_plan_total(self, billing_engine):
        """No alert when plan_total is 0 (no subscription)."""
        balance = {
            "plan_seconds_remaining": 0,
            "plan_seconds_total": 0,
            "extra_seconds_remaining": 0,
        }
        alert = await billing_engine.check_balance_and_alert("c1", balance)
        assert alert is None

    @pytest.mark.asyncio
    async def test_alert_contains_details(self, billing_engine):
        """Alert includes all expected detail fields."""
        balance = {
            "plan_seconds_remaining": 3_000,
            "plan_seconds_total": 100_000,
            "extra_seconds_remaining": 5_000,
        }
        alert = await billing_engine.check_balance_and_alert("c1", balance)
        assert alert is not None
        assert "customer_id" in alert
        assert "level" in alert
        assert "alert_message" in alert
        assert "pct_remaining" in alert
        assert "plan_remaining" in alert
        assert "extra_remaining" in alert
        assert alert["extra_remaining"] == 5_000


class TestTopupPackages:
    """Test top-up package configuration."""

    def test_topup_packages_defined(self):
        """TOPUP_PACKAGES has small, medium, large."""
        from src.config import TOPUP_PACKAGES
        assert "small" in TOPUP_PACKAGES
        assert "medium" in TOPUP_PACKAGES
        assert "large" in TOPUP_PACKAGES

    def test_topup_package_prices(self):
        """Top-up packages have correct prices."""
        from src.config import TOPUP_PACKAGES
        assert TOPUP_PACKAGES["small"]["price"] == 20
        assert TOPUP_PACKAGES["medium"]["price"] == 50
        assert TOPUP_PACKAGES["large"]["price"] == 100

    def test_topup_package_seconds(self):
        """Top-up packages have correct seconds."""
        from src.config import TOPUP_PACKAGES
        assert TOPUP_PACKAGES["small"]["seconds"] == 10_000
        assert TOPUP_PACKAGES["medium"]["seconds"] == 25_000
        assert TOPUP_PACKAGES["large"]["seconds"] == 50_000

    def test_topup_price_per_second_decreases(self):
        """Larger packages offer better per-second rates."""
        from src.config import TOPUP_PACKAGES
        small_rate = TOPUP_PACKAGES["small"]["price"] / TOPUP_PACKAGES["small"]["seconds"]
        medium_rate = TOPUP_PACKAGES["medium"]["price"] / TOPUP_PACKAGES["medium"]["seconds"]
        large_rate = TOPUP_PACKAGES["large"]["price"] / TOPUP_PACKAGES["large"]["seconds"]
        assert large_rate <= medium_rate <= small_rate
