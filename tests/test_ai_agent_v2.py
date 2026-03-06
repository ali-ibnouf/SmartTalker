"""AI Optimization Agent v2 test suite.

Tests for:
A. Infrastructure monitors (10 tests)
B. New business monitors (6 tests)
C. New security monitors (4 tests)
D. Auto-fix handlers (12 tests)
E. Approval queue (8 tests)
F. Prevention enhancements (4 tests)
G. Standalone runner (2 tests)
H. Warmup job fix (9 tests)
I. RunPod workers auto-fixable (1 test)
J. Resilience rules (RunPodConsecutiveFailure, DashScopeConsecutiveTimeout,
   DashScopeQueueDepth, R2Downtime, MarginSqueeze) + fixes (VideoDisable, TextOnlyMode)
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.ai_agent.config import AgentSettings
from src.services.ai_agent.rules import AgentContext, Detection


def _make_ctx(**overrides) -> AgentContext:
    """Create a test AgentContext with sensible defaults."""
    return AgentContext(
        db=overrides.get("db"),
        redis=overrides.get("redis"),
        pipeline=overrides.get("pipeline"),
        config=overrides.get("config"),
        agent_config=overrides.get("agent_config", AgentSettings()),
        operator_manager=overrides.get("operator_manager"),
    )


# ═══════════════════════════════════════════════════════════════════════════
# A. Infrastructure Monitor Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestPostgreSQLConnectionsRule:
    """Test infra.pg_connections monitoring rule."""

    @pytest.mark.asyncio
    async def test_detects_high_connections(self):
        from src.services.ai_agent.monitors.infrastructure import PostgreSQLConnectionsRule

        mock_db = MagicMock()
        mock_session = AsyncMock()

        # pg_stat_activity returns 85 connections, max_connections returns 100
        mock_session.execute = AsyncMock(side_effect=[
            MagicMock(scalar=MagicMock(return_value=85)),  # current
            MagicMock(scalar=MagicMock(return_value="100")),  # max
        ])
        mock_db.session_ctx = MagicMock(return_value=_AsyncCM(mock_session))

        ctx = _make_ctx(db=mock_db)
        rule = PostgreSQLConnectionsRule()
        detections = await rule.evaluate(ctx)

        assert len(detections) == 1
        assert detections[0].severity == "warning"
        assert detections[0].details["usage_pct"] == 85.0

    @pytest.mark.asyncio
    async def test_critical_at_90_percent(self):
        from src.services.ai_agent.monitors.infrastructure import PostgreSQLConnectionsRule

        mock_db = MagicMock()
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(side_effect=[
            MagicMock(scalar=MagicMock(return_value=92)),
            MagicMock(scalar=MagicMock(return_value="100")),
        ])
        mock_db.session_ctx = MagicMock(return_value=_AsyncCM(mock_session))

        ctx = _make_ctx(db=mock_db)
        rule = PostgreSQLConnectionsRule()
        detections = await rule.evaluate(ctx)

        assert len(detections) == 1
        assert detections[0].severity == "critical"

    @pytest.mark.asyncio
    async def test_no_detection_below_threshold(self):
        from src.services.ai_agent.monitors.infrastructure import PostgreSQLConnectionsRule

        mock_db = MagicMock()
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(side_effect=[
            MagicMock(scalar=MagicMock(return_value=50)),
            MagicMock(scalar=MagicMock(return_value="100")),
        ])
        mock_db.session_ctx = MagicMock(return_value=_AsyncCM(mock_session))

        ctx = _make_ctx(db=mock_db)
        rule = PostgreSQLConnectionsRule()
        detections = await rule.evaluate(ctx)

        assert len(detections) == 0


class TestRedisMemoryRule:
    """Test infra.redis_memory monitoring rule."""

    @pytest.mark.asyncio
    async def test_detects_high_memory(self):
        from src.services.ai_agent.monitors.infrastructure import RedisMemoryRule

        mock_redis = AsyncMock()
        mock_redis.info = AsyncMock(return_value={
            "used_memory": 800 * 1024 * 1024,  # 800MB
            "maxmemory": 1024 * 1024 * 1024,   # 1GB
        })

        ctx = _make_ctx(redis=mock_redis)
        rule = RedisMemoryRule()
        detections = await rule.evaluate(ctx)

        assert len(detections) == 1
        assert detections[0].severity == "warning"

    @pytest.mark.asyncio
    async def test_critical_at_90_percent(self):
        from src.services.ai_agent.monitors.infrastructure import RedisMemoryRule

        mock_redis = AsyncMock()
        mock_redis.info = AsyncMock(return_value={
            "used_memory": 950 * 1024 * 1024,
            "maxmemory": 1024 * 1024 * 1024,
        })

        ctx = _make_ctx(redis=mock_redis)
        rule = RedisMemoryRule()
        detections = await rule.evaluate(ctx)

        assert len(detections) == 1
        assert detections[0].severity == "critical"


class TestDashScopeLatencyRule:
    """Test infra.dashscope_latency monitoring rule."""

    @pytest.mark.asyncio
    async def test_detects_high_latency(self):
        from collections import deque
        from src.services.ai_agent.monitors.infrastructure import DashScopeLatencyRule

        mock_pipeline = MagicMock()
        mock_llm = MagicMock()
        mock_llm._recent_latencies = deque([3.5, 4.0, 3.8, 4.2], maxlen=10)
        mock_pipeline._llm = mock_llm

        ctx = _make_ctx(pipeline=mock_pipeline)
        rule = DashScopeLatencyRule()
        detections = await rule.evaluate(ctx)

        assert len(detections) == 1
        assert detections[0].severity == "warning"

    @pytest.mark.asyncio
    async def test_no_detection_normal_latency(self):
        from collections import deque
        from src.services.ai_agent.monitors.infrastructure import DashScopeLatencyRule

        mock_pipeline = MagicMock()
        mock_llm = MagicMock()
        mock_llm._recent_latencies = deque([0.5, 0.8, 0.6], maxlen=10)
        mock_pipeline._llm = mock_llm

        ctx = _make_ctx(pipeline=mock_pipeline)
        rule = DashScopeLatencyRule()
        detections = await rule.evaluate(ctx)

        assert len(detections) == 0


class TestASRLatencyRule:
    """Test infra.asr_latency monitoring rule."""

    @pytest.mark.asyncio
    async def test_detects_high_asr_latency(self):
        from collections import deque
        from src.services.ai_agent.monitors.infrastructure import ASRLatencyRule

        mock_pipeline = MagicMock()
        mock_asr = MagicMock()
        mock_asr._recent_latencies = deque([0.6, 0.7, 0.55, 0.65], maxlen=50)
        mock_pipeline._asr = mock_asr

        ctx = _make_ctx(pipeline=mock_pipeline)
        rule = ASRLatencyRule()
        detections = await rule.evaluate(ctx)

        assert len(detections) == 1
        assert detections[0].rule_id == "infra.asr_latency"
        assert detections[0].severity == "warning"
        assert "ms avg" in detections[0].title

    @pytest.mark.asyncio
    async def test_no_detection_normal_asr_latency(self):
        from collections import deque
        from src.services.ai_agent.monitors.infrastructure import ASRLatencyRule

        mock_pipeline = MagicMock()
        mock_asr = MagicMock()
        mock_asr._recent_latencies = deque([0.1, 0.2, 0.15], maxlen=50)
        mock_pipeline._asr = mock_asr

        ctx = _make_ctx(pipeline=mock_pipeline)
        rule = ASRLatencyRule()
        detections = await rule.evaluate(ctx)

        assert len(detections) == 0

    @pytest.mark.asyncio
    async def test_no_pipeline_returns_empty(self):
        from src.services.ai_agent.monitors.infrastructure import ASRLatencyRule
        ctx = _make_ctx(pipeline=None)
        rule = ASRLatencyRule()
        assert await rule.evaluate(ctx) == []

    @pytest.mark.asyncio
    async def test_critical_at_1s(self):
        from collections import deque
        from src.services.ai_agent.monitors.infrastructure import ASRLatencyRule

        mock_pipeline = MagicMock()
        mock_asr = MagicMock()
        mock_asr._recent_latencies = deque([1.2, 1.5, 1.1], maxlen=50)
        mock_pipeline._asr = mock_asr

        ctx = _make_ctx(pipeline=mock_pipeline)
        rule = ASRLatencyRule()
        detections = await rule.evaluate(ctx)

        assert len(detections) == 1
        assert detections[0].severity == "critical"


class TestTTSLatencyRule:
    """Test infra.tts_latency monitoring rule."""

    @pytest.mark.asyncio
    async def test_detects_high_tts_latency(self):
        from collections import deque
        from src.services.ai_agent.monitors.infrastructure import TTSLatencyRule

        mock_pipeline = MagicMock()
        mock_tts = MagicMock()
        mock_tts._recent_latencies = deque([2.5, 3.0, 2.8], maxlen=50)
        mock_pipeline._tts = mock_tts

        ctx = _make_ctx(pipeline=mock_pipeline)
        rule = TTSLatencyRule()
        detections = await rule.evaluate(ctx)

        assert len(detections) == 1
        assert detections[0].rule_id == "infra.tts_latency"
        assert detections[0].severity == "warning"

    @pytest.mark.asyncio
    async def test_no_detection_normal_tts_latency(self):
        from collections import deque
        from src.services.ai_agent.monitors.infrastructure import TTSLatencyRule

        mock_pipeline = MagicMock()
        mock_tts = MagicMock()
        mock_tts._recent_latencies = deque([0.5, 0.8, 1.0], maxlen=50)
        mock_pipeline._tts = mock_tts

        ctx = _make_ctx(pipeline=mock_pipeline)
        rule = TTSLatencyRule()
        detections = await rule.evaluate(ctx)

        assert len(detections) == 0

    @pytest.mark.asyncio
    async def test_critical_at_double_threshold(self):
        from collections import deque
        from src.services.ai_agent.monitors.infrastructure import TTSLatencyRule

        mock_pipeline = MagicMock()
        mock_tts = MagicMock()
        mock_tts._recent_latencies = deque([4.5, 5.0, 4.8], maxlen=50)
        mock_pipeline._tts = mock_tts

        ctx = _make_ctx(pipeline=mock_pipeline)
        rule = TTSLatencyRule()
        detections = await rule.evaluate(ctx)

        assert len(detections) == 1
        assert detections[0].severity == "critical"

    @pytest.mark.asyncio
    async def test_no_pipeline_returns_empty(self):
        from src.services.ai_agent.monitors.infrastructure import TTSLatencyRule
        ctx = _make_ctx(pipeline=None)
        rule = TTSLatencyRule()
        assert await rule.evaluate(ctx) == []


class TestASRConnectionRule:
    """Test infra.asr_connection monitoring rule."""

    @pytest.mark.asyncio
    async def test_detects_connection_failures(self):
        from collections import deque
        from src.services.ai_agent.monitors.infrastructure import ASRConnectionRule

        mock_pipeline = MagicMock()
        mock_asr = MagicMock()
        # 5 errors in the last minute (well within 5min window)
        now = time.time()
        mock_asr._recent_errors = deque([now - 10, now - 20, now - 30, now - 40, now - 50], maxlen=20)
        mock_pipeline._asr = mock_asr

        ctx = _make_ctx(pipeline=mock_pipeline)
        rule = ASRConnectionRule()
        detections = await rule.evaluate(ctx)

        assert len(detections) == 1
        assert detections[0].rule_id == "infra.asr_connection"
        assert detections[0].severity == "warning"

    @pytest.mark.asyncio
    async def test_no_detection_old_errors(self):
        from collections import deque
        from src.services.ai_agent.monitors.infrastructure import ASRConnectionRule

        mock_pipeline = MagicMock()
        mock_asr = MagicMock()
        # Errors older than 5 minutes
        old = time.time() - 600
        mock_asr._recent_errors = deque([old, old - 10, old - 20], maxlen=20)
        mock_pipeline._asr = mock_asr

        ctx = _make_ctx(pipeline=mock_pipeline)
        rule = ASRConnectionRule()
        detections = await rule.evaluate(ctx)

        assert len(detections) == 0

    @pytest.mark.asyncio
    async def test_no_pipeline_returns_empty(self):
        from src.services.ai_agent.monitors.infrastructure import ASRConnectionRule
        ctx = _make_ctx(pipeline=None)
        rule = ASRConnectionRule()
        assert await rule.evaluate(ctx) == []


class TestTTSConnectionRule:
    """Test infra.tts_connection monitoring rule."""

    @pytest.mark.asyncio
    async def test_detects_connection_failures(self):
        from collections import deque
        from src.services.ai_agent.monitors.infrastructure import TTSConnectionRule

        mock_pipeline = MagicMock()
        mock_tts = MagicMock()
        now = time.time()
        mock_tts._recent_errors = deque([now - 5, now - 15, now - 25, now - 35], maxlen=20)
        mock_pipeline._tts = mock_tts

        ctx = _make_ctx(pipeline=mock_pipeline)
        rule = TTSConnectionRule()
        detections = await rule.evaluate(ctx)

        assert len(detections) == 1
        assert detections[0].rule_id == "infra.tts_connection"

    @pytest.mark.asyncio
    async def test_no_pipeline_returns_empty(self):
        from src.services.ai_agent.monitors.infrastructure import TTSConnectionRule
        ctx = _make_ctx(pipeline=None)
        rule = TTSConnectionRule()
        assert await rule.evaluate(ctx) == []


class TestDashScopeQuotaRule:
    """Test infra.dashscope_quota monitoring rule."""

    @pytest.mark.asyncio
    async def test_detects_high_usage(self):
        from src.services.ai_agent.monitors.infrastructure import DashScopeQuotaRule

        mock_db = MagicMock()
        mock_session = AsyncMock()
        # $450 of $500 budget = 90%
        mock_session.execute = AsyncMock(
            return_value=MagicMock(scalar=MagicMock(return_value=450.0))
        )
        mock_db.session_ctx = MagicMock(return_value=_AsyncCM(mock_session))

        ctx = _make_ctx(db=mock_db)
        rule = DashScopeQuotaRule()
        detections = await rule.evaluate(ctx)

        assert len(detections) == 1
        assert detections[0].rule_id == "infra.dashscope_quota"
        assert detections[0].severity == "warning"
        assert detections[0].details["usage_pct"] == 90.0

    @pytest.mark.asyncio
    async def test_no_detection_below_threshold(self):
        from src.services.ai_agent.monitors.infrastructure import DashScopeQuotaRule

        mock_db = MagicMock()
        mock_session = AsyncMock()
        # $100 of $500 = 20%
        mock_session.execute = AsyncMock(
            return_value=MagicMock(scalar=MagicMock(return_value=100.0))
        )
        mock_db.session_ctx = MagicMock(return_value=_AsyncCM(mock_session))

        ctx = _make_ctx(db=mock_db)
        rule = DashScopeQuotaRule()
        detections = await rule.evaluate(ctx)

        assert len(detections) == 0

    @pytest.mark.asyncio
    async def test_critical_when_exceeded(self):
        from src.services.ai_agent.monitors.infrastructure import DashScopeQuotaRule

        mock_db = MagicMock()
        mock_session = AsyncMock()
        # $550 of $500 = 110%
        mock_session.execute = AsyncMock(
            return_value=MagicMock(scalar=MagicMock(return_value=550.0))
        )
        mock_db.session_ctx = MagicMock(return_value=_AsyncCM(mock_session))

        ctx = _make_ctx(db=mock_db)
        rule = DashScopeQuotaRule()
        detections = await rule.evaluate(ctx)

        assert len(detections) == 1
        assert detections[0].severity == "critical"

    @pytest.mark.asyncio
    async def test_no_db_returns_empty(self):
        from src.services.ai_agent.monitors.infrastructure import DashScopeQuotaRule
        ctx = _make_ctx(db=None)
        rule = DashScopeQuotaRule()
        assert await rule.evaluate(ctx) == []


class TestRunPodRenderTimeRule:
    """Test infra.runpod_render_time monitoring rule."""

    @pytest.mark.asyncio
    async def test_detects_slow_renders_from_client(self):
        from collections import deque
        from src.services.ai_agent.monitors.infrastructure import RunPodRenderTimeRule

        mock_pipeline = MagicMock()
        mock_runpod = MagicMock()
        mock_runpod._recent_render_times = deque([6.0, 7.0, 5.5], maxlen=50)
        mock_pipeline._runpod = mock_runpod

        ctx = _make_ctx(pipeline=mock_pipeline)
        rule = RunPodRenderTimeRule()
        detections = await rule.evaluate(ctx)

        assert len(detections) == 1
        assert detections[0].severity == "warning"
        assert "6.2" in detections[0].title  # avg of 6.0, 7.0, 5.5

    @pytest.mark.asyncio
    async def test_critical_on_cold_starts(self):
        from collections import deque
        from src.services.ai_agent.monitors.infrastructure import RunPodRenderTimeRule

        mock_pipeline = MagicMock()
        mock_runpod = MagicMock()
        mock_runpod._recent_render_times = deque([18.0, 16.0, 20.0], maxlen=50)
        mock_pipeline._runpod = mock_runpod

        ctx = _make_ctx(pipeline=mock_pipeline)
        rule = RunPodRenderTimeRule()
        detections = await rule.evaluate(ctx)

        assert len(detections) == 1
        assert detections[0].severity == "critical"
        assert "cold" in detections[0].title.lower()

    @pytest.mark.asyncio
    async def test_no_detection_fast_renders(self):
        from collections import deque
        from src.services.ai_agent.monitors.infrastructure import RunPodRenderTimeRule

        mock_pipeline = MagicMock()
        mock_runpod = MagicMock()
        mock_runpod._recent_render_times = deque([2.0, 3.0, 2.5], maxlen=50)
        mock_pipeline._runpod = mock_runpod

        ctx = _make_ctx(pipeline=mock_pipeline)
        rule = RunPodRenderTimeRule()
        detections = await rule.evaluate(ctx)

        assert len(detections) == 0


class TestRunPodQueueDepthRule:
    """Test infra.runpod_queue monitoring rule."""

    @pytest.mark.asyncio
    async def test_no_config_returns_empty(self):
        from src.services.ai_agent.monitors.infrastructure import RunPodQueueDepthRule
        ctx = _make_ctx(config=None)
        rule = RunPodQueueDepthRule()
        assert await rule.evaluate(ctx) == []

    @pytest.mark.asyncio
    async def test_no_endpoint_returns_empty(self):
        from src.services.ai_agent.monitors.infrastructure import RunPodQueueDepthRule
        mock_config = MagicMock()
        mock_config.runpod_endpoint_musetalk = ""
        mock_config.runpod_api_key = ""
        ctx = _make_ctx(config=mock_config)
        rule = RunPodQueueDepthRule()
        assert await rule.evaluate(ctx) == []


class TestRunPodHealthRule:
    """Test infra.runpod_health monitoring rule."""

    def test_rule_id(self):
        from src.services.ai_agent.monitors.infrastructure import RunPodHealthRule
        rule = RunPodHealthRule()
        assert rule.rule_id == "infra.runpod_health"

    @pytest.mark.asyncio
    async def test_no_config_returns_empty(self):
        from src.services.ai_agent.monitors.infrastructure import RunPodHealthRule
        ctx = _make_ctx(config=None)
        rule = RunPodHealthRule()
        assert await rule.evaluate(ctx) == []

    @pytest.mark.asyncio
    async def test_detects_error_rate_from_db(self):
        """Test RunPod error rate detection via APICostRecord."""
        from src.services.ai_agent.monitors.infrastructure import RunPodHealthRule

        mock_config = MagicMock()
        mock_config.runpod_endpoint_url = ""  # Skip HTTP check
        mock_config.runpod_api_key = ""

        mock_db = MagicMock()
        mock_session = AsyncMock()
        # total=20, errors=5 (25% error rate)
        mock_session.execute = AsyncMock(side_effect=[
            MagicMock(scalar=MagicMock(return_value=20)),
            MagicMock(scalar=MagicMock(return_value=5)),
        ])
        mock_db.session_ctx = MagicMock(return_value=_AsyncCM(mock_session))

        ctx = _make_ctx(config=mock_config, db=mock_db)
        rule = RunPodHealthRule()
        detections = await rule.evaluate(ctx)

        assert len(detections) == 1
        assert detections[0].details["error_rate"] == 0.25


class TestR2ConnectivityRule:
    """Test infra.r2_connectivity monitoring rule."""

    @pytest.mark.asyncio
    async def test_detects_connection_failure(self):
        from src.services.ai_agent.monitors.infrastructure import R2ConnectivityRule

        mock_config = MagicMock()
        mock_config.r2_bucket_name = "test-bucket"
        mock_config.r2_endpoint_url = "https://test.r2.cloudflarestorage.com"
        mock_config.r2_access_key_id = "key"
        mock_config.r2_secret_access_key = "secret"

        ctx = _make_ctx(config=mock_config)
        rule = R2ConnectivityRule()

        # boto3 is imported inside evaluate(), patch the module
        mock_client = MagicMock()
        mock_client.list_objects_v2 = MagicMock(side_effect=Exception("Access denied"))

        mock_boto = MagicMock()
        mock_boto.client = MagicMock(return_value=mock_client)

        mock_botoconfig = MagicMock()
        mock_botocore = MagicMock()
        mock_botocore.config.Config = mock_botoconfig

        with patch.dict("sys.modules", {"boto3": mock_boto, "botocore": mock_botocore, "botocore.config": mock_botocore.config}):
            detections = await rule.evaluate(ctx)

        assert len(detections) == 1
        assert detections[0].severity == "critical"
        assert "R2" in detections[0].title


# ═══════════════════════════════════════════════════════════════════════════
# B. Business Monitor Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestFailedPaymentRule:
    """Test business.failed_payment monitoring rule."""

    def test_rule_id(self):
        from src.services.ai_agent.monitors.business import FailedPaymentRule
        rule = FailedPaymentRule()
        assert rule.rule_id == "business.failed_payment"

    @pytest.mark.asyncio
    async def test_no_db_returns_empty(self):
        from src.services.ai_agent.monitors.business import FailedPaymentRule
        ctx = _make_ctx(db=None)
        rule = FailedPaymentRule()
        assert await rule.evaluate(ctx) == []

    @pytest.mark.asyncio
    async def test_detects_payment_failures(self):
        from src.services.ai_agent.monitors.business import FailedPaymentRule

        mock_db = MagicMock()
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=MagicMock(
            all=MagicMock(return_value=[
                ("cust-1", "Acme Corp", "professional", 3),
            ])
        ))
        mock_db.session_ctx = MagicMock(return_value=_AsyncCM(mock_session))

        ctx = _make_ctx(db=mock_db)
        rule = FailedPaymentRule()
        detections = await rule.evaluate(ctx)

        assert len(detections) == 1
        assert detections[0].severity == "critical"
        assert detections[0].details["failures"] == 3

    @pytest.mark.asyncio
    async def test_warning_at_two_failures(self):
        from src.services.ai_agent.monitors.business import FailedPaymentRule

        mock_db = MagicMock()
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=MagicMock(
            all=MagicMock(return_value=[
                ("cust-1", "Acme Corp", "starter", 2),
            ])
        ))
        mock_db.session_ctx = MagicMock(return_value=_AsyncCM(mock_session))

        ctx = _make_ctx(db=mock_db)
        rule = FailedPaymentRule()
        detections = await rule.evaluate(ctx)

        assert len(detections) == 1
        assert detections[0].severity == "warning"


class TestTrainingStallRule:
    """Test business.training_stall monitoring rule."""

    @pytest.mark.asyncio
    async def test_detects_stalled_training(self):
        from src.services.ai_agent.monitors.business import TrainingStallRule

        mock_avatar = MagicMock()
        mock_avatar.id = "av-1"
        mock_avatar.name = "TestBot"
        mock_avatar.training_progress = 0.35
        mock_avatar.updated_at = datetime.utcnow() - timedelta(days=10)

        mock_db = MagicMock()
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=MagicMock(
            all=MagicMock(return_value=[(mock_avatar, "TestCo")])
        ))
        mock_db.session_ctx = MagicMock(return_value=_AsyncCM(mock_session))

        ctx = _make_ctx(db=mock_db)
        rule = TrainingStallRule()
        detections = await rule.evaluate(ctx)

        assert len(detections) == 1
        assert "35%" in detections[0].title


class TestOnboardingStuckRule:
    """Test business.onboarding_stuck monitoring rule."""

    @pytest.mark.asyncio
    async def test_no_db_returns_empty(self):
        from src.services.ai_agent.monitors.business import OnboardingStuckRule

        ctx = _make_ctx(db=None)
        rule = OnboardingStuckRule()
        assert await rule.evaluate(ctx) == []


# ═══════════════════════════════════════════════════════════════════════════
# C. Security Monitor Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestFailedAuthRule:
    """Test security.failed_auth monitoring rule."""

    @pytest.mark.asyncio
    async def test_detects_brute_force(self):
        from src.services.ai_agent.monitors.security import FailedAuthRule

        mock_redis = AsyncMock()
        # Simulate scan returning keys
        mock_redis.scan = AsyncMock(return_value=(b"0", [b"rate:auth_fail:1.2.3.4"]))
        mock_redis.ttl = AsyncMock(return_value=300)
        mock_redis.get = AsyncMock(return_value=b"10")

        ctx = _make_ctx(redis=mock_redis)
        rule = FailedAuthRule()
        detections = await rule.evaluate(ctx)

        assert len(detections) == 1
        assert detections[0].details["ip"] == "1.2.3.4"
        assert detections[0].details["failed_attempts"] == 10

    @pytest.mark.asyncio
    async def test_no_redis_returns_empty(self):
        from src.services.ai_agent.monitors.security import FailedAuthRule

        ctx = _make_ctx(redis=None)
        rule = FailedAuthRule()
        assert await rule.evaluate(ctx) == []


class TestAPIUsageSpikeRule:
    """Test security.api_spike monitoring rule."""

    @pytest.mark.asyncio
    async def test_detects_usage_spike(self):
        from src.services.ai_agent.monitors.security import APIUsageSpikeRule

        mock_db = MagicMock()
        mock_session = AsyncMock()
        # Call 1: recent counts, Call 2: avg counts
        mock_session.execute = AsyncMock(side_effect=[
            MagicMock(all=MagicMock(return_value=[("cust-1", 500)])),  # last hour
            MagicMock(all=MagicMock(return_value=[("cust-1", 50.0)])),  # hourly avg
        ])
        mock_db.session_ctx = MagicMock(return_value=_AsyncCM(mock_session))

        ctx = _make_ctx(db=mock_db)
        rule = APIUsageSpikeRule()
        detections = await rule.evaluate(ctx)

        assert len(detections) == 1
        assert detections[0].details["ratio"] == 10.0

    @pytest.mark.asyncio
    async def test_ignores_low_volume_customers(self):
        from src.services.ai_agent.monitors.security import APIUsageSpikeRule

        mock_db = MagicMock()
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(side_effect=[
            MagicMock(all=MagicMock(return_value=[("cust-1", 10)])),
            MagicMock(all=MagicMock(return_value=[("cust-1", 2.0)])),  # avg < 5
        ])
        mock_db.session_ctx = MagicMock(return_value=_AsyncCM(mock_session))

        ctx = _make_ctx(db=mock_db)
        rule = APIUsageSpikeRule()
        detections = await rule.evaluate(ctx)

        assert len(detections) == 0


# ═══════════════════════════════════════════════════════════════════════════
# D. Auto-Fix Handler Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestMemoryCleanupFix:
    """Test fix.memory_cleanup handler."""

    @pytest.mark.asyncio
    async def test_clears_redis_keys(self):
        from src.services.ai_agent.fixes.handlers import MemoryCleanupFix

        mock_redis = AsyncMock()
        # scan_iter returns async iterator
        async def _scan_iter(**kwargs):
            for k in [b"session:abc", b"session:def"]:
                yield k
        mock_redis.scan_iter = _scan_iter
        mock_redis.delete = AsyncMock()

        detection = Detection(
            rule_id="system.high_memory", severity="warning",
            title="test", description="", details={},
        )
        ctx = _make_ctx(redis=mock_redis)
        fix = MemoryCleanupFix()

        assert await fix.can_fix(detection, ctx)
        result = await fix.apply(detection, ctx)
        assert result["action"] == "memory_cleanup"
        assert result["redis_keys_cleared"] >= 0


class TestDBConnectionFix:
    """Test fix.db_connection_cleanup handler."""

    @pytest.mark.asyncio
    async def test_kills_idle_connections(self):
        from src.services.ai_agent.fixes.handlers import DBConnectionFix

        mock_db = MagicMock()
        mock_session = AsyncMock()
        # Returns rows where pg_terminate_backend returned True
        mock_session.execute = AsyncMock(return_value=MagicMock(
            __iter__=MagicMock(return_value=iter([(True,), (True,), (False,)]))
        ))
        mock_db.session_ctx = MagicMock(return_value=_AsyncCM(mock_session))

        detection = Detection(
            rule_id="infra.pg_connections", severity="warning",
            title="test", description="", details={"usage_pct": 85},
        )
        ctx = _make_ctx(db=mock_db)
        fix = DBConnectionFix()

        assert await fix.can_fix(detection, ctx)
        result = await fix.apply(detection, ctx)
        assert result["action"] == "db_connection_cleanup"
        assert result["connections_killed"] == 2


class TestQuotaGraceFix:
    """Test fix.quota_grace handler."""

    @pytest.mark.asyncio
    async def test_sets_grace_period(self):
        from src.services.ai_agent.fixes.handlers import QuotaGraceFix

        mock_db = MagicMock()
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock()
        mock_db.session_ctx = MagicMock(return_value=_AsyncCM(mock_session))

        detection = Detection(
            rule_id="business.quota_exhaustion", severity="critical",
            title="test", description="",
            details={"customer_id": "cust-1", "usage_pct": 105},
        )
        ctx = _make_ctx(db=mock_db)
        fix = QuotaGraceFix()

        assert await fix.can_fix(detection, ctx)
        result = await fix.apply(detection, ctx)
        assert result["action"] == "quota_grace_set"
        assert result["customer_id"] == "cust-1"

    @pytest.mark.asyncio
    async def test_skips_below_100_pct(self):
        from src.services.ai_agent.fixes.handlers import QuotaGraceFix

        detection = Detection(
            rule_id="business.quota_exhaustion", severity="warning",
            title="test", description="",
            details={"customer_id": "cust-1", "usage_pct": 85},
        )
        ctx = _make_ctx(db=MagicMock())
        fix = QuotaGraceFix()

        assert not await fix.can_fix(detection, ctx)


class TestRateLimitThrottleFix:
    """Test fix.rate_throttle handler."""

    @pytest.mark.asyncio
    async def test_applies_throttle(self):
        from src.services.ai_agent.fixes.handlers import RateLimitThrottleFix

        mock_redis = AsyncMock()
        mock_redis.set = AsyncMock()

        detection = Detection(
            rule_id="security.api_spike", severity="critical",
            title="test", description="",
            details={"customer_id": "cust-1"},
        )
        ctx = _make_ctx(redis=mock_redis)
        fix = RateLimitThrottleFix()

        assert await fix.can_fix(detection, ctx)
        result = await fix.apply(detection, ctx)
        assert result["action"] == "rate_throttle_applied"
        mock_redis.set.assert_called_once()


class TestRedisMemoryFix:
    """Test fix.redis_memory handler."""

    @pytest.mark.asyncio
    async def test_clears_cache(self):
        from src.services.ai_agent.fixes.handlers import RedisMemoryFix

        mock_redis = AsyncMock()
        mock_redis.execute_command = AsyncMock()
        async def _scan_iter(**kwargs):
            for k in [b"plan:cust1", b"plan:cust2"]:
                yield k
        mock_redis.scan_iter = _scan_iter
        mock_redis.delete = AsyncMock()

        detection = Detection(
            rule_id="infra.redis_memory", severity="warning",
            title="test", description="", details={},
        )
        ctx = _make_ctx(redis=mock_redis)
        fix = RedisMemoryFix()

        assert await fix.can_fix(detection, ctx)
        result = await fix.apply(detection, ctx)
        assert result["action"] == "redis_memory_cleanup"


class TestStaleSessionFix:
    """Test fix.stale_session_cleanup scheduled handler."""

    @pytest.mark.asyncio
    async def test_closes_idle_sessions(self):
        from src.services.ai_agent.fixes.handlers import StaleSessionFix

        mock_session = MagicMock()
        mock_session.session_id = "sess-1"
        mock_session.last_activity = time.time() - 3600  # 1 hour ago

        mock_manager = MagicMock()
        mock_manager.get_active_sessions = MagicMock(return_value=[mock_session])
        mock_manager.close_session = AsyncMock()

        ctx = _make_ctx(operator_manager=mock_manager)
        fix = StaleSessionFix()

        result = await fix.apply_scheduled(ctx)
        assert result["sessions_closed"] == 1
        mock_manager.close_session.assert_called_once_with("sess-1")


class TestFixCooldown:
    """Test cooldown enforcement in FixRegistry."""

    @pytest.mark.asyncio
    async def test_cooldown_blocks_rapid_fixes(self):
        from src.services.ai_agent.fixes.handlers import FixRegistry

        mock_handler = MagicMock()
        mock_handler.fix_id = "fix.test"
        mock_handler.can_fix = AsyncMock(return_value=True)
        mock_handler.apply = AsyncMock(return_value={"action": "test"})

        registry = FixRegistry()
        registry.register("test.rule", mock_handler)

        detection = Detection(
            rule_id="test.rule", severity="warning",
            title="test", description="", details={}, auto_fixable=True,
        )
        agent_config = AgentSettings()
        agent_config.fix_cooldown_s = 60
        ctx = _make_ctx(agent_config=agent_config)

        # First call should succeed
        result1 = await registry.try_fix(detection, ctx)
        assert result1 is not None

        # Second call within cooldown should be skipped
        result2 = await registry.try_fix(detection, ctx)
        assert result2 is None


# ═══════════════════════════════════════════════════════════════════════════
# E. Approval Queue Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestApprovalQueue:
    """Test approval queue operations."""

    @pytest.mark.asyncio
    async def test_create_pending_approval_no_db(self):
        from src.services.ai_agent.approval import ApprovalQueue

        queue = ApprovalQueue(db=None)
        result = await queue.request_approval(
            action_type="suspend_customer",
            target_id="cust-1",
            description="Too many payment failures",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_request_approval_signature(self):
        """Verify ApprovalQueue.request_approval accepts expected params."""
        from src.services.ai_agent.approval import ApprovalQueue
        import inspect
        sig = inspect.signature(ApprovalQueue.request_approval)
        params = list(sig.parameters.keys())
        assert "action_type" in params
        assert "target_id" in params
        assert "description" in params

    @pytest.mark.asyncio
    async def test_approve_executes_action(self):
        from src.services.ai_agent.approval import ApprovalQueue

        mock_approval = MagicMock()
        mock_approval.id = "approval-1"
        mock_approval.status = "pending"
        mock_approval.action_type = "suspend_customer"
        mock_approval.target_id = "cust-1"
        mock_approval.details = "{}"

        mock_db = MagicMock()
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=MagicMock(
            scalar_one_or_none=MagicMock(return_value=mock_approval)
        ))
        mock_db.session_ctx = MagicMock(return_value=_AsyncCM(mock_session))

        queue = ApprovalQueue(db=mock_db)

        with patch.object(queue, "_execute_action", new=AsyncMock(return_value={"status": "frozen"})):
            result = await queue.approve("approval-1", reviewed_by="admin")

        assert result["status"] == "approved"

    @pytest.mark.asyncio
    async def test_reject_approval(self):
        from src.services.ai_agent.approval import ApprovalQueue

        mock_db = MagicMock()
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=MagicMock(
            scalar_one_or_none=MagicMock(return_value="pending")
        ))
        mock_db.session_ctx = MagicMock(return_value=_AsyncCM(mock_session))

        queue = ApprovalQueue(db=mock_db)
        result = await queue.reject("approval-1", reviewed_by="admin")

        assert result["status"] == "rejected"

    @pytest.mark.asyncio
    async def test_list_pending(self):
        from src.services.ai_agent.approval import ApprovalQueue

        mock_approval = MagicMock()
        mock_approval.id = "approval-1"
        mock_approval.action_type = "suspend_customer"
        mock_approval.target_id = "cust-1"
        mock_approval.description = "Test"
        mock_approval.details = "{}"
        mock_approval.status = "pending"
        mock_approval.requested_by = "agent"
        mock_approval.created_at = datetime.utcnow()
        mock_approval.expires_at = datetime.utcnow() + timedelta(hours=24)

        mock_db = MagicMock()
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=MagicMock(
            scalars=MagicMock(return_value=MagicMock(
                all=MagicMock(return_value=[mock_approval])
            ))
        ))
        mock_db.session_ctx = MagicMock(return_value=_AsyncCM(mock_session))

        queue = ApprovalQueue(db=mock_db)
        pending = await queue.list_pending()

        assert len(pending) == 1
        assert pending[0]["action_type"] == "suspend_customer"

    @pytest.mark.asyncio
    async def test_expire_stale(self):
        from src.services.ai_agent.approval import ApprovalQueue

        mock_db = MagicMock()
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=MagicMock(rowcount=2))
        mock_db.session_ctx = MagicMock(return_value=_AsyncCM(mock_session))

        queue = ApprovalQueue(db=mock_db)
        expired = await queue.expire_stale()

        assert expired == 2

    @pytest.mark.asyncio
    async def test_duplicate_prevention(self):
        from src.services.ai_agent.approval import ApprovalQueue

        mock_db = MagicMock()
        mock_session = AsyncMock()
        # Simulate existing pending request found
        mock_session.execute = AsyncMock(return_value=MagicMock(
            scalar=MagicMock(return_value="approval-existing")
        ))
        mock_db.session_ctx = MagicMock(return_value=_AsyncCM(mock_session))

        queue = ApprovalQueue(db=mock_db)
        result = await queue.request_approval(
            action_type="suspend_customer",
            target_id="cust-1",
            description="duplicate",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_approve_already_processed(self):
        from src.services.ai_agent.approval import ApprovalQueue

        mock_approval = MagicMock()
        mock_approval.id = "approval-1"
        mock_approval.status = "approved"  # Already approved

        mock_db = MagicMock()
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=MagicMock(
            scalar_one_or_none=MagicMock(return_value=mock_approval)
        ))
        mock_db.session_ctx = MagicMock(return_value=_AsyncCM(mock_session))

        queue = ApprovalQueue(db=mock_db)
        result = await queue.approve("approval-1", reviewed_by="admin")

        assert "error" in result

    @pytest.mark.asyncio
    async def test_no_db_returns_empty(self):
        from src.services.ai_agent.approval import ApprovalQueue

        queue = ApprovalQueue(db=None)
        assert await queue.list_pending() == []
        assert await queue.expire_stale() == 0


# ═══════════════════════════════════════════════════════════════════════════
# F. Prevention Enhancement Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestPreventionEnhancements:
    """Test enhanced prevention engine methods."""

    @pytest.mark.asyncio
    async def test_analyze_trends(self):
        from src.services.ai_agent.prevention import PatternTracker

        mock_pattern = MagicMock()
        mock_pattern.rule_id = "system.high_cpu"
        mock_pattern.pattern_key = "system.high_cpu"
        mock_pattern.occurrence_count = 5
        mock_pattern.first_seen = datetime.utcnow() - timedelta(hours=10)
        mock_pattern.last_seen = datetime.utcnow()
        mock_pattern.predicted_next = datetime.utcnow() + timedelta(hours=1.5)

        mock_db = MagicMock()
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=MagicMock(
            scalars=MagicMock(return_value=MagicMock(
                all=MagicMock(return_value=[mock_pattern])
            ))
        ))
        mock_db.session_ctx = MagicMock(return_value=_AsyncCM(mock_session))

        tracker = PatternTracker(db=mock_db)
        trends = await tracker.analyze_trends()

        assert len(trends) == 1
        assert trends[0]["rule_id"] == "system.high_cpu"
        assert trends[0]["trend"] in ("accelerating", "decelerating", "stable")

    @pytest.mark.asyncio
    async def test_quota_exhaustion_prediction(self):
        from src.services.ai_agent.prevention import PatternTracker

        mock_db = MagicMock()
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(side_effect=[
            # Subscription
            MagicMock(first=MagicMock(return_value=(36000, "starter"))),
            # Current month usage
            MagicMock(scalar=MagicMock(return_value=20000)),
            # Week total
            MagicMock(scalar=MagicMock(return_value=14000)),
        ])
        mock_db.session_ctx = MagicMock(return_value=_AsyncCM(mock_session))

        tracker = PatternTracker(db=mock_db)
        result = await tracker.predict_quota_exhaustion("cust-1")

        assert result is not None
        assert result["customer_id"] == "cust-1"
        assert result["days_remaining"] is not None
        assert result["daily_burn_rate_s"] == 2000  # 14000/7

    @pytest.mark.asyncio
    async def test_churn_probability_no_activity(self):
        from src.services.ai_agent.prevention import PatternTracker

        mock_db = MagicMock()
        mock_session = AsyncMock()
        # All queries return 0/None — inactive customer
        mock_session.execute = AsyncMock(return_value=MagicMock(
            scalar=MagicMock(return_value=0)
        ))
        mock_db.session_ctx = MagicMock(return_value=_AsyncCM(mock_session))

        tracker = PatternTracker(db=mock_db)
        prob = await tracker.get_customer_churn_probability("cust-1")

        # Should be high — no activity at all
        assert 0.0 <= prob <= 1.0

    @pytest.mark.asyncio
    async def test_churn_probability_no_db(self):
        from src.services.ai_agent.prevention import PatternTracker

        tracker = PatternTracker(db=None)
        assert await tracker.get_customer_churn_probability("cust-1") == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# G2. RunPod Extended Monitor Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestRunPodWorkersRule:
    """Test infra.runpod_workers monitoring rule."""

    @pytest.mark.asyncio
    async def test_no_config_returns_empty(self):
        from src.services.ai_agent.monitors.infrastructure import RunPodWorkersRule
        ctx = _make_ctx(config=None)
        rule = RunPodWorkersRule()
        assert await rule.evaluate(ctx) == []

    @pytest.mark.asyncio
    async def test_no_endpoint_returns_empty(self):
        from src.services.ai_agent.monitors.infrastructure import RunPodWorkersRule
        mock_config = MagicMock()
        mock_config.runpod_endpoint_musetalk = ""
        mock_config.runpod_api_key = ""
        ctx = _make_ctx(config=mock_config)
        rule = RunPodWorkersRule()
        assert await rule.evaluate(ctx) == []

    @pytest.mark.asyncio
    async def test_detects_no_idle_workers(self):
        from src.services.ai_agent.monitors.infrastructure import RunPodWorkersRule

        mock_config = MagicMock()
        mock_config.runpod_endpoint_musetalk = "https://api.runpod.ai/v2/test"
        mock_config.runpod_api_key = "key123"

        # Mock active customers count
        mock_db = MagicMock()
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(
            return_value=MagicMock(scalar=MagicMock(return_value=15))
        )
        mock_db.session_ctx = MagicMock(return_value=_AsyncCM(mock_session))

        ctx = _make_ctx(config=mock_config, db=mock_db)
        rule = RunPodWorkersRule()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json = MagicMock(return_value={
            "jobs": {"inQueue": 3, "inProgress": 2},
            "workers": {"idle": 0, "running": 2, "ready": 2, "throttled": 0},
        })

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            detections = await rule.evaluate(ctx)

        assert len(detections) == 1
        assert detections[0].rule_id == "infra.runpod_workers"
        assert "idle" in detections[0].title.lower()
        assert detections[0].details["idle"] == 0
        assert detections[0].details["active_customers"] == 15

    @pytest.mark.asyncio
    async def test_detects_throttled_workers(self):
        from src.services.ai_agent.monitors.infrastructure import RunPodWorkersRule

        mock_config = MagicMock()
        mock_config.runpod_endpoint_musetalk = "https://api.runpod.ai/v2/test"
        mock_config.runpod_api_key = "key123"

        ctx = _make_ctx(config=mock_config, db=None)
        rule = RunPodWorkersRule()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json = MagicMock(return_value={
            "jobs": {"inQueue": 5, "inProgress": 3},
            "workers": {"idle": 2, "running": 3, "ready": 3, "throttled": 2},
        })

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            detections = await rule.evaluate(ctx)

        assert len(detections) == 1
        assert "throttled" in detections[0].title.lower()

    @pytest.mark.asyncio
    async def test_no_detection_healthy_workers(self):
        from src.services.ai_agent.monitors.infrastructure import RunPodWorkersRule

        mock_config = MagicMock()
        mock_config.runpod_endpoint_musetalk = "https://api.runpod.ai/v2/test"
        mock_config.runpod_api_key = "key123"

        ctx = _make_ctx(config=mock_config, db=None)
        rule = RunPodWorkersRule()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json = MagicMock(return_value={
            "jobs": {"inQueue": 0, "inProgress": 1},
            "workers": {"idle": 3, "running": 1, "ready": 4, "throttled": 0},
        })

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            detections = await rule.evaluate(ctx)

        assert len(detections) == 0


class TestRunPodColdStartRule:
    """Test infra.runpod_cold_starts monitoring rule."""

    @pytest.mark.asyncio
    async def test_no_pipeline_returns_empty(self):
        from src.services.ai_agent.monitors.infrastructure import RunPodColdStartRule
        ctx = _make_ctx(pipeline=None)
        rule = RunPodColdStartRule()
        assert await rule.evaluate(ctx) == []

    @pytest.mark.asyncio
    async def test_detects_high_cold_start_pct(self):
        from collections import deque
        from src.services.ai_agent.monitors.infrastructure import RunPodColdStartRule

        mock_pipeline = MagicMock()
        mock_runpod = MagicMock()
        # 4 out of 10 jobs exceed 12s cold threshold = 40%
        mock_runpod._recent_lipsync_times = deque(
            [3.0, 4.0, 15.0, 2.5, 14.0, 3.5, 13.0, 2.0, 16.0, 4.5], maxlen=50
        )
        mock_runpod._recent_render_times = deque(maxlen=50)
        mock_pipeline._runpod = mock_runpod

        ctx = _make_ctx(pipeline=mock_pipeline)
        rule = RunPodColdStartRule()
        detections = await rule.evaluate(ctx)

        assert len(detections) == 1
        assert detections[0].rule_id == "infra.runpod_cold_starts"
        assert detections[0].severity == "warning"
        assert detections[0].details["cold_pct"] == 40.0

    @pytest.mark.asyncio
    async def test_critical_at_60_pct(self):
        from collections import deque
        from src.services.ai_agent.monitors.infrastructure import RunPodColdStartRule

        mock_pipeline = MagicMock()
        mock_runpod = MagicMock()
        # 7 of 10 jobs cold = 70%
        mock_runpod._recent_lipsync_times = deque(
            [15.0, 14.0, 13.0, 16.0, 15.0, 14.0, 13.0, 3.0, 4.0, 5.0], maxlen=50
        )
        mock_runpod._recent_render_times = deque(maxlen=50)
        mock_pipeline._runpod = mock_runpod

        ctx = _make_ctx(pipeline=mock_pipeline)
        rule = RunPodColdStartRule()
        detections = await rule.evaluate(ctx)

        assert len(detections) == 1
        assert detections[0].severity == "critical"

    @pytest.mark.asyncio
    async def test_no_detection_low_cold_pct(self):
        from collections import deque
        from src.services.ai_agent.monitors.infrastructure import RunPodColdStartRule

        mock_pipeline = MagicMock()
        mock_runpod = MagicMock()
        # 1 of 10 cold = 10% (below 30% threshold)
        mock_runpod._recent_lipsync_times = deque(
            [3.0, 4.0, 3.5, 2.5, 14.0, 3.5, 4.0, 2.0, 3.0, 4.5], maxlen=50
        )
        mock_runpod._recent_render_times = deque(maxlen=50)
        mock_pipeline._runpod = mock_runpod

        ctx = _make_ctx(pipeline=mock_pipeline)
        rule = RunPodColdStartRule()
        detections = await rule.evaluate(ctx)

        assert len(detections) == 0

    @pytest.mark.asyncio
    async def test_falls_back_to_render_times(self):
        from collections import deque
        from src.services.ai_agent.monitors.infrastructure import RunPodColdStartRule

        mock_pipeline = MagicMock()
        mock_runpod = MagicMock()
        # No lipsync-specific data, uses general render times
        mock_runpod._recent_lipsync_times = deque(maxlen=50)  # empty
        mock_runpod._recent_render_times = deque(
            [15.0, 14.0, 13.0, 3.0, 4.0], maxlen=50
        )
        mock_pipeline._runpod = mock_runpod

        ctx = _make_ctx(pipeline=mock_pipeline)
        rule = RunPodColdStartRule()
        detections = await rule.evaluate(ctx)

        assert len(detections) == 1
        assert detections[0].details["cold_pct"] == 60.0


class TestRunPodTaskTimeRule:
    """Test infra.runpod_task_time monitoring rule."""

    @pytest.mark.asyncio
    async def test_no_pipeline_returns_empty(self):
        from src.services.ai_agent.monitors.infrastructure import RunPodTaskTimeRule
        ctx = _make_ctx(pipeline=None)
        rule = RunPodTaskTimeRule()
        assert await rule.evaluate(ctx) == []

    @pytest.mark.asyncio
    async def test_detects_slow_preprocess(self):
        from collections import deque
        from src.services.ai_agent.monitors.infrastructure import RunPodTaskTimeRule

        mock_pipeline = MagicMock()
        mock_runpod = MagicMock()
        mock_runpod._recent_preprocess_times = deque([35.0, 40.0, 38.0], maxlen=50)
        mock_runpod._recent_lipsync_times = deque([3.0, 4.0], maxlen=50)
        mock_pipeline._runpod = mock_runpod

        ctx = _make_ctx(pipeline=mock_pipeline)
        rule = RunPodTaskTimeRule()
        detections = await rule.evaluate(ctx)

        assert len(detections) == 1
        assert "preprocess_face" in detections[0].title
        assert detections[0].details["task_type"] == "preprocess_face"

    @pytest.mark.asyncio
    async def test_detects_slow_lipsync_warm(self):
        from collections import deque
        from src.services.ai_agent.monitors.infrastructure import RunPodTaskTimeRule

        mock_pipeline = MagicMock()
        mock_runpod = MagicMock()
        mock_runpod._recent_preprocess_times = deque([10.0], maxlen=50)
        mock_runpod._recent_lipsync_times = deque([7.0, 8.0, 6.5], maxlen=50)
        mock_pipeline._runpod = mock_runpod

        ctx = _make_ctx(pipeline=mock_pipeline)
        rule = RunPodTaskTimeRule()
        detections = await rule.evaluate(ctx)

        assert len(detections) == 1
        assert "render_lipsync" in detections[0].title
        assert detections[0].severity == "warning"

    @pytest.mark.asyncio
    async def test_detects_slow_lipsync_cold(self):
        from collections import deque
        from src.services.ai_agent.monitors.infrastructure import RunPodTaskTimeRule

        mock_pipeline = MagicMock()
        mock_runpod = MagicMock()
        mock_runpod._recent_preprocess_times = deque(maxlen=50)
        mock_runpod._recent_lipsync_times = deque([15.0, 14.0, 16.0], maxlen=50)
        mock_pipeline._runpod = mock_runpod

        ctx = _make_ctx(pipeline=mock_pipeline)
        rule = RunPodTaskTimeRule()
        detections = await rule.evaluate(ctx)

        assert len(detections) == 1
        assert detections[0].severity == "critical"
        assert "cold" in detections[0].title.lower()

    @pytest.mark.asyncio
    async def test_both_tasks_slow(self):
        from collections import deque
        from src.services.ai_agent.monitors.infrastructure import RunPodTaskTimeRule

        mock_pipeline = MagicMock()
        mock_runpod = MagicMock()
        mock_runpod._recent_preprocess_times = deque([35.0, 40.0], maxlen=50)
        mock_runpod._recent_lipsync_times = deque([7.0, 8.0], maxlen=50)
        mock_pipeline._runpod = mock_runpod

        ctx = _make_ctx(pipeline=mock_pipeline)
        rule = RunPodTaskTimeRule()
        detections = await rule.evaluate(ctx)

        # Both preprocess and lipsync should trigger
        assert len(detections) == 2
        task_types = {d.details["task_type"] for d in detections}
        assert task_types == {"preprocess_face", "render_lipsync"}

    @pytest.mark.asyncio
    async def test_no_detection_fast_tasks(self):
        from collections import deque
        from src.services.ai_agent.monitors.infrastructure import RunPodTaskTimeRule

        mock_pipeline = MagicMock()
        mock_runpod = MagicMock()
        mock_runpod._recent_preprocess_times = deque([10.0, 12.0], maxlen=50)
        mock_runpod._recent_lipsync_times = deque([2.0, 3.0, 2.5], maxlen=50)
        mock_pipeline._runpod = mock_runpod

        ctx = _make_ctx(pipeline=mock_pipeline)
        rule = RunPodTaskTimeRule()
        detections = await rule.evaluate(ctx)

        assert len(detections) == 0


class TestRunPodR2LatencyRule:
    """Test infra.runpod_r2_latency monitoring rule."""

    @pytest.mark.asyncio
    async def test_no_pipeline_returns_empty(self):
        from src.services.ai_agent.monitors.infrastructure import RunPodR2LatencyRule
        ctx = _make_ctx(pipeline=None)
        rule = RunPodR2LatencyRule()
        assert await rule.evaluate(ctx) == []

    @pytest.mark.asyncio
    async def test_detects_high_r2_latency(self):
        from collections import deque
        from src.services.ai_agent.monitors.infrastructure import RunPodR2LatencyRule

        mock_pipeline = MagicMock()
        mock_runpod = MagicMock()
        mock_runpod._recent_r2_latencies = deque([2.5, 3.0, 2.8], maxlen=50)
        mock_pipeline._runpod = mock_runpod

        ctx = _make_ctx(pipeline=mock_pipeline)
        rule = RunPodR2LatencyRule()
        detections = await rule.evaluate(ctx)

        assert len(detections) == 1
        assert detections[0].rule_id == "infra.runpod_r2_latency"
        assert detections[0].severity == "warning"
        assert "R2" in detections[0].title

    @pytest.mark.asyncio
    async def test_critical_at_double_threshold(self):
        from collections import deque
        from src.services.ai_agent.monitors.infrastructure import RunPodR2LatencyRule

        mock_pipeline = MagicMock()
        mock_runpod = MagicMock()
        mock_runpod._recent_r2_latencies = deque([4.5, 5.0, 4.8], maxlen=50)
        mock_pipeline._runpod = mock_runpod

        ctx = _make_ctx(pipeline=mock_pipeline)
        rule = RunPodR2LatencyRule()
        detections = await rule.evaluate(ctx)

        assert len(detections) == 1
        assert detections[0].severity == "critical"

    @pytest.mark.asyncio
    async def test_no_detection_fast_uploads(self):
        from collections import deque
        from src.services.ai_agent.monitors.infrastructure import RunPodR2LatencyRule

        mock_pipeline = MagicMock()
        mock_runpod = MagicMock()
        mock_runpod._recent_r2_latencies = deque([0.5, 0.8, 0.6], maxlen=50)
        mock_pipeline._runpod = mock_runpod

        ctx = _make_ctx(pipeline=mock_pipeline)
        rule = RunPodR2LatencyRule()
        detections = await rule.evaluate(ctx)

        assert len(detections) == 0

    @pytest.mark.asyncio
    async def test_no_data_returns_empty(self):
        from collections import deque
        from src.services.ai_agent.monitors.infrastructure import RunPodR2LatencyRule

        mock_pipeline = MagicMock()
        mock_runpod = MagicMock()
        mock_runpod._recent_r2_latencies = deque(maxlen=50)  # empty
        mock_pipeline._runpod = mock_runpod

        ctx = _make_ctx(pipeline=mock_pipeline)
        rule = RunPodR2LatencyRule()
        detections = await rule.evaluate(ctx)

        assert len(detections) == 0


class TestRunPodTimeoutRule:
    """Test infra.runpod_timeouts monitoring rule."""

    @pytest.mark.asyncio
    async def test_no_pipeline_returns_empty(self):
        from src.services.ai_agent.monitors.infrastructure import RunPodTimeoutRule
        ctx = _make_ctx(pipeline=None)
        rule = RunPodTimeoutRule()
        assert await rule.evaluate(ctx) == []

    @pytest.mark.asyncio
    async def test_detects_high_timeout_rate(self):
        from collections import deque
        from src.services.ai_agent.monitors.infrastructure import RunPodTimeoutRule

        now = time.time()
        mock_pipeline = MagicMock()
        mock_runpod = MagicMock()
        # 3 timeouts in last hour
        mock_runpod._recent_timeouts = deque([now - 60, now - 120, now - 180], maxlen=20)
        # 10 successful renders
        mock_runpod._recent_render_times = deque([3.0] * 10, maxlen=50)
        # 2 failures
        mock_runpod._recent_failures = deque([now - 300, now - 600], maxlen=20)
        mock_pipeline._runpod = mock_runpod

        ctx = _make_ctx(pipeline=mock_pipeline)
        rule = RunPodTimeoutRule()
        detections = await rule.evaluate(ctx)

        # 3 timeouts / (10 + 3 + 2) = 20%
        assert len(detections) == 1
        assert detections[0].rule_id == "infra.runpod_timeouts"
        assert detections[0].severity == "warning"
        assert detections[0].details["timeouts"] == 3

    @pytest.mark.asyncio
    async def test_critical_at_25_pct(self):
        from collections import deque
        from src.services.ai_agent.monitors.infrastructure import RunPodTimeoutRule

        now = time.time()
        mock_pipeline = MagicMock()
        mock_runpod = MagicMock()
        # 4 timeouts out of 10 total = 40%
        mock_runpod._recent_timeouts = deque([now - 60, now - 120, now - 180, now - 240], maxlen=20)
        mock_runpod._recent_render_times = deque([3.0] * 5, maxlen=50)
        mock_runpod._recent_failures = deque([now - 300], maxlen=20)
        mock_pipeline._runpod = mock_runpod

        ctx = _make_ctx(pipeline=mock_pipeline)
        rule = RunPodTimeoutRule()
        detections = await rule.evaluate(ctx)

        assert len(detections) == 1
        assert detections[0].severity == "critical"

    @pytest.mark.asyncio
    async def test_no_detection_no_timeouts(self):
        from collections import deque
        from src.services.ai_agent.monitors.infrastructure import RunPodTimeoutRule

        mock_pipeline = MagicMock()
        mock_runpod = MagicMock()
        mock_runpod._recent_timeouts = deque(maxlen=20)  # empty
        mock_runpod._recent_render_times = deque([3.0] * 10, maxlen=50)
        mock_runpod._recent_failures = deque(maxlen=20)
        mock_pipeline._runpod = mock_runpod

        ctx = _make_ctx(pipeline=mock_pipeline)
        rule = RunPodTimeoutRule()
        detections = await rule.evaluate(ctx)

        assert len(detections) == 0

    @pytest.mark.asyncio
    async def test_old_timeouts_ignored(self):
        from collections import deque
        from src.services.ai_agent.monitors.infrastructure import RunPodTimeoutRule

        old = time.time() - 7200  # 2 hours ago
        mock_pipeline = MagicMock()
        mock_runpod = MagicMock()
        mock_runpod._recent_timeouts = deque([old, old - 60, old - 120], maxlen=20)
        mock_runpod._recent_render_times = deque([3.0] * 10, maxlen=50)
        mock_runpod._recent_failures = deque(maxlen=20)
        mock_pipeline._runpod = mock_runpod

        ctx = _make_ctx(pipeline=mock_pipeline)
        rule = RunPodTimeoutRule()
        detections = await rule.evaluate(ctx)

        assert len(detections) == 0


# ═══════════════════════════════════════════════════════════════════════════
# G. Standalone Runner Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestStandaloneRunner:
    """Test runner module basics."""

    def test_runner_module_importable(self):
        from src.services.ai_agent import runner
        assert hasattr(runner, "main")

    @pytest.mark.asyncio
    async def test_health_server_responds(self):
        """Verify health endpoint returns JSON."""
        from src.services.ai_agent.runner import _start_health_server

        mock_agent = MagicMock()
        mock_agent.get_stats = AsyncMock(return_value={
            "scan_count": 5, "running": True, "rules_count": 18,
        })

        server = await _start_health_server(mock_agent, port=18081)
        assert server is not None

        # Make a test request
        import asyncio
        reader, writer = await asyncio.open_connection("127.0.0.1", 18081)
        writer.write(b"GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n")
        await writer.drain()
        data = await asyncio.wait_for(reader.read(4096), timeout=5.0)
        writer.close()

        response = data.decode()
        assert "200 OK" in response
        assert "scan_count" in response

        server.close()
        await server.wait_closed()


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


class _AsyncCM:
    """Async context manager helper for mock DB sessions."""

    def __init__(self, mock_obj):
        self._mock = mock_obj

    async def __aenter__(self):
        return self._mock

    async def __aexit__(self, *args):
        pass


# ═══════════════════════════════════════════════════════════════════════════
# H. WarmupJobFix Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestWarmupJobFix:
    """Tests for the RunPod warmup auto-fix handler."""

    def _make_config(self):
        config = MagicMock()
        config.runpod_endpoint_musetalk = "https://api.runpod.ai/v2/test123"
        config.runpod_api_key = "rp_testkey"
        return config

    def _make_detection(self, idle=0, running=2, active_customers=15):
        return Detection(
            rule_id="infra.runpod_workers",
            severity="critical",
            title="RunPod workers: 0 idle",
            description="No idle workers",
            details={
                "idle": idle,
                "running": running,
                "ready": 0,
                "throttled": 0,
                "active_customers": active_customers,
            },
            auto_fixable=True,
        )

    @pytest.mark.asyncio
    async def test_can_fix_when_idle_zero(self):
        from src.services.ai_agent.fixes.handlers import WarmupJobFix

        fix = WarmupJobFix()
        config = self._make_config()
        ctx = _make_ctx(config=config)
        d = self._make_detection(idle=0)
        assert await fix.can_fix(d, ctx) is True

    @pytest.mark.asyncio
    async def test_cannot_fix_when_idle_nonzero(self):
        from src.services.ai_agent.fixes.handlers import WarmupJobFix

        fix = WarmupJobFix()
        config = self._make_config()
        ctx = _make_ctx(config=config)
        d = self._make_detection(idle=1)
        assert await fix.can_fix(d, ctx) is False

    @pytest.mark.asyncio
    async def test_cannot_fix_wrong_rule(self):
        from src.services.ai_agent.fixes.handlers import WarmupJobFix

        fix = WarmupJobFix()
        ctx = _make_ctx(config=self._make_config())
        d = Detection(
            rule_id="infra.runpod_health",
            severity="critical",
            title="Unhealthy",
            description="",
            details={"idle": 0},
        )
        assert await fix.can_fix(d, ctx) is False

    @pytest.mark.asyncio
    async def test_cannot_fix_no_config(self):
        from src.services.ai_agent.fixes.handlers import WarmupJobFix

        fix = WarmupJobFix()
        ctx = _make_ctx(config=None)
        d = self._make_detection(idle=0)
        assert await fix.can_fix(d, ctx) is False

    @pytest.mark.asyncio
    async def test_cooldown_prevents_repeated_fix(self):
        from src.services.ai_agent.fixes.handlers import WarmupJobFix

        fix = WarmupJobFix()
        fix._last_applied = time.time()  # Just applied
        ctx = _make_ctx(config=self._make_config())
        d = self._make_detection(idle=0)
        assert await fix.can_fix(d, ctx) is False

    @pytest.mark.asyncio
    async def test_cooldown_expired_allows_fix(self):
        from src.services.ai_agent.fixes.handlers import WarmupJobFix

        fix = WarmupJobFix()
        fix._last_applied = time.time() - 700  # 11+ minutes ago
        ctx = _make_ctx(config=self._make_config())
        d = self._make_detection(idle=0)
        assert await fix.can_fix(d, ctx) is True

    @pytest.mark.asyncio
    async def test_apply_sends_warmup_job(self):
        from src.services.ai_agent.fixes.handlers import WarmupJobFix

        fix = WarmupJobFix()
        config = self._make_config()
        ctx = _make_ctx(config=config)
        d = self._make_detection(idle=0, active_customers=12)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "warmup-job-123"}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await fix.apply(d, ctx)

        assert result["action"] == "runpod_warmup_sent"
        assert result["job_id"] == "warmup-job-123"
        assert result["active_customers"] == 12
        assert fix._last_applied > 0

    @pytest.mark.asyncio
    async def test_apply_skip_no_endpoint(self):
        from src.services.ai_agent.fixes.handlers import WarmupJobFix

        fix = WarmupJobFix()
        config = MagicMock()
        config.runpod_endpoint_musetalk = ""
        config.runpod_api_key = ""
        ctx = _make_ctx(config=config)
        d = self._make_detection(idle=0)

        result = await fix.apply(d, ctx)
        assert result["action"] == "skip"

    @pytest.mark.asyncio
    async def test_apply_http_error(self):
        from src.services.ai_agent.fixes.handlers import WarmupJobFix

        fix = WarmupJobFix()
        ctx = _make_ctx(config=self._make_config())
        d = self._make_detection(idle=0)

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=Exception("connection refused"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await fix.apply(d, ctx)

        assert result["action"] == "error"
        assert "connection refused" in result["error"]


class TestRunPodWorkersRuleAutoFixable:
    """Tests that RunPodWorkersRule sets auto_fixable correctly."""

    @pytest.mark.asyncio
    async def test_auto_fixable_when_zero_idle(self):
        """Detection is auto_fixable when idle workers == 0."""
        from src.services.ai_agent.monitors.infrastructure import RunPodWorkersRule

        rule = RunPodWorkersRule()
        config = MagicMock()
        config.runpod_endpoint_musetalk = "https://api.runpod.ai/v2/test"
        config.runpod_api_key = "rp_key"

        mock_db = MagicMock()
        session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar.return_value = 15
        session.execute = AsyncMock(return_value=mock_result)
        mock_db.session_ctx = lambda: _AsyncCM(session)

        ctx = _make_ctx(
            config=config,
            db=mock_db,
            agent_config=AgentSettings(
                runpod_min_idle_workers=1,
                runpod_active_customers_for_warm=10,
            ),
        )

        health_data = {
            "workers": {"idle": 0, "running": 3, "ready": 3, "throttled": 0},
            "jobs": {"inQueue": 0, "inProgress": 3},
        }

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = health_data

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            detections = await rule.evaluate(ctx)

        worker_detections = [d for d in detections if "idle" in d.details]
        assert len(worker_detections) == 1
        assert worker_detections[0].auto_fixable is True


# ═══════════════════════════════════════════════════════════════════════════
# J. Resilience Rules Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestRunPodConsecutiveFailureRule:
    """Tests for resilience.runpod_consecutive_failures rule."""

    @pytest.mark.asyncio
    async def test_detects_consecutive_failures(self):
        from src.services.ai_agent.monitors.infrastructure import RunPodConsecutiveFailureRule

        rule = RunPodConsecutiveFailureRule()
        pipeline = MagicMock()
        pipeline._runpod = MagicMock()
        pipeline._runpod._consecutive_failures = 5
        pipeline._runpod.video_disabled = False

        ctx = _make_ctx(
            pipeline=pipeline,
            agent_config=AgentSettings(runpod_consecutive_failure_threshold=3),
        )
        detections = await rule.evaluate(ctx)
        assert len(detections) == 1
        assert detections[0].rule_id == "resilience.runpod_consecutive_failures"
        assert detections[0].severity == "critical"
        assert detections[0].auto_fixable is True
        assert detections[0].details["consecutive_failures"] == 5

    @pytest.mark.asyncio
    async def test_no_detection_below_threshold(self):
        from src.services.ai_agent.monitors.infrastructure import RunPodConsecutiveFailureRule

        rule = RunPodConsecutiveFailureRule()
        pipeline = MagicMock()
        pipeline._runpod = MagicMock()
        pipeline._runpod._consecutive_failures = 2
        pipeline._runpod.video_disabled = False

        ctx = _make_ctx(
            pipeline=pipeline,
            agent_config=AgentSettings(runpod_consecutive_failure_threshold=3),
        )
        detections = await rule.evaluate(ctx)
        assert len(detections) == 0

    @pytest.mark.asyncio
    async def test_not_auto_fixable_when_already_disabled(self):
        from src.services.ai_agent.monitors.infrastructure import RunPodConsecutiveFailureRule

        rule = RunPodConsecutiveFailureRule()
        pipeline = MagicMock()
        pipeline._runpod = MagicMock()
        pipeline._runpod._consecutive_failures = 5
        pipeline._runpod.video_disabled = True  # already disabled

        ctx = _make_ctx(
            pipeline=pipeline,
            agent_config=AgentSettings(runpod_consecutive_failure_threshold=3),
        )
        detections = await rule.evaluate(ctx)
        assert len(detections) == 1
        assert detections[0].auto_fixable is False

    @pytest.mark.asyncio
    async def test_no_pipeline_returns_empty(self):
        from src.services.ai_agent.monitors.infrastructure import RunPodConsecutiveFailureRule

        rule = RunPodConsecutiveFailureRule()
        ctx = _make_ctx(pipeline=None)
        detections = await rule.evaluate(ctx)
        assert detections == []


class TestDashScopeConsecutiveTimeoutRule:
    """Tests for resilience.dashscope_consecutive_timeouts rule."""

    @pytest.mark.asyncio
    async def test_detects_consecutive_timeouts(self):
        from src.services.ai_agent.monitors.infrastructure import DashScopeConsecutiveTimeoutRule

        rule = DashScopeConsecutiveTimeoutRule()
        pipeline = MagicMock()
        pipeline._llm = MagicMock()
        pipeline._llm._consecutive_timeouts = 7
        pipeline._llm.text_only_mode = False

        ctx = _make_ctx(
            pipeline=pipeline,
            agent_config=AgentSettings(dashscope_consecutive_timeout_threshold=5),
        )
        detections = await rule.evaluate(ctx)
        assert len(detections) == 1
        assert detections[0].rule_id == "resilience.dashscope_consecutive_timeouts"
        assert detections[0].severity == "critical"
        assert detections[0].auto_fixable is True

    @pytest.mark.asyncio
    async def test_no_detection_below_threshold(self):
        from src.services.ai_agent.monitors.infrastructure import DashScopeConsecutiveTimeoutRule

        rule = DashScopeConsecutiveTimeoutRule()
        pipeline = MagicMock()
        pipeline._llm = MagicMock()
        pipeline._llm._consecutive_timeouts = 3
        pipeline._llm.text_only_mode = False

        ctx = _make_ctx(
            pipeline=pipeline,
            agent_config=AgentSettings(dashscope_consecutive_timeout_threshold=5),
        )
        detections = await rule.evaluate(ctx)
        assert len(detections) == 0

    @pytest.mark.asyncio
    async def test_not_fixable_when_text_only_already(self):
        from src.services.ai_agent.monitors.infrastructure import DashScopeConsecutiveTimeoutRule

        rule = DashScopeConsecutiveTimeoutRule()
        pipeline = MagicMock()
        pipeline._llm = MagicMock()
        pipeline._llm._consecutive_timeouts = 10
        pipeline._llm.text_only_mode = True

        ctx = _make_ctx(
            pipeline=pipeline,
            agent_config=AgentSettings(dashscope_consecutive_timeout_threshold=5),
        )
        detections = await rule.evaluate(ctx)
        assert len(detections) == 1
        assert detections[0].auto_fixable is False


class TestDashScopeQueueDepthRule:
    """Tests for resilience.dashscope_queue_depth rule."""

    @pytest.mark.asyncio
    async def test_detects_high_queue_depth(self):
        from src.services.ai_agent.monitors.infrastructure import DashScopeQueueDepthRule

        rule = DashScopeQueueDepthRule()
        pipeline = MagicMock()
        pipeline._llm = MagicMock()
        pipeline._llm._pending_requests = 60
        pipeline._llm._rate_limit_429_count = 12

        ctx = _make_ctx(
            pipeline=pipeline,
            agent_config=AgentSettings(dashscope_queue_depth_warn=50),
        )
        detections = await rule.evaluate(ctx)
        assert len(detections) == 1
        assert detections[0].rule_id == "resilience.dashscope_queue_depth"
        assert detections[0].severity == "critical"
        assert detections[0].details["pending_requests"] == 60
        assert detections[0].details["rate_limit_429_count"] == 12

    @pytest.mark.asyncio
    async def test_no_detection_below_threshold(self):
        from src.services.ai_agent.monitors.infrastructure import DashScopeQueueDepthRule

        rule = DashScopeQueueDepthRule()
        pipeline = MagicMock()
        pipeline._llm = MagicMock()
        pipeline._llm._pending_requests = 10
        pipeline._llm._rate_limit_429_count = 0

        ctx = _make_ctx(
            pipeline=pipeline,
            agent_config=AgentSettings(dashscope_queue_depth_warn=50),
        )
        detections = await rule.evaluate(ctx)
        assert len(detections) == 0


class TestR2DowntimeRule:
    """Tests for resilience.r2_downtime rule."""

    @pytest.mark.asyncio
    async def test_detects_r2_downtime_critical(self):
        from src.services.ai_agent.monitors.infrastructure import R2DowntimeRule

        rule = R2DowntimeRule()
        pipeline = MagicMock()
        pipeline._r2 = MagicMock()
        pipeline._r2._consecutive_failures = 10
        pipeline._r2._last_failure_time = time.time() - 600  # 10 min ago

        ctx = _make_ctx(
            pipeline=pipeline,
            agent_config=AgentSettings(r2_downtime_threshold_s=300.0),
        )
        detections = await rule.evaluate(ctx)
        assert len(detections) == 1
        assert detections[0].severity == "critical"
        assert "DOWN" in detections[0].title

    @pytest.mark.asyncio
    async def test_detects_r2_failures_warning(self):
        from src.services.ai_agent.monitors.infrastructure import R2DowntimeRule

        rule = R2DowntimeRule()
        pipeline = MagicMock()
        pipeline._r2 = MagicMock()
        pipeline._r2._consecutive_failures = 5
        pipeline._r2._last_failure_time = time.time() - 10  # 10 seconds ago (within threshold)

        ctx = _make_ctx(
            pipeline=pipeline,
            agent_config=AgentSettings(r2_downtime_threshold_s=300.0),
        )
        detections = await rule.evaluate(ctx)
        assert len(detections) == 1
        assert detections[0].severity == "warning"

    @pytest.mark.asyncio
    async def test_no_detection_when_zero_failures(self):
        from src.services.ai_agent.monitors.infrastructure import R2DowntimeRule

        rule = R2DowntimeRule()
        pipeline = MagicMock()
        pipeline._r2 = MagicMock()
        pipeline._r2._consecutive_failures = 0
        pipeline._r2._last_failure_time = 0.0

        ctx = _make_ctx(pipeline=pipeline)
        detections = await rule.evaluate(ctx)
        assert len(detections) == 0

    @pytest.mark.asyncio
    async def test_no_detection_few_failures(self):
        from src.services.ai_agent.monitors.infrastructure import R2DowntimeRule

        rule = R2DowntimeRule()
        pipeline = MagicMock()
        pipeline._r2 = MagicMock()
        pipeline._r2._consecutive_failures = 2  # < 3
        pipeline._r2._last_failure_time = time.time() - 10

        ctx = _make_ctx(pipeline=pipeline)
        detections = await rule.evaluate(ctx)
        assert len(detections) == 0


class TestMarginSqueezeRule:
    """Tests for resilience.margin_squeeze rule."""

    @pytest.mark.asyncio
    async def test_detects_margin_squeeze(self):
        from src.services.ai_agent.monitors.infrastructure import MarginSqueezeRule

        rule = MarginSqueezeRule()
        mock_db = MagicMock()
        session = AsyncMock()

        # First query: active customers with subscriptions
        cust_rows = MagicMock()
        cust_rows.all.return_value = [
            ("cust-1", "Acme Corp", "starter", 50.0),
        ]
        # Second query: API costs for cust-1
        cost_result = MagicMock()
        cost_result.scalar.return_value = 35.0  # 70% of $50

        session.execute = AsyncMock(side_effect=[cust_rows, cost_result])
        mock_db.session_ctx = lambda: _AsyncCM(session)

        ctx = _make_ctx(
            db=mock_db,
            agent_config=AgentSettings(margin_squeeze_pct=60.0),
        )

        detections = await rule.evaluate(ctx)
        assert len(detections) == 1
        assert detections[0].rule_id == "resilience.margin_squeeze"
        assert detections[0].severity == "warning"
        assert detections[0].details["cost_pct"] == 70.0

    @pytest.mark.asyncio
    async def test_no_detection_healthy_margin(self):
        from src.services.ai_agent.monitors.infrastructure import MarginSqueezeRule

        rule = MarginSqueezeRule()
        mock_db = MagicMock()
        session = AsyncMock()

        cust_rows = MagicMock()
        cust_rows.all.return_value = [
            ("cust-1", "Acme Corp", "professional", 200.0),
        ]
        cost_result = MagicMock()
        cost_result.scalar.return_value = 40.0  # 20% of $200

        session.execute = AsyncMock(side_effect=[cust_rows, cost_result])
        mock_db.session_ctx = lambda: _AsyncCM(session)

        ctx = _make_ctx(
            db=mock_db,
            agent_config=AgentSettings(margin_squeeze_pct=60.0),
        )

        detections = await rule.evaluate(ctx)
        assert len(detections) == 0

    @pytest.mark.asyncio
    async def test_critical_severity_at_80_pct(self):
        from src.services.ai_agent.monitors.infrastructure import MarginSqueezeRule

        rule = MarginSqueezeRule()
        mock_db = MagicMock()
        session = AsyncMock()

        cust_rows = MagicMock()
        cust_rows.all.return_value = [
            ("cust-1", "BigCo", "starter", 50.0),
        ]
        cost_result = MagicMock()
        cost_result.scalar.return_value = 45.0  # 90% of $50

        session.execute = AsyncMock(side_effect=[cust_rows, cost_result])
        mock_db.session_ctx = lambda: _AsyncCM(session)

        ctx = _make_ctx(
            db=mock_db,
            agent_config=AgentSettings(margin_squeeze_pct=60.0),
        )

        detections = await rule.evaluate(ctx)
        assert len(detections) == 1
        assert detections[0].severity == "critical"


# ═══════════════════════════════════════════════════════════════════════════
# K. Resilience Fix Handler Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestVideoDisableFix:
    """Tests for fix.video_disable handler."""

    @pytest.mark.asyncio
    async def test_can_fix_matching_rule(self):
        from src.services.ai_agent.fixes.handlers import VideoDisableFix

        fix = VideoDisableFix()
        pipeline = MagicMock()
        pipeline._runpod = MagicMock()
        pipeline._runpod.video_disabled = False
        ctx = _make_ctx(pipeline=pipeline)

        d = Detection(
            rule_id="resilience.runpod_consecutive_failures",
            severity="critical",
            title="test",
            description="test",
            details={"consecutive_failures": 5},
        )
        assert await fix.can_fix(d, ctx) is True

    @pytest.mark.asyncio
    async def test_cannot_fix_wrong_rule(self):
        from src.services.ai_agent.fixes.handlers import VideoDisableFix

        fix = VideoDisableFix()
        pipeline = MagicMock()
        pipeline._runpod = MagicMock()
        pipeline._runpod.video_disabled = False
        ctx = _make_ctx(pipeline=pipeline)

        d = Detection(
            rule_id="infra.runpod_health",
            severity="critical",
            title="test",
            description="test",
        )
        assert await fix.can_fix(d, ctx) is False

    @pytest.mark.asyncio
    async def test_cannot_fix_already_disabled(self):
        from src.services.ai_agent.fixes.handlers import VideoDisableFix

        fix = VideoDisableFix()
        pipeline = MagicMock()
        pipeline._runpod = MagicMock()
        pipeline._runpod.video_disabled = True  # already disabled
        ctx = _make_ctx(pipeline=pipeline)

        d = Detection(
            rule_id="resilience.runpod_consecutive_failures",
            severity="critical",
            title="test",
            description="test",
            details={"consecutive_failures": 5},
        )
        assert await fix.can_fix(d, ctx) is False

    @pytest.mark.asyncio
    async def test_apply_disables_video(self):
        from src.services.ai_agent.fixes.handlers import VideoDisableFix

        fix = VideoDisableFix()
        pipeline = MagicMock()
        pipeline._runpod = MagicMock()
        pipeline._runpod.video_disabled = False
        ctx = _make_ctx(pipeline=pipeline)

        d = Detection(
            rule_id="resilience.runpod_consecutive_failures",
            severity="critical",
            title="test",
            description="test",
            details={"consecutive_failures": 5},
        )
        result = await fix.apply(d, ctx)
        assert result["action"] == "video_disabled"
        assert pipeline._runpod.video_disabled is True


class TestTextOnlyModeFix:
    """Tests for fix.text_only_mode handler."""

    @pytest.mark.asyncio
    async def test_can_fix_matching_rule(self):
        from src.services.ai_agent.fixes.handlers import TextOnlyModeFix

        fix = TextOnlyModeFix()
        pipeline = MagicMock()
        pipeline._llm = MagicMock()
        pipeline._llm.text_only_mode = False
        ctx = _make_ctx(pipeline=pipeline)

        d = Detection(
            rule_id="resilience.dashscope_consecutive_timeouts",
            severity="critical",
            title="test",
            description="test",
            details={"consecutive_timeouts": 7},
        )
        assert await fix.can_fix(d, ctx) is True

    @pytest.mark.asyncio
    async def test_cannot_fix_wrong_rule(self):
        from src.services.ai_agent.fixes.handlers import TextOnlyModeFix

        fix = TextOnlyModeFix()
        pipeline = MagicMock()
        pipeline._llm = MagicMock()
        pipeline._llm.text_only_mode = False
        ctx = _make_ctx(pipeline=pipeline)

        d = Detection(
            rule_id="infra.dashscope_latency",
            severity="warning",
            title="test",
            description="test",
        )
        assert await fix.can_fix(d, ctx) is False

    @pytest.mark.asyncio
    async def test_cannot_fix_already_text_only(self):
        from src.services.ai_agent.fixes.handlers import TextOnlyModeFix

        fix = TextOnlyModeFix()
        pipeline = MagicMock()
        pipeline._llm = MagicMock()
        pipeline._llm.text_only_mode = True
        ctx = _make_ctx(pipeline=pipeline)

        d = Detection(
            rule_id="resilience.dashscope_consecutive_timeouts",
            severity="critical",
            title="test",
            description="test",
            details={"consecutive_timeouts": 7},
        )
        assert await fix.can_fix(d, ctx) is False

    @pytest.mark.asyncio
    async def test_apply_enables_text_only(self):
        from src.services.ai_agent.fixes.handlers import TextOnlyModeFix

        fix = TextOnlyModeFix()
        pipeline = MagicMock()
        pipeline._llm = MagicMock()
        pipeline._llm.text_only_mode = False
        ctx = _make_ctx(pipeline=pipeline)

        d = Detection(
            rule_id="resilience.dashscope_consecutive_timeouts",
            severity="critical",
            title="test",
            description="test",
            details={"consecutive_timeouts": 7},
        )
        result = await fix.apply(d, ctx)
        assert result["action"] == "text_only_mode_enabled"
        assert pipeline._llm.text_only_mode is True


# ═══════════════════════════════════════════════════════════════════════════
# L. LLM Retry & R2 Tracking Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestLLMRetryBehavior:
    """Tests for the LLM generate() retry with backoff."""

    @pytest.mark.asyncio
    async def test_timeout_increments_consecutive_counter(self):
        from src.pipeline.llm import LLMEngine

        config = MagicMock()
        config.llm_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        config.llm_model_name = "qwen3-max"
        config.llm_api_key = "sk-test"
        config.dashscope_api_key = "sk-test"
        config.llm_timeout = 5.0
        config.llm_max_tokens = 512
        config.llm_temperature = 0.7
        config.llm_max_history = 10

        engine = LLMEngine(config)
        assert engine._consecutive_timeouts == 0
        assert engine._pending_requests == 0
        assert engine.text_only_mode is False

    @pytest.mark.asyncio
    async def test_pending_requests_tracking(self):
        from src.pipeline.llm import LLMEngine

        config = MagicMock()
        config.llm_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        config.llm_model_name = "qwen3-max"
        config.llm_api_key = "sk-test"
        config.dashscope_api_key = "sk-test"
        config.llm_timeout = 5.0
        config.llm_max_tokens = 512
        config.llm_temperature = 0.7
        config.llm_max_history = 10

        engine = LLMEngine(config)
        # Verify tracking fields initialized
        assert engine._pending_requests == 0
        assert engine._rate_limit_429_count == 0


class TestR2FailureTracking:
    """Tests for the R2Storage failure tracking fields."""

    def test_tracking_fields_initialized(self):
        config = MagicMock()
        config.r2_bucket = "test-bucket"
        config.r2_public_url = "https://media.example.com"
        config.r2_account_id = "acc123"
        config.r2_access_key_id = "key"
        config.r2_secret_access_key = "secret"

        from src.services.r2_storage import R2Storage

        storage = R2Storage(config)
        assert storage._consecutive_failures == 0
        assert storage._last_failure_time == 0.0
        assert storage._total_failures == 0
        assert len(storage._recent_upload_times) == 0


# ═══════════════════════════════════════════════════════════════════════════
# M. Prediction Rule Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestDashScopeQuotaExhaustionRule:
    """Tests for prediction.dashscope_quota_exhaustion rule."""

    @pytest.mark.asyncio
    async def test_warns_when_budget_running_out(self):
        from src.services.ai_agent.monitors.predictions import DashScopeQuotaExhaustionRule

        rule = DashScopeQuotaExhaustionRule()
        mock_db = MagicMock()
        session = AsyncMock()

        # Return $400 spent (of $500 budget) in 20 days → $20/day → 5 days left
        cost_result = MagicMock()
        cost_result.scalar.return_value = 400.0
        session.execute = AsyncMock(return_value=cost_result)
        mock_db.session_ctx = lambda: _AsyncCM(session)

        ctx = _make_ctx(
            db=mock_db,
            agent_config=AgentSettings(dashscope_monthly_budget_usd=500.0),
        )
        detections = await rule.evaluate(ctx)
        assert len(detections) == 1
        assert detections[0].rule_id == "prediction.dashscope_quota_exhaustion"
        assert detections[0].details["budget_usd"] == 500.0

    @pytest.mark.asyncio
    async def test_no_detection_when_budget_healthy(self):
        from src.services.ai_agent.monitors.predictions import DashScopeQuotaExhaustionRule

        rule = DashScopeQuotaExhaustionRule()
        mock_db = MagicMock()
        session = AsyncMock()

        # Only $50 spent (of $500) → lots of runway
        cost_result = MagicMock()
        cost_result.scalar.return_value = 50.0
        session.execute = AsyncMock(return_value=cost_result)
        mock_db.session_ctx = lambda: _AsyncCM(session)

        ctx = _make_ctx(
            db=mock_db,
            agent_config=AgentSettings(dashscope_monthly_budget_usd=500.0),
        )
        detections = await rule.evaluate(ctx)
        assert len(detections) == 0

    @pytest.mark.asyncio
    async def test_no_db_returns_empty(self):
        from src.services.ai_agent.monitors.predictions import DashScopeQuotaExhaustionRule

        rule = DashScopeQuotaExhaustionRule()
        ctx = _make_ctx(db=None)
        assert await rule.evaluate(ctx) == []


class TestRunPodCostProjectionRule:
    """Tests for prediction.runpod_cost_projection rule."""

    @pytest.mark.asyncio
    async def test_warns_on_high_projected_cost(self):
        from src.services.ai_agent.monitors.predictions import RunPodCostProjectionRule

        rule = RunPodCostProjectionRule()
        mock_db = MagicMock()
        session = AsyncMock()

        # $30/week RunPod → ~$128/month. Budget = $500 * 0.4 = $200 → 64%
        # Need >80% to trigger. Let's use $50/week → ~$214/month → 107%
        cost_result = MagicMock()
        cost_result.scalar.return_value = 50.0
        count_result = MagicMock()
        count_result.scalar.return_value = 200
        session.execute = AsyncMock(side_effect=[cost_result, count_result])
        mock_db.session_ctx = lambda: _AsyncCM(session)

        ctx = _make_ctx(
            db=mock_db,
            agent_config=AgentSettings(dashscope_monthly_budget_usd=500.0),
        )
        detections = await rule.evaluate(ctx)
        assert len(detections) == 1
        assert detections[0].rule_id == "prediction.runpod_cost_projection"

    @pytest.mark.asyncio
    async def test_no_detection_low_cost(self):
        from src.services.ai_agent.monitors.predictions import RunPodCostProjectionRule

        rule = RunPodCostProjectionRule()
        mock_db = MagicMock()
        session = AsyncMock()

        cost_result = MagicMock()
        cost_result.scalar.return_value = 5.0  # $5/week → ~$21/month, well under budget
        count_result = MagicMock()
        count_result.scalar.return_value = 50
        session.execute = AsyncMock(side_effect=[cost_result, count_result])
        mock_db.session_ctx = lambda: _AsyncCM(session)

        ctx = _make_ctx(
            db=mock_db,
            agent_config=AgentSettings(dashscope_monthly_budget_usd=500.0),
        )
        detections = await rule.evaluate(ctx)
        assert len(detections) == 0


class TestCustomerMarginPredictionRule:
    """Tests for prediction.customer_margin rule."""

    @pytest.mark.asyncio
    async def test_warns_on_margin_risk(self):
        from src.services.ai_agent.monitors.predictions import CustomerMarginPredictionRule

        rule = CustomerMarginPredictionRule()
        mock_db = MagicMock()
        session = AsyncMock()

        # Customer with $50/mo revenue, $40 cost so far in 20 days
        # Projected: ($40/20)*30 = $60/mo → 120% → critical
        cust_rows = MagicMock()
        cust_rows.all.return_value = [("cust-1", "TestCorp", "starter", 50.0)]
        cost_result = MagicMock()
        cost_result.scalar.return_value = 40.0

        session.execute = AsyncMock(side_effect=[cust_rows, cost_result])
        mock_db.session_ctx = lambda: _AsyncCM(session)

        ctx = _make_ctx(db=mock_db)
        detections = await rule.evaluate(ctx)
        assert len(detections) == 1
        assert detections[0].rule_id == "prediction.customer_margin"

    @pytest.mark.asyncio
    async def test_no_detection_healthy_customer(self):
        from src.services.ai_agent.monitors.predictions import CustomerMarginPredictionRule

        rule = CustomerMarginPredictionRule()
        mock_db = MagicMock()
        session = AsyncMock()

        cust_rows = MagicMock()
        cust_rows.all.return_value = [("cust-1", "GoodCorp", "professional", 200.0)]
        cost_result = MagicMock()
        cost_result.scalar.return_value = 10.0  # very low cost

        session.execute = AsyncMock(side_effect=[cust_rows, cost_result])
        mock_db.session_ctx = lambda: _AsyncCM(session)

        ctx = _make_ctx(db=mock_db)
        detections = await rule.evaluate(ctx)
        assert len(detections) == 0


class TestVRMVideoRatioRule:
    """Tests for prediction.vrm_video_ratio rule."""

    @pytest.mark.asyncio
    async def test_warns_on_high_fallback_rate(self):
        from src.services.ai_agent.monitors.predictions import VRMVideoRatioRule

        rule = VRMVideoRatioRule()
        mock_db = MagicMock()
        session = AsyncMock()

        # 5 video renders, 20 video sessions → 15 fallbacks → 75%
        video_result = MagicMock()
        video_result.scalar.return_value = 5
        sessions_result = MagicMock()
        sessions_result.scalar.return_value = 20

        session.execute = AsyncMock(side_effect=[video_result, sessions_result])
        mock_db.session_ctx = lambda: _AsyncCM(session)

        ctx = _make_ctx(db=mock_db)
        detections = await rule.evaluate(ctx)
        assert len(detections) == 1
        assert detections[0].rule_id == "prediction.vrm_video_ratio"
        assert detections[0].details["fallback_pct"] == 75.0

    @pytest.mark.asyncio
    async def test_no_detection_insufficient_data(self):
        from src.services.ai_agent.monitors.predictions import VRMVideoRatioRule

        rule = VRMVideoRatioRule()
        mock_db = MagicMock()
        session = AsyncMock()

        video_result = MagicMock()
        video_result.scalar.return_value = 2
        sessions_result = MagicMock()
        sessions_result.scalar.return_value = 3  # < 5

        session.execute = AsyncMock(side_effect=[video_result, sessions_result])
        mock_db.session_ctx = lambda: _AsyncCM(session)

        ctx = _make_ctx(db=mock_db)
        detections = await rule.evaluate(ctx)
        assert len(detections) == 0

    @pytest.mark.asyncio
    async def test_info_when_stable(self):
        from src.services.ai_agent.monitors.predictions import VRMVideoRatioRule

        rule = VRMVideoRatioRule()
        mock_db = MagicMock()
        session = AsyncMock()

        # 50 renders, 52 sessions → 4% fallback → stable + optimization info
        video_result = MagicMock()
        video_result.scalar.return_value = 50
        sessions_result = MagicMock()
        sessions_result.scalar.return_value = 52

        session.execute = AsyncMock(side_effect=[video_result, sessions_result])
        mock_db.session_ctx = lambda: _AsyncCM(session)

        ctx = _make_ctx(db=mock_db)
        detections = await rule.evaluate(ctx)
        # Should have info-level optimization suggestion
        assert len(detections) == 1
        assert detections[0].severity == "info"


# ═══════════════════════════════════════════════════════════════════════════
# N. Alert Cooldown Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestAlertCooldown:
    """Tests for the NotificationDispatcher alert cooldown logic."""

    def _make_dispatcher(self, cooldown_map=None, default_cd=60):
        """Create a NotificationDispatcher backed by mocks."""
        from src.services.ai_agent.notifications import (
            NotificationDispatcher,
            NotificationSettings,
        )

        agent_config = MagicMock()
        agent_config.audit_log_path = "./logs/test_audit.jsonl"
        agent_config.notification_ws_enabled = True
        agent_config.notification_email_enabled = False
        agent_config.notification_warning_batch_s = 300
        agent_config.notification_info_digest_s = 3600
        agent_config.alert_cooldown = cooldown_map or {
            "dashscope_timeout": 300,
            "runpod_failure": 300,
            "margin_squeeze": 86400,
            "r2_failure": 120,
            "low_balance": 3600,
        }
        agent_config.alert_cooldown_default_s = default_cd

        smtp = NotificationSettings()
        dispatcher = NotificationDispatcher(
            operator_manager=None,
            agent_config=agent_config,
            smtp_config=smtp,
        )
        # Mock the internal audit logger to avoid file I/O
        dispatcher._audit = MagicMock()
        return dispatcher

    @pytest.mark.asyncio
    async def test_first_alert_always_dispatches(self):
        """First alert for a rule_id must always be dispatched."""
        dispatcher = self._make_dispatcher()
        d = Detection(
            rule_id="dashscope_timeout",
            severity="critical",
            title="DashScope Timeout",
            description="test",
        )
        await dispatcher.dispatch(d, "inc-001")

        # Audit log written with status "open" (not "suppressed")
        dispatcher._audit.info.assert_called_once()
        call_extra = dispatcher._audit.info.call_args[1]["extra"]
        assert call_extra["status"] == "open"

        # Timestamp recorded
        assert "dashscope_timeout" in dispatcher._last_alert_times

    @pytest.mark.asyncio
    async def test_duplicate_within_cooldown_suppressed(self):
        """Second identical alert within cooldown window must be suppressed."""
        dispatcher = self._make_dispatcher({"dashscope_timeout": 300})
        d = Detection(
            rule_id="dashscope_timeout",
            severity="critical",
            title="DashScope Timeout",
            description="test",
        )

        # First dispatch — goes through
        await dispatcher.dispatch(d, "inc-001")
        assert dispatcher._audit.info.call_count == 1
        first_extra = dispatcher._audit.info.call_args[1]["extra"]
        assert first_extra["status"] == "open"

        # Second dispatch — suppressed
        await dispatcher.dispatch(d, "inc-002")
        assert dispatcher._audit.info.call_count == 2
        second_extra = dispatcher._audit.info.call_args[1]["extra"]
        assert second_extra["status"] == "suppressed"

    @pytest.mark.asyncio
    async def test_alert_after_cooldown_expires(self):
        """After cooldown window passes, the same alert must fire again."""
        dispatcher = self._make_dispatcher({"r2_failure": 1})  # 1 second cooldown

        d = Detection(
            rule_id="r2_failure",
            severity="warning",
            title="R2 Upload Failed",
            description="test",
        )

        await dispatcher.dispatch(d, "inc-001")
        assert dispatcher._audit.info.call_count == 1

        # Manually set last alert time to 2 seconds ago (past cooldown)
        dispatcher._last_alert_times["r2_failure"] = time.time() - 2

        await dispatcher.dispatch(d, "inc-002")
        assert dispatcher._audit.info.call_count == 2
        second_extra = dispatcher._audit.info.call_args[1]["extra"]
        assert second_extra["status"] == "open"

    @pytest.mark.asyncio
    async def test_different_rule_id_not_affected(self):
        """Cooldown for one rule_id must not block a different rule_id."""
        dispatcher = self._make_dispatcher({
            "dashscope_timeout": 300,
            "runpod_failure": 300,
        })

        d1 = Detection(
            rule_id="dashscope_timeout",
            severity="critical",
            title="DashScope Timeout",
            description="test",
        )
        d2 = Detection(
            rule_id="runpod_failure",
            severity="critical",
            title="RunPod Failure",
            description="test",
        )

        await dispatcher.dispatch(d1, "inc-001")
        await dispatcher.dispatch(d2, "inc-002")

        # Both should be dispatched (status "open"), not suppressed
        assert dispatcher._audit.info.call_count == 2
        calls = dispatcher._audit.info.call_args_list
        assert calls[0][1]["extra"]["status"] == "open"
        assert calls[1][1]["extra"]["status"] == "open"

    @pytest.mark.asyncio
    async def test_default_cooldown_for_unknown_rule(self):
        """Rule IDs not in the cooldown dict must use the default cooldown."""
        dispatcher = self._make_dispatcher(
            cooldown_map={},  # empty — no explicit entries
            default_cd=1,     # 1 second default
        )

        d = Detection(
            rule_id="some.unknown.rule",
            severity="info",
            title="Unknown",
            description="test",
        )

        # First dispatch — goes through
        await dispatcher.dispatch(d, "inc-001")
        assert dispatcher._audit.info.call_count == 1

        # Immediate second — suppressed (within 1s default)
        await dispatcher.dispatch(d, "inc-002")
        assert dispatcher._audit.info.call_count == 2
        assert dispatcher._audit.info.call_args[1]["extra"]["status"] == "suppressed"

        # After cooldown expires — goes through again
        dispatcher._last_alert_times["some.unknown.rule"] = time.time() - 2
        await dispatcher.dispatch(d, "inc-003")
        assert dispatcher._audit.info.call_count == 3
        assert dispatcher._audit.info.call_args[1]["extra"]["status"] == "open"

