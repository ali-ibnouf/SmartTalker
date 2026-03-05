"""Tests for SupervisorEngine."""

from __future__ import annotations

import importlib

import pytest

aiosqlite_available = importlib.util.find_spec("aiosqlite") is not None
needs_aiosqlite = pytest.mark.skipif(not aiosqlite_available, reason="aiosqlite not installed")


class TestSupervisorEngine:
    """Tests for SupervisorEngine."""

    def test_init(self, config):
        """Engine initializes without loading."""
        from src.pipeline.supervisor import SupervisorEngine
        engine = SupervisorEngine(config)
        assert not engine.is_loaded

    @needs_aiosqlite
    @pytest.mark.asyncio
    async def test_load_creates_tables(self, config, tmp_path):
        """load() creates SQLite tables."""
        from src.pipeline.supervisor import SupervisorEngine
        config.training_db_path = tmp_path / "test_supervisor.db"
        engine = SupervisorEngine(config)
        await engine.load()
        assert engine.is_loaded

        cursor = await engine._sqlite_conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in await cursor.fetchall()}
        assert "operator_actions" in tables
        assert "decision_reviews" in tables

        await engine.unload()

    @needs_aiosqlite
    @pytest.mark.asyncio
    async def test_record_action(self, config, tmp_path):
        """Record an operator action."""
        from src.pipeline.supervisor import SupervisorEngine
        config.training_db_path = tmp_path / "test_supervisor.db"
        engine = SupervisorEngine(config)
        await engine.load()

        action_id = await engine.record_operator_action(
            operator_id="op_001",
            action_type="response",
            session_id="sess1",
            details={"text_length": 100},
        )
        assert action_id is not None

        await engine.unload()

    @needs_aiosqlite
    @pytest.mark.asyncio
    async def test_operator_metrics(self, config, tmp_path):
        """Get metrics for an operator."""
        from src.pipeline.supervisor import SupervisorEngine
        config.training_db_path = tmp_path / "test_supervisor.db"
        engine = SupervisorEngine(config)
        await engine.load()

        # Record some actions (use keyword arg for details)
        await engine.record_operator_action("op_001", "response", "sess1", details={"text_length": 100})
        await engine.record_operator_action("op_001", "response", "sess2", details={"text_length": 200})
        await engine.record_operator_action("op_001", "correction", "sess1", details={"quality": "bad"})

        metrics = await engine.get_operator_metrics("op_001", days=30)
        assert metrics.operator_id == "op_001"
        assert metrics.total_responses == 2
        assert metrics.corrections_made == 1

        await engine.unload()

    @needs_aiosqlite
    @pytest.mark.asyncio
    async def test_list_operators(self, config, tmp_path):
        """List all operator metrics."""
        from src.pipeline.supervisor import SupervisorEngine
        config.training_db_path = tmp_path / "test_supervisor.db"
        engine = SupervisorEngine(config)
        await engine.load()

        await engine.record_operator_action("op_001", "response", "sess1", details={})
        await engine.record_operator_action("op_002", "response", "sess2", details={})

        metrics_list = await engine.list_operator_metrics(days=30)
        assert len(metrics_list) == 2
        operator_ids = {m.operator_id for m in metrics_list}
        assert "op_001" in operator_ids
        assert "op_002" in operator_ids

        await engine.unload()

    @needs_aiosqlite
    @pytest.mark.asyncio
    async def test_flag_for_review(self, config, tmp_path):
        """Flag a decision for review."""
        from src.pipeline.supervisor import SupervisorEngine
        config.training_db_path = tmp_path / "test_supervisor.db"
        engine = SupervisorEngine(config)
        await engine.load()

        review_id = await engine.flag_for_review(
            session_id="sess1",
            avatar_id="avatar1",
            question="What is the return policy?",
            ai_response="We accept returns within 30 days.",
            confidence=0.45,
            reason="low_confidence",
        )
        assert review_id is not None
        assert len(review_id) > 0

        # Check it appears in queue
        queue = await engine.list_review_queue(reviewed=False)
        assert len(queue) == 1
        assert queue[0].id == review_id
        assert queue[0].question == "What is the return policy?"

        await engine.unload()

    @needs_aiosqlite
    @pytest.mark.asyncio
    async def test_submit_review(self, config, tmp_path):
        """Submit a review verdict."""
        from src.pipeline.supervisor import SupervisorEngine
        config.training_db_path = tmp_path / "test_supervisor.db"
        engine = SupervisorEngine(config)
        await engine.load()

        review_id = await engine.flag_for_review(
            session_id="sess1",
            avatar_id="avatar1",
            question="Test question",
            ai_response="Test response",
            confidence=0.5,
            reason="low_confidence",
        )

        result = await engine.submit_review(
            review_id=review_id,
            reviewer_id="op_001",
            verdict="approved",
        )
        assert result.reviewed is True
        assert result.review_verdict == "approved"
        assert result.reviewer_id == "op_001"

        # Check queue is empty now
        queue = await engine.list_review_queue(reviewed=False)
        assert len(queue) == 0

        await engine.unload()

    @needs_aiosqlite
    @pytest.mark.asyncio
    async def test_activity_timeline(self, config, tmp_path):
        """Activity timeline returns entries."""
        from src.pipeline.supervisor import SupervisorEngine
        config.training_db_path = tmp_path / "test_supervisor.db"
        engine = SupervisorEngine(config)
        await engine.load()

        await engine.record_operator_action("op_001", "response", "sess1", details={})
        await engine.record_operator_action("op_002", "correction", "sess2", details={})

        timeline = await engine.get_activity_timeline(days=7, limit=100)
        assert len(timeline) == 2
        # Most recent first
        assert timeline[0].operator_id in {"op_001", "op_002"}

        await engine.unload()

    @needs_aiosqlite
    @pytest.mark.asyncio
    async def test_quality_score(self, config, tmp_path):
        """Quality score is calculated correctly."""
        from src.pipeline.supervisor import SupervisorEngine
        config.training_db_path = tmp_path / "test_supervisor.db"
        engine = SupervisorEngine(config)
        await engine.load()

        # 9 responses, 1 correction → quality = 9/(9+1) = 0.9
        for i in range(9):
            await engine.record_operator_action("op_001", "response", f"sess{i}", details={})
        await engine.record_operator_action("op_001", "correction", "sess9", details={})

        metrics = await engine.get_operator_metrics("op_001")
        assert abs(metrics.quality_score - 0.9) < 0.01

        await engine.unload()
