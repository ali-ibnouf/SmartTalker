"""Tests for LearningAnalytics engine."""

from __future__ import annotations

import importlib

import pytest

aiosqlite_available = importlib.util.find_spec("aiosqlite") is not None
needs_aiosqlite = pytest.mark.skipif(not aiosqlite_available, reason="aiosqlite not installed")


async def _setup_training_tables(config, tmp_path, db_name="test_analytics.db"):
    """Helper: create a TrainingEngine to set up qa_pairs/skills tables, then close it."""
    from src.pipeline.training import TrainingEngine
    config.training_db_path = tmp_path / db_name
    engine = TrainingEngine(config, kb_engine=None)
    await engine.load()
    return engine


class TestLearningAnalytics:
    """Tests for LearningAnalytics engine."""

    def test_init(self, config):
        """Engine initializes without loading."""
        from src.pipeline.learning_analytics import LearningAnalytics
        engine = LearningAnalytics(config)
        assert not engine.is_loaded

    @needs_aiosqlite
    @pytest.mark.asyncio
    async def test_load_creates_tables(self, config, tmp_path):
        """load() creates SQLite tables."""
        from src.pipeline.learning_analytics import LearningAnalytics
        config.training_db_path = tmp_path / "test_analytics.db"
        engine = LearningAnalytics(config)
        await engine.load()
        assert engine.is_loaded

        cursor = await engine._sqlite_conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in await cursor.fetchall()}
        assert "learning_metrics" in tables

        await engine.unload()

    @needs_aiosqlite
    @pytest.mark.asyncio
    async def test_quality_stats_empty(self, config, tmp_path):
        """Quality stats with no data returns zeros."""
        from src.pipeline.learning_analytics import LearningAnalytics

        # Need TrainingEngine to create qa_pairs/skills tables
        training = await _setup_training_tables(config, tmp_path)
        skill = await training.define_skill("avatar1", "Empty Skill")

        engine = LearningAnalytics(config)
        await engine.load()

        stats = await engine.get_skill_quality_stats("avatar1", skill.skill_id)
        assert stats.total_qa == 0
        assert stats.good_count == 0
        assert stats.bad_count == 0

        await engine.unload()
        await training.unload()

    @needs_aiosqlite
    @pytest.mark.asyncio
    async def test_quality_stats_with_data(self, config, tmp_path):
        """Quality stats reflect Q&A pairs from training DB."""
        from src.pipeline.learning_analytics import LearningAnalytics
        from src.pipeline.training import TrainingEngine

        config.training_db_path = tmp_path / "test_analytics.db"
        training = TrainingEngine(config, kb_engine=None)
        await training.load()

        # Create skill and add Q&A pairs
        skill = await training.define_skill("avatar1", "Greetings", target_threshold=0.7)
        await training.learn_from_human("avatar1", skill.skill_id, "Hi", "Hello!", quality="good")
        await training.learn_from_human("avatar1", skill.skill_id, "Hey", "Hi there!", quality="good")
        await training.learn_from_human("avatar1", skill.skill_id, "Yo", "Bad response", quality="bad")

        engine = LearningAnalytics(config)
        await engine.load()

        stats = await engine.get_skill_quality_stats("avatar1", skill.skill_id)
        assert stats.total_qa == 3
        assert stats.good_count == 2
        assert stats.bad_count == 1
        assert abs(stats.bad_ratio - 1 / 3) < 0.01

        await engine.unload()
        await training.unload()

    @needs_aiosqlite
    @pytest.mark.asyncio
    async def test_recalculate_threshold_raises(self, config, tmp_path):
        """Threshold recalculation with high bad ratio raises threshold."""
        from src.pipeline.learning_analytics import LearningAnalytics
        from src.pipeline.training import TrainingEngine

        config.training_db_path = tmp_path / "test_analytics.db"
        training = TrainingEngine(config, kb_engine=None)
        await training.load()

        skill = await training.define_skill("avatar1", "Returns", target_threshold=0.7)
        # Add many bad Q&A pairs to make bad_ratio > 0.3
        for i in range(5):
            await training.learn_from_human("avatar1", skill.skill_id, f"Q{i}", f"Bad{i}", quality="bad")
        await training.learn_from_human("avatar1", skill.skill_id, "Q6", "Good", quality="good")

        engine = LearningAnalytics(config)
        await engine.load()

        new_threshold = await engine.recalculate_effective_threshold(skill.skill_id)
        # bad_ratio > 0.3 so threshold should increase by 0.1
        assert new_threshold >= 0.8

        await engine.unload()
        await training.unload()

    @needs_aiosqlite
    @pytest.mark.asyncio
    async def test_daily_consolidation(self, config, tmp_path):
        """Daily consolidation returns result."""
        from src.pipeline.learning_analytics import LearningAnalytics

        # Need TrainingEngine for skills table
        training = await _setup_training_tables(config, tmp_path)

        engine = LearningAnalytics(config)
        await engine.load()

        result = await engine.consolidate_daily("avatar1")
        assert result.date is not None
        assert result.skills_updated >= 0

        await engine.unload()
        await training.unload()

    @needs_aiosqlite
    @pytest.mark.asyncio
    async def test_export_jsonl_empty(self, config, tmp_path):
        """Export with no data returns empty string."""
        from src.pipeline.learning_analytics import LearningAnalytics

        # Need TrainingEngine for qa_pairs table
        training = await _setup_training_tables(config, tmp_path)

        engine = LearningAnalytics(config)
        await engine.load()

        content = await engine.export_qa_pairs("avatar1")
        assert content == ""

        await engine.unload()
        await training.unload()

    @needs_aiosqlite
    @pytest.mark.asyncio
    async def test_improvement_timeline(self, config, tmp_path):
        """Timeline returns list of points."""
        from src.pipeline.learning_analytics import LearningAnalytics
        config.training_db_path = tmp_path / "test_analytics.db"
        engine = LearningAnalytics(config)
        await engine.load()

        timeline = await engine.get_improvement_timeline("avatar1", days=7)
        assert isinstance(timeline, list)

        await engine.unload()

    @needs_aiosqlite
    @pytest.mark.asyncio
    async def test_weak_areas(self, config, tmp_path):
        """Weak areas returns list."""
        from src.pipeline.learning_analytics import LearningAnalytics

        # Need TrainingEngine for qa_pairs table
        training = await _setup_training_tables(config, tmp_path)

        engine = LearningAnalytics(config)
        await engine.load()

        areas = await engine.get_weak_areas("avatar1", "skill1")
        assert isinstance(areas, list)

        await engine.unload()
        await training.unload()
