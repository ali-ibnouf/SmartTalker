"""Tests for AnalyticsEngine."""

from __future__ import annotations

import importlib

import pytest

aiosqlite_available = importlib.util.find_spec("aiosqlite") is not None
needs_aiosqlite = pytest.mark.skipif(not aiosqlite_available, reason="aiosqlite not installed")


class TestAnalyticsEngine:
    """Tests for AnalyticsEngine."""

    def test_init(self, config):
        """Engine initializes without loading."""
        from src.pipeline.analytics import AnalyticsEngine
        engine = AnalyticsEngine(config)
        assert not engine.is_loaded

    @needs_aiosqlite
    @pytest.mark.asyncio
    async def test_load_creates_tables(self, config, tmp_path):
        """load() creates SQLite tables."""
        from src.pipeline.analytics import AnalyticsEngine
        config.training_db_path = tmp_path / "test_analytics.db"
        engine = AnalyticsEngine(config)
        await engine.load()
        assert engine.is_loaded

        cursor = await engine._sqlite_conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in await cursor.fetchall()}
        assert "analytics_snapshots" in tables

        await engine.unload()

    @needs_aiosqlite
    @pytest.mark.asyncio
    async def test_compute_kpis_empty(self, config, tmp_path):
        """KPIs with no data returns zeros."""
        from src.pipeline.analytics import AnalyticsEngine
        config.training_db_path = tmp_path / "test_analytics.db"
        engine = AnalyticsEngine(config)
        await engine.load()

        kpis = await engine.compute_kpis("avatar1")
        assert kpis.total_conversations == 0
        assert kpis.total_messages == 0
        assert kpis.escalation_rate == 0.0

        await engine.unload()

    @needs_aiosqlite
    @pytest.mark.asyncio
    async def test_timeseries_daily(self, config, tmp_path):
        """Timeseries returns list of points."""
        from src.pipeline.analytics import AnalyticsEngine
        config.training_db_path = tmp_path / "test_analytics.db"
        engine = AnalyticsEngine(config)
        await engine.load()

        points = await engine.get_timeseries("avatar1", metric="conversations", period="daily", days=7)
        assert isinstance(points, list)

        await engine.unload()

    @needs_aiosqlite
    @pytest.mark.asyncio
    async def test_timeseries_weekly(self, config, tmp_path):
        """Timeseries with weekly period."""
        from src.pipeline.analytics import AnalyticsEngine
        config.training_db_path = tmp_path / "test_analytics.db"
        engine = AnalyticsEngine(config)
        await engine.load()

        points = await engine.get_timeseries("avatar1", metric="autonomy", period="weekly", days=30)
        assert isinstance(points, list)

        await engine.unload()

    @needs_aiosqlite
    @pytest.mark.asyncio
    async def test_dashboard_data(self, config, tmp_path):
        """Dashboard data returns structured dict."""
        from src.pipeline.analytics import AnalyticsEngine
        config.training_db_path = tmp_path / "test_analytics.db"
        engine = AnalyticsEngine(config)
        await engine.load()

        data = await engine.get_dashboard_data("avatar1", days=30)
        assert "kpis" in data
        assert "trends" in data
        assert "top_skills" in data
        assert "bottom_skills" in data
        assert data["kpis"]["total_conversations"] >= 0

        await engine.unload()

    @needs_aiosqlite
    @pytest.mark.asyncio
    async def test_drift_detection_no_data(self, config, tmp_path):
        """Drift detection with no data returns empty list."""
        from src.pipeline.analytics import AnalyticsEngine
        config.training_db_path = tmp_path / "test_analytics.db"
        engine = AnalyticsEngine(config)
        await engine.load()

        alerts = await engine.check_drift("avatar1")
        assert isinstance(alerts, list)
        assert len(alerts) == 0

        await engine.unload()

    @needs_aiosqlite
    @pytest.mark.asyncio
    async def test_export_report(self, config, tmp_path):
        """Report export returns structured dict."""
        from src.pipeline.analytics import AnalyticsEngine
        config.training_db_path = tmp_path / "test_analytics.db"
        engine = AnalyticsEngine(config)
        await engine.load()

        report = await engine.export_report("avatar1", days=30)
        assert "kpis" in report
        assert "top_skills" in report
        assert "trends" in report

        await engine.unload()

    @needs_aiosqlite
    @pytest.mark.asyncio
    async def test_compute_kpis_with_training_data(self, config, tmp_path):
        """KPIs reflect data from training DB (SQLite fallback)."""
        from src.pipeline.analytics import AnalyticsEngine
        from src.pipeline.training import TrainingEngine

        config.training_db_path = tmp_path / "test_analytics.db"
        training = TrainingEngine(config, kb_engine=None)
        await training.load()

        # Create skill and data
        skill = await training.define_skill("avatar1", "Support")
        await training.learn_from_human("avatar1", skill.skill_id, "Q1", "A1", quality="good")
        await training.create_escalation("sess1", "avatar1", skill.skill_id, "Q2", 0.3)

        engine = AnalyticsEngine(config)
        await engine.load()

        kpis = await engine.compute_kpis("avatar1")
        # SQLite fallback: limited KPI support, but should not error
        assert kpis is not None

        await engine.unload()
        await training.unload()

    @needs_aiosqlite
    @pytest.mark.asyncio
    async def test_unload(self, config, tmp_path):
        """unload() closes connection and resets state."""
        from src.pipeline.analytics import AnalyticsEngine
        config.training_db_path = tmp_path / "test_analytics.db"
        engine = AnalyticsEngine(config)
        await engine.load()
        await engine.unload()
        assert not engine.is_loaded
