"""Tests for SmartTalker Training engine."""

from __future__ import annotations

import importlib
from unittest.mock import AsyncMock, MagicMock

import pytest

aiosqlite_available = importlib.util.find_spec("aiosqlite") is not None
needs_aiosqlite = pytest.mark.skipif(not aiosqlite_available, reason="aiosqlite not installed")


# =============================================================================
# TrainingEngine Tests
# =============================================================================


class TestTrainingEngine:
    """Tests for TrainingEngine."""

    def test_init(self, config):
        """Engine initializes without loading."""
        from src.pipeline.training import TrainingEngine
        engine = TrainingEngine(config, kb_engine=None)
        assert not engine.is_loaded

    @needs_aiosqlite
    @pytest.mark.asyncio
    async def test_load_creates_tables(self, config, tmp_path):
        """load() creates SQLite tables."""
        from src.pipeline.training import TrainingEngine
        config.training_db_path = tmp_path / "test_training.db"
        engine = TrainingEngine(config, kb_engine=None)

        await engine.load()
        assert engine.is_loaded

        # Verify tables exist by trying to query them (SQLite fallback mode)
        cursor = await engine._sqlite_conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in await cursor.fetchall()}
        assert "skills" in tables
        assert "qa_pairs" in tables
        assert "escalations" in tables

        await engine.unload()

    @needs_aiosqlite
    @pytest.mark.asyncio
    async def test_unload(self, config, tmp_path):
        """unload() closes DB and resets state."""
        from src.pipeline.training import TrainingEngine
        config.training_db_path = tmp_path / "test_training.db"
        engine = TrainingEngine(config, kb_engine=None)
        await engine.load()
        await engine.unload()
        assert not engine.is_loaded

    @needs_aiosqlite
    @pytest.mark.asyncio
    async def test_define_skill(self, config, tmp_path):
        """define_skill creates a new skill record."""
        from src.pipeline.training import TrainingEngine
        config.training_db_path = tmp_path / "test_training.db"
        engine = TrainingEngine(config, kb_engine=None)
        await engine.load()

        skill = await engine.define_skill(
            avatar_id="test-avatar",
            name="Customer Greeting",
            description="How to greet customers",
            target_threshold=0.8,
        )

        assert skill.skill_id is not None
        assert skill.avatar_id == "test-avatar"
        assert skill.name == "Customer Greeting"
        assert skill.progress == 0.0

        await engine.unload()

    @needs_aiosqlite
    @pytest.mark.asyncio
    async def test_list_skills(self, config, tmp_path):
        """list_skills returns skills for avatar."""
        from src.pipeline.training import TrainingEngine
        config.training_db_path = tmp_path / "test_training.db"
        engine = TrainingEngine(config, kb_engine=None)
        await engine.load()

        await engine.define_skill("avatar-1", "Skill A", "Desc A")
        await engine.define_skill("avatar-1", "Skill B", "Desc B")
        await engine.define_skill("avatar-2", "Skill C", "Desc C")

        skills = await engine.list_skills("avatar-1")
        assert len(skills) == 2
        names = {s.name for s in skills}
        assert "Skill A" in names
        assert "Skill B" in names

        await engine.unload()

    @needs_aiosqlite
    @pytest.mark.asyncio
    async def test_delete_skill(self, config, tmp_path):
        """delete_skill removes a skill."""
        from src.pipeline.training import TrainingEngine
        config.training_db_path = tmp_path / "test_training.db"
        engine = TrainingEngine(config, kb_engine=None)
        await engine.load()

        skill = await engine.define_skill("avatar-1", "To Delete", "")
        deleted = await engine.delete_skill(skill.skill_id)
        assert deleted is True

        skills = await engine.list_skills("avatar-1")
        assert len(skills) == 0

        # Delete non-existent
        deleted2 = await engine.delete_skill("nonexistent")
        assert deleted2 is False

        await engine.unload()

    @needs_aiosqlite
    @pytest.mark.asyncio
    async def test_learn_from_human(self, config, tmp_path):
        """learn_from_human stores Q&A and updates progress."""
        from src.pipeline.training import TrainingEngine
        config.training_db_path = tmp_path / "test_training.db"

        mock_kb = MagicMock()
        mock_kb.ingest_text = AsyncMock(return_value=MagicMock(doc_id="d1", chunk_count=1))
        mock_kb.is_loaded = True

        engine = TrainingEngine(config, kb_engine=mock_kb)
        await engine.load()

        skill = await engine.define_skill("avatar-1", "Greetings", "How to greet")

        qa = await engine.learn_from_human(
            avatar_id="avatar-1",
            skill_id=skill.skill_id,
            question="How do I say hello?",
            human_answer="You say 'marhaba'",
            ai_answer="I'm not sure",
            quality="good",
        )

        assert qa.qa_id is not None
        assert qa.question == "How do I say hello?"
        assert qa.human_answer == "You say 'marhaba'"

        # KB should have been called to ingest
        mock_kb.ingest_text.assert_called_once()

        await engine.unload()

    @needs_aiosqlite
    @pytest.mark.asyncio
    async def test_skill_progress_updates(self, config, tmp_path):
        """Skill progress increases as Q&A pairs are added."""
        from src.pipeline.training import TrainingEngine
        config.training_db_path = tmp_path / "test_training.db"
        engine = TrainingEngine(config, kb_engine=None)
        await engine.load()

        skill = await engine.define_skill("avatar-1", "Products", "Product knowledge")

        # Add several Q&A pairs
        for i in range(5):
            await engine.learn_from_human(
                avatar_id="avatar-1",
                skill_id=skill.skill_id,
                question=f"Question {i}",
                human_answer=f"Answer {i}",
                quality="good",
            )

        # Check progress increased
        skills = await engine.list_skills("avatar-1")
        assert skills[0].progress > 0.0
        assert skills[0].qa_count == 5

        await engine.unload()

    @needs_aiosqlite
    @pytest.mark.asyncio
    async def test_escalation_create_and_resolve(self, config, tmp_path):
        """Escalations can be created and resolved."""
        from src.pipeline.training import TrainingEngine
        config.training_db_path = tmp_path / "test_training.db"
        engine = TrainingEngine(config, kb_engine=None)
        await engine.load()

        skill = await engine.define_skill("avatar-1", "Support", "")

        esc = await engine.create_escalation(
            session_id="session-123",
            avatar_id="avatar-1",
            skill_id=skill.skill_id,
            question="Complex question?",
            confidence=0.3,
        )

        assert esc.event_id is not None
        assert not esc.resolved

        # List unresolved
        escalations = await engine.list_escalations("avatar-1", unresolved_only=True)
        assert len(escalations) == 1

        # Resolve
        resolved = await engine.resolve_escalation(esc.event_id, "Here's the answer")
        assert resolved.resolved
        assert resolved.resolution == "Here's the answer"

        # Unresolved list should be empty
        escalations2 = await engine.list_escalations("avatar-1", unresolved_only=True)
        assert len(escalations2) == 0

        await engine.unload()

    @needs_aiosqlite
    @pytest.mark.asyncio
    async def test_should_escalate_low_confidence(self, config, tmp_path):
        """Low confidence triggers escalation."""
        from src.pipeline.training import TrainingEngine
        config.training_db_path = tmp_path / "test_training.db"
        config.training_escalation_threshold = 0.5
        engine = TrainingEngine(config, kb_engine=None)
        await engine.load()

        should, reason = await engine.should_escalate(
            avatar_id="avatar-1",
            question="Hard question",
            ai_confidence=0.2,
        )
        assert should is True

        await engine.unload()

    @needs_aiosqlite
    @pytest.mark.asyncio
    async def test_should_escalate_high_confidence(self, config, tmp_path):
        """High confidence does not trigger escalation."""
        from src.pipeline.training import TrainingEngine
        config.training_db_path = tmp_path / "test_training.db"
        config.training_escalation_threshold = 0.5
        engine = TrainingEngine(config, kb_engine=None)
        await engine.load()

        should, reason = await engine.should_escalate(
            avatar_id="avatar-1",
            question="Easy question",
            ai_confidence=0.9,
        )
        assert should is False

        await engine.unload()

    @needs_aiosqlite
    @pytest.mark.asyncio
    async def test_get_status(self, config, tmp_path):
        """get_status returns comprehensive training status."""
        from src.pipeline.training import TrainingEngine
        config.training_db_path = tmp_path / "test_training.db"
        engine = TrainingEngine(config, kb_engine=None)
        await engine.load()

        await engine.define_skill("avatar-1", "Skill A", "")
        await engine.define_skill("avatar-1", "Skill B", "")

        status = await engine.get_status("avatar-1")
        assert status.avatar_id == "avatar-1"
        assert len(status.skills) == 2
        assert status.overall_progress == 0.0
        assert not status.is_live

        await engine.unload()

    @needs_aiosqlite
    @pytest.mark.asyncio
    async def test_go_live_check(self, config, tmp_path):
        """check_go_live returns False when progress is low."""
        from src.pipeline.training import TrainingEngine
        config.training_db_path = tmp_path / "test_training.db"
        config.training_go_live_threshold = 100.0
        engine = TrainingEngine(config, kb_engine=None)
        await engine.load()

        is_live = await engine.check_go_live("avatar-1")
        assert is_live is False

        await engine.unload()

    def test_progress_calculation(self, config):
        """Progress formula returns correct values."""
        from src.pipeline.training import TrainingEngine
        engine = TrainingEngine(config, kb_engine=None)

        # 0 Q&A pairs = 0 progress
        assert engine._calculate_progress(0, 0) == 0.0

        # 50 pairs, all good = 100%
        progress = engine._calculate_progress(50, 50)
        assert progress == 100.0

        # 25 pairs, 20 good = partial
        progress = engine._calculate_progress(25, 20)
        assert 0.0 < progress < 100.0

        # Progress capped at 100
        progress = engine._calculate_progress(100, 100)
        assert progress == 100.0


# =============================================================================
# Training DB Integrity Tests
# =============================================================================


class TestTrainingDB:
    """Tests for SQLite database integrity."""

    @needs_aiosqlite
    @pytest.mark.asyncio
    async def test_tables_created(self, config, tmp_path):
        """All required tables are created on load."""
        from src.pipeline.training import TrainingEngine
        config.training_db_path = tmp_path / "test.db"
        engine = TrainingEngine(config, kb_engine=None)
        await engine.load()

        cursor = await engine._sqlite_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in await cursor.fetchall()]

        assert "skills" in tables
        assert "qa_pairs" in tables
        assert "escalations" in tables

        await engine.unload()

    @needs_aiosqlite
    @pytest.mark.asyncio
    async def test_concurrent_writes(self, config, tmp_path):
        """Multiple concurrent writes don't corrupt the database."""
        import asyncio
        from src.pipeline.training import TrainingEngine
        config.training_db_path = tmp_path / "test.db"
        engine = TrainingEngine(config, kb_engine=None)
        await engine.load()

        skill = await engine.define_skill("avatar-1", "Concurrent", "")

        async def write_qa(i):
            await engine.learn_from_human(
                avatar_id="avatar-1",
                skill_id=skill.skill_id,
                question=f"Q{i}",
                human_answer=f"A{i}",
                quality="good",
            )

        await asyncio.gather(*[write_qa(i) for i in range(10)])

        skills = await engine.list_skills("avatar-1")
        assert skills[0].qa_count == 10

        await engine.unload()
