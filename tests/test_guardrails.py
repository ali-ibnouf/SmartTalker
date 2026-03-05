"""Tests for GuardrailsEngine."""

from __future__ import annotations

import importlib

import pytest

aiosqlite_available = importlib.util.find_spec("aiosqlite") is not None
needs_aiosqlite = pytest.mark.skipif(not aiosqlite_available, reason="aiosqlite not installed")


@pytest.fixture
async def guardrails_engine(config, tmp_path):
    """Create a GuardrailsEngine with SQLite."""
    from src.pipeline.guardrails import GuardrailsEngine
    config.training_db_path = tmp_path / "test_guardrails.db"
    engine = GuardrailsEngine(config)
    await engine.load()
    yield engine
    await engine.unload()


class TestGuardrailsEngine:
    """Tests for GuardrailsEngine."""

    def test_init(self, config):
        """Engine initializes without loading."""
        from src.pipeline.guardrails import GuardrailsEngine
        engine = GuardrailsEngine(config)
        assert not engine.is_loaded

    @needs_aiosqlite
    @pytest.mark.asyncio
    async def test_load_creates_tables(self, config, tmp_path):
        """load() creates SQLite tables."""
        from src.pipeline.guardrails import GuardrailsEngine
        config.training_db_path = tmp_path / "test_guardrails.db"
        engine = GuardrailsEngine(config)
        await engine.load()
        assert engine.is_loaded

        cursor = await engine._sqlite_conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in await cursor.fetchall()}
        assert "guardrail_policies" in tables
        assert "policy_violations" in tables

        await engine.unload()

    @needs_aiosqlite
    @pytest.mark.asyncio
    async def test_default_policy(self, config, tmp_path):
        """Get policy with no custom policy returns defaults."""
        from src.pipeline.guardrails import GuardrailsEngine
        config.training_db_path = tmp_path / "test_guardrails.db"
        engine = GuardrailsEngine(config)
        await engine.load()

        policy = await engine.get_policy("avatar1")
        assert policy.max_response_length == 2000
        assert isinstance(policy.blocked_topics, list)

        await engine.unload()

    @needs_aiosqlite
    @pytest.mark.asyncio
    async def test_set_policy(self, config, tmp_path):
        """Set and retrieve a custom policy."""
        from src.pipeline.guardrails import GuardrailsEngine, PolicyConfig
        config.training_db_path = tmp_path / "test_guardrails.db"
        engine = GuardrailsEngine(config)
        await engine.load()

        policy = PolicyConfig(
            blocked_topics=["violence", "politics"],
            required_disclaimers=["This is AI-generated"],
            max_response_length=1000,
            escalation_keywords=["emergency"],
        )
        await engine.set_policy("avatar1", policy)

        retrieved = await engine.get_policy("avatar1")
        assert "violence" in retrieved.blocked_topics
        assert "politics" in retrieved.blocked_topics
        assert retrieved.max_response_length == 1000
        assert "emergency" in retrieved.escalation_keywords

        await engine.unload()

    @needs_aiosqlite
    @pytest.mark.asyncio
    async def test_delete_policy(self, config, tmp_path):
        """Delete a policy."""
        from src.pipeline.guardrails import GuardrailsEngine, PolicyConfig
        config.training_db_path = tmp_path / "test_guardrails.db"
        engine = GuardrailsEngine(config)
        await engine.load()

        await engine.set_policy("avatar1", PolicyConfig(blocked_topics=["test"]))
        deleted = await engine.delete_policy("avatar1")
        assert deleted is True

        # Delete non-existent
        deleted = await engine.delete_policy("avatar_nonexist")
        assert deleted is False

        await engine.unload()

    @needs_aiosqlite
    @pytest.mark.asyncio
    async def test_check_clean_response(self, config, tmp_path):
        """Clean response passes all checks."""
        from src.pipeline.guardrails import GuardrailsEngine
        config.training_db_path = tmp_path / "test_guardrails.db"
        engine = GuardrailsEngine(config)
        await engine.load()

        result = await engine.check_response(
            avatar_id="avatar1",
            session_id="sess1",
            response_text="Hello! How can I help you today?",
            user_question="Hi",
        )
        assert result.passed is True
        assert len(result.violations) == 0
        assert result.escalation_triggered is False

        await engine.unload()

    @needs_aiosqlite
    @pytest.mark.asyncio
    async def test_blocked_topic_violation(self, config, tmp_path):
        """Response containing blocked topic is flagged."""
        from src.pipeline.guardrails import GuardrailsEngine, PolicyConfig
        config.training_db_path = tmp_path / "test_guardrails.db"
        engine = GuardrailsEngine(config)
        await engine.load()

        await engine.set_policy("avatar1", PolicyConfig(
            blocked_topics=["competitor"],
        ))

        result = await engine.check_response(
            avatar_id="avatar1",
            session_id="sess1",
            response_text="Our competitor offers a similar product.",
            user_question="What about other companies?",
        )
        assert result.passed is False
        assert any(v.get("type") == "blocked_topic" for v in result.violations)

        await engine.unload()

    @needs_aiosqlite
    @pytest.mark.asyncio
    async def test_length_violation(self, config, tmp_path):
        """Response exceeding max length is flagged."""
        from src.pipeline.guardrails import GuardrailsEngine, PolicyConfig
        config.training_db_path = tmp_path / "test_guardrails.db"
        engine = GuardrailsEngine(config)
        await engine.load()

        await engine.set_policy("avatar1", PolicyConfig(
            max_response_length=50,
        ))

        long_text = "A" * 100
        result = await engine.check_response(
            avatar_id="avatar1",
            session_id="sess1",
            response_text=long_text,
            user_question="test",
        )
        assert result.passed is False
        assert any(v.get("type") == "length_exceeded" for v in result.violations)
        # Sanitized text should be truncated
        assert len(result.sanitized_text) <= 53  # 50 + "..."

        await engine.unload()

    @needs_aiosqlite
    @pytest.mark.asyncio
    async def test_escalation_keyword(self, config, tmp_path):
        """Response with escalation keyword triggers escalation."""
        from src.pipeline.guardrails import GuardrailsEngine, PolicyConfig
        config.training_db_path = tmp_path / "test_guardrails.db"
        engine = GuardrailsEngine(config)
        await engine.load()

        await engine.set_policy("avatar1", PolicyConfig(
            escalation_keywords=["emergency", "urgent"],
        ))

        result = await engine.check_response(
            avatar_id="avatar1",
            session_id="sess1",
            response_text="This is an emergency situation that needs attention.",
            user_question="Help",
        )
        assert result.escalation_triggered is True

        await engine.unload()

    @needs_aiosqlite
    @pytest.mark.asyncio
    async def test_disclaimer_injection(self, config, tmp_path):
        """Required disclaimers are appended if missing."""
        from src.pipeline.guardrails import GuardrailsEngine, PolicyConfig
        config.training_db_path = tmp_path / "test_guardrails.db"
        engine = GuardrailsEngine(config)
        await engine.load()

        await engine.set_policy("avatar1", PolicyConfig(
            required_disclaimers=["This is AI-generated content"],
        ))

        result = await engine.check_response(
            avatar_id="avatar1",
            session_id="sess1",
            response_text="Here is my answer.",
            user_question="test",
        )
        assert "This is AI-generated content" in result.sanitized_text
        assert len(result.disclaimers_added) > 0

        await engine.unload()

    @needs_aiosqlite
    @pytest.mark.asyncio
    async def test_violation_recording(self, config, tmp_path):
        """Violations are recorded and can be listed."""
        from src.pipeline.guardrails import GuardrailsEngine, PolicyConfig
        config.training_db_path = tmp_path / "test_guardrails.db"
        engine = GuardrailsEngine(config)
        await engine.load()

        await engine.set_policy("avatar1", PolicyConfig(
            blocked_topics=["forbidden"],
        ))

        # Trigger a violation
        await engine.check_response(
            avatar_id="avatar1",
            session_id="sess1",
            response_text="This is about forbidden topics.",
            user_question="test",
        )

        # List violations
        violations = await engine.list_violations("avatar1")
        assert len(violations) > 0
        assert violations[0].avatar_id == "avatar1"

        await engine.unload()
