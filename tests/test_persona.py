"""Tests for PersonaEngine."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from src.pipeline.persona import PersonaEngine, PersonaInfo, PersonaMatch


@pytest.fixture
def persona_config():
    """Create a mock config."""
    return MagicMock()


@pytest.fixture
def persona_engine(persona_config):
    """Create a PersonaEngine with no DB."""
    return PersonaEngine(persona_config, db=None)


class TestPersonaEngineInit:
    """Test PersonaEngine initialization."""

    def test_init(self, persona_config):
        engine = PersonaEngine(persona_config)
        assert engine.is_loaded is False

    @pytest.mark.asyncio
    async def test_load_unload(self, persona_engine):
        assert persona_engine.is_loaded is False
        await persona_engine.load()
        assert persona_engine.is_loaded is True
        await persona_engine.unload()
        assert persona_engine.is_loaded is False


class TestPersonaNoDb:
    """Test PersonaEngine operations without database."""

    @pytest.mark.asyncio
    async def test_list_personas_no_db(self, persona_engine):
        result = await persona_engine.list_personas()
        assert result == []

    @pytest.mark.asyncio
    async def test_list_personas_with_industry_no_db(self, persona_engine):
        result = await persona_engine.list_personas(industry="tech")
        assert result == []

    @pytest.mark.asyncio
    async def test_extract_persona_no_db_raises(self, persona_engine):
        with pytest.raises(RuntimeError, match="Database required"):
            await persona_engine.extract_persona("avatar1", "Test Persona")

    @pytest.mark.asyncio
    async def test_match_persona_no_db(self, persona_engine):
        result = await persona_engine.match_persona("tech", ["python", "sql"])
        assert result == []

    @pytest.mark.asyncio
    async def test_apply_persona_no_db_raises(self, persona_engine):
        with pytest.raises(RuntimeError, match="Database required"):
            await persona_engine.apply_persona("avatar1", "persona1")


class TestPersonaDataclasses:
    """Test persona dataclass behavior."""

    def test_persona_info_defaults(self):
        info = PersonaInfo(persona_id="p1", name="Test")
        assert info.industry == "general"
        assert info.description == ""
        assert info.skill_count == 0
        assert info.is_public is False
        assert info.source_avatar_id == ""

    def test_persona_info_custom(self):
        info = PersonaInfo(
            persona_id="p1",
            name="Sales Rep",
            industry="retail",
            description="A sales persona",
            skill_count=5,
            is_public=True,
            source_avatar_id="avatar1",
        )
        assert info.industry == "retail"
        assert info.skill_count == 5
        assert info.is_public is True

    def test_persona_match_defaults(self):
        match = PersonaMatch(persona_id="p1", name="Test")
        assert match.industry == "general"
        assert match.match_score == 0.0
        assert match.pre_populated_skills == []

    def test_persona_match_with_skills(self):
        match = PersonaMatch(
            persona_id="p1",
            name="Test",
            match_score=0.85,
            pre_populated_skills=[
                {"name": "Greeting", "description": "Welcome customers"},
            ],
        )
        assert match.match_score == 0.85
        assert len(match.pre_populated_skills) == 1
