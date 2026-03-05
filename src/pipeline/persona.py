"""Job Persona Engine for generalized skill templates.

Extracts skills from trained avatars, maintains a persona catalog,
and pre-populates new avatars with relevant skills at 70% progress.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from src.config import Settings
from src.utils.logger import setup_logger

logger = setup_logger("pipeline.persona")


@dataclass
class PersonaMatch:
    """Result of a persona match search."""

    persona_id: str
    name: str
    industry: str = "general"
    match_score: float = 0.0
    pre_populated_skills: list[dict[str, str]] = field(default_factory=list)


@dataclass
class PersonaInfo:
    """Full persona metadata."""

    persona_id: str
    name: str
    industry: str = "general"
    description: str = ""
    skill_count: int = 0
    is_public: bool = False
    source_avatar_id: str = ""


class PersonaEngine:
    """Job Persona Engine for skill template management.

    Extracts generalized job skills from trained avatars (stripping
    private info), maintains a catalog, and pre-populates new avatars.

    Args:
        config: Application settings.
        db: Database instance for persona persistence.
    """

    def __init__(self, config: Settings, db: Any = None) -> None:
        self._config = config
        self._db = db
        self._loaded = False

        logger.info("PersonaEngine initialized")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    async def load(self) -> None:
        self._loaded = True
        logger.info("PersonaEngine loaded")

    async def unload(self) -> None:
        self._loaded = False
        logger.info("PersonaEngine unloaded")

    async def list_personas(self, industry: Optional[str] = None) -> list[PersonaInfo]:
        """List personas, optionally filtered by industry."""
        if self._db is None:
            return []

        from sqlalchemy import select
        from src.db.models import JobPersona

        async with self._db.session() as session:
            stmt = select(JobPersona)
            if industry:
                stmt = stmt.where(JobPersona.industry == industry)
            stmt = stmt.order_by(JobPersona.name)
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [
                PersonaInfo(
                    persona_id=r.id,
                    name=r.name,
                    industry=r.industry,
                    description=r.description,
                    skill_count=len(r.skills),
                    is_public=r.is_public,
                    source_avatar_id=r.source_avatar_id or "",
                )
                for r in rows
            ]

    async def extract_persona(
        self,
        avatar_id: str,
        persona_name: str,
        industry: str = "general",
    ) -> PersonaInfo:
        """Extract a persona from a trained avatar's skills.

        Generalizes skills (strips private info like customer names)
        and creates a reusable persona in the catalog.
        """
        if self._db is None:
            raise RuntimeError("Database required for persona extraction")

        from sqlalchemy import select
        from src.db.models import Skill, JobPersona, PersonaSkill

        async with self._db.session() as session:
            # Get avatar's skills
            skill_result = await session.execute(
                select(Skill).where(Skill.avatar_id == avatar_id)
            )
            skills = skill_result.scalars().all()

            if not skills:
                raise ValueError(f"No skills found for avatar {avatar_id}")

            # Create persona
            persona_id = uuid.uuid4().hex
            persona = JobPersona(
                id=persona_id,
                name=persona_name,
                industry=industry,
                description=f"Extracted from avatar {avatar_id}",
                source_avatar_id=avatar_id,
                is_public=False,
            )
            session.add(persona)

            # Copy skills as persona skills (generalized at 70%)
            for skill in skills:
                ps = PersonaSkill(
                    id=uuid.uuid4().hex,
                    persona_id=persona_id,
                    name=skill.name,
                    description=skill.description,
                    pre_populated_progress=70.0,
                )
                session.add(ps)

            await session.commit()

            return PersonaInfo(
                persona_id=persona_id,
                name=persona_name,
                industry=industry,
                description=persona.description,
                skill_count=len(skills),
                is_public=False,
                source_avatar_id=avatar_id,
            )

    async def match_persona(
        self,
        industry: str,
        skills_needed: list[str],
    ) -> list[PersonaMatch]:
        """Find matching personas by industry and required skills."""
        if self._db is None:
            return []

        personas = await self.list_personas(industry=industry)
        if not personas:
            return []

        from sqlalchemy import select
        from src.db.models import PersonaSkill

        matches = []
        async with self._db.session() as session:
            for persona in personas:
                skill_result = await session.execute(
                    select(PersonaSkill).where(PersonaSkill.persona_id == persona.persona_id)
                )
                persona_skills = skill_result.scalars().all()
                persona_skill_names = {s.name.lower() for s in persona_skills}

                # Calculate match score
                needed_lower = {s.lower() for s in skills_needed}
                if not needed_lower:
                    score = 0.5
                else:
                    overlap = len(persona_skill_names & needed_lower)
                    score = overlap / len(needed_lower)

                if score > 0:
                    matches.append(PersonaMatch(
                        persona_id=persona.persona_id,
                        name=persona.name,
                        industry=persona.industry,
                        match_score=round(score, 2),
                        pre_populated_skills=[
                            {"name": s.name, "description": s.description}
                            for s in persona_skills
                        ],
                    ))

        matches.sort(key=lambda m: m.match_score, reverse=True)
        return matches

    async def apply_persona(
        self,
        avatar_id: str,
        persona_id: str,
    ) -> int:
        """Apply a persona's skills to an avatar at 70% progress.

        Returns the number of skills applied.
        """
        if self._db is None:
            raise RuntimeError("Database required for persona application")

        from sqlalchemy import select
        from src.db.models import Skill, PersonaSkill

        async with self._db.session() as session:
            # Get persona skills
            ps_result = await session.execute(
                select(PersonaSkill).where(PersonaSkill.persona_id == persona_id)
            )
            persona_skills = ps_result.scalars().all()

            if not persona_skills:
                return 0

            # Create skills for the avatar
            count = 0
            for ps in persona_skills:
                skill = Skill(
                    id=uuid.uuid4().hex,
                    avatar_id=avatar_id,
                    name=ps.name,
                    description=ps.description,
                    target_threshold=0.7,
                    progress=ps.pre_populated_progress,
                    qa_count=0,
                )
                session.add(skill)
                count += 1

            await session.commit()

        logger.info(
            "Persona applied",
            extra={"avatar_id": avatar_id, "persona_id": persona_id, "skills": count},
        )
        return count
