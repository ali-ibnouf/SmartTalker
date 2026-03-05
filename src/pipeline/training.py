"""Training Engine for skill tracking, escalation, and learning from humans.

Uses PostgreSQL (via SQLAlchemy async) for persistent storage of:
- Skill definitions and progress per avatar
- Q&A pairs captured from human operator responses
- Escalation events and their resolutions

Falls back to aiosqlite when no PostgreSQL Database is provided.
Integrates with KnowledgeBaseEngine for learning (ingest Q&A into KB).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from src.config import Settings
from src.utils.exceptions import TrainingError
from src.utils.logger import setup_logger, log_with_latency

logger = setup_logger("pipeline.training")


# ── Dataclasses ──────────────────────────────────────────────────────────────


@dataclass
class SkillDefinition:
    """A single trainable skill for an avatar."""

    skill_id: str
    avatar_id: str
    name: str
    description: str = ""
    target_threshold: float = 0.7
    effective_threshold: float = 0.7
    progress: float = 0.0
    qa_count: int = 0
    bad_ratio: float = 0.0
    created_at: float = 0.0
    updated_at: float = 0.0


@dataclass
class QAPair:
    """A question-answer pair captured from a human operator."""

    qa_id: str
    skill_id: str
    avatar_id: str
    question: str
    human_answer: str
    ai_answer: str = ""
    quality: str = "none"  # "good", "bad", "none"
    ingested: bool = False
    weight: float = 1.0
    correction_of: str = ""
    confidence_at_time: float = 0.0
    created_at: float = 0.0


@dataclass
class EscalationEvent:
    """An escalation event where AI deferred to a human."""

    event_id: str
    session_id: str
    avatar_id: str
    skill_id: str = "unknown"
    question: str = ""
    confidence: float = 0.0
    resolved: bool = False
    resolution: str = ""
    created_at: float = 0.0


@dataclass
class TrainingStatus:
    """Overall training status for an avatar."""

    avatar_id: str = "default"
    skills: list[SkillDefinition] = field(default_factory=list)
    overall_progress: float = 0.0
    is_live: bool = False
    total_qa_pairs: int = 0
    total_escalations: int = 0
    unresolved_escalations: int = 0
    latency_ms: int = 0


# Target number of Q&A pairs for 100% quantity score
_TARGET_QA_COUNT = 50


# ── Engine ───────────────────────────────────────────────────────────────────


class TrainingEngine:
    """Training engine for skill tracking, escalation, and human learning.

    Follows the standard SmartTalker engine pattern.
    Uses PostgreSQL via SQLAlchemy async when a Database instance is provided,
    falls back to aiosqlite otherwise.

    Args:
        config: Application settings.
        kb_engine: Reference to the KnowledgeBaseEngine (for learning ingestion).
        db: Optional Database instance for PostgreSQL.
    """

    def __init__(self, config: Settings, kb_engine: Any = None, db: Any = None, analytics: Any = None) -> None:
        self._config = config
        self._kb_engine = kb_engine
        self._db = db  # PostgreSQL Database instance
        self._analytics = analytics  # LearningAnalytics instance
        self._sqlite_conn: Any = None  # aiosqlite fallback
        self._loaded = False

        logger.info("TrainingEngine initialized")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ── Lifecycle ─────────────────────────────────────────────────────

    async def load(self) -> None:
        """Initialize database — uses PostgreSQL if available."""
        if self._db is not None:
            # PostgreSQL mode — tables are created by Database.connect()
            self._loaded = True
            logger.info("TrainingEngine loaded (PostgreSQL mode)")
            return

        # Fallback: aiosqlite
        try:
            import aiosqlite
        except ImportError:
            raise TrainingError(
                message="No database available",
                detail="Provide a PostgreSQL Database instance or install aiosqlite",
            )

        try:
            db_path = self._config.training_db_path
            db_path.parent.mkdir(parents=True, exist_ok=True)

            self._sqlite_conn = await aiosqlite.connect(str(db_path))
            self._sqlite_conn.row_factory = aiosqlite.Row
            await self._sqlite_conn.executescript(_SQLITE_CREATE_TABLES)
            await self._sqlite_conn.commit()

            self._loaded = True
            logger.info("TrainingEngine loaded (SQLite fallback)", extra={"db_path": str(db_path)})
        except Exception as exc:
            raise TrainingError(
                message="Failed to initialize Training Engine database",
                detail=str(exc),
                original_exception=exc,
            ) from exc

    async def unload(self) -> None:
        """Close database connection."""
        if self._sqlite_conn is not None:
            await self._sqlite_conn.close()
            self._sqlite_conn = None
        self._loaded = False
        logger.info("TrainingEngine unloaded")

    # ── Skill Management ──────────────────────────────────────────────

    async def define_skill(
        self,
        avatar_id: str,
        name: str,
        description: str = "",
        target_threshold: float = 0.7,
    ) -> SkillDefinition:
        """Define a new trainable skill for an avatar."""
        self._check_loaded()

        now = time.time()
        skill_id = f"skill_{uuid.uuid4().hex[:12]}"

        if self._db is not None:
            from src.db.models import Skill as SkillModel
            async with self._db.session() as session:
                skill_row = SkillModel(
                    id=skill_id,
                    avatar_id=avatar_id,
                    name=name,
                    description=description,
                    target_threshold=target_threshold,
                    progress=0.0,
                    qa_count=0,
                )
                session.add(skill_row)
                await session.commit()
        else:
            await self._sqlite_conn.execute(
                """INSERT INTO skills
                   (skill_id, avatar_id, name, description, target_threshold, progress, qa_count, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, 0.0, 0, ?, ?)""",
                (skill_id, avatar_id, name, description, target_threshold, now, now),
            )
            await self._sqlite_conn.commit()

        skill = SkillDefinition(
            skill_id=skill_id,
            avatar_id=avatar_id,
            name=name,
            description=description,
            target_threshold=target_threshold,
            progress=0.0,
            qa_count=0,
            created_at=now,
            updated_at=now,
        )
        logger.info(
            "Skill defined",
            extra={"skill_id": skill_id, "avatar_id": avatar_id, "skill_name": name},
        )
        return skill

    async def list_skills(self, avatar_id: str) -> list[SkillDefinition]:
        """List all skills for an avatar."""
        self._check_loaded()

        if self._db is not None:
            from sqlalchemy import select
            from src.db.models import Skill as SkillModel
            async with self._db.session() as session:
                result = await session.execute(
                    select(SkillModel).where(SkillModel.avatar_id == avatar_id).order_by(SkillModel.name)
                )
                rows = result.scalars().all()
                return [self._orm_to_skill(row) for row in rows]
        else:
            cursor = await self._sqlite_conn.execute(
                "SELECT * FROM skills WHERE avatar_id = ? ORDER BY name",
                (avatar_id,),
            )
            rows = await cursor.fetchall()
            return [self._row_to_skill(row) for row in rows]

    async def delete_skill(self, skill_id: str) -> bool:
        """Delete a skill and its associated Q&A pairs."""
        self._check_loaded()

        if self._db is not None:
            from sqlalchemy import delete
            from src.db.models import Skill as SkillModel, QAPair as QAModel
            async with self._db.session() as session:
                await session.execute(delete(QAModel).where(QAModel.skill_id == skill_id))
                result = await session.execute(delete(SkillModel).where(SkillModel.id == skill_id))
                await session.commit()
                deleted = result.rowcount > 0
        else:
            await self._sqlite_conn.execute(
                "DELETE FROM qa_pairs WHERE skill_id = ?", (skill_id,)
            )
            cursor = await self._sqlite_conn.execute(
                "DELETE FROM skills WHERE skill_id = ?", (skill_id,)
            )
            await self._sqlite_conn.commit()
            deleted = cursor.rowcount > 0

        if deleted:
            logger.info("Skill deleted", extra={"skill_id": skill_id})
        return deleted

    async def update_skill_progress(self, skill_id: str) -> SkillDefinition:
        """Recalculate skill progress based on Q&A count and quality."""
        self._check_loaded()

        if self._db is not None:
            from sqlalchemy import select, func as sa_func
            from src.db.models import Skill as SkillModel, QAPair as QAModel
            async with self._db.session() as session:
                result = await session.execute(
                    select(
                        sa_func.count(QAModel.id).label("total"),
                        sa_func.count(QAModel.id).filter(QAModel.quality == "good").label("good"),
                    ).where(QAModel.skill_id == skill_id)
                )
                row = result.one()
                total, good = row.total or 0, row.good or 0
                progress = self._calculate_progress(total, good)

                skill_result = await session.execute(
                    select(SkillModel).where(SkillModel.id == skill_id)
                )
                skill_row = skill_result.scalar_one_or_none()
                if skill_row is None:
                    raise TrainingError(message=f"Skill not found: {skill_id}")

                skill_row.progress = progress
                skill_row.qa_count = total
                await session.commit()
                await session.refresh(skill_row)
                return self._orm_to_skill(skill_row)
        else:
            cursor = await self._sqlite_conn.execute(
                "SELECT COUNT(*) as total, SUM(CASE WHEN quality = 'good' THEN 1 ELSE 0 END) as good "
                "FROM qa_pairs WHERE skill_id = ?",
                (skill_id,),
            )
            row = await cursor.fetchone()
            total = row[0] or 0
            good = row[1] or 0

            progress = self._calculate_progress(total, good)
            now = time.time()

            await self._sqlite_conn.execute(
                "UPDATE skills SET progress = ?, qa_count = ?, updated_at = ? WHERE skill_id = ?",
                (progress, total, now, skill_id),
            )
            await self._sqlite_conn.commit()

            cursor = await self._sqlite_conn.execute(
                "SELECT * FROM skills WHERE skill_id = ?", (skill_id,)
            )
            row = await cursor.fetchone()
            if row is None:
                raise TrainingError(message=f"Skill not found: {skill_id}")
            return self._row_to_skill(row)

    # ── Escalation ────────────────────────────────────────────────────

    async def should_escalate(
        self,
        avatar_id: str,
        question: str,
        ai_confidence: float,
    ) -> tuple[bool, str]:
        """Determine whether to escalate based on confidence and skill matching."""
        self._check_loaded()

        threshold = self._config.training_escalation_threshold

        if ai_confidence < 0.3:
            return True, "unknown"

        matched_skill = await self._match_skill(avatar_id, question)

        if matched_skill:
            # Use effective_threshold (auto-adjusted) if available, else target
            skill_threshold = matched_skill.effective_threshold or matched_skill.target_threshold
            if ai_confidence < skill_threshold:
                return True, matched_skill.skill_id
            if matched_skill.progress >= 80.0:
                return False, matched_skill.skill_id
        elif ai_confidence < threshold:
            return True, "unknown"

        return False, matched_skill.skill_id if matched_skill else "unknown"

    async def create_escalation(
        self,
        session_id: str,
        avatar_id: str,
        skill_id: str,
        question: str,
        confidence: float,
    ) -> EscalationEvent:
        """Record an escalation event."""
        self._check_loaded()

        now = time.time()
        event_id = f"esc_{uuid.uuid4().hex[:12]}"

        if self._db is not None:
            from src.db.models import Escalation as EscModel
            async with self._db.session() as session:
                esc_row = EscModel(
                    id=event_id,
                    session_id=session_id,
                    avatar_id=avatar_id,
                    skill_id=skill_id,
                    question=question,
                    confidence=confidence,
                    resolved=False,
                    resolution="",
                )
                session.add(esc_row)
                await session.commit()
        else:
            await self._sqlite_conn.execute(
                """INSERT INTO escalations
                   (event_id, session_id, avatar_id, skill_id, question, confidence, resolved, resolution, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, 0, '', ?)""",
                (event_id, session_id, avatar_id, skill_id, question, confidence, now),
            )
            await self._sqlite_conn.commit()

        event = EscalationEvent(
            event_id=event_id,
            session_id=session_id,
            avatar_id=avatar_id,
            skill_id=skill_id,
            question=question,
            confidence=confidence,
            created_at=now,
        )
        logger.info(
            "Escalation created",
            extra={"event_id": event_id, "confidence": confidence, "skill_id": skill_id},
        )
        return event

    async def resolve_escalation(
        self,
        event_id: str,
        resolution: str,
    ) -> EscalationEvent:
        """Resolve an escalation with the human's response."""
        self._check_loaded()

        if self._db is not None:
            from sqlalchemy import select
            from src.db.models import Escalation as EscModel
            async with self._db.session() as session:
                result = await session.execute(
                    select(EscModel).where(EscModel.id == event_id)
                )
                row = result.scalar_one_or_none()
                if row is None:
                    raise TrainingError(message=f"Escalation not found: {event_id}")
                row.resolved = True
                row.resolution = resolution
                await session.commit()
                return self._orm_to_escalation(row)
        else:
            await self._sqlite_conn.execute(
                "UPDATE escalations SET resolved = 1, resolution = ? WHERE event_id = ?",
                (resolution, event_id),
            )
            await self._sqlite_conn.commit()

            cursor = await self._sqlite_conn.execute(
                "SELECT * FROM escalations WHERE event_id = ?", (event_id,)
            )
            row = await cursor.fetchone()
            if row is None:
                raise TrainingError(message=f"Escalation not found: {event_id}")
            return self._row_to_escalation(row)

    async def list_escalations(
        self,
        avatar_id: str,
        unresolved_only: bool = False,
    ) -> list[EscalationEvent]:
        """List escalation events for an avatar."""
        self._check_loaded()

        if self._db is not None:
            from sqlalchemy import select
            from src.db.models import Escalation as EscModel
            async with self._db.session() as session:
                stmt = select(EscModel).where(EscModel.avatar_id == avatar_id)
                if unresolved_only:
                    stmt = stmt.where(EscModel.resolved == False)  # noqa: E712
                stmt = stmt.order_by(EscModel.created_at.desc())
                result = await session.execute(stmt)
                rows = result.scalars().all()
                return [self._orm_to_escalation(row) for row in rows]
        else:
            if unresolved_only:
                cursor = await self._sqlite_conn.execute(
                    "SELECT * FROM escalations WHERE avatar_id = ? AND resolved = 0 ORDER BY created_at DESC",
                    (avatar_id,),
                )
            else:
                cursor = await self._sqlite_conn.execute(
                    "SELECT * FROM escalations WHERE avatar_id = ? ORDER BY created_at DESC",
                    (avatar_id,),
                )
            rows = await cursor.fetchall()
            return [self._row_to_escalation(row) for row in rows]

    # ── Learning from Human ───────────────────────────────────────────

    async def learn_from_human(
        self,
        avatar_id: str,
        skill_id: str,
        question: str,
        human_answer: str,
        ai_answer: str = "",
        quality: str = "none",
        correction_of: str = "",
        confidence_at_time: float = 0.0,
    ) -> QAPair:
        """Record a Q&A pair and ingest into KB for learning."""
        self._check_loaded()

        if quality not in {"good", "bad", "none"}:
            raise TrainingError(message=f"Invalid quality: {quality}")

        now = time.time()
        qa_id = f"qa_{uuid.uuid4().hex[:12]}"
        ingested = False
        weight = 1.5 if quality == "good" else (0.5 if quality == "bad" else 1.0)

        if self._db is not None:
            from src.db.models import QAPair as QAModel
            async with self._db.session() as session:
                qa_row = QAModel(
                    id=qa_id,
                    skill_id=skill_id,
                    avatar_id=avatar_id,
                    question=question,
                    human_answer=human_answer,
                    ai_answer=ai_answer,
                    quality=quality,
                    ingested=False,
                    weight=weight,
                    correction_of=correction_of or None,
                    confidence_at_time=confidence_at_time or None,
                )
                session.add(qa_row)
                await session.commit()
        else:
            await self._sqlite_conn.execute(
                """INSERT INTO qa_pairs
                   (qa_id, skill_id, avatar_id, question, human_answer, ai_answer, quality, ingested, weight, correction_of, confidence_at_time, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?)""",
                (qa_id, skill_id, avatar_id, question, human_answer, ai_answer, quality, weight, correction_of or None, confidence_at_time or None, now),
            )
            await self._sqlite_conn.commit()

        # Ingest into KB with weight metadata
        if self._kb_engine is not None and self._kb_engine.is_loaded:
            try:
                qa_text = f"Question: {question}\nAnswer: {human_answer}"
                await self._kb_engine.ingest_text(
                    text=qa_text,
                    source_name=f"training_{skill_id}",
                    metadata={"skill_id": skill_id, "avatar_id": avatar_id, "qa_id": qa_id, "quality": quality, "weight": weight},
                )
                ingested = True
                if self._db is not None:
                    from sqlalchemy import update
                    from src.db.models import QAPair as QAModel2
                    async with self._db.session() as session:
                        await session.execute(
                            update(QAModel2).where(QAModel2.id == qa_id).values(ingested=True)
                        )
                        await session.commit()
                else:
                    await self._sqlite_conn.execute(
                        "UPDATE qa_pairs SET ingested = 1 WHERE qa_id = ?", (qa_id,)
                    )
                    await self._sqlite_conn.commit()
            except Exception as exc:
                logger.warning(
                    "Failed to ingest Q&A into KB",
                    extra={"qa_id": qa_id, "error": str(exc)},
                )

        # Update skill progress
        await self.update_skill_progress(skill_id)

        qa_pair = QAPair(
            qa_id=qa_id,
            skill_id=skill_id,
            avatar_id=avatar_id,
            question=question,
            human_answer=human_answer,
            ai_answer=ai_answer,
            quality=quality,
            ingested=ingested,
            weight=weight,
            correction_of=correction_of,
            confidence_at_time=confidence_at_time,
            created_at=now,
        )

        # Auto-adjust threshold via LearningAnalytics
        if self._analytics is not None and self._analytics.is_loaded:
            try:
                await self._analytics.recalculate_effective_threshold(skill_id)
            except Exception as exc:
                logger.warning("Failed to recalculate threshold: %s", exc)

        log_with_latency(
            logger,
            "Learned from human",
            0,
            extra={"qa_id": qa_id, "skill_id": skill_id, "quality": quality, "weight": weight},
        )
        return qa_pair

    async def list_qa_pairs(
        self,
        avatar_id: str,
        skill_id: Optional[str] = None,
    ) -> list[QAPair]:
        """List Q&A pairs, optionally filtered by skill."""
        self._check_loaded()

        if self._db is not None:
            from sqlalchemy import select
            from src.db.models import QAPair as QAModel
            async with self._db.session() as session:
                stmt = select(QAModel).where(QAModel.avatar_id == avatar_id)
                if skill_id:
                    stmt = stmt.where(QAModel.skill_id == skill_id)
                stmt = stmt.order_by(QAModel.created_at.desc())
                result = await session.execute(stmt)
                rows = result.scalars().all()
                return [self._orm_to_qa(row) for row in rows]
        else:
            if skill_id:
                cursor = await self._sqlite_conn.execute(
                    "SELECT * FROM qa_pairs WHERE avatar_id = ? AND skill_id = ? ORDER BY created_at DESC",
                    (avatar_id, skill_id),
                )
            else:
                cursor = await self._sqlite_conn.execute(
                    "SELECT * FROM qa_pairs WHERE avatar_id = ? ORDER BY created_at DESC",
                    (avatar_id,),
                )
            rows = await cursor.fetchall()
            return [self._row_to_qa(row) for row in rows]

    # ── Training Status ───────────────────────────────────────────────

    async def get_status(self, avatar_id: str) -> TrainingStatus:
        """Get comprehensive training status for an avatar."""
        self._check_loaded()
        start = time.perf_counter()

        skills = await self.list_skills(avatar_id)

        if self._db is not None:
            from sqlalchemy import select, func as sa_func
            from src.db.models import QAPair as QAModel, Escalation as EscModel
            async with self._db.session() as session:
                qa_result = await session.execute(
                    select(sa_func.count(QAModel.id)).where(QAModel.avatar_id == avatar_id)
                )
                total_qa = qa_result.scalar() or 0

                esc_result = await session.execute(
                    select(sa_func.count(EscModel.id)).where(EscModel.avatar_id == avatar_id)
                )
                total_esc = esc_result.scalar() or 0

                unresolved_result = await session.execute(
                    select(sa_func.count(EscModel.id)).where(
                        EscModel.avatar_id == avatar_id, EscModel.resolved == False  # noqa: E712
                    )
                )
                unresolved_esc = unresolved_result.scalar() or 0
        else:
            cursor = await self._sqlite_conn.execute(
                "SELECT COUNT(*) FROM qa_pairs WHERE avatar_id = ?", (avatar_id,)
            )
            total_qa = (await cursor.fetchone())[0]

            cursor = await self._sqlite_conn.execute(
                "SELECT COUNT(*) FROM escalations WHERE avatar_id = ?", (avatar_id,)
            )
            total_esc = (await cursor.fetchone())[0]

            cursor = await self._sqlite_conn.execute(
                "SELECT COUNT(*) FROM escalations WHERE avatar_id = ? AND resolved = 0",
                (avatar_id,),
            )
            unresolved_esc = (await cursor.fetchone())[0]

        overall = 0.0
        if skills:
            overall = sum(s.progress for s in skills) / len(skills)

        is_live = overall >= self._config.training_go_live_threshold

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return TrainingStatus(
            avatar_id=avatar_id,
            skills=skills,
            overall_progress=round(overall, 1),
            is_live=is_live,
            total_qa_pairs=total_qa,
            total_escalations=total_esc,
            unresolved_escalations=unresolved_esc,
            latency_ms=elapsed_ms,
        )

    async def check_go_live(self, avatar_id: str) -> bool:
        """Check if avatar has reached go-live threshold."""
        status = await self.get_status(avatar_id)
        return status.is_live

    # ── Internal ──────────────────────────────────────────────────────

    def _check_loaded(self) -> None:
        if not self._loaded:
            raise TrainingError(message="Training engine not loaded")

    async def _match_skill(
        self, avatar_id: str, question: str
    ) -> Optional[SkillDefinition]:
        """Match a question to a skill by keyword overlap."""
        skills = await self.list_skills(avatar_id)
        if not skills:
            return None

        question_lower = question.lower()
        question_words = set(question_lower.split())

        best_match: Optional[SkillDefinition] = None
        best_score = 0

        for skill in skills:
            skill_words = set(skill.name.lower().split())
            if skill.description:
                skill_words.update(skill.description.lower().split())

            overlap = len(question_words & skill_words)
            if overlap > best_score:
                best_score = overlap
                best_match = skill

        return best_match if best_score > 0 else None

    @staticmethod
    def _calculate_progress(qa_count: int, good_count: int) -> float:
        """Calculate skill progress from Q&A metrics."""
        if qa_count == 0:
            return 0.0
        quantity_score = min(qa_count / _TARGET_QA_COUNT, 1.0) * 70.0
        quality_ratio = good_count / qa_count if qa_count > 0 else 0.0
        quality_score = quality_ratio * 30.0
        return min(quantity_score + quality_score, 100.0)

    # ── ORM conversions (PostgreSQL) ──────────────────────────────────

    @staticmethod
    def _orm_to_skill(row: Any) -> SkillDefinition:
        return SkillDefinition(
            skill_id=row.id,
            avatar_id=row.avatar_id,
            name=row.name,
            description=row.description or "",
            target_threshold=row.target_threshold,
            effective_threshold=getattr(row, "effective_threshold", 0.7) or 0.7,
            bad_ratio=getattr(row, "bad_ratio", 0.0) or 0.0,
            progress=row.progress,
            qa_count=row.qa_count,
            created_at=row.created_at.timestamp() if row.created_at else 0.0,
            updated_at=row.updated_at.timestamp() if row.updated_at else 0.0,
        )

    @staticmethod
    def _orm_to_qa(row: Any) -> QAPair:
        return QAPair(
            qa_id=row.id,
            skill_id=row.skill_id,
            avatar_id=row.avatar_id,
            question=row.question,
            human_answer=row.human_answer,
            ai_answer=row.ai_answer or "",
            quality=row.quality or "none",
            ingested=bool(row.ingested),
            weight=getattr(row, "weight", 1.0) or 1.0,
            correction_of=getattr(row, "correction_of", "") or "",
            confidence_at_time=getattr(row, "confidence_at_time", 0.0) or 0.0,
            created_at=row.created_at.timestamp() if row.created_at else 0.0,
        )

    @staticmethod
    def _orm_to_escalation(row: Any) -> EscalationEvent:
        return EscalationEvent(
            event_id=row.id,
            session_id=row.session_id,
            avatar_id=row.avatar_id,
            skill_id=row.skill_id or "unknown",
            question=row.question or "",
            confidence=row.confidence,
            resolved=bool(row.resolved),
            resolution=row.resolution or "",
            created_at=row.created_at.timestamp() if row.created_at else 0.0,
        )

    # ── SQLite row conversions (legacy fallback) ──────────────────────

    @staticmethod
    def _row_to_skill(row: Any) -> SkillDefinition:
        return SkillDefinition(
            skill_id=row["skill_id"],
            avatar_id=row["avatar_id"],
            name=row["name"],
            description=row["description"] or "",
            target_threshold=row["target_threshold"],
            effective_threshold=row["effective_threshold"] if "effective_threshold" in row.keys() else 0.7,
            bad_ratio=row["bad_ratio"] if "bad_ratio" in row.keys() else 0.0,
            progress=row["progress"],
            qa_count=row["qa_count"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _row_to_qa(row: Any) -> QAPair:
        return QAPair(
            qa_id=row["qa_id"],
            skill_id=row["skill_id"],
            avatar_id=row["avatar_id"],
            question=row["question"],
            human_answer=row["human_answer"],
            ai_answer=row["ai_answer"] or "",
            quality=row["quality"] or "none",
            ingested=bool(row["ingested"]),
            weight=row["weight"] if "weight" in row.keys() else 1.0,
            correction_of=row["correction_of"] or "" if "correction_of" in row.keys() else "",
            confidence_at_time=row["confidence_at_time"] or 0.0 if "confidence_at_time" in row.keys() else 0.0,
            created_at=row["created_at"],
        )

    @staticmethod
    def _row_to_escalation(row: Any) -> EscalationEvent:
        return EscalationEvent(
            event_id=row["event_id"],
            session_id=row["session_id"],
            avatar_id=row["avatar_id"],
            skill_id=row["skill_id"] or "unknown",
            question=row["question"] or "",
            confidence=row["confidence"],
            resolved=bool(row["resolved"]),
            resolution=row["resolution"] or "",
            created_at=row["created_at"],
        )


# ── SQLite Schema (legacy fallback) ──────────────────────────────────────

_SQLITE_CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS skills (
    skill_id TEXT PRIMARY KEY,
    avatar_id TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    target_threshold REAL DEFAULT 0.7,
    effective_threshold REAL DEFAULT 0.7,
    bad_ratio REAL DEFAULT 0.0,
    progress REAL DEFAULT 0.0,
    qa_count INTEGER DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS qa_pairs (
    qa_id TEXT PRIMARY KEY,
    skill_id TEXT NOT NULL,
    avatar_id TEXT NOT NULL,
    question TEXT NOT NULL,
    human_answer TEXT NOT NULL,
    ai_answer TEXT DEFAULT '',
    quality TEXT DEFAULT 'none',
    ingested INTEGER DEFAULT 0,
    weight REAL DEFAULT 1.0,
    correction_of TEXT DEFAULT NULL,
    confidence_at_time REAL DEFAULT NULL,
    created_at REAL NOT NULL,
    FOREIGN KEY (skill_id) REFERENCES skills(skill_id)
);

CREATE TABLE IF NOT EXISTS escalations (
    event_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    avatar_id TEXT NOT NULL,
    skill_id TEXT DEFAULT 'unknown',
    question TEXT DEFAULT '',
    confidence REAL DEFAULT 0.0,
    resolved INTEGER DEFAULT 0,
    resolution TEXT DEFAULT '',
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_skills_avatar ON skills(avatar_id);
CREATE INDEX IF NOT EXISTS idx_qa_skill ON qa_pairs(skill_id);
CREATE INDEX IF NOT EXISTS idx_qa_avatar ON qa_pairs(avatar_id);
CREATE INDEX IF NOT EXISTS idx_escalations_avatar ON escalations(avatar_id);
CREATE INDEX IF NOT EXISTS idx_escalations_unresolved
    ON escalations(resolved) WHERE resolved = 0;
"""
