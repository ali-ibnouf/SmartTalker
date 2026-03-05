"""Learning Analytics — tracks correction quality, auto-adjusts thresholds, exports data.

Enhances the training pipeline with:
- Correction tracking and quality metrics per skill
- Automatic escalation threshold adjustment based on bad_ratio
- Daily consolidation of learning metrics
- Q&A export for future fine-tuning
- Improvement timeline for trend analysis
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from src.config import Settings
from src.utils.exceptions import TrainingError
from src.utils.logger import setup_logger

logger = setup_logger("pipeline.learning_analytics")

# Threshold adjustment constants
_BAD_RATIO_RAISE = 0.3  # If bad_ratio > this, raise effective_threshold
_BAD_RATIO_LOWER = 0.1  # If bad_ratio < this, lower effective_threshold
_THRESHOLD_STEP_UP = 0.1
_THRESHOLD_STEP_DOWN = 0.05
_THRESHOLD_MIN = 0.3
_THRESHOLD_MAX = 0.95


@dataclass
class QualityStats:
    """Quality statistics for a skill."""
    skill_id: str = ""
    total_qa: int = 0
    good_count: int = 0
    bad_count: int = 0
    none_count: int = 0
    correction_count: int = 0
    bad_ratio: float = 0.0
    effective_threshold: float = 0.7
    improvement_rate: float = 0.0  # good_ratio change over last 7 days


@dataclass
class DailyConsolidation:
    """Result of daily consolidation."""
    avatar_id: str = ""
    date: str = ""
    skills_updated: int = 0
    thresholds_adjusted: int = 0
    total_qa_today: int = 0
    good_today: int = 0
    bad_today: int = 0


@dataclass
class TimelinePoint:
    """Single point in the improvement timeline."""
    date: str = ""
    qa_added: int = 0
    good_count: int = 0
    bad_count: int = 0
    corrections: int = 0
    progress: float = 0.0


class LearningAnalytics:
    """Tracks correction patterns, improvement rates, and adjusts thresholds."""

    def __init__(self, config: Settings, db: Any = None) -> None:
        self._config = config
        self._db = db
        self._sqlite_conn: Any = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    async def load(self) -> None:
        """Initialize storage. Uses PostgreSQL if db provided, else SQLite."""
        if self._db is not None:
            self._loaded = True
            logger.info("LearningAnalytics loaded (PostgreSQL)")
            return

        try:
            import aiosqlite
            db_path = str(self._config.training_db_path)
            self._sqlite_conn = await aiosqlite.connect(db_path)
            await self._sqlite_conn.execute("PRAGMA journal_mode=WAL")
            await self._create_sqlite_tables()
            self._loaded = True
            logger.info("LearningAnalytics loaded (SQLite: %s)", db_path)
        except Exception as exc:
            raise TrainingError("Failed to load LearningAnalytics", detail=str(exc), original_exception=exc)

    async def _create_sqlite_tables(self) -> None:
        assert self._sqlite_conn is not None
        await self._sqlite_conn.executescript("""
            CREATE TABLE IF NOT EXISTS learning_metrics (
                id TEXT PRIMARY KEY,
                avatar_id TEXT NOT NULL,
                skill_id TEXT NOT NULL,
                date TEXT NOT NULL,
                qa_added INTEGER DEFAULT 0,
                good_count INTEGER DEFAULT 0,
                bad_count INTEGER DEFAULT 0,
                corrections_count INTEGER DEFAULT 0,
                avg_confidence_before REAL DEFAULT 0.0,
                avg_confidence_after REAL DEFAULT 0.0,
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_lm_avatar_skill_date
                ON learning_metrics(avatar_id, skill_id, date);
        """)
        await self._sqlite_conn.commit()

    async def unload(self) -> None:
        if self._sqlite_conn is not None:
            await self._sqlite_conn.close()
            self._sqlite_conn = None
        self._loaded = False
        logger.info("LearningAnalytics unloaded")

    # ── Quality Stats ────────────────────────────────────────────────────

    async def get_skill_quality_stats(self, avatar_id: str, skill_id: str) -> QualityStats:
        """Get quality statistics for a specific skill."""
        if not self._loaded:
            raise TrainingError("LearningAnalytics not loaded")

        stats = QualityStats(skill_id=skill_id)

        if self._db is not None:
            from sqlalchemy import select, func as sa_func
            from src.db.models import QAPair as QAPairModel, Skill as SkillModel

            async with self._db.session() as session:
                # Get skill info
                skill_row = await session.execute(
                    select(SkillModel).where(SkillModel.id == skill_id)
                )
                skill = skill_row.scalar_one_or_none()
                if skill:
                    stats.effective_threshold = skill.effective_threshold
                    stats.bad_ratio = skill.bad_ratio

                # Count Q&A by quality
                for quality in ("good", "bad", "none"):
                    result = await session.execute(
                        select(sa_func.count()).where(
                            QAPairModel.skill_id == skill_id,
                            QAPairModel.quality == quality,
                        )
                    )
                    count = result.scalar() or 0
                    if quality == "good":
                        stats.good_count = count
                    elif quality == "bad":
                        stats.bad_count = count
                    else:
                        stats.none_count = count

                # Count corrections
                result = await session.execute(
                    select(sa_func.count()).where(
                        QAPairModel.skill_id == skill_id,
                        QAPairModel.correction_of.isnot(None),
                    )
                )
                stats.correction_count = result.scalar() or 0
        else:
            conn = self._sqlite_conn
            assert conn is not None
            cursor = await conn.execute(
                "SELECT quality, COUNT(*) FROM qa_pairs WHERE skill_id = ? GROUP BY quality",
                (skill_id,),
            )
            for row in await cursor.fetchall():
                if row[0] == "good":
                    stats.good_count = row[1]
                elif row[0] == "bad":
                    stats.bad_count = row[1]
                else:
                    stats.none_count = row[1]

            cursor = await conn.execute(
                "SELECT COUNT(*) FROM qa_pairs WHERE skill_id = ? AND correction_of IS NOT NULL",
                (skill_id,),
            )
            row = await cursor.fetchone()
            stats.correction_count = row[0] if row else 0

            cursor = await conn.execute(
                "SELECT effective_threshold, bad_ratio FROM skills WHERE skill_id = ?",
                (skill_id,),
            )
            row = await cursor.fetchone()
            if row:
                stats.effective_threshold = row[0]
                stats.bad_ratio = row[1]

        stats.total_qa = stats.good_count + stats.bad_count + stats.none_count
        if stats.total_qa > 0:
            stats.bad_ratio = stats.bad_count / stats.total_qa

        # Calculate improvement rate (comparing last 7 days vs previous 7 days)
        stats.improvement_rate = await self._calc_improvement_rate(avatar_id, skill_id)

        return stats

    async def _calc_improvement_rate(self, avatar_id: str, skill_id: str) -> float:
        """Calculate the improvement rate as change in good_ratio over recent periods."""
        now = datetime.now(timezone.utc)
        week_ago = now - timedelta(days=7)
        two_weeks_ago = now - timedelta(days=14)

        if self._db is not None:
            from sqlalchemy import select, func as sa_func
            from src.db.models import QAPair as QAPairModel

            async with self._db.session() as session:
                # Recent 7 days
                recent_total = (await session.execute(
                    select(sa_func.count()).where(
                        QAPairModel.skill_id == skill_id,
                        QAPairModel.created_at >= week_ago,
                    )
                )).scalar() or 0
                recent_good = (await session.execute(
                    select(sa_func.count()).where(
                        QAPairModel.skill_id == skill_id,
                        QAPairModel.quality == "good",
                        QAPairModel.created_at >= week_ago,
                    )
                )).scalar() or 0

                # Previous 7 days
                prev_total = (await session.execute(
                    select(sa_func.count()).where(
                        QAPairModel.skill_id == skill_id,
                        QAPairModel.created_at >= two_weeks_ago,
                        QAPairModel.created_at < week_ago,
                    )
                )).scalar() or 0
                prev_good = (await session.execute(
                    select(sa_func.count()).where(
                        QAPairModel.skill_id == skill_id,
                        QAPairModel.quality == "good",
                        QAPairModel.created_at >= two_weeks_ago,
                        QAPairModel.created_at < week_ago,
                    )
                )).scalar() or 0
        else:
            conn = self._sqlite_conn
            assert conn is not None
            week_ts = week_ago.timestamp()
            two_weeks_ts = two_weeks_ago.timestamp()

            cursor = await conn.execute(
                "SELECT COUNT(*) FROM qa_pairs WHERE skill_id = ? AND created_at >= ?",
                (skill_id, week_ts),
            )
            recent_total = (await cursor.fetchone())[0]
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM qa_pairs WHERE skill_id = ? AND quality = 'good' AND created_at >= ?",
                (skill_id, week_ts),
            )
            recent_good = (await cursor.fetchone())[0]

            cursor = await conn.execute(
                "SELECT COUNT(*) FROM qa_pairs WHERE skill_id = ? AND created_at >= ? AND created_at < ?",
                (skill_id, two_weeks_ts, week_ts),
            )
            prev_total = (await cursor.fetchone())[0]
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM qa_pairs WHERE skill_id = ? AND quality = 'good' AND created_at >= ? AND created_at < ?",
                (skill_id, two_weeks_ts, week_ts),
            )
            prev_good = (await cursor.fetchone())[0]

        recent_ratio = (recent_good / recent_total) if recent_total > 0 else 0.0
        prev_ratio = (prev_good / prev_total) if prev_total > 0 else 0.0
        return round(recent_ratio - prev_ratio, 4)

    # ── Threshold Adjustment ─────────────────────────────────────────────

    async def recalculate_effective_threshold(self, skill_id: str) -> float:
        """Auto-adjust escalation threshold based on quality metrics."""
        if not self._loaded:
            raise TrainingError("LearningAnalytics not loaded")

        if self._db is not None:
            from sqlalchemy import select, func as sa_func
            from src.db.models import QAPair as QAPairModel, Skill as SkillModel

            async with self._db.session() as session:
                skill_row = await session.execute(
                    select(SkillModel).where(SkillModel.id == skill_id)
                )
                skill = skill_row.scalar_one_or_none()
                if not skill:
                    return 0.7

                total = (await session.execute(
                    select(sa_func.count()).where(QAPairModel.skill_id == skill_id)
                )).scalar() or 0
                bad = (await session.execute(
                    select(sa_func.count()).where(
                        QAPairModel.skill_id == skill_id,
                        QAPairModel.quality == "bad",
                    )
                )).scalar() or 0

                bad_ratio = (bad / total) if total > 0 else 0.0
                new_threshold = self._adjust_threshold(skill.target_threshold, bad_ratio)

                skill.effective_threshold = new_threshold
                skill.bad_ratio = bad_ratio
                await session.commit()
                return new_threshold
        else:
            conn = self._sqlite_conn
            assert conn is not None

            cursor = await conn.execute(
                "SELECT target_threshold FROM skills WHERE skill_id = ?", (skill_id,)
            )
            row = await cursor.fetchone()
            if not row:
                return 0.7
            base_threshold = row[0]

            cursor = await conn.execute(
                "SELECT COUNT(*) FROM qa_pairs WHERE skill_id = ?", (skill_id,)
            )
            total = (await cursor.fetchone())[0]
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM qa_pairs WHERE skill_id = ? AND quality = 'bad'",
                (skill_id,),
            )
            bad = (await cursor.fetchone())[0]

            bad_ratio = (bad / total) if total > 0 else 0.0
            new_threshold = self._adjust_threshold(base_threshold, bad_ratio)

            await conn.execute(
                "UPDATE skills SET effective_threshold = ?, bad_ratio = ? WHERE skill_id = ?",
                (new_threshold, bad_ratio, skill_id),
            )
            await conn.commit()
            return new_threshold

    @staticmethod
    def _adjust_threshold(base: float, bad_ratio: float) -> float:
        """Compute effective threshold from base and bad_ratio."""
        if bad_ratio > _BAD_RATIO_RAISE:
            adjusted = base + _THRESHOLD_STEP_UP
        elif bad_ratio < _BAD_RATIO_LOWER:
            adjusted = base - _THRESHOLD_STEP_DOWN
        else:
            adjusted = base
        return max(_THRESHOLD_MIN, min(_THRESHOLD_MAX, round(adjusted, 3)))

    # ── Daily Consolidation ──────────────────────────────────────────────

    async def consolidate_daily(self, avatar_id: str) -> DailyConsolidation:
        """Summarize today's learning, update metrics, adjust thresholds."""
        if not self._loaded:
            raise TrainingError("LearningAnalytics not loaded")

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        result = DailyConsolidation(avatar_id=avatar_id, date=today)

        if self._db is not None:
            from sqlalchemy import select, func as sa_func
            from src.db.models import Skill as SkillModel, QAPair as QAPairModel, LearningMetric

            async with self._db.session() as session:
                skills_result = await session.execute(
                    select(SkillModel).where(SkillModel.avatar_id == avatar_id)
                )
                skills = skills_result.scalars().all()

                for skill in skills:
                    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

                    total_today = (await session.execute(
                        select(sa_func.count()).where(
                            QAPairModel.skill_id == skill.id,
                            QAPairModel.created_at >= today_start,
                        )
                    )).scalar() or 0

                    good_today = (await session.execute(
                        select(sa_func.count()).where(
                            QAPairModel.skill_id == skill.id,
                            QAPairModel.quality == "good",
                            QAPairModel.created_at >= today_start,
                        )
                    )).scalar() or 0

                    bad_today = (await session.execute(
                        select(sa_func.count()).where(
                            QAPairModel.skill_id == skill.id,
                            QAPairModel.quality == "bad",
                            QAPairModel.created_at >= today_start,
                        )
                    )).scalar() or 0

                    corrections_today = (await session.execute(
                        select(sa_func.count()).where(
                            QAPairModel.skill_id == skill.id,
                            QAPairModel.correction_of.isnot(None),
                            QAPairModel.created_at >= today_start,
                        )
                    )).scalar() or 0

                    # Store daily metric
                    metric = LearningMetric(
                        id=uuid.uuid4().hex,
                        avatar_id=avatar_id,
                        skill_id=skill.id,
                        date=today_start,
                        qa_added=total_today,
                        good_count=good_today,
                        bad_count=bad_today,
                        corrections_count=corrections_today,
                        created_at=datetime.now(timezone.utc),
                    )
                    session.add(metric)

                    result.total_qa_today += total_today
                    result.good_today += good_today
                    result.bad_today += bad_today
                    result.skills_updated += 1

                await session.commit()

                # Adjust thresholds for all skills
                for skill in skills:
                    old = skill.effective_threshold
                    new_val = await self.recalculate_effective_threshold(skill.id)
                    if abs(old - new_val) > 0.001:
                        result.thresholds_adjusted += 1
        else:
            conn = self._sqlite_conn
            assert conn is not None

            cursor = await conn.execute(
                "SELECT skill_id FROM skills WHERE avatar_id = ?", (avatar_id,)
            )
            skill_rows = await cursor.fetchall()
            now_ts = time.time()
            today_start_ts = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            ).timestamp()

            for (skill_id,) in skill_rows:
                cursor = await conn.execute(
                    "SELECT COUNT(*) FROM qa_pairs WHERE skill_id = ? AND created_at >= ?",
                    (skill_id, today_start_ts),
                )
                total_today = (await cursor.fetchone())[0]

                cursor = await conn.execute(
                    "SELECT quality, COUNT(*) FROM qa_pairs WHERE skill_id = ? AND created_at >= ? GROUP BY quality",
                    (skill_id, today_start_ts),
                )
                good_today = bad_today = 0
                for row in await cursor.fetchall():
                    if row[0] == "good":
                        good_today = row[1]
                    elif row[0] == "bad":
                        bad_today = row[1]

                cursor = await conn.execute(
                    "SELECT COUNT(*) FROM qa_pairs WHERE skill_id = ? AND correction_of IS NOT NULL AND created_at >= ?",
                    (skill_id, today_start_ts),
                )
                corrections_today = (await cursor.fetchone())[0]

                await conn.execute(
                    "INSERT INTO learning_metrics (id, avatar_id, skill_id, date, qa_added, good_count, bad_count, corrections_count, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (uuid.uuid4().hex, avatar_id, skill_id, today, total_today, good_today, bad_today, corrections_today, now_ts),
                )

                result.total_qa_today += total_today
                result.good_today += good_today
                result.bad_today += bad_today
                result.skills_updated += 1

                old_threshold = 0.7
                cursor = await conn.execute(
                    "SELECT effective_threshold FROM skills WHERE skill_id = ?", (skill_id,)
                )
                row = await cursor.fetchone()
                if row:
                    old_threshold = row[0]

                new_val = await self.recalculate_effective_threshold(skill_id)
                if abs(old_threshold - new_val) > 0.001:
                    result.thresholds_adjusted += 1

            await conn.commit()

        logger.info(
            "Daily consolidation for %s: %d skills, %d QA, %d thresholds adjusted",
            avatar_id, result.skills_updated, result.total_qa_today, result.thresholds_adjusted,
        )
        return result

    # ── Export ────────────────────────────────────────────────────────────

    async def export_qa_pairs(
        self,
        avatar_id: str,
        skill_id: Optional[str] = None,
        fmt: str = "jsonl",
    ) -> str:
        """Export Q&A pairs as JSON or JSONL for fine-tuning."""
        if not self._loaded:
            raise TrainingError("LearningAnalytics not loaded")

        pairs: list[dict] = []

        if self._db is not None:
            from sqlalchemy import select
            from src.db.models import QAPair as QAPairModel

            async with self._db.session() as session:
                query = select(QAPairModel).where(QAPairModel.avatar_id == avatar_id)
                if skill_id:
                    query = query.where(QAPairModel.skill_id == skill_id)
                result = await session.execute(query.order_by(QAPairModel.created_at))
                for row in result.scalars().all():
                    pairs.append({
                        "qa_id": row.id,
                        "skill_id": row.skill_id,
                        "question": row.question,
                        "human_answer": row.human_answer,
                        "ai_answer": row.ai_answer,
                        "quality": row.quality,
                        "weight": row.weight,
                        "correction_of": row.correction_of,
                    })
        else:
            conn = self._sqlite_conn
            assert conn is not None
            sql = "SELECT qa_id, skill_id, question, human_answer, ai_answer, quality, weight, correction_of FROM qa_pairs WHERE avatar_id = ?"
            params: list[Any] = [avatar_id]
            if skill_id:
                sql += " AND skill_id = ?"
                params.append(skill_id)
            sql += " ORDER BY created_at"

            cursor = await conn.execute(sql, params)
            for row in await cursor.fetchall():
                pairs.append({
                    "qa_id": row[0], "skill_id": row[1],
                    "question": row[2], "human_answer": row[3],
                    "ai_answer": row[4], "quality": row[5],
                    "weight": row[6], "correction_of": row[7],
                })

        if fmt == "jsonl":
            return "\n".join(json.dumps(p, ensure_ascii=False) for p in pairs)
        return json.dumps(pairs, ensure_ascii=False, indent=2)

    # ── Improvement Timeline ─────────────────────────────────────────────

    async def get_improvement_timeline(
        self,
        avatar_id: str,
        skill_id: Optional[str] = None,
        days: int = 30,
    ) -> list[TimelinePoint]:
        """Get daily learning metrics over the past N days."""
        if not self._loaded:
            raise TrainingError("LearningAnalytics not loaded")

        points: list[TimelinePoint] = []

        if self._db is not None:
            from sqlalchemy import select
            from src.db.models import LearningMetric

            since = datetime.now(timezone.utc) - timedelta(days=days)
            async with self._db.session() as session:
                query = select(LearningMetric).where(
                    LearningMetric.avatar_id == avatar_id,
                    LearningMetric.date >= since,
                )
                if skill_id:
                    query = query.where(LearningMetric.skill_id == skill_id)
                query = query.order_by(LearningMetric.date)
                result = await session.execute(query)
                for row in result.scalars().all():
                    points.append(TimelinePoint(
                        date=row.date.strftime("%Y-%m-%d") if isinstance(row.date, datetime) else str(row.date),
                        qa_added=row.qa_added,
                        good_count=row.good_count,
                        bad_count=row.bad_count,
                        corrections=row.corrections_count,
                    ))
        else:
            conn = self._sqlite_conn
            assert conn is not None
            since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
            sql = "SELECT date, SUM(qa_added), SUM(good_count), SUM(bad_count), SUM(corrections_count) FROM learning_metrics WHERE avatar_id = ? AND date >= ?"
            params: list[Any] = [avatar_id, since]
            if skill_id:
                sql += " AND skill_id = ?"
                params.append(skill_id)
            sql += " GROUP BY date ORDER BY date"

            cursor = await conn.execute(sql, params)
            for row in await cursor.fetchall():
                points.append(TimelinePoint(
                    date=row[0],
                    qa_added=row[1] or 0,
                    good_count=row[2] or 0,
                    bad_count=row[3] or 0,
                    corrections=row[4] or 0,
                ))

        return points

    # ── Weak Areas Detection ─────────────────────────────────────────────

    async def get_weak_areas(self, avatar_id: str, skill_id: str, limit: int = 10) -> list[dict]:
        """Identify question patterns with high bad Q&A ratio."""
        if not self._loaded:
            raise TrainingError("LearningAnalytics not loaded")

        weak: list[dict] = []

        if self._db is not None:
            from sqlalchemy import select
            from src.db.models import QAPair as QAPairModel

            async with self._db.session() as session:
                result = await session.execute(
                    select(QAPairModel).where(
                        QAPairModel.skill_id == skill_id,
                        QAPairModel.quality == "bad",
                    ).order_by(QAPairModel.created_at.desc()).limit(limit)
                )
                for row in result.scalars().all():
                    weak.append({
                        "qa_id": row.id,
                        "question": row.question,
                        "ai_answer": row.ai_answer,
                        "human_answer": row.human_answer,
                        "created_at": row.created_at.timestamp() if isinstance(row.created_at, datetime) else 0.0,
                    })
        else:
            conn = self._sqlite_conn
            assert conn is not None
            cursor = await conn.execute(
                "SELECT qa_id, question, ai_answer, human_answer, created_at FROM qa_pairs WHERE skill_id = ? AND quality = 'bad' ORDER BY created_at DESC LIMIT ?",
                (skill_id, limit),
            )
            for row in await cursor.fetchall():
                weak.append({
                    "qa_id": row[0], "question": row[1],
                    "ai_answer": row[2], "human_answer": row[3],
                    "created_at": row[4] or 0.0,
                })

        return weak
