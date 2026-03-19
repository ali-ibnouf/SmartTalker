"""Supervisor Engine — operator performance tracking and team management.

Provides:
- Operator action recording (responses, corrections, escalation resolution)
- Performance metrics per operator (response time, quality score)
- Decision review queue for AI quality assurance
- Team activity timeline
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

logger = setup_logger("pipeline.supervisor")


@dataclass
class OperatorMetrics:
    """Aggregated performance metrics for a single operator."""
    operator_id: str = ""
    total_responses: int = 0
    avg_response_time_ms: int = 0
    escalations_resolved: int = 0
    corrections_made: int = 0
    sessions_handled: int = 0
    quality_score: float = 0.0
    active_since: float = 0.0


@dataclass
class DecisionReviewItem:
    """An AI decision flagged for human review."""
    id: str = ""
    session_id: str = ""
    avatar_id: str = ""
    question: str = ""
    ai_response: str = ""
    confidence: float = 0.0
    flagged_reason: str = ""
    reviewed: bool = False
    reviewer_id: str = ""
    review_verdict: str = ""
    corrected_response: str = ""
    created_at: float = 0.0
    reviewed_at: float = 0.0


@dataclass
class ActivityEntry:
    """A single entry in the team activity timeline."""
    timestamp: float = 0.0
    operator_id: str = ""
    action_type: str = ""
    session_id: str = ""
    avatar_id: str = ""
    details: dict = field(default_factory=dict)


class SupervisorEngine:
    """Operator performance tracking and decision review management."""

    def __init__(self, config: Settings, db: Any = None) -> None:
        self._config = config
        self._db = db
        self._sqlite_conn: Any = None
        self._loaded = False
        self._ws_manager: Any = None

    def set_ws_manager(self, ws_manager: Any) -> None:
        """Set the WebSocket manager for active session introspection."""
        self._ws_manager = ws_manager

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    async def load(self) -> None:
        if self._db is not None:
            self._loaded = True
            logger.info("SupervisorEngine loaded (PostgreSQL)")
            return

        try:
            import aiosqlite
            db_path = str(self._config.training_db_path).replace("training.db", "supervisor.db")
            self._sqlite_conn = await aiosqlite.connect(db_path)
            await self._sqlite_conn.execute("PRAGMA journal_mode=WAL")
            await self._create_sqlite_tables()
            self._loaded = True
            logger.info("SupervisorEngine loaded (SQLite: %s)", db_path)
        except Exception as exc:
            raise TrainingError("Failed to load SupervisorEngine", detail=str(exc), original_exception=exc)

    async def _create_sqlite_tables(self) -> None:
        assert self._sqlite_conn is not None
        await self._sqlite_conn.executescript("""
            CREATE TABLE IF NOT EXISTS operator_actions (
                id TEXT PRIMARY KEY,
                operator_id TEXT NOT NULL,
                action_type TEXT NOT NULL,
                session_id TEXT DEFAULT '',
                avatar_id TEXT DEFAULT '',
                details TEXT DEFAULT '{}',
                response_time_ms INTEGER DEFAULT 0,
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_oa_operator ON operator_actions(operator_id);
            CREATE INDEX IF NOT EXISTS idx_oa_created ON operator_actions(created_at);

            CREATE TABLE IF NOT EXISTS decision_reviews (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                avatar_id TEXT NOT NULL,
                question TEXT NOT NULL,
                ai_response TEXT NOT NULL,
                confidence REAL DEFAULT 0.0,
                flagged_reason TEXT NOT NULL,
                reviewed INTEGER DEFAULT 0,
                reviewer_id TEXT DEFAULT '',
                review_verdict TEXT DEFAULT '',
                corrected_response TEXT DEFAULT '',
                created_at REAL NOT NULL,
                reviewed_at REAL DEFAULT 0.0
            );
            CREATE INDEX IF NOT EXISTS idx_dr_reviewed ON decision_reviews(reviewed);
            CREATE INDEX IF NOT EXISTS idx_dr_created ON decision_reviews(created_at);
        """)
        await self._sqlite_conn.commit()

    async def unload(self) -> None:
        if self._sqlite_conn is not None:
            await self._sqlite_conn.close()
            self._sqlite_conn = None
        self._loaded = False
        logger.info("SupervisorEngine unloaded")

    # ── Operator Action Recording ────────────────────────────────────────

    async def record_operator_action(
        self,
        operator_id: str,
        action_type: str,
        session_id: str = "",
        avatar_id: str = "",
        details: Optional[dict] = None,
        response_time_ms: int = 0,
    ) -> str:
        """Record an operator action for performance tracking."""
        if not self._loaded:
            raise TrainingError("SupervisorEngine not loaded")

        action_id = uuid.uuid4().hex
        now = time.time()
        details_json = json.dumps(details or {}, ensure_ascii=False)

        if self._db is not None:
            from src.db.models import OperatorAction

            async with self._db.session_ctx() as session:
                session.add(OperatorAction(
                    id=action_id,
                    operator_id=operator_id,
                    action_type=action_type,
                    session_id=session_id,
                    avatar_id=avatar_id,
                    details=details_json,
                    response_time_ms=response_time_ms,
                    created_at=datetime.now(timezone.utc),
                ))
        else:
            conn = self._sqlite_conn
            assert conn is not None
            await conn.execute(
                "INSERT INTO operator_actions (id, operator_id, action_type, session_id, avatar_id, details, response_time_ms, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (action_id, operator_id, action_type, session_id, avatar_id, details_json, response_time_ms, now),
            )
            await conn.commit()

        logger.info("Operator action [%s] by %s in session %s", action_type, operator_id, session_id)
        return action_id

    # ── Operator Metrics ─────────────────────────────────────────────────

    async def get_operator_metrics(self, operator_id: str, days: int = 30) -> OperatorMetrics:
        """Get aggregated performance metrics for a specific operator."""
        if not self._loaded:
            raise TrainingError("SupervisorEngine not loaded")

        metrics = OperatorMetrics(operator_id=operator_id)

        if self._db is not None:
            from sqlalchemy import select, func as sa_func
            from src.db.models import OperatorAction

            since = datetime.now(timezone.utc) - timedelta(days=days)
            async with self._db.session() as session:
                base = select(OperatorAction).where(
                    OperatorAction.operator_id == operator_id,
                    OperatorAction.created_at >= since,
                )
                # Total responses
                result = await session.execute(
                    select(sa_func.count()).where(
                        OperatorAction.operator_id == operator_id,
                        OperatorAction.action_type == "response",
                        OperatorAction.created_at >= since,
                    )
                )
                metrics.total_responses = result.scalar() or 0

                # Avg response time
                result = await session.execute(
                    select(sa_func.avg(OperatorAction.response_time_ms)).where(
                        OperatorAction.operator_id == operator_id,
                        OperatorAction.action_type == "response",
                        OperatorAction.created_at >= since,
                    )
                )
                avg = result.scalar()
                metrics.avg_response_time_ms = int(avg) if avg else 0

                # Escalations resolved
                result = await session.execute(
                    select(sa_func.count()).where(
                        OperatorAction.operator_id == operator_id,
                        OperatorAction.action_type == "escalation_resolve",
                        OperatorAction.created_at >= since,
                    )
                )
                metrics.escalations_resolved = result.scalar() or 0

                # Corrections
                result = await session.execute(
                    select(sa_func.count()).where(
                        OperatorAction.operator_id == operator_id,
                        OperatorAction.action_type == "correction",
                        OperatorAction.created_at >= since,
                    )
                )
                metrics.corrections_made = result.scalar() or 0

                # Unique sessions
                result = await session.execute(
                    select(sa_func.count(sa_func.distinct(OperatorAction.session_id))).where(
                        OperatorAction.operator_id == operator_id,
                        OperatorAction.created_at >= since,
                    )
                )
                metrics.sessions_handled = result.scalar() or 0

                # Earliest action
                result = await session.execute(
                    select(sa_func.min(OperatorAction.created_at)).where(
                        OperatorAction.operator_id == operator_id,
                    )
                )
                earliest = result.scalar()
                if earliest and isinstance(earliest, datetime):
                    metrics.active_since = earliest.timestamp()
        else:
            conn = self._sqlite_conn
            assert conn is not None
            since_ts = (datetime.now(timezone.utc) - timedelta(days=days)).timestamp()

            cursor = await conn.execute(
                "SELECT COUNT(*) FROM operator_actions WHERE operator_id = ? AND action_type = 'response' AND created_at >= ?",
                (operator_id, since_ts),
            )
            metrics.total_responses = (await cursor.fetchone())[0]

            cursor = await conn.execute(
                "SELECT AVG(response_time_ms) FROM operator_actions WHERE operator_id = ? AND action_type = 'response' AND created_at >= ?",
                (operator_id, since_ts),
            )
            avg = (await cursor.fetchone())[0]
            metrics.avg_response_time_ms = int(avg) if avg else 0

            cursor = await conn.execute(
                "SELECT COUNT(*) FROM operator_actions WHERE operator_id = ? AND action_type = 'escalation_resolve' AND created_at >= ?",
                (operator_id, since_ts),
            )
            metrics.escalations_resolved = (await cursor.fetchone())[0]

            cursor = await conn.execute(
                "SELECT COUNT(*) FROM operator_actions WHERE operator_id = ? AND action_type = 'correction' AND created_at >= ?",
                (operator_id, since_ts),
            )
            metrics.corrections_made = (await cursor.fetchone())[0]

            cursor = await conn.execute(
                "SELECT COUNT(DISTINCT session_id) FROM operator_actions WHERE operator_id = ? AND created_at >= ?",
                (operator_id, since_ts),
            )
            metrics.sessions_handled = (await cursor.fetchone())[0]

            cursor = await conn.execute(
                "SELECT MIN(created_at) FROM operator_actions WHERE operator_id = ?",
                (operator_id,),
            )
            row = await cursor.fetchone()
            if row and row[0]:
                metrics.active_since = row[0]

        # Quality score: ratio of responses to corrections (lower corrections = higher quality)
        total = metrics.total_responses + metrics.corrections_made
        if total > 0:
            metrics.quality_score = round(metrics.total_responses / total, 2)

        return metrics

    async def list_operator_metrics(self, days: int = 30) -> list[OperatorMetrics]:
        """Get metrics for all operators."""
        if not self._loaded:
            raise TrainingError("SupervisorEngine not loaded")

        operator_ids: list[str] = []

        if self._db is not None:
            from sqlalchemy import select, func as sa_func
            from src.db.models import OperatorAction

            since = datetime.now(timezone.utc) - timedelta(days=days)
            async with self._db.session() as session:
                result = await session.execute(
                    select(sa_func.distinct(OperatorAction.operator_id)).where(
                        OperatorAction.created_at >= since,
                    )
                )
                operator_ids = [row[0] for row in result.all()]
        else:
            conn = self._sqlite_conn
            assert conn is not None
            since_ts = (datetime.now(timezone.utc) - timedelta(days=days)).timestamp()
            cursor = await conn.execute(
                "SELECT DISTINCT operator_id FROM operator_actions WHERE created_at >= ?",
                (since_ts,),
            )
            operator_ids = [row[0] for row in await cursor.fetchall()]

        metrics_list: list[OperatorMetrics] = []
        for op_id in operator_ids:
            m = await self.get_operator_metrics(op_id, days)
            metrics_list.append(m)

        return sorted(metrics_list, key=lambda m: m.total_responses, reverse=True)

    # ── Decision Review Queue ────────────────────────────────────────────

    async def flag_for_review(
        self,
        session_id: str,
        avatar_id: str,
        question: str,
        ai_response: str,
        confidence: float,
        reason: str,
    ) -> str:
        """Flag an AI decision for human review."""
        if not self._loaded:
            raise TrainingError("SupervisorEngine not loaded")

        review_id = uuid.uuid4().hex
        now = time.time()

        if self._db is not None:
            from src.db.models import DecisionReview

            async with self._db.session_ctx() as session:
                session.add(DecisionReview(
                    id=review_id,
                    session_id=session_id,
                    avatar_id=avatar_id,
                    question=question[:5000],
                    ai_response=ai_response[:5000],
                    confidence=confidence,
                    flagged_reason=reason,
                    created_at=datetime.now(timezone.utc),
                ))
        else:
            conn = self._sqlite_conn
            assert conn is not None
            await conn.execute(
                "INSERT INTO decision_reviews (id, session_id, avatar_id, question, ai_response, confidence, flagged_reason, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (review_id, session_id, avatar_id, question[:5000], ai_response[:5000], confidence, reason, now),
            )
            await conn.commit()

        logger.info("Flagged for review: %s (reason: %s, confidence: %.2f)", review_id, reason, confidence)
        return review_id

    async def list_review_queue(
        self,
        reviewed: Optional[bool] = None,
        limit: int = 50,
    ) -> list[DecisionReviewItem]:
        """List decision review items."""
        if not self._loaded:
            raise TrainingError("SupervisorEngine not loaded")

        items: list[DecisionReviewItem] = []

        if self._db is not None:
            from sqlalchemy import select
            from src.db.models import DecisionReview

            async with self._db.session() as session:
                query = select(DecisionReview)
                if reviewed is not None:
                    query = query.where(DecisionReview.reviewed == reviewed)
                query = query.order_by(DecisionReview.created_at.desc()).limit(limit)
                result = await session.execute(query)
                for row in result.scalars().all():
                    items.append(DecisionReviewItem(
                        id=row.id,
                        session_id=row.session_id,
                        avatar_id=row.avatar_id,
                        question=row.question,
                        ai_response=row.ai_response,
                        confidence=row.confidence,
                        flagged_reason=row.flagged_reason,
                        reviewed=row.reviewed,
                        reviewer_id=row.reviewer_id,
                        review_verdict=row.review_verdict,
                        corrected_response=row.corrected_response,
                        created_at=row.created_at.timestamp() if isinstance(row.created_at, datetime) else 0.0,
                        reviewed_at=row.reviewed_at.timestamp() if isinstance(row.reviewed_at, datetime) else 0.0,
                    ))
        else:
            conn = self._sqlite_conn
            assert conn is not None
            sql = "SELECT id, session_id, avatar_id, question, ai_response, confidence, flagged_reason, reviewed, reviewer_id, review_verdict, corrected_response, created_at, reviewed_at FROM decision_reviews"
            params: list[Any] = []
            if reviewed is not None:
                sql += " WHERE reviewed = ?"
                params.append(int(reviewed))
            sql += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            cursor = await conn.execute(sql, params)
            for row in await cursor.fetchall():
                items.append(DecisionReviewItem(
                    id=row[0], session_id=row[1], avatar_id=row[2],
                    question=row[3], ai_response=row[4], confidence=row[5],
                    flagged_reason=row[6], reviewed=bool(row[7]),
                    reviewer_id=row[8] or "", review_verdict=row[9] or "",
                    corrected_response=row[10] or "",
                    created_at=row[11] or 0.0, reviewed_at=row[12] or 0.0,
                ))

        return items

    async def submit_review(
        self,
        review_id: str,
        reviewer_id: str,
        verdict: str,
        corrected_response: str = "",
    ) -> DecisionReviewItem:
        """Submit a review verdict for a flagged decision."""
        if not self._loaded:
            raise TrainingError("SupervisorEngine not loaded")

        now = time.time()

        if self._db is not None:
            from sqlalchemy import select
            from src.db.models import DecisionReview

            async with self._db.session_ctx() as session:
                result = await session.execute(
                    select(DecisionReview).where(DecisionReview.id == review_id)
                )
                row = result.scalar_one_or_none()
                if not row:
                    raise TrainingError("Review item not found", detail=review_id)

                row.reviewed = True
                row.reviewer_id = reviewer_id
                row.review_verdict = verdict
                row.corrected_response = corrected_response
                row.reviewed_at = datetime.now(timezone.utc)

                return DecisionReviewItem(
                    id=row.id, session_id=row.session_id, avatar_id=row.avatar_id,
                    question=row.question, ai_response=row.ai_response,
                    confidence=row.confidence, flagged_reason=row.flagged_reason,
                    reviewed=True, reviewer_id=reviewer_id,
                    review_verdict=verdict, corrected_response=corrected_response,
                    created_at=row.created_at.timestamp() if isinstance(row.created_at, datetime) else 0.0,
                    reviewed_at=now,
                )
        else:
            conn = self._sqlite_conn
            assert conn is not None
            cursor = await conn.execute(
                "UPDATE decision_reviews SET reviewed=1, reviewer_id=?, review_verdict=?, corrected_response=?, reviewed_at=? WHERE id=?",
                (reviewer_id, verdict, corrected_response, now, review_id),
            )
            if cursor.rowcount == 0:
                raise TrainingError("Review item not found", detail=review_id)
            await conn.commit()

            cursor = await conn.execute(
                "SELECT id, session_id, avatar_id, question, ai_response, confidence, flagged_reason, reviewed, reviewer_id, review_verdict, corrected_response, created_at, reviewed_at FROM decision_reviews WHERE id = ?",
                (review_id,),
            )
            row = await cursor.fetchone()
            assert row is not None
            return DecisionReviewItem(
                id=row[0], session_id=row[1], avatar_id=row[2],
                question=row[3], ai_response=row[4], confidence=row[5],
                flagged_reason=row[6], reviewed=True,
                reviewer_id=row[8] or "", review_verdict=row[9] or "",
                corrected_response=row[10] or "",
                created_at=row[11] or 0.0, reviewed_at=row[12] or 0.0,
            )

    # ── Active Sessions ─────────────────────────────────────────────────

    async def get_active_sessions_summary(self) -> list[dict]:
        """Return summary of active chat sessions from the WebSocket manager."""
        ws_mgr = self._ws_manager
        if ws_mgr is None:
            return []

        sessions: list[dict] = []
        session_map: dict = getattr(ws_mgr, '_sessions', None) or {}
        for sid, sess in session_map.items():
            cfg = getattr(sess, 'config', None)
            sessions.append({
                "session_id": sid,
                "customer_id": getattr(sess, 'client_ip', ''),
                "avatar_id": getattr(cfg, 'avatar_id', '') if cfg else '',
                "started_at": getattr(sess, 'connected_at', 0.0),
                "training_mode": getattr(cfg, 'training_mode', 'digital') if cfg else 'digital',
                "ai_paused": getattr(sess, 'ai_paused', False),
                "operator_id": getattr(sess, 'operator_id', None),
                "message_count": 0,
            })
        return sessions

    # ── Activity Timeline ────────────────────────────────────────────────

    async def get_activity_timeline(self, days: int = 7, limit: int = 100) -> list[ActivityEntry]:
        """Get recent team activity timeline."""
        if not self._loaded:
            raise TrainingError("SupervisorEngine not loaded")

        entries: list[ActivityEntry] = []

        if self._db is not None:
            from sqlalchemy import select
            from src.db.models import OperatorAction

            since = datetime.now(timezone.utc) - timedelta(days=days)
            async with self._db.session() as session:
                result = await session.execute(
                    select(OperatorAction).where(
                        OperatorAction.created_at >= since,
                    ).order_by(OperatorAction.created_at.desc()).limit(limit)
                )
                for row in result.scalars().all():
                    entries.append(ActivityEntry(
                        timestamp=row.created_at.timestamp() if isinstance(row.created_at, datetime) else 0.0,
                        operator_id=row.operator_id,
                        action_type=row.action_type,
                        session_id=row.session_id,
                        avatar_id=row.avatar_id,
                        details=json.loads(row.details) if row.details else {},
                    ))
        else:
            conn = self._sqlite_conn
            assert conn is not None
            since_ts = (datetime.now(timezone.utc) - timedelta(days=days)).timestamp()
            cursor = await conn.execute(
                "SELECT created_at, operator_id, action_type, session_id, avatar_id, details FROM operator_actions WHERE created_at >= ? ORDER BY created_at DESC LIMIT ?",
                (since_ts, limit),
            )
            for row in await cursor.fetchall():
                entries.append(ActivityEntry(
                    timestamp=row[0] or 0.0,
                    operator_id=row[1],
                    action_type=row[2],
                    session_id=row[3] or "",
                    avatar_id=row[4] or "",
                    details=json.loads(row[5]) if row[5] else {},
                ))

        return entries
