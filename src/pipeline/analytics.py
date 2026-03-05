"""Analytics Engine — KPI aggregation, time-series, drift detection, reports.

Aggregates data from conversations, escalations, Q&A pairs, and usage records
into actionable performance metrics.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from src.config import Settings
from src.utils.exceptions import AnalyticsError
from src.utils.logger import setup_logger

logger = setup_logger("pipeline.analytics")


@dataclass
class KPISnapshot:
    """Key Performance Indicators at a point in time."""
    period_start: float = 0.0
    period_end: float = 0.0
    total_conversations: int = 0
    total_messages: int = 0
    avg_response_time_ms: float = 0.0
    avg_kb_confidence: float = 0.0
    escalation_rate: float = 0.0
    autonomy_percent: float = 0.0
    resolution_time_avg_s: float = 0.0
    accuracy_score: float = 0.0
    total_cost: float = 0.0
    unique_users: int = 0


@dataclass
class DriftAlert:
    """Performance drift detection alert."""
    metric_name: str = ""
    current_value: float = 0.0
    baseline_value: float = 0.0
    drift_percent: float = 0.0
    severity: str = "warning"
    detected_at: float = 0.0


@dataclass
class TimeseriesPoint:
    """A single point in a time series."""
    date: str = ""
    value: float = 0.0


class AnalyticsEngine:
    """Aggregates data into actionable KPIs and time-series."""

    def __init__(self, config: Settings, db: Any = None) -> None:
        self._config = config
        self._db = db
        self._sqlite_conn: Any = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    async def load(self) -> None:
        if self._db is not None:
            self._loaded = True
            logger.info("AnalyticsEngine loaded (PostgreSQL)")
            return

        try:
            import aiosqlite
            db_path = str(self._config.training_db_path).replace("training.db", "analytics.db")
            self._sqlite_conn = await aiosqlite.connect(db_path)
            await self._sqlite_conn.execute("PRAGMA journal_mode=WAL")
            await self._create_sqlite_tables()
            self._loaded = True
            logger.info("AnalyticsEngine loaded (SQLite: %s)", db_path)
        except Exception as exc:
            raise AnalyticsError("Failed to load AnalyticsEngine", detail=str(exc), original_exception=exc)

    async def _create_sqlite_tables(self) -> None:
        assert self._sqlite_conn is not None
        await self._sqlite_conn.executescript("""
            CREATE TABLE IF NOT EXISTS analytics_snapshots (
                id TEXT PRIMARY KEY,
                avatar_id TEXT NOT NULL,
                period TEXT NOT NULL,
                period_date TEXT NOT NULL,
                total_conversations INTEGER DEFAULT 0,
                total_messages INTEGER DEFAULT 0,
                avg_response_time_ms REAL DEFAULT 0.0,
                avg_kb_confidence REAL DEFAULT 0.0,
                escalation_rate REAL DEFAULT 0.0,
                autonomy_percent REAL DEFAULT 0.0,
                resolution_time_avg_s REAL DEFAULT 0.0,
                accuracy_score REAL DEFAULT 0.0,
                total_cost REAL DEFAULT 0.0,
                unique_users INTEGER DEFAULT 0,
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_as_avatar_period
                ON analytics_snapshots(avatar_id, period, period_date);
        """)
        await self._sqlite_conn.commit()

    async def unload(self) -> None:
        if self._sqlite_conn is not None:
            await self._sqlite_conn.close()
            self._sqlite_conn = None
        self._loaded = False
        logger.info("AnalyticsEngine unloaded")

    # ── KPI Computation ──────────────────────────────────────────────────

    async def compute_kpis(
        self,
        avatar_id: str,
        period: str = "daily",
        start: Optional[float] = None,
        end: Optional[float] = None,
    ) -> KPISnapshot:
        """Compute KPIs for an avatar over a time period."""
        if not self._loaded:
            raise AnalyticsError("AnalyticsEngine not loaded")

        now = time.time()
        if end is None:
            end = now
        if start is None:
            if period == "daily":
                start = end - 86400
            elif period == "weekly":
                start = end - 7 * 86400
            else:  # monthly
                start = end - 30 * 86400

        kpis = KPISnapshot(period_start=start, period_end=end)

        if self._db is not None:
            kpis = await self._compute_kpis_pg(avatar_id, start, end, kpis)
        else:
            kpis = await self._compute_kpis_sqlite(avatar_id, start, end, kpis)

        return kpis

    async def _compute_kpis_pg(
        self, avatar_id: str, start: float, end: float, kpis: KPISnapshot,
    ) -> KPISnapshot:
        """Compute KPIs from PostgreSQL."""
        from sqlalchemy import select, func as sa_func
        from src.db.models import (
            Conversation, ConversationMessage, Escalation, QAPair as QAPairModel,
        )

        start_dt = datetime.fromtimestamp(start, tz=timezone.utc)
        end_dt = datetime.fromtimestamp(end, tz=timezone.utc)

        async with self._db.session() as session:
            # Conversations
            result = await session.execute(
                select(sa_func.count()).select_from(Conversation).where(
                    Conversation.avatar_id == avatar_id,
                    Conversation.started_at >= start_dt,
                    Conversation.started_at <= end_dt,
                )
            )
            kpis.total_conversations = result.scalar() or 0

            # Messages
            result = await session.execute(
                select(sa_func.count()).select_from(ConversationMessage).join(
                    Conversation, ConversationMessage.conversation_id == Conversation.id,
                ).where(
                    Conversation.avatar_id == avatar_id,
                    ConversationMessage.created_at >= start_dt,
                    ConversationMessage.created_at <= end_dt,
                )
            )
            kpis.total_messages = result.scalar() or 0

            # Avg response time
            result = await session.execute(
                select(sa_func.avg(ConversationMessage.latency_ms)).join(
                    Conversation, ConversationMessage.conversation_id == Conversation.id,
                ).where(
                    Conversation.avatar_id == avatar_id,
                    ConversationMessage.role == "assistant",
                    ConversationMessage.created_at >= start_dt,
                    ConversationMessage.created_at <= end_dt,
                )
            )
            avg = result.scalar()
            kpis.avg_response_time_ms = round(avg, 1) if avg else 0.0

            # Avg KB confidence
            result = await session.execute(
                select(sa_func.avg(ConversationMessage.kb_confidence)).join(
                    Conversation, ConversationMessage.conversation_id == Conversation.id,
                ).where(
                    Conversation.avatar_id == avatar_id,
                    ConversationMessage.kb_confidence.isnot(None),
                    ConversationMessage.created_at >= start_dt,
                    ConversationMessage.created_at <= end_dt,
                )
            )
            avg = result.scalar()
            kpis.avg_kb_confidence = round(avg, 3) if avg else 0.0

            # Escalation rate
            total_esc = (await session.execute(
                select(sa_func.count()).select_from(Escalation).where(
                    Escalation.avatar_id == avatar_id,
                    Escalation.created_at >= start_dt,
                    Escalation.created_at <= end_dt,
                )
            )).scalar() or 0
            if kpis.total_conversations > 0:
                kpis.escalation_rate = round(total_esc / kpis.total_conversations, 3)

            # Autonomy: messages handled by assistant / total messages
            assistant_msgs = (await session.execute(
                select(sa_func.count()).select_from(ConversationMessage).join(
                    Conversation, ConversationMessage.conversation_id == Conversation.id,
                ).where(
                    Conversation.avatar_id == avatar_id,
                    ConversationMessage.role == "assistant",
                    ConversationMessage.escalated == False,  # noqa: E712
                    ConversationMessage.created_at >= start_dt,
                    ConversationMessage.created_at <= end_dt,
                )
            )).scalar() or 0
            total_responses = (await session.execute(
                select(sa_func.count()).select_from(ConversationMessage).join(
                    Conversation, ConversationMessage.conversation_id == Conversation.id,
                ).where(
                    Conversation.avatar_id == avatar_id,
                    ConversationMessage.role.in_(["assistant", "operator"]),
                    ConversationMessage.created_at >= start_dt,
                    ConversationMessage.created_at <= end_dt,
                )
            )).scalar() or 0
            if total_responses > 0:
                kpis.autonomy_percent = round(assistant_msgs / total_responses * 100, 1)

            # Accuracy from Q&A quality ratings
            total_qa = (await session.execute(
                select(sa_func.count()).select_from(QAPairModel).where(
                    QAPairModel.avatar_id == avatar_id,
                    QAPairModel.created_at >= start_dt,
                    QAPairModel.created_at <= end_dt,
                )
            )).scalar() or 0
            good_qa = (await session.execute(
                select(sa_func.count()).select_from(QAPairModel).where(
                    QAPairModel.avatar_id == avatar_id,
                    QAPairModel.quality == "good",
                    QAPairModel.created_at >= start_dt,
                    QAPairModel.created_at <= end_dt,
                )
            )).scalar() or 0
            if total_qa > 0:
                kpis.accuracy_score = round(good_qa / total_qa, 3)

            # Total cost
            result = await session.execute(
                select(sa_func.sum(Conversation.total_cost)).where(
                    Conversation.avatar_id == avatar_id,
                    Conversation.started_at >= start_dt,
                    Conversation.started_at <= end_dt,
                )
            )
            kpis.total_cost = round(result.scalar() or 0.0, 2)

            # Unique users
            result = await session.execute(
                select(sa_func.count(sa_func.distinct(Conversation.caller_id))).where(
                    Conversation.avatar_id == avatar_id,
                    Conversation.started_at >= start_dt,
                    Conversation.started_at <= end_dt,
                )
            )
            kpis.unique_users = result.scalar() or 0

        return kpis

    async def _compute_kpis_sqlite(
        self, avatar_id: str, start: float, end: float, kpis: KPISnapshot,
    ) -> KPISnapshot:
        """Compute KPIs from SQLite (limited — uses training DB tables)."""
        # SQLite fallback: we only have access to training tables
        # (skills, qa_pairs, escalations), not full conversation tables.
        conn = self._sqlite_conn
        assert conn is not None

        # Use training DB for what we can compute
        try:
            import aiosqlite
            training_path = str(self._config.training_db_path)
            async with aiosqlite.connect(training_path) as tdb:
                # Escalations count
                cursor = await tdb.execute(
                    "SELECT COUNT(*) FROM escalations WHERE avatar_id = ? AND created_at >= ? AND created_at <= ?",
                    (avatar_id, start, end),
                )
                total_esc = (await cursor.fetchone())[0]

                # Q&A pairs
                cursor = await tdb.execute(
                    "SELECT COUNT(*) FROM qa_pairs WHERE avatar_id = ? AND created_at >= ? AND created_at <= ?",
                    (avatar_id, start, end),
                )
                total_qa = (await cursor.fetchone())[0]

                cursor = await tdb.execute(
                    "SELECT COUNT(*) FROM qa_pairs WHERE avatar_id = ? AND quality = 'good' AND created_at >= ? AND created_at <= ?",
                    (avatar_id, start, end),
                )
                good_qa = (await cursor.fetchone())[0]

                kpis.total_messages = total_qa
                if total_qa > 0:
                    kpis.escalation_rate = round(total_esc / max(total_qa, 1), 3)
                    kpis.accuracy_score = round(good_qa / total_qa, 3)
                    kpis.autonomy_percent = round((1 - kpis.escalation_rate) * 100, 1)
        except Exception:
            pass  # Degrade gracefully

        return kpis

    # ── Time Series ──────────────────────────────────────────────────────

    async def get_timeseries(
        self,
        avatar_id: str,
        metric: str,
        period: str = "daily",
        days: int = 30,
    ) -> list[TimeseriesPoint]:
        """Get time-series data for a specific metric."""
        if not self._loaded:
            raise AnalyticsError("AnalyticsEngine not loaded")

        # Check cached snapshots first
        points = await self._get_cached_timeseries(avatar_id, metric, period, days)
        if points:
            return points

        # Compute on-the-fly
        now = datetime.now(timezone.utc)
        points = []

        for d in range(days, 0, -1):
            day = now - timedelta(days=d)
            day_start = day.replace(hour=0, minute=0, second=0, microsecond=0)
            day_end = day_start + timedelta(days=1)
            date_str = day_start.strftime("%Y-%m-%d")

            kpis = await self.compute_kpis(
                avatar_id, period="daily",
                start=day_start.timestamp(),
                end=day_end.timestamp(),
            )

            value = 0.0
            if metric == "conversations":
                value = float(kpis.total_conversations)
            elif metric == "escalation_rate":
                value = kpis.escalation_rate
            elif metric == "autonomy":
                value = kpis.autonomy_percent
            elif metric == "response_time":
                value = kpis.avg_response_time_ms
            elif metric == "accuracy":
                value = kpis.accuracy_score
            elif metric == "cost":
                value = kpis.total_cost
            elif metric == "confidence":
                value = kpis.avg_kb_confidence

            points.append(TimeseriesPoint(date=date_str, value=round(value, 3)))

        return points

    async def _get_cached_timeseries(
        self, avatar_id: str, metric: str, period: str, days: int,
    ) -> list[TimeseriesPoint]:
        """Try to load from cached analytics_snapshots."""
        if self._db is not None:
            from sqlalchemy import select
            from src.db.models import AnalyticsSnapshot

            since = datetime.now(timezone.utc) - timedelta(days=days)
            async with self._db.session() as session:
                result = await session.execute(
                    select(AnalyticsSnapshot).where(
                        AnalyticsSnapshot.avatar_id == avatar_id,
                        AnalyticsSnapshot.period == period,
                        AnalyticsSnapshot.period_date >= since,
                    ).order_by(AnalyticsSnapshot.period_date)
                )
                rows = result.scalars().all()
                if not rows:
                    return []

                points = []
                for row in rows:
                    date_str = row.period_date.strftime("%Y-%m-%d") if isinstance(row.period_date, datetime) else str(row.period_date)
                    value = getattr(row, metric.replace("response_time", "avg_response_time_ms")
                                    .replace("confidence", "avg_kb_confidence")
                                    .replace("autonomy", "autonomy_percent")
                                    .replace("conversations", "total_conversations")
                                    .replace("cost", "total_cost")
                                    .replace("accuracy", "accuracy_score"),
                                    0.0)
                    points.append(TimeseriesPoint(date=date_str, value=round(float(value), 3)))
                return points

        return []  # SQLite fallback doesn't cache

    # ── Dashboard Bundle ─────────────────────────────────────────────────

    async def get_dashboard_data(self, avatar_id: str, days: int = 30) -> dict:
        """Get all analytics data needed for the dashboard."""
        if not self._loaded:
            raise AnalyticsError("AnalyticsEngine not loaded")

        kpis = await self.compute_kpis(avatar_id, period="monthly", start=time.time() - days * 86400)
        drift = await self.check_drift(avatar_id)

        # Compute a few key trends (conversations, autonomy)
        conv_trend = await self.get_timeseries(avatar_id, "conversations", "daily", min(days, 14))
        autonomy_trend = await self.get_timeseries(avatar_id, "autonomy", "daily", min(days, 14))

        # Get top/bottom skills from training engine data
        top_skills: list[dict] = []
        bottom_skills: list[dict] = []
        if self._db is not None:
            from sqlalchemy import select
            from src.db.models import Skill as SkillModel

            async with self._db.session() as session:
                result = await session.execute(
                    select(SkillModel).where(SkillModel.avatar_id == avatar_id).order_by(SkillModel.progress.desc())
                )
                all_skills = result.scalars().all()
                for s in all_skills[:5]:
                    top_skills.append({"skill_id": s.id, "name": s.name, "progress": s.progress})
                for s in all_skills[-5:]:
                    bottom_skills.append({"skill_id": s.id, "name": s.name, "progress": s.progress})

        return {
            "kpis": {
                "total_conversations": kpis.total_conversations,
                "total_messages": kpis.total_messages,
                "avg_response_time_ms": kpis.avg_response_time_ms,
                "avg_kb_confidence": kpis.avg_kb_confidence,
                "escalation_rate": kpis.escalation_rate,
                "autonomy_percent": kpis.autonomy_percent,
                "accuracy_score": kpis.accuracy_score,
                "total_cost": kpis.total_cost,
                "unique_users": kpis.unique_users,
            },
            "trends": {
                "conversations": [{"date": p.date, "value": p.value} for p in conv_trend],
                "autonomy": [{"date": p.date, "value": p.value} for p in autonomy_trend],
            },
            "drift_alerts": [
                {"metric": a.metric_name, "current": a.current_value, "baseline": a.baseline_value,
                 "drift_percent": a.drift_percent, "severity": a.severity}
                for a in drift
            ],
            "top_skills": top_skills,
            "bottom_skills": bottom_skills,
        }

    # ── Performance Drift Detection ──────────────────────────────────────

    async def check_drift(
        self,
        avatar_id: str,
        baseline_days: int = 30,
        recent_days: int = 7,
    ) -> list[DriftAlert]:
        """Detect performance drift by comparing recent vs baseline periods."""
        if not self._loaded:
            raise AnalyticsError("AnalyticsEngine not loaded")

        now = time.time()
        baseline = await self.compute_kpis(
            avatar_id, period="monthly",
            start=now - baseline_days * 86400,
            end=now - recent_days * 86400,
        )
        recent = await self.compute_kpis(
            avatar_id, period="weekly",
            start=now - recent_days * 86400,
            end=now,
        )

        alerts: list[DriftAlert] = []

        # Check each metric for significant drift
        checks = [
            ("escalation_rate", baseline.escalation_rate, recent.escalation_rate, True),   # higher = worse
            ("accuracy_score", baseline.accuracy_score, recent.accuracy_score, False),      # lower = worse
            ("autonomy_percent", baseline.autonomy_percent, recent.autonomy_percent, False),# lower = worse
            ("avg_response_time_ms", baseline.avg_response_time_ms, recent.avg_response_time_ms, True),  # higher = worse
        ]

        for metric, baseline_val, recent_val, higher_is_worse in checks:
            if baseline_val == 0:
                continue

            drift_pct = ((recent_val - baseline_val) / baseline_val) * 100

            # Determine if this drift is concerning
            is_concerning = (
                (higher_is_worse and drift_pct > 20) or
                (not higher_is_worse and drift_pct < -20)
            )
            is_critical = (
                (higher_is_worse and drift_pct > 50) or
                (not higher_is_worse and drift_pct < -50)
            )

            if is_concerning:
                alerts.append(DriftAlert(
                    metric_name=metric,
                    current_value=round(recent_val, 3),
                    baseline_value=round(baseline_val, 3),
                    drift_percent=round(drift_pct, 1),
                    severity="critical" if is_critical else "warning",
                    detected_at=now,
                ))

        return alerts

    # ── Report Export ─────────────────────────────────────────────────────

    async def export_report(self, avatar_id: str, days: int = 30) -> dict:
        """Export a full analytics report."""
        if not self._loaded:
            raise AnalyticsError("AnalyticsEngine not loaded")

        dashboard = await self.get_dashboard_data(avatar_id, days)
        dashboard["avatar_id"] = avatar_id
        dashboard["period_days"] = days
        dashboard["generated_at"] = time.time()

        return dashboard
