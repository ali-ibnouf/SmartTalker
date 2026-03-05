"""Guardrails Engine — content policy enforcement and output validation.

Provides:
- Per-avatar content policies (blocked topics, disclaimers, length limits, keyword escalation)
- LLM output validation before sending to user
- Policy violation recording and audit trail
- Integration point between LLM generation and TTS synthesis
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from src.config import Settings
from src.utils.exceptions import GuardrailsError
from src.utils.logger import setup_logger
from src.utils.metrics import GUARDRAIL_VIOLATIONS

logger = setup_logger("pipeline.guardrails")


@dataclass
class PolicyConfig:
    """Content policy for an avatar."""
    blocked_topics: list[str] = field(default_factory=list)
    required_disclaimers: list[str] = field(default_factory=list)
    max_response_length: int = 2000
    escalation_keywords: list[str] = field(default_factory=list)
    is_active: bool = True


@dataclass
class GuardrailCheckResult:
    """Result of a guardrails check on LLM output."""
    passed: bool = True
    violations: list[dict] = field(default_factory=list)
    sanitized_text: str = ""
    disclaimers_added: list[str] = field(default_factory=list)
    escalation_triggered: bool = False
    escalation_reason: str = ""


@dataclass
class ViolationRecord:
    """A recorded policy violation."""
    id: str = ""
    avatar_id: str = ""
    session_id: str = ""
    violation_type: str = ""
    original_response: str = ""
    sanitized_response: str = ""
    details: dict = field(default_factory=dict)
    severity: str = "warning"
    created_at: float = 0.0


class GuardrailsEngine:
    """Content policy enforcement and output validation engine."""

    def __init__(self, config: Settings, db: Any = None) -> None:
        self._config = config
        self._db = db
        self._sqlite_conn: Any = None
        self._loaded = False
        # Default policy from config
        self._default_policy = PolicyConfig(
            blocked_topics=[t.strip() for t in config.guardrails_blocked_topics.split(",") if t.strip()],
            required_disclaimers=[d.strip() for d in config.guardrails_required_disclaimers.split(",") if d.strip()],
            max_response_length=config.guardrails_max_response_length,
        )

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    async def load(self) -> None:
        if self._db is not None:
            self._loaded = True
            logger.info("GuardrailsEngine loaded (PostgreSQL)")
            return

        try:
            import aiosqlite
            db_path = str(self._config.training_db_path).replace("training.db", "guardrails.db")
            self._sqlite_conn = await aiosqlite.connect(db_path)
            await self._sqlite_conn.execute("PRAGMA journal_mode=WAL")
            await self._create_sqlite_tables()
            self._loaded = True
            logger.info("GuardrailsEngine loaded (SQLite: %s)", db_path)
        except Exception as exc:
            raise GuardrailsError("Failed to load GuardrailsEngine", detail=str(exc), original_exception=exc)

    async def _create_sqlite_tables(self) -> None:
        assert self._sqlite_conn is not None
        await self._sqlite_conn.executescript("""
            CREATE TABLE IF NOT EXISTS guardrail_policies (
                id TEXT PRIMARY KEY,
                avatar_id TEXT NOT NULL UNIQUE,
                blocked_topics TEXT DEFAULT '[]',
                required_disclaimers TEXT DEFAULT '[]',
                max_response_length INTEGER DEFAULT 2000,
                escalation_keywords TEXT DEFAULT '[]',
                is_active INTEGER DEFAULT 1,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_gp_avatar ON guardrail_policies(avatar_id);

            CREATE TABLE IF NOT EXISTS policy_violations (
                id TEXT PRIMARY KEY,
                avatar_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                violation_type TEXT NOT NULL,
                original_response TEXT DEFAULT '',
                sanitized_response TEXT DEFAULT '',
                details TEXT DEFAULT '{}',
                severity TEXT DEFAULT 'warning',
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_pv_avatar ON policy_violations(avatar_id);
            CREATE INDEX IF NOT EXISTS idx_pv_session ON policy_violations(session_id);
            CREATE INDEX IF NOT EXISTS idx_pv_type ON policy_violations(violation_type);
            CREATE INDEX IF NOT EXISTS idx_pv_created ON policy_violations(created_at);
        """)
        await self._sqlite_conn.commit()

    async def unload(self) -> None:
        if self._sqlite_conn is not None:
            await self._sqlite_conn.close()
            self._sqlite_conn = None
        self._loaded = False
        logger.info("GuardrailsEngine unloaded")

    # ── Policy CRUD ──────────────────────────────────────────────────────

    async def get_policy(self, avatar_id: str) -> PolicyConfig:
        """Get the content policy for an avatar, falling back to defaults."""
        if not self._loaded:
            raise GuardrailsError("GuardrailsEngine not loaded")

        if self._db is not None:
            from sqlalchemy import select
            from src.db.models import GuardrailPolicy

            async with self._db.session() as session:
                result = await session.execute(
                    select(GuardrailPolicy).where(GuardrailPolicy.avatar_id == avatar_id)
                )
                row = result.scalar_one_or_none()
                if row:
                    return PolicyConfig(
                        blocked_topics=json.loads(row.blocked_topics) if row.blocked_topics else [],
                        required_disclaimers=json.loads(row.required_disclaimers) if row.required_disclaimers else [],
                        max_response_length=row.max_response_length,
                        escalation_keywords=json.loads(row.escalation_keywords) if row.escalation_keywords else [],
                        is_active=row.is_active,
                    )
        else:
            conn = self._sqlite_conn
            assert conn is not None
            cursor = await conn.execute(
                "SELECT blocked_topics, required_disclaimers, max_response_length, escalation_keywords, is_active FROM guardrail_policies WHERE avatar_id = ?",
                (avatar_id,),
            )
            row = await cursor.fetchone()
            if row:
                return PolicyConfig(
                    blocked_topics=json.loads(row[0]) if row[0] else [],
                    required_disclaimers=json.loads(row[1]) if row[1] else [],
                    max_response_length=row[2],
                    escalation_keywords=json.loads(row[3]) if row[3] else [],
                    is_active=bool(row[4]),
                )

        return PolicyConfig(
            blocked_topics=list(self._default_policy.blocked_topics),
            required_disclaimers=list(self._default_policy.required_disclaimers),
            max_response_length=self._default_policy.max_response_length,
        )

    async def set_policy(self, avatar_id: str, policy: PolicyConfig) -> None:
        """Create or update the content policy for an avatar."""
        if not self._loaded:
            raise GuardrailsError("GuardrailsEngine not loaded")

        now = time.time()
        blocked_json = json.dumps(policy.blocked_topics, ensure_ascii=False)
        disclaimers_json = json.dumps(policy.required_disclaimers, ensure_ascii=False)
        keywords_json = json.dumps(policy.escalation_keywords, ensure_ascii=False)

        if self._db is not None:
            from sqlalchemy import select
            from src.db.models import GuardrailPolicy

            async with self._db.session() as session:
                result = await session.execute(
                    select(GuardrailPolicy).where(GuardrailPolicy.avatar_id == avatar_id)
                )
                existing = result.scalar_one_or_none()
                if existing:
                    existing.blocked_topics = blocked_json
                    existing.required_disclaimers = disclaimers_json
                    existing.max_response_length = policy.max_response_length
                    existing.escalation_keywords = keywords_json
                    existing.is_active = policy.is_active
                else:
                    session.add(GuardrailPolicy(
                        id=uuid.uuid4().hex,
                        avatar_id=avatar_id,
                        blocked_topics=blocked_json,
                        required_disclaimers=disclaimers_json,
                        max_response_length=policy.max_response_length,
                        escalation_keywords=keywords_json,
                        is_active=policy.is_active,
                        created_at=datetime.now(timezone.utc),
                        updated_at=datetime.now(timezone.utc),
                    ))
                await session.commit()
        else:
            conn = self._sqlite_conn
            assert conn is not None
            cursor = await conn.execute(
                "SELECT id FROM guardrail_policies WHERE avatar_id = ?", (avatar_id,)
            )
            row = await cursor.fetchone()
            if row:
                await conn.execute(
                    "UPDATE guardrail_policies SET blocked_topics=?, required_disclaimers=?, max_response_length=?, escalation_keywords=?, is_active=?, updated_at=? WHERE avatar_id=?",
                    (blocked_json, disclaimers_json, policy.max_response_length, keywords_json, int(policy.is_active), now, avatar_id),
                )
            else:
                await conn.execute(
                    "INSERT INTO guardrail_policies (id, avatar_id, blocked_topics, required_disclaimers, max_response_length, escalation_keywords, is_active, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (uuid.uuid4().hex, avatar_id, blocked_json, disclaimers_json, policy.max_response_length, keywords_json, int(policy.is_active), now, now),
                )
            await conn.commit()

        logger.info("Policy updated for avatar %s", avatar_id)

    async def delete_policy(self, avatar_id: str) -> bool:
        """Delete the content policy for an avatar."""
        if not self._loaded:
            raise GuardrailsError("GuardrailsEngine not loaded")

        if self._db is not None:
            from sqlalchemy import delete
            from src.db.models import GuardrailPolicy

            async with self._db.session() as session:
                result = await session.execute(
                    delete(GuardrailPolicy).where(GuardrailPolicy.avatar_id == avatar_id)
                )
                await session.commit()
                return result.rowcount > 0
        else:
            conn = self._sqlite_conn
            assert conn is not None
            cursor = await conn.execute(
                "DELETE FROM guardrail_policies WHERE avatar_id = ?", (avatar_id,)
            )
            await conn.commit()
            return cursor.rowcount > 0

    # ── Output Validation ────────────────────────────────────────────────

    async def check_response(
        self,
        avatar_id: str,
        session_id: str,
        response_text: str,
        user_question: str,
    ) -> GuardrailCheckResult:
        """Validate LLM response against the avatar's content policy."""
        if not self._loaded:
            raise GuardrailsError("GuardrailsEngine not loaded")

        policy = await self.get_policy(avatar_id)
        if not policy.is_active:
            return GuardrailCheckResult(passed=True, sanitized_text=response_text)

        result = GuardrailCheckResult(sanitized_text=response_text)

        # Check blocked topics
        topic_violations = self._check_blocked_topics(response_text, policy.blocked_topics)
        if topic_violations:
            result.violations.extend(topic_violations)
            result.passed = False
            result.sanitized_text = "I'm unable to discuss that topic. Please ask me something else."
            for v in topic_violations:
                GUARDRAIL_VIOLATIONS.labels(avatar_id=avatar_id, violation_type="blocked_topic").inc()
                await self.record_violation(
                    avatar_id, session_id, "blocked_topic",
                    response_text, result.sanitized_text,
                    v, "blocked",
                )

        # Check response length
        length_violations = self._check_response_length(response_text, policy.max_response_length)
        if length_violations:
            result.violations.extend(length_violations)
            if result.passed:  # Only truncate if not already blocked
                result.sanitized_text = response_text[:policy.max_response_length]
                result.passed = False
            GUARDRAIL_VIOLATIONS.labels(avatar_id=avatar_id, violation_type="length_exceeded").inc()
            await self.record_violation(
                avatar_id, session_id, "length_exceeded",
                response_text, result.sanitized_text,
                length_violations[0], "warning",
            )

        # Check escalation keywords
        should_escalate, keyword = self._check_escalation_keywords(
            user_question + " " + response_text, policy.escalation_keywords,
        )
        if should_escalate:
            result.escalation_triggered = True
            result.escalation_reason = f"Keyword match: {keyword}"
            GUARDRAIL_VIOLATIONS.labels(avatar_id=avatar_id, violation_type="keyword_escalation").inc()

        # Add disclaimers if needed
        text_to_check = result.sanitized_text if not result.passed else response_text
        new_text, added = self._check_disclaimers(text_to_check, policy.required_disclaimers)
        if added:
            result.sanitized_text = new_text
            result.disclaimers_added = added

        return result

    @staticmethod
    def _check_blocked_topics(text: str, blocked: list[str]) -> list[dict]:
        """Check if response contains any blocked topic keywords."""
        violations: list[dict] = []
        text_lower = text.lower()
        for topic in blocked:
            if not topic:
                continue
            pattern = re.compile(re.escape(topic.lower()))
            if pattern.search(text_lower):
                violations.append({
                    "type": "blocked_topic",
                    "topic": topic,
                    "message": f"Response contains blocked topic: {topic}",
                })
        return violations

    @staticmethod
    def _check_response_length(text: str, max_len: int) -> list[dict]:
        """Check if response exceeds maximum length."""
        if len(text) > max_len:
            return [{
                "type": "length_exceeded",
                "actual_length": len(text),
                "max_length": max_len,
                "message": f"Response length {len(text)} exceeds max {max_len}",
            }]
        return []

    @staticmethod
    def _check_escalation_keywords(text: str, keywords: list[str]) -> tuple[bool, str]:
        """Check if text contains any escalation trigger keywords."""
        text_lower = text.lower()
        for keyword in keywords:
            if not keyword:
                continue
            if keyword.lower() in text_lower:
                return True, keyword
        return False, ""

    @staticmethod
    def _check_disclaimers(text: str, required: list[str]) -> tuple[str, list[str]]:
        """Ensure required disclaimers are present, adding them if missing."""
        added: list[str] = []
        text_lower = text.lower()
        for disclaimer in required:
            if not disclaimer:
                continue
            if disclaimer.lower() not in text_lower:
                added.append(disclaimer)
        if added:
            disclaimer_text = "\n\n" + " ".join(added)
            return text + disclaimer_text, added
        return text, []

    # ── Violation Recording ──────────────────────────────────────────────

    async def record_violation(
        self,
        avatar_id: str,
        session_id: str,
        violation_type: str,
        original: str,
        sanitized: str,
        details: dict,
        severity: str = "warning",
    ) -> str:
        """Record a policy violation."""
        violation_id = uuid.uuid4().hex
        now = time.time()

        if self._db is not None:
            from src.db.models import PolicyViolation

            async with self._db.session() as session:
                session.add(PolicyViolation(
                    id=violation_id,
                    avatar_id=avatar_id,
                    session_id=session_id,
                    violation_type=violation_type,
                    original_response=original[:5000],  # Limit stored size
                    sanitized_response=sanitized[:5000],
                    details=json.dumps(details, ensure_ascii=False),
                    severity=severity,
                    created_at=datetime.now(timezone.utc),
                ))
                await session.commit()
        else:
            conn = self._sqlite_conn
            assert conn is not None
            await conn.execute(
                "INSERT INTO policy_violations (id, avatar_id, session_id, violation_type, original_response, sanitized_response, details, severity, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (violation_id, avatar_id, session_id, violation_type, original[:5000], sanitized[:5000], json.dumps(details, ensure_ascii=False), severity, now),
            )
            await conn.commit()

        logger.warning(
            "Policy violation [%s] for avatar %s in session %s: %s",
            violation_type, avatar_id, session_id, severity,
        )
        return violation_id

    # ── Audit Trail ──────────────────────────────────────────────────────

    async def list_violations(
        self,
        avatar_id: str,
        violation_type: Optional[str] = None,
        since: Optional[float] = None,
        limit: int = 100,
    ) -> list[ViolationRecord]:
        """Query recorded violations for audit."""
        if not self._loaded:
            raise GuardrailsError("GuardrailsEngine not loaded")

        records: list[ViolationRecord] = []

        if self._db is not None:
            from sqlalchemy import select
            from src.db.models import PolicyViolation

            async with self._db.session() as session:
                query = select(PolicyViolation).where(PolicyViolation.avatar_id == avatar_id)
                if violation_type:
                    query = query.where(PolicyViolation.violation_type == violation_type)
                if since:
                    since_dt = datetime.fromtimestamp(since, tz=timezone.utc)
                    query = query.where(PolicyViolation.created_at >= since_dt)
                query = query.order_by(PolicyViolation.created_at.desc()).limit(limit)
                result = await session.execute(query)
                for row in result.scalars().all():
                    records.append(ViolationRecord(
                        id=row.id,
                        avatar_id=row.avatar_id,
                        session_id=row.session_id,
                        violation_type=row.violation_type,
                        original_response=row.original_response,
                        sanitized_response=row.sanitized_response,
                        details=json.loads(row.details) if row.details else {},
                        severity=row.severity,
                        created_at=row.created_at.timestamp() if isinstance(row.created_at, datetime) else 0.0,
                    ))
        else:
            conn = self._sqlite_conn
            assert conn is not None
            sql = "SELECT id, avatar_id, session_id, violation_type, original_response, sanitized_response, details, severity, created_at FROM policy_violations WHERE avatar_id = ?"
            params: list[Any] = [avatar_id]
            if violation_type:
                sql += " AND violation_type = ?"
                params.append(violation_type)
            if since:
                sql += " AND created_at >= ?"
                params.append(since)
            sql += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            cursor = await conn.execute(sql, params)
            for row in await cursor.fetchall():
                records.append(ViolationRecord(
                    id=row[0], avatar_id=row[1], session_id=row[2],
                    violation_type=row[3], original_response=row[4],
                    sanitized_response=row[5],
                    details=json.loads(row[6]) if row[6] else {},
                    severity=row[7], created_at=row[8] or 0.0,
                ))

        return records
