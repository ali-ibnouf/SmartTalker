"""SQLAlchemy ORM models for SmartTalker Central.

Tables:
- customers          Multi-tenant customer accounts
- subscriptions      Billing plans per customer
- avatars            Maskki digital employee avatars
- conversations      Chat sessions
- conversation_messages  Individual messages within conversations
- skills             Trainable skills per avatar
- qa_pairs           Q&A captured from human operators
- escalations        Escalation events
- job_personas       Generalized job persona catalog
- persona_skills     Skills pre-populated in a persona
- usage_records      Per-second billing metering
- render_nodes       GPU Render Node registry (deprecated — RunPod Serverless)
- learning_metrics   Daily learning quality metrics per skill
- guardrail_policies Per-avatar content policy configuration
- policy_violations  Logged guardrail violations
- operator_actions   Operator activity log for supervisor
- decision_reviews   Flagged AI decisions for review
- analytics_snapshots Cached KPI aggregations
- employees          Digital employee definitions (Phase 2)
- employee_knowledge KB Q&A entries per employee
- employee_learning  Learning queue (pending human review)
- tool_registry      Custom API tool definitions
- employee_tools     N:M employee ↔ tool assignment
- visitor_profiles   Per-visitor identity & metadata
- visitor_memories   Per-visitor contextual memories
- tool_execution_log Tool call audit trail
- workflows          Workflow definitions with step sequences
- workflow_executions Workflow run instances
- api_cost_records   Per-call cost tracking by service
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


def _uuid() -> str:
    return uuid.uuid4().hex


# =============================================================================
# Customer & Subscription
# =============================================================================


class Customer(Base):
    __tablename__ = "customers"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    company: Mapped[str] = mapped_column(String(255), default="")
    api_key: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    suspended: Mapped[bool] = mapped_column(Boolean, default=False)
    suspended_reason: Mapped[str] = mapped_column(Text, default="")
    extra_seconds_remaining: Mapped[int] = mapped_column(Integer, default=0)
    operator_language: Mapped[str] = mapped_column(String(10), default="ar")
    data_language: Mapped[str] = mapped_column(String(10), default="ar")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    subscriptions: Mapped[list["Subscription"]] = relationship(back_populates="customer")
    avatars: Mapped[list["Avatar"]] = relationship(back_populates="customer")
    usage_records: Mapped[list["UsageRecord"]] = relationship(back_populates="customer")


class Subscription(Base):
    __tablename__ = "subscriptions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    customer_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("customers.id"), nullable=False
    )
    plan: Mapped[str] = mapped_column(String(50), default="starter")  # starter, professional, business, enterprise
    monthly_seconds: Mapped[int] = mapped_column(Integer, default=50000)
    rate_per_second: Mapped[float] = mapped_column(Float, default=0.001)
    max_avatars: Mapped[int] = mapped_column(Integer, default=1)
    max_concurrent_sessions: Mapped[int] = mapped_column(Integer, default=1)
    price_monthly: Mapped[float] = mapped_column(Float, default=50.0)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    payment_failures: Mapped[int] = mapped_column(Integer, default=0)
    grace_period_until: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    starts_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    customer: Mapped["Customer"] = relationship(back_populates="subscriptions")


# =============================================================================
# Avatar & Conversations
# =============================================================================


class Avatar(Base):
    __tablename__ = "avatars"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    customer_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("customers.id"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    photo_url: Mapped[str] = mapped_column(Text, default="")
    voice_id: Mapped[str] = mapped_column(String(128), default="default")
    persona_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    language: Mapped[str] = mapped_column(String(10), default="ar")
    is_live: Mapped[bool] = mapped_column(Boolean, default=False)
    training_progress: Mapped[float] = mapped_column(Float, default=0.0)
    avatar_type: Mapped[str] = mapped_column(String(20), default="vrm")  # "video" | "vrm"
    vrm_url: Mapped[str] = mapped_column(Text, default="")
    photo_preprocessed: Mapped[bool] = mapped_column(Boolean, default=False)
    face_data_url: Mapped[str] = mapped_column(Text, default="")
    voice_model: Mapped[str] = mapped_column(String(100), default="qwen3-tts-vc-realtime")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    customer: Mapped["Customer"] = relationship(back_populates="avatars")
    conversations: Mapped[list["Conversation"]] = relationship(back_populates="avatar")
    skills: Mapped[list["Skill"]] = relationship(back_populates="avatar")

    __table_args__ = (Index("idx_avatars_customer", "customer_id"),)


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    avatar_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("avatars.id"), nullable=False
    )
    channel: Mapped[str] = mapped_column(String(50), default="web")  # web, whatsapp, api
    caller_id: Mapped[str] = mapped_column(String(255), default="")
    language: Mapped[str] = mapped_column(String(10), default="auto")
    started_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    ended_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    duration_s: Mapped[float] = mapped_column(Float, default=0.0)
    message_count: Mapped[int] = mapped_column(Integer, default=0)
    total_cost: Mapped[float] = mapped_column(Float, default=0.0)
    gpu_cost: Mapped[float] = mapped_column(Float, default=0.0)

    avatar: Mapped["Avatar"] = relationship(back_populates="conversations")
    messages: Mapped[list["ConversationMessage"]] = relationship(back_populates="conversation")

    __table_args__ = (
        Index("idx_conversations_avatar", "avatar_id"),
        Index("idx_conversations_started", "started_at"),
    )


class ConversationMessage(Base):
    __tablename__ = "conversation_messages"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    conversation_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("conversations.id"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # user, assistant, operator
    content: Mapped[str] = mapped_column(Text, nullable=False)
    emotion: Mapped[str] = mapped_column(String(50), default="neutral")
    language: Mapped[str] = mapped_column(String(10), default="")
    kb_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    escalated: Mapped[bool] = mapped_column(Boolean, default=False)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")

    __table_args__ = (Index("idx_messages_conversation", "conversation_id"),)


# =============================================================================
# Training: Skills, QA Pairs, Escalations
# =============================================================================


class Skill(Base):
    __tablename__ = "skills"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    avatar_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("avatars.id"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    target_threshold: Mapped[float] = mapped_column(Float, default=0.7)
    effective_threshold: Mapped[float] = mapped_column(Float, default=0.7)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    qa_count: Mapped[int] = mapped_column(Integer, default=0)
    bad_ratio: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    avatar: Mapped["Avatar"] = relationship(back_populates="skills")
    qa_pairs: Mapped[list["QAPair"]] = relationship(back_populates="skill")

    __table_args__ = (Index("idx_skills_avatar", "avatar_id"),)


class QAPair(Base):
    __tablename__ = "qa_pairs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    skill_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("skills.id"), nullable=False
    )
    avatar_id: Mapped[str] = mapped_column(String(64), nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    human_answer: Mapped[str] = mapped_column(Text, nullable=False)
    ai_answer: Mapped[str] = mapped_column(Text, default="")
    quality: Mapped[str] = mapped_column(String(20), default="none")  # good, bad, none
    ingested: Mapped[bool] = mapped_column(Boolean, default=False)
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    correction_of: Mapped[str | None] = mapped_column(String(64), nullable=True)
    confidence_at_time: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    skill: Mapped["Skill"] = relationship(back_populates="qa_pairs")

    __table_args__ = (
        Index("idx_qa_skill", "skill_id"),
        Index("idx_qa_avatar", "avatar_id"),
    )


class Escalation(Base):
    __tablename__ = "escalations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False)
    avatar_id: Mapped[str] = mapped_column(String(64), nullable=False)
    skill_id: Mapped[str] = mapped_column(String(64), default="unknown")
    question: Mapped[str] = mapped_column(Text, default="")
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    resolved: Mapped[bool] = mapped_column(Boolean, default=False)
    resolution: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    __table_args__ = (
        Index("idx_escalations_avatar", "avatar_id"),
        Index("idx_escalations_unresolved", "resolved"),
    )


# =============================================================================
# Job Persona Engine
# =============================================================================


class JobPersona(Base):
    __tablename__ = "job_personas"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    industry: Mapped[str] = mapped_column(String(128), default="general")
    description: Mapped[str] = mapped_column(Text, default="")
    source_avatar_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    is_public: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    skills: Mapped[list["PersonaSkill"]] = relationship(back_populates="persona")

    __table_args__ = (Index("idx_personas_industry", "industry"),)


class PersonaSkill(Base):
    __tablename__ = "persona_skills"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    persona_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("job_personas.id"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    pre_populated_progress: Mapped[float] = mapped_column(Float, default=70.0)

    persona: Mapped["JobPersona"] = relationship(back_populates="skills")


# =============================================================================
# Billing
# =============================================================================


class UsageRecord(Base):
    __tablename__ = "usage_records"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    customer_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("customers.id"), nullable=False
    )
    session_id: Mapped[str] = mapped_column(String(64), nullable=False)
    avatar_id: Mapped[str] = mapped_column(String(64), default="")
    channel: Mapped[str] = mapped_column(String(50), default="web")
    duration_s: Mapped[float] = mapped_column(Float, nullable=False)
    cost: Mapped[float] = mapped_column(Float, nullable=False)
    runpod_job_id: Mapped[str] = mapped_column(String(100), default="")
    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    ended_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    customer: Mapped["Customer"] = relationship(back_populates="usage_records")

    __table_args__ = (
        Index("idx_usage_customer", "customer_id"),
        Index("idx_usage_session", "session_id"),
        Index("idx_usage_period", "started_at", "ended_at"),
    )


# =============================================================================
# Render Nodes (deprecated — GPU rendering now via RunPod Serverless)
# =============================================================================


class RenderNode(Base):
    __tablename__ = "render_nodes"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    hostname: Mapped[str] = mapped_column(String(255), nullable=False)
    license_key: Mapped[str] = mapped_column(String(128), default="")
    customer_id: Mapped[str] = mapped_column(String(64), default="")
    gpu_type: Mapped[str] = mapped_column(String(128), default="")
    vram_mb: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(20), default="offline")  # online, offline, busy
    current_fps: Mapped[float] = mapped_column(Float, default=0.0)
    last_heartbeat: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    registered_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


# =============================================================================
# Learning Analytics
# =============================================================================


class LearningMetric(Base):
    __tablename__ = "learning_metrics"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    avatar_id: Mapped[str] = mapped_column(String(64), nullable=False)
    skill_id: Mapped[str] = mapped_column(String(64), nullable=False)
    date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    qa_added: Mapped[int] = mapped_column(Integer, default=0)
    good_count: Mapped[int] = mapped_column(Integer, default=0)
    bad_count: Mapped[int] = mapped_column(Integer, default=0)
    corrections_count: Mapped[int] = mapped_column(Integer, default=0)
    avg_confidence_before: Mapped[float] = mapped_column(Float, default=0.0)
    avg_confidence_after: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    __table_args__ = (
        Index("idx_lm_avatar_skill_date", "avatar_id", "skill_id", "date"),
    )


# =============================================================================
# Guardrails
# =============================================================================


class GuardrailPolicy(Base):
    __tablename__ = "guardrail_policies"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    avatar_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    blocked_topics: Mapped[str] = mapped_column(Text, default="[]")  # JSON array
    required_disclaimers: Mapped[str] = mapped_column(Text, default="[]")  # JSON array
    max_response_length: Mapped[int] = mapped_column(Integer, default=2000)
    escalation_keywords: Mapped[str] = mapped_column(Text, default="[]")  # JSON array
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (Index("idx_guardrail_avatar", "avatar_id"),)


class PolicyViolation(Base):
    __tablename__ = "policy_violations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    avatar_id: Mapped[str] = mapped_column(String(64), nullable=False)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False)
    violation_type: Mapped[str] = mapped_column(String(50), nullable=False)
    original_response: Mapped[str] = mapped_column(Text, default="")
    sanitized_response: Mapped[str] = mapped_column(Text, default="")
    details: Mapped[str] = mapped_column(Text, default="{}")  # JSON
    severity: Mapped[str] = mapped_column(String(20), default="warning")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    __table_args__ = (
        Index("idx_violations_avatar", "avatar_id"),
        Index("idx_violations_session", "session_id"),
        Index("idx_violations_type", "violation_type"),
        Index("idx_violations_created", "created_at"),
    )


# =============================================================================
# Supervisor
# =============================================================================


class OperatorAction(Base):
    __tablename__ = "operator_actions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    operator_id: Mapped[str] = mapped_column(String(64), nullable=False)
    action_type: Mapped[str] = mapped_column(String(50), nullable=False)
    session_id: Mapped[str] = mapped_column(String(64), default="")
    avatar_id: Mapped[str] = mapped_column(String(64), default="")
    details: Mapped[str] = mapped_column(Text, default="{}")  # JSON
    response_time_ms: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    __table_args__ = (
        Index("idx_opactions_operator", "operator_id"),
        Index("idx_opactions_created", "created_at"),
    )


class DecisionReview(Base):
    __tablename__ = "decision_reviews"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False)
    avatar_id: Mapped[str] = mapped_column(String(64), nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    ai_response: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    flagged_reason: Mapped[str] = mapped_column(String(50), nullable=False)
    reviewed: Mapped[bool] = mapped_column(Boolean, default=False)
    reviewer_id: Mapped[str] = mapped_column(String(64), default="")
    review_verdict: Mapped[str] = mapped_column(String(20), default="")
    corrected_response: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    __table_args__ = (
        Index("idx_reviews_reviewed", "reviewed"),
        Index("idx_reviews_created", "created_at"),
    )


# =============================================================================
# Analytics
# =============================================================================


class AnalyticsSnapshot(Base):
    __tablename__ = "analytics_snapshots"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    avatar_id: Mapped[str] = mapped_column(String(64), nullable=False)
    period: Mapped[str] = mapped_column(String(20), nullable=False)
    period_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    total_conversations: Mapped[int] = mapped_column(Integer, default=0)
    total_messages: Mapped[int] = mapped_column(Integer, default=0)
    avg_response_time_ms: Mapped[float] = mapped_column(Float, default=0.0)
    avg_kb_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    escalation_rate: Mapped[float] = mapped_column(Float, default=0.0)
    autonomy_percent: Mapped[float] = mapped_column(Float, default=0.0)
    resolution_time_avg_s: Mapped[float] = mapped_column(Float, default=0.0)
    accuracy_score: Mapped[float] = mapped_column(Float, default=0.0)
    total_cost: Mapped[float] = mapped_column(Float, default=0.0)
    unique_users: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    __table_args__ = (
        Index("idx_snapshots_avatar_period", "avatar_id", "period", "period_date"),
    )


# =============================================================================
# AI Optimization Agent
# =============================================================================


class AgentIncident(Base):
    __tablename__ = "agent_incidents"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    rule_id: Mapped[str] = mapped_column(String(100), nullable=False)
    severity: Mapped[str] = mapped_column(String(20), nullable=False)  # info, warning, critical
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    details: Mapped[str] = mapped_column(Text, default="{}")  # JSON
    recommendation: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(20), default="open")  # open, acknowledged, resolved, auto_fixed
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    actions: Mapped[list["AgentAction"]] = relationship(back_populates="incident")

    __table_args__ = (
        Index("idx_agent_incidents_status", "status"),
        Index("idx_agent_incidents_rule", "rule_id"),
        Index("idx_agent_incidents_created", "created_at"),
    )


class AgentAction(Base):
    __tablename__ = "agent_actions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    incident_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("agent_incidents.id"), nullable=False
    )
    action_type: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    result: Mapped[str] = mapped_column(Text, default="{}")  # JSON
    auto: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    incident: Mapped["AgentIncident"] = relationship(back_populates="actions")

    __table_args__ = (
        Index("idx_agent_actions_incident", "incident_id"),
    )


class AgentApproval(Base):
    __tablename__ = "agent_approvals"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    action_type: Mapped[str] = mapped_column(String(100), nullable=False)  # suspend_customer, kill_switch, plan_downgrade, data_deletion
    target_id: Mapped[str] = mapped_column(String(100), nullable=False)  # customer_id or entity affected
    description: Mapped[str] = mapped_column(Text, default="")
    details: Mapped[str] = mapped_column(Text, default="{}")  # JSON
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending, approved, rejected, expired
    requested_by: Mapped[str] = mapped_column(String(100), default="agent")  # agent or rule_id
    reviewed_by: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    __table_args__ = (
        Index("idx_agent_approvals_status", "status"),
        Index("idx_agent_approvals_created", "created_at"),
    )


class AgentPattern(Base):
    __tablename__ = "agent_patterns"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    rule_id: Mapped[str] = mapped_column(String(100), nullable=False)
    pattern_key: Mapped[str] = mapped_column(String(255), nullable=False)
    occurrence_count: Mapped[int] = mapped_column(Integer, default=1)
    first_seen: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    last_seen: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    predicted_next: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    __table_args__ = (
        Index("idx_agent_patterns_rule_key", "rule_id", "pattern_key", unique=True),
    )


# =============================================================================
# Phase 2: Employee & Knowledge
# =============================================================================


class Employee(Base):
    """Digital employee definition — personality, role, guardrails."""
    __tablename__ = "employees"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    customer_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("customers.id"), nullable=False
    )
    avatar_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("avatars.id"), nullable=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    role_title: Mapped[str] = mapped_column(String(255), default="")
    role_description: Mapped[str] = mapped_column(Text, default="")
    personality: Mapped[str] = mapped_column(Text, default="{}")  # JSON
    guardrails: Mapped[str] = mapped_column(Text, default="{}")  # JSON
    language: Mapped[str] = mapped_column(String(10), default="ar")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    knowledge: Mapped[list["EmployeeKnowledge"]] = relationship(back_populates="employee")
    learning: Mapped[list["EmployeeLearning"]] = relationship(back_populates="employee")
    tools: Mapped[list["EmployeeTools"]] = relationship(back_populates="employee")

    __table_args__ = (
        Index("idx_employees_customer", "customer_id"),
        Index("idx_employees_avatar", "avatar_id"),
    )


class EmployeeKnowledge(Base):
    """Knowledge-base Q&A entry for an employee."""
    __tablename__ = "employee_knowledge"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    employee_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("employees.id"), nullable=False
    )
    category: Mapped[str] = mapped_column(String(128), default="general")
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    approved: Mapped[bool] = mapped_column(Boolean, default=False)
    times_used: Mapped[int] = mapped_column(Integer, default=0)
    success_rate: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    employee: Mapped["Employee"] = relationship(back_populates="knowledge")

    __table_args__ = (
        Index("idx_ek_employee", "employee_id"),
        Index("idx_ek_category", "category"),
    )


class EmployeeLearning(Base):
    """Learning queue entry — pending human review or auto-approved."""
    __tablename__ = "employee_learning"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    employee_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("employees.id"), nullable=False
    )
    customer_id: Mapped[str] = mapped_column(String(64), nullable=False)
    session_id: Mapped[str] = mapped_column(String(64), default="")
    learning_type: Mapped[str] = mapped_column(String(50), nullable=False)  # qa_pair, preference, correction
    old_value: Mapped[str] = mapped_column(Text, default="")
    new_value: Mapped[str] = mapped_column(Text, default="")
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending, approved, rejected
    source: Mapped[str] = mapped_column(String(50), default="auto")  # auto, operator, visitor
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    employee: Mapped["Employee"] = relationship(back_populates="learning")

    __table_args__ = (
        Index("idx_el_employee", "employee_id"),
        Index("idx_el_status", "status"),
        Index("idx_el_created", "created_at"),
    )


# =============================================================================
# Phase 2: Tool Registry & Assignment
# =============================================================================


class ToolRegistry(Base):
    """Custom API tool definition."""
    __tablename__ = "tool_registry"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    customer_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("customers.id"), nullable=False
    )
    tool_id: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)  # slug
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    category: Mapped[str] = mapped_column(String(64), default="custom")
    input_schema: Mapped[str] = mapped_column(Text, default="{}")  # JSON
    api_url: Mapped[str] = mapped_column(Text, default="")
    api_method: Mapped[str] = mapped_column(String(10), default="POST")  # GET, POST, PUT, DELETE
    api_headers: Mapped[str] = mapped_column(Text, default="{}")  # JSON
    api_body_template: Mapped[str] = mapped_column(Text, default="{}")  # JSON with {{param}} placeholders
    response_mapping: Mapped[str] = mapped_column(Text, default="{}")  # JSON
    timeout_ms: Mapped[int] = mapped_column(Integer, default=10000)
    requires_confirmation: Mapped[bool] = mapped_column(Boolean, default=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    employees: Mapped[list["EmployeeTools"]] = relationship(back_populates="tool")

    __table_args__ = (
        Index("idx_tr_customer", "customer_id"),
        Index("idx_tr_tool_id", "tool_id"),
    )


class EmployeeTools(Base):
    """N:M assignment of tools to employees."""
    __tablename__ = "employee_tools"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    employee_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("employees.id"), nullable=False
    )
    tool_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("tool_registry.id"), nullable=False
    )
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    employee: Mapped["Employee"] = relationship(back_populates="tools")
    tool: Mapped["ToolRegistry"] = relationship(back_populates="employees")

    __table_args__ = (
        Index("idx_et_employee", "employee_id"),
        Index("idx_et_tool", "tool_id"),
        Index("idx_et_employee_tool", "employee_id", "tool_id", unique=True),
    )


# =============================================================================
# Phase 2: Visitor Profile & Memory
# =============================================================================


class VisitorProfile(Base):
    """Visitor identity and metadata."""
    __tablename__ = "visitor_profiles"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    visitor_id: Mapped[str] = mapped_column(String(128), nullable=False)
    employee_id: Mapped[str] = mapped_column(String(64), default="")
    customer_id: Mapped[str] = mapped_column(String(64), default="")
    display_name: Mapped[str] = mapped_column(String(255), default="")
    email: Mapped[str] = mapped_column(String(255), default="")
    phone: Mapped[str] = mapped_column(String(50), default="")
    language: Mapped[str] = mapped_column(String(10), default="")
    tags: Mapped[str] = mapped_column(Text, default="[]")  # JSON array
    interaction_count: Mapped[int] = mapped_column(Integer, default=0)
    last_seen: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    memories: Mapped[list["VisitorMemory"]] = relationship(back_populates="profile")

    __table_args__ = (
        Index("idx_vp_visitor", "visitor_id"),
        Index("idx_vp_customer", "customer_id"),
        Index("idx_vp_visitor_employee", "visitor_id", "employee_id"),
    )


class VisitorMemory(Base):
    """Per-visitor contextual memory entries."""
    __tablename__ = "visitor_memories"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    visitor_id: Mapped[str] = mapped_column(String(128), nullable=False)
    profile_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("visitor_profiles.id"), nullable=True
    )
    employee_id: Mapped[str] = mapped_column(String(64), default="")
    memory_type: Mapped[str] = mapped_column(String(50), nullable=False)  # preference, fact, intent
    content: Mapped[str] = mapped_column(Text, nullable=False)
    source_session: Mapped[str] = mapped_column(String(64), default="")
    importance: Mapped[float] = mapped_column(Float, default=0.5)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    profile: Mapped["VisitorProfile"] = relationship(back_populates="memories")

    __table_args__ = (
        Index("idx_vm_visitor", "visitor_id"),
        Index("idx_vm_employee", "employee_id"),
        Index("idx_vm_visitor_employee", "visitor_id", "employee_id"),
    )


# =============================================================================
# Phase 2: Tool Execution Log
# =============================================================================


class ToolExecutionLog(Base):
    """Audit trail for tool calls."""
    __tablename__ = "tool_execution_log"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    tool_id: Mapped[str] = mapped_column(String(128), nullable=False)
    employee_id: Mapped[str] = mapped_column(String(64), default="")
    session_id: Mapped[str] = mapped_column(String(64), default="")
    visitor_id: Mapped[str] = mapped_column(String(128), default="")
    input_data: Mapped[str] = mapped_column(Text, default="{}")  # JSON
    output_data: Mapped[str] = mapped_column(Text, default="{}")  # JSON
    status: Mapped[str] = mapped_column(String(20), default="success")  # success, error, timeout
    error_message: Mapped[str] = mapped_column(Text, default="")
    execution_time_ms: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    __table_args__ = (
        Index("idx_tel_tool", "tool_id"),
        Index("idx_tel_session", "session_id"),
        Index("idx_tel_employee", "employee_id"),
        Index("idx_tel_created", "created_at"),
    )


# =============================================================================
# Phase 2: Workflows
# =============================================================================


class Workflow(Base):
    """Workflow definition with step sequence."""
    __tablename__ = "workflows"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    customer_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("customers.id"), nullable=False
    )
    employee_id: Mapped[str] = mapped_column(String(64), default="")
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    trigger_type: Mapped[str] = mapped_column(String(50), default="manual")  # manual, keyword, intent, tool_result
    trigger_config: Mapped[str] = mapped_column(Text, default="{}")  # JSON
    steps: Mapped[str] = mapped_column(Text, default="[]")  # JSON array of step definitions
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    template_id: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    executions: Mapped[list["WorkflowExecution"]] = relationship(back_populates="workflow")

    __table_args__ = (
        Index("idx_wf_customer", "customer_id"),
        Index("idx_wf_employee", "employee_id"),
        Index("idx_wf_active", "is_active"),
    )


class WorkflowExecution(Base):
    """Single run of a workflow."""
    __tablename__ = "workflow_executions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    workflow_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("workflows.id"), nullable=False
    )
    session_id: Mapped[str] = mapped_column(String(64), default="")
    visitor_id: Mapped[str] = mapped_column(String(128), default="")
    status: Mapped[str] = mapped_column(String(20), default="running")  # running, waiting, completed, failed
    current_step: Mapped[int] = mapped_column(Integer, default=0)
    context: Mapped[str] = mapped_column(Text, default="{}")  # JSON — accumulated step results
    error: Mapped[str] = mapped_column(Text, default="")
    started_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    workflow: Mapped["Workflow"] = relationship(back_populates="executions")

    __table_args__ = (
        Index("idx_we_workflow", "workflow_id"),
        Index("idx_we_session", "session_id"),
        Index("idx_we_status", "status"),
    )


# =============================================================================
# Phase 2: API Cost Tracking
# =============================================================================


class APICostRecord(Base):
    """Per-call cost tracking by service (ASR, LLM, TTS, RunPod)."""
    __tablename__ = "api_cost_records"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    service: Mapped[str] = mapped_column(String(20), nullable=False)  # asr, llm, tts, runpod
    customer_id: Mapped[str] = mapped_column(String(64), default="")
    session_id: Mapped[str] = mapped_column(String(64), default="")
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    tokens_used: Mapped[int] = mapped_column(Integer, default=0)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    details: Mapped[str] = mapped_column(Text, default="{}")  # JSON
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    __table_args__ = (
        Index("idx_acr_service", "service"),
        Index("idx_acr_customer", "customer_id"),
        Index("idx_acr_session", "session_id"),
        Index("idx_acr_created", "created_at"),
    )


# =============================================================================
# Phase 3: Cross-Learning & Industry Categories
# =============================================================================


class IndustryCategory(Base):
    """Industry category with shared knowledge templates."""
    __tablename__ = "industry_categories"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    slug: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    default_qa_pairs: Mapped[str] = mapped_column(Text, default="[]")  # JSON list of {q, a}
    default_personality: Mapped[str] = mapped_column(Text, default="{}")  # JSON
    employee_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    __table_args__ = (
        Index("idx_ic_slug", "slug"),
    )


class EmployeeIndustry(Base):
    """N:M mapping between employees and industries."""
    __tablename__ = "employee_industries"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    employee_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("employees.id"), nullable=False
    )
    industry_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("industry_categories.id"), nullable=False
    )
    adopted_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    __table_args__ = (
        Index("idx_ei_employee", "employee_id"),
        Index("idx_ei_industry", "industry_id"),
    )
