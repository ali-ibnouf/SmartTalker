"""Configuration for the AI Optimization Agent."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentSettings(BaseSettings):
    """Thresholds and intervals for the AI Agent."""

    model_config = SettingsConfigDict(env_prefix="AGENT_", env_file=".env", extra="ignore")

    # Master toggle
    agent_enabled: bool = True
    auto_fix_enabled: bool = True

    # Scan loop
    scan_interval_s: int = 60

    # System thresholds
    cpu_warn_pct: float = 85.0
    memory_warn_pct: float = 85.0
    disk_warn_pct: float = 90.0

    # GPU / render node thresholds
    fps_min: float = 20.0
    vram_warn_pct: float = 90.0
    heartbeat_timeout_s: int = 120

    # Business thresholds
    churn_days_inactive: int = 7
    quota_warn_pct: float = 90.0
    escalation_rate_warn: float = 0.20  # 20%

    # Security thresholds
    violation_spike_24h: int = 10
    rapid_session_threshold: int = 50  # sessions per minute

    # Notification
    notification_ws_enabled: bool = True
    notification_email_enabled: bool = True
    notification_email_to: str = "contact@lsmarttech.com"
    notification_warning_batch_s: int = 300   # 5 minutes
    notification_info_digest_s: int = 3600    # 1 hour
    audit_log_path: str = "./logs/agent_audit.jsonl"
