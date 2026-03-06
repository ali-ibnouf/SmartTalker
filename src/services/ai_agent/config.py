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

    # GPU / render node thresholds (legacy — kept for unregistered gpu.py rules)
    fps_min: float = 20.0
    vram_warn_pct: float = 90.0

    # Business thresholds
    churn_days_inactive: int = 7
    quota_warn_pct: float = 90.0
    escalation_rate_warn: float = 0.20  # 20%

    # Security thresholds
    violation_spike_24h: int = 10
    rapid_session_threshold: int = 50  # sessions per minute
    failed_auth_threshold: int = 5
    failed_auth_window_min: int = 10
    api_spike_multiplier: float = 3.0

    # Infrastructure thresholds
    pg_connections_warn_pct: float = 80.0
    redis_memory_warn_pct: float = 75.0
    dashscope_latency_warn_s: float = 3.0
    asr_latency_warn_s: float = 0.5  # 500ms
    tts_latency_warn_s: float = 2.0  # 2x realtime (~1s speech = 2s max)
    asr_connection_fail_threshold: int = 3  # failures in 5min
    tts_connection_fail_threshold: int = 3  # failures in 5min
    dashscope_monthly_budget_usd: float = 500.0  # monthly DashScope budget
    dashscope_quota_warn_pct: float = 80.0  # alert at 80% of budget
    runpod_error_rate_warn: float = 0.05  # 5% error rate
    runpod_queue_depth_warn: int = 10  # pending jobs
    runpod_render_time_warn_s: float = 5.0  # warm render threshold
    runpod_render_time_cold_s: float = 15.0  # cold start threshold
    runpod_preprocess_time_warn_s: float = 30.0  # preprocess_face threshold
    runpod_lipsync_warm_warn_s: float = 5.0  # render_lipsync warm threshold
    runpod_lipsync_cold_warn_s: float = 12.0  # render_lipsync cold threshold
    runpod_cold_start_pct_warn: float = 30.0  # alert if >30% of requests hit cold start
    runpod_min_idle_workers: int = 1  # minimum idle workers when busy
    runpod_active_customers_for_warm: int = 10  # require warm pool above N customers
    runpod_r2_latency_warn_s: float = 2.0  # R2 upload latency from worker
    runpod_timeout_rate_warn: float = 0.10  # 10% timeout rate

    # Business thresholds (extended)
    failed_payment_threshold: int = 2
    training_stall_days: int = 7
    onboarding_stuck_hours: int = 48

    # Resilience escalation thresholds
    runpod_consecutive_failure_threshold: int = 3  # disable video after N consecutive
    dashscope_consecutive_timeout_threshold: int = 5  # text-only mode after N consecutive
    dashscope_queue_depth_warn: int = 50  # alert if pending requests > N
    r2_downtime_threshold_s: float = 300.0  # escalate if R2 down > 5 minutes
    margin_squeeze_pct: float = 60.0  # alert if cost > N% of revenue

    # Alert cooldown — suppress duplicate alerts within these windows (seconds)
    alert_cooldown: dict[str, int] = {
        "dashscope_timeout": 300,       # 5 min
        "runpod_failure": 300,          # 5 min
        "margin_squeeze": 86400,        # 24h
        "r2_failure": 120,              # 2 min
        "low_balance": 3600,            # 1h
    }
    alert_cooldown_default_s: int = 60  # default for rule_ids not in the dict

    # Auto-fix
    fix_cooldown_s: int = 300  # 5 minutes between same fix type
    stale_session_timeout_min: int = 30
    stale_session_check_interval_s: int = 300  # 5 minutes
    quota_grace_hours: int = 2
    throttle_duration_s: int = 600
    throttle_rate_limit: int = 50

    # Approval queue
    approval_expiry_hours: int = 24

    # Notification
    notification_ws_enabled: bool = True
    notification_email_enabled: bool = True
    notification_email_to: str = "contact@lsmarttech.com"
    notification_warning_batch_s: int = 300   # 5 minutes
    notification_info_digest_s: int = 3600    # 1 hour
    audit_log_path: str = "./logs/agent_audit.jsonl"
