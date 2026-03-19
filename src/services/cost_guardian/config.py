"""Cost Guardian budget thresholds and anomaly detection configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ServiceBudget:
    """Budget limits per service per time window."""

    service: str
    hourly_limit: float       # max $/hour — triggers WARNING
    hourly_kill: float        # max $/hour — triggers AUTO-STOP
    daily_limit: float        # max $/day — triggers WARNING
    daily_kill: float         # max $/day — triggers AUTO-STOP
    monthly_limit: float      # max $/month — triggers WARNING
    monthly_kill: float       # max $/month — triggers AUTO-STOP
    per_request_max: float    # max $ per single API call — triggers alert


# Based on Maskki cost analysis (5,000 min = $130 total).
# Normal daily usage for a healthy platform:

BUDGETS: dict[str, ServiceBudget] = {
    "llm": ServiceBudget(
        service="DashScope LLM (qwen3-max)",
        hourly_limit=5.0,
        hourly_kill=20.0,
        daily_limit=30.0,
        daily_kill=100.0,
        monthly_limit=100.0,
        monthly_kill=500.0,
        per_request_max=0.50,
    ),
    "asr": ServiceBudget(
        service="DashScope ASR (qwen3-asr-flash)",
        hourly_limit=2.0,
        hourly_kill=10.0,
        daily_limit=10.0,
        daily_kill=50.0,
        monthly_limit=50.0,
        monthly_kill=200.0,
        per_request_max=0.10,
    ),
    "tts": ServiceBudget(
        service="DashScope TTS (qwen3-tts-vc)",
        hourly_limit=3.0,
        hourly_kill=15.0,
        daily_limit=20.0,
        daily_kill=80.0,
        monthly_limit=80.0,
        monthly_kill=300.0,
        per_request_max=0.20,
    ),
    "voice_clone": ServiceBudget(
        service="DashScope Voice Clone",
        hourly_limit=2.0,
        hourly_kill=5.0,
        daily_limit=5.0,
        daily_kill=20.0,
        monthly_limit=20.0,
        monthly_kill=50.0,
        per_request_max=0.25,
    ),
    "gpu_render": ServiceBudget(
        service="RunPod GPU Render (MuseTalk)",
        hourly_limit=3.0,
        hourly_kill=15.0,
        daily_limit=15.0,
        daily_kill=60.0,
        monthly_limit=60.0,
        monthly_kill=250.0,
        per_request_max=0.50,
    ),
    "gpu_preprocess": ServiceBudget(
        service="RunPod GPU Preprocess (Face)",
        hourly_limit=1.0,
        hourly_kill=5.0,
        daily_limit=3.0,
        daily_kill=10.0,
        monthly_limit=10.0,
        monthly_kill=30.0,
        per_request_max=0.30,
    ),
}

# Per-customer limits (relative to their plan revenue)
CUSTOMER_COST_RATIO: dict[str, float] = {
    "warning": 0.50,     # customer API cost > 50% of monthly revenue -> warning
    "critical": 0.70,    # > 70% -> critical alert
    "kill": 0.90,        # > 90% -> auto-pause customer sessions (losing money)
}

# Anomaly detection
ANOMALY_CONFIG: dict[str, object] = {
    "spike_multiplier": 5.0,       # 5x normal rate = spike
    "lookback_hours": 24,          # compare against last 24h average
    "min_data_points": 10,         # need at least 10 data points to detect anomaly
    "rapid_fire_threshold": 20,    # > 20 API calls/minute from one customer = rapid fire
    "zero_cost_alert": True,       # alert if API calls return cost=0 (billing might be broken)
}
