"""Prometheus metrics definition for SmartTalker."""

from prometheus_client import Counter, Gauge, Histogram

# Labels common to inference metrics
INFERENCE_LABELS = ["model_type"]

# ── Inference Metrics ────────────────────────────────────────────────────────

INFERENCE_LATENCY = Histogram(
    "smarttalker_inference_duration_seconds",
    "Time spent running inference (seconds)",
    labelnames=INFERENCE_LABELS,
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

# ── Business Metrics ─────────────────────────────────────────────────────────

ACTIVE_SESSIONS = Gauge(
    "smarttalker_active_sessions",
    "Number of active user sessions in memory",
)

# ── Training & Analytics Metrics ─────────────────────────────────────────────

ESCALATION_TOTAL = Counter(
    "smarttalker_escalations_total",
    "Total escalation events",
    labelnames=["avatar_id", "reason"],
)

GUARDRAIL_VIOLATIONS = Counter(
    "smarttalker_guardrail_violations_total",
    "Total guardrail policy violations",
    labelnames=["avatar_id", "violation_type"],
)

KB_CONFIDENCE = Histogram(
    "smarttalker_kb_confidence",
    "Knowledge Base confidence scores",
    labelnames=["avatar_id"],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

AUTONOMY_RATE = Gauge(
    "smarttalker_autonomy_rate",
    "Current autonomy percentage (digital vs human handling)",
    labelnames=["avatar_id"],
)
