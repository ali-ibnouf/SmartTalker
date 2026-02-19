"""Prometheus metrics definition for SmartTalker."""

from prometheus_client import Gauge, Histogram

# Labels common to inference metrics
INFERENCE_LABELS = ["model_type"]

# ── Inference Metrics ────────────────────────────────────────────────────────

INFERENCE_LATENCY = Histogram(
    "smarttalker_inference_duration_seconds",
    "Time spent running inference (seconds)",
    labelnames=INFERENCE_LABELS,
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

GPU_MEMORY_USAGE = Gauge(
    "smarttalker_gpu_memory_bytes",
    "GPU memory allocated (bytes)",
    labelnames=["device_index"],
)

GPU_QUEUE_DEPTH = Gauge(
    "smarttalker_gpu_queue_depth",
    "Number of tasks waiting for GPU execution",
)

# ── Business Metrics ─────────────────────────────────────────────────────────

ACTIVE_SESSIONS = Gauge(
    "smarttalker_active_sessions",
    "Number of active user sessions in memory",
)
