"""Cost Guardian — real-time cost monitoring and auto-protection.

Monitors DashScope (LLM/ASR/TTS), RunPod GPU, and per-customer spend.
Detects anomalies, sends alerts, and can auto-pause services to prevent
runaway costs from damaging the business.
"""

from src.services.cost_guardian.guardian import CostGuardian

__all__ = ["CostGuardian"]
