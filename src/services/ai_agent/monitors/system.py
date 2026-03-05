"""System resource monitoring rules (CPU, memory, disk)."""

from __future__ import annotations

from src.services.ai_agent.rules import AgentContext, Detection
from src.utils.logger import setup_logger

logger = setup_logger("ai_agent.monitors.system")


class HighCPURule:
    """Detects sustained high CPU usage."""

    rule_id = "system.high_cpu"

    async def evaluate(self, ctx: AgentContext) -> list[Detection]:
        try:
            import psutil
        except ImportError:
            return []

        cpu_pct = psutil.cpu_percent(interval=1)
        threshold = ctx.agent_config.cpu_warn_pct
        if cpu_pct >= threshold:
            severity = "critical" if cpu_pct >= 95 else "warning"
            return [Detection(
                rule_id=self.rule_id,
                severity=severity,
                title=f"High CPU usage: {cpu_pct:.0f}%",
                description=f"CPU is at {cpu_pct:.0f}% (threshold: {threshold:.0f}%). This may degrade response latency.",
                details={"cpu_pct": cpu_pct, "threshold": threshold},
                recommendation="Check for runaway processes. Consider scaling horizontally or reducing concurrent sessions.",
                auto_fixable=False,
            )]
        return []


class HighMemoryRule:
    """Detects high memory usage."""

    rule_id = "system.high_memory"

    async def evaluate(self, ctx: AgentContext) -> list[Detection]:
        try:
            import psutil
        except ImportError:
            return []

        mem = psutil.virtual_memory()
        threshold = ctx.agent_config.memory_warn_pct
        if mem.percent >= threshold:
            severity = "critical" if mem.percent >= 95 else "warning"
            return [Detection(
                rule_id=self.rule_id,
                severity=severity,
                title=f"High memory usage: {mem.percent:.0f}%",
                description=(
                    f"Memory at {mem.used // (1024**2)}MB / {mem.total // (1024**2)}MB "
                    f"({mem.percent:.0f}%). Risk of OOM if load increases."
                ),
                details={
                    "memory_pct": mem.percent,
                    "used_mb": mem.used // (1024**2),
                    "total_mb": mem.total // (1024**2),
                },
                recommendation="Clear Redis cache or restart unused services to free memory.",
                auto_fixable=True,
            )]
        return []


class DiskSpaceRule:
    """Detects low disk space."""

    rule_id = "system.disk_space"

    async def evaluate(self, ctx: AgentContext) -> list[Detection]:
        try:
            import psutil
        except ImportError:
            return []

        disk = psutil.disk_usage("/")
        threshold = ctx.agent_config.disk_warn_pct
        if disk.percent >= threshold:
            severity = "critical" if disk.percent >= 95 else "warning"
            free_gb = disk.free / (1024**3)
            return [Detection(
                rule_id=self.rule_id,
                severity=severity,
                title=f"Low disk space: {free_gb:.1f}GB free",
                description=(
                    f"Disk usage at {disk.percent:.0f}% ({free_gb:.1f}GB free). "
                    f"Audio files and logs may fill remaining space."
                ),
                details={
                    "disk_pct": disk.percent,
                    "free_gb": round(free_gb, 1),
                    "total_gb": round(disk.total / (1024**3), 1),
                },
                recommendation="Run storage cleanup to remove old audio files and log archives.",
                auto_fixable=False,
            )]
        return []
