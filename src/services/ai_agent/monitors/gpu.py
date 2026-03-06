"""GPU render node monitoring rules."""

from __future__ import annotations

import time

from src.services.ai_agent.rules import AgentContext, Detection
from src.utils.logger import setup_logger

logger = setup_logger("ai_agent.monitors.gpu")


class NodeFPSDropRule:
    """Detects GPU nodes with FPS below minimum threshold."""

    rule_id = "gpu.fps_drop"

    async def evaluate(self, ctx: AgentContext) -> list[Detection]:
        detections: list[Detection] = []
        nm = ctx.node_manager
        if nm is None:
            return detections

        fps_min = ctx.agent_config.fps_min
        for node in nm.list_nodes():
            if node.status != "online":
                continue
            if node.current_fps < fps_min and node.current_fps > 0:
                severity = "critical" if node.current_fps < fps_min * 0.5 else "warning"
                detections.append(Detection(
                    rule_id=self.rule_id,
                    severity=severity,
                    title=f"Low FPS on {node.hostname}: {node.current_fps:.0f}",
                    description=(
                        f"Node {node.node_id} ({node.gpu_type}) running at "
                        f"{node.current_fps:.0f} FPS (min: {fps_min:.0f}). "
                        f"Active sessions: {node.active_sessions}."
                    ),
                    details={
                        "node_id": node.node_id,
                        "hostname": node.hostname,
                        "gpu_type": node.gpu_type,
                        "fps": node.current_fps,
                        "active_sessions": node.active_sessions,
                        "max_concurrent": node.max_concurrent,
                    },
                    recommendation=(
                        f"Reduce concurrent sessions from {node.active_sessions} "
                        f"or upgrade GPU from {node.gpu_type}."
                    ),
                    auto_fixable=True,
                ))
        return detections


class NodeVRAMRule:
    """Detects GPU nodes with high VRAM usage."""

    rule_id = "gpu.vram_high"

    async def evaluate(self, ctx: AgentContext) -> list[Detection]:
        detections: list[Detection] = []
        nm = ctx.node_manager
        if nm is None:
            return detections

        warn_pct = ctx.agent_config.vram_warn_pct
        for node in nm.list_nodes():
            if node.status != "online" or node.vram_mb == 0:
                continue
            vram_pct = (node.vram_used / node.vram_mb) * 100 if node.vram_mb > 0 else 0
            if vram_pct >= warn_pct:
                detections.append(Detection(
                    rule_id=self.rule_id,
                    severity="warning",
                    title=f"High VRAM on {node.hostname}: {vram_pct:.0f}%",
                    description=(
                        f"Node {node.node_id} VRAM at {node.vram_used}MB / {node.vram_mb}MB "
                        f"({vram_pct:.0f}%). OOM crash risk if new sessions start."
                    ),
                    details={
                        "node_id": node.node_id,
                        "vram_used_mb": node.vram_used,
                        "vram_total_mb": node.vram_mb,
                        "vram_pct": round(vram_pct, 1),
                    },
                    recommendation="Reduce avatars loaded or limit concurrent sessions.",
                ))
        return detections


