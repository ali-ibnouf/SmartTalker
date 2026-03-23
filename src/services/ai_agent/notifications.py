"""Notification dispatcher for the AI Optimization Agent.

Routes detections to three channels based on severity:
- Audit log (all severities)
- WebSocket dashboard push (critical immediately, warning batched 5min)
- Email (critical only, via aiosmtplib)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from email.message import EmailMessage
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

from src.services.ai_agent.rules import Detection
from src.utils.async_utils import background_task_error_handler
from src.utils.logger import SmartTalkerJsonFormatter, setup_logger

logger = setup_logger("ai_agent.notifications")


class NotificationSettings(BaseSettings):
    """SMTP configuration for email alerts."""

    model_config = SettingsConfigDict(env_prefix="SMTP_", env_file=".env", extra="ignore")

    host: str = "smtp.gmail.com"
    port: int = 587
    username: str = ""
    password: str = ""
    use_tls: bool = True
    from_email: str = "agent@lsmarttech.com"
    from_name: str = "SmartTalker AI Agent"


def _setup_audit_logger(path: str) -> logging.Logger:
    """Create a dedicated file-only logger for audit trail.

    If the logger already has handlers pointing to a different path,
    they are replaced (supports reconfiguration in tests).
    """
    audit = logging.getLogger("ai_agent.audit")
    resolved = str(Path(path).resolve())

    # Check if already configured with the same path
    if audit.handlers:
        existing_path = getattr(audit.handlers[0], "baseFilename", None)
        if existing_path == resolved:
            return audit
        # Path changed — remove old handlers
        for h in audit.handlers[:]:
            h.close()
            audit.removeHandler(h)

    audit.setLevel(logging.INFO)
    audit.propagate = False

    # Ensure directory exists
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    handler = RotatingFileHandler(
        path,
        maxBytes=50 * 1024 * 1024,  # 50 MB
        backupCount=5,
    )
    handler.setFormatter(SmartTalkerJsonFormatter(
        fmt="%(timestamp)s %(level)s %(module)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    ))
    audit.addHandler(handler)
    return audit


class NotificationDispatcher:
    """Routes agent detections to WebSocket, email, and audit log channels."""

    def __init__(
        self,
        operator_manager: Any,
        agent_config: Any,
        smtp_config: NotificationSettings,
    ) -> None:
        self._operator_manager = operator_manager
        self._agent_config = agent_config
        self._smtp_config = smtp_config

        # Audit logger
        self._audit = _setup_audit_logger(agent_config.audit_log_path)

        # Batch queues
        self._warning_queue: asyncio.Queue[tuple[Detection, Optional[str]]] = asyncio.Queue()
        self._info_queue: asyncio.Queue[tuple[Detection, Optional[str]]] = asyncio.Queue()

        # Background flush tasks
        self._warning_task: Optional[asyncio.Task[None]] = None
        self._info_task: Optional[asyncio.Task[None]] = None

        # Alert cooldown — last dispatch time per rule_id
        self._last_alert_times: dict[str, float] = {}

    async def start(self) -> None:
        """Start background batch flush loops."""
        self._warning_task = asyncio.create_task(self._warning_flush_loop())
        self._warning_task.add_done_callback(background_task_error_handler)
        self._info_task = asyncio.create_task(self._info_flush_loop())
        self._info_task.add_done_callback(background_task_error_handler)
        logger.info("NotificationDispatcher started")

    async def stop(self) -> None:
        """Drain remaining items and cancel flush tasks."""
        # Final drain
        await self._flush_warning_queue()
        await self._flush_info_queue()

        for task in (self._warning_task, self._info_task):
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._warning_task = None
        self._info_task = None
        logger.info("NotificationDispatcher stopped")

    # ── Public API ────────────────────────────────────────────────────────

    async def dispatch(self, d: Detection, incident_id: Optional[str]) -> None:
        """Route a detection to the appropriate channels by severity.

        Suppresses duplicate alerts for the same rule_id within the
        configured cooldown window (audit log still written).
        """
        # Check cooldown — suppress WS/email but always write audit
        if self._is_cooled_down(d.rule_id):
            self._write_audit_log(d, incident_id, "suppressed")
            logger.debug(
                f"Alert suppressed (cooldown): {d.rule_id}",
                extra={"rule_id": d.rule_id},
            )
            return

        # Record this dispatch time
        self._last_alert_times[d.rule_id] = time.time()

        # Always write audit log
        self._write_audit_log(d, incident_id, "open")

        if d.severity == "critical":
            await self._send_ws_immediate(d, incident_id)
            await self._send_email(d, incident_id)
        elif d.severity == "warning":
            await self._warning_queue.put((d, incident_id))
        elif d.severity == "info":
            await self._info_queue.put((d, incident_id))

    def _is_cooled_down(self, rule_id: str) -> bool:
        """Return True if the same rule_id was dispatched within its cooldown window."""
        last = self._last_alert_times.get(rule_id)
        if last is None:
            return False

        cooldown_map: dict[str, int] = getattr(
            self._agent_config, "alert_cooldown", {}
        )
        default_cd: int = getattr(
            self._agent_config, "alert_cooldown_default_s", 60
        )
        cooldown_s = cooldown_map.get(rule_id, default_cd)
        return (time.time() - last) < cooldown_s

    async def dispatch_status_change(
        self, incident_id: str, status: str
    ) -> None:
        """Log a status change (resolved, auto_fixed) to the audit trail."""
        self._audit.info(
            f"Incident status changed: {status}",
            extra={
                "event": "status_change",
                "incident_id": incident_id,
                "status": status,
                "timestamp_epoch": time.time(),
            },
        )

    # ── WebSocket ─────────────────────────────────────────────────────────

    async def _send_ws_immediate(
        self, d: Detection, incident_id: Optional[str]
    ) -> None:
        """Push a single alert to all authenticated operators."""
        mgr = self._operator_manager
        if mgr is None or not self._agent_config.notification_ws_enabled:
            return

        msg = {
            "type": "agent_alert",
            "severity": d.severity,
            "incident": {
                "id": incident_id,
                "rule_id": d.rule_id,
                "title": d.title,
                "description": d.description,
                "recommendation": d.recommendation,
                "details": d.details,
                "auto_fixable": d.auto_fixable,
            },
            "timestamp": time.time(),
        }
        await self._broadcast_to_operators(msg)

    async def _send_ws_batch(
        self, items: list[tuple[Detection, Optional[str]]], severity: str
    ) -> None:
        """Push a batched alert to all authenticated operators."""
        mgr = self._operator_manager
        if mgr is None or not items or not self._agent_config.notification_ws_enabled:
            return

        msg = {
            "type": "agent_alert_batch",
            "severity": severity,
            "incidents": [
                {
                    "id": iid,
                    "rule_id": d.rule_id,
                    "title": d.title,
                    "description": d.description,
                    "recommendation": d.recommendation,
                    "auto_fixable": d.auto_fixable,
                }
                for d, iid in items
            ],
            "count": len(items),
            "timestamp": time.time(),
        }
        await self._broadcast_to_operators(msg)

    async def _broadcast_to_operators(self, msg: dict[str, Any]) -> None:
        """Send a JSON message to all authenticated operators."""
        mgr = self._operator_manager
        if mgr is None:
            return

        for operator in mgr._operators.values():
            if operator.authenticated:
                try:
                    await mgr._send_json(operator.websocket, msg)
                except Exception:
                    pass

    # ── Email ─────────────────────────────────────────────────────────────

    async def _send_email(
        self, d: Detection, incident_id: Optional[str]
    ) -> None:
        """Send an email alert for critical detections."""
        if not self._agent_config.notification_email_enabled:
            return
        if not self._smtp_config.username:
            logger.debug("SMTP username not configured, skipping email")
            return

        try:
            import aiosmtplib
        except ImportError:
            logger.warning("aiosmtplib not installed, skipping email notification")
            return

        subject = f"[CRITICAL] {d.title}"
        body = self._build_email_html(d, incident_id)

        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = f"{self._smtp_config.from_name} <{self._smtp_config.from_email}>"
        msg["To"] = self._agent_config.notification_email_to
        msg.set_content(f"CRITICAL: {d.title}\n\n{d.description}\n\nRecommendation: {d.recommendation}")
        msg.add_alternative(body, subtype="html")

        try:
            await aiosmtplib.send(
                msg,
                hostname=self._smtp_config.host,
                port=self._smtp_config.port,
                username=self._smtp_config.username,
                password=self._smtp_config.password,
                start_tls=self._smtp_config.use_tls,
            )
            logger.info(
                f"Critical alert email sent for {d.rule_id}",
                extra={"incident_id": incident_id},
            )
        except Exception as exc:
            logger.error(f"Failed to send alert email: {exc}")

    def _build_email_html(
        self, d: Detection, incident_id: Optional[str]
    ) -> str:
        """Build an HTML email body for a critical detection."""
        details_rows = "".join(
            f"<tr><td style='padding:4px 8px;font-weight:bold'>{k}</td>"
            f"<td style='padding:4px 8px'>{v}</td></tr>"
            for k, v in d.details.items()
        )
        return f"""<html><body style="font-family:sans-serif;background:#1a1a2e;color:#e0e0e0;padding:20px">
<div style="max-width:600px;margin:0 auto">
<div style="background:#dc2626;color:white;padding:12px 20px;border-radius:8px 8px 0 0;font-size:18px">
  CRITICAL ALERT
</div>
<div style="background:#16213e;padding:20px;border-radius:0 0 8px 8px">
  <h2 style="margin:0 0 12px;color:#ff6b6b">{d.title}</h2>
  <p style="color:#ccc">{d.description}</p>
  <table style="width:100%;border-collapse:collapse;margin:16px 0">
    <tr><td style="padding:4px 8px;font-weight:bold">Rule</td><td style="padding:4px 8px">{d.rule_id}</td></tr>
    <tr><td style="padding:4px 8px;font-weight:bold">Incident ID</td><td style="padding:4px 8px">{incident_id or 'N/A'}</td></tr>
    {details_rows}
  </table>
  <div style="background:#0f3460;padding:12px;border-radius:4px;margin-top:12px">
    <strong>Recommendation:</strong> {d.recommendation}
  </div>
  <p style="color:#888;font-size:12px;margin-top:20px">
    SmartTalker AI Optimization Agent &mdash; auto-generated alert
  </p>
</div>
</div>
</body></html>"""

    # ── Audit Log ─────────────────────────────────────────────────────────

    def _write_audit_log(
        self,
        d: Detection,
        incident_id: Optional[str],
        status: str,
    ) -> None:
        """Write a structured JSON line to the audit log."""
        self._audit.info(
            d.title,
            extra={
                "event": "detection",
                "incident_id": incident_id,
                "rule_id": d.rule_id,
                "severity": d.severity,
                "status": status,
                "description": d.description,
                "recommendation": d.recommendation,
                "details": json.dumps(d.details),
                "auto_fixable": d.auto_fixable,
                "timestamp_epoch": time.time(),
            },
        )

    # ── Batch Flush Loops ─────────────────────────────────────────────────

    async def _warning_flush_loop(self) -> None:
        """Drain warning queue every N seconds and send batched WS alert."""
        interval = self._agent_config.notification_warning_batch_s
        while True:
            await asyncio.sleep(interval)
            await self._flush_warning_queue()

    async def _info_flush_loop(self) -> None:
        """Drain info queue every N seconds and send digest WS alert."""
        interval = self._agent_config.notification_info_digest_s
        while True:
            await asyncio.sleep(interval)
            await self._flush_info_queue()

    async def _flush_warning_queue(self) -> None:
        """Drain all pending warning items and send as batch."""
        items: list[tuple[Detection, Optional[str]]] = []
        while not self._warning_queue.empty():
            try:
                items.append(self._warning_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        if items:
            await self._send_ws_batch(items, "warning")

    async def _flush_info_queue(self) -> None:
        """Drain all pending info items and send as digest."""
        items: list[tuple[Detection, Optional[str]]] = []
        while not self._info_queue.empty():
            try:
                items.append(self._info_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        if items:
            await self._send_ws_batch(items, "info")
