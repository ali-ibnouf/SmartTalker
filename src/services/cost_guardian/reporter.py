"""Cost Reporter — sends alert emails and daily cost reports via Resend."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from src.services.cost_guardian.analyzer import AlertLevel, CostAlert
from src.services.cost_guardian.monitor import CostMonitor
from src.utils.logger import setup_logger

logger = setup_logger("cost_guardian.reporter")


class CostReporter:
    """Generates and sends cost reports via Resend email API."""

    def __init__(self, resend_api_key: Optional[str], monitor: CostMonitor) -> None:
        self._api_key = resend_api_key
        self.monitor = monitor

    async def send_alert_email(self, alerts: list[CostAlert]) -> None:
        """Send email for WARNING/CRITICAL/EMERGENCY alerts."""
        if not alerts or not self._api_key:
            return

        emergencies = [a for a in alerts if a.level == AlertLevel.EMERGENCY]
        criticals = [a for a in alerts if a.level == AlertLevel.CRITICAL]
        warnings = [a for a in alerts if a.level == AlertLevel.WARNING]

        prefix = (
            "EMERGENCY" if emergencies
            else "CRITICAL" if criticals
            else "WARNING"
        )
        subject = f"{prefix}: Maskki Cost Alert — {len(alerts)} issue(s) detected"
        html = self._build_alert_email(emergencies, criticals, warnings)

        await self._send(subject, html)

    async def send_daily_report(self) -> None:
        """Send daily cost summary."""
        if not self._api_key:
            return

        spend = await self.monitor.get_total_monthly_spend()
        top_customers = await self.monitor.get_top_spending_customers(10)
        total = sum(s["cost"] for s in spend.values())

        html = self._build_daily_report(spend, top_customers, total)
        await self._send(f"Maskki Daily Cost Report — ${total:.2f} today", html)

    async def send_emergency_report(
        self, alert: CostAlert, action_result: dict[str, Any]
    ) -> None:
        """Immediate email when auto-action is taken."""
        if not self._api_key:
            return

        html = f"""
        <h2 style="color: red;">Emergency Cost Action Taken</h2>
        <table style="border-collapse: collapse; width: 100%;">
            <tr style="background: #fee;">
                <td style="padding: 10px; border: 1px solid #ddd;"><strong>Alert</strong></td>
                <td style="padding: 10px; border: 1px solid #ddd;">{alert.message}</td>
            </tr>
            <tr>
                <td style="padding: 10px; border: 1px solid #ddd;"><strong>Level</strong></td>
                <td style="padding: 10px; border: 1px solid #ddd;">{alert.level.value}</td>
            </tr>
            <tr>
                <td style="padding: 10px; border: 1px solid #ddd;"><strong>Service</strong></td>
                <td style="padding: 10px; border: 1px solid #ddd;">{alert.service}</td>
            </tr>
            <tr>
                <td style="padding: 10px; border: 1px solid #ddd;"><strong>Current Value</strong></td>
                <td style="padding: 10px; border: 1px solid #ddd;">${alert.current_value:.4f}</td>
            </tr>
            <tr>
                <td style="padding: 10px; border: 1px solid #ddd;"><strong>Threshold</strong></td>
                <td style="padding: 10px; border: 1px solid #ddd;">${alert.threshold:.4f}</td>
            </tr>
            <tr style="background: #ffe;">
                <td style="padding: 10px; border: 1px solid #ddd;"><strong>Action Taken</strong></td>
                <td style="padding: 10px; border: 1px solid #ddd;">{action_result}</td>
            </tr>
            <tr>
                <td style="padding: 10px; border: 1px solid #ddd;"><strong>Time</strong></td>
                <td style="padding: 10px; border: 1px solid #ddd;">{datetime.now(timezone.utc).replace(tzinfo=None).isoformat()}</td>
            </tr>
        </table>
        <p style="margin-top: 20px; color: #666;">
            To unpause, remove the Redis key or wait for auto-expiry.<br>
            Check dashboard: <a href="https://admin.maskki.com">admin.maskki.com</a>
        </p>
        """

        await self._send(
            f"EMERGENCY: {alert.service} auto-stopped — ${alert.current_value:.2f}",
            html,
        )

    # ── Email Builder ───────────────────────────────────────────────────

    def _build_alert_email(
        self,
        emergencies: list[CostAlert],
        criticals: list[CostAlert],
        warnings: list[CostAlert],
    ) -> str:
        html = "<h2>Maskki Cost Alert Report</h2>"

        if emergencies:
            html += '<h3 style="color: red;">EMERGENCY (Auto-Action Taken)</h3><ul>'
            for a in emergencies:
                html += f"<li><strong>{a.service}</strong>: {a.message}<br>Action: {a.action}</li>"
            html += "</ul>"

        if criticals:
            html += '<h3 style="color: orange;">CRITICAL</h3><ul>'
            for a in criticals:
                html += f"<li><strong>{a.service}</strong>: {a.message}</li>"
            html += "</ul>"

        if warnings:
            html += '<h3 style="color: #cc0;">WARNING</h3><ul>'
            for a in warnings:
                html += f"<li><strong>{a.service}</strong>: {a.message}</li>"
            html += "</ul>"

        html += f'<p style="color: #999;">Generated at {datetime.now(timezone.utc).replace(tzinfo=None).isoformat()} UTC</p>'
        return html

    def _build_daily_report(
        self,
        spend: dict[str, dict[str, Any]],
        top_customers: list[dict[str, Any]],
        total: float,
    ) -> str:
        html = f"""
        <h2>Maskki Daily Cost Report</h2>
        <h3>Total Month-to-Date: ${total:.2f}</h3>
        <h4>By Service:</h4>
        <table style="border-collapse: collapse;">
            <tr style="background: #f5f5f5;">
                <th style="padding: 8px; border: 1px solid #ddd;">Service</th>
                <th style="padding: 8px; border: 1px solid #ddd;">Cost</th>
                <th style="padding: 8px; border: 1px solid #ddd;">Calls</th>
            </tr>
        """
        for service, data in spend.items():
            html += f"""
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;">{service}</td>
                <td style="padding: 8px; border: 1px solid #ddd;">${data['cost']:.2f}</td>
                <td style="padding: 8px; border: 1px solid #ddd;">{data['calls']}</td>
            </tr>"""
        html += "</table>"

        html += """<h4>Top Spending Customers:</h4>
        <table style="border-collapse: collapse;">
            <tr style="background: #f5f5f5;">
                <th style="padding: 8px; border: 1px solid #ddd;">Customer</th>
                <th style="padding: 8px; border: 1px solid #ddd;">Plan</th>
                <th style="padding: 8px; border: 1px solid #ddd;">API Cost</th>
                <th style="padding: 8px; border: 1px solid #ddd;">Calls</th>
            </tr>"""
        for c in top_customers:
            html += f"""
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;">{c['company_name']}</td>
                <td style="padding: 8px; border: 1px solid #ddd;">{c['plan_tier']}</td>
                <td style="padding: 8px; border: 1px solid #ddd;">${c['total_cost']:.2f}</td>
                <td style="padding: 8px; border: 1px solid #ddd;">{c['total_calls']}</td>
            </tr>"""
        html += "</table>"
        return html

    # ── Low-Level Email ─────────────────────────────────────────────────

    async def _send(self, subject: str, html: str) -> None:
        """Send email via Resend API."""
        if not self._api_key:
            logger.debug(f"Email skipped (no API key): {subject}")
            return
        try:
            import resend
            resend.api_key = self._api_key
            resend.Emails.send({
                "from": "Maskki Cost Guardian <alerts@maskki.ai>",
                "to": ["contact@lsmarttech.com"],
                "subject": subject,
                "html": html,
            })
            logger.info(f"Cost alert email sent: {subject}")
        except Exception as e:
            logger.error(f"Failed to send cost alert email: {e}")
