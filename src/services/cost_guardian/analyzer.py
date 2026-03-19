"""Cost Analyzer — detects budget breaches, spikes, and anomalies."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from src.services.cost_guardian.config import (
    ANOMALY_CONFIG,
    BUDGETS,
    CUSTOMER_COST_RATIO,
    ServiceBudget,
)
from src.services.cost_guardian.monitor import CostMonitor
from src.utils.logger import setup_logger

logger = setup_logger("cost_guardian.analyzer")

# Plan prices for margin calculation
_PLAN_PRICES: dict[str, int] = {
    "starter": 100,
    "professional": 200,
    "business": 400,
    "enterprise": 800,
}


class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class CostAlert:
    """Represents a detected cost anomaly or budget breach."""

    def __init__(
        self,
        level: AlertLevel,
        service: str,
        message: str,
        current_value: float,
        threshold: float,
        action: Optional[str] = None,
        customer_id: Optional[str] = None,
    ) -> None:
        self.level = level
        self.service = service
        self.message = message
        self.current_value = current_value
        self.threshold = threshold
        self.action = action
        self.customer_id = customer_id
        self.timestamp = datetime.now(timezone.utc)

    def __repr__(self) -> str:
        return f"CostAlert({self.level.value}: {self.service} — {self.message})"


class CostAnalyzer:
    """Analyzes cost data and detects anomalies."""

    def __init__(self, monitor: CostMonitor) -> None:
        self.monitor = monitor

    async def run_full_analysis(self) -> list[CostAlert]:
        """Run all analysis checks. Returns list of alerts."""
        alerts: list[CostAlert] = []

        # 1. Budget checks (per service)
        for service_key, budget in BUDGETS.items():
            alerts.extend(await self._check_service_budget(service_key, budget))

        # 2. Spike detection
        for service_key in BUDGETS:
            spike_alert = await self._check_spike(service_key)
            if spike_alert:
                alerts.append(spike_alert)

        # 3. Per-request anomaly
        for service_key, budget in BUDGETS.items():
            req_alert = await self._check_per_request(service_key, budget)
            if req_alert:
                alerts.append(req_alert)

        # 4. Customer cost ratio
        alerts.extend(await self._check_customer_margins())

        # 5. Rapid fire detection
        alerts.extend(await self._check_rapid_fire())

        # 6. Stuck RunPod jobs
        alerts.extend(await self._check_stuck_jobs())

        # 7. Zero cost detection (billing broken?)
        alerts.extend(await self._check_zero_costs())

        return alerts

    async def _check_service_budget(
        self, service_key: str, budget: ServiceBudget
    ) -> list[CostAlert]:
        alerts: list[CostAlert] = []
        hourly = await self.monitor.get_hourly_spend(service_key)
        daily = await self.monitor.get_daily_spend(service_key)
        monthly = await self.monitor.get_monthly_spend(service_key)

        # Hourly
        if hourly >= budget.hourly_kill:
            alerts.append(CostAlert(
                AlertLevel.EMERGENCY, budget.service,
                f"EMERGENCY: {budget.service} hourly spend ${hourly:.2f} exceeded kill threshold ${budget.hourly_kill:.2f}",
                hourly, budget.hourly_kill,
                action="pause_service",
            ))
        elif hourly >= budget.hourly_limit:
            alerts.append(CostAlert(
                AlertLevel.WARNING, budget.service,
                f"WARNING: {budget.service} hourly spend ${hourly:.2f} approaching limit ${budget.hourly_limit:.2f}",
                hourly, budget.hourly_limit,
            ))

        # Daily
        if daily >= budget.daily_kill:
            alerts.append(CostAlert(
                AlertLevel.EMERGENCY, budget.service,
                f"EMERGENCY: {budget.service} daily spend ${daily:.2f} exceeded kill threshold ${budget.daily_kill:.2f}",
                daily, budget.daily_kill,
                action="pause_service",
            ))
        elif daily >= budget.daily_limit:
            alerts.append(CostAlert(
                AlertLevel.WARNING, budget.service,
                f"WARNING: {budget.service} daily spend ${daily:.2f} approaching limit ${budget.daily_limit:.2f}",
                daily, budget.daily_limit,
            ))

        # Monthly
        if monthly >= budget.monthly_kill:
            alerts.append(CostAlert(
                AlertLevel.EMERGENCY, budget.service,
                f"EMERGENCY: {budget.service} monthly spend ${monthly:.2f} exceeded kill threshold ${budget.monthly_kill:.2f}",
                monthly, budget.monthly_kill,
                action="pause_all_sessions",
            ))
        elif monthly >= budget.monthly_limit:
            alerts.append(CostAlert(
                AlertLevel.CRITICAL, budget.service,
                f"CRITICAL: {budget.service} monthly spend ${monthly:.2f} approaching limit ${budget.monthly_limit:.2f}",
                monthly, budget.monthly_limit,
            ))

        return alerts

    async def _check_spike(self, service_key: str) -> Optional[CostAlert]:
        """Detect sudden spending spikes vs historical average."""
        current_hourly = await self.monitor.get_hourly_spend(service_key)
        avg_hourly = await self.monitor.get_average_hourly_spend(
            service_key, int(ANOMALY_CONFIG["lookback_hours"]),
        )

        spike_mult = float(ANOMALY_CONFIG["spike_multiplier"])
        if avg_hourly > 0 and current_hourly > avg_hourly * spike_mult:
            return CostAlert(
                AlertLevel.CRITICAL, BUDGETS[service_key].service,
                f"SPIKE: {BUDGETS[service_key].service} current hour ${current_hourly:.2f} "
                f"is {current_hourly / avg_hourly:.1f}x the 24h average ${avg_hourly:.2f}/hr",
                current_hourly, avg_hourly * spike_mult,
                action="investigate",
            )
        return None

    async def _check_per_request(
        self, service_key: str, budget: ServiceBudget
    ) -> Optional[CostAlert]:
        """Check if the most recent request cost is abnormally high."""
        last_cost = await self.monitor.get_last_request_cost(service_key)
        if last_cost is not None and last_cost > budget.per_request_max:
            return CostAlert(
                AlertLevel.WARNING, budget.service,
                f"ANOMALY: Single {budget.service} request cost ${last_cost:.4f} "
                f"exceeds max ${budget.per_request_max:.2f}",
                last_cost, budget.per_request_max,
            )
        return None

    async def _check_customer_margins(self) -> list[CostAlert]:
        """Check if any customer's API cost exceeds safe ratio of their revenue."""
        alerts: list[CostAlert] = []
        top_customers = await self.monitor.get_top_spending_customers(20)

        for customer in top_customers:
            revenue = _PLAN_PRICES.get(customer["plan_tier"], 100)
            cost = customer["total_cost"]
            ratio = cost / revenue if revenue > 0 else 0

            if ratio >= CUSTOMER_COST_RATIO["kill"]:
                alerts.append(CostAlert(
                    AlertLevel.EMERGENCY, "customer_margin",
                    f"MARGIN KILL: {customer['company_name'] or customer['customer_id']} "
                    f"cost ${cost:.2f} = {ratio * 100:.0f}% of ${revenue}/mo revenue",
                    cost, revenue * CUSTOMER_COST_RATIO["kill"],
                    action="pause_customer",
                    customer_id=customer["customer_id"],
                ))
            elif ratio >= CUSTOMER_COST_RATIO["critical"]:
                alerts.append(CostAlert(
                    AlertLevel.CRITICAL, "customer_margin",
                    f"MARGIN CRITICAL: {customer['company_name'] or customer['customer_id']} "
                    f"cost ${cost:.2f} = {ratio * 100:.0f}% of ${revenue}/mo revenue",
                    cost, revenue * CUSTOMER_COST_RATIO["critical"],
                    customer_id=customer["customer_id"],
                ))
            elif ratio >= CUSTOMER_COST_RATIO["warning"]:
                alerts.append(CostAlert(
                    AlertLevel.WARNING, "customer_margin",
                    f"MARGIN WARNING: {customer['company_name'] or customer['customer_id']} "
                    f"cost ${cost:.2f} = {ratio * 100:.0f}% of ${revenue}/mo revenue",
                    cost, revenue * CUSTOMER_COST_RATIO["warning"],
                    customer_id=customer["customer_id"],
                ))

        return alerts

    async def _check_rapid_fire(self) -> list[CostAlert]:
        """Detect customers making too many API calls per minute."""
        alerts: list[CostAlert] = []
        threshold = int(ANOMALY_CONFIG["rapid_fire_threshold"])
        customer_ids = await self.monitor.get_recent_active_customer_ids(minutes=5)

        for cid in customer_ids:
            rate = await self.monitor.get_customer_request_rate(cid, minutes=1)
            if rate > threshold:
                alerts.append(CostAlert(
                    AlertLevel.CRITICAL, "rapid_fire",
                    f"RAPID FIRE: Customer {cid} making {rate} API calls/minute",
                    float(rate), float(threshold),
                    action="throttle_customer",
                    customer_id=cid,
                ))

        return alerts

    async def _check_stuck_jobs(self) -> list[CostAlert]:
        """Detect RunPod jobs that started but never completed."""
        alerts: list[CostAlert] = []
        active_jobs = await self.monitor.get_runpod_active_jobs()

        now = datetime.now(timezone.utc)
        for job in active_jobs:
            created = job["created_at"]
            # Handle naive datetimes from SQLite/test
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            age_minutes = (now - created).total_seconds() / 60
            if age_minutes > 10:
                alerts.append(CostAlert(
                    AlertLevel.CRITICAL, "RunPod stuck job",
                    f"STUCK JOB: RunPod record {job['id']} running for {age_minutes:.0f} minutes",
                    age_minutes, 10.0,
                    action="cancel_runpod_job",
                ))

        return alerts

    async def _check_zero_costs(self) -> list[CostAlert]:
        """Detect if API calls are logging $0 cost (billing might be broken)."""
        if not ANOMALY_CONFIG["zero_cost_alert"]:
            return []

        alerts: list[CostAlert] = []
        rows = await self.monitor.get_zero_cost_counts()

        for row in rows:
            alerts.append(CostAlert(
                AlertLevel.WARNING, row["service"],
                f"ZERO COST: {row['zero_count']} {row['service']} calls with $0 cost "
                f"in last hour. Billing broken?",
                float(row["zero_count"]), 5.0,
                action="investigate",
            ))

        return alerts
