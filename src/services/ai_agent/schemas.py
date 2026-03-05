"""Pydantic schemas for AI Agent API endpoints."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class AgentStatsResponse(BaseModel):
    scan_count: int = 0
    last_scan_at: Optional[str] = None
    rules_count: int = 0
    running: bool = False
    incidents_total: int = 0
    open_incidents: int = 0
    auto_fixes_applied: int = 0
    patterns_tracked: int = 0


class IncidentItem(BaseModel):
    id: str
    rule_id: str
    severity: str
    title: str
    description: str = ""
    details: dict[str, Any] = {}
    recommendation: str = ""
    status: str = "open"
    created_at: Optional[str] = None
    resolved_at: Optional[str] = None


class IncidentListResponse(BaseModel):
    incidents: list[IncidentItem] = []
    total: int = 0


class PredictionItem(BaseModel):
    rule_id: str
    pattern_key: str
    occurrences: int = 0
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None
    predicted_next: Optional[str] = None


class PredictionListResponse(BaseModel):
    predictions: list[PredictionItem] = []


class DetectionItem(BaseModel):
    rule_id: str
    severity: str
    title: str
    description: str = ""
    recommendation: str = ""
    auto_fixable: bool = False


class ScanResponse(BaseModel):
    detections: list[DetectionItem] = []
    count: int = 0


class IncidentActionResponse(BaseModel):
    incident_id: str
    status: str
