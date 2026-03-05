"""Tests for the new analytical API routes (Analytics, Supervisor, Guardrails, Training)."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient

from src.main import app

# API Key for endpoints
API_KEY = "test_key_123"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

@pytest.fixture
def client(monkeypatch):
    """TestClient with mocked analytical engines."""
    # Ensure middleware auth accepts our test key
    monkeypatch.setenv("API_KEY", API_KEY)

    # Mock AnalyticsEngine
    mock_analytics = MagicMock()
    mock_analytics.is_loaded = True
    mock_kpis = MagicMock(
        total_conversations=100,
        total_messages=500,
        avg_response_time_ms=850.5,
        avg_kb_confidence=0.88,
        escalation_rate=0.05,
        autonomy_percent=95.0,
        resolution_time_avg_s=120.0,
        accuracy_score=0.92,
        total_cost=5.40,
    )
    mock_analytics.compute_kpis = AsyncMock(return_value=mock_kpis)
    
    mock_ts_point = MagicMock(date="2026-03-01", value=42.0)
    mock_analytics.get_timeseries = AsyncMock(return_value=[mock_ts_point])
    
    mock_drift = MagicMock(metric="response_time", baseline_value=800.0, current_value=950.0, change_percent=18.75, severity="warning")
    mock_analytics.check_drift = AsyncMock(return_value=[mock_drift])

    mock_report = {
        "kpis": mock_kpis,
        "skill_breakdown": [{"skill_id": "sales", "count": 10}],
        "daily_trends": {"msgs": [mock_ts_point]}
    }
    mock_analytics.export_report = AsyncMock(return_value=mock_report)
    
    mock_dashboard = {
        "kpis": mock_kpis,
        "trends": {"autonomy": [mock_ts_point]},
        "top_skills": [{"skill_id": "support", "count": 50}],
        "bottom_skills": [{"skill_id": "billing", "count": 2}]
    }
    mock_analytics.get_dashboard_data = AsyncMock(return_value=mock_dashboard)
    
    # Mock SupervisorEngine
    mock_supervisor = MagicMock()
    mock_supervisor.is_loaded = True
    mock_op_metric = MagicMock(
        operator_id="op_1", total_responses=50, avg_response_time_ms=5000, 
        escalations_resolved=10, corrections_made=5, sessions_handled=20, quality_score=0.95
    )
    mock_supervisor.list_operator_metrics = AsyncMock(return_value=[mock_op_metric])
    mock_supervisor.get_operator_metrics = AsyncMock(return_value=mock_op_metric)
    mock_supervisor.get_active_sessions_summary = AsyncMock(return_value=[
        {"session_id": "s1", "avatar_id": "a1", "operator_id": "op_1", "started_at": 1000.0, "message_count": 5}
    ])
    
    mock_review = MagicMock(
        id="r1", session_id="s1", avatar_id="a1", question="q", ai_response="a",
        confidence=0.4, flagged_reason="low_confidence", reviewed=False, reviewer_id="",
        review_verdict="", corrected_response="", created_at=1000.0
    )
    mock_supervisor.list_review_queue = AsyncMock(return_value=[mock_review])
    mock_supervisor.submit_review = AsyncMock(return_value=mock_review)
    
    mock_timeline_entry = MagicMock(
        timestamp=1000.0, operator_id="op_1", action_type="override",
        session_id="s1", avatar_id="a1", details={"action": "took over"},
    )
    mock_supervisor.get_activity_timeline = AsyncMock(return_value=[mock_timeline_entry])
    
    # Mock GuardrailsEngine
    mock_guardrails = MagicMock()
    mock_guardrails.is_loaded = True
    mock_policy = MagicMock(
        blocked_topics=["politics"], required_disclaimers=["I am AI"],
        max_response_length=500, escalation_keywords=["human"]
    )
    mock_guardrails.get_policy = AsyncMock(return_value=mock_policy)
    mock_guardrails.set_policy = AsyncMock()
    mock_guardrails.delete_policy = AsyncMock(return_value=True)
    
    mock_violation = MagicMock(
        id="v1", avatar_id="test_avatar", session_id="s1", violation_type="blocked",
        original_response="bad", sanitized_response="good", severity="high", created_at=1000.0
    )
    mock_guardrails.list_violations = AsyncMock(return_value=[mock_violation, mock_violation])

    # Mock Learning Analytics
    mock_learning = MagicMock()
    mock_learning.is_loaded = True
    mock_quality = MagicMock(
        total_qa=100, good_count=80, bad_count=10, none_count=10, correction_count=5,
        bad_ratio=0.1, improvement_rate=0.05, effective_threshold=0.75
    )
    mock_learning.get_skill_quality_stats = AsyncMock(return_value=mock_quality)
    
    mock_impr = MagicMock(date="2026-03-01", qa_added=5, good_count=4, bad_count=1, avg_confidence=0.8)
    mock_learning.get_improvement_timeline = AsyncMock(return_value=[mock_impr])
    
    mock_consolidation = MagicMock(date="2026-03-01", skills_updated=3)
    mock_learning.consolidate_daily = AsyncMock(return_value=mock_consolidation)
    
    mock_learning.export_qa_pairs = AsyncMock(return_value='{"q":"a"}\n{"q2":"a2"}')
    mock_learning.get_weak_areas = AsyncMock(return_value=[
        {"skill_id": "weak_1", "skill_name": "Math", "bad_ratio": 0.4, "correction_count": 8, "effective_threshold": 0.8}
    ])

    with TestClient(app) as test_client:
        app.state.analytics = mock_analytics
        app.state.supervisor = mock_supervisor
        app.state.guardrails = mock_guardrails
        app.state.learning_analytics = mock_learning
        yield test_client

# =============================================================================
# Analytics Routes
# =============================================================================

def test_get_kpis(client):
    response = client.get("/api/v1/analytics/test_avatar/kpis?period=weekly", headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert data["avatar_id"] == "test_avatar"
    assert data["total_conversations"] == 100
    assert data["autonomy_percent"] == 95.0

def test_get_timeseries(client):
    response = client.get("/api/v1/analytics/test_avatar/timeseries?metric=autonomy", headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert data["metric"] == "autonomy"
    assert len(data["points"]) == 1
    assert data["points"][0]["value"] == 42.0

def test_get_dashboard(client):
    response = client.get("/api/v1/analytics/test_avatar/dashboard", headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert data["kpis"]["total_conversations"] == 100
    assert "autonomy" in data["trends"]

def test_get_drift(client):
    response = client.get("/api/v1/analytics/test_avatar/drift", headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 1
    assert data["alerts"][0]["metric"] == "response_time"

def test_export_report(client):
    response = client.get("/api/v1/analytics/test_avatar/report", headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert data["period_days"] == 30
    assert data["kpis"]["accuracy_score"] == 0.92

# =============================================================================
# Supervisor Routes
# =============================================================================

def test_list_operators(client):
    response = client.get("/api/v1/supervisor/operators", headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 1
    assert data["operators"][0]["operator_id"] == "op_1"

def test_get_operator(client):
    response = client.get("/api/v1/supervisor/operators/op_1", headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert data["operator_id"] == "op_1"
    assert data["quality_score"] == 0.95

def test_active_sessions(client):
    response = client.get("/api/v1/supervisor/sessions/active", headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 1
    assert data["sessions"][0]["session_id"] == "s1"

def test_review_queue(client):
    response = client.get("/api/v1/supervisor/review-queue", headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 1
    assert data["reviews"][0]["review_id"] == "r1"

def test_submit_review(client):
    payload = {"reviewer_id": "sup1", "verdict": "approved"}
    response = client.post("/api/v1/supervisor/review-queue/r1", json=payload, headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert data["review_id"] == "r1"

def test_activity_timeline(client):
    response = client.get("/api/v1/supervisor/activity-timeline", headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 1
    assert data["entries"][0]["action_type"] == "override"

# =============================================================================
# Guardrails Routes
# =============================================================================

def test_get_policy(client):
    response = client.get("/api/v1/guardrails/test_avatar/policy", headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert "politics" in data["blocked_topics"]

def test_set_policy(client):
    payload = {
        "blocked_topics": ["nsfw"],
        "required_disclaimers": [],
        "max_response_length": 1000,
        "escalation_keywords": ["agent"]
    }
    response = client.put("/api/v1/guardrails/test_avatar/policy", json=payload, headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert data["max_response_length"] == 1000

def test_delete_policy(client):
    response = client.delete("/api/v1/guardrails/test_avatar/policy", headers=HEADERS)
    assert response.status_code == 200
    assert response.json()["message"] == "Policy deleted"

def test_list_violations(client):
    response = client.get("/api/v1/guardrails/test_avatar/violations", headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2

def test_guardrails_audit(client):
    response = client.get("/api/v1/guardrails/test_avatar/audit", headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert data["total_violations"] == 2
    assert data["violation_types"]["blocked"] == 2

# =============================================================================
# Learning Analytics Routes
# =============================================================================

def test_quality_stats(client):
    response = client.get("/api/v1/training/test_avatar/quality-stats?skill_id=s1", headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert data["skill_id"] == "s1"
    assert data["total"] == 100
    assert data["good"] == 80

def test_improvement_timeline(client):
    response = client.get("/api/v1/training/test_avatar/improvement-timeline", headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert len(data["timeline"]) == 1
    assert data["timeline"][0]["qa_added"] == 5

def test_consolidate_daily(client):
    response = client.post("/api/v1/training/test_avatar/consolidate", headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert data["skills_consolidated"] == 3

def test_export_qa_pairs(client):
    response = client.get("/api/v1/training/test_avatar/export", headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert data["format"] == "jsonl"
    assert data["record_count"] == 1
    assert "q2" in data["content"]

def test_weak_areas(client):
    response = client.get("/api/v1/training/test_avatar/skills/all/weak-areas", headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 1
    assert data["weak_areas"][0]["skill_id"] == "weak_1"
