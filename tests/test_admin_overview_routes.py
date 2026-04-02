"""Tests for admin overview routes (admin_overview_routes.py)."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.admin_overview_routes import router


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_db():
    """Mock DB with async session context manager."""
    db = MagicMock()
    session = AsyncMock()
    db.session.return_value.__aenter__ = AsyncMock(return_value=session)
    db.session.return_value.__aexit__ = AsyncMock(return_value=False)
    return db, session


@pytest.fixture
def app(mock_db):
    application = FastAPI()
    application.include_router(router)

    db, _ = mock_db
    application.state.db = db

    @application.middleware("http")
    async def inject_customer(request, call_next):
        request.state.customer_id = "admin"
        return await call_next(request)

    return application


@pytest.fixture
def client(app) -> TestClient:
    return TestClient(app)


# ---------------------------------------------------------------------------
# GET /admin/training/overview
# ---------------------------------------------------------------------------


def test_admin_training_overview(client, mock_db):
    """Training overview returns aggregated stats."""
    _, session = mock_db

    # Mock: status totals query
    status_result = MagicMock()
    status_result.__iter__ = lambda self: iter([("pending", 5), ("approved", 20), ("rejected", 3)])

    # Mock: avg confidence query
    avg_result = MagicMock()
    avg_result.scalar.return_value = 0.7234

    # Mock: per-customer query
    customer_row = MagicMock()
    customer_row.customer_id = "cust-1"
    customer_row.name = "Test Corp"
    customer_row.pending = 3
    customer_row.approved = 15
    customer_row.rejected = 2
    customer_row.avg_conf = 0.68
    customer_result = MagicMock()
    customer_result.__iter__ = lambda self: iter([customer_row])

    # Mock: recent pending query
    entry = MagicMock()
    entry.id = "learn-1"
    entry.customer_id = "cust-1"
    entry.employee_id = "emp-1"
    entry.learning_type = "qa_pair"
    entry.old_value = "What is your return policy?"
    entry.new_value = "We accept returns within 30 days."
    entry.confidence = 0.65
    entry.created_at = datetime(2026, 1, 15, 10, 30)
    recent_result = MagicMock()
    recent_result.__iter__ = lambda self: iter([(entry, "Test Corp")])

    session.execute = AsyncMock(
        side_effect=[status_result, avg_result, customer_result, recent_result]
    )

    resp = client.get("/api/v1/admin/training/overview")
    assert resp.status_code == 200

    data = resp.json()
    assert data["total_pending"] == 5
    assert data["total_approved"] == 20
    assert data["total_rejected"] == 3
    assert data["avg_confidence"] == 0.7234
    assert len(data["customers"]) == 1
    assert data["customers"][0]["customer_name"] == "Test Corp"
    assert len(data["recent_pending"]) == 1
    assert data["recent_pending"][0]["learning_type"] == "qa_pair"


def test_admin_training_overview_empty(client, mock_db):
    """Training overview with no data returns zeros."""
    _, session = mock_db

    empty_iter = MagicMock()
    empty_iter.__iter__ = lambda self: iter([])

    scalar_none = MagicMock()
    scalar_none.scalar.return_value = None

    session.execute = AsyncMock(
        side_effect=[empty_iter, scalar_none, empty_iter, empty_iter]
    )

    resp = client.get("/api/v1/admin/training/overview")
    assert resp.status_code == 200

    data = resp.json()
    assert data["total_pending"] == 0
    assert data["total_approved"] == 0
    assert data["avg_confidence"] == 0.0
    assert data["customers"] == []
    assert data["recent_pending"] == []


# ---------------------------------------------------------------------------
# GET /admin/knowledge/overview
# ---------------------------------------------------------------------------


def test_admin_knowledge_overview(client, mock_db):
    """Knowledge overview returns aggregated stats."""
    _, session = mock_db

    # Mock: totals query
    totals_result = MagicMock()
    totals_result.one.return_value = (42, 0.85)

    # Mock: per-customer query
    customer_row = MagicMock()
    customer_row.customer_id = "cust-1"
    customer_row.name = "Test Corp"
    customer_row.count = 30
    customer_row.avg_rate = 0.88
    customer_result = MagicMock()
    customer_result.__iter__ = lambda self: iter([customer_row])

    # Mock: category query
    cat_result = MagicMock()
    cat_result.__iter__ = lambda self: iter([("general", 20), ("learned", 15), ("faq", 7)])

    session.execute = AsyncMock(
        side_effect=[totals_result, customer_result, cat_result]
    )

    resp = client.get("/api/v1/admin/knowledge/overview")
    assert resp.status_code == 200

    data = resp.json()
    assert data["total_knowledge_items"] == 42
    assert data["avg_success_rate"] == 0.85
    assert data["total_categories"] == 3
    assert len(data["customers"]) == 1
    assert data["customers"][0]["knowledge_count"] == 30
    assert len(data["categories"]) == 3
    assert data["categories"][0]["category"] == "general"


def test_admin_knowledge_overview_empty(client, mock_db):
    """Knowledge overview with no data returns zeros."""
    _, session = mock_db

    totals_result = MagicMock()
    totals_result.one.return_value = (0, None)

    empty_iter = MagicMock()
    empty_iter.__iter__ = lambda self: iter([])

    session.execute = AsyncMock(
        side_effect=[totals_result, empty_iter, empty_iter]
    )

    resp = client.get("/api/v1/admin/knowledge/overview")
    assert resp.status_code == 200

    data = resp.json()
    assert data["total_knowledge_items"] == 0
    assert data["avg_success_rate"] == 0.0
    assert data["total_categories"] == 0
    assert data["customers"] == []
    assert data["categories"] == []


# ---------------------------------------------------------------------------
# Database unavailable
# ---------------------------------------------------------------------------


def test_training_overview_no_db(client, app):
    """Returns 503 when DB is unavailable."""
    app.state.db = None
    resp = client.get("/api/v1/admin/training/overview")
    assert resp.status_code == 503


def test_knowledge_overview_no_db(client, app):
    """Returns 503 when DB is unavailable."""
    app.state.db = None
    resp = client.get("/api/v1/admin/knowledge/overview")
    assert resp.status_code == 503
