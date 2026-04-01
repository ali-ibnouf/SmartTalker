"""Tests for agent templates & customer agents API routes."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.agent_templates import (
    admin_router,
    customer_router,
    public_router,
)


# ---------------------------------------------------------------------------
# Fake ORM objects
# ---------------------------------------------------------------------------


@dataclass
class FakeTemplate:
    id: str = "tpl-1"
    slug: str = "test-template"
    name_ar: str = "قالب تجريبي"
    name_en: str = "Test Template"
    description_ar: str = "وصف"
    description_en: str = "Description"
    job_title_ar: str = "موظف"
    job_title_en: str = "Officer"
    category: str = "general"
    icon_emoji: str = "🤖"
    color_accent: str = "#00D4AA"
    default_language: str = "ar"
    default_personality: str = "professional"
    system_prompt: str = "You are an assistant."
    kb_template: str = '{"sample_faqs": [{"q": "test?", "a": "yes"}]}'
    is_published: bool = True
    sort_order: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    agents: list = field(default_factory=list)


@dataclass
class FakeAgent:
    id: str = "agent-1"
    customer_id: str = "cust-123"
    template_id: str = "tpl-1"
    name_ar: str = "وكيل تجريبي"
    name_en: str = "Test Agent"
    description: str = "desc"
    photo_url: str = ""
    photo_r2_key: str = ""
    photo_preprocessed: bool = False
    voice_id: str = ""
    voice_cloned: bool = False
    personality: str = "professional"
    language: str = "ar"
    kb_document_count: int = 0
    kb_faq_count: int = 0
    kb_status: str = "empty"
    is_active: bool = True
    channels: str = '{"widget": true}'
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    customer: object = None
    template: object = None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


_UNSET = object()


def _mock_session_with(scalars_all=_UNSET, scalar_one_or_none=_UNSET, scalar=_UNSET):
    """Build a mock async session that returns given data for execute().

    Uses a sentinel instead of None to avoid the MagicMock getattr trap:
    if we skip configuring scalar_one_or_none, MagicMock auto-creates a
    truthy sub-mock instead of returning None.
    """
    session = AsyncMock()

    result = MagicMock()
    if scalars_all is not _UNSET:
        result.scalars.return_value.all.return_value = scalars_all
    if scalar_one_or_none is not _UNSET:
        result.scalar_one_or_none.return_value = scalar_one_or_none
    if scalar is not _UNSET:
        result.scalar.return_value = scalar

    session.execute = AsyncMock(return_value=result)
    return session


def _make_db(session):
    db = MagicMock()
    db.session.return_value.__aenter__ = AsyncMock(return_value=session)
    db.session.return_value.__aexit__ = AsyncMock(return_value=False)
    return db


@pytest.fixture
def public_app():
    """App with public router only (no auth)."""
    app = FastAPI()
    app.include_router(public_router)

    session = _mock_session_with(scalars_all=[FakeTemplate()])
    app.state.db = _make_db(session)
    return app


@pytest.fixture
def public_client(public_app) -> TestClient:
    return TestClient(public_app)


@pytest.fixture
def customer_app():
    """App with customer router (injects customer_id via middleware)."""
    app = FastAPI()
    app.include_router(customer_router)

    @app.middleware("http")
    async def inject_customer(request, call_next):
        request.state.customer_id = "cust-123"
        return await call_next(request)

    return app


@pytest.fixture
def admin_app():
    """App with admin router."""
    app = FastAPI()
    app.include_router(admin_router)
    return app


# ---------------------------------------------------------------------------
# PUBLIC — GET /api/v1/public/agent-templates
# ---------------------------------------------------------------------------


class TestPublicTemplates:
    def test_list_published(self, public_client):
        resp = public_client.get("/api/v1/public/agent-templates")
        assert resp.status_code == 200
        data = resp.json()
        assert "templates" in data
        assert len(data["templates"]) == 1
        assert data["templates"][0]["slug"] == "test-template"
        assert data["templates"][0]["is_published"] is True

    def test_template_dict_fields(self, public_client):
        resp = public_client.get("/api/v1/public/agent-templates")
        tpl = resp.json()["templates"][0]
        required_fields = {
            "id", "slug", "name_ar", "name_en", "description_ar",
            "description_en", "category", "icon_emoji", "color_accent",
            "system_prompt", "kb_template", "is_published",
        }
        assert required_fields.issubset(set(tpl.keys()))
        # kb_template should be parsed dict, not raw string
        assert isinstance(tpl["kb_template"], dict)


# ---------------------------------------------------------------------------
# CUSTOMER — GET /api/v1/agents
# ---------------------------------------------------------------------------


class TestListAgents:
    def test_list_empty(self, customer_app):
        session = _mock_session_with(scalars_all=[])
        customer_app.state.db = _make_db(session)
        client = TestClient(customer_app)

        resp = client.get("/api/v1/agents")
        assert resp.status_code == 200
        assert resp.json()["agents"] == []

    def test_list_with_agents(self, customer_app):
        session = _mock_session_with(scalars_all=[FakeAgent()])
        customer_app.state.db = _make_db(session)
        client = TestClient(customer_app)

        resp = client.get("/api/v1/agents")
        assert resp.status_code == 200
        agents = resp.json()["agents"]
        assert len(agents) == 1
        assert agents[0]["name_ar"] == "وكيل تجريبي"
        assert isinstance(agents[0]["channels"], dict)


# ---------------------------------------------------------------------------
# CUSTOMER — GET /api/v1/agents/{agent_id}
# ---------------------------------------------------------------------------


class TestGetAgent:
    def test_found(self, customer_app):
        session = _mock_session_with(scalar_one_or_none=FakeAgent())
        customer_app.state.db = _make_db(session)
        client = TestClient(customer_app)

        resp = client.get("/api/v1/agents/agent-1")
        assert resp.status_code == 200
        assert resp.json()["id"] == "agent-1"

    def test_not_found(self, customer_app):
        session = _mock_session_with(scalar_one_or_none=None)
        customer_app.state.db = _make_db(session)
        client = TestClient(customer_app)

        resp = client.get("/api/v1/agents/nonexistent")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# CUSTOMER — POST /api/v1/agents/from-template
# ---------------------------------------------------------------------------


class TestCreateAgent:
    def test_create_without_template(self, customer_app):
        session = AsyncMock()
        session.add = MagicMock()
        session.commit = AsyncMock()
        customer_app.state.db = _make_db(session)
        client = TestClient(customer_app)

        resp = client.post("/api/v1/agents/from-template", json={
            "name_ar": "وكيل جديد",
            "name_en": "New Agent",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["name_ar"] == "وكيل جديد"
        assert len(data["id"]) == 32  # uuid4().hex

    def test_create_from_template(self, customer_app):
        template = FakeTemplate()
        session = AsyncMock()

        # First execute: select template
        tpl_result = MagicMock()
        tpl_result.scalar_one_or_none.return_value = template
        # Second execute: not called (session.add + commit)
        session.execute = AsyncMock(return_value=tpl_result)
        session.add = MagicMock()
        session.commit = AsyncMock()
        customer_app.state.db = _make_db(session)
        client = TestClient(customer_app)

        resp = client.post("/api/v1/agents/from-template", json={
            "name_ar": "وكيل من قالب",
            "template_id": "tpl-1",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["template_id"] == "tpl-1"
        assert data["personality"] == "professional"
        assert data["kb_faq_count"] == 1
        assert data["kb_status"] == "seeded"

    def test_create_missing_name(self, customer_app):
        session = AsyncMock()
        customer_app.state.db = _make_db(session)
        client = TestClient(customer_app)

        resp = client.post("/api/v1/agents/from-template", json={
            "name_en": "No Arabic Name",
        })
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# CUSTOMER — DELETE /api/v1/agents/{agent_id}
# ---------------------------------------------------------------------------


class TestDeleteAgent:
    def test_delete_success(self, customer_app):
        agent = FakeAgent()
        session = AsyncMock()
        result = MagicMock()
        result.scalar_one_or_none.return_value = agent
        session.execute = AsyncMock(return_value=result)
        session.delete = AsyncMock()
        session.commit = AsyncMock()
        customer_app.state.db = _make_db(session)
        client = TestClient(customer_app)

        resp = client.delete("/api/v1/agents/agent-1")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_delete_not_found(self, customer_app):
        session = _mock_session_with(scalar_one_or_none=None)
        customer_app.state.db = _make_db(session)
        client = TestClient(customer_app)

        resp = client.delete("/api/v1/agents/nonexistent")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# ADMIN — GET /api/v1/admin/agent-templates
# ---------------------------------------------------------------------------


class TestAdminTemplates:
    def test_list_all(self, admin_app):
        session = _mock_session_with(scalars_all=[FakeTemplate(), FakeTemplate(id="tpl-2", slug="second")])
        admin_app.state.db = _make_db(session)
        client = TestClient(admin_app)

        resp = client.get("/api/v1/admin/agent-templates")
        assert resp.status_code == 200
        assert len(resp.json()["templates"]) == 2


# ---------------------------------------------------------------------------
# ADMIN — POST /api/v1/admin/agent-templates
# ---------------------------------------------------------------------------


class TestAdminCreateTemplate:
    def test_create_success(self, admin_app):
        session = AsyncMock()
        session.add = MagicMock()
        session.commit = AsyncMock()
        admin_app.state.db = _make_db(session)
        client = TestClient(admin_app)

        resp = client.post("/api/v1/admin/agent-templates", json={
            "slug": "new-template",
            "name_ar": "قالب جديد",
            "name_en": "New Template",
            "category": "general",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["slug"] == "new-template"
        assert data["is_published"] is False

    def test_create_missing_slug(self, admin_app):
        session = AsyncMock()
        admin_app.state.db = _make_db(session)
        client = TestClient(admin_app)

        resp = client.post("/api/v1/admin/agent-templates", json={
            "name_ar": "بلا slug",
        })
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# ADMIN — publish / unpublish
# ---------------------------------------------------------------------------


class TestAdminPublish:
    def test_publish(self, admin_app):
        template = FakeTemplate(is_published=False)
        session = AsyncMock()
        result = MagicMock()
        result.scalar_one_or_none.return_value = template
        session.execute = AsyncMock(return_value=result)
        session.commit = AsyncMock()
        admin_app.state.db = _make_db(session)
        client = TestClient(admin_app)

        resp = client.post("/api/v1/admin/agent-templates/tpl-1/publish")
        assert resp.status_code == 200
        assert resp.json()["is_published"] is True

    def test_unpublish(self, admin_app):
        template = FakeTemplate(is_published=True)
        session = AsyncMock()
        result = MagicMock()
        result.scalar_one_or_none.return_value = template
        session.execute = AsyncMock(return_value=result)
        session.commit = AsyncMock()
        admin_app.state.db = _make_db(session)
        client = TestClient(admin_app)

        resp = client.post("/api/v1/admin/agent-templates/tpl-1/unpublish")
        assert resp.status_code == 200
        assert resp.json()["is_published"] is False

    def test_publish_not_found(self, admin_app):
        session = _mock_session_with(scalar_one_or_none=None)
        admin_app.state.db = _make_db(session)
        client = TestClient(admin_app)

        resp = client.post("/api/v1/admin/agent-templates/nonexistent/publish")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# ADMIN — GET /api/v1/admin/stats
# ---------------------------------------------------------------------------


class TestAdminStats:
    def test_returns_counts(self, admin_app):
        session = AsyncMock()
        # 4 count queries: template_count, agent_count, active_agents, kb_ready
        results = []
        for val in [5, 10, 8, 6]:
            r = MagicMock()
            r.scalar.return_value = val
            results.append(r)
        session.execute = AsyncMock(side_effect=results)
        admin_app.state.db = _make_db(session)
        client = TestClient(admin_app)

        resp = client.get("/api/v1/admin/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["agent_templates"] == 5
        assert data["customer_agents"] == 10
        assert data["active_agents"] == 8
        assert data["agents_kb_ready"] == 6


# ---------------------------------------------------------------------------
# DB unavailable
# ---------------------------------------------------------------------------


class TestDBUnavailable:
    def test_public_503(self):
        app = FastAPI()
        app.include_router(public_router)
        app.state.db = None
        client = TestClient(app)

        resp = client.get("/api/v1/public/agent-templates")
        assert resp.status_code == 503

    def test_customer_503(self, customer_app):
        customer_app.state.db = None
        client = TestClient(customer_app)

        resp = client.get("/api/v1/agents")
        assert resp.status_code == 503
