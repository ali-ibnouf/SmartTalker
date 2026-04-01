"""Tests for extended KB routes (kb_routes.py)."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.kb_routes import router


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@dataclass
class FakeKBDocument:
    doc_id: str = "doc-1"
    filename: str = "test.txt"
    doc_type: str = "manual"
    chunk_count: int = 3
    created_at: float = 1700000000.0
    file_hash: str = ""
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@pytest.fixture
def mock_kb():
    kb = MagicMock()
    kb.is_loaded = True
    kb.ingest_text = AsyncMock(return_value=FakeKBDocument())
    kb.list_documents = MagicMock(return_value=[FakeKBDocument()])
    kb.delete_document = MagicMock(return_value=True)
    return kb


@pytest.fixture
def mock_db():
    """Mock DB with async session context manager."""
    db = MagicMock()
    session = AsyncMock()
    # Make session context manager
    db.session.return_value.__aenter__ = AsyncMock(return_value=session)
    db.session.return_value.__aexit__ = AsyncMock(return_value=False)
    return db


@pytest.fixture
def app(mock_kb, mock_db):
    application = FastAPI()
    application.include_router(router)

    pipeline = MagicMock()
    pipeline._kb = mock_kb
    application.state.pipeline = pipeline
    application.state.db = mock_db

    # Inject customer_id via middleware
    @application.middleware("http")
    async def inject_customer(request, call_next):
        request.state.customer_id = "cust-123"
        return await call_next(request)

    return application


@pytest.fixture
def client(app) -> TestClient:
    return TestClient(app)


# ---------------------------------------------------------------------------
# POST /kb/ingest-text
# ---------------------------------------------------------------------------


class TestIngestText:
    def test_success(self, client, mock_kb):
        resp = client.post("/api/v1/kb/ingest-text", json={
            "title": "Test Entry",
            "content": "This is test content for the KB.",
            "tags": ["test"],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["doc_id"] == "doc-1"
        assert data["doc_type"] == "manual"
        assert data["chunk_count"] == 3
        mock_kb.ingest_text.assert_awaited_once()

    def test_empty_content_returns_422(self, client):
        resp = client.post("/api/v1/kb/ingest-text", json={
            "title": "Test",
            "content": "",
        })
        assert resp.status_code == 422

    def test_missing_title_returns_422(self, client):
        resp = client.post("/api/v1/kb/ingest-text", json={
            "content": "Some content",
        })
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /kb/scrape
# ---------------------------------------------------------------------------


class TestScrape:
    def test_valid_url(self, client, mock_kb):
        html = "<html><body><p>Hello world from the website.</p></body></html>"
        mock_resp = MagicMock()
        mock_resp.text = html
        mock_resp.raise_for_status = MagicMock()

        with patch("src.api.kb_routes.httpx.AsyncClient") as MockClient:
            ctx = AsyncMock()
            ctx.get = AsyncMock(return_value=mock_resp)
            MockClient.return_value.__aenter__ = AsyncMock(return_value=ctx)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            resp = client.post("/api/v1/kb/scrape", json={"url": "https://example.com"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["doc_id"] == "doc-1"
        assert "preview" in data

    def test_invalid_url_scheme(self, client):
        resp = client.post("/api/v1/kb/scrape", json={"url": "ftp://bad.com"})
        assert resp.status_code == 400
        assert "http" in resp.json()["detail"].lower()

    def test_unreachable_url(self, client):
        import httpx
        with patch("src.api.kb_routes.httpx.AsyncClient") as MockClient:
            ctx = AsyncMock()
            ctx.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
            MockClient.return_value.__aenter__ = AsyncMock(return_value=ctx)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            resp = client.post("/api/v1/kb/scrape", json={"url": "https://nope.invalid"})

        assert resp.status_code == 400
        assert "fetch" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# DELETE /kb/documents/all
# ---------------------------------------------------------------------------


class TestDeleteAllDocuments:
    def test_success(self, client, mock_kb):
        resp = client.delete("/api/v1/kb/documents/all", headers={"X-Confirm": "delete-all"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["deleted"] == 1

    def test_missing_confirm_header(self, client):
        resp = client.delete("/api/v1/kb/documents/all")
        assert resp.status_code == 400
        assert "X-Confirm" in resp.json()["detail"]

    def test_wrong_confirm_value(self, client):
        resp = client.delete("/api/v1/kb/documents/all", headers={"X-Confirm": "wrong"})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# GET /kb/knowledge (mocked DB)
# ---------------------------------------------------------------------------


class TestListKnowledge:
    def test_returns_list(self, client, mock_db):
        """With mocked empty results, returns empty list."""
        mock_session = AsyncMock()
        # count query
        count_result = MagicMock()
        count_result.scalar.return_value = 0
        # rows query
        rows_result = MagicMock()
        rows_result.scalars.return_value.all.return_value = []

        mock_session.execute = AsyncMock(side_effect=[count_result, rows_result])
        mock_db.session.return_value.__aenter__ = AsyncMock(return_value=mock_session)

        resp = client.get("/api/v1/kb/knowledge")
        assert resp.status_code == 200
        data = resp.json()
        assert data["items"] == []
        assert data["count"] == 0


# ---------------------------------------------------------------------------
# GET /kb/analytics
# ---------------------------------------------------------------------------


class TestAnalytics:
    def test_returns_stats(self, client, mock_kb, mock_db):
        mock_session = AsyncMock()
        # knowledge_count, avg_confidence, unanswered — 3 queries
        count_result = MagicMock()
        count_result.scalar.return_value = 10

        avg_result = MagicMock()
        avg_result.scalar.return_value = 0.75

        unanswered_result = MagicMock()
        unanswered_result.scalar.return_value = 2

        mock_session.execute = AsyncMock(
            side_effect=[count_result, avg_result, unanswered_result]
        )
        mock_db.session.return_value.__aenter__ = AsyncMock(return_value=mock_session)

        resp = client.get("/api/v1/kb/analytics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_documents"] == 1
        assert data["total_chunks"] == 3
        assert data["total_knowledge_items"] == 10
        assert data["avg_confidence"] == 0.75
        assert data["unanswered_count"] == 2
        assert len(data["growth"]) == 30


# ---------------------------------------------------------------------------
# KB unavailable
# ---------------------------------------------------------------------------


class TestKBUnavailable:
    def test_ingest_text_503(self, client, mock_kb):
        mock_kb.is_loaded = False
        resp = client.post("/api/v1/kb/ingest-text", json={
            "title": "Test", "content": "Content here",
        })
        assert resp.status_code == 503

    def test_analytics_503(self, client, mock_kb, mock_db):
        mock_kb.is_loaded = False
        resp = client.get("/api/v1/kb/analytics")
        assert resp.status_code == 503
