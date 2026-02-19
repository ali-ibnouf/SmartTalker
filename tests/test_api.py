"""Tests for SmartTalker API endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def app(config, mock_pipeline):
    """Create a test FastAPI app with mocked pipeline."""
    from fastapi import FastAPI
    from src.api.routes import router
    from src.api.middleware import RequestIDMiddleware, LoggingMiddleware

    application = FastAPI()
    application.add_middleware(LoggingMiddleware)
    application.add_middleware(RequestIDMiddleware)
    application.include_router(router)

    application.state.config = config
    application.state.pipeline = mock_pipeline

    return application


@pytest.fixture
def client(app) -> TestClient:
    """Create a test client."""
    return TestClient(app)


# =============================================================================
# Health Endpoint
# =============================================================================


class TestHealthEndpoint:
    """Tests for GET /api/v1/health."""

    def test_health_returns_200(self, client):
        """Health check returns 200 with valid structure."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("healthy", "degraded")
        assert "models_loaded" in data

    def test_health_has_request_id(self, client):
        """Health response includes X-Request-ID header."""
        response = client.get("/api/v1/health")
        assert "x-request-id" in response.headers


# =============================================================================
# Text-to-Speech Endpoint
# =============================================================================


class TestTextToSpeechEndpoint:
    """Tests for POST /api/v1/text-to-speech."""

    def test_empty_text_returns_422(self, client):
        """Empty text triggers validation error."""
        response = client.post("/api/v1/text-to-speech", json={"text": ""})
        assert response.status_code == 422

    def test_valid_text_calls_pipeline(self, client, mock_pipeline):
        """Valid request calls process_text on the pipeline."""
        from src.pipeline.orchestrator import PipelineResult

        mock_pipeline.process_text = AsyncMock(return_value=PipelineResult(
            audio_path="test.wav",
            response_text="Hello!",
            total_latency_ms=100,
            breakdown={"llm_ms": 50, "tts_ms": 50},
        ))

        # Need to patch _to_file_url to avoid file copy
        with patch("src.api.routes._to_file_url", return_value="http://test/files/test.wav"):
            response = client.post(
                "/api/v1/text-to-speech",
                json={"text": "Hello", "language": "en"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["response_text"] == "Hello!"
        assert data["total_latency_ms"] == 100

    def test_text_too_long_returns_422(self, client):
        """Text exceeding max length returns 422."""
        response = client.post(
            "/api/v1/text-to-speech",
            json={"text": "x" * 2001},
        )
        assert response.status_code == 422


# =============================================================================
# Voices Endpoint
# =============================================================================


class TestVoicesEndpoint:
    """Tests for GET /api/v1/voices."""

    def test_list_voices_empty(self, client, mock_pipeline):
        """Empty voice list returns correctly."""
        response = client.get("/api/v1/voices")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 0
        assert data["voices"] == []


# =============================================================================
# Middleware Tests
# =============================================================================


class TestMiddleware:
    """Tests for API middleware."""

    def test_request_id_generated(self, client):
        """Requests without X-Request-ID get one generated."""
        response = client.get("/api/v1/health")
        assert len(response.headers.get("x-request-id", "")) > 0

    def test_request_id_forwarded(self, client):
        """Requests with X-Request-ID get it echoed back."""
        custom_id = "test-id-12345"
        response = client.get(
            "/api/v1/health",
            headers={"X-Request-ID": custom_id},
        )
        assert response.headers.get("x-request-id") == custom_id


# =============================================================================
# Schema Validation Tests
# =============================================================================


class TestSchemas:
    """Tests for Pydantic request/response schemas."""

    def test_text_request_valid(self):
        """Valid TextRequest passes validation."""
        from src.api.schemas import TextRequest
        req = TextRequest(text="Hello")
        assert req.text == "Hello"
        assert req.language == "ar"
        assert req.emotion == "neutral"

    def test_text_request_defaults(self):
        """TextRequest has correct defaults."""
        from src.api.schemas import TextRequest
        req = TextRequest(text="Test")
        assert req.avatar_id == "default"
        assert req.voice_id is None

    def test_error_response_serialization(self):
        """ErrorResponse serializes correctly."""
        from src.api.schemas import ErrorResponse
        err = ErrorResponse(error="test error", detail="details")
        data = err.model_dump()
        assert data["error"] == "test error"
        assert data["detail"] == "details"

    def test_health_response_model(self):
        """HealthResponse validates correctly."""
        from src.api.schemas import HealthResponse
        health = HealthResponse(
            status="healthy",
            gpu_available=True,
            gpu_memory_used_mb=1024.0,
        )
        assert health.status == "healthy"
        assert health.gpu_available is True
