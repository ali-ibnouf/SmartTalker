"""Tests for the SessionLinkService.

All Redis calls are mocked — no real Redis needed.
"""

import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from src.services.session_link_service import SessionLinkService, SESSION_LINK_PREFIX


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_redis():
    """Mock async Redis client with all needed methods."""
    redis = AsyncMock()
    redis.setex = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    redis.delete = AsyncMock(return_value=1)
    redis.ttl = AsyncMock(return_value=1200)
    return redis


@pytest.fixture
def service(mock_redis):
    return SessionLinkService(mock_redis, "https://app.maskki.com")


# ── create_link ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_link_returns_token_url_expires(service):
    """create_link() returns dict with token, url, expires_at, expires_minutes."""
    result = await service.create_link(
        customer_id="cust_001",
        avatar_id="avatar_001",
    )

    assert "token" in result
    assert "url" in result
    assert "expires_at" in result
    assert result["expires_minutes"] == 30


@pytest.mark.asyncio
async def test_create_link_stores_in_redis(service, mock_redis):
    """create_link() stores session data in Redis with correct TTL."""
    result = await service.create_link(
        customer_id="cust_001",
        avatar_id="avatar_001",
        expires_minutes=15,
    )

    mock_redis.setex.assert_called_once()
    call_args = mock_redis.setex.call_args
    key = call_args[0][0]
    ttl = call_args[0][1]
    data_json = call_args[0][2]

    assert key == f"{SESSION_LINK_PREFIX}{result['token']}"
    assert ttl == 15 * 60  # 900 seconds
    data = json.loads(data_json)
    assert data["customer_id"] == "cust_001"
    assert data["avatar_id"] == "avatar_001"
    assert data["status"] == "pending"


@pytest.mark.asyncio
async def test_create_link_url_format(service):
    """URL format is base_url/s/token."""
    result = await service.create_link(
        customer_id="cust_001",
        avatar_id="avatar_001",
    )
    assert result["url"] == f"https://app.maskki.com/s/{result['token']}"


@pytest.mark.asyncio
async def test_create_link_token_length(service):
    """Token is at least 16 characters (secrets.token_urlsafe(16) = 22 chars)."""
    result = await service.create_link(
        customer_id="cust_001",
        avatar_id="avatar_001",
    )
    assert len(result["token"]) >= 16


# ── get_session ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_session_returns_data(service, mock_redis):
    """get_session() returns session data for a valid token."""
    session_data = {
        "token": "test_token",
        "customer_id": "cust_001",
        "avatar_id": "avatar_001",
        "status": "pending",
        "expires_at": (datetime.utcnow() + timedelta(minutes=30)).isoformat(),
    }
    mock_redis.get.return_value = json.dumps(session_data)

    result = await service.get_session("test_token")

    assert result is not None
    assert result["customer_id"] == "cust_001"
    assert result["status"] == "pending"


@pytest.mark.asyncio
async def test_get_session_expired_returns_none(service, mock_redis):
    """get_session() returns None for an expired token (past expires_at)."""
    session_data = {
        "token": "expired_token",
        "customer_id": "cust_001",
        "avatar_id": "avatar_001",
        "status": "pending",
        "expires_at": (datetime.utcnow() - timedelta(minutes=5)).isoformat(),
    }
    mock_redis.get.return_value = json.dumps(session_data)

    result = await service.get_session("expired_token")

    assert result is None
    mock_redis.delete.assert_called_once()


@pytest.mark.asyncio
async def test_get_session_unknown_returns_none(service, mock_redis):
    """get_session() returns None for an unknown token."""
    mock_redis.get.return_value = None

    result = await service.get_session("nonexistent")

    assert result is None


# ── activate_session ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_activate_session_changes_status(service, mock_redis):
    """activate_session() changes status to 'active'."""
    session_data = {
        "token": "test_token",
        "customer_id": "cust_001",
        "avatar_id": "avatar_001",
        "status": "pending",
        "expires_at": (datetime.utcnow() + timedelta(minutes=30)).isoformat(),
    }
    mock_redis.get.return_value = json.dumps(session_data)

    result = await service.activate_session("test_token")

    assert result is True
    # Verify Redis was updated
    assert mock_redis.setex.call_count == 1
    updated_data = json.loads(mock_redis.setex.call_args[0][2])
    assert updated_data["status"] == "active"
    assert "activated_at" in updated_data


# ── complete_session ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_complete_session_changes_status(service, mock_redis):
    """complete_session() changes status to 'completed' with 1hr TTL."""
    session_data = {
        "token": "test_token",
        "customer_id": "cust_001",
        "avatar_id": "avatar_001",
        "status": "active",
        "expires_at": (datetime.utcnow() + timedelta(minutes=30)).isoformat(),
    }
    mock_redis.get.return_value = json.dumps(session_data)

    result = await service.complete_session("test_token")

    assert result is True
    call_args = mock_redis.setex.call_args[0]
    assert call_args[1] == 3600  # 1 hour audit retention
    updated_data = json.loads(call_args[2])
    assert updated_data["status"] == "completed"
    assert "completed_at" in updated_data


# ── invalidate_link ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_invalidate_link_deletes_redis_key(service, mock_redis):
    """invalidate_link() deletes the Redis key."""
    mock_redis.delete.return_value = 1

    result = await service.invalidate_link("test_token")

    assert result is True
    mock_redis.delete.assert_called_once_with(f"{SESSION_LINK_PREFIX}test_token")
