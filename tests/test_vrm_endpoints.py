"""Tests for VRM avatar API endpoints."""

import io
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from src.main import app

API_KEY = "test_key_123"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}


@pytest.fixture
def client(monkeypatch, tmp_path):
    """TestClient with mocked config pointing to tmp_path for static files."""
    monkeypatch.setenv("API_KEY", API_KEY)

    with TestClient(app) as test_client:
        # Point static_files_dir at tmp_path so VRM uploads go there
        mock_config = MagicMock()
        mock_config.static_files_dir = tmp_path / "files"
        (tmp_path / "files").mkdir(exist_ok=True)
        app.state.config = mock_config
        yield test_client


def test_upload_vrm_success(client, tmp_path):
    """Upload a valid .vrm file."""
    fake_vrm = b"\x00" * 1024  # 1KB fake VRM
    response = client.post(
        "/api/v1/avatars/test_avatar/vrm",
        files={"file": ("model.vrm", io.BytesIO(fake_vrm), "application/octet-stream")},
        headers=HEADERS,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["avatar_id"] == "test_avatar"
    assert data["vrm_url"] == "/files/vrm/test_avatar/model.vrm"
    assert data["file_size_bytes"] == 1024

    # Verify file exists on disk
    vrm_path = tmp_path / "files" / "vrm" / "test_avatar" / "model.vrm"
    assert vrm_path.exists()
    assert vrm_path.read_bytes() == fake_vrm


def test_upload_vrm_wrong_extension(client):
    """Reject non-.vrm files."""
    response = client.post(
        "/api/v1/avatars/test_avatar/vrm",
        files={"file": ("model.glb", io.BytesIO(b"data"), "application/octet-stream")},
        headers=HEADERS,
    )
    assert response.status_code == 400
    assert "vrm" in response.json()["detail"].lower()


def test_get_vrm_info_no_file(client):
    """VRM info when no file uploaded."""
    response = client.get("/api/v1/avatars/test_avatar/vrm-info", headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert data["avatar_id"] == "test_avatar"
    assert data["has_vrm"] is False
    assert data["vrm_url"] is None


def test_get_vrm_info_with_file(client, tmp_path):
    """VRM info after upload."""
    # Create the VRM file manually
    vrm_dir = tmp_path / "files" / "vrm" / "test_avatar"
    vrm_dir.mkdir(parents=True, exist_ok=True)
    (vrm_dir / "model.vrm").write_bytes(b"\x00" * 100)

    response = client.get("/api/v1/avatars/test_avatar/vrm-info", headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert data["has_vrm"] is True
    assert data["vrm_url"] == "/files/vrm/test_avatar/model.vrm"
    assert data["avatar_type"] == "vrm"


def test_set_avatar_type_to_vrm_without_file(client):
    """Cannot switch to VRM without uploading a file first."""
    response = client.put(
        "/api/v1/avatars/test_avatar/type",
        json={"avatar_type": "vrm"},
        headers=HEADERS,
    )
    assert response.status_code == 400
    assert "No VRM file" in response.json()["detail"]


def test_set_avatar_type_to_vrm_with_file(client, tmp_path):
    """Switch to VRM when file exists."""
    vrm_dir = tmp_path / "files" / "vrm" / "test_avatar"
    vrm_dir.mkdir(parents=True, exist_ok=True)
    (vrm_dir / "model.vrm").write_bytes(b"\x00" * 100)

    response = client.put(
        "/api/v1/avatars/test_avatar/type",
        json={"avatar_type": "vrm"},
        headers=HEADERS,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["avatar_type"] == "vrm"


def test_set_avatar_type_to_video(client):
    """Switch to video mode (always allowed)."""
    response = client.put(
        "/api/v1/avatars/test_avatar/type",
        json={"avatar_type": "video"},
        headers=HEADERS,
    )
    assert response.status_code == 200
    assert response.json()["avatar_type"] == "video"


def test_set_avatar_type_invalid(client):
    """Invalid avatar type rejected by schema validation."""
    response = client.put(
        "/api/v1/avatars/test_avatar/type",
        json={"avatar_type": "invalid"},
        headers=HEADERS,
    )
    assert response.status_code == 422
