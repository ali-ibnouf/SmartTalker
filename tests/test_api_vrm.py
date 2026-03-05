"""Tests for VRM avatar endpoints."""

import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from src.main import app

API_KEY = "test_key_123"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

@pytest.fixture
def client(monkeypatch, tmp_path):
    """TestClient with mocked config pointing to tmp_path for static files."""
    monkeypatch.setenv("API_KEY", API_KEY)

    with TestClient(app) as test_client:
        mock_config = MagicMock()
        mock_config.static_files_dir = tmp_path / "files"
        (tmp_path / "files").mkdir(exist_ok=True)
        app.state.config = mock_config
        yield test_client

def test_get_vrm_info_no_vrm(client):
    response = client.get("/api/v1/avatars/test_avatar/vrm-info", headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert data["avatar_id"] == "test_avatar"
    assert data["avatar_type"] == "video"
    assert data["has_vrm"] is False
    assert data["vrm_url"] is None

def test_set_avatar_type_vrm_without_file_fails(client):
    payload = {"avatar_type": "vrm"}
    response = client.put("/api/v1/avatars/test_avatar/type", json=payload, headers=HEADERS)
    assert response.status_code == 400
    assert "No VRM file uploaded" in response.json()["detail"]

def test_upload_vrm_success(client):
    # Create a small dummy file simulating a .vrm file
    file_content = b"fake vrm data"
    files = {"file": ("model.vrm", file_content, "application/octet-stream")}
    
    response = client.post("/api/v1/avatars/test_avatar/vrm", files=files, headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert data["avatar_id"] == "test_avatar"
    assert "model.vrm" in data["vrm_url"]
    assert data["file_size_bytes"] == len(file_content)

def test_upload_vrm_wrong_extension(client):
    files = {"file": ("model.txt", b"fake data", "text/plain")}
    response = client.post("/api/v1/avatars/test_avatar/vrm", files=files, headers=HEADERS)
    assert response.status_code == 400
    assert "File must have .vrm extension" in response.json()["detail"]

def test_get_vrm_info_with_vrm(client):
    # First upload
    files = {"file": ("model.vrm", b"data", "application/octet-stream")}
    client.post("/api/v1/avatars/test_avatar/vrm", files=files, headers=HEADERS)
    
    # Then get info
    response = client.get("/api/v1/avatars/test_avatar/vrm-info", headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert data["avatar_type"] == "vrm"
    assert data["has_vrm"] is True
    assert data["vrm_url"] == "/files/vrm/test_avatar/model.vrm"

def test_set_avatar_type_video(client):
    payload = {"avatar_type": "video"}
    response = client.put("/api/v1/avatars/test_avatar/type", json=payload, headers=HEADERS)
    assert response.status_code == 200
    assert response.json()["avatar_type"] == "video"

def test_set_avatar_type_vrm_success(client):
    # First upload
    files = {"file": ("model.vrm", b"data", "application/octet-stream")}
    client.post("/api/v1/avatars/test_avatar/vrm", files=files, headers=HEADERS)
    
    # Then set type to vrm
    payload = {"avatar_type": "vrm"}
    response = client.put("/api/v1/avatars/test_avatar/type", json=payload, headers=HEADERS)
    assert response.status_code == 200
    assert response.json()["avatar_type"] == "vrm"
