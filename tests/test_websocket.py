"""Tests for SmartTalker WebSocket chat endpoint."""

from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient

from src.api.websocket import WebSocketManager, websocket_chat_endpoint
from src.pipeline.orchestrator import PipelineResult

@pytest.fixture
def app(mock_pipeline, tmp_path):
    """Create a test FastAPI app with manual WebSocket setup."""
    application = FastAPI()

    # Initialize manager with mock pipeline and temp storage
    ws_storage = tmp_path / "ws_test"
    manager = WebSocketManager(pipeline=mock_pipeline, storage_dir=ws_storage)
    
    # Store in state
    application.state.ws_manager = manager
    
    # Mount endpoint exactly as in main.py
    @application.websocket("/ws/chat")
    async def ws_chat(websocket: WebSocket):
        await websocket_chat_endpoint(websocket, manager)
        
    return application

@pytest.fixture
def client(app):
    return TestClient(app)

class TestWebSocketChat:
    """Tests for /ws/chat WebSocket endpoint."""

    def test_connection_lifecycle(self, client):
        """Test successful connection and session init."""
        with client.websocket_connect("/ws/chat") as websocket:
            # Should receive session_init immediately
            data = websocket.receive_json()
            assert data["type"] == "session_init"
            assert "session_id" in data
            assert data["message"] == "Connected to SmartTalker"

    def test_ping_pong(self, client):
        """Test ping/pong functionality."""
        with client.websocket_connect("/ws/chat") as websocket:
            websocket.receive_json()  # init
            
            # Send ping
            websocket.send_json({"type": "ping"})
            
            # Expect pong
            response = websocket.receive_json()
            assert response["type"] == "pong"
            assert "timestamp" in response

    def test_text_chat_flow(self, client, mock_pipeline):
        """Test sending text and receiving processing response."""
        # Setup mock pipeline response
        mock_pipeline.process_text = AsyncMock(return_value=PipelineResult(
            response_text="Hello back",
            detected_emotion="joy",
            audio_path="output.wav",
            video_path=None,
            total_latency_ms=100,
            breakdown={"llm": 50, "tts": 50}
        ))

        with client.websocket_connect("/ws/chat") as websocket:
            websocket.receive_json()  # init

            # Send text chat message
            websocket.send_json({
                "type": "text_chat",
                "text": "Hello",
                "language": "en"
            })

            # Should receive text_response
            response = websocket.receive_json()
            assert response["type"] == "text_response"
            assert response["text"] == "Hello back"
            assert response["emotion"] == "joy"
            assert response["audio_url"] == "/files/output.wav"
            
            # Verify pipeline called correctly
            mock_pipeline.process_text.assert_called_once()
            call_args = mock_pipeline.process_text.call_args[1]
            assert call_args["text"] == "Hello"
            assert call_args["language"] == "en"

    def test_config_update(self, client):
        """Test updating session configuration."""
        with client.websocket_connect("/ws/chat") as websocket:
            websocket.receive_json()  # init
            
            # Update config
            websocket.send_json({
                "type": "config",
                "avatar_id": "avatar-123",
                "enable_video": True
            })
            
            # Expect config_ack
            response = websocket.receive_json()
            assert response["type"] == "config_ack"
            assert response["avatar_id"] == "avatar-123"
            assert response["enable_video"] is True

    def test_invalid_json_handling(self, client):
        """Test that invalid JSON returns error message."""
        with client.websocket_connect("/ws/chat") as websocket:
            websocket.receive_json()  # init
            
            # Send plain text string (not JSON)
            websocket.send_text("This is not JSON")
            
            # Expect error
            response = websocket.receive_json()
            assert response["type"] == "error"
            assert response["error"] == "Invalid JSON"
