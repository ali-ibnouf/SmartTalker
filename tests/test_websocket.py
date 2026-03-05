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
        """Test sending text and receiving multi-message response with KB fields."""
        # Setup mock pipeline response with KB/escalation fields
        mock_pipeline.process_text = AsyncMock(return_value=PipelineResult(
            response_text="Hello back",
            detected_emotion="happy",
            body_state="talking_happy",
            audio_path="output.wav",
            lip_sync={"words": []},
            total_latency_ms=100,
            breakdown={"llm_ms": 50, "tts_ms": 50},
            kb_confidence=0.85,
            kb_sources=["doc1.txt"],
            escalated=False,
            escalation_id=None,
        ))

        with client.websocket_connect("/ws/chat") as websocket:
            websocket.receive_json()  # init

            # Send text chat message
            websocket.send_json({
                "type": "text_chat",
                "text": "Hello",
                "language": "en",
            })

            # 1. thinking
            msg1 = websocket.receive_json()
            assert msg1["type"] == "thinking"

            # 2. body_state
            msg2 = websocket.receive_json()
            assert msg2["type"] == "body_state"
            assert msg2["state"] == "talking_happy"
            assert "clip_url" in msg2

            # 3. voice
            msg3 = websocket.receive_json()
            assert msg3["type"] == "voice"
            assert msg3["audio_url"] == "/files/output.wav"

            # 4. response (now includes KB/escalation fields)
            msg4 = websocket.receive_json()
            assert msg4["type"] == "response"
            assert msg4["text"] == "Hello back"
            assert msg4["emotion"] == "happy"
            assert msg4["latency_ms"] == 100
            assert msg4["kb_confidence"] == 0.85
            assert msg4["kb_sources"] == ["doc1.txt"]
            assert msg4["escalated"] is False
            assert msg4["escalation_id"] is None

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
                "language": "en",
            })

            # Expect config_ack
            response = websocket.receive_json()
            assert response["type"] == "config_ack"
            assert response["avatar_id"] == "avatar-123"
            assert response["language"] == "en"

    def test_set_state(self, client):
        """Test manual clip state switching."""
        with client.websocket_connect("/ws/chat") as websocket:
            websocket.receive_json()  # init

            websocket.send_json({"type": "set_state", "state": "thinking"})
            response = websocket.receive_json()
            assert response["type"] == "body_state"
            assert response["state"] == "thinking"

    def test_set_state_invalid(self, client):
        """Test invalid state returns error."""
        with client.websocket_connect("/ws/chat") as websocket:
            websocket.receive_json()  # init

            websocket.send_json({"type": "set_state", "state": "dancing"})
            response = websocket.receive_json()
            assert response["type"] == "error"
            assert "Invalid state" in response["error"]

    def test_stop_returns_idle(self, client):
        """Test stop returns avatar to idle."""
        with client.websocket_connect("/ws/chat") as websocket:
            websocket.receive_json()  # init

            websocket.send_json({"type": "stop"})
            response = websocket.receive_json()
            assert response["type"] == "body_state"
            assert response["state"] == "idle"

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

    def test_training_mode_switch(self, client):
        """Test switching training mode between digital and human."""
        with client.websocket_connect("/ws/chat") as websocket:
            websocket.receive_json()  # init

            # Switch to human mode
            websocket.send_json({"type": "training_mode", "mode": "human"})
            response = websocket.receive_json()
            assert response["type"] == "training_mode_ack"
            assert response["mode"] == "human"

            # Switch back to digital
            websocket.send_json({"type": "training_mode", "mode": "digital"})
            response = websocket.receive_json()
            assert response["type"] == "training_mode_ack"
            assert response["mode"] == "digital"

    def test_training_mode_invalid(self, client):
        """Test invalid training mode returns error."""
        with client.websocket_connect("/ws/chat") as websocket:
            websocket.receive_json()  # init

            websocket.send_json({"type": "training_mode", "mode": "invalid"})
            response = websocket.receive_json()
            assert response["type"] == "error"
            assert "Invalid training mode" in response["error"]

    def test_human_mode_awaits_operator(self, client, mock_pipeline):
        """In human mode, text_chat sends awaiting_operator instead of pipeline."""
        with client.websocket_connect("/ws/chat") as websocket:
            websocket.receive_json()  # init

            # Switch to human mode
            websocket.send_json({"type": "training_mode", "mode": "human"})
            websocket.receive_json()  # training_mode_ack

            # Send text — should NOT call pipeline
            websocket.send_json({
                "type": "text_chat",
                "text": "Hello operator",
            })

            response = websocket.receive_json()
            assert response["type"] == "awaiting_operator"
            assert response["text"] == "Hello operator"

            # Pipeline should NOT have been called
            mock_pipeline.process_text.assert_not_called()

    def test_config_includes_training_mode(self, client):
        """Config update can set training_mode."""
        with client.websocket_connect("/ws/chat") as websocket:
            websocket.receive_json()  # init

            websocket.send_json({
                "type": "config",
                "training_mode": "human",
            })

            response = websocket.receive_json()
            assert response["type"] == "config_ack"
            assert response["training_mode"] == "human"
