import sys
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
import pytest
from fastapi import WebSocket, WebSocketDisconnect

# Fixtures
@pytest.fixture
def mock_pipeline():
    return MagicMock()

@pytest.fixture
def mock_config():
    config = MagicMock()
    config.webrtc_stun_servers = "stun:test.com"
    config.webrtc_turn_server = None
    return config

class TestWebRTCSignalingHandler:
    """Tests for WebRTC signaling logic."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Setup fresh mocks for each test."""
        # Create fresh mocks
        self.mock_aiortc = MagicMock()
        self.mock_pc_cls = MagicMock()
        self.mock_pc_instance = MagicMock()
        self.mock_pc_instance.close = AsyncMock()
        self.mock_pc_instance.createAnswer = AsyncMock(return_value=MagicMock(sdp="server_sdp"))
        self.mock_pc_instance.setLocalDescription = AsyncMock()
        self.mock_pc_instance.setRemoteDescription = AsyncMock()
        type(self.mock_pc_instance).localDescription = PropertyMock(return_value=MagicMock(sdp="server_sdp"))
        
        self.mock_pc_cls.return_value = self.mock_pc_instance
        self.mock_aiortc.RTCPeerConnection = self.mock_pc_cls
        self.mock_aiortc.RTCSessionDescription = MagicMock()
        self.mock_aiortc.RTCConfiguration = MagicMock()
        self.mock_aiortc.RTCIceServer = MagicMock()
        self.mock_aiortc.contrib.media = MagicMock()

    async def get_handler(self, mock_pipeline, mock_config):
        """Helper to get handler instance with patches applied."""
        # Patch sys.modules to inject our mock aiortc
        with patch.dict(sys.modules, {"aiortc": self.mock_aiortc, "aiortc.contrib.media": MagicMock()}):
            # We must import INSIDE the patch to pick up the mock
            if "src.integrations.webrtc" in sys.modules:
                del sys.modules["src.integrations.webrtc"]
            
            from src.integrations.webrtc import WebRTCSignalingHandler
            return WebRTCSignalingHandler(mock_pipeline, mock_config)

    @pytest.mark.asyncio
    async def test_handle_connection_flow(self, mock_pipeline, mock_config):
        """Test full signaling flow: ready -> offer -> answer."""
        handler = await self.get_handler(mock_pipeline, mock_config)
        mock_ws = AsyncMock(spec=WebSocket)
        
        offer_msg = {"type": "offer", "sdp": "dummy_sdp"}
        hangup_msg = {"type": "hangup"}
        mock_ws.receive_json.side_effect = [offer_msg, hangup_msg]
        
        await handler.handle_connection(mock_ws)
        
        # Verification
        assert mock_ws.accept.called
        assert mock_ws.send_json.call_count >= 2
        
        self.mock_pc_instance.close.assert_awaited()

    @pytest.mark.asyncio
    async def test_handle_disconnect(self, mock_pipeline, mock_config):
        """Test cleanup on WebSocket disconnect."""
        handler = await self.get_handler(mock_pipeline, mock_config)
        mock_ws = AsyncMock(spec=WebSocket)
        mock_ws.receive_json.side_effect = WebSocketDisconnect()
        
        await handler.handle_connection(mock_ws)
        
        self.mock_pc_instance.close.assert_awaited()

    @pytest.mark.asyncio
    async def test_ice_candidate_handling(self, mock_pipeline, mock_config):
        """Test receiving ICE candidates."""
        handler = await self.get_handler(mock_pipeline, mock_config)
        mock_ws = AsyncMock(spec=WebSocket)
        mock_ws.receive_json.side_effect = [
            {"type": "ice_candidate", "candidate": "foo"},
            {"type": "hangup"}
        ]
        
        await handler.handle_connection(mock_ws)
        
        assert mock_ws.accept.called

    @pytest.mark.asyncio
    async def test_unknown_message(self, mock_pipeline, mock_config):
        """Test handling unknown message types."""
        handler = await self.get_handler(mock_pipeline, mock_config)
        mock_ws = AsyncMock(spec=WebSocket)
        mock_ws.receive_json.side_effect = [
            {"type": "unknown_stuff"},
            {"type": "hangup"}
        ]
        
        await handler.handle_connection(mock_ws)

        assert mock_ws.accept.called
