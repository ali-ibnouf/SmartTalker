"""Tests for WhatsApp integration."""

import hashlib
import hmac
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.integrations.whatsapp import WhatsAppClient

@pytest.fixture
def whatsapp_client(config):
    """Create a WhatsAppClient instance."""
    return WhatsAppClient(config)

class TestWhatsAppClient:
    """Tests for WhatsAppClient."""

    def test_init(self, whatsapp_client, config):
        """Client initializes with correct config."""
        assert whatsapp_client._verify_token == (config.whatsapp_verify_token or "")
        assert whatsapp_client._access_token == (config.whatsapp_access_token or "")

    def test_verify_webhook_valid(self, whatsapp_client):
        """Webhook verification succeeds with correct token."""
        mode = "subscribe"
        token = whatsapp_client._verify_token
        challenge = "12345"
        
        response = whatsapp_client.verify_webhook(mode, token, challenge)
        assert response == challenge

    def test_verify_webhook_invalid_token(self, whatsapp_client):
        """Webhook verification returns None with incorrect token."""
        mode = "subscribe"
        token = "wrong_token"
        challenge = "12345"
        
        response = whatsapp_client.verify_webhook(mode, token, challenge)
        assert response is None

    def test_verify_signature_valid(self, whatsapp_client):
        """Signature verification succeeds for valid payload."""
        whatsapp_client._app_secret = "test_secret"
        payload = b'{"object":"whatsapp_business_account"}'
        secret = whatsapp_client._app_secret.encode()
        signature = "sha256=" + hmac.new(secret, payload, hashlib.sha256).hexdigest()
        
        # Test valid
        assert whatsapp_client.verify_signature(payload, signature) is True

    def test_verify_signature_invalid(self, whatsapp_client):
        """Signature verification returns False for invalid signature."""
        whatsapp_client._app_secret = "test_secret"
        payload = b"{}"
        signature = "sha256=fake_signature"
        
        assert whatsapp_client.verify_signature(payload, signature) is False
    
    def test_deduplication(self, whatsapp_client):
        """Message deduplication works correctly."""
        msg_id = "wamid.HBgLM..."
        assert whatsapp_client.is_duplicate(msg_id) is False  # First time (not duplicate)
        assert whatsapp_client.is_duplicate(msg_id) is True   # Second time (duplicate)

    @pytest.mark.asyncio
    async def test_send_message_retry(self, whatsapp_client):
        """Send message retries on failure."""
        import httpx
        with patch("src.integrations.whatsapp.WhatsAppClient._get_client", new_callable=AsyncMock) as mock_get_client:
            mock_client = MagicMock()
            mock_client.post = AsyncMock()
            
            # Create a proper response mock
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"id": "123"}
            
            # Side effect: first call raises retriable error, second returns proper response
            mock_client.post.side_effect = [httpx.ConnectError("Fail"), mock_response]
            mock_get_client.return_value = mock_client
            
            with patch("asyncio.sleep", new_callable=AsyncMock):
                await whatsapp_client.send_text("1234567890", "Hello")
            
            assert mock_client.post.call_count == 2

from src.api.routes import _build_public_audio_url, _process_whatsapp_payload
from pathlib import Path

class TestAudioUrlBuilder:
    """Tests for audio URL construction."""

    def test_build_public_audio_url_with_webhook_url(self):
        """Constructs correct URL when webhook_url is configured."""
        app = MagicMock()
        app.state.config.static_files_dir = Path("/tmp/static")
        app.state.config.whatsapp_webhook_url = "https://example.com/api/v1/whatsapp/webhook"
        
        # Mock FS operations
        with patch("pathlib.Path.mkdir"), patch("shutil.copy2"):
            # Use real Path logic (no need to mock .name)
            url = _build_public_audio_url("/tmp/audio.wav", app)
            assert url == "https://example.com/files/audio.wav"

    def test_build_public_audio_url_fallback(self):
        """Falls back to relative URL when webhook_url is missing."""
        app = MagicMock()
        app.state.config.static_files_dir = Path("/tmp/static")
        app.state.config.whatsapp_webhook_url = None
        
        with patch("pathlib.Path.mkdir"), patch("shutil.copy2"):
            url = _build_public_audio_url("/tmp/audio.wav", app)
            assert url == "/files/audio.wav"

@pytest.mark.asyncio
async def test_process_payload_sends_audio():
    """Background task sends audio if generated."""
    app = MagicMock()
    app.state.whatsapp = MagicMock()
    app.state.whatsapp.send_text = AsyncMock()
    app.state.whatsapp.send_audio = AsyncMock()
    app.state.whatsapp.mark_read = AsyncMock()
    
    app.state.pipeline = AsyncMock()
    
    # Mock pipeline result
    result = MagicMock()
    result.response_text = "Hello"
    result.audio_path = "/tmp/response.wav"
    app.state.pipeline.process_text.return_value = result
    
    # Mock payload processing
    payload = {"object": "whatsapp_business_account"}
    app.state.whatsapp.parse_incoming.return_value = [{
        "type": "text", 
        "text": "Hi", 
        "from_number": "123", 
        "message_id": "msg1"
    }]
    app.state.whatsapp.is_duplicate.return_value = False
    
    # Mock URL builder
    with patch("src.api.routes._build_public_audio_url", return_value="https://url/file.wav"):
        await _process_whatsapp_payload(payload, app)
        
    app.state.whatsapp.send_text.assert_awaited_with("123", "Hello")
    app.state.whatsapp.send_audio.assert_awaited_with("123", "https://url/file.wav")

@pytest.mark.asyncio
async def test_process_payload_skips_audio_if_empty():
    """Background task does not send audio if path is empty."""
    app = MagicMock()
    app.state.whatsapp = MagicMock()
    app.state.whatsapp.send_text = AsyncMock()
    app.state.whatsapp.send_audio = AsyncMock()
    app.state.whatsapp.mark_read = AsyncMock()
    
    app.state.pipeline = AsyncMock()
    
    # Mock pipeline result with no audio
    result = MagicMock()
    result.response_text = "Hello"
    result.audio_path = ""
    app.state.pipeline.process_text.return_value = result
    
    payload = {"object": "whatsapp_business_account"}
    app.state.whatsapp.parse_incoming.return_value = [{
        "type": "text", 
        "text": "Hi", 
        "from_number": "123", 
        "message_id": "msg1"
    }]
    app.state.whatsapp.is_duplicate.return_value = False
    
    await _process_whatsapp_payload(payload, app)
        
    app.state.whatsapp.send_text.assert_awaited_with("123", "Hello")
    app.state.whatsapp.send_audio.assert_not_awaited()
