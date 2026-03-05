"""Tests for VRM WebSocket streaming flow."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from starlette.websockets import WebSocketState

from src.api.websocket import SessionConfig, Session, WebSocketManager


def _make_mock_ws():
    """Create a mock WebSocket with CONNECTED state so _send_json works."""
    mock_ws = AsyncMock()
    mock_ws.client_state = WebSocketState.CONNECTED
    return mock_ws


class TestSessionConfigVRM:
    """Tests for avatar_type in SessionConfig."""

    def test_default_avatar_type(self):
        """Default avatar_type is video."""
        config = SessionConfig()
        assert config.avatar_type == "video"

    def test_vrm_avatar_type(self):
        """Can set avatar_type to vrm."""
        config = SessionConfig(avatar_type="vrm")
        assert config.avatar_type == "vrm"


class TestVRMConfigUpdate:
    """Tests for avatar_type in config update handler."""

    @pytest.mark.asyncio
    async def test_config_update_accepts_avatar_type(self):
        """Config update handler accepts avatar_type field."""
        mock_pipeline = MagicMock()
        manager = WebSocketManager(mock_pipeline, storage_dir=MagicMock())

        mock_ws = _make_mock_ws()
        session = Session(session_id="test_session", websocket=mock_ws)

        await manager._handle_config_update(session, {"avatar_type": "vrm"})

        assert session.config.avatar_type == "vrm"
        # Verify ack was sent with avatar_type
        mock_ws.send_json.assert_called()
        ack_data = mock_ws.send_json.call_args[0][0]
        assert ack_data["type"] == "config_ack"
        assert ack_data["avatar_type"] == "vrm"

    @pytest.mark.asyncio
    async def test_config_update_rejects_invalid_avatar_type(self):
        """Config update ignores invalid avatar_type values."""
        mock_pipeline = MagicMock()
        manager = WebSocketManager(mock_pipeline, storage_dir=MagicMock())

        mock_ws = _make_mock_ws()
        session = Session(session_id="test_session", websocket=mock_ws)

        await manager._handle_config_update(session, {"avatar_type": "invalid"})

        # Should remain default "video"
        assert session.config.avatar_type == "video"


class TestVRMResponseFlow:
    """Tests for _send_vrm_response method."""

    @pytest.mark.asyncio
    async def test_send_vrm_response_no_audio(self):
        """VRM response with no audio sends only vrm_audio_end."""
        mock_pipeline = MagicMock()
        manager = WebSocketManager(mock_pipeline, storage_dir=MagicMock())

        mock_ws = _make_mock_ws()
        session = Session(session_id="test", websocket=mock_ws)

        result = MagicMock()
        result.audio_path = None

        await manager._send_vrm_response(session, result)

        calls = mock_ws.send_json.call_args_list
        assert any(c[0][0].get("type") == "vrm_audio_end" for c in calls)

    @pytest.mark.asyncio
    async def test_send_vrm_response_with_audio(self, tmp_path):
        """VRM response with audio sends vrm_audio chunks + vrm_audio_end."""
        mock_pipeline = MagicMock()
        manager = WebSocketManager(mock_pipeline, storage_dir=MagicMock())

        mock_ws = _make_mock_ws()
        session = Session(session_id="test", websocket=mock_ws)

        # 0.4s of silence = 22050 * 2 * 0.4 = 17640 bytes
        audio_data = b"\x00" * 17640
        audio_path = tmp_path / "test_audio.raw"
        audio_path.write_bytes(audio_data)

        result = MagicMock()
        result.audio_path = str(audio_path)
        result.response_text = "Hello world"
        result.detected_emotion = "neutral"
        result.lip_sync = {"words": [
            {"word": "Hello", "start": 0.0, "end": 0.2},
            {"word": "world", "start": 0.2, "end": 0.4},
        ]}

        await manager._send_vrm_response(session, result)

        calls = mock_ws.send_json.call_args_list
        msg_types = [c[0][0].get("type") for c in calls]

        assert "vrm_audio" in msg_types
        assert msg_types[-1] == "vrm_audio_end"

        audio_msgs = [c[0][0] for c in calls if c[0][0].get("type") == "vrm_audio"]
        assert len(audio_msgs) >= 1

        first = audio_msgs[0]
        assert first["seq"] == 0
        assert "audio_b64" in first
        assert "duration_ms" in first
        assert "visemes" in first
        assert isinstance(first["visemes"], list)
        assert first["emotion"] == "neutral"

    @pytest.mark.asyncio
    async def test_send_vrm_response_with_wav_header(self, tmp_path):
        """VRM response correctly strips WAV header from audio."""
        mock_pipeline = MagicMock()
        manager = WebSocketManager(mock_pipeline, storage_dir=MagicMock())

        mock_ws = _make_mock_ws()
        session = Session(session_id="test", websocket=mock_ws)

        wav_header = b"RIFF" + b"\x00" * 40  # 44 bytes
        pcm_data = b"\x00" * 8820  # 0.2s of audio
        audio_path = tmp_path / "test.wav"
        audio_path.write_bytes(wav_header + pcm_data)

        result = MagicMock()
        result.audio_path = str(audio_path)
        result.response_text = "Test"
        result.detected_emotion = "neutral"
        result.lip_sync = None

        await manager._send_vrm_response(session, result)

        calls = mock_ws.send_json.call_args_list
        audio_msgs = [c[0][0] for c in calls if c[0][0].get("type") == "vrm_audio"]
        assert len(audio_msgs) >= 1

    @pytest.mark.asyncio
    async def test_text_chat_vrm_mode_branches(self, tmp_path):
        """text_chat with avatar_type=vrm calls _send_vrm_response not body_state/voice."""
        mock_pipeline = MagicMock()
        mock_result = MagicMock()
        mock_result.audio_path = None
        mock_result.response_text = "Hello"
        mock_result.detected_emotion = "neutral"
        mock_result.body_state = "idle"
        mock_result.total_latency_ms = 100
        mock_result.breakdown = {}
        mock_result.kb_confidence = 0.9
        mock_result.kb_sources = []
        mock_result.escalated = False
        mock_result.escalation_id = None
        mock_result.lip_sync = None

        mock_pipeline.process_text = AsyncMock(return_value=mock_result)
        manager = WebSocketManager(mock_pipeline, storage_dir=tmp_path)

        mock_ws = _make_mock_ws()
        session = Session(session_id="test", websocket=mock_ws)
        session.config.avatar_type = "vrm"
        session.config.training_mode = "digital"

        await manager._handle_text_chat(session, {"text": "Hello"})

        calls = mock_ws.send_json.call_args_list
        msg_types = [c[0][0].get("type") for c in calls]

        # Should NOT have body_state or voice messages
        assert "body_state" not in msg_types
        assert "voice" not in msg_types
        # Should have vrm_audio_end (from _send_vrm_response with no audio)
        assert "vrm_audio_end" in msg_types
        # Should still have response message
        assert "response" in msg_types

    @pytest.mark.asyncio
    async def test_viseme_data_has_vrm_names(self, tmp_path):
        """Visemes in vrm_audio messages use VRM blend shape names."""
        mock_pipeline = MagicMock()
        manager = WebSocketManager(mock_pipeline, storage_dir=MagicMock())

        mock_ws = _make_mock_ws()
        session = Session(session_id="test", websocket=mock_ws)

        # 0.3s of audio
        audio_data = b"\x01\x00" * (22050 * 3 // 10)
        audio_path = tmp_path / "test.raw"
        audio_path.write_bytes(audio_data)

        result = MagicMock()
        result.audio_path = str(audio_path)
        result.response_text = "hello"
        result.detected_emotion = "happy"
        result.lip_sync = {"words": [{"word": "hello", "start": 0.0, "end": 0.3}]}

        await manager._send_vrm_response(session, result)

        calls = mock_ws.send_json.call_args_list
        audio_msgs = [c[0][0] for c in calls if c[0][0].get("type") == "vrm_audio"]
        assert len(audio_msgs) >= 1

        valid_vrm = {"neutral", "aa", "ih", "ou", "ee", "oh"}
        for msg in audio_msgs:
            for v in msg["visemes"]:
                assert v["viseme"] in valid_vrm, f"Invalid VRM viseme: {v['viseme']}"
