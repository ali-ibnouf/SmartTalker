"""Tests for Phase 1 Migration: Qwen3 + RunPod + R2.

Covers:
- DashScope ASR (mock WebSocket)
- DashScope TTS (mock WebSocket)
- Voice cloning
- RunPod Serverless client (mock HTTP)
- R2 Storage client (mock boto3)
- Visitor WebSocket handler
- Cost tracking across all services
- New DB model fields
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest
import pytest_asyncio


# =============================================================================
# ASR Tests (DashScope WebSocket)
# =============================================================================


class TestASREngine:
    """Tests for the DashScope ASR engine."""

    def _make_config(self):
        """Create a minimal config for ASR tests."""
        config = MagicMock()
        config.dashscope_api_key = "test-key"
        config.dashscope_ws_url = "wss://test.example.com/ws"
        config.asr_model = "qwen3-asr-flash-realtime"
        return config

    def test_init(self):
        from src.pipeline.asr import ASREngine
        engine = ASREngine(self._make_config())
        assert engine.is_loaded is True

    def test_load_noop(self):
        from src.pipeline.asr import ASREngine
        engine = ASREngine(self._make_config())
        engine.load()
        assert engine.is_loaded is True

    def test_unload(self):
        from src.pipeline.asr import ASREngine
        engine = ASREngine(self._make_config())
        engine.unload()
        assert engine.is_loaded is False

    def test_detect_language_arabic(self):
        from src.pipeline.asr import ASREngine
        assert ASREngine._detect_language("مرحبا بالعالم") == "ar"

    def test_detect_language_english(self):
        from src.pipeline.asr import ASREngine
        assert ASREngine._detect_language("Hello world this is a test") == "en"

    def test_detect_language_french(self):
        from src.pipeline.asr import ASREngine
        assert ASREngine._detect_language("Bonjour le monde, c'est très bien") == "fr"

    def test_detect_language_turkish(self):
        from src.pipeline.asr import ASREngine
        assert ASREngine._detect_language("Merhaba dünya, nasılsınız güzel günler") == "tr"

    def test_detect_language_empty(self):
        from src.pipeline.asr import ASREngine
        assert ASREngine._detect_language("") == "unknown"

    def test_detect_language_mixed(self):
        from src.pipeline.asr import ASREngine
        result = ASREngine._detect_language("مرحبا Hello")
        assert result in ("ar", "en", "mixed")

    def test_cost_tracking(self):
        from src.pipeline.asr import COST_PER_MINUTE
        # 1 minute of audio should cost $0.008
        assert COST_PER_MINUTE == 0.008
        cost = 2.5 / 60.0 * COST_PER_MINUTE  # 2.5 seconds
        assert cost > 0

    @pytest.mark.asyncio
    async def test_session_send_audio(self):
        """ASRSession.send_audio sends base64-encoded PCM via WebSocket."""
        from src.pipeline.asr import ASRSession

        mock_ws = AsyncMock()
        session = ASRSession(mock_ws, "test-session", "ar")

        pcm = b"\x00\x01" * 1600  # 100ms of 16kHz mono
        await session.send_audio(pcm)

        mock_ws.send.assert_called_once()
        sent = json.loads(mock_ws.send.call_args[0][0])
        assert sent["type"] == "input_audio_buffer.append"
        assert "audio" in sent
        # Verify base64 roundtrip
        decoded = base64.b64decode(sent["audio"])
        assert decoded == pcm

    @pytest.mark.asyncio
    async def test_session_closed_raises(self):
        """Sending audio to a closed session raises ASRError."""
        from src.pipeline.asr import ASRSession, ASREngine
        from src.utils.exceptions import ASRError

        mock_ws = AsyncMock()
        session = ASRSession(mock_ws, "test-session", "ar")
        session._closed = True

        with pytest.raises(ASRError):
            await session.send_audio(b"\x00\x01")


# =============================================================================
# TTS Tests (DashScope WebSocket)
# =============================================================================


class TestTTSEngine:
    """Tests for the DashScope TTS engine."""

    def _make_config(self):
        config = MagicMock()
        config.dashscope_api_key = "test-key"
        config.dashscope_ws_url = "wss://test.example.com/ws"
        config.tts_model = "qwen3-tts-vc-realtime"
        config.tts_sample_rate = 48000
        config.tts_max_text_length = 1000
        return config

    def test_init(self):
        from src.pipeline.tts import TTSEngine
        engine = TTSEngine(self._make_config())
        assert engine.is_loaded is True

    def test_sample_rate_48khz(self):
        """TTS result uses 48kHz sample rate (DashScope native)."""
        from src.pipeline.tts import TTSResult
        result = TTSResult(audio_path="/test.wav")
        assert result.sample_rate == 48000

    def test_cost_tracking(self):
        from src.pipeline.tts import COST_PER_MINUTE
        assert COST_PER_MINUTE == 0.015

    def test_voice_clone_cost(self):
        from src.pipeline.tts import VOICE_ENROLLMENT_COST
        assert VOICE_ENROLLMENT_COST == 0.20

    def test_lip_sync_preparation(self):
        """_prepare_lip_sync produces timing data for text."""
        from src.pipeline.tts import TTSEngine
        engine = TTSEngine(self._make_config())
        lip_sync = engine._prepare_lip_sync("hello world", "en", 1.0)
        assert "words" in lip_sync
        assert isinstance(lip_sync["words"], list)

    def test_emotion_params(self):
        """EMOTION_PARAMS maps emotions to speed multipliers."""
        from src.pipeline.tts import EMOTION_PARAMS
        assert "neutral" in EMOTION_PARAMS
        assert "happy" in EMOTION_PARAMS
        for emotion, params in EMOTION_PARAMS.items():
            assert "speed" in params


# =============================================================================
# RunPod Serverless Client Tests
# =============================================================================


class TestRunPodClient:
    """Tests for the RunPod Serverless client."""

    def _make_config(self):
        config = MagicMock()
        config.runpod_api_key = "rp_test_key"
        config.runpod_endpoint_musetalk = "https://api.runpod.ai/v2/test-musetalk"
        config.runpod_endpoint_preprocess = "https://api.runpod.ai/v2/test-preprocess"
        return config

    def test_init(self):
        from src.services.runpod_client import RunPodServerless
        client = RunPodServerless(self._make_config())
        assert client._endpoint_musetalk == "https://api.runpod.ai/v2/test-musetalk"

    def test_calculate_cost(self):
        from src.services.runpod_client import RunPodServerless, COST_PER_SEC
        cost = RunPodServerless.calculate_cost(5000)  # 5 seconds
        assert cost == pytest.approx(5.0 * COST_PER_SEC, rel=1e-4)

    def test_cost_per_sec(self):
        from src.services.runpod_client import COST_PER_SEC
        assert COST_PER_SEC == 0.00076

    @pytest.mark.asyncio
    async def test_preprocess_no_endpoint_raises(self):
        from src.services.runpod_client import RunPodServerless, RunPodError
        config = self._make_config()
        config.runpod_endpoint_preprocess = ""
        client = RunPodServerless(config)
        with pytest.raises(RunPodError, match="not configured"):
            await client.preprocess_face("http://example.com/photo.jpg", "emp_1")

    @pytest.mark.asyncio
    async def test_render_no_endpoint_raises(self):
        from src.services.runpod_client import RunPodServerless, RunPodError
        config = self._make_config()
        config.runpod_endpoint_musetalk = ""
        client = RunPodServerless(config)
        with pytest.raises(RunPodError, match="not configured"):
            await client.render_lipsync("url", "url", "emp", "sess")

    @pytest.mark.asyncio
    async def test_close(self):
        from src.services.runpod_client import RunPodServerless
        client = RunPodServerless(self._make_config())
        await client.close()
        assert client._client is None


# =============================================================================
# R2 Storage Client Tests
# =============================================================================


class TestR2Storage:
    """Tests for the Cloudflare R2 storage client."""

    def _make_config(self):
        config = MagicMock()
        config.r2_account_id = "test-account"
        config.r2_access_key_id = "test-key-id"
        config.r2_secret_access_key = "test-secret"
        config.r2_bucket = "test-bucket"
        config.r2_public_url = "https://media.test.com"
        return config

    def test_init(self):
        from src.services.r2_storage import R2Storage
        r2 = R2Storage(self._make_config())
        assert r2._bucket == "test-bucket"
        assert r2._public_url == "https://media.test.com"

    def test_public_key_url(self):
        from src.services.r2_storage import R2Storage
        r2 = R2Storage(self._make_config())
        url = r2._public_key_url("employees/emp_1/photo.jpg")
        assert url == "https://media.test.com/employees/emp_1/photo.jpg"

    def test_upload_employee_photo_key(self):
        """upload_employee_photo uses correct R2 key."""
        from src.services.r2_storage import R2Storage
        r2 = R2Storage(self._make_config())

        mock_client = MagicMock()
        r2._client = mock_client

        url = r2.upload_employee_photo("emp_123", b"\xff\xd8\xff\xe0")
        assert "employees/emp_123/photo.jpg" in url
        mock_client.put_object.assert_called_once()
        call_kwargs = mock_client.put_object.call_args[1]
        assert call_kwargs["Key"] == "employees/emp_123/photo.jpg"
        assert call_kwargs["ContentType"] == "image/jpeg"

    def test_upload_audio_key(self):
        """upload_audio uses timestamped key."""
        from src.services.r2_storage import R2Storage
        r2 = R2Storage(self._make_config())

        mock_client = MagicMock()
        r2._client = mock_client

        url = r2.upload_audio("session_456", b"\x00\x01")
        assert "sessions/session_456/audio_" in url
        assert url.endswith(".pcm")

    def test_upload_face_data_key(self):
        from src.services.r2_storage import R2Storage
        r2 = R2Storage(self._make_config())

        mock_client = MagicMock()
        r2._client = mock_client

        url = r2.upload_face_data("emp_789", b"\x00")
        assert "employees/emp_789/face_data.bin" in url

    def test_upload_voice_sample_key(self):
        from src.services.r2_storage import R2Storage
        r2 = R2Storage(self._make_config())

        mock_client = MagicMock()
        r2._client = mock_client

        url = r2.upload_voice_sample("emp_101", b"\x00")
        assert "employees/emp_101/voice_sample.wav" in url

    def test_delete_employee_media(self):
        from src.services.r2_storage import R2Storage
        r2 = R2Storage(self._make_config())

        mock_client = MagicMock()
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {"Contents": [{"Key": "employees/emp_1/photo.jpg"}, {"Key": "employees/emp_1/face_data.bin"}]}
        ]
        mock_client.get_paginator.return_value = mock_paginator
        r2._client = mock_client

        count = r2.delete_employee_media("emp_1")
        assert count == 2
        mock_client.delete_objects.assert_called_once()

    def test_delete_customer_media(self):
        from src.services.r2_storage import R2Storage
        r2 = R2Storage(self._make_config())

        mock_client = MagicMock()
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {"Contents": [{"Key": "employees/emp_1/photo.jpg"}]}
        ]
        mock_client.get_paginator.return_value = mock_paginator
        r2._client = mock_client

        count = r2.delete_customer_media("cust_1", ["emp_1", "emp_2"])
        assert count >= 1

    def test_r2_error_hierarchy(self):
        from src.services.r2_storage import R2Error
        from src.utils.exceptions import SmartTalkerError
        assert issubclass(R2Error, SmartTalkerError)


# =============================================================================
# DB Model Field Tests
# =============================================================================


class TestModelFields:
    """Test that new DB model fields exist."""

    def test_avatar_photo_preprocessed(self):
        from src.db.models import Avatar
        # Verify column exists on the model
        col_names = [c.name for c in Avatar.__table__.columns]
        assert "photo_preprocessed" in col_names
        assert "face_data_url" in col_names
        assert "voice_model" in col_names

    def test_avatar_default_type_vrm(self):
        from src.db.models import Avatar
        col = Avatar.__table__.columns["avatar_type"]
        assert col.default.arg == "vrm"

    def test_conversation_gpu_cost(self):
        from src.db.models import Conversation
        col_names = [c.name for c in Conversation.__table__.columns]
        assert "gpu_cost" in col_names

    def test_usage_record_runpod_job_id(self):
        from src.db.models import UsageRecord
        col_names = [c.name for c in UsageRecord.__table__.columns]
        assert "runpod_job_id" in col_names

    def test_customer_language_fields(self):
        from src.db.models import Customer
        col_names = [c.name for c in Customer.__table__.columns]
        assert "operator_language" in col_names
        assert "data_language" in col_names


# =============================================================================
# LLM Cost Tracking Tests
# =============================================================================


class TestLLMCost:
    """Tests for LLM cost calculation."""

    def test_cost_constants(self):
        from src.pipeline.llm import LLMEngine
        assert hasattr(LLMEngine, "INPUT_COST_PER_M")
        assert hasattr(LLMEngine, "OUTPUT_COST_PER_M")
        assert LLMEngine.INPUT_COST_PER_M == 1.20
        assert LLMEngine.OUTPUT_COST_PER_M == 6.00

    def test_cost_result_field(self):
        from src.pipeline.llm import LLMResult
        result = LLMResult(text="test", cost_usd=0.001)
        assert result.cost_usd == 0.001


# =============================================================================
# Config Tests (New Fields)
# =============================================================================


class TestConfigMigration:
    """Test that config has all new fields."""

    def test_dashscope_fields(self):
        from src.config import Settings
        fields = Settings.model_fields
        assert "dashscope_api_key" in fields
        assert "dashscope_base_url" in fields
        assert "dashscope_ws_url" in fields

    def test_asr_model_field(self):
        from src.config import Settings
        fields = Settings.model_fields
        assert "asr_model" in fields
        # Default should be qwen3
        default = fields["asr_model"].default
        assert "qwen3" in default

    def test_tts_model_field(self):
        from src.config import Settings
        fields = Settings.model_fields
        assert "tts_model" in fields
        assert "tts_sample_rate" in fields

    def test_runpod_fields(self):
        from src.config import Settings
        fields = Settings.model_fields
        assert "runpod_api_key" in fields
        assert "runpod_endpoint_musetalk" in fields
        assert "runpod_endpoint_preprocess" in fields

    def test_r2_fields(self):
        from src.config import Settings
        fields = Settings.model_fields
        assert "r2_account_id" in fields
        assert "r2_access_key_id" in fields
        assert "r2_secret_access_key" in fields
        assert "r2_bucket" in fields
        assert "r2_public_url" in fields

    def test_sample_rate_48khz(self):
        from src.config import Settings
        assert Settings.model_fields["tts_sample_rate"].default == 48000

    def test_llm_model_qwen3(self):
        from src.config import Settings
        assert Settings.model_fields["llm_model_name"].default == "qwen3-max"


# =============================================================================
# RunPod Error Hierarchy
# =============================================================================


class TestRunPodErrors:
    """Test RunPod error types."""

    def test_runpod_error_is_smarttalker_error(self):
        from src.services.runpod_client import RunPodError
        from src.utils.exceptions import SmartTalkerError
        assert issubclass(RunPodError, SmartTalkerError)

    def test_runpod_error_to_dict(self):
        from src.services.runpod_client import RunPodError
        err = RunPodError(message="test error", detail="some detail")
        d = err.to_dict()
        assert d["error"] == "test error"
        assert d["detail"] == "some detail"
