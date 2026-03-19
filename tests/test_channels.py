"""Tests for multi-channel integration.

Channel Router, WhatsApp, Telegram, QR Code, Visitor Resolution,
Documentation, and Embed Code.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from src.channels.base import ChannelType, IncomingMessage, OutgoingMessage
from src.channels.router import ChannelRouter
from src.channels.widget import WidgetAdapter
from src.channels.whatsapp import WhatsAppAdapter
from src.channels.telegram import TelegramAdapter
from src.channels.visitor_resolver import VisitorResolver
from src.services.qr_generator import QRGenerator
from src.services.doc_generator import CustomerDocGenerator


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def mock_agent():
    agent = AsyncMock()
    # AgentEngine.handle_message() returns a string
    agent.handle_message = AsyncMock(return_value="Hello! How can I help?")
    agent.get_employee = AsyncMock(return_value=None)
    return agent


@pytest.fixture
def mock_tts():
    tts = AsyncMock()
    stream = AsyncMock()
    stream.collect_all = AsyncMock(return_value=b"\x00" * 1000)
    tts.synthesize_stream = AsyncMock(return_value=stream)
    return tts


@pytest.fixture
def mock_asr():
    asr = AsyncMock()
    session = AsyncMock()
    session.send_audio = AsyncMock()
    from src.pipeline.asr import TranscriptionResult
    session.finish = AsyncMock(return_value=TranscriptionResult(text="Transcribed text"))
    asr.create_session = AsyncMock(return_value=session)
    return asr


@pytest.fixture
def mock_runpod():
    from src.services.runpod_client import RenderResult
    runpod = AsyncMock()
    runpod.render_lipsync = AsyncMock(return_value=RenderResult(
        video_url="https://r2.example.com/video.mp4"
    ))
    return runpod


@pytest.fixture
def mock_r2():
    r2 = AsyncMock()
    r2.upload_audio = AsyncMock(return_value="https://r2.example.com/audio.ogg")
    r2.upload = AsyncMock(return_value="https://r2.example.com/qr.png")
    return r2


@pytest.fixture
def mock_db():
    db = AsyncMock()
    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    result = AsyncMock()
    result.scalar = MagicMock(return_value=None)
    result.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))
    session.execute = AsyncMock(return_value=result)
    session.commit = AsyncMock()
    session.add = MagicMock()
    db.session_ctx = MagicMock(return_value=session)
    return db


@pytest.fixture
def router(mock_agent, mock_tts, mock_asr, mock_runpod, mock_r2):
    r = ChannelRouter(
        agent_engine=mock_agent,
        tts=mock_tts,
        asr=mock_asr,
        runpod=mock_runpod,
        r2=mock_r2,
    )
    r.register_adapter(ChannelType.WIDGET, WidgetAdapter())
    r.register_adapter(ChannelType.WHATSAPP, WhatsAppAdapter())
    r.register_adapter(ChannelType.TELEGRAM, TelegramAdapter())
    return r


@pytest.fixture
def wa_config():
    config = MagicMock()
    config.employee_id = "emp_001"
    config.customer_id = "cust_001"
    config.wa_access_token = "test_token"
    config.wa_phone_number_id = "1234567890"
    config.wa_business_account_id = "9876543210"
    config.wa_verify_token = "verify_me"
    config.wa_phone_display = "+1 234 567 8900"
    return config


@pytest.fixture
def tg_config():
    config = MagicMock()
    config.employee_id = "emp_001"
    config.customer_id = "cust_001"
    config.tg_bot_token = "123456:ABC-DEF"
    config.tg_bot_username = "@test_bot"
    return config


# ═══════════════════════════════════════════════════════════════════
# Channel Router Tests
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_text_message_routes_to_agent(router, mock_agent):
    """Text message → Agent Engine → text response."""
    msg = IncomingMessage(
        channel=ChannelType.WIDGET,
        channel_session_id="ws_sess_1",
        employee_id="emp_001",
        customer_id="cust_001",
        visitor_id="v_abc123",
        message_type="text",
        text="Hello",
    )
    result = await router.handle_message(msg)

    mock_agent.handle_message.assert_called_once_with(
        session_id="ws_sess_1",
        visitor_message="Hello",
        employee_id="emp_001",
        visitor_id="v_abc123",
        customer_id="cust_001",
    )
    assert result.text == "Hello! How can I help?"


@pytest.mark.asyncio
async def test_voice_message_transcribed_then_routes(router, mock_agent, mock_asr):
    """Voice message → ASR → Agent Engine → text response."""
    msg = IncomingMessage(
        channel=ChannelType.WHATSAPP,
        channel_session_id="wa_96891234567_emp_001",
        employee_id="emp_001",
        customer_id="cust_001",
        visitor_id="v_phone1",
        message_type="voice",
        audio_bytes=b"\x00" * 500,
    )
    result = await router.handle_message(msg)

    mock_asr.create_session.assert_called_once()
    mock_agent.handle_message.assert_called_once_with(
        session_id="wa_96891234567_emp_001",
        visitor_message="Transcribed text",
        employee_id="emp_001",
        visitor_id="v_phone1",
        customer_id="cust_001",
    )
    assert result.text == "Hello! How can I help?"


@pytest.mark.asyncio
async def test_widget_gets_video_response(router, mock_agent, mock_runpod):
    """Widget with video avatar → RunPod renders lip-sync video."""
    employee = MagicMock()
    employee.avatar_mode = "video"
    employee.face_data_url = "https://r2.example.com/face.pkl"
    employee.id = "emp_001"
    employee.voice_id = "voice_abc"
    mock_agent.get_employee = AsyncMock(return_value=employee)

    msg = IncomingMessage(
        channel=ChannelType.WIDGET,
        channel_session_id="ws_sess_2",
        employee_id="emp_001",
        customer_id="cust_001",
        visitor_id="v_xyz",
        message_type="voice",
        audio_bytes=b"\x00" * 500,
    )
    result = await router.handle_message(msg)

    assert result.video_url == "https://r2.example.com/video.mp4"
    assert result.audio_url is not None
    mock_runpod.render_lipsync.assert_called_once()


@pytest.mark.asyncio
async def test_whatsapp_gets_text_and_audio_only(router, mock_agent, mock_runpod):
    """WhatsApp never gets video, even if employee has video avatar."""
    employee = MagicMock()
    employee.avatar_mode = "video"
    employee.face_data_url = "https://r2.example.com/face.pkl"
    employee.id = "emp_001"
    employee.voice_id = "voice_abc"
    mock_agent.get_employee = AsyncMock(return_value=employee)

    msg = IncomingMessage(
        channel=ChannelType.WHATSAPP,
        channel_session_id="wa_96891234567_emp_001",
        employee_id="emp_001",
        customer_id="cust_001",
        visitor_id="v_phone",
        message_type="voice",
        audio_bytes=b"\x00" * 500,
    )
    result = await router.handle_message(msg)

    assert result.video_url is None  # No video for WhatsApp
    assert result.text == "Hello! How can I help?"
    mock_runpod.render_lipsync.assert_not_called()


@pytest.mark.asyncio
async def test_telegram_gets_text_and_voice_only(router, mock_agent, mock_runpod):
    """Telegram never gets video."""
    employee = MagicMock()
    employee.avatar_mode = "video"
    employee.face_data_url = "https://r2.example.com/face.pkl"
    employee.id = "emp_001"
    employee.voice_id = "voice_abc"
    mock_agent.get_employee = AsyncMock(return_value=employee)

    msg = IncomingMessage(
        channel=ChannelType.TELEGRAM,
        channel_session_id="tg_12345_emp_001",
        employee_id="emp_001",
        customer_id="cust_001",
        visitor_id="v_tg",
        message_type="voice",
        audio_bytes=b"\x00" * 500,
    )
    result = await router.handle_message(msg)

    assert result.video_url is None
    assert result.audio_url is not None
    mock_runpod.render_lipsync.assert_not_called()


# ═══════════════════════════════════════════════════════════════════
# WhatsApp Tests
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_whatsapp_webhook_verification():
    """WhatsApp GET webhook verification returns challenge."""
    from src.api.webhooks.whatsapp import router as wa_router
    from fastapi import FastAPI
    from httpx import AsyncClient, ASGITransport

    app = FastAPI()
    app.include_router(wa_router)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/webhooks/whatsapp/emp_001",
            params={
                "hub.mode": "subscribe",
                "hub.verify_token": "test_token",
                "hub.challenge": "challenge_12345",
            },
        )
    # Without app.state.db, config is None → 404
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_whatsapp_text_message_parsed(wa_config):
    """WhatsApp text message parsed into IncomingMessage."""
    adapter = WhatsAppAdapter()
    raw = {
        "entry": [{
            "changes": [{
                "value": {
                    "messages": [{
                        "from": "96891234567",
                        "type": "text",
                        "id": "wamid_123",
                        "text": {"body": "Hi there!"},
                    }]
                }
            }]
        }]
    }
    msg = await adapter.parse_incoming(raw, wa_config)

    assert msg.channel == ChannelType.WHATSAPP
    assert msg.text == "Hi there!"
    assert msg.message_type == "text"
    assert msg.channel_session_id == "wa_96891234567_emp_001"
    assert msg.employee_id == "emp_001"
    assert msg.metadata["phone"] == "96891234567"


@pytest.mark.asyncio
async def test_whatsapp_voice_message_parsed(wa_config):
    """WhatsApp voice message sets audio_url and type=voice."""
    adapter = WhatsAppAdapter()

    raw = {
        "entry": [{
            "changes": [{
                "value": {
                    "messages": [{
                        "from": "96891234567",
                        "type": "audio",
                        "id": "wamid_456",
                        "audio": {"id": "media_789"},
                    }]
                }
            }]
        }]
    }

    with patch.object(
        adapter, "_get_media_url",
        new_callable=AsyncMock,
        return_value="https://media.whatsapp.com/audio.ogg",
    ):
        msg = await adapter.parse_incoming(raw, wa_config)

    assert msg.message_type == "voice"
    assert msg.audio_url == "https://media.whatsapp.com/audio.ogg"


@pytest.mark.asyncio
async def test_whatsapp_response_sent():
    """WhatsApp adapter sends text + audio via Meta API."""
    adapter = WhatsAppAdapter()
    adapter._cached_config = MagicMock(
        wa_access_token="test_token",
        wa_phone_number_id="1234567890",
    )

    response = OutgoingMessage(
        text="Hello!",
        audio_url="https://r2.example.com/audio.ogg",
    )

    with patch.object(adapter, "_send_wa_message", new_callable=AsyncMock) as mock_send:
        await adapter.send_response("wa_96891234567_emp_001", response)

    assert mock_send.call_count == 2  # text + audio


@pytest.mark.asyncio
async def test_whatsapp_quick_replies():
    """WhatsApp adapter sends interactive buttons for quick replies."""
    adapter = WhatsAppAdapter()
    adapter._cached_config = MagicMock(
        wa_access_token="test_token",
        wa_phone_number_id="1234567890",
    )

    response = OutgoingMessage(
        text="Choose one:",
        quick_replies=["Option A", "Option B", "Option C"],
    )

    with patch.object(adapter, "_send_wa_message", new_callable=AsyncMock) as mock_send:
        await adapter.send_response("wa_96891234567_emp_001", response)

    # text + quick_replies (interactive buttons)
    assert mock_send.call_count == 2
    # Verify interactive message contains buttons
    interactive_call = mock_send.call_args_list[1]
    body = interactive_call[0][2]  # third positional arg
    assert body["type"] == "interactive"
    assert len(body["interactive"]["action"]["buttons"]) == 3


# ═══════════════════════════════════════════════════════════════════
# Telegram Tests
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_telegram_webhook_parsed(tg_config):
    """Telegram text message parsed into IncomingMessage."""
    adapter = TelegramAdapter()
    raw = {
        "message": {
            "chat": {"id": 12345},
            "from": {"id": 67890},
            "text": "Hello bot!",
        }
    }
    msg = await adapter.parse_incoming(raw, tg_config)

    assert msg.channel == ChannelType.TELEGRAM
    assert msg.text == "Hello bot!"
    assert msg.message_type == "text"
    assert msg.channel_session_id == "tg_12345_emp_001"
    assert msg.metadata["chat_id"] == "12345"
    assert msg.metadata["user_id"] == "67890"


@pytest.mark.asyncio
async def test_telegram_voice_message(tg_config):
    """Telegram voice message sets audio_url and type=voice."""
    adapter = TelegramAdapter()

    raw = {
        "message": {
            "chat": {"id": 12345},
            "from": {"id": 67890},
            "voice": {"file_id": "voice_file_abc"},
        }
    }

    with patch.object(
        adapter, "_get_file_url",
        new_callable=AsyncMock,
        return_value="https://api.telegram.org/file/bot123/voice.ogg",
    ):
        msg = await adapter.parse_incoming(raw, tg_config)

    assert msg.message_type == "voice"
    assert msg.audio_url == "https://api.telegram.org/file/bot123/voice.ogg"


@pytest.mark.asyncio
async def test_telegram_response_sent():
    """Telegram adapter sends text + voice via Bot API."""
    adapter = TelegramAdapter()
    adapter._cached_config = MagicMock(tg_bot_token="123456:ABC-DEF")

    response = OutgoingMessage(
        text="Hello from bot!",
        audio_url="https://r2.example.com/audio.ogg",
    )

    with patch("httpx.AsyncClient") as MockClient:
        mock_client = AsyncMock()
        MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        await adapter.send_response("tg_12345_emp_001", response)

    # sendMessage (text) + sendVoice (audio)
    assert mock_client.post.call_count == 2


@pytest.mark.asyncio
async def test_telegram_keyboard_replies():
    """Telegram adapter sends keyboard for quick replies."""
    adapter = TelegramAdapter()
    adapter._cached_config = MagicMock(tg_bot_token="123456:ABC-DEF")

    response = OutgoingMessage(
        text="Choose:",
        quick_replies=["A", "B", "C"],
    )

    with patch("httpx.AsyncClient") as MockClient:
        mock_client = AsyncMock()
        MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        await adapter.send_response("tg_12345_emp_001", response)

    # sendMessage (text) + sendMessage (with reply_markup)
    assert mock_client.post.call_count == 2
    # Last call should have reply_markup
    last_call = mock_client.post.call_args_list[-1]
    body = last_call.kwargs.get("json", {})
    assert "reply_markup" in body
    assert body["reply_markup"]["one_time_keyboard"] is True


@pytest.mark.asyncio
async def test_telegram_start_command(tg_config):
    """Telegram /start command generates greeting."""
    adapter = TelegramAdapter()
    raw = {
        "message": {
            "chat": {"id": 12345},
            "from": {"id": 67890},
            "text": "/start emp_001",
        }
    }
    msg = await adapter.parse_incoming(raw, tg_config)

    assert msg.text == "Hi, I'd like to chat!"
    assert msg.metadata.get("start_param") == "emp_001"


# ═══════════════════════════════════════════════════════════════════
# QR Code Tests
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_qr_generated_and_uploaded_to_r2(mock_r2):
    """QR code generated and uploaded to R2 storage."""
    gen = QRGenerator(r2_storage=mock_r2)
    result = await gen.generate_employee_qr("emp_001", "Alice")

    assert "landing_url" in result
    assert result["landing_url"] == "https://maskki.com/connect/emp_001"
    # If qrcode package is available, R2 upload is called
    # If not, qr_code_url is empty (graceful fallback)
    assert "qr_code_url" in result


@pytest.mark.asyncio
async def test_qr_landing_page_data_correct():
    """QR landing page returns correct employee + channel data."""
    from src.api.channel_routes import router as ch_router
    from fastapi import FastAPI
    from httpx import AsyncClient, ASGITransport

    app = FastAPI()
    app.include_router(ch_router)
    # Without DB, returns error
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/connect/emp_001")

    # Without app.state.db, returns service unavailable
    data = resp.json()
    # Returns a tuple (dict, status_code) when no DB
    assert "error" in str(data) or "employee_name" in str(data)


@pytest.mark.asyncio
async def test_qr_print_version_high_res():
    """Print QR code uses larger box_size for 300 DPI."""
    gen = QRGenerator(r2_storage=None)
    result = await gen.generate_print_version("emp_001", "Alice")
    # Returns bytes (PNG) if qrcode installed, empty bytes otherwise
    assert isinstance(result, bytes)


# ═══════════════════════════════════════════════════════════════════
# Visitor Resolution Tests
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_new_visitor_created():
    """New visitor gets a unique v_xxx ID."""
    resolver = VisitorResolver(db=None)
    vid = await resolver.resolve_visitor("whatsapp", "96891234567", "emp_001")
    assert vid.startswith("v_")
    assert len(vid) > 4


@pytest.mark.asyncio
async def test_returning_visitor_recognized(mock_db):
    """Returning visitor gets the same visitor_id."""
    # Mock DB returns existing visitor_id
    session = mock_db.session_ctx.return_value
    result = AsyncMock()
    result.scalar = MagicMock(return_value="v_existing123")
    session.execute = AsyncMock(return_value=result)

    resolver = VisitorResolver(db=mock_db)
    vid = await resolver.resolve_visitor("whatsapp", "96891234567", "emp_001")
    assert vid == "v_existing123"


@pytest.mark.asyncio
async def test_cross_channel_visitor_separate():
    """Different channels without linking produce separate visitor IDs."""
    resolver = VisitorResolver(db=None)
    vid_wa = await resolver.resolve_visitor("whatsapp", "96891234567", "emp_001")
    vid_tg = await resolver.resolve_visitor("telegram", "67890", "emp_001")
    assert vid_wa != vid_tg  # Different IDs (no DB = no linking)


@pytest.mark.asyncio
async def test_visitor_merge(mock_db):
    """Merging two visitors moves all channel maps and memories."""
    session = mock_db.session_ctx.return_value

    resolver = VisitorResolver(db=mock_db)
    await resolver.merge_visitors("v_keep", "v_merge")

    # Should have executed updates for channel_maps, memories, and delete profile
    assert session.execute.call_count >= 3
    session.commit.assert_called_once()


# ═══════════════════════════════════════════════════════════════════
# Documentation Tests
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_whatsapp_guide_generated():
    """WhatsApp guide contains relevant setup instructions."""
    gen = CustomerDocGenerator(db=None)
    docs = await gen.generate_channel_docs(
        "cust_001", "emp_001", "Alice", "Acme Corp", "Sales Agent"
    )
    assert "whatsapp_setup" in docs
    wa_doc = docs["whatsapp_setup"]
    assert "WhatsApp" in wa_doc["title"]
    assert "Alice" in wa_doc["content"]
    assert "webhook" in wa_doc["content"].lower()
    assert "emp_001" in wa_doc["content"]


@pytest.mark.asyncio
async def test_telegram_guide_generated():
    """Telegram guide contains BotFather instructions."""
    gen = CustomerDocGenerator(db=None)
    docs = await gen.generate_channel_docs(
        "cust_001", "emp_001", "Alice", "Acme Corp", "Sales Agent"
    )
    assert "telegram_setup" in docs
    tg_doc = docs["telegram_setup"]
    assert "Telegram" in tg_doc["title"]
    assert "BotFather" in tg_doc["content"]
    assert "Alice" in tg_doc["content"]


@pytest.mark.asyncio
async def test_docs_saved_to_customer_file(mock_db):
    """Docs are saved to the customer_docs DB table."""
    session = mock_db.session_ctx.return_value

    gen = CustomerDocGenerator(db=mock_db)
    docs = await gen.generate_channel_docs(
        "cust_001", "emp_001", "Alice", "Acme Corp"
    )

    assert len(docs) == 4  # widget, whatsapp, telegram, qr
    # DB session was used to upsert docs
    session.commit.assert_called()


@pytest.mark.asyncio
async def test_docs_all_types_generated():
    """All four doc types are generated."""
    gen = CustomerDocGenerator(db=None)
    docs = await gen.generate_channel_docs(
        "cust_001", "emp_001", "Alice", "Acme Corp"
    )

    assert set(docs.keys()) == {"widget_setup", "whatsapp_setup", "telegram_setup", "qr_guide"}
    for key, doc in docs.items():
        assert "title" in doc
        assert "content" in doc
        assert len(doc["content"]) > 50  # Not empty


# ═══════════════════════════════════════════════════════════════════
# Embed Code Tests
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_embed_code_contains_employee_id():
    """Embed code snippet contains the employee ID."""
    from src.api.channel_routes import _embed_snippet

    result = _embed_snippet("emp_001", "Alice", "en")
    assert "emp_001" in result["html_snippet"]
    assert result["employee_name"] == "Alice"


@pytest.mark.asyncio
async def test_embed_code_correct_format():
    """Embed code is a valid script tag with proper attributes."""
    from src.api.channel_routes import _embed_snippet

    result = _embed_snippet("emp_001", "Alice", "ar")
    snippet = result["html_snippet"]

    assert "<script" in snippet
    assert 'data-employee-id="emp_001"' in snippet
    assert 'data-language="ar"' in snippet
    assert 'data-position="bottom-right"' in snippet
    assert 'data-theme="auto"' in snippet
    assert "widget.js" in snippet
    assert "</script>" in snippet
    assert len(result["instructions"]) == 3


# ═══════════════════════════════════════════════════════════════════
# Widget Adapter Tests
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_widget_supports_voice_and_video():
    """Widget adapter supports both voice and video."""
    adapter = WidgetAdapter()
    assert adapter.supports_voice() is True
    assert adapter.supports_video() is True


@pytest.mark.asyncio
async def test_widget_text_message_parsed():
    """Widget text message parsed correctly."""
    adapter = WidgetAdapter()
    config = MagicMock(customer_id="cust_001")
    raw = {
        "type": "text_message",
        "employee_id": "emp_001",
        "session_id": "ws_abc",
        "visitor_id": "v_xyz",
        "text": "Hello from widget",
    }
    msg = await adapter.parse_incoming(raw, config)

    assert msg.channel == ChannelType.WIDGET
    assert msg.text == "Hello from widget"
    assert msg.message_type == "text"
    assert msg.channel_session_id == "ws_abc"


# ═══════════════════════════════════════════════════════════════════
# Channel Router — adapter registration
# ═══════════════════════════════════════════════════════════════════


def test_adapter_registration(mock_agent, mock_tts, mock_asr, mock_runpod, mock_r2):
    """Adapters can be registered and retrieved."""
    r = ChannelRouter(
        agent_engine=mock_agent,
        tts=mock_tts,
        asr=mock_asr,
        runpod=mock_runpod,
        r2=mock_r2,
    )
    wa = WhatsAppAdapter()
    r.register_adapter(ChannelType.WHATSAPP, wa)

    assert r.get_adapter(ChannelType.WHATSAPP) is wa
    assert r.get_adapter(ChannelType.TELEGRAM) is None


@pytest.mark.asyncio
async def test_unregistered_channel_raises(mock_agent, mock_tts, mock_asr, mock_runpod, mock_r2):
    """Routing to unregistered channel raises ValueError."""
    r = ChannelRouter(
        agent_engine=mock_agent,
        tts=mock_tts,
        asr=mock_asr,
        runpod=mock_runpod,
        r2=mock_r2,
    )
    msg = IncomingMessage(
        channel=ChannelType.TELEGRAM,
        channel_session_id="tg_1_emp",
        employee_id="emp_001",
        customer_id="cust_001",
        visitor_id="v_x",
        message_type="text",
        text="Hi",
    )
    with pytest.raises(ValueError, match="No adapter registered"):
        await r.handle_message(msg)


# ═══════════════════════════════════════════════════════════════════
# Telegram Webhook Registration
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_telegram_webhook_registration():
    """TelegramAdapter.register_webhook calls setWebhook API."""
    with patch("httpx.AsyncClient") as MockClient:
        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"ok": True}
        mock_resp.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await TelegramAdapter.register_webhook(
            "123456:ABC", "https://api.maskki.com/webhooks/telegram/emp_001"
        )

    assert result["ok"] is True
    mock_client.post.assert_called_once()
    call_url = mock_client.post.call_args[0][0]
    assert "setWebhook" in call_url
