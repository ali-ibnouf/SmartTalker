"""Tests for DocumentFlowHandler, TTSRestService, and webhook wiring.

All external calls (Redis, LLM, OCR, SessionLinks, TTS) are mocked.
"""

import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.knowledge.services_kb import (
    SERVICES,
    get_all_service_names_ar,
    get_required_docs,
    get_service,
)
from src.services.document_flow_handler import (
    FLOW_REDIS_PREFIX,
    FLOW_TTL_SECONDS,
    DocumentFlowHandler,
    DocumentFlowResult,
    FlowState,
    _is_document_valid,
    _msg,
    _parse_json_from_llm,
)


# ── Helpers / fixtures ────────────────────────────────────────────────────────


@dataclass
class FakeLLMResult:
    text: str = ""
    cost_usd: float = 0.0


@dataclass
class FakeOCRResult:
    text: str = ""
    confidence: float = 0.9
    document_type: MagicMock = field(default_factory=lambda: MagicMock(value="national_id"))
    fields: dict = field(default_factory=dict)
    method: str = "tesseract"

    def to_dict(self):
        return {"text": self.text, "confidence": self.confidence}


@pytest.fixture
def mock_redis():
    redis = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    redis.setex = AsyncMock()
    redis.delete = AsyncMock()
    return redis


@pytest.fixture
def mock_llm():
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value=FakeLLMResult(
        text='{"service_id": "license_renewal", "confidence": 0.95}'
    ))
    return llm


@pytest.fixture
def mock_ocr():
    ocr = AsyncMock()
    ocr.analyze_image = AsyncMock(return_value=FakeOCRResult(
        text="Name: Ali  ID: 12345  Expiry: 2028-01-01",
        confidence=0.9,
        fields={"name": "Ali", "id_number": "12345", "expiry_date": "2028-01-01"},
    ))
    return ocr


@pytest.fixture
def mock_session_links():
    svc = AsyncMock()
    svc.create_link = AsyncMock(return_value={
        "token": "abc123",
        "url": "https://app.maskki.com/s/abc123",
        "expires_at": "2026-01-01T00:00:00",
        "expires_minutes": 30,
    })
    return svc


@pytest.fixture
def handler(mock_redis, mock_llm, mock_ocr, mock_session_links):
    return DocumentFlowHandler(
        redis_client=mock_redis,
        llm_service=mock_llm,
        ocr_service=mock_ocr,
        session_link_service=mock_session_links,
    )


# ── 1. Services KB ───────────────────────────────────────────────────────────


def test_services_kb_has_all_entries():
    """Knowledge base has 4 services."""
    assert len(SERVICES) == 4
    assert "license_renewal" in SERVICES
    assert "visa_application" in SERVICES


def test_get_required_docs_excludes_optional():
    """get_required_docs() excludes non-required docs."""
    docs = get_required_docs("residency_renewal")
    ids = [d["id"] for d in docs]
    assert "salary_slip" not in ids
    assert "passport" in ids


def test_get_all_service_names_ar():
    """get_all_service_names_ar() returns Arabic names."""
    names = get_all_service_names_ar()
    assert len(names) == 4
    assert "تجديد رخصة القيادة" in names


# ── 2. Service identification ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_identify_service_success(handler, mock_redis):
    """First text message identifies the service and asks for first document."""
    result = await handler.handle(
        visitor_id="visitor_001",
        customer_id="cust_001",
        avatar_id="avatar_001",
        message={"type": "text", "content": "أريد تجديد رخصة القيادة"},
    )

    assert isinstance(result, DocumentFlowResult)
    assert result.state == FlowState.VERIFYING_DOCS
    assert "رخصة" in result.voice_text or "License" in result.voice_text
    # State saved to Redis
    mock_redis.setex.assert_called_once()


@pytest.mark.asyncio
async def test_identify_service_low_confidence(handler, mock_llm, mock_redis):
    """When LLM is not confident, ask user to clarify."""
    mock_llm.generate.return_value = FakeLLMResult(
        text='{"service_id": null, "confidence": 0.3}'
    )

    result = await handler.handle(
        visitor_id="visitor_002",
        customer_id="cust_001",
        avatar_id="avatar_001",
        message={"type": "text", "content": "مرحباً"},
    )

    assert result.state == FlowState.IDENTIFYING_SERVICE
    # Should list available services
    assert "نقدم" in result.voice_text or "offer" in result.voice_text.lower()


@pytest.mark.asyncio
async def test_identify_service_llm_failure(handler, mock_llm, mock_redis):
    """When LLM throws, gracefully ask user to clarify."""
    mock_llm.generate.side_effect = RuntimeError("API timeout")

    result = await handler.handle(
        visitor_id="visitor_003",
        customer_id="cust_001",
        avatar_id="avatar_001",
        message={"type": "text", "content": "أريد خدمة"},
    )

    assert result.state == FlowState.IDENTIFYING_SERVICE


# ── 3. Document verification ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_verify_document_success(handler, mock_redis, mock_ocr):
    """Image in VERIFYING_DOCS state runs OCR and advances to next doc."""
    state = {
        "status": FlowState.VERIFYING_DOCS,
        "service_id": "license_renewal",
        "language": "ar",
        "required_docs": get_required_docs("license_renewal"),
        "verified_docs": {},
        "current_doc_index": 0,
        "customer_id": "cust_001",
        "avatar_id": "avatar_001",
    }
    mock_redis.get.return_value = json.dumps(state)

    result = await handler.handle(
        visitor_id="visitor_004",
        customer_id="cust_001",
        avatar_id="avatar_001",
        message={"type": "image", "content": b"fake_image", "format": "jpeg"},
    )

    assert result.state == FlowState.VERIFYING_DOCS
    assert "تم التحقق" in result.voice_text or "verified" in result.voice_text.lower()
    mock_ocr.analyze_image.assert_called_once()


@pytest.mark.asyncio
async def test_verify_document_low_quality(handler, mock_redis, mock_ocr):
    """Low-confidence OCR result asks for a clearer photo."""
    state = {
        "status": FlowState.VERIFYING_DOCS,
        "service_id": "license_renewal",
        "language": "ar",
        "required_docs": get_required_docs("license_renewal"),
        "verified_docs": {},
        "current_doc_index": 0,
    }
    mock_redis.get.return_value = json.dumps(state)
    mock_ocr.analyze_image.return_value = FakeOCRResult(confidence=0.3)

    result = await handler.handle(
        visitor_id="visitor_005",
        customer_id="cust_001",
        avatar_id="avatar_001",
        message={"type": "image", "content": b"blurry", "format": "jpeg"},
    )

    assert result.state == FlowState.VERIFYING_DOCS
    assert "واضح" in result.voice_text or "unclear" in result.voice_text.lower()


@pytest.mark.asyncio
async def test_verify_document_type_mismatch(handler, mock_redis, mock_ocr):
    """Wrong document type asks for correct one."""
    state = {
        "status": FlowState.VERIFYING_DOCS,
        "service_id": "license_renewal",
        "language": "ar",
        "required_docs": get_required_docs("license_renewal"),
        "verified_docs": {},
        "current_doc_index": 0,  # expecting national_id
    }
    mock_redis.get.return_value = json.dumps(state)
    mock_ocr.analyze_image.return_value = FakeOCRResult(
        confidence=0.9,
        document_type=MagicMock(value="passport"),  # wrong type
    )

    result = await handler.handle(
        visitor_id="visitor_006",
        customer_id="cust_001",
        avatar_id="avatar_001",
        message={"type": "image", "content": b"passport_img", "format": "jpeg"},
    )

    assert result.state == FlowState.VERIFYING_DOCS
    assert "لا يبدو" in result.voice_text or "doesn't appear" in result.voice_text.lower()


@pytest.mark.asyncio
async def test_verify_document_expired(handler, mock_redis, mock_ocr):
    """Expired document is rejected."""
    state = {
        "status": FlowState.VERIFYING_DOCS,
        "service_id": "license_renewal",
        "language": "ar",
        "required_docs": get_required_docs("license_renewal"),
        "verified_docs": {},
        "current_doc_index": 0,
    }
    mock_redis.get.return_value = json.dumps(state)

    # Expiry within 90 days → invalid
    soon = (date.today() + timedelta(days=30)).isoformat()
    mock_ocr.analyze_image.return_value = FakeOCRResult(
        confidence=0.9,
        fields={"expiry_date": soon},
    )

    result = await handler.handle(
        visitor_id="visitor_007",
        customer_id="cust_001",
        avatar_id="avatar_001",
        message={"type": "image", "content": b"expiring_id", "format": "jpeg"},
    )

    assert result.state == FlowState.VERIFYING_DOCS
    assert "منتهي" in result.voice_text or "expired" in result.voice_text.lower()


# ── 4. Full flow completion ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_all_docs_complete_no_video(handler, mock_redis, mock_ocr):
    """All docs verified for a non-video service → fees + completed."""
    docs = get_required_docs("license_renewal")
    # Simulate all docs but last verified
    verified = {}
    for d in docs[:-1]:
        verified[d["id"]] = {"confidence": 0.9, "fields": {}, "r2_key": ""}

    last_doc = docs[-1]  # eye_test_certificate

    state = {
        "status": FlowState.VERIFYING_DOCS,
        "service_id": "license_renewal",
        "language": "ar",
        "required_docs": docs,
        "verified_docs": verified,
        "current_doc_index": len(docs) - 1,
        "customer_id": "cust_001",
        "avatar_id": "avatar_001",
    }
    mock_redis.get.return_value = json.dumps(state)

    # OCR returns matching type for the last expected doc
    mock_ocr.analyze_image.return_value = FakeOCRResult(
        text="Eye test OK",
        confidence=0.9,
        document_type=MagicMock(value=last_doc["id"]),
        fields={"expiry_date": (date.today() + timedelta(days=365)).isoformat()},
    )

    result = await handler.handle(
        visitor_id="visitor_008",
        customer_id="cust_001",
        avatar_id="avatar_001",
        message={"type": "image", "content": b"last_doc", "format": "jpeg"},
    )

    assert result.state == FlowState.COMPLETED
    assert "20" in result.voice_text  # fee amount
    assert result.session_link is None


@pytest.mark.asyncio
async def test_all_docs_complete_with_video(handler, mock_redis, mock_ocr, mock_session_links):
    """All docs verified for video service → session link generated."""
    docs = get_required_docs("visa_application")
    verified = {}
    for d in docs[:-1]:
        verified[d["id"]] = {"confidence": 0.9, "fields": {}, "r2_key": ""}

    last_doc = docs[-1]

    state = {
        "status": FlowState.VERIFYING_DOCS,
        "service_id": "visa_application",
        "language": "ar",
        "required_docs": docs,
        "verified_docs": verified,
        "current_doc_index": len(docs) - 1,
        "customer_id": "cust_001",
        "avatar_id": "avatar_001",
    }
    mock_redis.get.return_value = json.dumps(state)

    mock_ocr.analyze_image.return_value = FakeOCRResult(
        text="National ID",
        confidence=0.9,
        document_type=MagicMock(value=last_doc["id"]),
        fields={"expiry_date": (date.today() + timedelta(days=365)).isoformat()},
    )

    result = await handler.handle(
        visitor_id="visitor_009",
        customer_id="cust_001",
        avatar_id="avatar_001",
        message={"type": "image", "content": b"passport_img", "format": "jpeg"},
    )

    assert result.state == FlowState.VIDEO_SESSION
    assert result.session_link == "https://app.maskki.com/s/abc123"
    mock_session_links.create_link.assert_called_once()


# ── 5. Helper functions ─────────────────────────────────────────────────────


def test_msg_returns_arabic():
    assert _msg("ar", ar="مرحبا", en="Hello") == "مرحبا"


def test_msg_returns_english():
    assert _msg("en", ar="مرحبا", en="Hello") == "Hello"


def test_is_document_valid_low_confidence():
    ocr = FakeOCRResult(confidence=0.3)
    assert _is_document_valid(ocr) is False


def test_is_document_valid_expired():
    soon = (date.today() + timedelta(days=30)).isoformat()
    ocr = FakeOCRResult(confidence=0.9, fields={"expiry_date": soon})
    assert _is_document_valid(ocr) is False


def test_is_document_valid_ok():
    far = (date.today() + timedelta(days=365)).isoformat()
    ocr = FakeOCRResult(confidence=0.9, fields={"expiry_date": far})
    assert _is_document_valid(ocr) is True


def test_parse_json_from_llm_plain():
    result = _parse_json_from_llm('{"service_id": "license_renewal"}')
    assert result["service_id"] == "license_renewal"


def test_parse_json_from_llm_markdown_fenced():
    result = _parse_json_from_llm('```json\n{"service_id": "x"}\n```')
    assert result["service_id"] == "x"


# ── 6. Webhook integration ──────────────────────────────────────────────────


def test_looks_like_service_request():
    """Keyword detection for government service requests."""
    from src.api.webhooks.whatsapp import _looks_like_service_request

    assert _looks_like_service_request("أريد تجديد رخصة القيادة") is True
    assert _looks_like_service_request("تأشيرة سفر") is True
    assert _looks_like_service_request("Hello, how are you?") is False
    assert _looks_like_service_request("") is False


def test_incoming_to_flow_message_text():
    """Text message converts to flow dict correctly."""
    from src.api.webhooks.whatsapp import _incoming_to_flow_message
    from src.channels.base import ChannelType, IncomingMessage

    msg = IncomingMessage(
        channel=ChannelType.WHATSAPP,
        channel_session_id="wa_123_emp",
        employee_id="emp",
        customer_id="cust",
        visitor_id="vis",
        message_type="text",
        text="أريد تجديد الإقامة",
    )

    result = _incoming_to_flow_message(msg)
    assert result["type"] == "text"
    assert "تجديد" in result["content"]
