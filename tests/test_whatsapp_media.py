"""Tests for WhatsApp media handlers (video, image, document).

All downloads, OCR, video extraction, and R2 uploads are mocked.
"""

from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from src.channels.base import ChannelType, IncomingMessage
from src.channels.whatsapp import WhatsAppAdapter


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def wa_config():
    config = MagicMock()
    config.employee_id = "emp_001"
    config.customer_id = "cust_001"
    config.wa_access_token = "test_token"
    config.wa_phone_number_id = "1234567890"
    return config


@pytest.fixture
def mock_r2():
    r2 = MagicMock()
    r2.upload_incoming_media = MagicMock(return_value="https://r2.example.com/incoming/media.jpg")
    return r2


@pytest.fixture
def mock_ocr():
    ocr = AsyncMock()
    result = MagicMock()
    result.text = "محمد علي — جواز سفر رقم A12345"
    result.confidence = 0.95
    result.method = "qwen_vl"
    result.document_type = MagicMock()
    result.document_type.value = "passport"
    result.to_dict = MagicMock(return_value={
        "text": "محمد علي — جواز سفر رقم A12345",
        "confidence": 0.95,
        "method": "qwen_vl",
        "document_type": "passport",
        "fields": {"full_name": "محمد علي"},
        "quality_issues": [],
    })
    ocr.analyze_image = AsyncMock(return_value=result)
    return ocr


def _build_wa_payload(msg_type: str, msg_content: dict) -> dict:
    """Build a standard WhatsApp webhook payload."""
    msg = {"from": "96891234567", "id": "wamid_test", "type": msg_type}
    msg.update(msg_content)
    return {
        "entry": [{
            "changes": [{
                "value": {
                    "messages": [msg],
                }
            }]
        }]
    }


# ─── Video tests ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_video_extracts_audio_as_voice(wa_config, mock_r2):
    """Video message → extract audio → returns as voice for ASR."""
    adapter = WhatsAppAdapter(r2_storage=mock_r2)

    payload = _build_wa_payload("video", {
        "video": {"id": "media_vid_001", "mime_type": "video/mp4"},
    })

    with patch.object(adapter, "_download_media", new_callable=AsyncMock, return_value=b"fake_video_bytes"):
        with patch("src.utils.video.extract_audio_from_bytes", return_value=b"fake_wav_audio") as mock_extract:
            msg = await adapter.parse_incoming(payload, wa_config)

    assert msg.message_type == "voice"
    assert msg.audio_bytes == b"fake_wav_audio"
    assert msg.metadata["media_type"] == "video"
    mock_extract.assert_called_once_with(b"fake_video_bytes", "mp4")


@pytest.mark.asyncio
async def test_video_extraction_failure_fallback(wa_config):
    """Video extraction fails → Arabic fallback text message."""
    adapter = WhatsAppAdapter()

    payload = _build_wa_payload("video", {
        "video": {"id": "media_vid_002", "mime_type": "video/mp4"},
    })

    with patch.object(adapter, "_download_media", new_callable=AsyncMock, side_effect=RuntimeError("ffmpeg failed")):
        msg = await adapter.parse_incoming(payload, wa_config)

    assert msg.message_type == "text"
    assert "لم أتمكن من معالجته" in msg.text
    assert msg.metadata.get("media_error")


@pytest.mark.asyncio
async def test_video_no_media_id(wa_config):
    """Video with no media ID → default text message (unchanged)."""
    adapter = WhatsAppAdapter()

    payload = _build_wa_payload("video", {
        "video": {},  # no "id" key
    })

    msg = await adapter.parse_incoming(payload, wa_config)

    # Stays as default text type since handler returns early
    assert msg.message_type == "text"
    assert msg.audio_bytes is None


# ─── Image tests ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_image_ocr_extracts_text(wa_config, mock_r2, mock_ocr):
    """Image message → OCR → returns text with document context."""
    adapter = WhatsAppAdapter(r2_storage=mock_r2, ocr_service=mock_ocr)

    payload = _build_wa_payload("image", {
        "image": {"id": "media_img_001", "mime_type": "image/jpeg"},
    })

    with patch.object(adapter, "_download_media", new_callable=AsyncMock, return_value=b"fake_jpg"):
        msg = await adapter.parse_incoming(payload, wa_config)

    assert msg.message_type == "text"
    assert "محتوى المستند المرفق" in msg.text
    assert "محمد علي" in msg.text
    assert msg.metadata["media_type"] == "image"
    assert msg.metadata["ocr_result"]["document_type"] == "passport"
    mock_ocr.analyze_image.assert_called_once_with(b"fake_jpg", "jpeg", extract_fields=True)


@pytest.mark.asyncio
async def test_image_caption_preserved(wa_config, mock_r2, mock_ocr):
    """Image with caption → caption appears before OCR text."""
    adapter = WhatsAppAdapter(r2_storage=mock_r2, ocr_service=mock_ocr)

    payload = _build_wa_payload("image", {
        "image": {"id": "media_img_002", "mime_type": "image/jpeg", "caption": "هذا جوازي"},
    })

    with patch.object(adapter, "_download_media", new_callable=AsyncMock, return_value=b"fake_jpg"):
        msg = await adapter.parse_incoming(payload, wa_config)

    assert msg.message_type == "text"
    # Caption comes first
    assert msg.text.startswith("هذا جوازي")
    # OCR text follows
    assert "محتوى المستند المرفق" in msg.text


@pytest.mark.asyncio
async def test_image_ocr_failure_fallback(wa_config):
    """Image processing fails → Arabic fallback message."""
    adapter = WhatsAppAdapter()  # No OCR service

    payload = _build_wa_payload("image", {
        "image": {"id": "media_img_003", "mime_type": "image/jpeg"},
    })

    with patch.object(adapter, "_download_media", new_callable=AsyncMock, side_effect=RuntimeError("download failed")):
        msg = await adapter.parse_incoming(payload, wa_config)

    assert msg.message_type == "text"
    assert "لم أتمكن من قراءتها" in msg.text


# ─── Document tests ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_pdf_text_extracted(wa_config, mock_r2):
    """PDF document → PyPDF2 text extraction → text message."""
    adapter = WhatsAppAdapter(r2_storage=mock_r2)

    payload = _build_wa_payload("document", {
        "document": {
            "id": "media_doc_001",
            "mime_type": "application/pdf",
            "filename": "contract.pdf",
        },
    })

    with patch.object(adapter, "_download_media", new_callable=AsyncMock, return_value=b"fake_pdf"):
        with patch.object(WhatsAppAdapter, "_parse_pdf_bytes", return_value="عقد عمل بين الطرفين"):
            msg = await adapter.parse_incoming(payload, wa_config)

    assert msg.message_type == "text"
    assert "contract.pdf" in msg.text
    assert "عقد عمل بين الطرفين" in msg.text
    assert msg.metadata["filename"] == "contract.pdf"
    assert msg.metadata["media_type"] == "document"


@pytest.mark.asyncio
async def test_document_image_uses_ocr(wa_config, mock_r2, mock_ocr):
    """Document with image MIME type → OCR service is used."""
    adapter = WhatsAppAdapter(r2_storage=mock_r2, ocr_service=mock_ocr)

    payload = _build_wa_payload("document", {
        "document": {
            "id": "media_doc_002",
            "mime_type": "image/jpeg",
            "filename": "id_card.jpg",
        },
    })

    with patch.object(adapter, "_download_media", new_callable=AsyncMock, return_value=b"fake_img"):
        msg = await adapter.parse_incoming(payload, wa_config)

    assert msg.message_type == "text"
    assert "محمد علي" in msg.text
    mock_ocr.analyze_image.assert_called_once()


@pytest.mark.asyncio
async def test_document_unsupported_type(wa_config, mock_r2):
    """Document with unsupported MIME → fallback message with filename."""
    adapter = WhatsAppAdapter(r2_storage=mock_r2)

    payload = _build_wa_payload("document", {
        "document": {
            "id": "media_doc_003",
            "mime_type": "application/zip",
            "filename": "archive.zip",
        },
    })

    with patch.object(adapter, "_download_media", new_callable=AsyncMock, return_value=b"fake_zip"):
        msg = await adapter.parse_incoming(payload, wa_config)

    assert msg.message_type == "text"
    assert "archive.zip" in msg.text
    assert "لا يمكن معالجته" in msg.text


# ─── Unsupported type ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_unsupported_message_type_sticker(wa_config):
    """Sticker or other unsupported type → Arabic unsupported message."""
    adapter = WhatsAppAdapter()

    payload = _build_wa_payload("sticker", {
        "sticker": {"id": "sticker_001"},
    })

    msg = await adapter.parse_incoming(payload, wa_config)

    assert msg.message_type == "text"
    assert "لا أستطيع معالجة" in msg.text
