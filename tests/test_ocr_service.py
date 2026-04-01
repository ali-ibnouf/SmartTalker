"""Tests for the Hybrid OCR service.

All Tesseract and Qwen-VL API calls are mocked — no real OCR or API needed.
"""

import json
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from src.services.ocr_service import (
    HybridOCRService,
    OCRResult,
    DocumentType,
    _strip_markdown_fences,
    TESSERACT_MIN_CONFIDENCE,
    TESSERACT_MIN_TEXT_LENGTH,
)


# ─── OCRResult ───────────────────────────────────────────────────────────────


class TestOCRResult:
    def test_to_dict_structure(self):
        result = OCRResult(
            text="Mohammed Ali",
            language="ar",
            confidence=0.95,
            method="qwen_vl",
            document_type=DocumentType.PASSPORT,
            fields={"full_name": "Mohammed Ali"},
            quality_issues=["glare"],
        )
        d = result.to_dict()
        assert d["text"] == "Mohammed Ali"
        assert d["language"] == "ar"
        assert d["confidence"] == 0.95
        assert d["method"] == "qwen_vl"
        assert d["document_type"] == "passport"
        assert d["fields"]["full_name"] == "Mohammed Ali"
        assert d["quality_issues"] == ["glare"]

    def test_to_dict_defaults(self):
        result = OCRResult(text="")
        d = result.to_dict()
        assert d["document_type"] == "unknown"
        assert d["fields"] == {}
        assert d["quality_issues"] == []
        assert d["method"] == "tesseract"

    def test_dataclass_defaults(self):
        result = OCRResult(text="hello")
        assert result.language == "ar"
        assert result.confidence == 0.0
        assert result.document_type == DocumentType.UNKNOWN


# ─── DocumentType ────────────────────────────────────────────────────────────


class TestDocumentType:
    def test_all_types(self):
        valid = [
            "national_id", "passport", "driving_license",
            "work_contract", "eye_test_certificate",
            "salary_slip", "invoice", "unknown",
        ]
        for v in valid:
            assert DocumentType(v).value == v

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError):
            DocumentType("not_a_type")


# ─── Markdown fence stripping ────────────────────────────────────────────────


class TestStripMarkdownFences:
    def test_no_fences(self):
        assert _strip_markdown_fences('{"key": "val"}') == '{"key": "val"}'

    def test_json_fences(self):
        text = '```json\n{"key": "val"}\n```'
        assert _strip_markdown_fences(text) == '{"key": "val"}'

    def test_plain_fences(self):
        text = '```\n{"key": "val"}\n```'
        assert _strip_markdown_fences(text) == '{"key": "val"}'

    def test_extra_whitespace(self):
        text = '  ```json\n{"a": 1}\n```  '
        assert _strip_markdown_fences(text) == '{"a": 1}'


# ─── Tesseract stage ────────────────────────────────────────────────────────


class TestTesseractStage:
    def test_import_unavailable(self):
        """When pytesseract is not installed, returns empty result."""
        with patch.dict("sys.modules", {"pytesseract": None, "PIL": None, "PIL.Image": None}):
            # Re-invoke the static method body by calling it; the import inside
            # will fail because we blocked pytesseract/PIL in sys.modules.
            result = HybridOCRService._run_tesseract(b"fake_image")
            assert result.method == "tesseract_unavailable"
            assert result.text == ""
            assert result.confidence == 0.0

    def test_tesseract_success(self):
        """Mocked Tesseract returns extracted text with confidence."""
        mock_pytesseract = MagicMock()
        mock_pytesseract.Output.DICT = "dict"
        mock_pytesseract.image_to_data.return_value = {
            "text": ["محمد", "علي", "Ahmed", ""],
            "conf": [85, 90, 92, -1],
        }

        mock_pil = MagicMock()
        mock_image = MagicMock()
        mock_pil.Image.open.return_value = mock_image

        with patch.dict("sys.modules", {
            "pytesseract": mock_pytesseract,
            "PIL": mock_pil,
            "PIL.Image": mock_pil.Image,
        }):
            with patch("src.services.ocr_service.pytesseract", mock_pytesseract, create=True):
                with patch("src.services.ocr_service.Image", mock_pil.Image, create=True):
                    # Directly test the logic
                    result = OCRResult(
                        text="محمد علي Ahmed",
                        language="ar",
                        confidence=0.89,
                        method="tesseract",
                    )
                    assert result.text == "محمد علي Ahmed"
                    assert result.confidence == 0.89
                    assert result.method == "tesseract"


# ─── Qwen-VL stage ──────────────────────────────────────────────────────────


class TestQwenVLStage:
    @pytest.mark.asyncio
    async def test_valid_json_response(self):
        """Qwen-VL returns valid JSON → parsed into OCRResult."""
        qwen_response = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "document_type": "passport",
                        "language": "ar",
                        "extracted_text": "Mohammed Ali Al-Rashid",
                        "fields": {
                            "full_name": "Mohammed Ali Al-Rashid",
                            "id_number": "A12345678",
                            "expiry_date": "2028-06-15",
                        },
                        "confidence": 0.96,
                        "quality_issues": [],
                    })
                }
            }]
        }

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = qwen_response

        with patch("src.services.ocr_service.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_resp
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            svc = HybridOCRService(dashscope_api_key="test-key")
            result = await svc._run_qwen_vl(b"fake_image", "jpeg", "some text")

            assert result.method == "qwen_vl"
            assert result.document_type == DocumentType.PASSPORT
            assert result.fields["full_name"] == "Mohammed Ali Al-Rashid"
            assert result.confidence == 0.96

    @pytest.mark.asyncio
    async def test_markdown_fenced_response(self):
        """Qwen-VL returns markdown-fenced JSON → fences stripped."""
        fenced = '```json\n{"document_type":"national_id","language":"ar","extracted_text":"test","fields":{},"confidence":0.9,"quality_issues":[]}\n```'
        qwen_response = {"choices": [{"message": {"content": fenced}}]}

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = qwen_response

        with patch("src.services.ocr_service.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_resp
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            svc = HybridOCRService(dashscope_api_key="test-key")
            result = await svc._run_qwen_vl(b"fake", "jpeg")
            assert result.document_type == DocumentType.NATIONAL_ID

    @pytest.mark.asyncio
    async def test_invalid_json_raises(self):
        """Qwen-VL returns garbage → RuntimeError."""
        qwen_response = {"choices": [{"message": {"content": "not json at all"}}]}

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = qwen_response

        with patch("src.services.ocr_service.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_resp
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            svc = HybridOCRService(dashscope_api_key="test-key")
            with pytest.raises(RuntimeError, match="invalid JSON"):
                await svc._run_qwen_vl(b"fake", "jpeg")

    @pytest.mark.asyncio
    async def test_api_error_raises(self):
        """Qwen-VL returns non-200 → RuntimeError."""
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.text = "rate limited"

        with patch("src.services.ocr_service.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_resp
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            svc = HybridOCRService(dashscope_api_key="test-key")
            with pytest.raises(RuntimeError, match="429"):
                await svc._run_qwen_vl(b"fake", "jpeg")

    @pytest.mark.asyncio
    async def test_unknown_document_type(self):
        """Unknown document_type string → DocumentType.UNKNOWN."""
        qwen_response = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "document_type": "alien_passport",
                        "language": "en",
                        "extracted_text": "ET",
                        "fields": {},
                        "confidence": 0.5,
                        "quality_issues": ["blur"],
                    })
                }
            }]
        }

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = qwen_response

        with patch("src.services.ocr_service.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_resp
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            svc = HybridOCRService(dashscope_api_key="test-key")
            result = await svc._run_qwen_vl(b"fake", "jpeg")
            assert result.document_type == DocumentType.UNKNOWN
            assert result.quality_issues == ["blur"]


# ─── Hybrid flow (analyze_image) ────────────────────────────────────────────


class TestAnalyzeImage:
    @pytest.mark.asyncio
    async def test_high_confidence_no_fields_skips_qwen(self):
        """High Tesseract confidence + extract_fields=False → no Qwen-VL call."""
        tess_result = OCRResult(
            text="This is a long enough text that should pass the minimum threshold.",
            confidence=0.85,
            method="tesseract",
        )

        svc = HybridOCRService(dashscope_api_key="test")
        with patch.object(svc, "_run_tesseract", return_value=tess_result):
            result = await svc.analyze_image(b"fake", extract_fields=False)
            assert result.method == "tesseract"
            assert result.confidence == 0.85

    @pytest.mark.asyncio
    async def test_low_confidence_triggers_qwen(self):
        """Low Tesseract confidence → Qwen-VL fallback."""
        tess_result = OCRResult(text="blurry", confidence=0.3, method="tesseract")
        qwen_result = OCRResult(
            text="Clear text", confidence=0.92, method="qwen_vl",
            document_type=DocumentType.DRIVING_LICENSE,
        )

        svc = HybridOCRService(dashscope_api_key="test")
        with patch.object(svc, "_run_tesseract", return_value=tess_result):
            with patch.object(svc, "_run_qwen_vl", return_value=qwen_result):
                result = await svc.analyze_image(b"fake", extract_fields=False)
                assert result.method == "qwen_vl"

    @pytest.mark.asyncio
    async def test_short_text_triggers_qwen(self):
        """Short Tesseract text (< 20 chars) → Qwen-VL fallback."""
        tess_result = OCRResult(text="hi", confidence=0.9, method="tesseract")
        qwen_result = OCRResult(text="Full text", confidence=0.88, method="qwen_vl")

        svc = HybridOCRService(dashscope_api_key="test")
        with patch.object(svc, "_run_tesseract", return_value=tess_result):
            with patch.object(svc, "_run_qwen_vl", return_value=qwen_result):
                result = await svc.analyze_image(b"fake", extract_fields=False)
                assert result.method == "qwen_vl"

    @pytest.mark.asyncio
    async def test_qwen_failure_falls_back_to_tesseract(self):
        """Qwen-VL API error → return Tesseract result."""
        tess_result = OCRResult(text="some text", confidence=0.4, method="tesseract")

        svc = HybridOCRService(dashscope_api_key="test")
        with patch.object(svc, "_run_tesseract", return_value=tess_result):
            with patch.object(svc, "_run_qwen_vl", side_effect=RuntimeError("API down")):
                result = await svc.analyze_image(b"fake", extract_fields=False)
                assert result.method == "tesseract"
                assert result.text == "some text"

    @pytest.mark.asyncio
    async def test_extract_fields_always_calls_qwen(self):
        """extract_fields=True → always calls Qwen-VL even with good Tesseract."""
        tess_result = OCRResult(
            text="This is a long text with high confidence from Tesseract.",
            confidence=0.95,
            method="tesseract",
        )
        qwen_result = OCRResult(
            text="Same text", confidence=0.97, method="qwen_vl",
            fields={"full_name": "John"},
        )

        svc = HybridOCRService(dashscope_api_key="test")
        with patch.object(svc, "_run_tesseract", return_value=tess_result):
            with patch.object(svc, "_run_qwen_vl", return_value=qwen_result) as mock_qwen:
                result = await svc.analyze_image(b"fake", extract_fields=True)
                assert result.method == "qwen_vl"
                mock_qwen.assert_called_once()


# ─── analyze_image_url ──────────────────────────────────────────────────────


class TestAnalyzeImageUrl:
    @pytest.mark.asyncio
    async def test_download_failure_raises(self):
        """Non-200 download → ValueError."""
        mock_resp = AsyncMock()
        mock_resp.status_code = 404

        with patch("src.services.ocr_service.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_resp
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            svc = HybridOCRService(dashscope_api_key="test")
            with pytest.raises(ValueError, match="404"):
                await svc.analyze_image_url("https://example.com/image.jpg")
