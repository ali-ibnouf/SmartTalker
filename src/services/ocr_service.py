"""Hybrid OCR service: Tesseract (free) with Qwen-VL Flash (paid) fallback.

Stage 1 — Tesseract: Fast, free, raw text extraction from document images.
Stage 2 — Qwen-VL: Structured field extraction, document type classification,
           validity checks. Used when Tesseract confidence is low or when
           structured fields are requested.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import httpx

from src.utils.logger import setup_logger

logger = setup_logger("services.ocr")


class DocumentType(str, Enum):
    NATIONAL_ID = "national_id"
    PASSPORT = "passport"
    DRIVING_LICENSE = "driving_license"
    WORK_CONTRACT = "work_contract"
    EYE_TEST_CERT = "eye_test_certificate"
    SALARY_SLIP = "salary_slip"
    INVOICE = "invoice"
    UNKNOWN = "unknown"


@dataclass
class OCRResult:
    """Result of an OCR analysis."""

    text: str
    language: str = "ar"
    confidence: float = 0.0
    method: str = "tesseract"
    document_type: DocumentType = DocumentType.UNKNOWN
    fields: dict[str, Any] = field(default_factory=dict)
    quality_issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "language": self.language,
            "confidence": self.confidence,
            "method": self.method,
            "document_type": self.document_type.value,
            "fields": self.fields,
            "quality_issues": self.quality_issues,
        }


# Thresholds for deciding whether to call Qwen-VL
TESSERACT_MIN_CONFIDENCE = 0.6
TESSERACT_MIN_TEXT_LENGTH = 20

# Qwen-VL model for document analysis
QWEN_VL_MODEL = "qwen-vl-max-latest"

DASHSCOPE_CHAT_URL = (
    "https://dashscope-intl.aliyuncs.com"
    "/compatible-mode/v1/chat/completions"
)


class HybridOCRService:
    """Two-stage OCR: Tesseract first, Qwen-VL fallback.

    Uses Tesseract for fast free text extraction. Falls back to
    Qwen-VL Flash when confidence is low or structured field
    extraction is requested.
    """

    def __init__(self, dashscope_api_key: str) -> None:
        self._api_key = dashscope_api_key

    async def analyze_image(
        self,
        image_bytes: bytes,
        image_format: str = "jpeg",
        extract_fields: bool = True,
    ) -> OCRResult:
        """Analyze a document image.

        Args:
            image_bytes: Raw image file bytes.
            image_format: Image format hint (jpeg, png).
            extract_fields: If True, always use Qwen-VL for structured fields.

        Returns:
            OCRResult with extracted text, fields, and document type.
        """
        # Stage 1: Tesseract
        tess_result = self._run_tesseract(image_bytes)
        logger.info(
            "Tesseract stage complete",
            extra={
                "confidence": round(tess_result.confidence, 2),
                "text_length": len(tess_result.text),
                "method": tess_result.method,
            },
        )

        needs_qwen = (
            tess_result.confidence < TESSERACT_MIN_CONFIDENCE
            or len(tess_result.text.strip()) < TESSERACT_MIN_TEXT_LENGTH
            or extract_fields
        )

        if not needs_qwen:
            return tess_result

        # Stage 2: Qwen-VL
        logger.info("Escalating to Qwen-VL for document analysis")
        try:
            return await self._run_qwen_vl(image_bytes, image_format, tess_result.text)
        except Exception as exc:
            logger.warning("Qwen-VL failed, returning Tesseract result", extra={"error": str(exc)})
            return tess_result

    async def analyze_image_url(
        self,
        image_url: str,
        extract_fields: bool = True,
    ) -> OCRResult:
        """Analyze a document image from URL.

        Downloads the image first, then runs the hybrid pipeline.

        Args:
            image_url: Public URL of the image.
            extract_fields: If True, use Qwen-VL for structured fields.

        Returns:
            OCRResult with extracted text and metadata.
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(image_url)
            if resp.status_code != 200:
                raise ValueError(f"Failed to download image: HTTP {resp.status_code}")
            image_bytes = resp.content
            ct = resp.headers.get("content-type", "image/jpeg")
            fmt = ct.split("/")[-1].split(";")[0].strip()
            if fmt in ("jpg", "jpeg"):
                fmt = "jpeg"

        return await self.analyze_image(image_bytes, fmt, extract_fields)

    # ── Stage 1: Tesseract ───────────────────────────────────────────────

    @staticmethod
    def _run_tesseract(image_bytes: bytes) -> OCRResult:
        """Run Tesseract OCR on image bytes.

        Returns an OCRResult. If pytesseract is not installed, returns
        an empty result with method="tesseract_unavailable".
        """
        try:
            import pytesseract
            from PIL import Image
            import io
        except ImportError:
            logger.warning("pytesseract or Pillow not installed — skipping Tesseract")
            return OCRResult(text="", confidence=0.0, method="tesseract_unavailable")

        try:
            img = Image.open(io.BytesIO(image_bytes))
            data = pytesseract.image_to_data(
                img,
                lang="ara+eng",
                output_type=pytesseract.Output.DICT,
                config="--psm 3",
            )

            words: list[str] = []
            confs: list[int] = []
            for word, conf in zip(data["text"], data["conf"]):
                conf_int = int(conf)
                if conf_int > 0 and word.strip():
                    words.append(word)
                    confs.append(conf_int)

            text = " ".join(words)
            avg_conf = (sum(confs) / len(confs) / 100.0) if confs else 0.0

            from src.services.language_detector import detect_language
            lang = detect_language(text) if text else "ar"

            return OCRResult(text=text, language=lang, confidence=avg_conf, method="tesseract")

        except Exception as exc:
            logger.error("Tesseract processing failed", extra={"error": str(exc)})
            return OCRResult(text="", confidence=0.0, method="tesseract_error")

    # ── Stage 2: Qwen-VL ────────────────────────────────────────────────

    async def _run_qwen_vl(
        self,
        image_bytes: bytes,
        image_format: str,
        tesseract_text: str = "",
    ) -> OCRResult:
        """Call Qwen-VL Flash for structured document analysis.

        Args:
            image_bytes: Raw image bytes.
            image_format: Format hint (jpeg, png).
            tesseract_text: Text from Tesseract stage (provided as context).

        Returns:
            OCRResult with structured fields and document type.
        """
        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        data_url = f"data:image/{image_format};base64,{image_b64}"

        tess_hint = f'\nPrevious OCR extracted: "{tesseract_text}"' if tesseract_text else ""

        prompt = (
            "Analyze this document image carefully."
            f"{tess_hint}\n\n"
            "Respond ONLY with valid JSON (no markdown, no extra text):\n"
            "{\n"
            '  "document_type": "national_id|passport|driving_license|work_contract'
            '|eye_test_certificate|salary_slip|invoice|unknown",\n'
            '  "language": "ar|en|other",\n'
            '  "extracted_text": "all visible text from document",\n'
            '  "fields": {\n'
            '    "full_name": "name or null",\n'
            '    "id_number": "ID/passport/license number or null",\n'
            '    "expiry_date": "YYYY-MM-DD or null",\n'
            '    "issue_date": "YYYY-MM-DD or null",\n'
            '    "nationality": "nationality or null",\n'
            '    "gender": "male|female|null",\n'
            '    "date_of_birth": "YYYY-MM-DD or null",\n'
            '    "issuing_authority": "issuing body or null"\n'
            "  },\n"
            '  "confidence": 0.95,\n'
            '  "quality_issues": []\n'
            "}\n\n"
            "quality_issues can include: blur, glare, cropped, low_resolution, partial"
        )

        payload = {
            "model": QWEN_VL_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": 800,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                DASHSCOPE_CHAT_URL,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"Qwen-VL API error {resp.status_code}: {resp.text[:200]}")
            result = resp.json()

        content = result["choices"][0]["message"]["content"].strip()
        content = _strip_markdown_fences(content)

        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Qwen-VL returned invalid JSON: {exc}") from exc

        doc_type_str = data.get("document_type", "unknown")
        try:
            doc_type = DocumentType(doc_type_str)
        except ValueError:
            doc_type = DocumentType.UNKNOWN

        return OCRResult(
            text=data.get("extracted_text", ""),
            language=data.get("language", "ar"),
            confidence=float(data.get("confidence", 0.9)),
            method="qwen_vl",
            document_type=doc_type,
            fields=data.get("fields", {}),
            quality_issues=data.get("quality_issues", []),
        )


def _strip_markdown_fences(text: str) -> str:
    """Strip ```json ... ``` fences from LLM output."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n", 1)
        text = lines[1] if len(lines) > 1 else ""
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()
