"""WhatsApp document collection flow — state machine.

Guides visitors through document collection for government services.
State stored in Redis per visitor. Each message advances the flow:

  identifying_service → verifying_docs → all_verified → completed/video_session

Dependencies: Redis, LLM (Qwen3), OCR (HybridOCRService), SessionLinkService.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Optional

from src.utils.logger import setup_logger

logger = setup_logger("services.document_flow")

FLOW_REDIS_PREFIX = "doc_flow:"
FLOW_TTL_SECONDS = 3600  # 1 hour


class FlowState:
    """Document flow state constants."""

    IDENTIFYING_SERVICE = "identifying_service"
    VERIFYING_DOCS = "verifying_docs"
    ALL_VERIFIED = "all_verified"
    COMPLETED = "completed"
    VIDEO_SESSION = "video_session"


@dataclass
class DocumentFlowResult:
    """What the flow returns for each message."""

    voice_text: str
    language: str = "ar"
    session_link: Optional[str] = None
    state: str = FlowState.IDENTIFYING_SERVICE
    notify_supervisor: bool = False
    supervisor_data: dict[str, Any] = field(default_factory=dict)


class DocumentFlowHandler:
    """State machine for WhatsApp document collection.

    Each visitor has a state stored in Redis.
    States: identifying_service -> verifying_docs ->
            all_verified -> completed / video_session
    """

    def __init__(
        self,
        redis_client: Any,
        llm_service: Any,
        ocr_service: Any,
        session_link_service: Any,
    ) -> None:
        self._redis = redis_client
        self._llm = llm_service
        self._ocr = ocr_service
        self._session_links = session_link_service

    # ── Main entry point ───────────────────────────────────────────────────

    async def handle(
        self,
        visitor_id: str,
        customer_id: str,
        avatar_id: str,
        message: dict[str, Any],
    ) -> DocumentFlowResult:
        """Process one message and return what to say back (voice text).

        Args:
            visitor_id: Unique visitor identifier (phone number).
            customer_id: Maskki customer ID.
            avatar_id: Employee/avatar ID for this customer.
            message: Dict with keys: type ("text"|"image"), content (str|bytes),
                     format (str, for images), r2_key (str, optional).
        """
        state = await self._load_state(visitor_id)
        msg_type = message.get("type", "text")
        language = state.get("language", "ar")
        status = state.get("status", FlowState.IDENTIFYING_SERVICE)

        logger.info(
            "Document flow step",
            extra={
                "visitor_prefix": visitor_id[:8],
                "status": status,
                "msg_type": msg_type,
            },
        )

        if status == FlowState.IDENTIFYING_SERVICE:
            return await self._identify_service(
                visitor_id, customer_id, avatar_id, message, state,
            )

        if status == FlowState.VERIFYING_DOCS and msg_type == "image":
            return await self._verify_document(
                visitor_id, customer_id, avatar_id, message, state,
            )

        if status == FlowState.VERIFYING_DOCS and msg_type == "text":
            return DocumentFlowResult(
                voice_text=_msg(
                    language,
                    ar="يرجى إرسال صورة المستند المطلوب",
                    en="Please send a photo of the required document",
                ),
                language=language,
                state=FlowState.VERIFYING_DOCS,
            )

        if status in (FlowState.ALL_VERIFIED, FlowState.COMPLETED, FlowState.VIDEO_SESSION):
            return DocumentFlowResult(
                voice_text=_msg(
                    language,
                    ar="تم استلام طلبك بنجاح. سيتواصل معك فريقنا قريباً",
                    en="Your request was received. Our team will contact you soon.",
                ),
                language=language,
                state=FlowState.COMPLETED,
            )

        # Fallback — restart
        return await self._restart(visitor_id, language)

    # ── State: IDENTIFYING_SERVICE ─────────────────────────────────────────

    async def _identify_service(
        self,
        visitor_id: str,
        customer_id: str,
        avatar_id: str,
        message: dict[str, Any],
        state: dict[str, Any],
    ) -> DocumentFlowResult:
        text = message.get("content", "")

        # Detect language from first message
        from src.services.language_detector import detect_language

        language = detect_language(text) if text else "ar"

        # Ask LLM to identify service
        from src.knowledge.services_kb import SERVICES, get_all_service_names_ar

        service_names = get_all_service_names_ar()
        service_ids = list(SERVICES.keys())

        prompt = (
            f'The user sent: "{text}"\n\n'
            f"Available services: {', '.join(service_names)}\n"
            f"Service IDs: {service_ids}\n\n"
            "Identify which service the user needs.\n"
            'Respond ONLY with valid JSON (no markdown):\n'
            '{"service_id": "<one of the IDs or null>", "confidence": 0.0-1.0}\n'
            "If unclear, set service_id to null."
        )

        service_id = None
        confidence = 0.0
        try:
            llm_result = await self._llm.generate(
                user_text=prompt,
                language=language,
                session_id=f"docflow_{visitor_id}",
            )
            parsed = _parse_json_from_llm(llm_result.text)
            service_id = parsed.get("service_id")
            confidence = float(parsed.get("confidence", 0.0))
        except Exception as exc:
            logger.error("LLM service identification failed", extra={"error": str(exc)})

        if not service_id or service_id not in SERVICES or confidence < 0.7:
            names = " / ".join(service_names)
            return DocumentFlowResult(
                voice_text=_msg(
                    language,
                    ar=f"أهلاً! ما الخدمة التي تريدها؟ نقدم: {names}",
                    en=f"Hello! What service do you need? We offer: {names}",
                ),
                language=language,
                state=FlowState.IDENTIFYING_SERVICE,
            )

        # Service identified — save state and ask for first document
        from src.knowledge.services_kb import get_required_docs

        service = SERVICES[service_id]
        required_docs = get_required_docs(service_id)
        doc_names_ar = [d["name_ar"] for d in required_docs]

        new_state = {
            "status": FlowState.VERIFYING_DOCS,
            "service_id": service_id,
            "language": language,
            "required_docs": required_docs,
            "verified_docs": {},
            "current_doc_index": 0,
            "customer_id": customer_id,
            "avatar_id": avatar_id,
            "started_at": datetime.utcnow().isoformat(),
        }
        await self._save_state(visitor_id, new_state)

        first_doc = required_docs[0]
        docs_list_ar = "، ".join(doc_names_ar)
        docs_list_en = ", ".join(d["name_en"] for d in required_docs)

        return DocumentFlowResult(
            voice_text=_msg(
                language,
                ar=(
                    f"أهلاً! سأساعدك في {service['name_ar']}. "
                    f"أحتاج منك {len(required_docs)} مستندات: {docs_list_ar}. "
                    f"لنبدأ بـ{first_doc['name_ar']}، أرسل صورة واضحة"
                ),
                en=(
                    f"Hello! I'll help you with {service['name_en']}. "
                    f"I need {len(required_docs)} documents: {docs_list_en}. "
                    f"Let's start with {first_doc['name_en']}, please send a clear photo."
                ),
            ),
            language=language,
            state=FlowState.VERIFYING_DOCS,
        )

    # ── State: VERIFYING_DOCS ──────────────────────────────────────────────

    async def _verify_document(
        self,
        visitor_id: str,
        customer_id: str,
        avatar_id: str,
        message: dict[str, Any],
        state: dict[str, Any],
    ) -> DocumentFlowResult:
        language = state.get("language", "ar")
        current_index = state.get("current_doc_index", 0)
        required_docs = state.get("required_docs", [])

        if current_index >= len(required_docs):
            return await self._all_docs_complete(
                visitor_id, customer_id, avatar_id, state,
            )

        current_doc = required_docs[current_index]
        image_bytes = message.get("content")
        image_format = message.get("format", "jpeg")

        # Run OCR
        try:
            ocr_result = await self._ocr.analyze_image(
                image_bytes=image_bytes,
                image_format=image_format,
                extract_fields=True,
            )
        except Exception as exc:
            logger.error("OCR failed in doc flow", extra={"error": str(exc)})
            return DocumentFlowResult(
                voice_text=_msg(
                    language,
                    ar="عذراً، لم أتمكن من تحليل الصورة. يرجى إرسال صورة أوضح",
                    en="Sorry, I couldn't analyze the image. Please send a clearer photo.",
                ),
                language=language,
                state=FlowState.VERIFYING_DOCS,
            )

        # Check quality
        if ocr_result.confidence < 0.5:
            return DocumentFlowResult(
                voice_text=_msg(
                    language,
                    ar="الصورة غير واضحة. يرجى التأكد من الإضاءة الجيدة وإرسال صورة أوضح",
                    en="The image is unclear. Please ensure good lighting and send a clearer photo.",
                ),
                language=language,
                state=FlowState.VERIFYING_DOCS,
            )

        # Check doc type matches expected
        expected_type = current_doc["id"]
        actual_type = ocr_result.document_type.value

        if (
            actual_type != "unknown"
            and actual_type != expected_type
            and ocr_result.confidence > 0.7
        ):
            return DocumentFlowResult(
                voice_text=_msg(
                    language,
                    ar=(
                        f"هذا المستند لا يبدو {current_doc['name_ar']}. "
                        f"يرجى إرسال {current_doc['name_ar']} الصحيح"
                    ),
                    en=(
                        f"This doesn't appear to be {current_doc['name_en']}. "
                        f"Please send the correct {current_doc['name_en']}."
                    ),
                ),
                language=language,
                state=FlowState.VERIFYING_DOCS,
            )

        # Check validity (expiry etc.)
        if not _is_document_valid(ocr_result):
            return DocumentFlowResult(
                voice_text=_msg(
                    language,
                    ar="المستند غير صالح أو منتهي الصلاحية. يرجى تجديده أولاً",
                    en="The document is invalid or expired. Please renew it first.",
                ),
                language=language,
                state=FlowState.VERIFYING_DOCS,
            )

        # Document verified
        state["verified_docs"][expected_type] = {
            "r2_key": message.get("r2_key", ""),
            "fields": ocr_result.fields,
            "verified_at": datetime.utcnow().isoformat(),
            "confidence": ocr_result.confidence,
        }
        state["current_doc_index"] = current_index + 1
        remaining = len(required_docs) - (current_index + 1)

        if remaining == 0:
            await self._save_state(visitor_id, state)
            return await self._all_docs_complete(
                visitor_id, customer_id, avatar_id, state,
            )

        # Ask for next document
        await self._save_state(visitor_id, state)
        next_doc = required_docs[current_index + 1]

        return DocumentFlowResult(
            voice_text=_msg(
                language,
                ar=(
                    f"تم التحقق من {current_doc['name_ar']}. "
                    f"متبقي {remaining} مستند. "
                    f"الآن أرسل {next_doc['name_ar']}"
                ),
                en=(
                    f"{current_doc['name_en']} verified. "
                    f"{remaining} document(s) remaining. "
                    f"Please send {next_doc['name_en']}."
                ),
            ),
            language=language,
            state=FlowState.VERIFYING_DOCS,
            notify_supervisor=True,
            supervisor_data={
                "event": "document_verified",
                "doc_type": expected_type,
                "remaining": remaining,
            },
        )

    # ── State: ALL_DOCS_COMPLETE ───────────────────────────────────────────

    async def _all_docs_complete(
        self,
        visitor_id: str,
        customer_id: str,
        avatar_id: str,
        state: dict[str, Any],
    ) -> DocumentFlowResult:
        from src.knowledge.services_kb import SERVICES

        language = state.get("language", "ar")
        service_id = state.get("service_id")
        service = SERVICES.get(service_id, {})
        fees = service.get("fees", {})

        state["status"] = FlowState.ALL_VERIFIED
        await self._save_state(visitor_id, state)

        # Does this service require a video session?
        if service.get("requires_video_session") and self._session_links:
            link_data = await self._session_links.create_link(
                customer_id=customer_id,
                avatar_id=avatar_id,
                language=language,
                service_type=service.get("name_ar"),
                collected_docs=state.get("verified_docs", {}),
                expires_minutes=30,
                channel_source="whatsapp",
            )

            state["status"] = FlowState.VIDEO_SESSION
            await self._save_state(visitor_id, state)

            return DocumentFlowResult(
                voice_text=_msg(
                    language,
                    ar=(
                        "ممتاز! جميع مستنداتك مكتملة وصحيحة. "
                        "معاملتك تحتاج مراجعة إضافية. "
                        "ستصلك رسالة برابط للتحدث مع موظفتنا مباشرة"
                    ),
                    en=(
                        "Excellent! All your documents are complete and valid. "
                        "Your request needs additional review. "
                        "You'll receive a link to video chat with our agent."
                    ),
                ),
                language=language,
                session_link=link_data["url"],
                state=FlowState.VIDEO_SESSION,
                notify_supervisor=True,
                supervisor_data={
                    "event": "transfer_to_video",
                    "service": service.get("name_ar"),
                    "session_url": link_data["url"],
                },
            )

        # No video needed — share fees
        amount = fees.get("amount", 0)
        currency = fees.get("currency", "OMR")
        time_ar = service.get("processing_time_ar", "قريباً")

        state["status"] = FlowState.COMPLETED
        await self._save_state(visitor_id, state)

        return DocumentFlowResult(
            voice_text=_msg(
                language,
                ar=(
                    "ممتاز! جميع مستنداتك مكتملة وصحيحة. "
                    f"رسوم {service.get('name_ar', 'المعاملة')}: "
                    f"{amount} {currency}. "
                    f"وقت الإنجاز: {time_ar}. "
                    "سيتواصل معك فريقنا قريباً لاستكمال الدفع"
                ),
                en=(
                    "Excellent! All documents verified. "
                    f"Service fee: {amount} {currency}. "
                    f"Processing time: {time_ar}. "
                    "Our team will contact you shortly."
                ),
            ),
            language=language,
            state=FlowState.COMPLETED,
            notify_supervisor=True,
            supervisor_data={
                "event": "request_ready",
                "service": service.get("name_ar"),
                "fees": f"{amount} {currency}",
                "docs_count": len(state.get("verified_docs", {})),
            },
        )

    # ── Helpers ────────────────────────────────────────────────────────────

    async def _restart(
        self, visitor_id: str, language: str,
    ) -> DocumentFlowResult:
        """Clear state and start over."""
        await self._redis.delete(f"{FLOW_REDIS_PREFIX}{visitor_id}")
        from src.knowledge.services_kb import get_all_service_names_ar

        names = " / ".join(get_all_service_names_ar())
        return DocumentFlowResult(
            voice_text=_msg(
                language,
                ar=f"أهلاً! كيف يمكنني مساعدتك؟ نقدم: {names}",
                en=f"Hello! How can I help? We offer: {names}",
            ),
            language=language,
            state=FlowState.IDENTIFYING_SERVICE,
        )

    async def _load_state(self, visitor_id: str) -> dict[str, Any]:
        data = await self._redis.get(f"{FLOW_REDIS_PREFIX}{visitor_id}")
        if not data:
            return {"status": FlowState.IDENTIFYING_SERVICE, "language": "ar"}
        return json.loads(data)

    async def _save_state(self, visitor_id: str, state: dict[str, Any]) -> None:
        await self._redis.setex(
            f"{FLOW_REDIS_PREFIX}{visitor_id}",
            FLOW_TTL_SECONDS,
            json.dumps(state, ensure_ascii=False),
        )


# ── Module-level helpers ─────────────────────────────────────────────────────


def _msg(language: str, ar: str, en: str) -> str:
    """Return Arabic or English text based on language."""
    return ar if language == "ar" else en


def _is_document_valid(ocr_result: Any) -> bool:
    """Check if document is valid (not expired, sufficient confidence)."""
    if ocr_result.confidence < 0.4:
        return False
    expiry = ocr_result.fields.get("expiry_date")
    if expiry:
        try:
            exp_date = date.fromisoformat(expiry)
            if exp_date < (date.today() + timedelta(days=90)):
                return False
        except ValueError:
            pass
    return True


def _parse_json_from_llm(text: str) -> dict[str, Any]:
    """Parse JSON from LLM output, stripping markdown fences if present."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n", 1)
        text = lines[1] if len(lines) > 1 else ""
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    return json.loads(text)
