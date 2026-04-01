"""Government services knowledge base.

Static catalogue of services, their document requirements, fees,
and processing times. Used by DocumentFlowHandler to guide visitors
through document collection on WhatsApp.
"""

from __future__ import annotations

from typing import Any

SERVICES: dict[str, dict[str, Any]] = {
    "license_renewal": {
        "name_ar": "تجديد رخصة القيادة",
        "name_en": "Driver's License Renewal",
        "documents_required": [
            {
                "id": "national_id",
                "name_ar": "البطاقة الهوية الوطنية",
                "name_en": "National ID",
                "required": True,
            },
            {
                "id": "driving_license",
                "name_ar": "الرخصة القديمة",
                "name_en": "Old License",
                "required": True,
            },
            {
                "id": "eye_test_certificate",
                "name_ar": "شهادة فحص النظر",
                "name_en": "Eye Test Certificate",
                "required": True,
            },
        ],
        "fees": {"amount": 20, "currency": "OMR"},
        "processing_time_ar": "فوري",
        "requires_video_session": False,
    },
    "residency_renewal": {
        "name_ar": "تجديد الإقامة",
        "name_en": "Residency Renewal",
        "documents_required": [
            {
                "id": "passport",
                "name_ar": "جواز السفر",
                "name_en": "Passport",
                "required": True,
            },
            {
                "id": "national_id",
                "name_ar": "البطاقة الهوية",
                "name_en": "National ID",
                "required": True,
            },
            {
                "id": "work_contract",
                "name_ar": "عقد العمل",
                "name_en": "Work Contract",
                "required": True,
            },
            {
                "id": "salary_slip",
                "name_ar": "كشف الراتب",
                "name_en": "Salary Slip",
                "required": False,
            },
        ],
        "fees": {"amount": 30, "currency": "OMR"},
        "processing_time_ar": "يوم عمل واحد",
        "requires_video_session": False,
    },
    "visa_application": {
        "name_ar": "طلب تأشيرة",
        "name_en": "Visa Application",
        "documents_required": [
            {
                "id": "passport",
                "name_ar": "جواز السفر",
                "name_en": "Passport",
                "required": True,
            },
            {
                "id": "national_id",
                "name_ar": "البطاقة الهوية",
                "name_en": "National ID",
                "required": True,
            },
        ],
        "fees": {"amount": 50, "currency": "OMR"},
        "processing_time_ar": "3-5 أيام عمل",
        "requires_video_session": True,
    },
    "vehicle_registration": {
        "name_ar": "تسجيل مركبة",
        "name_en": "Vehicle Registration",
        "documents_required": [
            {
                "id": "national_id",
                "name_ar": "البطاقة الهوية",
                "name_en": "National ID",
                "required": True,
            },
            {
                "id": "invoice",
                "name_ar": "فاتورة الشراء",
                "name_en": "Purchase Invoice",
                "required": True,
            },
        ],
        "fees": {"amount": 15, "currency": "OMR"},
        "processing_time_ar": "فوري",
        "requires_video_session": False,
    },
}


def get_service(service_id: str) -> dict[str, Any] | None:
    """Get a service definition by ID."""
    return SERVICES.get(service_id)


def get_all_service_names_ar() -> list[str]:
    """Get list of all service names in Arabic."""
    return [s["name_ar"] for s in SERVICES.values()]


def get_required_docs(service_id: str) -> list[dict[str, Any]]:
    """Get required documents for a service (excludes optional docs)."""
    service = SERVICES.get(service_id, {})
    return [d for d in service.get("documents_required", []) if d["required"]]
