"""Cross-Learning Engine (L3): Industry knowledge sharing.

Allows employees to adopt industry-specific knowledge templates.
When an employee joins an industry, its default Q&A pairs are copied
into EmployeeKnowledge (approved=True). Admin can view cross-learning
stats across industries.

15 seed industries are pre-defined with default Q&A pairs.
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import (
    Employee,
    EmployeeIndustry,
    EmployeeKnowledge,
    IndustryCategory,
)
from src.utils.logger import setup_logger

logger = setup_logger("agent.cross_learning")

# ── 15 Seed Industries ──────────────────────────────────────────────────

SEED_INDUSTRIES: list[dict[str, Any]] = [
    {
        "slug": "retail",
        "name": "Retail & E-commerce",
        "description": "Online and physical retail stores",
        "default_qa_pairs": [
            {"q": "What are your return policies?", "a": "We accept returns within 30 days of purchase with a valid receipt."},
            {"q": "Do you offer free shipping?", "a": "We offer free shipping on orders over $50."},
            {"q": "How can I track my order?", "a": "You can track your order using the tracking number sent to your email."},
        ],
        "default_personality": {"tone": "friendly", "style": "helpful"},
    },
    {
        "slug": "healthcare",
        "name": "Healthcare & Medical",
        "description": "Hospitals, clinics, and medical practices",
        "default_qa_pairs": [
            {"q": "How do I book an appointment?", "a": "You can book online through our portal or call us during business hours."},
            {"q": "Do you accept insurance?", "a": "Yes, we accept most major insurance providers. Please contact us for specifics."},
            {"q": "What are your emergency hours?", "a": "Our emergency department is open 24/7."},
        ],
        "default_personality": {"tone": "professional", "style": "empathetic"},
    },
    {
        "slug": "hospitality",
        "name": "Hospitality & Hotels",
        "description": "Hotels, resorts, and travel accommodation",
        "default_qa_pairs": [
            {"q": "What time is check-in?", "a": "Check-in is at 3:00 PM and check-out is at 11:00 AM."},
            {"q": "Do you have room service?", "a": "Yes, room service is available 24 hours a day."},
            {"q": "Is WiFi free?", "a": "Complimentary high-speed WiFi is available throughout the property."},
        ],
        "default_personality": {"tone": "warm", "style": "welcoming"},
    },
    {
        "slug": "banking",
        "name": "Banking & Finance",
        "description": "Banks, credit unions, and financial services",
        "default_qa_pairs": [
            {"q": "How do I open an account?", "a": "You can open an account online or visit any branch with a valid ID."},
            {"q": "What are your interest rates?", "a": "Our current rates vary by product. Please visit our rates page for details."},
            {"q": "How do I report a lost card?", "a": "Call our 24/7 hotline immediately to block your card and request a replacement."},
        ],
        "default_personality": {"tone": "formal", "style": "trustworthy"},
    },
    {
        "slug": "education",
        "name": "Education & Training",
        "description": "Schools, universities, and online learning",
        "default_qa_pairs": [
            {"q": "How do I enroll?", "a": "Visit our admissions page to start the enrollment process."},
            {"q": "What courses do you offer?", "a": "We offer a wide range of courses. Browse our catalog for the full list."},
            {"q": "Are there scholarships available?", "a": "Yes, we offer merit-based and need-based scholarships. Check our financial aid page."},
        ],
        "default_personality": {"tone": "encouraging", "style": "informative"},
    },
    {
        "slug": "real_estate",
        "name": "Real Estate",
        "description": "Property sales, rentals, and management",
        "default_qa_pairs": [
            {"q": "How do I schedule a viewing?", "a": "Contact us to schedule a property viewing at your convenience."},
            {"q": "What documents do I need to rent?", "a": "You'll need a valid ID, proof of income, and references."},
            {"q": "Do you handle property management?", "a": "Yes, we offer full property management services for landlords."},
        ],
        "default_personality": {"tone": "professional", "style": "knowledgeable"},
    },
    {
        "slug": "restaurants",
        "name": "Restaurants & Food Service",
        "description": "Restaurants, cafes, and catering",
        "default_qa_pairs": [
            {"q": "Can I make a reservation?", "a": "Yes, you can reserve a table online or by calling us."},
            {"q": "Do you cater for dietary restrictions?", "a": "We offer vegetarian, vegan, and gluten-free options."},
            {"q": "Do you offer delivery?", "a": "Yes, we deliver through our website and major delivery platforms."},
        ],
        "default_personality": {"tone": "friendly", "style": "casual"},
    },
    {
        "slug": "automotive",
        "name": "Automotive & Dealerships",
        "description": "Car dealerships, repair shops, and auto services",
        "default_qa_pairs": [
            {"q": "Can I schedule a test drive?", "a": "Absolutely! Book a test drive online or visit our showroom."},
            {"q": "Do you offer financing?", "a": "Yes, we offer competitive financing options with flexible terms."},
            {"q": "How do I book a service appointment?", "a": "Schedule service online or call our service department."},
        ],
        "default_personality": {"tone": "enthusiastic", "style": "consultative"},
    },
    {
        "slug": "insurance",
        "name": "Insurance",
        "description": "Health, auto, home, and life insurance",
        "default_qa_pairs": [
            {"q": "How do I file a claim?", "a": "File a claim online through your account portal or call our claims department."},
            {"q": "What does my policy cover?", "a": "Coverage details are in your policy documents. Contact us for a detailed review."},
            {"q": "How do I get a quote?", "a": "Get an instant quote online or speak with one of our agents."},
        ],
        "default_personality": {"tone": "reassuring", "style": "clear"},
    },
    {
        "slug": "telecom",
        "name": "Telecommunications",
        "description": "Internet, mobile, and phone services",
        "default_qa_pairs": [
            {"q": "What plans do you offer?", "a": "We offer a range of plans for mobile, internet, and bundled services."},
            {"q": "How do I check my data usage?", "a": "Check your usage in real-time through our app or website."},
            {"q": "My internet is slow, what should I do?", "a": "Try restarting your router. If the issue persists, contact technical support."},
        ],
        "default_personality": {"tone": "helpful", "style": "technical"},
    },
    {
        "slug": "legal",
        "name": "Legal Services",
        "description": "Law firms, legal consultancy, and notary",
        "default_qa_pairs": [
            {"q": "How do I schedule a consultation?", "a": "Book a consultation through our website or call our office."},
            {"q": "What areas of law do you practice?", "a": "We practice corporate, family, immigration, and real estate law."},
            {"q": "What are your fees?", "a": "Fees vary by case type. We offer a free initial consultation to discuss your needs."},
        ],
        "default_personality": {"tone": "formal", "style": "precise"},
    },
    {
        "slug": "government",
        "name": "Government & Public Services",
        "description": "Municipal services, permits, and public offices",
        "default_qa_pairs": [
            {"q": "How do I apply for a permit?", "a": "Applications are accepted online or in person at our office."},
            {"q": "What are your office hours?", "a": "We are open Sunday through Thursday, 8 AM to 3 PM."},
            {"q": "How do I pay my utility bill?", "a": "Pay online, through our app, at authorized payment centers, or at our office."},
        ],
        "default_personality": {"tone": "formal", "style": "informative"},
    },
    {
        "slug": "fitness",
        "name": "Fitness & Wellness",
        "description": "Gyms, spas, and wellness centers",
        "default_qa_pairs": [
            {"q": "What are your membership plans?", "a": "We offer monthly, quarterly, and annual memberships with various tiers."},
            {"q": "Do you offer personal training?", "a": "Yes, certified personal trainers are available for one-on-one sessions."},
            {"q": "What are your operating hours?", "a": "We're open from 6 AM to 11 PM, seven days a week."},
        ],
        "default_personality": {"tone": "energetic", "style": "motivating"},
    },
    {
        "slug": "logistics",
        "name": "Logistics & Shipping",
        "description": "Freight, courier, and supply chain services",
        "default_qa_pairs": [
            {"q": "How do I track a shipment?", "a": "Enter your tracking number on our website or app for real-time updates."},
            {"q": "What are your delivery timeframes?", "a": "Delivery times vary by destination. Check our rates page for estimates."},
            {"q": "Do you ship internationally?", "a": "Yes, we offer international shipping to over 200 countries."},
        ],
        "default_personality": {"tone": "efficient", "style": "clear"},
    },
    {
        "slug": "saas",
        "name": "SaaS & Technology",
        "description": "Software-as-a-service and tech companies",
        "default_qa_pairs": [
            {"q": "Do you offer a free trial?", "a": "Yes, we offer a 14-day free trial with full access to all features."},
            {"q": "How do I integrate with my existing tools?", "a": "We offer REST APIs and native integrations with popular platforms."},
            {"q": "What support options are available?", "a": "We provide email, chat, and phone support depending on your plan tier."},
        ],
        "default_personality": {"tone": "professional", "style": "technical"},
    },
]


class CrossLearningEngine:
    """Manages industry-based knowledge sharing across employees."""

    def __init__(self, db: Any):
        self._db = db

    async def seed_industries(self) -> int:
        """Insert seed industries if not already present. Returns count inserted."""
        if self._db is None:
            return 0

        inserted = 0
        async with self._db.session() as session:
            for ind in SEED_INDUSTRIES:
                existing = await session.execute(
                    select(IndustryCategory).where(IndustryCategory.slug == ind["slug"])
                )
                if existing.scalar_one_or_none() is not None:
                    continue

                record = IndustryCategory(
                    slug=ind["slug"],
                    name=ind["name"],
                    description=ind["description"],
                    default_qa_pairs=json.dumps(ind["default_qa_pairs"]),
                    default_personality=json.dumps(ind["default_personality"]),
                )
                session.add(record)
                inserted += 1

            await session.commit()

        if inserted:
            logger.info(f"Seeded {inserted} industries")
        return inserted

    async def list_industries(self) -> list[dict]:
        """Return all industry categories."""
        if self._db is None:
            return []

        async with self._db.session() as session:
            result = await session.execute(
                select(IndustryCategory).order_by(IndustryCategory.name)
            )
            rows = result.scalars().all()
            return [
                {
                    "id": r.id,
                    "slug": r.slug,
                    "name": r.name,
                    "description": r.description,
                    "employee_count": r.employee_count,
                    "qa_count": len(json.loads(r.default_qa_pairs or "[]")),
                }
                for r in rows
            ]

    async def adopt_industry(
        self, employee_id: str, industry_slug: str
    ) -> dict:
        """Adopt an industry's knowledge templates for an employee.

        Copies the industry's default Q&A pairs into EmployeeKnowledge.
        Returns {"adopted": count, "industry": name}.
        """
        if self._db is None:
            return {"error": "Database not available"}

        async with self._db.session() as session:
            # Find industry
            result = await session.execute(
                select(IndustryCategory).where(IndustryCategory.slug == industry_slug)
            )
            industry = result.scalar_one_or_none()
            if industry is None:
                return {"error": f"Industry not found: {industry_slug}"}

            # Check employee exists
            emp_result = await session.execute(
                select(Employee).where(Employee.id == employee_id)
            )
            if emp_result.scalar_one_or_none() is None:
                return {"error": f"Employee not found: {employee_id}"}

            # Check if already adopted
            existing = await session.execute(
                select(EmployeeIndustry).where(
                    EmployeeIndustry.employee_id == employee_id,
                    EmployeeIndustry.industry_id == industry.id,
                )
            )
            if existing.scalar_one_or_none() is not None:
                return {"error": "Industry already adopted"}

            # Create mapping
            mapping = EmployeeIndustry(
                employee_id=employee_id,
                industry_id=industry.id,
            )
            session.add(mapping)

            # Copy Q&A pairs
            qa_pairs = json.loads(industry.default_qa_pairs or "[]")
            adopted = 0
            for qa in qa_pairs:
                q = qa.get("q", "")
                a = qa.get("a", "")
                if not q or not a:
                    continue

                # Check for duplicate questions
                dup = await session.execute(
                    select(EmployeeKnowledge).where(
                        EmployeeKnowledge.employee_id == employee_id,
                        EmployeeKnowledge.question == q,
                    )
                )
                if dup.scalar_one_or_none() is not None:
                    continue

                kb_entry = EmployeeKnowledge(
                    employee_id=employee_id,
                    category=industry.slug,
                    question=q,
                    answer=a,
                    approved=True,
                )
                session.add(kb_entry)
                adopted += 1

            # Increment employee count
            industry.employee_count = (industry.employee_count or 0) + 1

            await session.commit()

        logger.info(
            f"Employee {employee_id} adopted industry {industry_slug}: {adopted} Q&A pairs"
        )
        return {"adopted": adopted, "industry": industry.name}

    async def get_stats(self) -> dict:
        """Return cross-learning statistics for admin dashboard."""
        if self._db is None:
            return {"total_industries": 0, "total_adoptions": 0, "industries": []}

        async with self._db.session() as session:
            # Count total adoptions
            adoption_count = await session.execute(
                select(func.count(EmployeeIndustry.id))
            )
            total_adoptions = adoption_count.scalar() or 0

            # Industry stats
            result = await session.execute(
                select(IndustryCategory).order_by(IndustryCategory.employee_count.desc())
            )
            industries = result.scalars().all()

            return {
                "total_industries": len(industries),
                "total_adoptions": total_adoptions,
                "industries": [
                    {
                        "slug": ind.slug,
                        "name": ind.name,
                        "employee_count": ind.employee_count,
                        "qa_count": len(json.loads(ind.default_qa_pairs or "[]")),
                    }
                    for ind in industries
                ],
            }
