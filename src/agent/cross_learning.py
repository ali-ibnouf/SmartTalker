"""Cross-Learning Engine (L3): Industry knowledge sharing.

Allows employees to adopt industry-specific knowledge templates.
When an employee joins an industry, its default Q&A pairs are copied
into EmployeeKnowledge (approved=True).

Level 3 additions:
- ``generalize_qa()``: uses qwen3-max to remove company-specific details
  from high-performing Q&A pairs.
- ``weekly_cross_learning_cycle()``: finds Q&A with success_rate > 0.8
  and times_used > 5, generalizes them, distributes to same-industry
  employees as "pending" learning items.

15 seed industries with Arabic names.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import httpx
from sqlalchemy import and_, func, select

from src.db.models import (
    Employee,
    EmployeeIndustry,
    EmployeeKnowledge,
    EmployeeLearning,
    IndustryCategory,
)
from src.utils.logger import setup_logger

logger = setup_logger("agent.cross_learning")

# Thresholds for cross-learning cycle
_MIN_SUCCESS_RATE = 0.8
_MIN_TIMES_USED = 5

# System prompt for Q&A generalization
_GENERALIZE_SYSTEM_PROMPT = (
    "You remove company-specific details from a Q&A pair so it becomes a "
    "generic industry template. Replace brand names, addresses, phone numbers, "
    "specific prices, and proprietary details with placeholders like "
    "[Company Name], [Phone Number], [Price], etc.\n\n"
    "Return a JSON object with exactly two keys:\n"
    '  - "question": the generalized question\n'
    '  - "answer": the generalized answer\n\n'
    "Keep the tone and structure. Do NOT add new information.\n"
    "Do NOT include any text outside the JSON object."
)

# ── 15 Seed Industries ──────────────────────────────────────────────────

SEED_INDUSTRIES: list[dict[str, Any]] = [
    {
        "slug": "ecommerce",
        "name": "E-commerce",
        "name_ar": "التجارة الإلكترونية",
        "description": "Online retail and e-commerce platforms",
        "default_qa_pairs": [
            {"q": "What are your return policies?", "a": "We accept returns within 30 days of purchase with a valid receipt."},
            {"q": "Do you offer free shipping?", "a": "We offer free shipping on orders over $50."},
            {"q": "How can I track my order?", "a": "You can track your order using the tracking number sent to your email."},
        ],
        "default_personality": {"tone": "friendly", "style": "helpful"},
    },
    {
        "slug": "banking",
        "name": "Banking",
        "name_ar": "البنوك",
        "description": "Banks, credit unions, and financial services",
        "default_qa_pairs": [
            {"q": "How do I open an account?", "a": "You can open an account online or visit any branch with a valid ID."},
            {"q": "What are your interest rates?", "a": "Our current rates vary by product. Please visit our rates page for details."},
            {"q": "How do I report a lost card?", "a": "Call our 24/7 hotline immediately to block your card and request a replacement."},
        ],
        "default_personality": {"tone": "formal", "style": "trustworthy"},
    },
    {
        "slug": "healthcare",
        "name": "Healthcare",
        "name_ar": "الرعاية الصحية",
        "description": "Hospitals, clinics, and medical practices",
        "default_qa_pairs": [
            {"q": "How do I book an appointment?", "a": "You can book online through our portal or call us during business hours."},
            {"q": "Do you accept insurance?", "a": "Yes, we accept most major insurance providers. Please contact us for specifics."},
            {"q": "What are your emergency hours?", "a": "Our emergency department is open 24/7."},
        ],
        "default_personality": {"tone": "professional", "style": "empathetic"},
    },
    {
        "slug": "telecom",
        "name": "Telecom",
        "name_ar": "الاتصالات",
        "description": "Internet, mobile, and phone services",
        "default_qa_pairs": [
            {"q": "What plans do you offer?", "a": "We offer a range of plans for mobile, internet, and bundled services."},
            {"q": "How do I check my data usage?", "a": "Check your usage in real-time through our app or website."},
            {"q": "My internet is slow, what should I do?", "a": "Try restarting your router. If the issue persists, contact technical support."},
        ],
        "default_personality": {"tone": "helpful", "style": "technical"},
    },
    {
        "slug": "real_estate",
        "name": "Real Estate",
        "name_ar": "العقارات",
        "description": "Property sales, rentals, and management",
        "default_qa_pairs": [
            {"q": "How do I schedule a viewing?", "a": "Contact us to schedule a property viewing at your convenience."},
            {"q": "What documents do I need to rent?", "a": "You'll need a valid ID, proof of income, and references."},
            {"q": "Do you handle property management?", "a": "Yes, we offer full property management services for landlords."},
        ],
        "default_personality": {"tone": "professional", "style": "knowledgeable"},
    },
    {
        "slug": "education",
        "name": "Education",
        "name_ar": "التعليم",
        "description": "Schools, universities, and online learning",
        "default_qa_pairs": [
            {"q": "How do I enroll?", "a": "Visit our admissions page to start the enrollment process."},
            {"q": "What courses do you offer?", "a": "We offer a wide range of courses. Browse our catalog for the full list."},
            {"q": "Are there scholarships available?", "a": "Yes, we offer merit-based and need-based scholarships. Check our financial aid page."},
        ],
        "default_personality": {"tone": "encouraging", "style": "informative"},
    },
    {
        "slug": "travel",
        "name": "Travel",
        "name_ar": "السفر والسياحة",
        "description": "Hotels, travel agencies, and tourism",
        "default_qa_pairs": [
            {"q": "What time is check-in?", "a": "Check-in is at 3:00 PM and check-out is at 11:00 AM."},
            {"q": "Do you have room service?", "a": "Yes, room service is available 24 hours a day."},
            {"q": "Is WiFi free?", "a": "Complimentary high-speed WiFi is available throughout the property."},
        ],
        "default_personality": {"tone": "warm", "style": "welcoming"},
    },
    {
        "slug": "government",
        "name": "Government",
        "name_ar": "الخدمات الحكومية",
        "description": "Municipal services, permits, and public offices",
        "default_qa_pairs": [
            {"q": "How do I apply for a permit?", "a": "Applications are accepted online or in person at our office."},
            {"q": "What are your office hours?", "a": "We are open Sunday through Thursday, 8 AM to 3 PM."},
            {"q": "How do I pay my utility bill?", "a": "Pay online, through our app, at authorized payment centers, or at our office."},
        ],
        "default_personality": {"tone": "formal", "style": "informative"},
    },
    {
        "slug": "automotive",
        "name": "Automotive",
        "name_ar": "السيارات",
        "description": "Car dealerships, repair shops, and auto services",
        "default_qa_pairs": [
            {"q": "Can I schedule a test drive?", "a": "Absolutely! Book a test drive online or visit our showroom."},
            {"q": "Do you offer financing?", "a": "Yes, we offer competitive financing options with flexible terms."},
            {"q": "How do I book a service appointment?", "a": "Schedule service online or call our service department."},
        ],
        "default_personality": {"tone": "enthusiastic", "style": "consultative"},
    },
    {
        "slug": "food",
        "name": "Food & Restaurants",
        "name_ar": "المطاعم والأغذية",
        "description": "Restaurants, cafes, and food services",
        "default_qa_pairs": [
            {"q": "Can I make a reservation?", "a": "Yes, you can reserve a table online or by calling us."},
            {"q": "Do you cater for dietary restrictions?", "a": "We offer vegetarian, vegan, and gluten-free options."},
            {"q": "Do you offer delivery?", "a": "Yes, we deliver through our website and major delivery platforms."},
        ],
        "default_personality": {"tone": "friendly", "style": "casual"},
    },
    {
        "slug": "legal",
        "name": "Legal",
        "name_ar": "الخدمات القانونية",
        "description": "Law firms, legal consultancy, and notary",
        "default_qa_pairs": [
            {"q": "How do I schedule a consultation?", "a": "Book a consultation through our website or call our office."},
            {"q": "What areas of law do you practice?", "a": "We practice corporate, family, immigration, and real estate law."},
            {"q": "What are your fees?", "a": "Fees vary by case type. We offer a free initial consultation to discuss your needs."},
        ],
        "default_personality": {"tone": "formal", "style": "precise"},
    },
    {
        "slug": "logistics",
        "name": "Logistics",
        "name_ar": "الخدمات اللوجستية",
        "description": "Freight, courier, and supply chain services",
        "default_qa_pairs": [
            {"q": "How do I track a shipment?", "a": "Enter your tracking number on our website or app for real-time updates."},
            {"q": "What are your delivery timeframes?", "a": "Delivery times vary by destination. Check our rates page for estimates."},
            {"q": "Do you ship internationally?", "a": "Yes, we offer international shipping to over 200 countries."},
        ],
        "default_personality": {"tone": "efficient", "style": "clear"},
    },
    {
        "slug": "hr",
        "name": "HR & Recruitment",
        "name_ar": "الموارد البشرية",
        "description": "Human resources, staffing, and recruitment agencies",
        "default_qa_pairs": [
            {"q": "How do I apply for a position?", "a": "Submit your application through our careers portal with your CV."},
            {"q": "What benefits do you offer?", "a": "We offer health insurance, paid time off, and professional development opportunities."},
            {"q": "How long does the hiring process take?", "a": "Typically 2-4 weeks from application to offer, depending on the role."},
        ],
        "default_personality": {"tone": "professional", "style": "supportive"},
    },
    {
        "slug": "technology",
        "name": "Technology",
        "name_ar": "التكنولوجيا",
        "description": "Software, SaaS, and tech companies",
        "default_qa_pairs": [
            {"q": "Do you offer a free trial?", "a": "Yes, we offer a 14-day free trial with full access to all features."},
            {"q": "How do I integrate with my existing tools?", "a": "We offer REST APIs and native integrations with popular platforms."},
            {"q": "What support options are available?", "a": "We provide email, chat, and phone support depending on your plan tier."},
        ],
        "default_personality": {"tone": "professional", "style": "technical"},
    },
    {
        "slug": "nonprofit",
        "name": "Nonprofit",
        "name_ar": "المنظمات غير الربحية",
        "description": "Charities, NGOs, and nonprofit organizations",
        "default_qa_pairs": [
            {"q": "How can I donate?", "a": "You can donate online through our website, by bank transfer, or in person."},
            {"q": "How do I volunteer?", "a": "Visit our volunteer page to see current opportunities and sign up."},
            {"q": "Where does the money go?", "a": "Our annual report details how every dollar is allocated to our programs."},
        ],
        "default_personality": {"tone": "grateful", "style": "transparent"},
    },
]


class CrossLearningEngine:
    """Manages industry-based knowledge sharing across employees.

    Level 3 features:
    - ``generalize_qa()``: LLM-based removal of company-specific details
    - ``weekly_cross_learning_cycle()``: find high-performing Q&A, generalize,
      distribute to same-industry employees as pending learning items
    """

    def __init__(self, db: Any, config: Any = None):
        self._db = db
        self._config = config
        self._client: Optional[httpx.AsyncClient] = None

    def _get_llm_client(self) -> httpx.AsyncClient:
        """Lazy-init the LLM client."""
        if self._client is None:
            if self._config is None:
                from src.config import get_settings
                self._config = get_settings()
            api_key = getattr(self._config, "llm_api_key", "") or getattr(self._config, "dashscope_api_key", "")
            self._client = httpx.AsyncClient(
                base_url=self._config.llm_base_url,
                timeout=httpx.Timeout(30.0, connect=10.0),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.close()

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
                    description=ind.get("name_ar", "") + " — " + ind["description"],
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

    # ── Level 3: Generalize + Distribute ─────────────────────────────────

    async def generalize_qa(self, question: str, answer: str) -> dict:
        """Use qwen3-max to remove company-specific details from a Q&A pair.

        Args:
            question: The original question.
            answer: The original answer.

        Returns:
            Dict with "question" and "answer" keys (generalized).
        """
        client = self._get_llm_client()
        model = getattr(self._config, "llm_model_name", "qwen3-max")

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": _GENERALIZE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n"
                        f"Answer: {answer}"
                    ),
                },
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }

        response = await client.post("/chat/completions", json=payload)
        response.raise_for_status()

        data = response.json()
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )

        if not content:
            return {"question": question, "answer": answer}

        try:
            parsed = json.loads(content)
            return {
                "question": parsed.get("question", question),
                "answer": parsed.get("answer", answer),
            }
        except json.JSONDecodeError:
            logger.warning("LLM returned invalid JSON for generalization")
            return {"question": question, "answer": answer}

    async def weekly_cross_learning_cycle(self) -> dict:
        """Find high-performing Q&A, generalize, distribute to same-industry employees.

        Criteria: success_rate > 0.8 AND times_used > 5.
        Generalized pairs are added as EmployeeLearning (status="pending")
        to other employees in the same industry.

        Returns:
            Summary dict with counts.
        """
        if self._db is None:
            return {"error": "Database not available"}

        generalized = 0
        distributed = 0

        async with self._db.session() as session:
            # Find high-performing Q&A pairs
            high_perf = await session.execute(
                select(EmployeeKnowledge).where(
                    and_(
                        EmployeeKnowledge.approved == True,  # noqa: E712
                        EmployeeKnowledge.success_rate >= _MIN_SUCCESS_RATE,
                        EmployeeKnowledge.times_used >= _MIN_TIMES_USED,
                    )
                )
            )
            candidates = high_perf.scalars().all()

            if not candidates:
                logger.info("No high-performing Q&A found for cross-learning")
                return {"generalized": 0, "distributed": 0, "candidates": 0}

            for qa in candidates:
                # Find the source employee's industries
                emp_industries = await session.execute(
                    select(EmployeeIndustry.industry_id).where(
                        EmployeeIndustry.employee_id == qa.employee_id
                    )
                )
                industry_ids = [row[0] for row in emp_industries.all()]
                if not industry_ids:
                    continue

                # Generalize the Q&A
                try:
                    gen = await self.generalize_qa(qa.question, qa.answer)
                except Exception as exc:
                    logger.warning(f"Generalization failed for {qa.id}: {exc}")
                    continue

                generalized += 1

                # Find same-industry employees (excluding source)
                peers = await session.execute(
                    select(EmployeeIndustry.employee_id).where(
                        and_(
                            EmployeeIndustry.industry_id.in_(industry_ids),
                            EmployeeIndustry.employee_id != qa.employee_id,
                        )
                    )
                )
                peer_ids = list({row[0] for row in peers.all()})

                for peer_id in peer_ids:
                    # Check for existing duplicate
                    existing = await session.execute(
                        select(EmployeeKnowledge).where(
                            and_(
                                EmployeeKnowledge.employee_id == peer_id,
                                EmployeeKnowledge.question == gen["question"],
                            )
                        )
                    )
                    if existing.scalar_one_or_none() is not None:
                        continue

                    # Check if already suggested as pending learning
                    existing_learning = await session.execute(
                        select(EmployeeLearning).where(
                            and_(
                                EmployeeLearning.employee_id == peer_id,
                                EmployeeLearning.new_value == gen["question"],
                                EmployeeLearning.status == "pending",
                            )
                        )
                    )
                    if existing_learning.scalar_one_or_none() is not None:
                        continue

                    # Get peer's customer_id
                    peer_emp = await session.execute(
                        select(Employee.customer_id).where(Employee.id == peer_id)
                    )
                    peer_row = peer_emp.first()
                    peer_customer_id = peer_row[0] if peer_row else ""

                    # Add as pending learning item
                    learning = EmployeeLearning(
                        employee_id=peer_id,
                        customer_id=peer_customer_id,
                        learning_type="qa_pair",
                        old_value=gen["question"],
                        new_value=gen["answer"],
                        confidence=qa.success_rate,
                        status="pending",
                        source="cross_learning",
                    )
                    session.add(learning)
                    distributed += 1

            await session.commit()

        logger.info(
            "Cross-learning cycle complete",
            extra={
                "candidates": len(candidates),
                "generalized": generalized,
                "distributed": distributed,
            },
        )
        return {
            "candidates": len(candidates),
            "generalized": generalized,
            "distributed": distributed,
        }
