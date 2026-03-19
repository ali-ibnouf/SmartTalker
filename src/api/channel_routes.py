"""Channel management API routes.

Endpoints for channel CRUD, QR codes, embed code, connect page,
and auto-generated documentation.
"""

from __future__ import annotations

from typing import Any

import html

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

from src.utils.logger import setup_logger

logger = setup_logger("api.channel_routes")

router = APIRouter(prefix="/api/v1", tags=["channels"])


# ═══════════════════════════════════════════════════════════════════
# Connect Page (Public — no auth)
# ═══════════════════════════════════════════════════════════════════


@router.get("/connect/{employee_id}")
async def get_connect_page_data(employee_id: str, request: Request):
    """Public endpoint — returns data for QR landing page.

    No API key required. Returns employee info + available channels.
    """
    db = getattr(request.app.state, "db", None)
    if db is None:
        return JSONResponse(status_code=503, content={"error": "Service unavailable"})

    try:
        from sqlalchemy import select
        from src.db.models import Customer, Employee, EmployeeChannel

        async with db.session_ctx() as session:
            # Get employee
            stmt = select(Employee).where(
                Employee.id == employee_id,
                Employee.is_active == True,  # noqa: E712
            )
            result = await session.execute(stmt)
            employee = result.scalar()

            if employee is None:
                return JSONResponse(status_code=404, content={"error": "Employee not found"})

            # Get customer
            cust_result = await session.execute(
                select(Customer).where(Customer.id == employee.customer_id)
            )
            customer = cust_result.scalar()

            # Get channels
            ch_result = await session.execute(
                select(EmployeeChannel).where(
                    EmployeeChannel.employee_id == employee_id,
                    EmployeeChannel.enabled == True,  # noqa: E712
                )
            )
            channels = ch_result.scalars().all()

            channel_data: dict[str, Any] = {
                "widget": {"enabled": True},  # Widget always available
            }
            for ch in channels:
                if ch.channel_type == "whatsapp":
                    channel_data["whatsapp"] = {
                        "enabled": True,
                        "phone": ch.wa_phone_display,
                    }
                elif ch.channel_type == "telegram":
                    channel_data["telegram"] = {
                        "enabled": True,
                        "bot_username": ch.tg_bot_username,
                    }

            # Get avatar photo URL
            avatar = None
            if employee.avatar_id:
                from src.db.models import Avatar
                av_result = await session.execute(
                    select(Avatar).where(Avatar.id == employee.avatar_id)
                )
                avatar = av_result.scalar()

            return {
                "employee_name": employee.name,
                "role_title": employee.role_title,
                "photo_url": avatar.photo_url if avatar else "",
                "company_name": customer.company if customer else "",
                "channels": channel_data,
            }
    except Exception as exc:
        logger.error(f"Connect page error: {exc}")
        return JSONResponse(status_code=500, content={"error": "Internal server error"})


# ═══════════════════════════════════════════════════════════════════
# Embed Code
# ═══════════════════════════════════════════════════════════════════


@router.get("/employees/{employee_id}/embed-code")
async def get_embed_code(employee_id: str, request: Request):
    """Generate embed code snippet for customer's website."""
    db = getattr(request.app.state, "db", None)
    if db is None:
        return _embed_snippet(employee_id, "AI Assistant", "en")

    try:
        from sqlalchemy import select
        from src.db.models import Employee

        async with db.session_ctx() as session:
            result = await session.execute(
                select(Employee).where(Employee.id == employee_id)
            )
            employee = result.scalar()
            name = employee.name if employee else "AI Assistant"
            lang = employee.language if employee else "en"

    except Exception:
        name = "AI Assistant"
        lang = "en"

    return _embed_snippet(employee_id, name, lang)


def _embed_snippet(employee_id: str, name: str, language: str) -> dict:
    safe_id = html.escape(employee_id, quote=True)
    safe_name = html.escape(name, quote=True)
    safe_lang = html.escape(language, quote=True)
    snippet = (
        f'<!-- Maskki AI Agent Widget -->\n'
        f'<script\n'
        f'  src="https://maskki.com/widget.js"\n'
        f'  data-employee-id="{safe_id}"\n'
        f'  data-position="bottom-right"\n'
        f'  data-greeting="{safe_name} is here to help!"\n'
        f'  data-language="{safe_lang}"\n'
        f'  data-theme="auto"\n'
        f'  async>\n'
        f'</script>'
    )
    return {
        "html_snippet": snippet,
        "employee_name": name,
        "instructions": [
            "Copy the code above",
            "Paste it before </body> in your website HTML",
            "The widget will appear in the bottom-right corner",
        ],
    }


# ═══════════════════════════════════════════════════════════════════
# Channel CRUD
# ═══════════════════════════════════════════════════════════════════


@router.get("/employees/{employee_id}/channels")
async def list_channels(employee_id: str, request: Request):
    """List all channel configurations for an employee."""
    db = getattr(request.app.state, "db", None)
    if db is None:
        return {"channels": [], "count": 0}

    from sqlalchemy import select
    from src.db.models import EmployeeChannel

    async with db.session_ctx() as session:
        result = await session.execute(
            select(EmployeeChannel).where(
                EmployeeChannel.employee_id == employee_id
            )
        )
        channels = result.scalars().all()
        return {
            "channels": [
                {
                    "id": ch.id,
                    "channel_type": ch.channel_type,
                    "enabled": ch.enabled,
                    "wa_phone_display": ch.wa_phone_display if ch.channel_type == "whatsapp" else None,
                    "tg_bot_username": ch.tg_bot_username if ch.channel_type == "telegram" else None,
                    "widget_domain": ch.widget_domain if ch.channel_type == "widget" else None,
                    "qr_code_url": ch.qr_code_url,
                    "created_at": ch.created_at.isoformat() if ch.created_at else None,
                }
                for ch in channels
            ],
            "count": len(channels),
        }


@router.post("/employees/{employee_id}/channels")
async def create_channel(employee_id: str, request: Request):
    """Create or update a channel configuration for an employee."""
    db = getattr(request.app.state, "db", None)
    if db is None:
        return JSONResponse(status_code=503, content={"error": "Database unavailable"})

    body = await request.json()
    channel_type = body.get("channel_type", "")
    if channel_type not in ("whatsapp", "telegram", "widget"):
        return JSONResponse(status_code=400, content={"error": "Invalid channel_type"})

    from sqlalchemy import select
    from src.db.models import EmployeeChannel, Employee

    async with db.session_ctx() as session:
        # Get employee for customer_id
        emp_result = await session.execute(
            select(Employee).where(Employee.id == employee_id)
        )
        employee = emp_result.scalar()
        if employee is None:
            return JSONResponse(status_code=404, content={"error": "Employee not found"})

        # Upsert channel
        existing_result = await session.execute(
            select(EmployeeChannel).where(
                EmployeeChannel.employee_id == employee_id,
                EmployeeChannel.channel_type == channel_type,
            )
        )
        channel = existing_result.scalar()

        if channel is None:
            channel = EmployeeChannel(
                employee_id=employee_id,
                customer_id=employee.customer_id,
                channel_type=channel_type,
            )
            session.add(channel)

        # Update fields
        if channel_type == "whatsapp":
            channel.wa_phone_number_id = body.get("wa_phone_number_id", channel.wa_phone_number_id)
            channel.wa_business_account_id = body.get("wa_business_account_id", channel.wa_business_account_id)
            channel.wa_access_token = body.get("wa_access_token", channel.wa_access_token)
            channel.wa_verify_token = body.get("wa_verify_token", channel.wa_verify_token)
            channel.wa_phone_display = body.get("wa_phone_display", channel.wa_phone_display)
        elif channel_type == "telegram":
            channel.tg_bot_token = body.get("tg_bot_token", channel.tg_bot_token)
            channel.tg_bot_username = body.get("tg_bot_username", channel.tg_bot_username)

            # Auto-register webhook
            if body.get("tg_bot_token"):
                try:
                    from src.channels.telegram import TelegramAdapter
                    from src.config import get_settings
                    _settings = get_settings()
                    base_url = getattr(_settings, "telegram_webhook_url", "") or "https://api.maskki.com"
                    webhook_url = f"{base_url.rstrip('/')}/webhooks/telegram/{employee_id}"
                    await TelegramAdapter.register_webhook(
                        body["tg_bot_token"], webhook_url
                    )
                except Exception as exc:
                    logger.warning(f"Failed to register TG webhook: {exc}")

        elif channel_type == "widget":
            channel.widget_domain = body.get("widget_domain", channel.widget_domain)

        channel.enabled = body.get("enabled", True)

        await session.commit()

        return {
            "id": channel.id,
            "channel_type": channel.channel_type,
            "enabled": channel.enabled,
            "status": "connected",
        }


@router.delete("/employees/{employee_id}/channels/{channel_type}")
async def delete_channel(employee_id: str, channel_type: str, request: Request):
    """Disable a channel for an employee."""
    db = getattr(request.app.state, "db", None)
    if db is None:
        return JSONResponse(status_code=503, content={"error": "Database unavailable"})

    from sqlalchemy import update
    from src.db.models import EmployeeChannel

    async with db.session_ctx() as session:
        await session.execute(
            update(EmployeeChannel)
            .where(
                EmployeeChannel.employee_id == employee_id,
                EmployeeChannel.channel_type == channel_type,
            )
            .values(enabled=False)
        )
        await session.commit()

    return {"status": "disabled"}


# ═══════════════════════════════════════════════════════════════════
# QR Code
# ═══════════════════════════════════════════════════════════════════


@router.post("/employees/{employee_id}/qr-code")
async def generate_qr_code(employee_id: str, request: Request):
    """Generate or regenerate QR code for an employee."""
    db = getattr(request.app.state, "db", None)
    r2 = getattr(request.app.state, "r2_storage", None)

    from src.services.qr_generator import QRGenerator

    # Get employee name
    name = "AI Assistant"
    if db:
        from sqlalchemy import select
        from src.db.models import Employee

        async with db.session_ctx() as session:
            result = await session.execute(
                select(Employee).where(Employee.id == employee_id)
            )
            emp = result.scalar()
            if emp:
                name = emp.name

    generator = QRGenerator(r2_storage=r2, db=db)
    result = await generator.generate_employee_qr(employee_id, name)
    return result


@router.get("/employees/{employee_id}/qr-code/print")
async def download_print_qr(employee_id: str, request: Request):
    """Download high-res QR code for print (300 DPI PNG)."""
    from src.services.qr_generator import QRGenerator

    r2 = getattr(request.app.state, "r2_storage", None)
    generator = QRGenerator(r2_storage=r2)

    name = "AI Assistant"
    png_bytes = await generator.generate_print_version(employee_id, name)

    if not png_bytes:
        return JSONResponse(status_code=500, content={"error": "QR generation failed (qrcode package not installed)"})

    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={"Content-Disposition": f"attachment; filename={employee_id}_qr_print.png"},
    )


# ═══════════════════════════════════════════════════════════════════
# Documentation
# ═══════════════════════════════════════════════════════════════════


@router.get("/employees/{employee_id}/channel-docs")
async def get_channel_docs(employee_id: str, request: Request):
    """Get all auto-generated channel docs for an employee."""
    db = getattr(request.app.state, "db", None)

    from src.services.doc_generator import CustomerDocGenerator

    generator = CustomerDocGenerator(db=db)

    # Get customer_id from employee
    customer_id = ""
    if db:
        from sqlalchemy import select
        from src.db.models import Employee

        async with db.session_ctx() as session:
            result = await session.execute(
                select(Employee).where(Employee.id == employee_id)
            )
            emp = result.scalar()
            if emp:
                customer_id = emp.customer_id

    docs = await generator.get_docs(customer_id, employee_id)
    return {"docs": docs, "count": len(docs)}


@router.get("/employees/{employee_id}/channel-docs/{doc_type}")
async def get_channel_doc(employee_id: str, doc_type: str, request: Request):
    """Get a specific channel doc by type."""
    db = getattr(request.app.state, "db", None)

    from src.services.doc_generator import CustomerDocGenerator

    generator = CustomerDocGenerator(db=db)

    customer_id = ""
    if db:
        from sqlalchemy import select
        from src.db.models import Employee

        async with db.session_ctx() as session:
            result = await session.execute(
                select(Employee).where(Employee.id == employee_id)
            )
            emp = result.scalar()
            if emp:
                customer_id = emp.customer_id

    doc = await generator.get_doc(customer_id, employee_id, doc_type)
    if doc is None:
        return JSONResponse(status_code=404, content={"error": "Document not found"})
    return doc


@router.post("/employees/{employee_id}/channel-docs/generate")
async def generate_channel_docs(employee_id: str, request: Request):
    """Regenerate all channel docs for an employee."""
    db = getattr(request.app.state, "db", None)

    from src.services.doc_generator import CustomerDocGenerator

    generator = CustomerDocGenerator(db=db)

    # Get employee + customer info
    name = "AI Assistant"
    company = ""
    customer_id = ""
    role_title = ""

    if db:
        from sqlalchemy import select
        from src.db.models import Customer, Employee

        async with db.session_ctx() as session:
            result = await session.execute(
                select(Employee).where(Employee.id == employee_id)
            )
            emp = result.scalar()
            if emp:
                name = emp.name
                customer_id = emp.customer_id
                role_title = emp.role_title

                cust_result = await session.execute(
                    select(Customer).where(Customer.id == emp.customer_id)
                )
                customer = cust_result.scalar()
                if customer:
                    company = customer.company

    docs = await generator.generate_channel_docs(
        customer_id, employee_id, name, company, role_title
    )
    return {"docs": {k: {"title": v["title"]} for k, v in docs.items()}, "count": len(docs)}
