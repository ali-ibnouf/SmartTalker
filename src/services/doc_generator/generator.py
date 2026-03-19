"""Customer documentation generator.

Auto-generates personalized setup guides for each channel
when a customer enables a new channel for their employee.
"""

from __future__ import annotations

from typing import Any

from src.utils.logger import setup_logger

logger = setup_logger("services.doc_generator")


class CustomerDocGenerator:
    """Generate personalized setup documentation for each customer."""

    def __init__(self, db: Any = None) -> None:
        self._db = db

    async def generate_channel_docs(
        self,
        customer_id: str,
        employee_id: str,
        employee_name: str,
        company_name: str,
        role_title: str = "",
    ) -> dict[str, dict[str, str]]:
        """Generate all channel setup docs.

        Returns:
            Dict keyed by doc_type with title, content, format.
        """
        docs = {
            "widget_setup": {
                "title": f"Widget Setup Guide — {employee_name}",
                "content": self._widget_guide(employee_id, employee_name, company_name),
                "format": "markdown",
            },
            "whatsapp_setup": {
                "title": f"WhatsApp Setup Guide — {employee_name}",
                "content": self._whatsapp_guide(employee_id, employee_name, company_name),
                "format": "markdown",
            },
            "telegram_setup": {
                "title": f"Telegram Setup Guide — {employee_name}",
                "content": self._telegram_guide(
                    employee_id, employee_name, company_name, role_title
                ),
                "format": "markdown",
            },
            "qr_guide": {
                "title": f"QR Code Guide — {employee_name}",
                "content": self._qr_guide(employee_id, employee_name),
                "format": "markdown",
            },
        }

        # Save to DB
        if self._db:
            await self._save_docs(customer_id, employee_id, docs)

        return docs

    async def get_docs(
        self, customer_id: str, employee_id: str
    ) -> list[dict[str, Any]]:
        """Get all saved docs for a customer/employee."""
        if self._db is None:
            return []

        from sqlalchemy import select
        from src.db.models import CustomerDoc

        async with self._db.session_ctx() as session:
            stmt = select(CustomerDoc).where(
                CustomerDoc.customer_id == customer_id,
                CustomerDoc.employee_id == employee_id,
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [
                {
                    "id": r.id,
                    "doc_type": r.doc_type,
                    "title": r.title,
                    "content": r.content,
                    "version": r.version,
                    "generated_at": r.generated_at.isoformat() if r.generated_at else None,
                }
                for r in rows
            ]

    async def get_doc(
        self, customer_id: str, employee_id: str, doc_type: str
    ) -> dict[str, Any] | None:
        """Get a specific doc by type."""
        if self._db is None:
            return None

        from sqlalchemy import select
        from src.db.models import CustomerDoc

        async with self._db.session_ctx() as session:
            stmt = select(CustomerDoc).where(
                CustomerDoc.customer_id == customer_id,
                CustomerDoc.employee_id == employee_id,
                CustomerDoc.doc_type == doc_type,
            )
            result = await session.execute(stmt)
            r = result.scalar()
            if r is None:
                return None
            return {
                "id": r.id,
                "doc_type": r.doc_type,
                "title": r.title,
                "content": r.content,
                "version": r.version,
                "generated_at": r.generated_at.isoformat() if r.generated_at else None,
            }

    async def _save_docs(
        self,
        customer_id: str,
        employee_id: str,
        docs: dict[str, dict[str, str]],
    ) -> None:
        """Upsert docs into customer_docs table."""
        from sqlalchemy import select
        from src.db.models import CustomerDoc

        try:
            async with self._db.session_ctx() as session:
                for doc_type, doc in docs.items():
                    stmt = select(CustomerDoc).where(
                        CustomerDoc.customer_id == customer_id,
                        CustomerDoc.employee_id == employee_id,
                        CustomerDoc.doc_type == doc_type,
                    )
                    result = await session.execute(stmt)
                    existing = result.scalar()

                    if existing:
                        existing.title = doc["title"]
                        existing.content = doc["content"]
                        existing.version = (existing.version or 0) + 1
                    else:
                        session.add(CustomerDoc(
                            customer_id=customer_id,
                            employee_id=employee_id,
                            doc_type=doc_type,
                            title=doc["title"],
                            content=doc["content"],
                        ))

                await session.commit()
        except Exception as exc:
            logger.error(f"Failed to save docs: {exc}")

    # ── Guide Templates ───────────────────────────────────────────────

    def _widget_guide(
        self, employee_id: str, employee_name: str, company_name: str
    ) -> str:
        return f"""# Widget Setup for {employee_name}

## Embed Code

Copy and paste the following code before `</body>` in your website HTML:

```html
<script
  src="https://maskki.com/widget.js"
  data-employee-id="{employee_id}"
  data-position="bottom-right"
  data-greeting="{employee_name} is here to help!"
  data-theme="auto"
  async>
</script>
```

## Configuration Options

| Attribute | Default | Description |
|-----------|---------|-------------|
| `data-position` | `bottom-right` | Widget position: `bottom-right`, `bottom-left` |
| `data-greeting` | Employee name | Initial greeting message |
| `data-language` | `auto` | Language code (ar, en, fr, etc.) |
| `data-theme` | `auto` | `light`, `dark`, or `auto` |

## Testing

1. Add the code to a test page
2. Open the page in a browser
3. Click the widget icon in the bottom-right corner
4. Send a test message — {employee_name} should reply

## Allowed Domains

For security, add your website domain in:
Maskki Dashboard -> {employee_name} -> Channels -> Widget -> Allowed Domains
"""

    def _whatsapp_guide(
        self, employee_id: str, employee_name: str, company_name: str
    ) -> str:
        return f"""# WhatsApp Setup for {employee_name}

## Prerequisites
- Facebook Business Manager account
- Verified business
- Phone number not already registered on WhatsApp

## Step 1: Create WhatsApp Business Account
1. Go to https://business.facebook.com
2. Navigate to "WhatsApp Accounts" -> "Add"
3. Follow the verification process

## Step 2: Add Phone Number
1. In Meta Business Suite -> WhatsApp -> Getting Started
2. Add your business phone number
3. Verify via SMS or voice call

## Step 3: Create System User
1. Go to Business Settings -> Users -> System Users
2. Click "Add" -> name it "Maskki Integration"
3. Grant it "whatsapp_business_messaging" permission
4. Generate a permanent access token

## Step 4: Configure in Maskki
1. Go to https://app.maskki.com -> {employee_name} -> Channels -> WhatsApp
2. Enter:
   - Phone Number ID: (from Meta dashboard)
   - Business Account ID: (from Meta dashboard)
   - Access Token: (from Step 3)
3. Click "Connect"

## Step 5: Set Webhook
In Meta Developer Console -> WhatsApp -> Configuration:
- Callback URL: `https://api.maskki.com/webhooks/whatsapp/{employee_id}`
- Verify Token: (shown in Maskki dashboard after connecting)
- Subscribe to: messages, message_status

## Step 6: Test
Send a message to your WhatsApp Business number.
{employee_name} should reply within seconds.

## Troubleshooting
- If no reply: check webhook URL is correct
- If "message failed": check access token hasn't expired
- Contact support: contact@lsmarttech.com
"""

    def _telegram_guide(
        self,
        employee_id: str,
        employee_name: str,
        company_name: str,
        role_title: str,
    ) -> str:
        safe_company = company_name.lower().replace(" ", "")
        safe_name = employee_name.lower().replace(" ", "")

        return f"""# Telegram Setup for {employee_name}

## Step 1: Create Bot
1. Open Telegram -> search for @BotFather
2. Send: /newbot
3. Choose name: "{employee_name} - {company_name}"
4. Choose username: e.g. {safe_company}_{safe_name}bot
5. Copy the bot token

## Step 2: Configure in Maskki
1. Go to https://app.maskki.com -> {employee_name} -> Channels -> Telegram
2. Paste the bot token
3. Click "Connect"
4. Maskki will auto-register the webhook

## Step 3: Test
Open Telegram -> search for your bot -> send /start
{employee_name} should reply.

## Optional: Set Bot Profile
Send to @BotFather:
- /setdescription -> "{role_title} at {company_name}"
- /setabouttext -> "Powered by Maskki AI"
- /setuserpic -> upload {employee_name}'s photo
"""

    def _qr_guide(self, employee_id: str, employee_name: str) -> str:
        return f"""# QR Code Guide for {employee_name}

## Your QR Code
Download from: https://app.maskki.com -> {employee_name} -> Channels -> QR Code

## Where to Use
- **Business cards**: add QR to employee/company cards
- **Storefront**: print and display at reception/entrance
- **Product packaging**: customers scan to get support
- **Brochures/flyers**: instant access to AI agent
- **Email signatures**: link to QR image
- **Social media**: share QR as post/story

## What Happens When Scanned
1. Visitor's phone opens: maskki.com/connect/{employee_id}
2. They see {employee_name}'s photo + role
3. They choose: Website Chat, WhatsApp, or Telegram
4. Conversation starts immediately

## Print Specifications
- Minimum QR size: 2cm x 2cm
- Download high-res PDF for print (300 DPI)
- Keep white border around QR (don't crop)
"""
