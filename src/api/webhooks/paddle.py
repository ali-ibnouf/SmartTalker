"""Paddle webhook handlers.

Processes subscription events from Paddle billing:
- subscription.canceled → freeze employees + purge R2 media
- subscription.updated (status=active) → reactivate employees
"""

from __future__ import annotations

import hmac
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from src.config import get_settings
from src.services.subscription import SubscriptionLifecycle
from src.utils.logger import setup_logger

logger = setup_logger("api.webhooks.paddle")

router = APIRouter(prefix="/api/v1/webhooks/paddle", tags=["webhooks", "paddle"])


def _verify_paddle_signature(request: Request, body: bytes, config: Any) -> bool:
    """Verify Paddle webhook signature.
    
    Simplified validation using Paddle v2 webhook signatures.
    Format: ts=...,h1=...
    """
    signature_header = request.headers.get("Paddle-Signature")
    if not signature_header or not getattr(config, "paddle_webhook_secret", None):
        return False

    try:
        parts = dict(part.split("=") for part in signature_header.split(","))
        ts = parts.get("ts")
        h1 = parts.get("h1")

        if not ts or not h1:
            return False

        # Tolerance: reject if older than 5 minutes
        if time.time() - int(ts) > 300:
            return False

        payload = f"{ts}:{body.decode('utf-8')}"
        expected_sig = hmac.new(
            getattr(config, "paddle_webhook_secret", "").encode("utf-8"),
            payload.encode("utf-8"),
            digestmod="sha256",
        ).hexdigest()

        return hmac.compare_digest(h1, expected_sig)
    except Exception:
        return False


@router.post("")
async def paddle_webhook(request: Request):
    """Handle incoming Paddle webhooks."""
    config = get_settings()
    
    # Optional: Read body for signature verification
    body_bytes = await request.body()
    
    # Check signature if secret is configured
    paddle_secret = getattr(config, "paddle_webhook_secret", None)
    if paddle_secret:
        if not _verify_paddle_signature(request, body_bytes, config):
            logger.warning("Invalid Paddle signature")
            raise HTTPException(status_code=401, detail="Invalid signature")

    try:
        from fastapi import Body
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    event_type = payload.get("event_type")
    
    # Ensure there's a valid data payload
    data = payload.get("data", {})
    if not data:
        return {"status": "ignored"}

    if event_type == "subscription.canceled":
        # In Paddle v2, we look for the internal custom data (the customer_id)
        custom_data = data.get("custom_data", {})
        customer_id = custom_data.get("customer_id")

        if not customer_id:
            logger.warning("subscription.canceled received without customer_id in custom_data")
            return {"status": "ignored", "reason": "missing customer_id"}

        db = getattr(request.app.state, "db", None)
        storage = getattr(request.app.state, "storage", None)
        
        # Determine R2 instance from storage if available
        r2_manager = getattr(storage, "_r2", None) if storage else None
        
        lifecycle = SubscriptionLifecycle(db=db, r2_storage=r2_manager, config=config)

        # Execute cancellation, which freezes employees and optionally purges media
        # We pass purge_media=True as per the requirements for permanent deletion
        try:
            result = await lifecycle.cancel(
                customer_id=customer_id, 
                reason="paddle_subscription_canceled",
                purge_media=True
            )
            return {"status": "success", "action": "cancelled", "result": result}
        except Exception as exc:
            logger.error(f"Failed to process cancellation for {customer_id}: {exc}")
            raise HTTPException(status_code=500, detail="Cancellation processing failed")

    if event_type == "subscription.updated":
        custom_data = data.get("custom_data", {})
        customer_id = custom_data.get("customer_id")
        status = data.get("status", "")

        if not customer_id:
            return {"status": "ignored", "reason": "missing customer_id"}

        # Reactivate if subscription was resumed
        if status == "active":
            db = getattr(request.app.state, "db", None)
            lifecycle = SubscriptionLifecycle(db=db, config=config)

            try:
                items = data.get("items", [])
                plan_id = ""
                if isinstance(items, list) and items:
                    plan_item = items[0]
                    if isinstance(plan_item, dict):
                        plan_id = plan_item.get("price", {}).get("id", "")
                result = await lifecycle.reactivate(
                    customer_id=customer_id,
                    plan_id=plan_id,
                )
                return {"status": "success", "action": "reactivated", "result": result}
            except Exception as exc:
                logger.error(f"Failed to reactivate {customer_id}: {exc}")
                raise HTTPException(status_code=500, detail="Reactivation failed")

        return {"status": "ignored", "reason": f"unhandled status: {status}"}

    return {"status": "ignored", "event_type": event_type}
