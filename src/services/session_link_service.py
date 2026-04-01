"""Shareable session link generator.

Creates pre-authenticated URLs that open a video chat session.
Link format: {base_url}/s/{token}
Token stored in Redis with full session context and TTL.

Use cases:
- WhatsApp bot sends link to visitor for video handoff
- Link opens video chat with specific avatar
- Session pre-loaded with visitor context + collected docs
"""

from __future__ import annotations

import json
import secrets
from datetime import datetime, timedelta
from typing import Any, Optional

from src.utils.logger import setup_logger

logger = setup_logger("services.session_link")

SESSION_LINK_PREFIX = "session_link:"


class SessionLinkService:
    """Generate and manage shareable session links backed by Redis."""

    def __init__(self, redis_client: Any, base_url: str) -> None:
        self._redis = redis_client
        self._base_url = base_url.rstrip("/")

    async def create_link(
        self,
        customer_id: str,
        avatar_id: str,
        visitor_phone: Optional[str] = None,
        visitor_name: Optional[str] = None,
        language: str = "ar",
        context: Optional[dict] = None,
        service_type: Optional[str] = None,
        collected_docs: Optional[dict] = None,
        expires_minutes: int = 30,
        channel_source: str = "whatsapp",
    ) -> dict[str, Any]:
        """Create a shareable session link.

        Returns dict with token, url, expires_at, expires_minutes.
        """
        token = secrets.token_urlsafe(16)
        now = datetime.utcnow()

        session_data = {
            "token": token,
            "customer_id": customer_id,
            "avatar_id": avatar_id,
            "visitor_phone": visitor_phone,
            "visitor_name": visitor_name,
            "language": language,
            "service_type": service_type,
            "collected_docs": collected_docs or {},
            "context": context or {},
            "channel_source": channel_source,
            "created_at": now.isoformat(),
            "expires_at": (now + timedelta(minutes=expires_minutes)).isoformat(),
            "status": "pending",
            "allow_camera": True,
            "allow_microphone": True,
        }

        redis_key = f"{SESSION_LINK_PREFIX}{token}"
        await self._redis.setex(
            redis_key,
            expires_minutes * 60,
            json.dumps(session_data, ensure_ascii=False),
        )

        url = f"{self._base_url}/s/{token}"

        logger.info(
            "Session link created",
            extra={
                "token_prefix": token[:8],
                "customer_id": customer_id,
                "avatar_id": avatar_id,
                "expires_minutes": expires_minutes,
            },
        )

        return {
            "token": token,
            "url": url,
            "expires_at": session_data["expires_at"],
            "expires_minutes": expires_minutes,
        }

    async def get_session(self, token: str) -> Optional[dict[str, Any]]:
        """Retrieve session data by token. Returns None if expired/missing."""
        redis_key = f"{SESSION_LINK_PREFIX}{token}"
        data = await self._redis.get(redis_key)

        if not data:
            return None

        session = json.loads(data)

        # Double-check expiry (belt-and-suspenders with Redis TTL)
        expires_at = datetime.fromisoformat(session["expires_at"])
        if datetime.utcnow() > expires_at:
            await self._redis.delete(redis_key)
            return None

        return session

    async def activate_session(self, token: str) -> bool:
        """Mark session as active when visitor opens the link."""
        session = await self.get_session(token)
        if not session:
            return False

        session["status"] = "active"
        session["activated_at"] = datetime.utcnow().isoformat()

        redis_key = f"{SESSION_LINK_PREFIX}{token}"
        ttl = await self._redis.ttl(redis_key)
        if ttl > 0:
            await self._redis.setex(
                redis_key,
                ttl,
                json.dumps(session, ensure_ascii=False),
            )
        return True

    async def complete_session(self, token: str) -> bool:
        """Mark session as completed."""
        session = await self.get_session(token)
        if not session:
            return False

        session["status"] = "completed"
        session["completed_at"] = datetime.utcnow().isoformat()

        redis_key = f"{SESSION_LINK_PREFIX}{token}"
        # Keep for 1 hour after completion for audit
        await self._redis.setex(
            redis_key,
            3600,
            json.dumps(session, ensure_ascii=False),
        )
        return True

    async def invalidate_link(self, token: str) -> bool:
        """Manually invalidate a link before expiry."""
        redis_key = f"{SESSION_LINK_PREFIX}{token}"
        result = await self._redis.delete(redis_key)
        return result > 0
