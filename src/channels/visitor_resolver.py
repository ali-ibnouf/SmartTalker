"""Unified visitor identity resolution across channels.

Same person on WhatsApp and Widget is recognized as the same visitor,
sharing conversation history and memory.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from src.utils.logger import setup_logger

logger = setup_logger("channels.visitor_resolver")


class VisitorResolver:
    """Resolve and merge visitor identities across channels."""

    def __init__(self, db: Any, redis: Any = None) -> None:
        self._db = db
        self._redis = redis

    async def resolve_visitor(
        self,
        channel_type: str,
        channel_user_id: str,
        employee_id: str,
    ) -> str:
        """Get or create a unified visitor_id.

        Args:
            channel_type: "whatsapp", "telegram", "widget"
            channel_user_id: Phone number / tg user_id / ws session
            employee_id: The employee being contacted

        Returns:
            Unified visitor ID string.
        """
        if self._db is None:
            return f"v_{uuid.uuid4().hex[:12]}"

        try:
            from sqlalchemy import select, update
            from src.db.models import VisitorChannelMap, VisitorProfile

            async with self._db.session_ctx() as session:
                # Check if this channel user is already mapped
                stmt = select(VisitorChannelMap.visitor_id).where(
                    VisitorChannelMap.channel_type == channel_type,
                    VisitorChannelMap.channel_user_id == channel_user_id,
                    VisitorChannelMap.employee_id == employee_id,
                )
                result = await session.execute(stmt)
                existing = result.scalar()

                if existing:
                    # Update last_seen
                    await session.execute(
                        update(VisitorChannelMap)
                        .where(
                            VisitorChannelMap.channel_type == channel_type,
                            VisitorChannelMap.channel_user_id == channel_user_id,
                            VisitorChannelMap.employee_id == employee_id,
                        )
                        .values(last_seen=datetime.utcnow())
                    )
                    await session.commit()
                    await self._record_resolve_metric(True)
                    return existing

                # Create new visitor
                visitor_id = f"v_{uuid.uuid4().hex[:12]}"

                # Create channel mapping
                mapping = VisitorChannelMap(
                    visitor_id=visitor_id,
                    channel_type=channel_type,
                    channel_user_id=channel_user_id,
                    employee_id=employee_id,
                )
                session.add(mapping)

                # Create visitor profile
                profile = VisitorProfile(
                    visitor_id=visitor_id,
                    employee_id=employee_id,
                )
                session.add(profile)

                await session.commit()

                logger.info(
                    "New visitor created",
                    extra={
                        "visitor_id": visitor_id,
                        "channel": channel_type,
                        "employee_id": employee_id,
                    },
                )
                await self._record_resolve_metric(True)
                return visitor_id

        except Exception as exc:
            logger.error(f"Visitor resolution failed: {exc}")
            await self._record_resolve_metric(False)
            return f"v_{uuid.uuid4().hex[:12]}"

    async def link_channel(
        self,
        visitor_id: str,
        channel_type: str,
        channel_user_id: str,
        employee_id: str,
    ) -> None:
        """Link a new channel to an existing visitor.

        Used when a visitor provides their phone number on the widget,
        allowing us to link their WhatsApp identity.
        """
        if self._db is None:
            return

        try:
            from src.db.models import VisitorChannelMap

            async with self._db.session_ctx() as session:
                mapping = VisitorChannelMap(
                    visitor_id=visitor_id,
                    channel_type=channel_type,
                    channel_user_id=channel_user_id,
                    employee_id=employee_id,
                )
                session.add(mapping)
                await session.commit()

                logger.info(
                    "Channel linked to visitor",
                    extra={
                        "visitor_id": visitor_id,
                        "channel": channel_type,
                    },
                )
        except Exception as exc:
            logger.error(f"Channel linking failed: {exc}")

    async def merge_visitors(
        self, keep_visitor_id: str, merge_visitor_id: str
    ) -> None:
        """Merge two visitor identities.

        Moves all memories, conversations, and channel maps from
        merge_visitor_id to keep_visitor_id, then deletes merge_visitor_id.
        """
        if self._db is None:
            return

        try:
            from sqlalchemy import update, delete
            from src.db.models import (
                VisitorChannelMap, VisitorProfile, VisitorMemory,
            )

            async with self._db.session_ctx() as session:
                # Move channel maps
                await session.execute(
                    update(VisitorChannelMap)
                    .where(VisitorChannelMap.visitor_id == merge_visitor_id)
                    .values(visitor_id=keep_visitor_id)
                )

                # Move memories
                await session.execute(
                    update(VisitorMemory)
                    .where(VisitorMemory.visitor_id == merge_visitor_id)
                    .values(visitor_id=keep_visitor_id)
                )

                # Delete merged profile
                await session.execute(
                    delete(VisitorProfile)
                    .where(VisitorProfile.visitor_id == merge_visitor_id)
                )

                await session.commit()

                logger.info(
                    "Visitors merged",
                    extra={
                        "keep": keep_visitor_id,
                        "merged": merge_visitor_id,
                    },
                )
        except Exception as exc:
            logger.error(f"Visitor merge failed: {exc}")

    async def _record_resolve_metric(self, success: bool) -> None:
        """Record visitor resolution success/failure for agent monitoring."""
        if self._redis is None:
            return
        try:
            hour = datetime.utcnow().strftime("%Y%m%d%H")
            key = f"visitor_resolve_{'ok' if success else 'fail'}:{hour}"
            await self._redis.incr(key)
            await self._redis.expire(key, 7200)
        except Exception:
            pass
