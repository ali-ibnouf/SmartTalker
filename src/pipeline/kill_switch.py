"""Kill Switch for suspending/resuming customer accounts.

Provides immediate account suspension with Redis-cached lookups
for fast middleware checks. Falls back to database queries.
"""

from __future__ import annotations

from typing import Any, Optional

from src.utils.logger import setup_logger

logger = setup_logger("pipeline.kill_switch")

# Redis key prefix for suspension cache
_REDIS_PREFIX = "suspended:"
_REDIS_TTL = 3600  # 1 hour cache


class KillSwitch:
    """Customer account suspension manager.

    Checks suspension status via: local cache -> Redis -> database.
    Suspension immediately disconnects all active WebSocket sessions.

    Args:
        db: Database instance for persistent suspension state.
        redis: Optional Redis client for fast cached lookups.
        ws_manager: Optional WebSocket manager for disconnecting sessions.
    """

    def __init__(
        self,
        db: Any = None,
        redis: Any = None,
        ws_manager: Any = None,
    ) -> None:
        self._db = db
        self._redis = redis
        self._ws_manager = ws_manager
        self._local_cache: dict[str, bool] = {}

        logger.info("KillSwitch initialized")

    async def suspend(self, customer_id: str, reason: str = "") -> bool:
        """Suspend a customer account.

        1. Update database
        2. Set Redis cache
        3. Disconnect active WebSocket sessions
        """
        # Update database
        if self._db is not None:
            from sqlalchemy import update
            from src.db.models import Customer
            async with self._db.session() as session:
                await session.execute(
                    update(Customer)
                    .where(Customer.id == customer_id)
                    .values(suspended=True, suspended_reason=reason)
                )
                await session.commit()

        # Set Redis cache
        if self._redis is not None:
            try:
                await self._redis.set(
                    f"{_REDIS_PREFIX}{customer_id}", "1", ex=_REDIS_TTL
                )
            except Exception as exc:
                logger.warning(f"Redis cache set failed: {exc}")

        # Update local cache
        self._local_cache[customer_id] = True

        # Disconnect active WebSocket sessions (best-effort)
        if self._ws_manager is not None:
            try:
                await self._disconnect_customer_sessions(customer_id)
            except Exception as exc:
                logger.warning(f"Session disconnect failed: {exc}")

        logger.info(
            "Customer suspended",
            extra={"customer_id": customer_id, "reason": reason},
        )
        return True

    async def resume(self, customer_id: str) -> bool:
        """Resume a suspended customer account."""
        # Update database
        if self._db is not None:
            from sqlalchemy import update
            from src.db.models import Customer
            async with self._db.session() as session:
                await session.execute(
                    update(Customer)
                    .where(Customer.id == customer_id)
                    .values(suspended=False, suspended_reason="")
                )
                await session.commit()

        # Clear Redis cache
        if self._redis is not None:
            try:
                await self._redis.delete(f"{_REDIS_PREFIX}{customer_id}")
            except Exception as exc:
                logger.warning(f"Redis cache delete failed: {exc}")

        # Update local cache
        self._local_cache[customer_id] = False

        logger.info("Customer resumed", extra={"customer_id": customer_id})
        return True

    async def is_suspended(self, customer_id: str) -> bool:
        """Check if a customer is suspended.

        Lookup order: local cache -> Redis -> database.
        """
        # 1. Local cache
        if customer_id in self._local_cache:
            return self._local_cache[customer_id]

        # 2. Redis
        if self._redis is not None:
            try:
                cached = await self._redis.get(f"{_REDIS_PREFIX}{customer_id}")
                if cached is not None:
                    is_susp = cached == "1"
                    self._local_cache[customer_id] = is_susp
                    return is_susp
            except Exception:
                pass

        # 3. Database
        if self._db is not None:
            from sqlalchemy import select
            from src.db.models import Customer
            async with self._db.session() as session:
                result = await session.execute(
                    select(Customer.suspended).where(Customer.id == customer_id)
                )
                row = result.scalar_one_or_none()
                is_susp = bool(row) if row is not None else False

                # Populate caches
                self._local_cache[customer_id] = is_susp
                if self._redis is not None:
                    try:
                        await self._redis.set(
                            f"{_REDIS_PREFIX}{customer_id}",
                            "1" if is_susp else "0",
                            ex=_REDIS_TTL,
                        )
                    except Exception:
                        pass

                return is_susp

        return False

    async def _disconnect_customer_sessions(self, customer_id: str) -> None:
        """Disconnect all active WebSocket sessions for a customer."""
        if self._ws_manager is None:
            return

        # The WS manager tracks sessions — iterate and close matching ones
        for session_id, conn in list(getattr(self._ws_manager, "_connections", {}).items()):
            if getattr(conn, "customer_id", None) == customer_id:
                try:
                    ws = getattr(conn, "websocket", None)
                    if ws:
                        await ws.close(code=4003, reason="Account suspended")
                except Exception:
                    pass

