"""Async SQLAlchemy engine and session factory.

Provides a Database class that manages the async engine lifecycle
and provides session factories for use in application code.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.utils.logger import setup_logger

logger = setup_logger("db.engine")


class Database:
    """Async database connection manager.

    Usage::

        db = Database(url="postgresql+asyncpg://user:pass@host/dbname")
        await db.connect()

        async with db.session() as session:
            result = await session.execute(...)

        await db.disconnect()
    """

    def __init__(self, url: str, echo: bool = False) -> None:
        self._url = url
        engine_kwargs: dict = {"echo": echo}
        if "sqlite" in url:
            # SQLite uses StaticPool — pool_size/max_overflow are not supported
            from sqlalchemy.pool import StaticPool
            engine_kwargs["poolclass"] = StaticPool
            engine_kwargs["connect_args"] = {"check_same_thread": False}
        else:
            engine_kwargs["pool_size"] = 10
            engine_kwargs["max_overflow"] = 20
            engine_kwargs["pool_pre_ping"] = True
        self._engine: AsyncEngine = create_async_engine(url, **engine_kwargs)
        self._session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    @property
    def engine(self) -> AsyncEngine:
        return self._engine

    async def connect(self) -> None:
        """Create tables and verify connectivity."""
        from src.db.models import Base

        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        logger.info("Database connected and tables synced")

    async def disconnect(self) -> None:
        """Dispose the engine connection pool."""
        await self._engine.dispose()
        logger.info("Database disconnected")

    def session(self) -> AsyncSession:
        """Create a new async session (use as context manager)."""
        return self._session_factory()

    @asynccontextmanager
    async def session_ctx(self) -> AsyncGenerator[AsyncSession, None]:
        """Async generator that yields a session and handles commit/rollback."""
        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
