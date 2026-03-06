"""Standalone runner for the AI Optimization Agent.

Can run as its own Docker container connecting to the same
PostgreSQL and Redis as Central Server.

Usage: python -m src.services.ai_agent.runner
"""

from __future__ import annotations

import asyncio
import signal
import sys
from typing import Any

from src.utils.logger import setup_logger

logger = setup_logger("ai_agent.runner")


async def main() -> None:
    """Entry point: set up DB/Redis, launch agent, serve health endpoint."""
    from src.config import get_settings
    from src.db.database import Database
    from src.services.ai_agent.agent import AIAgent
    from src.services.ai_agent.config import AgentSettings
    from src.services.ai_agent.rules import AgentContext

    config = get_settings()
    agent_config = AgentSettings()

    # Database
    db = Database(url=config.database_url)
    await db.connect()
    logger.info("Database connected")

    # Redis
    redis: Any = None
    try:
        import redis.asyncio as aioredis

        redis = aioredis.from_url(
            config.redis_url,
            decode_responses=False,
        )
        await redis.ping()
        logger.info("Redis connected")
    except Exception as exc:
        logger.warning(f"Redis unavailable, running without: {exc}")
        redis = None

    # Build context (no pipeline/operator_manager in standalone mode)
    ctx = AgentContext(
        db=db,
        redis=redis,
        pipeline=None,
        config=config,
        agent_config=agent_config,
        operator_manager=None,
    )

    # Start agent
    agent = AIAgent(ctx)
    await agent.start()
    logger.info("AI Optimization Agent running in standalone mode")

    # Health HTTP endpoint on port 8081
    health_server = await _start_health_server(agent, port=8081)

    # Wait for shutdown signal
    shutdown_event = asyncio.Event()

    def _signal_handler() -> None:
        logger.info("Shutdown signal received")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass

    # On Windows, also handle KeyboardInterrupt
    try:
        await shutdown_event.wait()
    except KeyboardInterrupt:
        pass

    # Cleanup
    logger.info("Shutting down...")
    await agent.stop()
    if health_server:
        health_server.close()
        await health_server.wait_closed()
    if redis:
        await redis.close()
    await db.disconnect()
    logger.info("AI Agent stopped cleanly")


async def _start_health_server(agent: Any, port: int = 8081) -> Any:
    """Start a minimal TCP health endpoint."""
    import json

    async def handle_request(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            data = await asyncio.wait_for(reader.read(1024), timeout=5.0)
            request_line = data.decode("utf-8", errors="replace").split("\r\n")[0]

            if "GET /health" in request_line:
                stats = await agent.get_stats()
                body = json.dumps({"status": "ok", **stats})
                response = (
                    f"HTTP/1.1 200 OK\r\n"
                    f"Content-Type: application/json\r\n"
                    f"Content-Length: {len(body)}\r\n"
                    f"\r\n{body}"
                )
            else:
                body = '{"error": "not found"}'
                response = (
                    f"HTTP/1.1 404 Not Found\r\n"
                    f"Content-Type: application/json\r\n"
                    f"Content-Length: {len(body)}\r\n"
                    f"\r\n{body}"
                )

            writer.write(response.encode())
            await writer.drain()
        except Exception:
            pass
        finally:
            writer.close()

    try:
        server = await asyncio.start_server(handle_request, "0.0.0.0", port)
        logger.info(f"Health endpoint listening on port {port}")
        return server
    except Exception as exc:
        logger.warning(f"Failed to start health server: {exc}")
        return None


if __name__ == "__main__":
    asyncio.run(main())
