"""Asynchronous utilities for SmartTalker."""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from src.utils.logger import setup_logger

logger = setup_logger("utils.async_utils")


def background_task_error_handler(task: asyncio.Task) -> None:
    """Log exceptions from background tasks to prevent silent failures.
    
    Usage:
        task = asyncio.create_task(coro())
        task.add_done_callback(background_task_error_handler)
    """
    if task.cancelled():
        return
    try:
        exc = task.exception()
        if exc is not None:
            logger.error(
                f"Background task failed: {exc}",
                extra={
                    "task_name": task.get_name(),
                    "exception_type": type(exc).__name__,
                },
                exc_info=exc,
            )
    except asyncio.InvalidStateError:
        # Task not finished yet
        pass
    except Exception as e:
        logger.error(f"Error in background_task_error_handler: {e}")
