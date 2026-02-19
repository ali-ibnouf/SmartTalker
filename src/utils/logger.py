"""Structured JSON logging with correlation ID support.

Provides setup_logger(name, level) for consistent, queryable
log output across all SmartTalker modules.
"""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from typing import Optional

from pythonjsonlogger import jsonlogger

# Context variable for request correlation IDs (thread/async-safe)
correlation_id_var: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)


def get_correlation_id() -> Optional[str]:
    """Get the current request correlation ID.

    Returns:
        The correlation ID string, or None if not set.
    """
    return correlation_id_var.get()


def set_correlation_id(cid: str) -> None:
    """Set the correlation ID for the current context.

    Args:
        cid: The correlation ID to set (typically a UUID).
    """
    correlation_id_var.set(cid)


class SmartTalkerJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter that injects correlation ID and latency.

    Output format per line:
        {
            "timestamp": "2025-01-15T12:00:00.000Z",
            "level": "INFO",
            "module": "pipeline.asr",
            "message": "Transcription complete",
            "correlation_id": "abc-123",
            "latency_ms": 142
        }
    """

    def add_fields(
        self,
        log_record: dict,
        record: logging.LogRecord,
        message_dict: dict,
    ) -> None:
        """Add custom fields to every log entry.

        Args:
            log_record: The output dict that will be serialized.
            record: The original LogRecord from Python logging.
            message_dict: Extra key-value pairs passed via `extra={}`.
        """
        super().add_fields(log_record, record, message_dict)

        # Standard fields
        log_record["timestamp"] = self.formatTime(record, self.datefmt)
        log_record["level"] = record.levelname
        log_record["module"] = record.name

        # Correlation ID from context
        cid = get_correlation_id()
        if cid:
            log_record["correlation_id"] = cid

        # Latency (injected via extra={"latency_ms": value})
        if hasattr(record, "latency_ms"):
            log_record["latency_ms"] = record.latency_ms

        # Remove default fields that duplicate our custom ones
        for field in ("levelname", "name", "asctime"):
            log_record.pop(field, None)


def setup_logger(
    name: str,
    level: str = "INFO",
) -> logging.Logger:
    """Create a structured JSON logger.

    Args:
        name: Logger name (typically module path, e.g. "pipeline.asr").
        level: Logging level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).

    Returns:
        Configured Logger instance with JSON output to stdout.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    # JSON handler → stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)

    formatter = SmartTalkerJsonFormatter(
        fmt="%(timestamp)s %(level)s %(module)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S.%f",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def log_with_latency(
    logger: logging.Logger,
    message: str,
    latency_ms: float,
    level: str = "info",
    **kwargs: object,
) -> None:
    """Log a message with latency information attached.

    Args:
        logger: The logger instance to use.
        message: Log message text.
        latency_ms: Latency in milliseconds to include in the log entry.
        level: Log level string (default: "info").
        **kwargs: Additional key-value pairs for the log extra dict.
    """
    log_fn = getattr(logger, level.lower(), logger.info)
    log_fn(message, extra={"latency_ms": round(latency_ms, 2), **kwargs})
