"""Structured JSON logging setup for the Sentinel trading system."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Any, Optional


class JSONFormatter(logging.Formatter):
    """Format log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "component": record.name,
            "event": record.getMessage(),
        }

        # Merge any extra data passed via the `extra` kwarg
        if hasattr(record, "__dict__"):
            for key, value in record.__dict__.items():
                if key not in (
                    "name", "msg", "args", "created", "relativeCreated",
                    "exc_info", "exc_text", "stack_info", "lineno", "funcName",
                    "pathname", "filename", "module", "levelno", "levelname",
                    "msecs", "message", "taskName", "processName", "process",
                    "thread", "threadName",
                ):
                    log_entry.setdefault("data", {})[key] = _safe_serialize(value)

        if record.exc_info and record.exc_info[1] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)


def _safe_serialize(value: Any) -> Any:
    """Convert a value to something JSON-serializable."""
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, (list, tuple)):
        return [_safe_serialize(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _safe_serialize(v) for k, v in value.items()}
    return str(value)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    max_size_mb: int = 100,
    rotation: str = "daily",
) -> logging.Logger:
    """Configure structured JSON logging for the entire application.

    Args:
        level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Path to the log file. If None, logs only to console.
        max_size_mb: Maximum log file size in MB before rotation (size-based).
        rotation: Rotation strategy — "daily", "weekly", or "size".

    Returns:
        The root logger, configured with JSON output.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers to avoid duplicates on re-init
    root_logger.handlers.clear()

    formatter = JSONFormatter()

    # Console handler — always present
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler — optional
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        if rotation == "size":
            file_handler: logging.Handler = RotatingFileHandler(
                str(log_path),
                maxBytes=max_size_mb * 1024 * 1024,
                backupCount=5,
            )
        else:
            when = "midnight" if rotation == "daily" else "W0"
            file_handler = TimedRotatingFileHandler(
                str(log_path),
                when=when,
                backupCount=30,
                utc=True,
            )

        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger


def get_logger(component: str) -> logging.Logger:
    """Return a named logger for a specific component.

    Usage::

        logger = get_logger("risk.manager")
        logger.info("Signal approved", extra={"symbol": "AAPL"})
    """
    return logging.getLogger(component)
