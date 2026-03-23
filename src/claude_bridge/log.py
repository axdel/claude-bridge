"""Structured logging for Claude Bridge — stdlib only.

Wraps the stdlib ``logging`` module with:
- A ``[BRIDGE:<module>]`` prefix on every line
- Per-request ID correlation via ``contextvars``
- ``LOG_LEVEL`` env var support (default: INFO)

Usage::

    from claude_bridge.log import configure_logging, get_logger, request_id_var

    configure_logging()                     # call once at startup
    logger = get_logger("proxy")            # one per module
    request_id_var.set("abc123")            # set per request
    logger.info("POST /v1/messages (1234B)")
    # => INFO  [BRIDGE:proxy] req=abc123 POST /v1/messages (1234B)
"""

from __future__ import annotations

import contextvars
import logging
import os
import sys
from typing import TextIO

_NAMESPACE = "claude_bridge"

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="")


class _BridgeFormatter(logging.Formatter):
    """Format log records as ``LEVEL  [BRIDGE:<module>] req=<id> <message>``."""

    def format(self, record: logging.LogRecord) -> str:
        # Extract the module suffix from the logger name (e.g., "claude_bridge.proxy" → "proxy")
        module = record.name.removeprefix(f"{_NAMESPACE}.")
        req_id = request_id_var.get("")
        req_part = f" req={req_id}" if req_id else ""
        return f"{record.levelname:<5} [BRIDGE:{module}]{req_part} {record.getMessage()}"


def configure_logging(
    *,
    level: str | None = None,
    stream: TextIO | None = None,
) -> None:
    """Configure the bridge logger hierarchy.

    Args:
        level: Log level name (DEBUG/INFO/WARNING/ERROR). Falls back to
            ``LOG_LEVEL`` env var, then ``INFO``.
        stream: Output stream. Defaults to ``sys.stderr``.
    """
    resolved_level = level or os.environ.get("LOG_LEVEL", "INFO")
    resolved_stream = stream or sys.stderr

    root_logger = logging.getLogger(_NAMESPACE)
    root_logger.setLevel(resolved_level.upper())

    # Remove existing handlers to avoid duplicate output on re-configure
    root_logger.handlers.clear()

    handler = logging.StreamHandler(resolved_stream)
    handler.setFormatter(_BridgeFormatter())
    root_logger.addHandler(handler)

    # Prevent propagation to the root logger (avoids double output)
    root_logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the ``claude_bridge`` namespace.

    Example: ``get_logger("proxy")`` returns a logger named ``claude_bridge.proxy``.
    The logger inherits its level from the parent configured by ``configure_logging()``.
    """
    return logging.getLogger(f"{_NAMESPACE}.{name}")
