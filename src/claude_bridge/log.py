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
import json
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


# ---------------------------------------------------------------------------
# Redacted compatibility trace — structural-only, env-gated, never raises
# ---------------------------------------------------------------------------

_TRACE_ENV = "CLAUDE_BRIDGE_TRACE_PATH"


def is_trace_enabled() -> bool:
    """Return True when compatibility tracing is enabled via ``CLAUDE_BRIDGE_TRACE_PATH``.

    Read at call time (not import time) so the environment can change between runs
    and tests can toggle it without re-importing the module.
    """
    return bool(os.environ.get(_TRACE_ENV))


def trace_event(event: str, fields: dict) -> None:
    """Append one structural trace line to the trace file when tracing is enabled.

    Writes a single JSON object per line: ``{"req": <id>, "event": <name>, **fields}``.
    The sink is content-agnostic: callers MUST pass only structural data (counts,
    types, names, ids, lengths) — redaction is the caller's contract, never the
    sink's. Any failure (disabled, unwritable path, serialization error) is
    swallowed so tracing can never break or fail a user request.
    """
    path = os.environ.get(_TRACE_ENV)
    if not path:
        return
    try:
        record = {"req": request_id_var.get(""), "event": event, **fields}
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
    except Exception:
        get_logger("trace").debug("trace write failed", exc_info=True)
