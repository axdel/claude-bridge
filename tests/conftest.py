"""Shared pytest fixtures for the claude-bridge test suite."""

from __future__ import annotations

import logging

import pytest


@pytest.fixture
def capture_logger():
    """Return a factory that captures records from a named bridge logger.

    Bridge loggers set ``propagate=False`` (see ``log.configure_logging``), so pytest's
    built-in ``caplog`` never sees them. This attaches a record-collecting handler directly
    to the named logger, forces it to DEBUG so lower-level records are not filtered before
    the handler, and restores the logger's prior handlers/level at teardown.

    Usage::

        def test_x(capture_logger):
            records = capture_logger("claude_bridge.request_view")
            ...  # trigger logging
            assert any(r.levelno == logging.DEBUG for r in records)
    """
    attached: list[tuple[logging.Logger, logging.Handler, int]] = []

    def _capture(name: str, level: int = logging.DEBUG) -> list[logging.LogRecord]:
        records: list[logging.LogRecord] = []

        class _Collector(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        logger = logging.getLogger(name)
        handler = _Collector()
        handler.setLevel(level)
        prev_level = logger.level
        logger.addHandler(handler)
        logger.setLevel(level)
        attached.append((logger, handler, prev_level))
        return records

    yield _capture

    for logger, handler, prev_level in attached:
        logger.removeHandler(handler)
        logger.setLevel(prev_level)
