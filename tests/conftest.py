"""Shared pytest fixtures for the claude-bridge test suite."""

from __future__ import annotations

import logging

import pytest

from claude_bridge import request_view


@pytest.fixture(autouse=True)
def reset_oversized_media_warnings():
    """Clear the process-wide oversized-media warn-once set before every test.

    ``request_view`` remembers which oversized media it has already warned about so a
    repeated ``count_tokens`` over one history does not re-warn. That memory outlives a
    test, so without this any two tests asserting on the warning would be order-dependent
    — and ``pytest-randomly`` reorders every run, which turns that into a flake rather
    than a stable failure. Autouse because the coupling is invisible at the call site.
    """
    request_view._warned_oversized_media.clear()
    yield
    request_view._warned_oversized_media.clear()


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
