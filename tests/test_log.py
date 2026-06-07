"""Tests for the structured logging module."""

from __future__ import annotations

import io
import json
import logging

from claude_bridge.log import (
    configure_logging,
    get_logger,
    is_trace_enabled,
    request_id_var,
    trace_event,
)


class TestConfigureLogging:
    """configure_logging() sets up the root bridge logger with correct level and format."""

    def test_default_level_is_info(self):
        configure_logging()
        logger = get_logger("test_default")
        assert logger.getEffectiveLevel() == logging.INFO

    def test_explicit_level(self):
        configure_logging(level="DEBUG")
        logger = get_logger("test_explicit")
        assert logger.getEffectiveLevel() == logging.DEBUG

    def test_env_var_level(self, monkeypatch):
        monkeypatch.setenv("LOG_LEVEL", "WARNING")
        configure_logging()
        logger = get_logger("test_env")
        assert logger.getEffectiveLevel() == logging.WARNING


class TestGetLogger:
    """get_logger() returns a child logger under the bridge namespace."""

    def test_returns_logger_with_bridge_prefix(self):
        logger = get_logger("proxy")
        assert logger.name == "claude_bridge.proxy"

    def test_different_names_different_loggers(self):
        a = get_logger("proxy")
        b = get_logger("router")
        assert a is not b
        assert a.name != b.name


class TestLogFormat:
    """Log output includes [BRIDGE:<module>] prefix and request ID when set."""

    def test_format_without_request_id(self):
        stream = io.StringIO()
        configure_logging(level="INFO", stream=stream)
        logger = get_logger("proxy")
        token = request_id_var.set("")
        try:
            logger.info("test message")
        finally:
            request_id_var.reset(token)
        output = stream.getvalue()
        assert "[BRIDGE:proxy]" in output
        assert "test message" in output

    def test_format_with_request_id(self):
        stream = io.StringIO()
        configure_logging(level="INFO", stream=stream)
        logger = get_logger("proxy")
        token = request_id_var.set("abc123")
        try:
            logger.info("hello")
        finally:
            request_id_var.reset(token)
        output = stream.getvalue()
        assert "[BRIDGE:proxy]" in output
        assert "req=abc123" in output
        assert "hello" in output

    def test_debug_filtered_at_info_level(self):
        stream = io.StringIO()
        configure_logging(level="INFO", stream=stream)
        logger = get_logger("proxy")
        logger.debug("should not appear")
        output = stream.getvalue()
        assert output == ""

    def test_debug_visible_at_debug_level(self):
        stream = io.StringIO()
        configure_logging(level="DEBUG", stream=stream)
        logger = get_logger("proxy")
        logger.debug("should appear")
        output = stream.getvalue()
        assert "should appear" in output


class TestTraceSink:
    """trace_event is the env-gated, content-agnostic trace sink. It writes only
    when CLAUDE_BRIDGE_TRACE_PATH is set, appends JSON lines, and never raises."""

    def test_disabled_by_default_writes_nothing(self, tmp_path, monkeypatch):
        # No env var → tracing is off and the sink is a no-op (no file created).
        monkeypatch.delenv("CLAUDE_BRIDGE_TRACE_PATH", raising=False)
        target = tmp_path / "trace.jsonl"
        assert is_trace_enabled() is False
        trace_event("inbound_request", {"model": "gpt-5.5"})
        assert not target.exists()

    def test_enabled_appends_one_json_line_per_event(self, tmp_path, monkeypatch):
        target = tmp_path / "trace.jsonl"
        monkeypatch.setenv("CLAUDE_BRIDGE_TRACE_PATH", str(target))
        assert is_trace_enabled() is True
        trace_event("inbound_request", {"model": "gpt-5.5", "message_count": 3})
        trace_event("warning", {"text": "stripped cache_control"})
        lines = target.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first["event"] == "inbound_request"
        assert first["model"] == "gpt-5.5"
        assert first["message_count"] == 3
        assert json.loads(lines[1])["event"] == "warning"

    def test_record_carries_request_id(self, tmp_path, monkeypatch):
        target = tmp_path / "trace.jsonl"
        monkeypatch.setenv("CLAUDE_BRIDGE_TRACE_PATH", str(target))
        token = request_id_var.set("deadbeef")
        try:
            trace_event("inbound_request", {"model": "gpt-5.5"})
        finally:
            request_id_var.reset(token)
        assert json.loads(target.read_text(encoding="utf-8").splitlines()[0])["req"] == "deadbeef"

    def test_unwritable_path_never_raises(self, tmp_path, monkeypatch):
        # An unwritable trace path (a directory, not a file) must be swallowed:
        # tracing can never fail a user request.
        a_directory = tmp_path / "trace_dir"
        a_directory.mkdir()
        monkeypatch.setenv("CLAUDE_BRIDGE_TRACE_PATH", str(a_directory))
        # No exception escapes even though opening a directory for append fails.
        trace_event("inbound_request", {"model": "gpt-5.5"})
