"""Tests for the structured logging module."""

from __future__ import annotations

import io
import logging

from claude_bridge.log import configure_logging, get_logger, request_id_var


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
