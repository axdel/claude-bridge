"""Tests for translation-warning log-level classification (``request_view``).

The bug these guard: routine, per-request translation notices (thinking passthrough,
effort clamp, and dropping the ``output_config.format`` hint Claude Code sends on every
request) were logged at WARNING and — because the launchers share bridge stderr with the
Claude Code TUI at LOG_LEVEL=WARNING — flooded the terminal. The emitter now routes routine
notices to DEBUG and keeps genuinely-lossy ones (content/tool drops, unrecognized values) at
WARNING. Only the proven-boilerplate ``output_config.format`` drop is demoted, by EXACT match:
every OTHER output_config subkey drop (e.g. ``task_budget``, a real cost control) stays WARNING,
loud by default — an allowlist demotes what it has proven routine, never a whole open class.

Oracle: the expected log level for each message class comes from the classification
spec (routine -> DEBUG=10, lossy/unknown -> WARNING=30), never from running the emitter.
The coupling tests drive the REAL translators so a future reword of a routine message
without updating the allowlist fails here rather than silently re-flooding.
"""

from __future__ import annotations

import logging

from claude_bridge.providers.openai import anthropic_to_openai
from claude_bridge.providers.xai import anthropic_to_xai
from claude_bridge.request_view import emit_translation_warnings

_LOGGER_NAME = "claude_bridge.request_view"


class TestTranslationWarningLevels:
    """emit_translation_warnings splits routine (DEBUG) from lossy (WARNING)."""

    def test_thinking_passthrough_notice_logged_at_debug(self, capture_logger):
        records = capture_logger(_LOGGER_NAME)
        emit_translation_warnings(
            ["Thinking config passed through (reasoning_mode=passthrough)"], {}
        )
        assert [r.levelno for r in records] == [logging.DEBUG]

    def test_thinking_drop_notice_logged_at_debug(self, capture_logger):
        records = capture_logger(_LOGGER_NAME)
        emit_translation_warnings(["Stripped 'thinking' config (reasoning_mode=drop)"], {})
        assert [r.levelno for r in records] == [logging.DEBUG]

    def test_effort_clamp_notice_logged_at_debug(self, capture_logger):
        """The grok max->high clamp fires on every request; at WARNING it would re-flood."""
        records = capture_logger(_LOGGER_NAME)
        emit_translation_warnings(
            ["output_config.effort 'max' clamped to 'high' (grok max effort)"], {}
        )
        assert [r.levelno for r in records] == [logging.DEBUG]

    def test_lossy_tool_choice_notice_logged_at_warning(self, capture_logger):
        records = capture_logger(_LOGGER_NAME)
        emit_translation_warnings(
            ["Unsupported tool_choice type 'wild', omitting tool_choice"], {}
        )
        assert [r.levelno for r in records] == [logging.WARNING]

    def test_dropped_output_config_format_logged_at_debug(self, capture_logger):
        """output_config.format is Claude Code per-request boilerplate the Responses path
        cannot honor; its drop is non-actionable and fires every request, so it is routine
        -> DEBUG (an exact-match allowlist entry). The trace still records it. This is the
        exact TUI-flood case the user reported."""
        records = capture_logger(_LOGGER_NAME)
        emit_translation_warnings(["Dropped unsupported output_config.format"], {})
        assert [r.levelno for r in records] == [logging.DEBUG]

    def test_dropped_task_budget_subkey_stays_warning(self, capture_logger):
        """Only output_config.format is demoted, not the subkey-drop class. task_budget is a
        real Anthropic token/cost control; if the bridge silently discards it the operator
        MUST see a WARNING. The allowlist enumerates proven-routine notices by EXACT match, so
        an unforeseen meaningful subkey stays loud — proving the demotion did not overreach."""
        records = capture_logger(_LOGGER_NAME)
        emit_translation_warnings(["Dropped unsupported output_config.task_budget"], {})
        assert [r.levelno for r in records] == [logging.WARNING]

    def test_unknown_notice_defaults_to_warning(self, capture_logger):
        """The classifier is an allowlist — an unrecognized notice is loud by default."""
        records = capture_logger(_LOGGER_NAME)
        emit_translation_warnings(["Some brand-new degradation nobody allowlisted"], {})
        assert [r.levelno for r in records] == [logging.WARNING]

    def test_lossy_effort_carrying_routine_marker_stays_warning(self, capture_logger):
        """A crafted effort value must not smuggle a lossy notice down to DEBUG.

        ``output_config.effort`` is client-controlled and is interpolated verbatim into
        the lossy 'Unrecognized' notice. A caller sending ``"turbo clamped to high"``
        embeds the routine marker ``clamped to`` inside a genuinely-lossy warning. The
        oracle: the notice starts with ``Unrecognized`` — a lossy prefix — so it is loud
        by default no matter what client text follows. Substring matching classified it
        DEBUG (the suppression bug); prefix-anchored matching keeps it WARNING.
        """
        records = capture_logger(_LOGGER_NAME)
        emit_translation_warnings(
            ["Unrecognized output_config.effort 'turbo clamped to high', using default 'max'"],
            {},
        )
        assert [r.levelno for r in records] == [logging.WARNING]

    def test_dropped_format_prefixed_subkey_stays_warning(self, capture_logger):
        """The format demotion is EXACT, not a ``...output_config.format`` prefix.

        A subkey whose name merely begins with ``format`` (e.g. a future ``format_version``)
        is a different, unproven control and stays WARNING — loud by default. This guards
        against anyone re-widening the exact-match entry back into a prefix that would demote
        unforeseen subkeys sight-unseen. Oracle: only the exact ``output_config.format``
        notice is routine -> DEBUG; a prefix-only match is not.
        """
        records = capture_logger(_LOGGER_NAME)
        emit_translation_warnings(["Dropped unsupported output_config.format_version"], {})
        assert [r.levelno for r in records] == [logging.WARNING]

    def test_mixed_batch_splits_levels(self, capture_logger):
        records = capture_logger(_LOGGER_NAME)
        emit_translation_warnings(
            [
                "Thinking config passed through (reasoning_mode=passthrough)",
                "Dropped unsupported output_config.format",
            ],
            {},
        )
        levels = {r.getMessage(): r.levelno for r in records}
        assert (
            levels["Translation: Thinking config passed through (reasoning_mode=passthrough)"]
            == logging.DEBUG
        )
        assert levels["Translation: Dropped unsupported output_config.format"] == logging.DEBUG


class TestRealTranslatorNoFlood:
    """Coupling tests: a normal Claude Code request must emit ZERO WARNING notices.

    These drive the real translators, so a reworded routine message that the emitter's
    allowlist no longer recognizes surfaces here as a resurrected WARNING flood.
    """

    def _claude_code_request(self) -> dict:
        # A normal Claude Code request as seen on the wire: output_config carries BOTH
        # effort=max AND format (a structured-output request, full {type, schema} shape)
        # alongside adaptive thinking. `format` is proven from live TUI traffic — the
        # "Dropped unsupported output_config.format" WARNING flood this branch fixes. The
        # combined effort+format+thinking shape is a deliberate superset: the emitter
        # classifies each notice independently, so one request exercising all three routine
        # notices is a stronger coupling probe than three separate ones. Every notice it
        # produces — thinking passthrough, effort handling, AND the format subkey drop —
        # must stay below WARNING so the shared TUI is not flooded. Omitting `format` here
        # is what let the flood slip past this very coupling test before.
        return {
            "model": "claude-opus-4-6",
            "max_tokens": 100,
            "output_config": {
                "effort": "max",
                "format": {
                    "type": "json_schema",
                    "schema": {"type": "object", "properties": {"answer": {"type": "string"}}},
                },
            },
            "thinking": {"type": "adaptive"},
            "messages": [{"role": "user", "content": "Hi"}],
        }

    def test_openai_normal_request_emits_no_warning(self, capture_logger):
        records = capture_logger(_LOGGER_NAME)
        _, warnings = anthropic_to_openai(self._claude_code_request())
        emit_translation_warnings(warnings, {})
        assert not [r for r in records if r.levelno >= logging.WARNING]

    def test_xai_normal_request_emits_no_warning(self, capture_logger, monkeypatch):
        monkeypatch.delenv("XAI_MODEL", raising=False)
        monkeypatch.delenv("XAI_REASONING_EFFORT", raising=False)
        records = capture_logger(_LOGGER_NAME)
        _, warnings = anthropic_to_xai(self._claude_code_request())
        emit_translation_warnings(warnings, {})
        assert not [r for r in records if r.levelno >= logging.WARNING]
