"""Tests for the redacted compatibility trace — structural-only summaries and
the self-guarding trace hooks. Extracted from test_proxy.py (QAL4): the redaction
oracle and trace I/O form a cohesive surface distinct from the proxy server tests.

Every summarizer and trace hook must emit structural shape only (counts, type
names, lengths, ids) and never the underlying prompt text, tool arguments, tool
results, reasoning, or credential patterns. The redaction oracle below plants a
marker in every content-bearing field; no marker may survive into any output.
"""

from __future__ import annotations

import base64
import json

# Every marker is planted in a content-bearing position of the fixtures below.
# The redaction oracle: none of these substrings may appear in any trace output.
# Values derived from the planted input — never from running the summarizer.
_SECRET_MARKERS = [
    "SUPER_SECRET_PROMPT",  # system prompt text
    "SECRET_TOOL_DESC",  # tool description
    "LEAKED_MESSAGE_TEXT",  # user message text
    "SECRET_REASONING",  # thinking block text
    "SECRET_SIGNATURE",  # thinking block signature (encrypted reasoning)
    "hunter2",  # tool_use input value
    "/etc/passwd",  # tool_use input value (file path)
    "LEAKED_TOOL_OUTPUT",  # tool_result content
    "sk-secretapikey123",  # api-key pattern embedded in content
    "Bearer tok_secret",  # bearer-token pattern embedded in content
]


def _secret_laden_request() -> dict:
    """An Anthropic request with a secret marker in every content-bearing field.

    Shape mirrors a Claude Code tool loop: system prompt, tool definitions,
    a user turn, an assistant thinking+tool_use turn, and a tool_result turn.
    """
    return {
        "model": "claude-opus-4-8",
        "stream": True,
        "system": [{"type": "text", "text": "SUPER_SECRET_PROMPT system instructions"}],
        "tools": [
            {
                "name": "Read",
                "description": "SECRET_TOOL_DESC read a file",
                "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
            {"name": "Bash", "description": "run a command", "input_schema": {"type": "object"}},
        ],
        "tool_choice": {"type": "tool", "name": "Read"},
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "LEAKED_MESSAGE_TEXT Bearer tok_secret sk-secretapikey123",
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "SECRET_REASONING let me read it",
                        "signature": "SECRET_SIGNATURE",
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_01",
                        "name": "Read",
                        "input": {"path": "/etc/passwd", "password": "hunter2"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_01",
                        "content": [{"type": "text", "text": "LEAKED_TOOL_OUTPUT contents"}],
                    }
                ],
            },
        ],
    }


def _secret_laden_response() -> dict:
    """An Anthropic response carrying secret markers in text and tool_use input."""
    return {
        "id": "msg_01",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-4-8",
        "stop_reason": "tool_use",
        "content": [
            {"type": "text", "text": "SECRET_RESPONSE_TEXT here is the answer"},
            {
                "type": "tool_use",
                "id": "toolu_02",
                "name": "Bash",
                "input": {"command": "echo SECRET_COMMAND"},
            },
        ],
        "usage": {"input_tokens": 1234, "output_tokens": 56},
    }


def _assert_no_secrets(summary: dict) -> None:
    """The redaction oracle — no planted secret may survive into the summary."""
    blob = json.dumps(summary)
    for marker in _SECRET_MARKERS:
        assert marker not in blob, f"secret leaked into trace: {marker!r}"


class TestSummarizeAnthropicRequest:
    """_summarize_anthropic_request emits structural shape only — counts, type
    names, tool names, lengths — and never the underlying prompt content."""

    def test_structural_fields_match_input(self):
        from claude_bridge.proxy import _summarize_anthropic_request

        summary = _summarize_anthropic_request(_secret_laden_request())
        assert summary["model"] == "claude-opus-4-8"
        assert summary["stream"] is True
        assert summary["message_count"] == 3
        assert summary["tool_count"] == 2
        assert summary["tool_names"] == ["Bash", "Read"]
        assert summary["tool_choice"] == "tool"
        assert summary["block_types"] == {
            "text": 1,
            "thinking": 1,
            "tool_use": 1,
            "tool_result": 1,
        }

    def test_system_chars_is_positive_length_not_content(self):
        from claude_bridge.proxy import _summarize_anthropic_request

        summary = _summarize_anthropic_request(_secret_laden_request())
        # A length, derived structurally — present and positive for a non-empty system.
        assert isinstance(summary["system_chars"], int)
        assert summary["system_chars"] > 0

    def test_no_secrets_leak(self):
        from claude_bridge.proxy import _summarize_anthropic_request

        _assert_no_secrets(_summarize_anthropic_request(_secret_laden_request()))

    def test_string_content_counts_as_text_block(self):
        from claude_bridge.proxy import _summarize_anthropic_request

        # Anthropic shorthand: content may be a bare string (one text block).
        request = {"model": "m", "messages": [{"role": "user", "content": "SECRET_STRING"}]}
        summary = _summarize_anthropic_request(request)
        assert summary["block_types"] == {"text": 1}
        assert "SECRET_STRING" not in json.dumps(summary)

    def test_absent_tools_and_choice_are_empty(self):
        from claude_bridge.proxy import _summarize_anthropic_request

        summary = _summarize_anthropic_request({"model": "m", "messages": []})
        assert summary["tool_count"] == 0
        assert summary["tool_names"] == []
        assert summary["tool_choice"] is None
        assert summary["system_chars"] == 0


class TestSummarizeProviderRequest:
    """_summarize_provider_request summarizes the translated provider request
    without leaking any translated input content."""

    def test_structural_fields_and_warning_count(self):
        from claude_bridge.providers.openai import anthropic_to_openai
        from claude_bridge.proxy import _summarize_provider_request

        translated, warnings = anthropic_to_openai(_secret_laden_request())
        summary = _summarize_provider_request(translated, warnings)
        assert summary["tool_count"] == 2
        assert summary["tool_names"] == ["Bash", "Read"]
        assert summary["input_items"] >= 3
        assert summary["warning_count"] == len(warnings)
        assert summary["stream"] is True

    def test_warning_strings_included_for_trace(self):
        # REQ1: the trace carries the sanitized warning *strings*, not just a count,
        # so a degraded translation is diagnosable from the trace alone (T-003 spec).
        from claude_bridge.proxy import _summarize_provider_request

        warnings = [
            "Stripped 'thinking' config (reasoning_mode=drop)",
            "Unsupported tool_choice type 'x', omitting tool_choice",
        ]
        summary = _summarize_provider_request({"model": "m", "input": []}, warnings)
        assert summary["warnings"] == warnings
        assert summary["warning_count"] == 2

    def test_forced_tool_choice_renders_structurally(self):
        from claude_bridge.proxy import _summarize_provider_request

        translated = {
            "model": "gpt-5.5",
            "input": [],
            "tools": [{"type": "function", "name": "Read"}],
            "tool_choice": {"type": "function", "name": "Read"},
        }
        summary = _summarize_provider_request(translated, [])
        assert summary["tool_choice"] == "function:Read"

    def test_parallel_flag_emitted_only_when_present(self):
        from claude_bridge.proxy import _summarize_provider_request

        with_flag = _summarize_provider_request(
            {"model": "m", "input": [], "parallel_tool_calls": False}, []
        )
        without_flag = _summarize_provider_request({"model": "m", "input": []}, [])
        assert with_flag["parallel_tool_calls"] is False
        assert "parallel_tool_calls" not in without_flag

    def test_no_secrets_leak(self):
        from claude_bridge.providers.openai import anthropic_to_openai
        from claude_bridge.proxy import _summarize_provider_request

        translated, warnings = anthropic_to_openai(_secret_laden_request())
        _assert_no_secrets(_summarize_provider_request(translated, warnings))

    def test_injected_encrypted_reasoning_never_leaks(self):
        # The provider echoes opaque encrypted reasoning into the outbound request
        # (T-005). The summarizer must count it as an input item, never serialize it.
        from claude_bridge.proxy import _summarize_provider_request

        translated = {
            "model": "gpt-5.5",
            "input": [
                {"type": "reasoning", "id": "rs_1", "encrypted_content": "SECRET_REASONING"},
                {"type": "function_call", "id": "fc_1", "call_id": "fc_1", "name": "Read"},
            ],
        }
        summary = _summarize_provider_request(translated, [])
        assert summary["input_items"] == 2
        assert "SECRET_REASONING" not in json.dumps(summary)


class TestMediaTraceSummary:
    """Media blocks are summarized in the inbound trace as {kind, media_type,
    approx_bytes} — structural metadata only. The base64 payload never appears,
    and _summarize_provider_request never echoes a translated data-URL part."""

    def test_inbound_media_summarized_as_kind_type_bytes(self):
        from claude_bridge.proxy import _summarize_anthropic_request

        # 404-byte payload (b"\x89PNG" + 400 zero bytes); approx_bytes recovers it.
        img_b64 = base64.b64encode(b"\x89PNG" + b"\x00" * 400).decode()
        request = {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "look"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": img_b64,
                            },
                        },
                    ],
                }
            ],
        }
        summary = _summarize_anthropic_request(request)
        media = summary["media"]
        assert len(media) == 1
        assert media[0]["kind"] == "image"
        assert media[0]["media_type"] == "image/png"
        assert abs(media[0]["approx_bytes"] - 404) <= 2  # recovers the 404-byte payload
        assert img_b64 not in json.dumps(summary)

    def test_tool_result_nested_media_is_summarized(self):
        from claude_bridge.proxy import _summarize_anthropic_request

        pdf_b64 = base64.b64encode(b"%PDF-1.4" + b"\x00" * 800).decode()
        request = {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": [
                                {
                                    "type": "document",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "application/pdf",
                                        "data": pdf_b64,
                                    },
                                }
                            ],
                        }
                    ],
                }
            ],
        }
        summary = _summarize_anthropic_request(request)
        kinds = [(m["kind"], m["media_type"]) for m in summary["media"]]
        assert ("document", "application/pdf") in kinds
        assert pdf_b64 not in json.dumps(summary)

    def test_no_media_yields_empty_media_list(self):
        from claude_bridge.proxy import _summarize_anthropic_request

        summary = _summarize_anthropic_request(
            {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        )
        assert summary["media"] == []

    def test_provider_request_summary_never_echoes_base64_dataurl(self):
        # _summarize_provider_request must not serialize translated input content —
        # even a real input_image data-URL part stays out of the trace.
        from claude_bridge.proxy import _summarize_provider_request

        img_b64 = base64.b64encode(b"\x89PNG" + b"\x00" * 400).decode()
        translated = {
            "model": "gpt-5.5",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "look"},
                        {"type": "input_image", "image_url": f"data:image/png;base64,{img_b64}"},
                    ],
                }
            ],
        }
        summary = _summarize_provider_request(translated, [])
        blob = json.dumps(summary)
        assert img_b64 not in blob
        assert "data:image/png;base64" not in blob
        assert summary["input_items"] == 1


class TestSummarizeAnthropicResponse:
    """_summarize_anthropic_response emits stop_reason, block counts, and token
    usage — never response text or tool_use arguments."""

    def test_structural_fields_match_input(self):
        from claude_bridge.proxy import _summarize_anthropic_response

        summary = _summarize_anthropic_response(_secret_laden_response())
        assert summary["model"] == "claude-opus-4-8"
        assert summary["stop_reason"] == "tool_use"
        assert summary["block_types"] == {"text": 1, "tool_use": 1}
        assert summary["input_tokens"] == 1234
        assert summary["output_tokens"] == 56

    def test_no_secrets_leak(self):
        from claude_bridge.proxy import _summarize_anthropic_response

        summary = _summarize_anthropic_response(_secret_laden_response())
        assert "SECRET_RESPONSE_TEXT" not in json.dumps(summary)
        assert "SECRET_COMMAND" not in json.dumps(summary)


class TestSummarizeStreamEvent:
    """_summarize_stream_event classifies one SSE event by its structural fields
    (event name, block index, block/delta type, stop_reason) and drops content."""

    def test_content_block_start_emits_block_type(self):
        from claude_bridge.proxy import _summarize_stream_event

        event = {
            "event": "content_block_start",
            "data": {
                "index": 1,
                "content_block": {"type": "tool_use", "id": "toolu_03", "name": "Read"},
            },
        }
        summary = _summarize_stream_event(event)
        assert summary["sse"] == "content_block_start"
        assert summary["index"] == 1
        assert summary["block_type"] == "tool_use"

    def test_text_delta_drops_text_keeps_type(self):
        from claude_bridge.proxy import _summarize_stream_event

        event = {
            "event": "content_block_delta",
            "data": {"index": 0, "delta": {"type": "text_delta", "text": "SECRET_DELTA_TEXT"}},
        }
        summary = _summarize_stream_event(event)
        assert summary["delta_type"] == "text_delta"
        assert summary["index"] == 0
        assert "SECRET_DELTA_TEXT" not in json.dumps(summary)

    def test_input_json_delta_drops_partial_json(self):
        from claude_bridge.proxy import _summarize_stream_event

        event = {
            "event": "content_block_delta",
            "data": {
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": '{"path": "SECRET_PATH"}'},
            },
        }
        summary = _summarize_stream_event(event)
        assert summary["delta_type"] == "input_json_delta"
        assert "SECRET_PATH" not in json.dumps(summary)

    def test_message_delta_emits_stop_reason(self):
        from claude_bridge.proxy import _summarize_stream_event

        event = {
            "event": "message_delta",
            "data": {"delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 99}},
        }
        summary = _summarize_stream_event(event)
        assert summary["sse"] == "message_delta"
        assert summary["stop_reason"] == "end_turn"
        assert summary["output_tokens"] == 99


class TestTraceHooks:
    """The self-guarding trace hooks: off by default, redacted when on,
    and never raise — tracing can never break a request."""

    def test_inbound_disabled_writes_nothing(self, tmp_path, monkeypatch):
        from claude_bridge.proxy import _trace_inbound_request

        monkeypatch.delenv("CLAUDE_BRIDGE_TRACE_PATH", raising=False)
        target = tmp_path / "trace.jsonl"
        _trace_inbound_request(json.dumps(_secret_laden_request()).encode())
        assert not target.exists()

    def test_inbound_enabled_writes_redacted_structural_line(self, tmp_path, monkeypatch):
        from claude_bridge.proxy import _trace_inbound_request

        target = tmp_path / "trace.jsonl"
        monkeypatch.setenv("CLAUDE_BRIDGE_TRACE_PATH", str(target))
        _trace_inbound_request(json.dumps(_secret_laden_request()).encode())
        content = target.read_text(encoding="utf-8")
        for marker in _SECRET_MARKERS:
            assert marker not in content, f"secret leaked into trace file: {marker!r}"
        record = json.loads(content.splitlines()[0])
        assert record["event"] == "inbound_request"
        assert record["model"] == "claude-opus-4-8"
        assert record["message_count"] == 3

    def test_inbound_malformed_body_never_raises(self, tmp_path, monkeypatch):
        from claude_bridge.proxy import _trace_inbound_request

        target = tmp_path / "trace.jsonl"
        monkeypatch.setenv("CLAUDE_BRIDGE_TRACE_PATH", str(target))
        # Must swallow the JSON parse error — no file line, no exception.
        _trace_inbound_request(b"not json{{{")

    def test_provider_request_hook_redacts(self, tmp_path, monkeypatch):
        from claude_bridge.providers.openai import anthropic_to_openai
        from claude_bridge.proxy import _trace_provider_request

        target = tmp_path / "trace.jsonl"
        monkeypatch.setenv("CLAUDE_BRIDGE_TRACE_PATH", str(target))
        translated, warnings = anthropic_to_openai(_secret_laden_request())
        _trace_provider_request(translated, warnings)
        content = target.read_text(encoding="utf-8")
        for marker in _SECRET_MARKERS:
            assert marker not in content
        assert json.loads(content.splitlines()[0])["event"] == "provider_request"

    def test_provider_response_hook_redacts(self, tmp_path, monkeypatch):
        from claude_bridge.proxy import _trace_provider_response

        target = tmp_path / "trace.jsonl"
        monkeypatch.setenv("CLAUDE_BRIDGE_TRACE_PATH", str(target))
        _trace_provider_response(_secret_laden_response())
        content = target.read_text(encoding="utf-8")
        assert "SECRET_RESPONSE_TEXT" not in content
        assert "SECRET_COMMAND" not in content
        assert json.loads(content.splitlines()[0])["stop_reason"] == "tool_use"

    def test_stream_event_hook_redacts(self, tmp_path, monkeypatch):
        from claude_bridge.proxy import _trace_stream_event

        target = tmp_path / "trace.jsonl"
        monkeypatch.setenv("CLAUDE_BRIDGE_TRACE_PATH", str(target))
        _trace_stream_event(
            {
                "event": "content_block_delta",
                "data": {"index": 0, "delta": {"type": "text_delta", "text": "SECRET_DELTA_TEXT"}},
            }
        )
        content = target.read_text(encoding="utf-8")
        assert "SECRET_DELTA_TEXT" not in content
        assert json.loads(content.splitlines()[0])["delta_type"] == "text_delta"

    def test_stream_event_disabled_writes_nothing(self, tmp_path, monkeypatch):
        from claude_bridge.proxy import _trace_stream_event

        monkeypatch.delenv("CLAUDE_BRIDGE_TRACE_PATH", raising=False)
        target = tmp_path / "trace.jsonl"
        _trace_stream_event({"event": "message_stop", "data": {}})
        assert not target.exists()
