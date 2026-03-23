"""Tests for SSE streaming utilities and OpenAI → Anthropic SSE translation."""

from __future__ import annotations

import json

import pytest

from claude_bridge.stream import format_anthropic_sse, parse_sse_events


# ---------------------------------------------------------------------------
# parse_sse_events
# ---------------------------------------------------------------------------


class TestParseSSEEvents:
    """Parse raw SSE bytes into structured event dicts."""

    def test_single_event(self):
        raw = b'event: message_start\ndata: {"type":"message_start"}\n\n'
        events = parse_sse_events(raw)
        assert len(events) == 1
        assert events[0]["event"] == "message_start"
        assert events[0]["data"] == {"type": "message_start"}

    def test_multiple_events(self):
        raw = (
            b"event: content_block_start\n"
            b'data: {"type":"content_block_start","index":0}\n\n'
            b"event: content_block_delta\n"
            b'data: {"type":"content_block_delta","index":0}\n\n'
        )
        events = parse_sse_events(raw)
        assert len(events) == 2
        assert events[0]["event"] == "content_block_start"
        assert events[1]["event"] == "content_block_delta"

    def test_data_only_event_no_event_field(self):
        raw = b'data: {"type":"ping"}\n\n'
        events = parse_sse_events(raw)
        assert len(events) == 1
        assert events[0]["event"] == ""
        assert events[0]["data"] == {"type": "ping"}

    def test_data_done_marker_skipped(self):
        raw = b"event: done\ndata: [DONE]\n\n"
        events = parse_sse_events(raw)
        assert len(events) == 0

    def test_empty_input(self):
        assert parse_sse_events(b"") == []

    def test_crlf_line_endings(self):
        raw = b'event: ping\r\ndata: {"type":"ping"}\r\n\r\n'
        events = parse_sse_events(raw)
        assert len(events) == 1
        assert events[0]["event"] == "ping"


# ---------------------------------------------------------------------------
# format_anthropic_sse
# ---------------------------------------------------------------------------


class TestFormatAnthropicSSE:
    """Format Anthropic SSE events as wire bytes."""

    def test_basic_format(self):
        result = format_anthropic_sse("message_start", {"type": "message_start"})
        assert result == b'event: message_start\ndata: {"type":"message_start"}\n\n'

    def test_content_block_delta(self):
        data = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "hi"},
        }
        result = format_anthropic_sse("content_block_delta", data)
        lines = result.decode().split("\n")
        assert lines[0] == "event: content_block_delta"
        assert lines[1].startswith("data: ")
        parsed = json.loads(lines[1][len("data: ") :])
        assert parsed["delta"]["text"] == "hi"
        # Must end with double newline
        assert result.endswith(b"\n\n")


# ---------------------------------------------------------------------------
# OpenAI → Anthropic SSE event translation
# ---------------------------------------------------------------------------


class TestOpenAIToAnthropicSSETranslation:
    """Translate OpenAI Responses API SSE events to Anthropic Messages SSE events."""

    def test_response_created_emits_message_start(self):
        from claude_bridge.providers.openai import translate_openai_sse_event

        event = {
            "event": "response.created",
            "data": {
                "type": "response.created",
                "response": {
                    "id": "resp_123",
                    "model": "gpt-5.4",
                    "status": "in_progress",
                    "usage": {"input_tokens": 42, "output_tokens": 0},
                },
            },
        }
        results = translate_openai_sse_event(event)
        assert len(results) == 2  # message_start + ping
        assert results[0]["event"] == "message_start"
        assert results[1]["event"] == "ping"
        msg = results[0]["data"]["message"]
        assert msg["id"].startswith("msg_bridge_")
        assert msg["role"] == "assistant"
        assert msg["usage"]["input_tokens"] == 42
        assert msg["usage"]["output_tokens"] == 0

    def test_content_part_added_emits_content_block_start(self):
        from claude_bridge.providers.openai import translate_openai_sse_event

        event = {
            "event": "response.content_part.added",
            "data": {
                "type": "response.content_part.added",
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": ""},
            },
        }
        results = translate_openai_sse_event(event)
        assert len(results) == 1
        assert results[0]["event"] == "content_block_start"
        assert results[0]["data"]["index"] == 0
        assert results[0]["data"]["content_block"]["type"] == "text"

    def test_output_text_delta_emits_content_block_delta(self):
        from claude_bridge.providers.openai import translate_openai_sse_event

        event = {
            "event": "response.output_text.delta",
            "data": {
                "type": "response.output_text.delta",
                "output_index": 0,
                "content_index": 0,
                "delta": "Hello world",
            },
        }
        results = translate_openai_sse_event(event)
        assert len(results) == 1
        assert results[0]["event"] == "content_block_delta"
        assert results[0]["data"]["delta"]["type"] == "text_delta"
        assert results[0]["data"]["delta"]["text"] == "Hello world"
        assert results[0]["data"]["index"] == 0

    def test_output_text_done_emits_content_block_stop(self):
        from claude_bridge.providers.openai import translate_openai_sse_event

        event = {
            "event": "response.output_text.done",
            "data": {
                "type": "response.output_text.done",
                "output_index": 0,
                "content_index": 0,
                "text": "Full text here",
            },
        }
        results = translate_openai_sse_event(event)
        assert len(results) == 1
        assert results[0]["event"] == "content_block_stop"
        assert results[0]["data"]["index"] == 0

    def test_response_completed_emits_message_delta_and_stop(self):
        from claude_bridge.providers.openai import translate_openai_sse_event

        event = {
            "event": "response.completed",
            "data": {
                "type": "response.completed",
                "response": {
                    "id": "resp_456",
                    "model": "gpt-5.4",
                    "status": "completed",
                    "usage": {"input_tokens": 10, "output_tokens": 25},
                },
            },
        }
        results = translate_openai_sse_event(event)
        assert len(results) == 2

        # First: message_delta with stop_reason + usage
        assert results[0]["event"] == "message_delta"
        assert results[0]["data"]["delta"]["stop_reason"] == "end_turn"
        assert results[0]["data"]["usage"]["output_tokens"] == 25

        # Second: message_stop
        assert results[1]["event"] == "message_stop"
        assert results[1]["data"]["type"] == "message_stop"

    def test_unknown_event_returns_empty(self):
        from claude_bridge.providers.openai import translate_openai_sse_event

        event = {
            "event": "response.output_item.added",
            "data": {"type": "response.output_item.added"},
        }
        results = translate_openai_sse_event(event)
        assert results == []

    def test_incomplete_status_maps_to_max_tokens(self):
        from claude_bridge.providers.openai import translate_openai_sse_event

        event = {
            "event": "response.completed",
            "data": {
                "type": "response.completed",
                "response": {
                    "id": "resp_789",
                    "model": "gpt-5.4",
                    "status": "incomplete",
                    "usage": {"input_tokens": 5, "output_tokens": 100},
                },
            },
        }
        results = translate_openai_sse_event(event)
        assert results[0]["data"]["delta"]["stop_reason"] == "max_tokens"


# ---------------------------------------------------------------------------
# translate_stream integration tests
# ---------------------------------------------------------------------------


async def _chunks_from(byte_list: list[bytes]):
    """Async generator yielding bytes from a list (simulates HTTP chunks)."""
    for chunk in byte_list:
        yield chunk


def _make_sse_event(event_type: str, data: dict) -> bytes:
    """Build raw SSE bytes for one event."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n".encode()


class TestTranslateStream:
    """Integration tests for OpenAIProvider.translate_stream()."""

    @pytest.mark.asyncio
    async def test_single_chunk_with_multiple_events(self):
        """Multiple SSE events in one chunk are all translated."""
        from claude_bridge.providers.openai import OpenAIProvider

        provider = OpenAIProvider.__new__(OpenAIProvider)

        created_event = _make_sse_event(
            "response.created",
            {
                "type": "response.created",
                "response": {
                    "id": "resp_1",
                    "model": "gpt-5.4",
                    "status": "in_progress",
                    "usage": {"input_tokens": 10, "output_tokens": 0},
                },
            },
        )
        text_delta = _make_sse_event(
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "output_index": 0,
                "content_index": 0,
                "delta": "Hello",
            },
        )

        chunks = [created_event + text_delta]
        events = []
        async for event in provider.translate_stream(_chunks_from(chunks)):
            events.append(event)

        # response.created → message_start + ping, text.delta → content_block_delta
        assert len(events) == 3
        assert events[0]["event"] == "message_start"
        assert events[1]["event"] == "ping"
        assert events[2]["event"] == "content_block_delta"
        assert events[2]["data"]["delta"]["text"] == "Hello"

    @pytest.mark.asyncio
    async def test_event_split_across_chunks(self):
        """An SSE event split across two byte chunks is correctly buffered."""
        from claude_bridge.providers.openai import OpenAIProvider

        provider = OpenAIProvider.__new__(OpenAIProvider)

        full_event = _make_sse_event(
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "output_index": 0,
                "content_index": 0,
                "delta": "world",
            },
        )

        # Split the event in the middle
        mid = len(full_event) // 2
        chunk1 = full_event[:mid]
        chunk2 = full_event[mid:]

        events = []
        async for event in provider.translate_stream(_chunks_from([chunk1, chunk2])):
            events.append(event)

        assert len(events) == 1
        assert events[0]["event"] == "content_block_delta"
        assert events[0]["data"]["delta"]["text"] == "world"

    @pytest.mark.asyncio
    async def test_crlf_events_handled(self):
        """CRLF line endings in SSE are normalized and parsed correctly."""
        from claude_bridge.providers.openai import OpenAIProvider

        provider = OpenAIProvider.__new__(OpenAIProvider)

        event_bytes = (
            b"event: response.output_text.delta\r\n"
            b'data: {"type":"response.output_text.delta",'
            b'"output_index":0,"content_index":0,"delta":"hi"}\r\n\r\n'
        )

        events = []
        async for event in provider.translate_stream(_chunks_from([event_bytes])):
            events.append(event)

        assert len(events) == 1
        assert events[0]["data"]["delta"]["text"] == "hi"

    @pytest.mark.asyncio
    async def test_skipped_events_produce_no_output(self):
        """Events with no Anthropic equivalent are silently skipped."""
        from claude_bridge.providers.openai import OpenAIProvider

        provider = OpenAIProvider.__new__(OpenAIProvider)

        # response.output_item.added for a non-function_call item produces no output
        event_bytes = _make_sse_event(
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"type": "message", "content": []},
            },
        )

        events = []
        async for event in provider.translate_stream(_chunks_from([event_bytes])):
            events.append(event)

        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_empty_chunks_skipped(self):
        """Empty byte chunks don't produce events or errors."""
        from claude_bridge.providers.openai import OpenAIProvider

        provider = OpenAIProvider.__new__(OpenAIProvider)

        text_event = _make_sse_event(
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "output_index": 0,
                "content_index": 0,
                "delta": "ok",
            },
        )

        events = []
        async for event in provider.translate_stream(
            _chunks_from([b"", text_event, b""])
        ):
            events.append(event)

        assert len(events) == 1
        assert events[0]["data"]["delta"]["text"] == "ok"


# ---------------------------------------------------------------------------
# _extract_completed_response tests
# ---------------------------------------------------------------------------


class TestExtractCompletedResponse:
    """Tests for proxy._extract_completed_response() SSE parser."""

    def test_finds_response_completed(self):
        from claude_bridge.proxy import _extract_completed_response

        sse = (
            b'event: response.created\ndata: {"type":"response.created","response":{"id":"r1"}}\n\n'
            b'event: response.completed\ndata: {"type":"response.completed",'
            b'"response":{"id":"r1","status":"completed","output":[]}}\n\n'
        )
        result = _extract_completed_response(sse)
        assert result is not None
        assert result["id"] == "r1"
        assert result["status"] == "completed"

    def test_returns_none_when_no_completed(self):
        from claude_bridge.proxy import _extract_completed_response

        sse = b'event: response.created\ndata: {"type":"response.created","response":{"id":"r1"}}\n\n'
        result = _extract_completed_response(sse)
        assert result is None

    def test_handles_malformed_json_lines(self):
        from claude_bridge.proxy import _extract_completed_response

        sse = (
            b"data: not-json\n\n"
            b'data: {"type":"response.completed","response":{"id":"r2"}}\n\n'
        )
        result = _extract_completed_response(sse)
        assert result is not None
        assert result["id"] == "r2"


# ---------------------------------------------------------------------------
# SSE format round-trip test
# ---------------------------------------------------------------------------


class TestSSEFormatRoundTrip:
    """Verify format_anthropic_sse output can be parsed back."""

    def test_format_then_parse_roundtrip(self):
        data = {"type": "content_block_delta", "index": 0, "delta": {"text": "hello"}}
        wire_bytes = format_anthropic_sse("content_block_delta", data)
        parsed = parse_sse_events(wire_bytes)
        assert len(parsed) == 1
        assert parsed[0]["event"] == "content_block_delta"
        assert parsed[0]["data"]["delta"]["text"] == "hello"
