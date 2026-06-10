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
                    "model": "gpt-5.5",
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
        assert msg["usage"]["input_tokens"] == 50
        assert msg["usage"]["output_tokens"] == 0

    def test_response_created_usage_coerced_to_int(self):
        # message_start carries the first input_tokens estimate Claude Code's
        # /context bar renders. A float there breaks the integer accounting just
        # like on the completion path — coerce both ends of the stream.
        from claude_bridge.providers.openai import translate_openai_sse_event

        event = {
            "event": "response.created",
            "data": {
                "type": "response.created",
                "response": {
                    "id": "resp_flt_start",
                    "model": "gpt-5.5",
                    "status": "in_progress",
                    "usage": {"input_tokens": 42.0, "output_tokens": 0},
                },
            },
        }
        results = translate_openai_sse_event(event)
        usage = results[0]["data"]["message"]["usage"]
        assert usage["input_tokens"] == 50
        assert isinstance(usage["input_tokens"], int)

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
                    "model": "gpt-5.5",
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
        assert results[0]["data"]["usage"] == {"input_tokens": 12, "output_tokens": 30}

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
                    "model": "gpt-5.5",
                    "status": "incomplete",
                    "usage": {"input_tokens": 5, "output_tokens": 100},
                },
            },
        }
        results = translate_openai_sse_event(event)
        assert results[0]["data"]["delta"]["stop_reason"] == "max_tokens"

    def test_completed_max_output_tokens_maps_to_max_tokens(self):
        # incomplete_details.reason == "max_output_tokens" is the genuine
        # token-budget exhaustion signal — must surface as max_tokens.
        from claude_bridge.providers.openai import translate_openai_sse_event

        event = {
            "event": "response.completed",
            "data": {
                "type": "response.completed",
                "response": {
                    "id": "resp_mot",
                    "model": "gpt-5.5",
                    "status": "incomplete",
                    "incomplete_details": {"reason": "max_output_tokens"},
                    "usage": {"input_tokens": 5, "output_tokens": 100},
                },
            },
        }
        results = translate_openai_sse_event(event)
        message_delta = next(r for r in results if r["event"] == "message_delta")
        assert message_delta["data"]["delta"]["stop_reason"] == "max_tokens"

    def test_completed_content_filter_maps_to_end_turn_not_max_tokens(self):
        # A content-filtered completion is NOT budget exhaustion. Mapping it to
        # max_tokens makes Claude Code's auto-compact think it ran out of room and
        # retry forever. content_filter must terminate the turn cleanly (end_turn).
        from claude_bridge.providers.openai import translate_openai_sse_event

        event = {
            "event": "response.completed",
            "data": {
                "type": "response.completed",
                "response": {
                    "id": "resp_cf",
                    "model": "gpt-5.5",
                    "status": "incomplete",
                    "incomplete_details": {"reason": "content_filter"},
                    "usage": {"input_tokens": 5, "output_tokens": 3},
                },
            },
        }
        results = translate_openai_sse_event(event)
        message_delta = next(r for r in results if r["event"] == "message_delta")
        assert message_delta["data"]["delta"]["stop_reason"] == "end_turn"

    def test_completed_content_filter_synthesizes_refusal_block(self):
        # A bare end_turn with empty content would render as a blank assistant
        # turn in Claude Code. Synthesize a visible refusal text block so the
        # user learns why the turn stopped — mirrors the Gemini SAFETY path.
        from claude_bridge.providers.openai import translate_openai_sse_event

        event = {
            "event": "response.completed",
            "data": {
                "type": "response.completed",
                "response": {
                    "id": "resp_cf2",
                    "model": "gpt-5.5",
                    "status": "incomplete",
                    "incomplete_details": {"reason": "content_filter"},
                    "usage": {"input_tokens": 5, "output_tokens": 0},
                },
            },
        }
        results = translate_openai_sse_event(event)
        events = [r["event"] for r in results]
        assert events == [
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
        start = results[0]["data"]
        assert start["content_block"] == {"type": "text", "text": ""}
        delta = results[1]["data"]
        assert delta["delta"]["type"] == "text_delta"
        assert delta["delta"]["text"].strip() != ""

    def test_message_delta_usage_coerced_to_int(self):
        # Some providers emit float token counts. Anthropic usage fields are
        # integers — a float leaks the provider's wire format into Claude Code's
        # context accounting. Coerce on the streaming path too.
        from claude_bridge.providers.openai import translate_openai_sse_event

        event = {
            "event": "response.completed",
            "data": {
                "type": "response.completed",
                "response": {
                    "id": "resp_flt",
                    "model": "gpt-5.5",
                    "status": "completed",
                    "usage": {"input_tokens": 10.0, "output_tokens": 25.7},
                },
            },
        }
        results = translate_openai_sse_event(event)
        message_delta = next(r for r in results if r["event"] == "message_delta")
        usage = message_delta["data"]["usage"]
        assert usage["input_tokens"] == 12
        assert usage["output_tokens"] == 30
        assert isinstance(usage["input_tokens"], int)
        assert isinstance(usage["output_tokens"], int)


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

        provider = OpenAIProvider()

        created_event = _make_sse_event(
            "response.created",
            {
                "type": "response.created",
                "response": {
                    "id": "resp_1",
                    "model": "gpt-5.5",
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

        provider = OpenAIProvider()

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

        provider = OpenAIProvider()

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

        provider = OpenAIProvider()

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

        provider = OpenAIProvider()

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
        async for event in provider.translate_stream(_chunks_from([b"", text_event, b""])):
            events.append(event)

        assert len(events) == 1
        assert events[0]["data"]["delta"]["text"] == "ok"

    @pytest.mark.asyncio
    async def test_content_filter_refusal_block_gets_sequential_index(self):
        """A content-filtered stream with prior text yields a refusal block whose
        index follows the real text block (1), not a duplicate of it (0)."""
        from claude_bridge.providers.openai import OpenAIProvider

        provider = OpenAIProvider()

        part_added = _make_sse_event(
            "response.content_part.added",
            {"type": "response.content_part.added", "content_index": 0},
        )
        text_delta = _make_sse_event(
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "output_index": 0,
                "content_index": 0,
                "delta": "partial",
            },
        )
        text_done = _make_sse_event(
            "response.output_text.done",
            {
                "type": "response.output_text.done",
                "output_index": 0,
                "content_index": 0,
                "text": "partial",
            },
        )
        completed = _make_sse_event(
            "response.completed",
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_cf3",
                    "model": "gpt-5.5",
                    "status": "incomplete",
                    "incomplete_details": {"reason": "content_filter"},
                    "usage": {"input_tokens": 5, "output_tokens": 2},
                },
            },
        )

        events = []
        async for event in provider.translate_stream(
            _chunks_from([part_added + text_delta + text_done + completed])
        ):
            events.append(event)

        # Real text block is index 0; the synthesized refusal block must be index 1.
        refusal_starts = [
            e for e in events if e["event"] == "content_block_start" and e["data"]["index"] == 1
        ]
        assert len(refusal_starts) == 1
        refusal_delta = next(
            e
            for e in events
            if e["event"] == "content_block_delta"
            and e["data"].get("delta", {}).get("type") == "text_delta"
            and e["data"]["index"] == 1
        )
        assert refusal_delta["data"]["delta"]["text"].strip() != ""
        message_delta = next(e for e in events if e["event"] == "message_delta")
        assert message_delta["data"]["delta"]["stop_reason"] == "end_turn"


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


class TestStreamBufferBounding:
    """A provider that streams without "\n\n" event terminators must not grow the
    SSE buffer without bound (OOM) or hang — translate_stream caps the undrained
    buffer and aborts fast. See SCL-2."""

    @pytest.mark.asyncio
    async def test_aborts_when_unterminated_buffer_exceeds_cap(self, monkeypatch):
        from claude_bridge.providers import openai as openai_mod

        monkeypatch.setattr(openai_mod, "_MAX_SSE_BUFFER", 64)
        provider = openai_mod.OpenAIProvider()
        # Five 32-byte chunks with no "\n\n" → 160 bytes accumulated, never a complete
        # event. Oracle: the cap contract aborts a malformed stream rather than
        # buffering it without limit.
        chunks = [b"x" * 32] * 5

        with pytest.raises(RuntimeError, match="malformed"):
            async for _ in provider.translate_stream(_chunks_from(chunks)):
                pass

    @pytest.mark.asyncio
    async def test_does_not_abort_at_exact_cap(self, monkeypatch):
        from claude_bridge.providers import openai as openai_mod

        monkeypatch.setattr(openai_mod, "_MAX_SSE_BUFFER", 64)
        provider = openai_mod.OpenAIProvider()
        # Exactly cap bytes, still unterminated: the bound is exclusive, so this must
        # NOT abort — kills the > vs >= off-by-one on the cap check. Garbage bytes
        # parse to zero events at the tail; the assertion is the absence of a raise.
        events = [e async for e in provider.translate_stream(_chunks_from([b"x" * 64]))]
        assert events == []
