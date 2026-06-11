"""Tests for SSE streaming utilities and OpenAI → Anthropic SSE translation."""

from __future__ import annotations

import json

import pytest

from claude_bridge.stream import (
    format_anthropic_sse,
    iter_sse_event_blobs,
    parse_sse_events,
)

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


class TestTerminalStreamEvents:
    """Every Responses terminal event type must produce an Anthropic stream
    terminator.

    Oracle: the OpenAI Responses streaming API ends a stream with ONE of four
    distinct top-level event types — ``response.completed`` (success),
    ``response.incomplete`` (``max_output_tokens``/``content_filter``),
    ``response.failed`` (``server_error``/...), or a top-level ``error`` event.
    A ``response.completed`` event always carries ``status: "completed"``; the
    incomplete/failed states arrive under their OWN event type, never nested in
    a completed event. Anthropic finalizes a turn only on ``message_stop`` (or an
    ``error`` event terminating the stream). So each non-completed terminal event
    must translate to a terminator — otherwise Claude Code halts mid-turn.
    """

    def test_response_incomplete_event_maps_to_max_tokens_and_stops(self):
        # Real wire shape: event TYPE is response.incomplete (not a completed
        # event with status nested inside). max_output_tokens → max_tokens.
        from claude_bridge.providers.openai import translate_openai_sse_event

        event = {
            "event": "response.incomplete",
            "data": {
                "type": "response.incomplete",
                "response": {
                    "id": "resp_inc",
                    "model": "gpt-5.5",
                    "status": "incomplete",
                    "incomplete_details": {"reason": "max_output_tokens"},
                    "usage": {"input_tokens": 5, "output_tokens": 100},
                },
            },
        }
        results = translate_openai_sse_event(event)
        names = [r["event"] for r in results]
        assert "message_stop" in names
        message_delta = next(r for r in results if r["event"] == "message_delta")
        assert message_delta["data"]["delta"]["stop_reason"] == "max_tokens"

    def test_response_incomplete_content_filter_ends_turn_with_refusal(self):
        # content_filter is NOT budget exhaustion → end_turn, plus a visible
        # refusal block so the turn does not render blank.
        from claude_bridge.providers.openai import translate_openai_sse_event

        event = {
            "event": "response.incomplete",
            "data": {
                "type": "response.incomplete",
                "response": {
                    "id": "resp_inc_cf",
                    "model": "gpt-5.5",
                    "status": "incomplete",
                    "incomplete_details": {"reason": "content_filter"},
                    "usage": {"input_tokens": 5, "output_tokens": 2},
                },
            },
        }
        results = translate_openai_sse_event(event)
        # A bare end_turn with empty content would render as a blank assistant turn
        # in Claude Code; a visible refusal text block must precede the terminator so
        # the user learns why the turn stopped (mirrors the Gemini SAFETY path).
        assert [r["event"] for r in results] == [
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
        assert results[0]["data"]["content_block"] == {"type": "text", "text": ""}
        refusal_delta = results[1]["data"]["delta"]
        assert refusal_delta["type"] == "text_delta"
        assert refusal_delta["text"].strip() != ""
        message_delta = next(r for r in results if r["event"] == "message_delta")
        assert message_delta["data"]["delta"]["stop_reason"] == "end_turn"

    def test_response_incomplete_without_details_defaults_to_max_tokens(self):
        # GPT-5 can emit status "incomplete" with null incomplete_details. The
        # conservative oracle: absent a reason, treat it as token exhaustion
        # (max_tokens) so Claude Code knows the turn was truncated, not clean.
        from claude_bridge.providers.openai import translate_openai_sse_event

        event = {
            "event": "response.incomplete",
            "data": {
                "type": "response.incomplete",
                "response": {
                    "id": "resp_inc_nodetails",
                    "model": "gpt-5.5",
                    "status": "incomplete",
                    "usage": {"input_tokens": 5, "output_tokens": 100},
                },
            },
        }
        results = translate_openai_sse_event(event)
        names = [r["event"] for r in results]
        assert "message_stop" in names
        message_delta = next(r for r in results if r["event"] == "message_delta")
        assert message_delta["data"]["delta"]["stop_reason"] == "max_tokens"

    def test_response_failed_event_emits_error_terminator(self):
        # A failed upstream response is an API error, not an assistant message —
        # faithful translation is an Anthropic error event carrying the reason.
        from claude_bridge.providers.openai import translate_openai_sse_event

        event = {
            "event": "response.failed",
            "data": {
                "type": "response.failed",
                "response": {
                    "id": "resp_fail",
                    "status": "failed",
                    "error": {
                        "code": "server_error",
                        "message": "The model failed to generate a response.",
                    },
                },
            },
        }
        results = translate_openai_sse_event(event)
        error_events = [r for r in results if r["event"] == "error"]
        assert len(error_events) == 1
        assert error_events[0]["data"]["type"] == "error"
        # The SPECIFIC upstream reason must be surfaced verbatim (so the user learns
        # WHY the turn failed) — not collapsed to a generic default.
        assert error_events[0]["data"]["error"]["message"] == (
            "The model failed to generate a response."
        )

    def test_top_level_error_event_emits_error_terminator(self):
        # The Responses stream can emit a bare top-level ``error`` event on a
        # mid-stream server failure; it must terminate the Anthropic stream.
        from claude_bridge.providers.openai import translate_openai_sse_event

        event = {
            "event": "error",
            "data": {
                "type": "error",
                "code": "server_error",
                "message": "upstream exploded",
            },
        }
        results = translate_openai_sse_event(event)
        error_events = [r for r in results if r["event"] == "error"]
        assert len(error_events) == 1
        assert error_events[0]["data"]["type"] == "error"
        assert error_events[0]["data"]["error"]["message"] == "upstream exploded"


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


async def _collect(aiter):
    """Drain an async iterator into a list."""
    return [item async for item in aiter]


class TestIterSSEEventBlobs:
    """Direct unit tests for the shared SSE byte-framing owner.

    Oracle: the SSE wire format delimits events with a blank line (``\\n\\n``),
    CRLF normalized to LF (W3C EventSource spec). A framer must yield each event
    INCLUDING its terminator and split exactly on the delimiter — values derived
    from the spec, never from running the framer.
    """

    @pytest.mark.asyncio
    async def test_single_event_yielded_with_terminator_intact(self):
        # One \n\n-terminated event in one chunk → exactly that event, byte-for-byte.
        blobs = await _collect(iter_sse_event_blobs(_chunks_from([b"data: a\n\n"])))
        assert blobs == [b"data: a\n\n"]

    @pytest.mark.asyncio
    async def test_two_events_split_on_blank_line_boundary(self):
        # Two events concatenated in one chunk → split exactly at each \n\n; the
        # boundary offset must include the terminator (kills a shifted cut point).
        chunk = b"event: a\ndata: 1\n\nevent: b\ndata: 2\n\n"
        blobs = await _collect(iter_sse_event_blobs(_chunks_from([chunk])))
        assert blobs == [b"event: a\ndata: 1\n\n", b"event: b\ndata: 2\n\n"]

    @pytest.mark.asyncio
    async def test_event_split_across_chunks_is_buffered(self):
        # A delimiter straddling two chunks must still frame as one event.
        blobs = await _collect(
            iter_sse_event_blobs(_chunks_from([b"data: a\n", b"\ndata: b\n\n"]))
        )
        assert blobs == [b"data: a\n\n", b"data: b\n\n"]

    @pytest.mark.asyncio
    async def test_unterminated_trailing_remainder_is_flushed(self):
        # A final fragment with no \n\n is yielded once the stream ends (default
        # max_buffer=None must not raise on the post-loop bound check).
        blobs = await _collect(iter_sse_event_blobs(_chunks_from([b"data: a\n\ndata: tail"])))
        assert blobs == [b"data: a\n\n", b"data: tail"]

    @pytest.mark.asyncio
    async def test_crlf_normalized_to_lf(self):
        # CRLF line endings collapse to LF before framing.
        blobs = await _collect(iter_sse_event_blobs(_chunks_from([b"data: a\r\n\r\n"])))
        assert blobs == [b"data: a\n\n"]

    @pytest.mark.asyncio
    async def test_max_buffer_not_exceeded_at_exact_cap_does_not_raise(self):
        # Exactly cap bytes, unterminated: the bound is EXCLUSIVE, so no raise.
        # The trailing remainder is flushed at stream end. Kills > vs >=.
        blobs = await _collect(iter_sse_event_blobs(_chunks_from([b"x" * 8]), max_buffer=8))
        assert blobs == [b"x" * 8]

    @pytest.mark.asyncio
    async def test_max_buffer_exceeded_raises_runtime_error(self):
        # Over-cap and unterminated → abort the malformed stream. Kills > vs <.
        with pytest.raises(RuntimeError, match="malformed"):
            await _collect(iter_sse_event_blobs(_chunks_from([b"x" * 9]), max_buffer=8))


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
        completed = _make_sse_event(
            "response.completed",
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_1",
                    "model": "gpt-5.5",
                    "status": "completed",
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                },
            },
        )

        chunks = [created_event + text_delta + completed]
        events = []
        async for event in provider.translate_stream(_chunks_from(chunks)):
            events.append(event)

        # response.created → message_start + ping, text.delta → content_block_delta,
        # response.completed → message_delta + message_stop (the terminator).
        assert events[0]["event"] == "message_start"
        assert events[1]["event"] == "ping"
        assert events[2]["event"] == "content_block_delta"
        assert events[2]["data"]["delta"]["text"] == "Hello"
        assert events[-1]["event"] == "message_stop"

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
        incomplete = _make_sse_event(
            "response.incomplete",
            {
                "type": "response.incomplete",
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
            _chunks_from([part_added + text_delta + text_done + incomplete])
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

    @pytest.mark.asyncio
    async def test_response_incomplete_stream_terminates_with_message_stop(self):
        """A streamed turn that ends with a real response.incomplete event still
        closes with message_stop — Claude Code must finalize, not hang."""
        from claude_bridge.providers.openai import OpenAIProvider

        provider = OpenAIProvider()

        created = _make_sse_event(
            "response.created",
            {
                "type": "response.created",
                "response": {
                    "id": "resp_inc_stream",
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
                "delta": "partial answer",
            },
        )
        incomplete = _make_sse_event(
            "response.incomplete",
            {
                "type": "response.incomplete",
                "response": {
                    "id": "resp_inc_stream",
                    "model": "gpt-5.5",
                    "status": "incomplete",
                    "incomplete_details": {"reason": "max_output_tokens"},
                    "usage": {"input_tokens": 10, "output_tokens": 4096},
                },
            },
        )

        events = []
        async for event in provider.translate_stream(
            _chunks_from([created + text_delta + incomplete])
        ):
            events.append(event)

        assert events[-1]["event"] == "message_stop"
        message_delta = next(e for e in events if e["event"] == "message_delta")
        assert message_delta["data"]["delta"]["stop_reason"] == "max_tokens"
        # The terminator arrived from the real terminal event, so the invariant must
        # NOT also synthesize one — exactly one message_stop / message_delta total.
        assert sum(e["event"] == "message_stop" for e in events) == 1
        assert sum(e["event"] == "message_delta" for e in events) == 1

    @pytest.mark.asyncio
    async def test_dropped_stream_with_tool_call_synthesizes_tool_use_stop(self):
        """A dropped stream that already emitted a tool call must synthesize
        stop_reason=tool_use — Claude Code has to RUN the tool, not treat the turn
        as a clean end_turn."""
        from claude_bridge.providers.openai import OpenAIProvider

        provider = OpenAIProvider()

        created = _make_sse_event(
            "response.created",
            {
                "type": "response.created",
                "response": {
                    "id": "resp_tool_drop",
                    "model": "gpt-5.5",
                    "status": "in_progress",
                    "usage": {"input_tokens": 10, "output_tokens": 0},
                },
            },
        )
        tool_call = _make_sse_event(
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "call_id": "call_abc",
                    "name": "read_file",
                },
            },
        )

        # Stream ends after the tool-call start — no terminal event.
        events = []
        async for event in provider.translate_stream(_chunks_from([created + tool_call])):
            events.append(event)

        assert events[-1]["event"] == "message_stop"
        message_delta = next(e for e in events if e["event"] == "message_delta")
        assert message_delta["data"]["delta"]["stop_reason"] == "tool_use"

    @pytest.mark.asyncio
    async def test_started_stream_without_terminator_synthesizes_message_stop(self):
        """A stream that emits message_start but no terminal event (connection
        drop mid-turn) is closed by a synthesized message_stop — the termination
        invariant. Claude Code's parser requires message_stop as the final event
        (see tests/test_contract.py)."""
        from claude_bridge.providers.openai import OpenAIProvider

        provider = OpenAIProvider()

        created = _make_sse_event(
            "response.created",
            {
                "type": "response.created",
                "response": {
                    "id": "resp_drop",
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
                "delta": "cut off here",
            },
        )

        # Stream ends after the delta — no response.completed/incomplete/failed.
        events = []
        async for event in provider.translate_stream(_chunks_from([created + text_delta])):
            events.append(event)

        assert events[0]["event"] == "message_start"
        assert events[-1]["event"] == "message_stop"

    @pytest.mark.asyncio
    async def test_response_failed_stream_emits_error_event(self):
        """A streamed turn that ends with response.failed emits an error event so
        the stream does not end silently."""
        from claude_bridge.providers.openai import OpenAIProvider

        provider = OpenAIProvider()

        created = _make_sse_event(
            "response.created",
            {
                "type": "response.created",
                "response": {
                    "id": "resp_fail_stream",
                    "model": "gpt-5.5",
                    "status": "in_progress",
                    "usage": {"input_tokens": 10, "output_tokens": 0},
                },
            },
        )
        failed = _make_sse_event(
            "response.failed",
            {
                "type": "response.failed",
                "response": {
                    "id": "resp_fail_stream",
                    "status": "failed",
                    "error": {"code": "server_error", "message": "boom"},
                },
            },
        )

        events = []
        async for event in provider.translate_stream(_chunks_from([created + failed])):
            events.append(event)

        assert any(e["event"] == "error" for e in events)


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
