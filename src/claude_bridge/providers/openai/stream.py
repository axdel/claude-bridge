"""OpenAI Responses API SSE event -> Anthropic SSE event translation.

Pure functions, no I/O. Derives token scaling, id conversion, stop-reason, and
content-filter handling from the translate submodule.
"""

from __future__ import annotations

from claude_bridge.providers.openai.translate import (
    _CONTENT_FILTER_REASON,
    _CONTENT_FILTER_REFUSAL,
    GPT_TOKEN_COUNT_MULTIPLIER,
    _anthropic_usage,
    _incomplete_reason,
    _scale_token_count,
    _stop_reason,
    _to_anthropic_id,
)


def _sse_response_created(data: dict, *, token_count_multiplier: float) -> list[dict]:
    """Translate response.created → message_start + ping."""
    resp = data.get("response", {})
    usage = resp.get("usage") or {}
    return [
        {
            "event": "message_start",
            "data": {
                "type": "message_start",
                "message": {
                    "id": f"msg_bridge_{resp.get('id', 'unknown')}",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": resp.get("model", ""),
                    "stop_reason": None,
                    "usage": {
                        "input_tokens": _scale_token_count(
                            usage.get("input_tokens", 0), token_count_multiplier
                        ),
                        "output_tokens": 0,
                    },
                },
            },
        },
        {"event": "ping", "data": {"type": "ping"}},
    ]


def _sse_output_item_added(data: dict) -> list[dict]:
    """Translate response.output_item.added → content_block_start for function_call items."""
    item = data.get("item", {})
    output_index = data.get("output_index", 0)
    if item.get("type") != "function_call":
        return []
    oai_id = item.get("call_id") or item.get("id", "")
    anthropic_id = _to_anthropic_id(oai_id) if oai_id else f"call_bridge_{output_index}"
    return [
        {
            "event": "content_block_start",
            "data": {
                "type": "content_block_start",
                "index": output_index,
                "content_block": {
                    "type": "tool_use",
                    "id": anthropic_id,
                    "name": item.get("name", ""),
                    "input": {},
                },
            },
        }
    ]


def _synthesize_refusal_block(text: str) -> list[dict]:
    """Build start/delta/stop SSE events for a synthetic refusal text block.

    Emitted when a streamed turn is content-filtered with no model text, so the stream
    does not end on an empty assistant message. The placeholder index 0 is reassigned to
    the next sequential Anthropic block index by ``_remap_block_index``.
    """
    return [
        {
            "event": "content_block_start",
            "data": {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        },
        {
            "event": "content_block_delta",
            "data": {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": text},
            },
        },
        {"event": "content_block_stop", "data": {"type": "content_block_stop", "index": 0}},
    ]


def _sse_terminal_response(data: dict, *, token_count_multiplier: float) -> list[dict]:
    """Translate a terminal Responses event (``response.completed`` /
    ``response.incomplete``) → [refusal block?] + message_delta + message_stop.

    Both terminal event types carry a ``response`` object whose ``status`` and
    ``incomplete_details`` drive the stop_reason: ``completed`` → ``end_turn``
    (or ``tool_use`` when tool calls were emitted); ``incomplete`` →
    ``max_tokens`` unless the reason is ``content_filter``, which ends the turn
    cleanly (``end_turn``) and is prefixed with a synthesized refusal text block.
    """
    resp = data.get("response", {})
    status = resp.get("status", "completed")
    output = resp.get("output", [])
    has_tool_calls = any(i.get("type") == "function_call" for i in output)
    incomplete_reason = _incomplete_reason(resp)
    stop_reason = _stop_reason(status, has_tool_calls, incomplete_reason)

    events: list[dict] = []
    if incomplete_reason == _CONTENT_FILTER_REASON:
        events.extend(_synthesize_refusal_block(_CONTENT_FILTER_REFUSAL))
    events.append(
        {
            "event": "message_delta",
            "data": {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason},
                "usage": _anthropic_usage(
                    resp.get("usage"), token_count_multiplier=token_count_multiplier
                ),
            },
        }
    )
    events.append({"event": "message_stop", "data": {"type": "message_stop"}})
    return events


# Upper bound on a provider-controlled error message surfaced in an Anthropic error
# event. json.dumps escapes control characters on the wire, so this only guards
# against a hostile/huge message bloating the stream — not log injection.
_ERROR_MESSAGE_MAX = 500


def _sse_error_event(message: str) -> list[dict]:
    """Build an Anthropic ``error`` SSE event that terminates the stream.

    A failed or errored upstream response is an API error, not assistant output;
    the Anthropic streaming protocol ends a stream with an ``error`` event rather
    than a ``message_stop``. The provider-controlled message is length-bounded.
    """
    text = (message or "Provider stream error").strip()[:_ERROR_MESSAGE_MAX]
    return [
        {
            "event": "error",
            "data": {"type": "error", "error": {"type": "api_error", "message": text}},
        }
    ]


def _sse_response_failed(data: dict) -> list[dict]:
    """Translate a ``response.failed`` event to an Anthropic error event.

    The ``response.error`` object carries a ``code`` (``server_error``,
    ``rate_limit_exceeded``, ...) and a human-readable ``message``.
    """
    resp = data.get("response")
    resp = resp if isinstance(resp, dict) else {}
    error = resp.get("error")
    error = error if isinstance(error, dict) else {}
    message = error.get("message") or error.get("code") or "Provider response failed"
    return _sse_error_event(message)


def _sse_top_level_error(data: dict) -> list[dict]:
    """Translate a bare top-level Responses ``error`` stream event (emitted on a
    mid-stream server failure) to an Anthropic error event."""
    error = data.get("error")
    error = error if isinstance(error, dict) else {}
    message = data.get("message") or error.get("message") or data.get("code")
    return _sse_error_event(message or "Provider stream error")


def _sse_synthetic_termination(has_tool_calls: bool) -> list[dict]:
    """Build the message_delta + message_stop for a stream that emitted a
    message_start but never received a terminal event (e.g. a dropped upstream
    connection).

    stop_reason is ``tool_use`` when tool calls were already emitted (Claude Code
    must run them), else ``end_turn`` — a clean stop that does NOT masquerade as
    token exhaustion and trigger an auto-compact retry loop. Usage is reported as
    zero output tokens since the true terminal usage never arrived.
    """
    stop_reason = "tool_use" if has_tool_calls else "end_turn"
    return [
        {
            "event": "message_delta",
            "data": {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason},
                "usage": {"output_tokens": 0},
            },
        },
        {"event": "message_stop", "data": {"type": "message_stop"}},
    ]


# Events that are informational — no Anthropic equivalent.
_SKIPPED_SSE_EVENTS = frozenset(
    {
        "response.in_progress",
        "response.queued",
        "response.content_part.done",
        "response.output_item.done",
    }
)


def translate_openai_sse_event(
    event: dict,
    *,
    token_count_multiplier: float = GPT_TOKEN_COUNT_MULTIPLIER,
) -> list[dict]:
    """Translate one OpenAI Responses API SSE event to Anthropic SSE events.

    Dispatches to sub-handlers by event type. Returns a list of ``{event, data}``
    dicts (may be 0, 1, or 2 items). Pure function — no I/O.
    """
    event_type = event.get("event", "")
    data = event.get("data", {})

    if event_type == "response.created":
        return _sse_response_created(data, token_count_multiplier=token_count_multiplier)

    if event_type == "response.content_part.added":
        return [
            {
                "event": "content_block_start",
                "data": {
                    "type": "content_block_start",
                    "index": data.get("content_index", 0),
                    "content_block": {"type": "text", "text": ""},
                },
            }
        ]

    if event_type == "response.output_text.delta":
        return [
            {
                "event": "content_block_delta",
                "data": {
                    "type": "content_block_delta",
                    "index": data.get("content_index", 0),
                    "delta": {"type": "text_delta", "text": data.get("delta", "")},
                },
            }
        ]

    if event_type in (
        "response.output_text.done",
        "response.function_call_arguments.done",
    ):
        return [
            {
                "event": "content_block_stop",
                "data": {
                    "type": "content_block_stop",
                    "index": data.get("content_index", data.get("output_index", 0)),
                },
            }
        ]

    if event_type == "response.output_item.added":
        return _sse_output_item_added(data)

    if event_type == "response.function_call_arguments.delta":
        return [
            {
                "event": "content_block_delta",
                "data": {
                    "type": "content_block_delta",
                    "index": data.get("output_index", 0),
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": data.get("delta", ""),
                    },
                },
            }
        ]

    if event_type in ("response.completed", "response.incomplete"):
        return _sse_terminal_response(data, token_count_multiplier=token_count_multiplier)

    if event_type == "response.failed":
        return _sse_response_failed(data)

    if event_type == "error":
        return _sse_top_level_error(data)

    if event_type in _SKIPPED_SSE_EVENTS:
        return []

    return []


def _remap_block_index(
    event: dict,
    index_map: dict[int, int],
    next_index: int,
    has_tool_calls: bool,
) -> tuple[dict, int, bool]:
    """Remap OpenAI output_index to sequential Anthropic block indices.

    Returns (possibly-modified event, updated next_index, updated has_tool_calls).
    """
    data = event.get("data", {})

    if event.get("event") == "content_block_start":
        oai_index = data.get("index", 0)
        index_map[oai_index] = next_index
        data["index"] = next_index
        if data.get("content_block", {}).get("type") == "tool_use":
            has_tool_calls = True
        return event, next_index + 1, has_tool_calls

    if event.get("event") in ("content_block_delta", "content_block_stop"):
        oai_index = data.get("index", 0)
        data["index"] = index_map.get(oai_index, oai_index)
        return event, next_index, has_tool_calls

    if event.get("event") == "message_delta" and has_tool_calls:
        delta = data.get("delta", {})
        if delta.get("stop_reason") == "end_turn":
            delta["stop_reason"] = "tool_use"
        return event, next_index, has_tool_calls

    return event, next_index, has_tool_calls
