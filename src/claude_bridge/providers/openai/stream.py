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
    _reasoning_surfaced,
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


# The three reasoning-summary events that carry a summary part's lifecycle.
# ``response.reasoning_summary_text.done`` is deliberately absent: its ``text`` field
# repeats what the deltas already delivered, so translating it would duplicate the
# whole summary inside the block. It is skipped explicitly below.
_REASONING_SUMMARY_EVENTS = frozenset(
    {
        "response.reasoning_summary_part.added",
        "response.reasoning_summary_text.delta",
        "response.reasoning_summary_part.done",
    }
)


def _sse_reasoning_summary(event_type: str, data: dict) -> list[dict]:
    """Translate one reasoning-summary event → one Anthropic thinking-block event.

    A reasoning item emits ``part.added`` → repeated ``text.delta`` → ``part.done`` per
    summary part, which maps exactly onto Anthropic's ``content_block_start`` →
    ``thinking_delta`` → ``content_block_stop``. Without this the entire reasoning
    phase is dropped and the client sees nothing until the first answer token.

    Blocks are keyed on ``output_index`` — the space ``function_call`` items already
    use, distinct from the ``content_index`` text blocks use — which
    ``_remap_block_index`` renumbers into sequential Anthropic block indices.

    Emits one block PER PART; ``ThinkingCoalescer`` downstream merges a turn's parts
    into the single thinking block the Anthropic shape specifies. Splitting the two
    keeps this function pure per event.

    The ``signature`` field is present but empty: the published wire format carries one
    on every thinking block, and the bridge cannot mint a verifiable signature for
    reasoning it did not produce. Declaring it empty states "unsigned" explicitly rather
    than omitting the field and leaving each client to invent a default (D-THINK-004).

    ``REASONING_MODE=drop`` suppresses the whole reasoning phase here — the one direction
    the knob still governs, now that a returned block is omitted under every mode
    (D-THINK-003).
    """
    if not _reasoning_surfaced():
        return []
    index = data.get("output_index", 0)
    if event_type == "response.reasoning_summary_part.added":
        return [
            {
                "event": "content_block_start",
                "data": {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {"type": "thinking", "thinking": "", "signature": ""},
                },
            }
        ]
    if event_type == "response.reasoning_summary_text.delta":
        return [
            {
                "event": "content_block_delta",
                "data": {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "thinking_delta", "thinking": data.get("delta", "")},
                },
            }
        ]
    return [
        {"event": "content_block_stop", "data": {"type": "content_block_stop", "index": index}}
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
        # Carries the assembled summary text the deltas already streamed.
        "response.reasoning_summary_text.done",
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

    if event_type in _REASONING_SUMMARY_EVENTS:
        return _sse_reasoning_summary(event_type, data)

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


# Separator joining two summary parts merged into one thinking block. Each part is a
# short bold heading, so a blank line keeps them readable as distinct steps.
_THINKING_PART_SEPARATOR = "\n\n"

# Events that end a turn's reasoning phase: once one is reached, a thinking block held
# open for coalescing must be closed before the event is forwarded.
_THINKING_CLOSING_EVENTS = frozenset({"message_delta", "message_stop", "error"})


class ThinkingCoalescer:
    """Merge a turn's reasoning-summary parts into ONE Anthropic thinking block.

    A turn commonly emits several reasoning ITEMS, each with several summary PARTS —
    five parts across three items in a captured live turn. Translating each part into
    its own thinking block put N blocks in one assistant turn where the published
    Anthropic shape carries one, and Claude Code echoes every block back on every
    subsequent request, so N compounded turn over turn (measured 0 → 14 over six turns
    of a trivial task).

    Holds the first part's block open, rewrites later parts' deltas onto it, and defers
    the close until the reasoning phase ends — the first non-thinking block or a
    terminal event. State lives here rather than in ``translate_openai_sse_event`` so
    that translation stays a pure function of one event, matching ``_remap_block_index``.

    Runs BEFORE ``_remap_block_index``, so every index it reads or writes is still a raw
    provider index. That ordering is what keeps the merge invisible downstream: remap
    sees one thinking ``content_block_start`` and therefore allocates one Anthropic
    index, and the absorbed parts' deltas resolve onto it through the existing map.

    Alternatives weighed in D-THINK-005.
    """

    def __init__(self) -> None:
        self._open_index: int | None = None
        self._absorbed: set[int] = set()

    def feed(self, event: dict) -> list[dict]:
        """Return the events to forward for ``event`` (0, 1, or 2 of them)."""
        name = event.get("event")
        data = event.get("data", {})
        index = data.get("index", 0)
        is_thinking_start = (
            name == "content_block_start"
            and data.get("content_block", {}).get("type") == "thinking"
        )

        if is_thinking_start:
            self._absorbed.add(index)
            if self._open_index is None:
                self._open_index = index
                return [event]
            # A later part of the same turn: keep the open block, insert a separator.
            return [self._separator_delta()]

        if self._open_index is not None:
            # Parts of a LATER reasoning item carry a different raw index, so membership
            # of the absorbed set — not equality with the open one — is the test.
            if index in self._absorbed and name in ("content_block_delta", "content_block_stop"):
                if name == "content_block_stop":
                    return []  # defer: more parts may still arrive
                data["index"] = self._open_index
                return [event]
            if name == "content_block_start" or name in _THINKING_CLOSING_EVENTS:
                return [self._close(), event]

        return [event]

    def _separator_delta(self) -> dict:
        """A thinking_delta carrying the blank line between two merged parts."""
        return {
            "event": "content_block_delta",
            "data": {
                "type": "content_block_delta",
                "index": self._open_index,
                "delta": {"type": "thinking_delta", "thinking": _THINKING_PART_SEPARATOR},
            },
        }

    def _close(self) -> dict:
        """Close the held-open thinking block and forget it."""
        index, self._open_index = self._open_index, None
        self._absorbed.clear()
        return {
            "event": "content_block_stop",
            "data": {"type": "content_block_stop", "index": index},
        }

    def flush(self) -> list[dict]:
        """Close a still-open thinking block at stream end (upstream dropped mid-turn)."""
        return [self._close()] if self._open_index is not None else []


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
