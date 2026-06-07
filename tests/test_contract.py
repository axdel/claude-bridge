"""Claude Code wire-contract fixtures — the compatibility oracle for the bridge.

These tests pin the Anthropic <-> OpenAI Responses contract that Claude Code
depends on: rules/skills system context, tool_use/tool_result ordering, streaming
block indices, usage/stop shape, and count_tokens estimation. Expected values are
derived from the Anthropic Messages and OpenAI Responses specifications and from
structural invariants (round-trip identity, index monotonicity, metamorphic
monotonicity) — never from running the translation under test.

Tests marked ``xfail(strict=True)`` encode behavior a later task implements (named
in the marker reason). Strict mode turns the eventual xpass into a failure, forcing
the marker's removal when the behavior lands — so the contract file doubles as the
TDD red for those tasks.
"""

from __future__ import annotations

import json

import pytest

from claude_bridge.providers.openai import (
    OpenAIProvider,
    _to_anthropic_id,
    _to_openai_id,
    anthropic_to_openai,
    openai_to_anthropic,
)
from claude_bridge.proxy import estimate_input_tokens

# ---------------------------------------------------------------------------
# Fixtures — realistic Claude Code traffic shapes
# ---------------------------------------------------------------------------

# A full Claude Code tool loop: user asks, assistant narrates + calls a tool,
# user returns the tool result, assistant concludes. This is the canonical
# serialized tool turn the bridge must preserve order-for-order.
CLAUDE_CODE_TOOL_LOOP: dict = {
    "model": "claude-opus-4-6",
    "system": [{"type": "text", "text": "You are Claude Code."}],
    "tools": [
        {
            "name": "Read",
            "description": "Read a file",
            "input_schema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        }
    ],
    "messages": [
        {"role": "user", "content": [{"type": "text", "text": "Read /etc/hosts"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Reading it now."},
                {
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "Read",
                    "input": {"path": "/etc/hosts"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_01",
                    "content": "127.0.0.1 localhost",
                }
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
    ],
}


def _sse(event_type: str, payload: dict) -> bytes:
    """Build one OpenAI Responses SSE wire event (``event:``/``data:`` block)."""
    return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n".encode()


# A text-then-tool-call OpenAI Responses stream. Text lives at content_index 0
# inside output_index 0; the function call is a distinct output_index 1 — exactly
# how the Responses API interleaves a narrated tool call.
OPENAI_TEXT_THEN_TOOL_STREAM: bytes = (
    _sse(
        "response.created",
        {
            "type": "response.created",
            "response": {
                "id": "resp_1",
                "model": "gpt-5.5",
                "usage": {"input_tokens": 50, "output_tokens": 0},
            },
        },
    )
    + _sse(
        "response.content_part.added", {"type": "response.content_part.added", "content_index": 0}
    )
    + _sse(
        "response.output_text.delta",
        {"type": "response.output_text.delta", "content_index": 0, "delta": "Let me "},
    )
    + _sse(
        "response.output_text.delta",
        {"type": "response.output_text.delta", "content_index": 0, "delta": "read it."},
    )
    + _sse("response.output_text.done", {"type": "response.output_text.done", "content_index": 0})
    + _sse(
        "response.output_item.added",
        {
            "type": "response.output_item.added",
            "output_index": 1,
            "item": {"type": "function_call", "id": "fc_abc", "call_id": "fc_abc", "name": "Read"},
        },
    )
    + _sse(
        "response.function_call_arguments.delta",
        {"type": "response.function_call_arguments.delta", "output_index": 1, "delta": '{"path"'},
    )
    + _sse(
        "response.function_call_arguments.delta",
        {"type": "response.function_call_arguments.delta", "output_index": 1, "delta": ': "/x"}'},
    )
    + _sse(
        "response.function_call_arguments.done",
        {"type": "response.function_call_arguments.done", "output_index": 1},
    )
    + _sse(
        "response.completed",
        {
            "type": "response.completed",
            "response": {
                "id": "resp_1",
                "model": "gpt-5.5",
                "status": "completed",
                "output": [{"type": "function_call", "name": "Read"}],
                "usage": {"input_tokens": 50, "output_tokens": 12},
            },
        },
    )
)


async def _collect_stream(
    provider: OpenAIProvider, raw: bytes, *, split_at: int | None = None
) -> list[dict]:
    """Drive ``translate_stream`` over ``raw`` and return the translated events.

    ``split_at`` cuts the byte stream into two chunks at that offset to exercise
    cross-chunk SSE buffering without changing the expected event sequence.
    """

    async def _chunks():
        if split_at is None:
            yield raw
        else:
            yield raw[:split_at]
            yield raw[split_at:]

    return [event async for event in provider.translate_stream(_chunks())]


def _kind(item: dict) -> str:
    """Classify an OpenAI Responses input item by its structural role."""
    if "type" in item:
        return item["type"]
    return f"message:{item.get('role', '?')}"


# ---------------------------------------------------------------------------
# System / rules / skills context
# ---------------------------------------------------------------------------


class TestSystemContextContract:
    """Claude Code injects rules/skills as ordinary system text — they must reach
    the provider as ``instructions`` verbatim, in order."""

    def test_string_system_becomes_instructions(self):
        result, _ = anthropic_to_openai(
            {"model": "claude-opus-4-6", "system": "Be terse.", "messages": []}
        )
        assert result["instructions"] == "Be terse."

    def test_multiblock_system_joins_rules_and_skills_in_order(self):
        # Oracle: list system blocks join on newline in declared order (openai.py
        # contract); a reordering or dropped-block bug changes this exact string.
        request = {
            "model": "claude-opus-4-6",
            "system": [
                {"type": "text", "text": "Rule: cite or flag."},
                {"type": "text", "text": "Skill: bugfix."},
            ],
            "messages": [],
        }
        result, _ = anthropic_to_openai(request)
        assert result["instructions"] == "Rule: cite or flag.\nSkill: bugfix."

    def test_absent_system_gets_default_instructions(self):
        result, _ = anthropic_to_openai({"model": "claude-opus-4-6", "messages": []})
        assert result["instructions"] == "You are a helpful assistant."


# ---------------------------------------------------------------------------
# tool_use / tool_result translation
# ---------------------------------------------------------------------------


class TestToolUseContract:
    """tool_use → OpenAI ``function_call``. Spec: Responses function_call carries
    BOTH ``id`` and ``call_id`` and a JSON-string ``arguments``."""

    def test_tool_use_emits_function_call_with_both_ids(self):
        request = {
            "model": "claude-opus-4-6",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_9",
                            "name": "Read",
                            "input": {"path": "/x"},
                        }
                    ],
                }
            ],
        }
        result, _ = anthropic_to_openai(request)
        calls = [i for i in result["input"] if i.get("type") == "function_call"]
        assert len(calls) == 1
        call = calls[0]
        assert call["id"] == "fc_9"
        assert call["call_id"] == "fc_9"
        assert call["name"] == "Read"
        # Arguments is a JSON string that round-trips to the original input dict.
        assert json.loads(call["arguments"]) == {"path": "/x"}

    def test_tool_result_emits_string_function_call_output(self):
        request = {
            "model": "claude-opus-4-6",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_9", "content": "hello"}
                    ],
                }
            ],
        }
        result, _ = anthropic_to_openai(request)
        outputs = [i for i in result["input"] if i.get("type") == "function_call_output"]
        assert len(outputs) == 1
        assert outputs[0]["call_id"] == "fc_9"
        # Spec: function_call_output.output is always a string, never null.
        assert outputs[0]["output"] == "hello"
        assert isinstance(outputs[0]["output"], str)

    def test_tool_result_error_is_marked_in_output(self):
        request = {
            "model": "claude-opus-4-6",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_9",
                            "content": "boom",
                            "is_error": True,
                        }
                    ],
                }
            ],
        }
        result, _ = anthropic_to_openai(request)
        output = next(i for i in result["input"] if i.get("type") == "function_call_output")
        # An error result must be distinguishable from a success result carrying
        # the same text — otherwise the model cannot tell the tool failed.
        assert output["output"] == "[Error] boom"


class TestToolIdRoundTrip:
    """Claude Code's ``toolu_`` IDs must survive a round trip through the provider's
    ``fc_`` namespace so tool_result can be matched to its tool_use."""

    def test_toolu_id_round_trips_through_openai_namespace(self):
        # Round-trip identity oracle: the suffix is preserved both directions.
        assert _to_anthropic_id(_to_openai_id("toolu_deadbeef")) == "toolu_deadbeef"

    def test_openai_fc_id_maps_to_toolu(self):
        # Spec: Claude Code requires the toolu_ prefix on tool_use IDs.
        assert _to_anthropic_id("fc_xyz") == "toolu_xyz"

    def test_anthropic_toolu_maps_to_fc(self):
        assert _to_openai_id("toolu_xyz") == "fc_xyz"


class TestMultiTurnToolHistory:
    """A serialized Claude Code tool loop must translate with order preserved and
    the tool call linked to its result by call_id."""

    def test_tool_loop_preserves_order_and_call_linkage(self):
        result, _ = anthropic_to_openai(CLAUDE_CODE_TOOL_LOOP)
        kinds = [_kind(i) for i in result["input"]]
        # Oracle: the Responses input must carry the user turn, the assistant's
        # narration, the function_call, its output, then the closing assistant turn
        # — in that exact order.
        assert kinds == [
            "message:user",
            "message:assistant",
            "function_call",
            "function_call_output",
            "message:assistant",
        ]
        call = next(i for i in result["input"] if i.get("type") == "function_call")
        output = next(i for i in result["input"] if i.get("type") == "function_call_output")
        assert call["call_id"] == output["call_id"] == "fc_01"
        assert result["instructions"] == "You are Claude Code."

    def test_assistant_text_becomes_output_text(self):
        result, _ = anthropic_to_openai(CLAUDE_CODE_TOOL_LOOP)
        assistant_msgs = [i for i in result["input"] if i.get("role") == "assistant"]
        # Spec: assistant text is output_text (not input_text) in the Responses API.
        for msg in assistant_msgs:
            for block in msg["content"]:
                assert block["type"] == "output_text"


# ---------------------------------------------------------------------------
# Request-level invariants Claude Code relies on
# ---------------------------------------------------------------------------


class TestRequestInvariants:
    """Stable provider-request shape regardless of content."""

    def test_stateless_streaming_invariants(self):
        result, _ = anthropic_to_openai(CLAUDE_CODE_TOOL_LOOP)
        # Privacy posture (store=False) and streaming are contract invariants.
        assert result["store"] is False
        assert result["stream"] is True
        assert result["model"] == "gpt-5.5"

    def test_tools_use_flat_function_shape_with_strict_false(self):
        result, _ = anthropic_to_openai(CLAUDE_CODE_TOOL_LOOP)
        tool = result["tools"][0]
        # Flat Responses function shape — name at top level, schema under parameters.
        assert tool["type"] == "function"
        assert tool["name"] == "Read"
        assert tool["parameters"] == CLAUDE_CODE_TOOL_LOOP["tools"][0]["input_schema"]
        # strict:false because Claude Code marks all params required but omits
        # values for params it does not need.
        assert tool["strict"] is False


# ---------------------------------------------------------------------------
# tool_choice + parallel controls — NOT YET IMPLEMENTED (T-002)
# ---------------------------------------------------------------------------


class TestToolChoiceContract:
    """Anthropic ``tool_choice`` policy must reach the OpenAI Responses request.

    Oracle (verified against OpenAI docs): Responses ``tool_choice`` is
    ``"auto"``/``"none"``/``"required"`` or the forced-tool object
    ``{"type": "function", "name": <name>}`` — name at the top level, matching the
    flat tool definition this bridge already emits.
    """

    @staticmethod
    def _request(tool_choice: dict) -> dict:
        return {
            "model": "claude-opus-4-6",
            "tools": [{"name": "Read", "description": "", "input_schema": {}}],
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": tool_choice,
        }

    @pytest.mark.xfail(strict=True, reason="T-002 implements tool_choice mapping")
    def test_auto_maps_to_auto(self):
        result, _ = anthropic_to_openai(self._request({"type": "auto"}))
        assert result["tool_choice"] == "auto"

    @pytest.mark.xfail(strict=True, reason="T-002 implements tool_choice mapping")
    def test_none_maps_to_none(self):
        result, _ = anthropic_to_openai(self._request({"type": "none"}))
        assert result["tool_choice"] == "none"

    @pytest.mark.xfail(strict=True, reason="T-002 implements tool_choice mapping")
    def test_any_maps_to_required(self):
        result, _ = anthropic_to_openai(self._request({"type": "any"}))
        assert result["tool_choice"] == "required"

    @pytest.mark.xfail(strict=True, reason="T-002 implements tool_choice mapping")
    def test_named_tool_maps_to_forced_function(self):
        result, _ = anthropic_to_openai(self._request({"type": "tool", "name": "Read"}))
        assert result["tool_choice"] == {"type": "function", "name": "Read"}


class TestParallelToolContract:
    """Anthropic ``disable_parallel_tool_use`` must serialize tool calls on the
    provider side, or Claude Code's one-at-a-time tool loop breaks."""

    @pytest.mark.xfail(strict=True, reason="T-002 implements parallel_tool_calls mapping")
    def test_disable_parallel_maps_to_parallel_tool_calls_false(self):
        request = {
            "model": "claude-opus-4-6",
            "tools": [{"name": "Read", "description": "", "input_schema": {}}],
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
        }
        result, _ = anthropic_to_openai(request)
        assert result["parallel_tool_calls"] is False


# ---------------------------------------------------------------------------
# Response stop_reason + usage shape (auto-compact relevant)
# ---------------------------------------------------------------------------


class TestResponseStopReason:
    """stop_reason mapping per the Anthropic spec — Claude Code uses ``max_tokens``
    to drive auto-compaction and ``tool_use`` to continue the loop."""

    def test_tool_call_output_maps_to_tool_use(self):
        response = {
            "status": "completed",
            "output": [{"type": "function_call", "name": "R", "call_id": "fc_1"}],
        }
        assert openai_to_anthropic(response)["stop_reason"] == "tool_use"

    def test_incomplete_maps_to_max_tokens(self):
        # Spec: Anthropic reports max_tokens when generation is truncated.
        response = {"status": "incomplete", "output": []}
        assert openai_to_anthropic(response)["stop_reason"] == "max_tokens"

    def test_plain_completion_maps_to_end_turn(self):
        response = {"status": "completed", "output": [{"type": "message", "content": []}]}
        assert openai_to_anthropic(response)["stop_reason"] == "end_turn"


class TestUsageShape:
    """Usage must be integers — Claude Code's /context math divides by them."""

    def test_usage_passes_through_as_integers(self):
        response = {
            "status": "completed",
            "output": [],
            "usage": {"input_tokens": 123, "output_tokens": 45},
        }
        usage = openai_to_anthropic(response)["usage"]
        assert usage == {"input_tokens": 123, "output_tokens": 45}
        assert isinstance(usage["input_tokens"], int)
        assert isinstance(usage["output_tokens"], int)

    def test_missing_usage_defaults_to_zero_integers(self):
        usage = openai_to_anthropic({"status": "completed", "output": []})["usage"]
        assert usage == {"input_tokens": 0, "output_tokens": 0}


# ---------------------------------------------------------------------------
# Streaming block-index monotonicity + finality
# ---------------------------------------------------------------------------


class TestStreamingBlockIndex:
    """Anthropic content-block indices must be sequential from 0 and balanced, and
    message_stop must be the final event — Claude Code's stream parser assumes it."""

    async def test_block_indices_are_sequential_and_balanced(self):
        provider = OpenAIProvider(auth_mode="api_key", api_key="test-key-placeholder")
        events = await _collect_stream(provider, OPENAI_TEXT_THEN_TOOL_STREAM)

        starts = [e["data"]["index"] for e in events if e["event"] == "content_block_start"]
        # Two blocks (text, then tool_use) → indices exactly [0, 1].
        assert starts == [0, 1]

        started = set(starts)
        for e in events:
            if e["event"] in ("content_block_delta", "content_block_stop"):
                assert e["data"]["index"] in started
        # Every started block is closed exactly once.
        stops = [e["data"]["index"] for e in events if e["event"] == "content_block_stop"]
        assert sorted(stops) == [0, 1]

    async def test_message_start_and_stop_bracket_the_stream(self):
        provider = OpenAIProvider(auth_mode="api_key", api_key="test-key-placeholder")
        events = await _collect_stream(provider, OPENAI_TEXT_THEN_TOOL_STREAM)
        names = [e["event"] for e in events]
        assert names[0] == "message_start"
        assert names[-1] == "message_stop"
        assert names.count("message_start") == 1
        assert names.count("message_stop") == 1

    async def test_tool_use_block_id_round_trips_and_stop_reason_is_tool_use(self):
        provider = OpenAIProvider(auth_mode="api_key", api_key="test-key-placeholder")
        events = await _collect_stream(provider, OPENAI_TEXT_THEN_TOOL_STREAM)
        tool_start = next(
            e
            for e in events
            if e["event"] == "content_block_start"
            and e["data"]["content_block"]["type"] == "tool_use"
        )
        assert tool_start["data"]["content_block"]["id"] == "toolu_abc"
        delta = next(e for e in events if e["event"] == "message_delta")
        assert delta["data"]["delta"]["stop_reason"] == "tool_use"

    async def test_streaming_usage_fields_are_integers(self):
        # Auto-compact reads input/output tokens off message_start/message_delta.
        provider = OpenAIProvider(auth_mode="api_key", api_key="test-key-placeholder")
        events = await _collect_stream(provider, OPENAI_TEXT_THEN_TOOL_STREAM)
        start_usage = next(e for e in events if e["event"] == "message_start")["data"]["message"][
            "usage"
        ]
        assert start_usage["input_tokens"] == 50
        assert isinstance(start_usage["input_tokens"], int)
        delta_usage = next(e for e in events if e["event"] == "message_delta")["data"]["usage"]
        assert delta_usage["output_tokens"] == 12
        assert isinstance(delta_usage["output_tokens"], int)

    async def test_cross_chunk_buffering_preserves_block_indices(self):
        # Splitting mid-event must not corrupt the block-index sequence — the
        # buffer reassembles events across chunk boundaries.
        provider = OpenAIProvider(auth_mode="api_key", api_key="test-key-placeholder")
        split = len(OPENAI_TEXT_THEN_TOOL_STREAM) // 2
        whole = await _collect_stream(provider, OPENAI_TEXT_THEN_TOOL_STREAM)
        chunked = await _collect_stream(provider, OPENAI_TEXT_THEN_TOOL_STREAM, split_at=split)
        assert [e["event"] for e in whole] == [e["event"] for e in chunked]
        whole_starts = [e["data"]["index"] for e in whole if e["event"] == "content_block_start"]
        chunked_starts = [
            e["data"]["index"] for e in chunked if e["event"] == "content_block_start"
        ]
        assert whole_starts == chunked_starts == [0, 1]


# ---------------------------------------------------------------------------
# count_tokens estimation (auto-compact / /context relevant)
# ---------------------------------------------------------------------------


class TestCountTokensContract:
    """``estimate_input_tokens`` backs ``/v1/messages/count_tokens`` ({"input_tokens": int}).
    The contract is shape + positivity + monotonic inclusion of every section —
    not the exact divisor, which is a tunable detail."""

    def test_empty_request_estimates_zero(self):
        assert estimate_input_tokens({}) == 0

    def test_nonempty_estimate_is_positive_int(self):
        estimate = estimate_input_tokens(
            {"messages": [{"role": "user", "content": "hello world"}]}
        )
        assert isinstance(estimate, int)
        assert estimate > 0

    def test_estimate_grows_with_more_messages(self):
        # Metamorphic monotonicity: a superset of content must not estimate fewer
        # tokens than its subset.
        small = {"messages": [{"role": "user", "content": "hello"}]}
        large = {
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "a much longer reply with more tokens"},
            ]
        }
        assert estimate_input_tokens(large) > estimate_input_tokens(small)

    def test_tool_definitions_are_counted(self):
        # Inclusion oracle: dropping tool counting (a plausible mutation) must lower
        # the estimate — so the same request with tools estimates strictly more.
        base = {"messages": [{"role": "user", "content": "hi"}]}
        with_tools = {
            **base,
            "tools": [
                {"name": "Read", "description": "Read a file", "input_schema": {"type": "object"}}
            ],
        }
        assert estimate_input_tokens(with_tools) > estimate_input_tokens(base)

    def test_system_context_is_counted(self):
        base = {"messages": [{"role": "user", "content": "hi"}]}
        with_system = {
            **base,
            "system": "You are Claude Code with a long set of rules and skills.",
        }
        assert estimate_input_tokens(with_system) > estimate_input_tokens(base)
