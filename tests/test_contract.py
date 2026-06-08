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

from claude_bridge.providers.openai import (
    OpenAIProvider,
    _safe_token,
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
# tool_choice + parallel controls
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

    def test_auto_maps_to_auto(self):
        result, _ = anthropic_to_openai(self._request({"type": "auto"}))
        assert result["tool_choice"] == "auto"

    def test_none_maps_to_none(self):
        result, _ = anthropic_to_openai(self._request({"type": "none"}))
        assert result["tool_choice"] == "none"

    def test_any_maps_to_required(self):
        result, _ = anthropic_to_openai(self._request({"type": "any"}))
        assert result["tool_choice"] == "required"

    def test_named_tool_maps_to_forced_function(self):
        result, _ = anthropic_to_openai(self._request({"type": "tool", "name": "Read"}))
        assert result["tool_choice"] == {"type": "function", "name": "Read"}

    def test_absent_tool_choice_omits_the_field(self):
        # No policy requested → provider default; the bridge must not invent one.
        result, _ = anthropic_to_openai(
            {"model": "claude-opus-4-6", "messages": [{"role": "user", "content": "hi"}]}
        )
        assert "tool_choice" not in result

    def test_unknown_tool_choice_type_warns_and_omits(self):
        # Unsupported policy degrades safely: warn, and omit rather than guess.
        result, warnings = anthropic_to_openai(self._request({"type": "totally_new"}))
        assert "tool_choice" not in result
        assert any("totally_new" in w for w in warnings)


class TestParallelToolContract:
    """Anthropic ``disable_parallel_tool_use`` must serialize tool calls on the
    provider side, or Claude Code's one-at-a-time tool loop breaks."""

    def test_disable_parallel_maps_to_parallel_tool_calls_false(self):
        request = {
            "model": "claude-opus-4-6",
            "tools": [{"name": "Read", "description": "", "input_schema": {}}],
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
        }
        result, _ = anthropic_to_openai(request)
        assert result["parallel_tool_calls"] is False

    def test_parallel_not_constrained_when_not_disabled(self):
        # Default Anthropic behavior allows parallel tools; the bridge must not
        # silently force serialization when the client did not ask for it.
        result, _ = anthropic_to_openai(
            {
                "model": "claude-opus-4-6",
                "tools": [{"name": "Read", "description": "", "input_schema": {}}],
                "messages": [{"role": "user", "content": "hi"}],
                "tool_choice": {"type": "auto"},
            }
        )
        assert "parallel_tool_calls" not in result


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

    def test_incomplete_max_output_tokens_maps_to_max_tokens(self):
        # Oracle: OpenAI incomplete_details.reason "max_output_tokens" is truncation,
        # which Anthropic reports as max_tokens — the signal Claude Code auto-compacts on.
        response = {
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [],
        }
        assert openai_to_anthropic(response)["stop_reason"] == "max_tokens"

    def test_incomplete_content_filter_does_not_mask_as_max_tokens(self):
        # A moderation block must NOT look like token exhaustion, or Claude Code
        # auto-compacts a context that is nowhere near full. Anthropic convention
        # (mirrors the Gemini provider): refusal text + end_turn.
        response = {
            "status": "incomplete",
            "incomplete_details": {"reason": "content_filter"},
            "output": [],
        }
        result = openai_to_anthropic(response)
        assert result["stop_reason"] == "end_turn"
        assert any(
            block.get("type") == "text" and block.get("text") for block in result["content"]
        )

    def test_refusal_output_item_becomes_text_block(self):
        # A model refusal item carries human-readable text; it must reach Claude Code
        # as a text block rather than being silently dropped.
        response = {
            "status": "completed",
            "output": [{"type": "refusal", "refusal": "I can't help with that."}],
        }
        result = openai_to_anthropic(response)
        assert result["stop_reason"] == "end_turn"
        assert {"type": "text", "text": "I can't help with that."} in result["content"]

    def test_tool_use_takes_precedence_over_incomplete(self):
        # A truncated turn that still emitted a tool call is a tool_use turn — Claude
        # Code must run the tool, not compact.
        response = {
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [{"type": "function_call", "name": "Read", "call_id": "fc_1"}],
        }
        assert openai_to_anthropic(response)["stop_reason"] == "tool_use"


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

    def test_float_usage_coerced_to_int(self):
        # Claude Code's /context math divides by these; a float would break integer
        # arithmetic. Coerce defensively to int regardless of provider quirks.
        response = {
            "status": "completed",
            "output": [],
            "usage": {"input_tokens": 100.0, "output_tokens": 7.0},
        }
        usage = openai_to_anthropic(response)["usage"]
        assert usage == {"input_tokens": 100, "output_tokens": 7}
        assert isinstance(usage["input_tokens"], int)
        assert isinstance(usage["output_tokens"], int)

    def test_non_numeric_usage_defaults_to_zero(self):
        response = {"status": "completed", "output": [], "usage": {"input_tokens": None}}
        usage = openai_to_anthropic(response)["usage"]
        assert usage == {"input_tokens": 0, "output_tokens": 0}

    def test_cache_and_reasoning_details_fold_into_totals(self):
        # OpenAI's input_tokens already INCLUDES cached_tokens, and output_tokens
        # already INCLUDES reasoning_tokens (both are subsets per the Responses
        # contract). Anthropic's input_tokens/output_tokens are the full totals, so
        # the bridge reports them flat — never splitting cached out as an extra field
        # (which would double-count under Anthropic's non-overlapping model). The
        # optional *_details objects must not leak or perturb the flat totals.
        # See D-USAGE-001.
        response = {
            "status": "completed",
            "output": [],
            "usage": {
                "input_tokens": 1000,
                "output_tokens": 200,
                "input_tokens_details": {"cached_tokens": 768},
                "output_tokens_details": {"reasoning_tokens": 150},
            },
        }
        usage = openai_to_anthropic(response)["usage"]
        assert usage == {"input_tokens": 1000, "output_tokens": 200}

    def test_missing_token_detail_objects_default_safely(self):
        # Providers may omit the optional *_details objects entirely; absence must
        # not raise or alter the flat totals.
        response = {
            "status": "completed",
            "output": [],
            "usage": {"input_tokens": 50, "output_tokens": 10},
        }
        usage = openai_to_anthropic(response)["usage"]
        assert usage == {"input_tokens": 50, "output_tokens": 10}


class TestOracleEnvelopeShape:
    """The Anthropic Messages response envelope every Claude Code turn must parse.

    Seeds the optional Moonshot/Kimi oracle workflow (README -> "Verifying Against an
    Anthropic-Compatible Reference"): the deterministic offline anchor a maintainer
    diffs a redacted live-reference response against. Expected values come from the
    Anthropic Messages API object schema (the ``type`` and ``role`` constants, the
    ``stop_reason`` enum) — never from running the translator under test.
    """

    @staticmethod
    def _completed_text_response() -> dict:
        """A minimal completed Responses payload: one assistant text turn."""
        return {
            "id": "resp_oracle",
            "model": "gpt-5.5",
            "status": "completed",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "Done."}]}],
            "usage": {"input_tokens": 12, "output_tokens": 3},
        }

    def test_envelope_is_an_assistant_message(self):
        # Anthropic spec constants: a Messages response is always type "message" with
        # role "assistant". The bridge sets both as literals; an oracle diff would flag
        # any drift here immediately.
        result = openai_to_anthropic(self._completed_text_response())
        assert result["type"] == "message"
        assert result["role"] == "assistant"

    def test_envelope_carries_every_required_anthropic_field(self):
        # Spec oracle: the Messages response object requires this field set. Dropping
        # any one breaks SDK parsing on the Claude Code side — the exact regression the
        # oracle workflow exists to catch before it ships.
        result = openai_to_anthropic(self._completed_text_response())
        required = {"id", "type", "role", "model", "content", "stop_reason", "usage"}
        assert required <= result.keys()
        assert isinstance(result["id"], str) and result["id"]
        assert isinstance(result["content"], list) and result["content"]

    def test_stop_reason_is_a_valid_anthropic_enum_value(self):
        # Spec table: stop_reason is one of these four (or null mid-stream; a completed
        # response is never null). A value outside the set is an SDK parse error.
        result = openai_to_anthropic(self._completed_text_response())
        assert result["stop_reason"] in {"end_turn", "max_tokens", "stop_sequence", "tool_use"}


# ---------------------------------------------------------------------------
# Unsupported / special content blocks (server-tool, MCP) — D-SRVTOOL-001
# ---------------------------------------------------------------------------


# Anthropic server-tool / MCP-connector blocks the bridge cannot route to OpenAI. Each
# carries a SECRET_* sentinel in its nested content that must never reach the provider
# request. Shapes follow the Anthropic server-tool / MCP content-block schemas.
UNSUPPORTED_SERVER_TOOL_BLOCKS: list[dict] = [
    {
        "type": "server_tool_use",
        "id": "srvtoolu_1",
        "name": "web_search",
        "input": {"query": "SECRET_SEARCH_QUERY"},
    },
    {
        "type": "web_search_tool_result",
        "tool_use_id": "srvtoolu_1",
        "content": [
            {"type": "web_search_result", "title": "SECRET_TITLE", "url": "https://SECRET_URL"}
        ],
    },
    {
        "type": "mcp_tool_use",
        "id": "mcptoolu_1",
        "name": "fetch",
        "server_name": "files",
        "input": {"path": "SECRET_MCP_PATH"},
    },
    {
        "type": "mcp_tool_result",
        "tool_use_id": "mcptoolu_1",
        "content": [{"type": "text", "text": "SECRET_MCP_OUTPUT"}],
    },
    {
        "type": "code_execution_tool_result",
        "tool_use_id": "ce_1",
        "content": {"type": "code_execution_result", "stdout": "SECRET_STDOUT"},
    },
]

# Sentinels embedded in the blocks above — none may appear in a translated request.
UNSUPPORTED_BLOCK_SECRETS: tuple[str, ...] = (
    "SECRET_SEARCH_QUERY",
    "SECRET_TITLE",
    "SECRET_URL",
    "SECRET_MCP_PATH",
    "SECRET_MCP_OUTPUT",
    "SECRET_STDOUT",
)


class TestUnsupportedContentBlocks:
    """Anthropic server-tool and MCP blocks have no OpenAI Responses route. They must
    degrade to a redacted, type-named placeholder — never the raw block dict, which
    would pollute the provider request with a Python repr AND leak the block's nested
    tool inputs/outputs. See D-SRVTOOL-001."""

    def _request_with(self, block: dict) -> dict:
        return {
            "model": "claude-opus-4-6",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": [block]}],
        }

    def test_unsupported_block_content_never_leaks_into_request(self):
        # Oracle: the nested tool input/output sentinels are private content; a correct
        # redaction emits none of them. Derived from the no-leak contract, not from
        # running the translator.
        for block in UNSUPPORTED_SERVER_TOOL_BLOCKS:
            result, _ = anthropic_to_openai(self._request_with(block))
            serialized = json.dumps(result)
            for sentinel in UNSUPPORTED_BLOCK_SECRETS:
                assert sentinel not in serialized, f"{block['type']} leaked {sentinel}"

    def test_unsupported_block_becomes_typed_placeholder(self):
        # The degraded block keeps the type name (so the turn is debuggable) but carries
        # no nested content.
        for block in UNSUPPORTED_SERVER_TOOL_BLOCKS:
            result, _ = anthropic_to_openai(self._request_with(block))
            user_items = [i for i in result["input"] if i.get("role") == "user"]
            assert len(user_items) == 1
            placeholder = user_items[0]["content"][0]
            assert placeholder["type"] == "input_text"
            assert block["type"] in placeholder["text"]
            for sentinel in UNSUPPORTED_BLOCK_SECRETS:
                assert sentinel not in placeholder["text"]

    def test_unsupported_block_emits_warning_naming_type(self):
        # A silent degradation hides a behavioral divergence; every degraded block must
        # surface a warning that names the dropped type.
        for block in UNSUPPORTED_SERVER_TOOL_BLOCKS:
            _, warnings = anthropic_to_openai(self._request_with(block))
            assert any(block["type"] in w for w in warnings), f"no warning for {block['type']}"

    def test_known_blocks_remain_lossless(self):
        # Regression: hardening the unknown path must not perturb text / tool_use /
        # tool_result translation for Claude Code's client tools.
        result, warnings = anthropic_to_openai(CLAUDE_CODE_TOOL_LOOP)
        kinds = [_kind(i) for i in result["input"]]
        assert "function_call" in kinds  # tool_use preserved
        assert "function_call_output" in kinds  # tool_result preserved
        # The canonical tool loop has no unsupported blocks → no degradation warnings.
        assert not any("redacted placeholder" in w for w in warnings)


class TestLogInjectionNeutralization:
    """A request's block / tool_choice ``type`` is attacker-controllable and flows
    verbatim into a translation warning that the proxy writes to the human log and
    the structural trace. An unsanitized newline lets a hostile request forge a
    second log record (CWE-117). Every such token is neutralized at construction
    via ``_safe_token``. See ADV1."""

    def test_safe_token_preserves_benign_identifier(self):
        # Oracle: an already-safe identifier is returned unchanged — identity, not
        # mutation. Derived from "neutralize control chars", which a clean token has none of.
        assert _safe_token("server_tool_use") == "server_tool_use"

    def test_safe_token_strips_control_characters(self):
        # Oracle: newline (0x0A), carriage return (0x0D), and tab (0x09) are
        # non-printable control characters; the CWE-117 contract removes every one.
        # Expected value hand-derived from the spec, not from running the sanitizer.
        assert _safe_token("a\nb\rc\td") == "abcd"

    def test_safe_token_removes_newline_that_would_forge_a_record(self):
        # Oracle: the canonical injection payload is a newline introducing a fake
        # log line; after sanitizing, no newline remains to split the record.
        forged = "x\n2026-01-01 12:00:00 WARNING forged-by-attacker"
        assert "\n" not in _safe_token(forged)
        assert "\r" not in _safe_token(forged)

    def test_safe_token_caps_oversized_token(self):
        # Oracle: an unbounded type floods the log/trace line; the bridge caps the
        # payload at _SAFE_TOKEN_MAX (64) printable chars plus a literal "..." marker.
        # Exact match (vs startswith/len bound) kills marker-drop and slice-width mutants.
        assert _safe_token("a" * 500) == "a" * 64 + "..."

    def test_safe_token_keeps_token_at_exact_cap(self):
        # Oracle: a token exactly at the cap is returned whole — the boundary truncates
        # only what exceeds 64. Kills the > vs >= off-by-one on the cap comparison.
        assert _safe_token("a" * 64) == "a" * 64

    def test_safe_token_coerces_non_string(self):
        # tool_choice.get("type") is None when the key is absent; the warning still
        # interpolates it, so the sanitizer must coerce to str before stripping.
        assert _safe_token(None) == "None"

    def test_unsupported_block_warning_neutralizes_injected_newline(self):
        # End-to-end: a hostile block type must not inject a control char into the
        # degradation warning, and the redacted placeholder carries the sanitized token.
        request = {
            "model": "claude-opus-4-6",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": [{"type": "evil\ntype\rx"}]}],
        }
        result, warnings = anthropic_to_openai(request)
        assert warnings, "a hostile block must still emit a degradation warning"
        for w in warnings:
            assert "\n" not in w and "\r" not in w
        assert any("eviltypex" in w for w in warnings)
        assert "eviltypex" in json.dumps(result)

    def test_unsupported_tool_choice_warning_neutralizes_injected_newline(self):
        request = {
            "model": "claude-opus-4-6",
            "tools": [{"name": "Read", "description": "", "input_schema": {}}],
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": {"type": "evil\ntype"},
        }
        _, warnings = anthropic_to_openai(request)
        assert any("eviltype" in w for w in warnings)
        for w in warnings:
            assert "\n" not in w and "\r" not in w


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

    def test_tool_result_content_is_counted(self):
        # Tool-result payloads (file contents, command output) dominate Claude Code's
        # context growth — they must contribute to the estimate, not be ignored.
        base = {"messages": [{"role": "user", "content": "ok"}]}
        with_tool_result = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": "x" * 5000,
                        }
                    ],
                }
            ]
        }
        assert estimate_input_tokens(with_tool_result) > estimate_input_tokens(base)


# ---------------------------------------------------------------------------
# Reasoning continuity across tool turns (T-005)
# ---------------------------------------------------------------------------


def _provider() -> OpenAIProvider:
    """A provider in api_key mode — translation never touches auth, so any mode works."""
    return OpenAIProvider(auth_mode="api_key", api_key="test-key-placeholder")


def _reasoning_item(item_id: str, encrypted: str) -> dict:
    """A Responses reasoning output item carrying opaque encrypted continuation state."""
    return {"type": "reasoning", "id": item_id, "encrypted_content": encrypted, "summary": []}


def _function_call_item(call_id: str, name: str = "Read", item_id: str | None = None) -> dict:
    """A Responses function_call output item. ``id`` defaults to ``call_id`` (the common
    case), but the two diverge in real traffic — see test_injection_keyed_by_call_id."""
    return {
        "type": "function_call",
        "id": item_id if item_id is not None else call_id,
        "call_id": call_id,
        "name": name,
        "arguments": "{}",
    }


def _response_with(output: list[dict]) -> dict:
    """A completed Responses payload wrapping the given output items."""
    return {
        "id": "resp_1",
        "model": "gpt-5.5",
        "status": "completed",
        "output": output,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def _tool_turn_request(*tool_use_ids: str) -> dict:
    """A Turn-2 Anthropic request replaying one assistant tool call per id, each with a
    matching user tool_result. The toolu_ ids round-trip back to fc_ call_ids."""
    assistant_blocks = [
        {"type": "tool_use", "id": tid, "name": "Read", "input": {"path": "/x"}}
        for tid in tool_use_ids
    ]
    result_blocks = [
        {"type": "tool_result", "tool_use_id": tid, "content": "data"} for tid in tool_use_ids
    ]
    return {
        "model": "claude-opus-4-6",
        "tools": [{"name": "Read", "description": "", "input_schema": {"type": "object"}}],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "go"}]},
            {"role": "assistant", "content": assistant_blocks},
            {"role": "user", "content": result_blocks},
        ],
    }


def _reasoning_before(items: list[dict], call_id: str) -> dict | None:
    """Return the input item immediately preceding the function_call with ``call_id``."""
    for index, item in enumerate(items):
        if item.get("type") == "function_call" and item.get("call_id") == call_id:
            return items[index - 1] if index > 0 else None
    return None


class TestReasoningContinuity:
    """T-005: gpt-5.5 with ``store: false`` rejects a function_call whose required
    reasoning item is absent from the input. The provider asks for encrypted reasoning,
    captures it from each response keyed by call_id, and re-injects it immediately before
    the matching function_call on the next request — opaque, in-memory, never leaked."""

    def test_request_asks_for_encrypted_reasoning(self):
        # Oracle: stateless (store:false) continuity requires the Responses opt-in
        # include flag "reasoning.encrypted_content" — from the OpenAI spec, not the code.
        result, _ = anthropic_to_openai(CLAUDE_CODE_TOOL_LOOP)
        assert result["include"] == ["reasoning.encrypted_content"]

    def test_capture_then_inject_places_reasoning_immediately_before_call(self):
        provider = _provider()
        provider.translate_response(
            _response_with([_reasoning_item("rs_1", "ENC_1"), _function_call_item("fc_xyz")])
        )
        translated, _ = provider.translate_request(_tool_turn_request("toolu_xyz"))
        preceding = _reasoning_before(translated["input"], "fc_xyz")
        assert preceding is not None
        assert preceding.get("type") == "reasoning"
        assert preceding.get("encrypted_content") == "ENC_1"

    def test_injection_keyed_by_call_id_not_item_id(self):
        # Response function_call: call_id and id diverge (real Responses behavior).
        # openai_to_anthropic mints the toolu_ id from call_id, so the cache must too.
        provider = _provider()
        provider.translate_response(
            _response_with(
                [
                    _reasoning_item("rs_1", "ENC_KEYED"),
                    _function_call_item("call_xyz", item_id="fc_unrelated"),
                ]
            )
        )
        translated, _ = provider.translate_request(_tool_turn_request("toolu_xyz"))
        preceding = _reasoning_before(translated["input"], "fc_xyz")
        assert preceding is not None
        assert preceding.get("encrypted_content") == "ENC_KEYED"

    def test_fresh_provider_injects_nothing(self):
        # No capture yet → translate_request is a pure passthrough of the translation,
        # with no reasoning items synthesized.
        provider = _provider()
        baseline, _ = anthropic_to_openai(_tool_turn_request("toolu_xyz"))
        translated, _ = provider.translate_request(_tool_turn_request("toolu_xyz"))
        assert translated["input"] == baseline["input"]
        assert all(item.get("type") != "reasoning" for item in translated["input"])

    def test_intervening_message_breaks_pairing(self):
        # A reasoning item separated from the function_call by a message is NOT its
        # continuation state — the call must not receive that reasoning on replay.
        provider = _provider()
        provider.translate_response(
            _response_with(
                [
                    _reasoning_item("rs_1", "ENC_STALE"),
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "thinking out loud"}],
                    },
                    _function_call_item("fc_xyz"),
                ]
            )
        )
        translated, _ = provider.translate_request(_tool_turn_request("toolu_xyz"))
        assert all(item.get("type") != "reasoning" for item in translated["input"])

    def test_serial_reasonings_pair_with_their_own_calls(self):
        # Two reasoning/call pairs in one response: each call gets ITS OWN reasoning,
        # not the other's. Kills a "pending reasoning is never replaced" mutant.
        provider = _provider()
        provider.translate_response(
            _response_with(
                [
                    _reasoning_item("rs_1", "ENC_A"),
                    _function_call_item("fc_a"),
                    _reasoning_item("rs_2", "ENC_B"),
                    _function_call_item("fc_b", name="Read"),
                ]
            )
        )
        translated, _ = provider.translate_request(_tool_turn_request("toolu_a", "toolu_b"))
        before_a = _reasoning_before(translated["input"], "fc_a")
        before_b = _reasoning_before(translated["input"], "fc_b")
        assert before_a is not None and before_a.get("encrypted_content") == "ENC_A"
        assert before_b is not None and before_b.get("encrypted_content") == "ENC_B"

    def test_parallel_calls_get_single_preceding_reasoning(self):
        # One reasoning item precedes two parallel calls. On replay the reasoning item
        # appears exactly once (a duplicate item id would be rejected), before the first.
        provider = _provider()
        provider.translate_response(
            _response_with(
                [
                    _reasoning_item("rs_1", "ENC_SHARED"),
                    _function_call_item("fc_a"),
                    _function_call_item("fc_b"),
                ]
            )
        )
        translated, _ = provider.translate_request(_tool_turn_request("toolu_a", "toolu_b"))
        kinds = [_kind(item) for item in translated["input"]]
        assert kinds.count("reasoning") == 1
        before_a = _reasoning_before(translated["input"], "fc_a")
        assert before_a is not None and before_a.get("encrypted_content") == "ENC_SHARED"

    def test_captured_reasoning_not_returned_to_claude_code(self):
        # The opaque encrypted blob must never appear in the Anthropic response —
        # reasoning items are captured for replay only, never surfaced as content.
        provider = _provider()
        anthropic = provider.translate_response(
            _response_with([_reasoning_item("rs_1", "ENC_SECRET"), _function_call_item("fc_xyz")])
        )
        assert "ENC_SECRET" not in json.dumps(anthropic)
        assert any(block.get("type") == "tool_use" for block in anthropic["content"])

    def test_reasoning_cache_evicts_oldest_when_over_capacity(self):
        from claude_bridge.providers.openai import _REASONING_CACHE_MAX

        provider = _provider()
        for n in range(_REASONING_CACHE_MAX + 1):  # one past the bound
            provider.translate_response(
                _response_with(
                    [_reasoning_item(f"rs_{n}", f"ENC_{n}"), _function_call_item(f"fc_{n}")]
                )
            )
        # Bounded memory.
        assert len(provider._reasoning_by_call_id) <= _REASONING_CACHE_MAX
        # Behavioral oracle: the oldest call was evicted (no injection); the newest survives.
        oldest, _ = provider.translate_request(_tool_turn_request("toolu_0"))
        newest, _ = provider.translate_request(_tool_turn_request(f"toolu_{_REASONING_CACHE_MAX}"))
        assert all(item.get("type") != "reasoning" for item in oldest["input"])
        assert any(item.get("type") == "reasoning" for item in newest["input"])

    def test_reasoning_cache_is_thread_safe_under_concurrent_eviction(self):
        # T-005 claims _reasoning_by_call_id is concurrency-safe via _reasoning_lock.
        # The eviction loop reads `oldest = next(iter(...))` then `del`s that key.
        # Without the lock two threads read the same oldest key and the second `del`
        # raises KeyError. The GIL does NOT close this window — it spans two bytecodes,
        # and a thread switch landing between them double-deletes. We force that switch
        # with a near-zero switch interval, pin the cache at the bound so every stash
        # runs the eviction critical section, and align all workers on a barrier so
        # they collide there simultaneously.
        import sys
        import threading

        from claude_bridge.providers.openai import _REASONING_CACHE_MAX

        provider = _provider()
        # Pre-fill to the bound: now every concurrent stash adds one and immediately
        # evicts one, so the critical section runs on every single operation.
        provider._stash_reasoning(
            {f"seed_{i}": {"id": f"rs_seed_{i}"} for i in range(_REASONING_CACHE_MAX)}
        )

        worker_count = 16
        errors: list[Exception] = []
        ready = threading.Barrier(worker_count)

        def stash_many(worker: int) -> None:
            ready.wait()  # all workers hit the eviction loop together
            try:
                for n in range(200):
                    provider.translate_response(
                        _response_with(
                            [
                                _reasoning_item(f"rs_{worker}_{n}", f"ENC_{worker}_{n}"),
                                _function_call_item(f"fc_{worker}_{n}"),
                            ]
                        )
                    )
            except Exception as exc:  # a lost-lock eviction race surfaces here
                errors.append(exc)

        original_interval = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)  # force preemption between next(iter()) and del
        try:
            threads = [threading.Thread(target=stash_many, args=(w,)) for w in range(worker_count)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        finally:
            sys.setswitchinterval(original_interval)

        assert errors == [], f"concurrent stash raised: {errors}"
        assert len(provider._reasoning_by_call_id) <= _REASONING_CACHE_MAX

    def test_streaming_completed_event_captures_reasoning(self):
        import asyncio

        provider = _provider()
        stream = _sse(
            "response.completed",
            {
                "type": "response.completed",
                "response": _response_with(
                    [_reasoning_item("rs_1", "ENC_STREAM"), _function_call_item("fc_s")]
                ),
            },
        )
        asyncio.run(_collect_stream(provider, stream))
        translated, _ = provider.translate_request(_tool_turn_request("toolu_s"))
        preceding = _reasoning_before(translated["input"], "fc_s")
        assert preceding is not None
        assert preceding.get("encrypted_content") == "ENC_STREAM"
