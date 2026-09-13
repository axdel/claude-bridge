"""Thinking-block round-trip symmetry — what the bridge emits, it must accept back.

PR #24 gave the bridge an OUTBOUND reasoning -> Anthropic thinking-block translation
without changing the INBOUND one, so the two directions stopped being inverses: a block
the bridge emitted came back as literal ``[thinking]...[/thinking]`` assistant text in
the upstream prompt, silently, once per block per request, accumulating every turn.

These tests pin the SYMMETRY rather than either direction alone. A per-direction test
passes happily while the pair is broken; only driving a block out and back catches it.

Fixture provenance (Boundary Fixture Fidelity): ``REASONING_SUMMARY_WIRE`` replays an
event sequence CAPTURED from the live OpenAI Responses endpoint on 2026-09-13 (probe in
the branch RunDir), not authored from memory of the API. It is the shape that actually
produced the defect: several reasoning ITEMS per turn, several summary PARTS per item,
one delta per part carrying a bold heading, and ``reasoning_summary_text.done``
interleaved before each ``part.done``.
"""

from __future__ import annotations

import json

import pytest

import claude_bridge.providers.openai.translate as openai_translate
import claude_bridge.providers.xai.translate as xai_translate
from claude_bridge.providers.openai import OpenAIProvider, anthropic_to_openai
from claude_bridge.providers.xai import XAIProvider, anthropic_to_xai

# The literal tag pair that must never reach an upstream payload. Asserted as data
# rather than inline strings so a partial leak ("[thinking" with no close) still fails.
THINKING_TAGS = ("[thinking]", "[/thinking]", "[thinking")


def _sse(event_type: str, payload: dict) -> bytes:
    """Build one Responses SSE wire event (``event:``/``data:`` block)."""
    return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n".encode()


def _summary_part(output_index: int, summary_index: int, text: str) -> bytes:
    """One complete summary part: added -> delta -> text.done -> part.done.

    Mirrors the captured wire order exactly, including the ``text.done`` the bridge
    deliberately skips (it repeats what the delta already carried).
    """
    common = {"output_index": output_index, "summary_index": summary_index}
    return b"".join(
        [
            _sse(
                "response.reasoning_summary_part.added",
                {
                    "type": "response.reasoning_summary_part.added",
                    **common,
                    "part": {"type": "summary_text", "text": ""},
                },
            ),
            _sse(
                "response.reasoning_summary_text.delta",
                {"type": "response.reasoning_summary_text.delta", **common, "delta": text},
            ),
            _sse(
                "response.reasoning_summary_text.done",
                {"type": "response.reasoning_summary_text.done", **common, "text": text},
            ),
            _sse(
                "response.reasoning_summary_part.done",
                {
                    "type": "response.reasoning_summary_part.done",
                    **common,
                    "part": {"type": "summary_text", "text": text},
                },
            ),
        ]
    )


# Captured shape: three reasoning items (output_index 0,1,2) carrying 2/2/1 summary
# parts, then the message item at output_index 3 whose text uses content_index 0.
CAPTURED_SUMMARY_HEADINGS = [
    (0, 0, "**Verifying the move table**"),
    (0, 1, "**Proving seven-step minimum**"),
    (1, 0, "**Verifying boat-trip safety**"),
    (1, 1, "**Refining proof details**"),
    (2, 0, "**Proving the crossing count**"),
]

REASONING_SUMMARY_WIRE = b"".join(
    [
        _sse(
            "response.created",
            {
                "type": "response.created",
                "response": {"id": "resp_1", "model": "gpt-6-astra", "usage": {"input_tokens": 9}},
            },
        ),
        *[_summary_part(oi, si, text) for oi, si, text in CAPTURED_SUMMARY_HEADINGS],
        _sse(
            "response.content_part.added",
            {
                "type": "response.content_part.added",
                "output_index": 3,
                "content_index": 0,
                "part": {"type": "output_text", "text": ""},
            },
        ),
        _sse(
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "output_index": 3,
                "content_index": 0,
                "delta": "Seven crossings.",
            },
        ),
        _sse(
            "response.output_text.done",
            {"type": "response.output_text.done", "output_index": 3, "content_index": 0},
        ),
        _sse(
            "response.completed",
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_1",
                    "model": "gpt-6-astra",
                    "status": "completed",
                    "output": [],
                    "usage": {"input_tokens": 9, "output_tokens": 40},
                },
            },
        ),
    ]
)


async def _collect_stream(provider, raw: bytes) -> list[dict]:
    """Drive ``translate_stream`` over ``raw`` and return the translated events."""

    async def _chunks():
        yield raw

    return [event async for event in provider.translate_stream(_chunks())]


def _assemble_assistant_content(events: list[dict]) -> list[dict]:
    """Rebuild the assistant content array a streaming client accumulates.

    Mirrors what Claude Code actually does, verified against a real session via
    ``--output-format stream-json``: blocks are keyed by index, deltas append, and a
    ``thinking`` block with no ``signature`` on the wire is stored with ``signature: ""``
    rather than dropped.
    """
    blocks: dict[int, dict] = {}
    order: list[int] = []
    for event in events:
        data = event.get("data", {})
        index = data.get("index")
        if event.get("event") == "content_block_start":
            block = dict(data.get("content_block", {}))
            if block.get("type") == "thinking":
                block.setdefault("signature", "")
            blocks[index] = block
            order.append(index)
        elif event.get("event") == "content_block_delta":
            delta = data.get("delta", {})
            block = blocks.get(index)
            if block is None:
                continue
            if delta.get("type") == "thinking_delta":
                block["thinking"] = block.get("thinking", "") + delta.get("thinking", "")
            elif delta.get("type") == "text_delta":
                block["text"] = block.get("text", "") + delta.get("text", "")
            elif delta.get("type") == "signature_delta":
                block["signature"] = block.get("signature", "") + delta.get("signature", "")
    return [blocks[i] for i in order]


def _replay_request(assistant_content: list[dict]) -> dict:
    """An Anthropic request replaying ``assistant_content`` as the prior turn."""
    return {
        "model": "claude-opus-5",
        "max_tokens": 400,
        "thinking": {"type": "adaptive"},
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "solve it"}]},
            {"role": "assistant", "content": assistant_content},
            {"role": "user", "content": [{"type": "text", "text": "continue"}]},
        ],
    }


def _openai_provider() -> OpenAIProvider:
    """A provider in api_key mode — translation never touches auth, so any mode works."""
    return OpenAIProvider(auth_mode="api_key", api_key="test-key-placeholder")


PROVIDER_CASES = [
    pytest.param(_openai_provider, anthropic_to_openai, id="openai"),
    pytest.param(XAIProvider, anthropic_to_xai, id="xai"),
]

# (factory, translate module, the module attribute caching REASONING_MODE). The two
# providers name that constant differently, so the attribute travels with the case.
REASONING_MODE_CASES = [
    pytest.param(_openai_provider, openai_translate, "_REASONING_MODE", id="openai"),
    pytest.param(XAIProvider, xai_translate, "_XAI_REASONING_MODE", id="xai"),
]


class TestThinkingRoundTrip:
    """The bridge's two thinking translations must be inverses of each other.

    D-XAI-002 keeps openai and xai as independent copies of the Responses translation,
    so every case runs against BOTH — a fix landing in one file only is the documented
    failure mode of that decision.
    """

    @pytest.mark.parametrize("make_provider, translate_request", PROVIDER_CASES)
    def test_emitted_thinking_block_replays_without_a_literal_tag(
        self, make_provider, translate_request
    ):
        """The defect itself: a block the bridge emitted comes back as literal text.

        Oracle: the Anthropic Messages API carries reasoning in a typed ``thinking``
        content block; no part of its published wire format is a bracketed tag inside
        assistant text. A tag appearing upstream is therefore bridge-invented, and it
        is what teaches the model to emit ``[thinking`` as its own output.
        """
        import asyncio

        events = asyncio.run(_collect_stream(make_provider(), REASONING_SUMMARY_WIRE))
        assistant_content = _assemble_assistant_content(events)
        assert any(block.get("type") == "thinking" for block in assistant_content), (
            "fixture did not produce a thinking block — the round trip is untested"
        )

        upstream, _warnings = translate_request(_replay_request(assistant_content))
        payload = json.dumps(upstream)
        for tag in THINKING_TAGS:
            assert tag not in payload, f"bridge-invented {tag!r} reached the upstream payload"

    @pytest.mark.parametrize("make_provider, translate_request", PROVIDER_CASES)
    def test_replayed_turn_carries_no_more_blocks_than_it_emitted(
        self, make_provider, translate_request
    ):
        """Replaying a turn must not grow the prompt beyond what the turn contained.

        Oracle: a round trip is lossless, not amplifying. The measured defect was
        monotonic growth (0 -> 14 thinking blocks over 6 turns of a trivial task),
        which is what made a long session unusable.
        """
        import asyncio

        events = asyncio.run(_collect_stream(make_provider(), REASONING_SUMMARY_WIRE))
        assistant_content = _assemble_assistant_content(events)
        thinking_blocks = [b for b in assistant_content if b.get("type") == "thinking"]

        upstream, _ = translate_request(_replay_request(assistant_content))
        assistant_items = [i for i in upstream["input"] if i.get("role") == "assistant"]
        emitted_parts = sum(len(item.get("content", [])) for item in assistant_items)
        assert emitted_parts <= len(assistant_content), (
            f"replay expanded {len(assistant_content)} blocks into {emitted_parts} parts"
        )
        assert len(thinking_blocks) == 1, (
            f"one assistant turn produced {len(thinking_blocks)} thinking blocks; "
            "the Anthropic shape is one per turn"
        )

    @pytest.mark.parametrize("make_provider, _translate_request", PROVIDER_CASES)
    def test_emitted_thinking_block_carries_a_signature(self, make_provider, _translate_request):
        """Oracle: the published Messages API wire format shows ``signature`` on every
        thinking block, and requires blocks be passed back "complete and unmodified"
        (platform.claude.com/docs/en/build-with-claude/thinking-tool-workflows). A block
        with no signature is not a valid thinking block.
        """
        import asyncio

        events = asyncio.run(_collect_stream(make_provider(), REASONING_SUMMARY_WIRE))
        starts = [
            event["data"]["content_block"]
            for event in events
            if event.get("event") == "content_block_start"
            and event["data"].get("content_block", {}).get("type") == "thinking"
        ]
        assert starts, "fixture produced no thinking block"
        for block in starts:
            assert "signature" in block, f"thinking block emitted without a signature: {block}"

    @pytest.mark.parametrize("make_provider, _translate_request", PROVIDER_CASES)
    def test_every_opened_block_is_closed_exactly_once(self, make_provider, _translate_request):
        """Framing invariant across the captured multi-item shape.

        Reasoning blocks are keyed on ``output_index`` and text blocks on
        ``content_index`` — two numbering spaces sharing one remap table. This pins the
        ordering assumption that keeps them from colliding, which no example-based test
        of a single reasoning item would exercise.
        """
        import asyncio

        events = asyncio.run(_collect_stream(make_provider(), REASONING_SUMMARY_WIRE))
        live: list[int] = []
        closed: list[int] = []
        for event in events:
            index = event.get("data", {}).get("index")
            if event.get("event") == "content_block_start":
                assert index not in live, f"index {index} re-opened while still open"
                live.append(index)
            elif event.get("event") == "content_block_stop":
                assert index in live, f"index {index} closed but never opened"
                live.remove(index)
                closed.append(index)
        assert live == [], f"blocks left unclosed at stream end: {live}"
        assert len(closed) == len(set(closed)), "a block was closed more than once"

    @pytest.mark.parametrize("make_provider, _translate_request", PROVIDER_CASES)
    def test_upstream_dropping_mid_reasoning_still_closes_the_block(
        self, make_provider, _translate_request
    ):
        """A merged block is held open across parts, so a truncated stream must flush it.

        Oracle: Anthropic's streaming contract pairs every ``content_block_start`` with a
        ``content_block_stop``. Coalescing defers that stop until the reasoning phase
        ends, which introduces a state an abrupt upstream disconnect can strand — the
        client would wait forever on a block that never closes.
        """
        import asyncio

        # Cut the captured wire after the first summary part's deltas — no part.done,
        # no text block, no terminal event: the shape of a dropped connection.
        truncated = REASONING_SUMMARY_WIRE.split(b"event: response.reasoning_summary_part.done")[0]
        events = asyncio.run(_collect_stream(make_provider(), truncated))

        starts = [e for e in events if e.get("event") == "content_block_start"]
        stops = [e for e in events if e.get("event") == "content_block_stop"]
        assert len(starts) == 1, "the truncated stream should open exactly one block"
        assert [e["data"]["index"] for e in stops] == [starts[0]["data"]["index"]]
        assert events[-1]["event"] == "message_stop", "the turn must still terminate"

    @pytest.mark.parametrize("make_provider, translate_module, mode_attr", REASONING_MODE_CASES)
    def test_drop_mode_surfaces_no_thinking_block(
        self, make_provider, translate_module, mode_attr, monkeypatch
    ):
        """REASONING_MODE=drop suppresses the reasoning phase — the knob's one remaining job.

        Oracle: with a returned block now omitted under every mode, ``drop`` would be
        dead config if it governed nothing. It governs the outbound direction: a user who
        does not want reasoning in their TUI gets the answer text and nothing else.
        """
        import asyncio

        monkeypatch.setattr(translate_module, mode_attr, "drop")
        events = asyncio.run(_collect_stream(make_provider(), REASONING_SUMMARY_WIRE))
        assistant_content = _assemble_assistant_content(events)

        assert [b.get("type") for b in assistant_content] == ["text"]
        assert not any(
            e.get("data", {}).get("delta", {}).get("type") == "thinking_delta" for e in events
        )
        assert events[-1]["event"] == "message_stop"
