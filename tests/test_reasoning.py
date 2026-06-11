"""Reasoning-continuity contract (T-005 / D-REASON-001) — the bridge's one
stateful surface, split out of test_contract.py (QAL3).

gpt-5.5 with ``store: false`` rejects a function_call whose required reasoning
item is absent. The provider captures encrypted reasoning from each response
keyed by call_id and re-injects it immediately before the matching function_call
on the next request — opaque, in-memory, bounded, never leaked. These tests
exercise provider STATE across calls; the pure-function translation contract
stays in test_contract.py.
"""

from __future__ import annotations

import json

from claude_bridge.providers.openai import OpenAIProvider, anthropic_to_openai

# A full Claude Code tool loop — the canonical request shape used to assert the
# Responses encrypted-reasoning opt-in flag. Mirrors the fixture in test_contract.py
# (shared test scaffolding; kept local so this module is self-contained).
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

    def test_streaming_incomplete_event_captures_reasoning(self):
        # A turn truncated by max_output_tokens (response.incomplete) can still have
        # emitted a function_call; its preceding encrypted reasoning must be stashed,
        # or the next request's tool echo is rejected (D-REASON-001). The capture set
        # must include response.incomplete, not only response.completed.
        import asyncio

        provider = _provider()
        incomplete_response = _response_with(
            [_reasoning_item("rs_inc", "ENC_INCOMPLETE"), _function_call_item("fc_inc")]
        )
        incomplete_response["status"] = "incomplete"
        incomplete_response["incomplete_details"] = {"reason": "max_output_tokens"}
        stream = _sse(
            "response.incomplete",
            {"type": "response.incomplete", "response": incomplete_response},
        )
        asyncio.run(_collect_stream(provider, stream))
        translated, _ = provider.translate_request(_tool_turn_request("toolu_inc"))
        preceding = _reasoning_before(translated["input"], "fc_inc")
        assert preceding is not None
        assert preceding.get("encrypted_content") == "ENC_INCOMPLETE"
