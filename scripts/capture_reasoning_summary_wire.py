#!/usr/bin/env python3
"""Capture the RAW upstream reasoning-summary SSE sequence from the Responses endpoint.

This is the regeneration command for ``REASONING_SUMMARY_WIRE`` in
``tests/test_thinking_roundtrip.py``. That fixture is a golden capture of an EXTERNAL
contract, so it is only as trustworthy as its provenance: a boundary fixture authored
from memory of an API tests the code against its own assumptions and can never falsify
them. Re-run this when the upstream event shape is suspected to have changed, and
rebuild the fixture from what it prints rather than from recollection.

The bridge's own trace records only TRANSLATED events, so it cannot show the upstream
shape the translator is fed. This drives the provider's real endpoint with its real
auth and prints the untranslated stream -- the only place the per-item/per-part
structure (several reasoning items per turn, several summary parts per item,
``reasoning_summary_text.done`` interleaved before each ``part.done``) is visible.

The prompt is deliberately one that forces multi-step reasoning; a trivial prompt
yields a single summary part and hides exactly the structure the fixture must carry.

NOT part of the package or the test suite -- a standalone diagnostic. Prints event
structure and deltas only; it never prints auth headers.

Usage:
    uv run python scripts/capture_reasoning_summary_wire.py
"""

import asyncio
import json
import sys

import httpx

from claude_bridge.providers.openai.provider import OpenAIProvider

PROMPT = (
    "A farmer must move a wolf, a goat and a cabbage across a river in a boat that "
    "holds the farmer plus one item. The wolf eats the goat if left alone with it; "
    "the goat eats the cabbage if left alone with it. Work out the full crossing "
    "sequence, then verify each step leaves no forbidden pair unattended, then state "
    "the minimum number of crossings and prove no shorter sequence exists."
)


async def main() -> int:
    provider = OpenAIProvider()
    headers = await provider.authenticate()

    request = {
        "model": "claude-opus-5",
        "max_tokens": 900,
        "stream": True,
        "thinking": {"type": "adaptive"},
        "messages": [{"role": "user", "content": PROMPT}],
    }
    translated, _warnings = provider.translate_request(request)

    interesting = ("reasoning", "output_item", "content_part", "output_text")
    counts: dict[str, int] = {}

    async with (
        httpx.AsyncClient(http2=True, timeout=180.0) as client,
        client.stream("POST", provider.endpoint, json=translated, headers=headers) as response,
    ):
        print(f"upstream status: {response.status_code}")
        if response.status_code != 200:
            print((await response.aread()).decode()[:800])
            return 1
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue
            event_type = event.get("type", "")
            counts[event_type] = counts.get(event_type, 0) + 1
            if not any(k in event_type for k in interesting):
                continue
            fields = {
                "output_index": event.get("output_index"),
                "summary_index": event.get("summary_index"),
                "content_index": event.get("content_index"),
                "item_id": (event.get("item_id") or "")[-8:] or None,
            }
            fields = {k: v for k, v in fields.items() if v is not None}
            extra = ""
            if "delta" in event:
                extra = f"  delta={event['delta']!r}"
            elif "part" in event:
                extra = f"  part={json.dumps(event['part'])[:120]}"
            elif "item" in event:
                item = event["item"]
                extra = f"  item.type={item.get('type')} id={(item.get('id') or '')[-8:]}"
            print(f"  {event_type:52s} {fields}{extra}")

    print("\n=== event type counts ===")
    for name, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {count:4d}  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
