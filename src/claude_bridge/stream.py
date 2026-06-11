"""SSE (Server-Sent Events) parsing and formatting utilities — stdlib only.

Shared by the proxy (for formatting outbound Anthropic events) and by
providers (for parsing inbound provider-specific SSE streams).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator


async def iter_sse_event_blobs(
    raw_chunks: AsyncIterator[bytes],
    *,
    max_buffer: int | None = None,
) -> AsyncIterator[bytes]:
    """Reframe a raw provider byte stream into complete SSE event blobs.

    Buffers ``raw_chunks`` and yields one blob per ``\\n\\n``-terminated event
    (CRLF normalized to LF), then any non-empty trailing remainder. This is the
    single owner of SSE byte-framing; every provider's ``translate_stream``
    derives its event boundaries here instead of re-implementing the buffer walk.

    When ``max_buffer`` is set and the undrained buffer exceeds it without a
    terminator, raises ``RuntimeError`` — a terminator-less stream would
    otherwise grow the buffer without bound (OOM) and make repeated
    concatenation quadratic.
    """
    buffer = b""
    async for chunk in raw_chunks:
        buffer += chunk
        buffer = buffer.replace(b"\r\n", b"\n")
        while b"\n\n" in buffer:
            event_end = buffer.index(b"\n\n") + 2
            event_bytes = buffer[:event_end]
            buffer = buffer[event_end:]
            yield event_bytes
        if max_buffer is not None and len(buffer) > max_buffer:
            msg = (
                f"Provider SSE stream exceeded {max_buffer} bytes without an "
                "event terminator; aborting malformed stream"
            )
            raise RuntimeError(msg)
    if buffer.strip():
        yield buffer


def parse_sse_events(raw_bytes: bytes) -> list[dict]:
    """Parse SSE-formatted bytes into a list of ``{event, data}`` dicts.

    Skips events whose data is the ``[DONE]`` sentinel.
    Each returned dict has:
      - ``event`` (str): the SSE event type, or ``""`` if not specified.
      - ``data`` (dict): the parsed JSON payload.
    """
    text = raw_bytes.decode("utf-8", errors="replace")
    # Split on double-newline boundaries (handles both \n\n and \r\n\r\n)
    blocks = text.replace("\r\n", "\n").split("\n\n")

    events: list[dict] = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue

        event_type = ""
        data_line = ""

        for line in block.split("\n"):
            if line.startswith("event:"):
                event_type = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_line = line[len("data:") :].strip()

        if not data_line or data_line == "[DONE]":
            continue

        try:
            data = json.loads(data_line)
        except (json.JSONDecodeError, ValueError):
            continue

        events.append({"event": event_type, "data": data})

    return events


def format_anthropic_sse(event_type: str, data: dict) -> bytes:
    """Format one Anthropic SSE event as wire bytes.

    Returns ``b'event: <type>\\ndata: <json>\\n\\n'``.
    """
    json_str = json.dumps(data, separators=(",", ":"))
    return f"event: {event_type}\ndata: {json_str}\n\n".encode()
