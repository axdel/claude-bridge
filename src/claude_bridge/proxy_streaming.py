"""SSE streaming data-plane for the proxy.

Opens streaming upstream/provider responses over the shared httpx client, validates the
response BEFORE any downstream byte is written (no bare 200 ahead of a failed body),
pumps translated Anthropic SSE events to the client, and folds a provider's SSE deltas
into a single Messages response for non-streaming clients. The connection lifecycle and
routing stay in ``proxy``; this module owns the stream translation and pumping.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import httpx

from claude_bridge.http_client import (
    _is_remote_protocol_error,
    open_stream,
    retire_pooled_connections,
    select_forward_headers,
)
from claude_bridge.log import get_logger
from claude_bridge.provider import Provider, provider_capabilities
from claude_bridge.request_view import emit_translation_warnings, trace_stream_event
from claude_bridge.stream import format_anthropic_sse
from claude_bridge.wire import (
    ClientDisconnected,
    StreamOutcome,
    anthropic_error_body,
    provider_error_log_summary,
    provider_error_message,
    safe_write,
    write_response,
    write_sse_headers,
)

logger = get_logger("proxy")

# SSE events too noisy for DEBUG — normal stream lifecycle, not interesting.
_QUIET_SSE_EVENTS = frozenset(
    {
        "content_block_delta",
        "content_block_start",
        "content_block_stop",
        "message_start",
        "message_delta",
        "message_stop",
        "ping",
    }
)


class _MessageAccumulator:
    """Folds Anthropic SSE event payloads into a single Messages response.

    Owns the in-progress message, its content blocks (keyed by index, kept in
    arrival order), and the per-block tool-argument JSON buffers. Each ``on_*``
    method consumes one event's ``data`` payload; ``build`` produces the final
    response or ``None`` if no ``message_start`` was ever seen.
    """

    def __init__(self) -> None:
        self._message: dict | None = None
        self._blocks: dict[int, dict] = {}
        self._tool_json: dict[int, str] = {}
        self._order: list[int] = []

    def on_message_start(self, data: dict) -> None:
        msg = data.get("message", {})
        self._message = {
            "id": msg.get("id", "msg_bridge_unknown"),
            "type": "message",
            "role": "assistant",
            "model": msg.get("model", ""),
            "stop_reason": None,
            "content": [],
            "usage": dict(msg.get("usage", {"input_tokens": 0, "output_tokens": 0})),
        }

    def on_content_block_start(self, data: dict) -> None:
        index = data.get("index", 0)
        block = dict(data.get("content_block", {}))
        self._blocks[index] = block
        self._order.append(index)
        if block.get("type") == "tool_use":
            self._tool_json[index] = ""

    def on_content_block_delta(self, data: dict) -> None:
        block = self._blocks.get(data.get("index", 0))
        if block is None:
            return
        delta = data.get("delta", {})
        if delta.get("type") == "text_delta":
            block["text"] = block.get("text", "") + delta.get("text", "")
        elif delta.get("type") == "input_json_delta":
            index = data.get("index", 0)
            self._tool_json[index] = self._tool_json.get(index, "") + delta.get("partial_json", "")

    def on_content_block_stop(self, data: dict) -> None:
        index = data.get("index", 0)
        block = self._blocks.get(index)
        if block is None or block.get("type") != "tool_use":
            return
        raw_args = self._tool_json.get(index, "")
        try:
            block["input"] = json.loads(raw_args) if raw_args else {}
        except (json.JSONDecodeError, ValueError):
            block["input"] = {"_raw": raw_args}

    def on_message_delta(self, data: dict) -> None:
        if self._message is None:
            return
        delta = data.get("delta", {})
        if "stop_reason" in delta:
            self._message["stop_reason"] = delta["stop_reason"]
        if data.get("usage"):
            self._message["usage"] = data["usage"]

    def build(self) -> dict | None:
        if self._message is None:
            return None
        self._message["content"] = [self._blocks[index] for index in self._order]
        return self._message


def aggregate_stream_to_message(events: list[dict]) -> dict | None:
    """Fold a sequence of Anthropic SSE events into a single Messages response.

    The inverse of the streaming translation: ``message_start`` seeds the message,
    ``content_block_*`` build the text/tool_use blocks in arrival order, and
    ``message_delta`` carries the final stop_reason and usage — producing the same
    shape ``openai_to_anthropic`` does. Returns ``None`` when no ``message_start``
    was seen (malformed/empty upstream — the caller maps this to 502).

    Pure function — no I/O.
    """
    accumulator = _MessageAccumulator()
    handlers = {
        "message_start": accumulator.on_message_start,
        "content_block_start": accumulator.on_content_block_start,
        "content_block_delta": accumulator.on_content_block_delta,
        "content_block_stop": accumulator.on_content_block_stop,
        "message_delta": accumulator.on_message_delta,
    }
    for event in events:
        handler = handlers.get(event.get("event", ""))
        if handler is not None:
            handler(event.get("data", {}))
    return accumulator.build()


async def validate_passthrough_response(
    response: httpx.Response, writer: asyncio.StreamWriter
) -> StreamOutcome | None:
    """Check an upstream passthrough response BEFORE any SSE headers are written.

    Anthropic's own error body is forwarded verbatim (passthrough is transparent); a
    200 that is not an SSE stream is buffered and forwarded as-is. Returns a
    ``StreamOutcome`` in either case, or ``None`` when a real SSE stream may be pumped.
    """
    content_type = response.headers.get("content-type", "").lower()
    if response.status_code != 200:
        err_body = await response.aread()
        await response.aclose()
        write_response(writer, response.status_code, err_body)
        return StreamOutcome(response.status_code)
    if not content_type.startswith("text/event-stream"):
        buffered = await response.aread()
        await response.aclose()
        write_response(writer, 200, buffered)
        return StreamOutcome(200)
    return None


async def _pump_passthrough_stream(
    response: httpx.Response,
    writer: asyncio.StreamWriter,
    client: httpx.AsyncClient,
) -> StreamOutcome:
    """Forward validated upstream SSE bytes to the client unchanged until exhausted.

    SSE headers are written inside this ``try`` so a raise still ``aclose``s the
    response. A mid-stream transport error (e.g. the idle timeout firing on a stalled
    upstream, INV-SSE-01) leaves the client on a committed 200, so it is flagged
    ``error=True`` rather than restated as a status. A StreamReset also retires the
    pooled HTTP/2 session so the next request does not reuse it.
    """
    try:
        write_sse_headers(writer)
        async for chunk in response.aiter_bytes():
            await safe_write(writer, chunk)
    except ClientDisconnected:
        logger.debug("Client disconnected during passthrough stream")
        return StreamOutcome(499)
    except httpx.TransportError as exc:
        if _is_remote_protocol_error(exc):
            await retire_pooled_connections(client)
        logger.error("Passthrough stream transport error: %s", exc)
        return StreamOutcome(200, error=True)
    finally:
        await response.aclose()
    return StreamOutcome(200)


async def stream_passthrough(
    upstream_url: str,
    body: bytes,
    client_headers: dict[str, str],
    writer: asyncio.StreamWriter,
    client: httpx.AsyncClient,
) -> StreamOutcome:
    """Stream an SSE response from the Anthropic upstream back to the client unchanged."""
    url = f"{upstream_url}/v1/messages"
    headers = select_forward_headers(client_headers)
    try:
        response = await open_stream(client, url, content=body, headers=headers)
    except httpx.TransportError as exc:
        logger.error("Passthrough stream connect failed: %s", exc)
        write_response(writer, 502, anthropic_error_body(502, "upstream unavailable"))
        return StreamOutcome(502)

    guard = await validate_passthrough_response(response, writer)
    if guard is not None:
        return guard

    return await _pump_passthrough_stream(response, writer, client)


async def _prepare_provider_stream(
    provider: Provider, body: bytes, writer: asyncio.StreamWriter
) -> tuple[dict, dict] | StreamOutcome:
    """Parse, authenticate, and translate a streaming request.

    Returns ``(translated, auth_headers)`` ready to send, or a ``StreamOutcome`` when a
    preflight step failed (having already written the error response to the client).
    """
    try:
        request_dict = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        write_response(writer, 400, anthropic_error_body(400, "Malformed JSON request"))
        return StreamOutcome(400)
    try:
        # Authenticate first — some providers need auth context before translation
        auth_headers = await provider.authenticate()
        translated, warnings = provider.translate_request(request_dict)
    except Exception:
        logger.exception("Provider preflight failed")
        write_response(writer, 502, anthropic_error_body(502, "Provider preflight failed"))
        return StreamOutcome(502)
    if not isinstance(translated, dict):
        logger.warning(
            "Provider %s translate_request returned %s, expected dict",
            provider.name,
            type(translated).__name__,
        )
        write_response(writer, 502, anthropic_error_body(502, "Provider translation failed"))
        return StreamOutcome(502)
    emit_translation_warnings(warnings, translated)

    # Enable streaming on the translated request when the provider declares body selection.
    if provider_capabilities(provider).stream_request_mode == "body_parameter":
        translated["stream"] = True
    return translated, auth_headers


async def validate_stream_response(
    response: httpx.Response, writer: asyncio.StreamWriter
) -> StreamOutcome | None:
    """Check a provider stream response BEFORE any SSE headers are written.

    The provider did not open a stream when the status is non-200, OR when a PRESENT
    content-type is not ``text/event-stream``; read the (buffered) error body, translate
    it, and write a normal error response — never a bare 200 followed by a failed body.
    An ABSENT/empty content-type on a 200 is accepted as SSE: the Codex backend
    (chatgpt.com, behind Cloudflare) streams valid SSE with no content-type header
    (D-STREAM-004), while a genuine non-stream error always carries one (application/json
    / text/html). Returns a ``StreamOutcome`` for the error case, or ``None`` when a real
    SSE stream may be pumped.
    """
    content_type = response.headers.get("content-type", "").lower()
    if response.status_code != 200:
        err_body = await response.aread()
        await response.aclose()
        logger.error(
            "Provider HTTP %d: %s", response.status_code, provider_error_log_summary(err_body)
        )
        write_response(
            writer,
            response.status_code,
            anthropic_error_body(response.status_code, provider_error_message(err_body)),
        )
        return StreamOutcome(response.status_code)
    # Reject a PRESENT non-SSE content-type; an absent/empty one is accepted SSE (D-STREAM-004).
    if content_type and not content_type.startswith("text/event-stream"):
        err_body = await response.aread()
        await response.aclose()
        logger.error("Provider returned non-SSE 200 (content-type=%r)", content_type)
        write_response(writer, 502, anthropic_error_body(502, "provider did not return a stream"))
        return StreamOutcome(502)
    return None


def _accumulate_usage(
    event_name: str, data: dict, tokens_in: int, tokens_out: int
) -> tuple[int, int]:
    """Fold usage from a message_start/message_delta event into running totals.

    Anthropic carries input_tokens on message_start and the authoritative output_tokens
    (and a final input_tokens) on the terminal message_delta — the latter wins, so we
    replace rather than sum (D-USAGE-001 flat totals, no double-count).
    """
    if event_name == "message_start":
        usage = data.get("message", {}).get("usage", {})
        tokens_in = usage.get("input_tokens", tokens_in) or tokens_in
    elif event_name == "message_delta":
        usage = data.get("usage") or {}
        tokens_out = usage.get("output_tokens", tokens_out)
        tokens_in = usage.get("input_tokens", tokens_in)
    return tokens_in, tokens_out


async def _emit_stream_error(
    writer: asyncio.StreamWriter, tokens_in: int, tokens_out: int
) -> StreamOutcome:
    """Write a terminal Anthropic error event mid-stream and return a failed outcome."""
    error_event = format_anthropic_sse(
        "error",
        {
            "type": "error",
            "error": {"type": "api_error", "message": "Unexpected provider stream failure"},
        },
    )
    try:
        await safe_write(writer, error_event)
    except ClientDisconnected:
        logger.debug("Client disconnected before provider stream error event")
        return StreamOutcome(499, tokens_in, tokens_out)
    return StreamOutcome(502, tokens_in, tokens_out, error=True)


async def _pump_provider_stream(
    provider: Provider,
    response: httpx.Response,
    writer: asyncio.StreamWriter,
    client: httpx.AsyncClient,
) -> StreamOutcome:
    """Translate a validated provider SSE stream to Anthropic events and write them.

    SSE headers are written here — only after ``validate_stream_response`` confirmed a
    real stream, so no downstream byte precedes a failed body. The header write sits
    inside the outer ``try`` (so ``aclose`` always runs) but outside the body-exception
    handlers: a raise there has not committed an SSE body and must propagate, not become
    a mid-stream error event. Tracks token usage and whether a terminal ``message_stop``
    was seen; a stream ending without one is a semantic failure (``error=True``) even
    though the client already received a 200. A mid-stream StreamReset retires pooled
    HTTP/2 connections on *client* so the next request does not reuse the session.
    """

    async def _raw_chunks() -> AsyncIterator[bytes]:
        async for chunk in response.aiter_bytes():
            yield chunk

    tokens_in = 0
    tokens_out = 0
    saw_terminal = False
    try:
        write_sse_headers(writer)
        try:
            async for anthropic_event in provider.translate_stream(_raw_chunks()):
                event_name = anthropic_event["event"]
                data = anthropic_event["data"]
                tokens_in, tokens_out = _accumulate_usage(event_name, data, tokens_in, tokens_out)
                if event_name == "message_stop":
                    saw_terminal = True
                trace_stream_event(anthropic_event)
                sse_bytes = format_anthropic_sse(event_name, data)
                if event_name not in _QUIET_SSE_EVENTS:
                    logger.debug("SSE -> %s", event_name)
                await safe_write(writer, sse_bytes)
        except ClientDisconnected:
            logger.debug("Client disconnected during provider stream")
            return StreamOutcome(499, tokens_in, tokens_out)
        except httpx.TransportError as exc:
            if _is_remote_protocol_error(exc):
                await retire_pooled_connections(client)
            logger.error("Provider stream transport error: %s", exc)
            return await _emit_stream_error(writer, tokens_in, tokens_out)
        except Exception:
            logger.exception("Unexpected error during provider stream")
            return await _emit_stream_error(writer, tokens_in, tokens_out)
    finally:
        await response.aclose()
    return StreamOutcome(200, tokens_in, tokens_out, error=not saw_terminal)


async def _open_stream_with_reauth(
    provider: Provider,
    translated: dict,
    auth_headers: dict[str, str],
    writer: asyncio.StreamWriter,
    client: httpx.AsyncClient,
) -> httpx.Response | StreamOutcome:
    """Open the provider stream, retrying once with a refreshed credential on a 401.

    A 401 before any SSE byte means the bearer was rejected upstream — rotated or expired since
    our proactive expiry check. Drain the rejected response, force ONE credential refresh, and
    re-open the stream exactly once, all before any downstream output, so the client never sees
    the transient 401. The x-grok-req-id contextvar is unchanged across the retry, letting the
    upstream dedup it against the first attempt. Returns the open response for the caller to
    validate and pump, or a ``StreamOutcome`` when the connection or the refresh itself failed
    (a 502 already written to the client).
    """
    content = json.dumps(translated).encode()
    try:
        response = await open_stream(
            client, provider.endpoint, content=content, headers=auth_headers
        )
    except httpx.TransportError as exc:
        logger.error("Provider connection error: %s", exc)
        write_response(writer, 502, anthropic_error_body(502, "provider unavailable"))
        return StreamOutcome(502)

    if response.status_code != 401:
        return response

    # 401 before any SSE byte: close the rejected response, force one refresh, re-open once.
    await response.aclose()
    logger.warning("Provider returned 401 on stream open; forcing a refresh and retrying once")
    try:
        auth_headers = await provider.authenticate(force_refresh=True)
    except Exception:
        logger.exception("Reactive credential refresh failed")
        write_response(writer, 502, anthropic_error_body(502, "Provider preflight failed"))
        return StreamOutcome(502)
    try:
        return await open_stream(client, provider.endpoint, content=content, headers=auth_headers)
    except httpx.TransportError as exc:
        logger.error("Provider connection error after refresh: %s", exc)
        write_response(writer, 502, anthropic_error_body(502, "provider unavailable"))
        return StreamOutcome(502)


async def stream_via_provider(
    provider: Provider,
    body: bytes,
    writer: asyncio.StreamWriter,
    client: httpx.AsyncClient,
) -> StreamOutcome:
    """Translate a request, stream from the provider, translate SSE back to Anthropic.

    Orchestrates preflight (parse/auth/translate), opening the stream (with one reactive
    401 refresh + retry), validating the response before any downstream byte, and pumping
    the translated events.
    """
    prepared = await _prepare_provider_stream(provider, body, writer)
    if isinstance(prepared, StreamOutcome):
        return prepared
    translated, auth_headers = prepared

    logger.debug(
        "Sending to provider: model=%s items=%d",
        translated.get("model"),
        len(translated.get("input", [])),
    )
    opened = await _open_stream_with_reauth(provider, translated, auth_headers, writer, client)
    if isinstance(opened, StreamOutcome):
        return opened
    response = opened

    guard = await validate_stream_response(response, writer)
    if guard is not None:
        return guard
    return await _pump_provider_stream(provider, response, writer, client)
