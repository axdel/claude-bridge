"""Async HTTP proxy server for LLM API requests — stdlib only.

Intercepts Anthropic Messages API traffic and routes it to either
the real Anthropic upstream (passthrough) or a configured provider
(direct/failover mode) with request/response translation.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import urllib.error
import urllib.request

from claude_bridge.log import get_logger, request_id_var
from claude_bridge.provider import PROVIDERS, Provider
from claude_bridge.router import Router
from claude_bridge.stats import BridgeStats
from claude_bridge.stream import format_anthropic_sse

logger = get_logger("proxy")

_DEFAULT_UPSTREAM = "https://api.anthropic.com"

# Headers to forward from the client to the upstream API.
_FORWARD_HEADERS = ("x-api-key", "content-type", "anthropic-version")

# Upstream HTTP status codes that trigger failover.
_FAILOVER_STATUSES = {429, 500, 502, 503}

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


async def start_proxy(
    *,
    host: str = "127.0.0.1",
    port: int = 9999,
    upstream_url: str | None = None,
    provider_name: str | None = None,
    provider_kwargs: dict | None = None,
) -> asyncio.Server:
    """Start the proxy server and return the asyncio.Server handle."""
    upstream = upstream_url or os.environ.get("ANTHROPIC_REAL_URL", _DEFAULT_UPSTREAM)

    provider = None
    if provider_name:
        provider_cls = PROVIDERS.get(provider_name)
        if provider_cls is None:
            msg = f"Unknown provider '{provider_name}'. Available: {list(PROVIDERS)}"
            raise ValueError(msg)
        provider = provider_cls(**(provider_kwargs or {}))

    router = Router()
    stats = BridgeStats()
    handler = _make_handler(upstream, router, provider, stats)
    return await asyncio.start_server(handler, host, port)


def _make_handler(
    upstream_url: str,
    router: Router,
    provider: Provider | None = None,
    stats: BridgeStats | None = None,
):
    """Return a connection callback bound to *upstream_url*."""

    async def _handle_connection(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            await _process_request(
                reader, writer, upstream_url, router, provider, stats
            )
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except (ConnectionResetError, BrokenPipeError, OSError):
                pass  # Client already disconnected — expected in proxy servers

    return _handle_connection


async def _parse_request(
    reader: asyncio.StreamReader,
) -> tuple[str, str, dict[str, str], bytes] | None:
    """Read one HTTP request from *reader*. Returns (method, path, headers, body) or None."""
    request_line = await reader.readline()
    if not request_line:
        return None

    parts = request_line.decode("utf-8", errors="replace").strip().split()
    if len(parts) < 3:  # noqa: PLR2004
        return None

    method, path = parts[0], parts[1]

    headers: dict[str, str] = {}
    while True:
        line = await reader.readline()
        if line in (b"\r\n", b"\n", b""):
            break
        decoded = line.decode("utf-8", errors="replace").strip()
        if ":" in decoded:
            key, value = decoded.split(":", 1)
            headers[key.strip().lower()] = value.strip()

    try:
        content_length = int(headers.get("content-length", "0"))
    except (ValueError, TypeError):
        return None  # Malformed Content-Length — caller sends 400
    body = await reader.readexactly(content_length) if content_length else b""
    return method, path, headers, body


def _estimate_tokens(body: bytes) -> int:
    """Estimate input tokens from a count_tokens request body."""
    try:
        request = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        return 0
    from claude_bridge.providers.openai import estimate_input_tokens

    return estimate_input_tokens(request)


def _record_sync_response(
    stats: BridgeStats | None,
    request_start: float,
    status_code: int,
    response_body: bytes,
) -> None:
    """Extract usage from a sync response and record stats."""
    if stats is None:
        return
    import time as _time

    latency_ms = (_time.monotonic() - request_start) * 1000
    tokens_in = tokens_out = 0
    try:
        data = json.loads(response_body)
        usage = data.get("usage", {})
        tokens_in = usage.get("input_tokens", 0)
        tokens_out = usage.get("output_tokens", 0)
    except (json.JSONDecodeError, ValueError, AttributeError):
        pass
    stats.record_response(status_code, latency_ms, tokens_in, tokens_out)


def _record_latency(stats: BridgeStats | None, request_start: float) -> None:
    """Record latency only (for streaming responses where tokens aren't easily available)."""
    if stats is None:
        return
    import time as _time

    latency_ms = (_time.monotonic() - request_start) * 1000
    stats.record_response(200, latency_ms, 0, 0)


def _is_streaming(body: bytes) -> bool:
    """Return True if the request body has ``"stream": true``."""
    try:
        return json.loads(body).get("stream") is True
    except (json.JSONDecodeError, ValueError):
        return False


async def _process_request(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    upstream_url: str,
    router: Router,
    provider: Provider | None = None,
    stats: BridgeStats | None = None,
) -> None:
    """Parse one HTTP request and proxy or reject it."""
    import time as _time

    # Assign a short request ID for log correlation
    request_id_var.set(secrets.token_hex(4))
    request_start = _time.monotonic()

    parsed = await _parse_request(reader)
    if parsed is None:
        _write_response(writer, 400, b'{"error": "malformed request"}')
        return

    method, path, headers, body = parsed
    logger.info("%s %s (%dB)", method, path, len(body))

    # Strip query string for path matching (e.g. /v1/messages?beta=true → /v1/messages)
    base_path = path.split("?")[0]

    # Stats endpoint (accepts any method — POST from curl/test helpers is fine)
    if base_path == "/stats":
        snap = stats.snapshot() if stats else {}
        _write_response(writer, 200, json.dumps(snap).encode())
        return

    # Handle count_tokens — estimate from request body
    if method == "POST" and base_path == "/v1/messages/count_tokens":
        token_count = _estimate_tokens(body)
        logger.debug("count_tokens -> %d", token_count)
        _write_response(writer, 200, json.dumps({"input_tokens": token_count}).encode())
        return

    if method != "POST" or base_path != "/v1/messages":
        logger.info("-> 404 (unsupported path)")
        _write_response(writer, 404, b'{"error": "not found"}')
        return

    # Track this as a real request
    if stats:
        stats.record_request()

    streaming = _is_streaming(body)
    request_model = _extract_model(body)

    await _route_request(
        provider, upstream_url, headers, body, writer, router,
        stats, streaming, request_model, request_start,
    )


def _extract_model(body: bytes) -> str:
    """Extract the model name from a request body, or 'unknown'."""
    try:
        return json.loads(body).get("model", "unknown")
    except (json.JSONDecodeError, ValueError):
        return "unknown"


async def _route_request(
    provider: Provider | None,
    upstream_url: str,
    headers: dict[str, str],
    body: bytes,
    writer: asyncio.StreamWriter,
    router: Router,
    stats: BridgeStats | None,
    streaming: bool,
    request_model: str,
    request_start: float,
) -> None:
    """Route a /v1/messages request to the appropriate backend."""
    if provider is not None:
        mode = "stream" if streaming else "sync"
        logger.info(
            "-> DIRECT %s (%s) model=%s", provider.name, mode, request_model
        )
        if stats:
            stats.set_provider_info(provider.name, request_model)
        if streaming:
            await _stream_via_provider(provider, body, writer)
            _record_latency(stats, request_start)
        else:
            status_code, response_body = await _forward_via_provider(provider, body)
            _write_response(writer, status_code, response_body)
            _record_sync_response(stats, request_start, status_code, response_body)
    elif streaming:
        logger.info("-> passthrough (stream) model=%s", request_model)
        if stats:
            stats.set_provider_info("anthropic", request_model)
        await _stream_passthrough(upstream_url, body, headers, writer)
        _record_latency(stats, request_start)
    else:
        logger.info("-> auto-route (sync) model=%s", request_model)
        if stats:
            stats.set_provider_info("anthropic", request_model)
        status_code, response_body = await _auto_route(
            upstream_url, headers, body, router
        )
        _write_response(writer, status_code, response_body)
        _record_sync_response(stats, request_start, status_code, response_body)


async def _try_failover(router: Router, body: bytes) -> tuple[int, bytes] | None:
    """Attempt failover to the registered provider. Returns None if not possible."""
    fallback = _get_fallback_provider()
    if fallback is None:
        return None

    request_dict = json.loads(body)
    eligible, reason = router.is_failover_eligible(request_dict)
    if not eligible:
        logger.warning("Failover ineligible: %s", reason)
        return None

    if not router.should_use_fallback():
        return None

    return await _forward_via_provider(fallback, body)


async def _auto_route(
    upstream_url: str, headers: dict[str, str], body: bytes, router: Router
) -> tuple[int, bytes]:
    """Auto mode: try Anthropic, failover on error."""
    # If circuit breaker is OPEN, try fallback first
    if router.should_use_fallback():
        result = await _try_failover(router, body)
        if result is not None:
            return result

    # Try Anthropic upstream
    status_code, response_body = await asyncio.to_thread(
        _forward_request, upstream_url, body, headers
    )

    if status_code not in _FAILOVER_STATUSES:
        await router.record_success()
        return status_code, response_body

    # Anthropic failed — record and try failover
    await router.record_failure()
    result = await _try_failover(router, body)
    if result is not None:
        return result

    return status_code, response_body


_provider_cache: dict[str, Provider] = {}


def _get_fallback_chain() -> list[str]:
    """Return the ordered list of fallback provider names.

    Reads ``LLM_BRIDGE_FALLBACK`` env var (comma-separated). Defaults to ``["openai"]``.
    """
    raw = os.environ.get("LLM_BRIDGE_FALLBACK")
    if raw is None:
        return ["openai"]
    return [name.strip() for name in raw.split(",") if name.strip()]


def _get_cached_provider(name: str) -> Provider | None:
    """Return a cached provider instance, creating it on first access."""
    if name in _provider_cache:
        return _provider_cache[name]
    provider_cls = PROVIDERS.get(name)
    if provider_cls is None:
        return None
    instance = provider_cls()
    _provider_cache[name] = instance
    return instance


def _get_fallback_provider() -> Provider | None:
    """Return the first available provider from the fallback chain."""
    for name in _get_fallback_chain():
        provider = _get_cached_provider(name)
        if provider is not None:
            return provider
    return None


async def _forward_via_provider(provider: Provider, body: bytes) -> tuple[int, bytes]:
    """Translate, authenticate, forward to provider, translate back.

    The Codex endpoint always streams, so we read the SSE stream and extract
    the final response from the ``response.completed`` event.
    """
    request_dict = json.loads(body)
    translated, warnings = provider.translate_request(request_dict)
    for w in warnings:
        logger.warning("Translation: %s", w)

    auth_headers = await provider.authenticate()

    # Open a streaming connection and collect the full response
    def _collect_response():
        data = json.dumps(translated).encode()
        req = urllib.request.Request(provider.endpoint, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        for key, value in auth_headers.items():
            req.add_header(key, value)
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                full_body = resp.read()
                return resp.status, full_body
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            logger.error("Provider connection error: %s", exc)
            return 502, json.dumps({"error": "provider unavailable"}).encode()

    status_code, raw_response = await asyncio.to_thread(_collect_response)
    logger.info("Provider response: %d (%dB)", status_code, len(raw_response))
    if status_code != 200:
        logger.error("Provider error: %s", raw_response[:500])
        return status_code, raw_response

    # Parse SSE stream to find the response.completed event with the full response
    response_dict = _extract_completed_response(raw_response)
    if response_dict is None:
        # Fallback: try plain JSON parse
        try:
            response_dict = json.loads(raw_response)
        except (json.JSONDecodeError, ValueError):
            return 502, json.dumps(
                {"error": "could not parse provider response"}
            ).encode()

    anthropic_response = provider.translate_response(response_dict)
    return 200, json.dumps(anthropic_response).encode()


def _extract_completed_response(raw_sse: bytes) -> dict | None:
    """Extract the full response dict from a ``response.completed`` SSE event."""
    for line in raw_sse.decode("utf-8", errors="replace").split("\n"):
        if line.startswith("data: "):
            try:
                event_data = json.loads(line[6:])
                if event_data.get("type") == "response.completed":
                    return event_data.get("response")
            except (json.JSONDecodeError, ValueError):
                continue
    return None


def _forward_request(
    upstream_url: str, body: bytes, client_headers: dict[str, str]
) -> tuple[int, bytes]:
    """Synchronous HTTP POST to the upstream — called from asyncio.to_thread."""
    url = f"{upstream_url}/v1/messages"
    req = urllib.request.Request(url, data=body, method="POST")

    for key in _FORWARD_HEADERS:
        if key in client_headers:
            req.add_header(key, client_headers[key])

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        # Upstream returned an error status — forward it
        return exc.code, exc.read()
    except (urllib.error.URLError, TimeoutError, OSError):
        error = json.dumps({"error": "upstream unavailable"}).encode()
        return 502, error


class _ClientDisconnected(Exception):
    """Raised when a write/drain fails because the client closed the connection."""


async def _safe_write(writer: asyncio.StreamWriter, data: bytes) -> None:
    """Write data and drain, raising _ClientDisconnected on broken pipe."""
    try:
        writer.write(data)
        await writer.drain()
    except (ConnectionResetError, BrokenPipeError, OSError) as exc:
        raise _ClientDisconnected from exc


def _write_sse_headers(writer: asyncio.StreamWriter) -> None:
    """Write HTTP/1.1 200 headers for an SSE stream."""
    writer.write(b"HTTP/1.1 200 OK\r\n")
    writer.write(b"Content-Type: text/event-stream\r\n")
    writer.write(b"Cache-Control: no-cache\r\n")
    writer.write(b"Connection: keep-alive\r\n")
    writer.write(b"\r\n")


async def _stream_passthrough(
    upstream_url: str,
    body: bytes,
    client_headers: dict[str, str],
    writer: asyncio.StreamWriter,
) -> None:
    """Stream an SSE response from the Anthropic upstream back to the client unchanged."""

    def _open_stream():
        url = f"{upstream_url}/v1/messages"
        req = urllib.request.Request(url, data=body, method="POST")
        for key in _FORWARD_HEADERS:
            if key in client_headers:
                req.add_header(key, client_headers[key])
        return urllib.request.urlopen(req, timeout=120)  # noqa: S310

    try:
        resp = await asyncio.to_thread(_open_stream)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError):
        _write_response(
            writer, 502, json.dumps({"error": "upstream unavailable"}).encode()
        )
        return

    _write_sse_headers(writer)

    try:
        while True:
            chunk = await asyncio.to_thread(resp.read, 4096)
            if not chunk:
                break
            await _safe_write(writer, chunk)
    except _ClientDisconnected:
        logger.debug("Client disconnected during passthrough stream")
    finally:
        resp.close()


async def _stream_via_provider(
    provider: Provider,
    body: bytes,
    writer: asyncio.StreamWriter,
) -> None:
    """Translate request, stream from provider, translate SSE events back to Anthropic format."""
    request_dict = json.loads(body)
    translated, warnings = provider.translate_request(request_dict)
    for w in warnings:
        logger.warning("Translation: %s", w)

    # Enable streaming on the translated request
    translated["stream"] = True

    auth_headers = await provider.authenticate()

    def _open_stream():
        data = json.dumps(translated).encode()
        req = urllib.request.Request(provider.endpoint, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        for key, value in auth_headers.items():
            req.add_header(key, value)
        return urllib.request.urlopen(req, timeout=120)  # noqa: S310

    logger.debug(
        "Sending to provider: model=%s items=%d",
        translated.get("model"),
        len(translated.get("input", [])),
    )

    try:
        resp = await asyncio.to_thread(_open_stream)
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")[:500]
        logger.error("Provider HTTP %d: %s", exc.code, err_body)
        _write_response(writer, exc.code, json.dumps({"error": err_body}).encode())
        return
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        logger.error("Provider connection error: %s", exc)
        _write_response(
            writer, 502, json.dumps({"error": "provider unavailable"}).encode()
        )
        return

    _write_sse_headers(writer)

    async def _raw_chunks():
        """Async generator yielding raw byte chunks from the HTTP response."""
        try:
            while True:
                chunk = await asyncio.to_thread(resp.read, 4096)
                if not chunk:
                    break
                yield chunk
        finally:
            resp.close()

    try:
        async for anthropic_event in provider.translate_stream(_raw_chunks()):
            sse_bytes = format_anthropic_sse(
                anthropic_event["event"], anthropic_event["data"]
            )
            event_name = anthropic_event["event"]
            if event_name not in _QUIET_SSE_EVENTS:
                logger.debug("SSE -> %s", event_name)
            await _safe_write(writer, sse_bytes)
    except _ClientDisconnected:
        logger.debug("Client disconnected during provider stream")


def _write_response(writer: asyncio.StreamWriter, status: int, body: bytes) -> None:
    """Write a minimal HTTP/1.1 response."""
    reasons = {200: "OK", 400: "Bad Request", 404: "Not Found", 502: "Bad Gateway"}
    reason = reasons.get(status, "Error")
    writer.write(f"HTTP/1.1 {status} {reason}\r\n".encode())
    writer.write(b"Content-Type: application/json\r\n")
    writer.write(f"Content-Length: {len(body)}\r\n".encode())
    writer.write(b"Connection: close\r\n")
    writer.write(b"\r\n")
    writer.write(body)
