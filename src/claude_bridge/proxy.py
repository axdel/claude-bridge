"""Async HTTP proxy server for LLM API requests.

Intercepts Anthropic Messages API traffic and routes it to either
the real Anthropic upstream (passthrough) or a configured provider
(direct/failover mode) with request/response translation.

Data-plane upstream calls run over a shared httpx HTTP/2 client (``http_client``);
provider token/OIDC refresh stays on stdlib urllib inside each provider (D-TRANSPORT-001).
Client-facing response writing and error shaping live in ``wire``; the SSE streaming
data-plane (validate-before-headers, pump, aggregate) lives in ``proxy_streaming``. This
module owns the server lifecycle, request parsing, routing, failover, and sync forwarding.
"""

from __future__ import annotations

import asyncio
import json
import secrets
import time as _time
import uuid

import httpx

import claude_bridge.config as config
from claude_bridge.http_client import create_client, forward_request, post_provider
from claude_bridge.log import get_logger, request_id_var, upstream_request_id_var
from claude_bridge.provider import PROVIDERS, Provider, provider_capabilities, validate_provider
from claude_bridge.proxy_streaming import (
    aggregate_stream_to_message,
    stream_passthrough,
    stream_via_provider,
)
from claude_bridge.request_view import (
    emit_translation_warnings,
    estimate_tokens,
    trace_inbound_request,
    trace_provider_response,
)
from claude_bridge.router import Router
from claude_bridge.stats import BridgeStats
from claude_bridge.wire import (
    StreamOutcome,
    anthropic_error_body,
    provider_error_log_summary,
    provider_error_message,
    safe_log_excerpt,
    write_response,
)

logger = get_logger("proxy")


def _warn_invalid_max_request_body(raw: str) -> None:
    """Log invalid import-time request body limit configuration."""
    logger.warning(
        "Invalid MAX_REQUEST_BODY=%r, using default %dB",
        raw,
        config.DEFAULT_MAX_REQUEST_BODY,
    )


_MAX_REQUEST_BODY = config.max_request_body(on_invalid=_warn_invalid_max_request_body)


# Upstream HTTP status codes that trigger failover.
_FAILOVER_STATUSES = {429, 500, 502, 503}


async def start_proxy(
    *,
    host: str = "127.0.0.1",
    port: int = 9999,
    upstream_url: str | None = None,
    provider_name: str | None = None,
    provider_kwargs: dict | None = None,
    http_client_instance: httpx.AsyncClient | None = None,
) -> tuple[asyncio.Server, httpx.AsyncClient]:
    """Start the proxy server; return the ``(server, http client)`` pair.

    The caller owns the returned client and MUST ``aclose`` it after the server stops.
    A client may be injected (tests pass an ``httpx.MockTransport`` client); otherwise
    one is created here. If the server fails to start, a self-created client is closed
    before the error propagates — an injected client stays the caller's to close.
    """
    upstream = config.validate_upstream_url(upstream_url or config.anthropic_real_url())

    provider = None
    if provider_name:
        provider_cls = PROVIDERS.get(provider_name)
        if provider_cls is None:
            msg = f"Unknown provider '{provider_name}'. Available: {list(PROVIDERS)}"
            raise ValueError(msg)
        provider = validate_provider(provider_cls(**(provider_kwargs or {})))

    router = Router()
    stats = BridgeStats()
    owns_client = http_client_instance is None
    client = http_client_instance or create_client()
    handler = _make_handler(upstream, router, provider, stats, client=client)
    try:
        server = await asyncio.start_server(handler, host, port)
    except BaseException:
        if owns_client:
            await client.aclose()
        raise
    return server, client


def _make_handler(
    upstream_url: str,
    router: Router,
    provider: Provider | None = None,
    stats: BridgeStats | None = None,
    *,
    client: httpx.AsyncClient,
):
    """Return a connection callback bound to *upstream_url* and the shared HTTP client."""

    async def _handle_connection(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            await _process_request(
                reader, writer, upstream_url, router, provider, stats, client=client
            )
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except (ConnectionResetError, BrokenPipeError, OSError):
                pass  # Client already disconnected — expected in proxy servers

    return _handle_connection


class _RequestTooLarge(Exception):
    """Raised when Content-Length exceeds MAX_REQUEST_BODY."""


async def _parse_request(
    reader: asyncio.StreamReader,
) -> tuple[str, str, dict[str, str], bytes] | None:
    """Read one HTTP request from *reader*. Returns (method, path, headers, body) or None."""
    request_line = await reader.readline()
    if not request_line:
        return None

    parts = request_line.decode("utf-8", errors="replace").strip().split()
    if len(parts) < 3:
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
    if content_length > _MAX_REQUEST_BODY:
        raise _RequestTooLarge
    body = await reader.readexactly(content_length) if content_length else b""
    return method, path, headers, body


def _record_sync_response(
    stats: BridgeStats | None,
    request_start: float,
    status_code: int,
    response_body: bytes,
) -> None:
    """Extract usage from a sync response and record stats."""
    if stats is None:
        return

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


def _record_stream_response(
    stats: BridgeStats | None,
    request_start: float,
    outcome: StreamOutcome,
) -> None:
    """Record streaming latency, token usage, and error state from a stream outcome."""
    if stats is None:
        return

    latency_ms = (_time.monotonic() - request_start) * 1000
    stats.record_response(
        outcome.status,
        latency_ms,
        outcome.tokens_in,
        outcome.tokens_out,
        error=outcome.error,
    )


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
    *,
    client: httpx.AsyncClient,
) -> None:
    """Parse one HTTP request and proxy or reject it."""

    # Assign a short request ID for log correlation
    request_id_var.set(secrets.token_hex(4))
    # Assign the per-request upstream id sent as x-grok-req-id. Set once here (not inside
    # authenticate) so it stays stable across the transport and reactive-401 retries, letting
    # the upstream dedup a retried request; a full uuid, distinct from the 8-hex log token.
    upstream_request_id_var.set(str(uuid.uuid4()))
    request_start = _time.monotonic()

    try:
        parsed = await _parse_request(reader)
    except _RequestTooLarge:
        error_body = json.dumps(
            {
                "type": "error",
                "error": {
                    "type": "request_too_large",
                    "message": f"Request body exceeds maximum size ({_MAX_REQUEST_BODY} bytes)",
                },
            }
        ).encode()
        write_response(writer, 413, error_body)
        return
    if parsed is None:
        write_response(writer, 400, b'{"error": "malformed request"}')
        return

    method, path, headers, body = parsed
    logger.info("%s %s (%dB)", method, path, len(body))

    # Strip query string for path matching (e.g. /v1/messages?beta=true → /v1/messages)
    base_path = path.split("?")[0]

    # Health check endpoint
    if base_path == "/health":
        write_response(writer, 200, json.dumps({"status": "ok"}).encode())
        return

    # Stats endpoint (accepts any method — POST from curl/test helpers is fine)
    if base_path == "/stats":
        snap = stats.snapshot() if stats else {}
        write_response(writer, 200, json.dumps(snap).encode())
        return

    # Handle count_tokens — estimate from request body
    if method == "POST" and base_path == "/v1/messages/count_tokens":
        token_count = estimate_tokens(body)
        logger.debug("count_tokens -> %d", token_count)
        write_response(writer, 200, json.dumps({"input_tokens": token_count}).encode())
        return

    if method != "POST" or base_path != "/v1/messages":
        logger.info("-> 404 (unsupported path)")
        write_response(writer, 404, b'{"error": "not found"}')
        return

    # Track this as a real request
    if stats:
        stats.record_request()

    streaming = _is_streaming(body)
    request_model = _extract_model(body)

    await _route_request(
        provider,
        upstream_url,
        headers,
        body,
        writer,
        router,
        stats,
        streaming,
        request_model,
        request_start,
        client,
    )


# A real model id is a short printable token (e.g. "grok-4.6", "gpt-5.6-sol").
# The extracted value is request-controlled and reaches INFO logs and stats, so it
# is bounded to keep a forged value from flooding the logs.
_MODEL_LOG_LIMIT = 100


def _extract_model(body: bytes) -> str:
    """Extract the request model as a bounded, control-stripped token, or 'unknown'.

    The value is request-controlled and reaches INFO logs and stats, so it is
    sanitized here (CWE-117 log-forging defense): safe_log_excerpt collapses control
    characters — an embedded newline would otherwise let a caller inject a forged log
    line — and bounds the length. A non-string or missing model reports 'unknown'.
    """
    try:
        parsed = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        return "unknown"
    model = parsed.get("model") if isinstance(parsed, dict) else None
    if not isinstance(model, str):
        return "unknown"
    return safe_log_excerpt(model, limit=_MODEL_LOG_LIMIT) or "unknown"


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
    client: httpx.AsyncClient,
) -> None:
    """Route a /v1/messages request to the appropriate backend."""
    trace_inbound_request(body)
    if provider is not None:
        mode = "stream" if streaming else "sync"
        logger.info("-> DIRECT %s (%s) model=%s", provider.name, mode, request_model)
        if stats:
            stats.set_provider_info(provider.name, request_model)
        if streaming:
            outcome = await stream_via_provider(provider, body, writer, client)
            _record_stream_response(stats, request_start, outcome)
        else:
            status_code, response_body = await _forward_via_provider(provider, body, client)
            write_response(writer, status_code, response_body)
            _record_sync_response(stats, request_start, status_code, response_body)
    elif streaming:
        logger.info("-> passthrough (stream) model=%s", request_model)
        if stats:
            stats.set_provider_info("anthropic", request_model)
        outcome = await stream_passthrough(upstream_url, body, headers, writer, client)
        _record_stream_response(stats, request_start, outcome)
    else:
        logger.info("-> auto-route (sync) model=%s", request_model)
        if stats:
            stats.set_provider_info("anthropic", request_model)
        status_code, response_body, rl_headers = await _auto_route(
            upstream_url, headers, body, router, stats, client
        )
        write_response(writer, status_code, response_body, rl_headers)
        _record_sync_response(stats, request_start, status_code, response_body)


async def _try_failover(
    router: Router,
    body: bytes,
    client: httpx.AsyncClient,
    stats: BridgeStats | None = None,
) -> tuple[int, bytes] | None:
    """Attempt failover to the registered provider. Returns None if not possible."""
    fallback = _get_fallback_provider()
    if fallback is None:
        return None

    request_dict = json.loads(body)
    eligible, reason = router.is_failover_eligible(request_dict)
    if not eligible:
        logger.warning("Failover ineligible: %s", reason)
        return None

    if not await router.should_use_fallback():
        return None

    result = await _forward_via_provider(fallback, body, client)
    if stats:
        stats.record_failover()
    return result


async def _auto_route(
    upstream_url: str,
    headers: dict[str, str],
    body: bytes,
    router: Router,
    stats: BridgeStats | None,
    client: httpx.AsyncClient,
) -> tuple[int, bytes, list[tuple[str, str]]]:
    """Auto mode: try Anthropic, failover on error."""
    # If circuit breaker is OPEN, try fallback first
    if await router.should_use_fallback():
        result = await _try_failover(router, body, client, stats)
        if result is not None:
            return result[0], result[1], []

    # Try Anthropic upstream
    status_code, response_body, rl_headers = await forward_request(
        client, upstream_url, body, headers
    )

    if status_code not in _FAILOVER_STATUSES:
        await router.record_success()
        return status_code, response_body, rl_headers

    # Anthropic failed — record and try failover
    await router.record_failure()
    result = await _try_failover(router, body, client, stats)
    if result is not None:
        return result[0], result[1], []

    return status_code, response_body, rl_headers


_provider_cache: dict[str, Provider] = {}


def _get_fallback_chain() -> list[str]:
    """Return the ordered list of fallback provider names."""
    return config.fallback_chain()


def _get_cached_provider(name: str) -> Provider | None:
    """Return a cached provider instance, creating it on first access."""
    if name in _provider_cache:
        return _provider_cache[name]
    provider_cls = PROVIDERS.get(name)
    if provider_cls is None:
        return None
    instance = validate_provider(provider_cls())
    _provider_cache[name] = instance
    return instance


def _get_fallback_provider() -> Provider | None:
    """Return the first available provider from the fallback chain."""
    for name in _get_fallback_chain():
        if name not in PROVIDERS:
            logger.warning("Fallback provider %r is not registered", name)
            continue
        try:
            provider = _get_cached_provider(name)
        except ValueError as exc:
            logger.error("Fallback provider %r unavailable: %s", name, exc)
            continue
        if provider is not None:
            return provider
    return None


async def _forward_via_provider(
    provider: Provider, body: bytes, client: httpx.AsyncClient
) -> tuple[int, bytes]:
    """Authenticate, translate, forward to provider, translate back.

    Providers declare whether non-streaming client requests receive provider JSON
    or provider SSE. JSON responses use ``translate_response`` directly; SSE
    responses use ``translate_stream`` and fold the translated Anthropic events into
    one Messages response so Codex/OpenAI delta-only content is preserved.
    """
    try:
        request_dict = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        return 400, anthropic_error_body(400, "Malformed JSON request")
    try:
        auth_headers = await provider.authenticate()
        translated, warnings = provider.translate_request(request_dict)
    except Exception:
        logger.exception("Provider preflight failed")
        return 502, anthropic_error_body(502, "Provider preflight failed")
    if not isinstance(translated, dict):
        logger.warning(
            "Provider %s translate_request returned %s, expected dict",
            provider.name,
            type(translated).__name__,
        )
        return 502, anthropic_error_body(502, "Provider translation failed")
    emit_translation_warnings(warnings, translated)

    # Buffer the full provider response (retries the transient connect once internally).
    status_code, raw_response = await post_provider(
        client, provider.endpoint, translated, auth_headers
    )
    logger.info("Provider response: %d (%dB)", status_code, len(raw_response))

    # A 401 means the bearer was rejected upstream — rotated or expired between our proactive
    # expiry check and this call. Force one credential refresh and retry the POST exactly once,
    # before any downstream output; the fresh bearer may succeed where the stale one failed. The
    # x-grok-req-id contextvar is unchanged across the retry, so the upstream can dedup it against
    # the first attempt. A failing refresh becomes a 502 (the 401 is not the client's to see).
    if status_code == 401:
        logger.warning("Provider returned 401; forcing a credential refresh and retrying once")
        try:
            auth_headers = await provider.authenticate(force_refresh=True)
        except Exception:
            logger.exception("Reactive credential refresh failed")
            return 502, anthropic_error_body(502, "Provider preflight failed")
        status_code, raw_response = await post_provider(
            client, provider.endpoint, translated, auth_headers
        )
        logger.info("Provider response after refresh: %d (%dB)", status_code, len(raw_response))

    if status_code != 200:
        logger.error("Provider HTTP %d: %s", status_code, provider_error_log_summary(raw_response))
        return status_code, anthropic_error_body(status_code, provider_error_message(raw_response))

    if provider_capabilities(provider).sync_response_mode == "json":
        try:
            provider_response = json.loads(raw_response)
        except (json.JSONDecodeError, ValueError):
            logger.exception("Provider JSON response did not parse")
            return 502, anthropic_error_body(502, "could not parse provider response")
        try:
            anthropic_response = provider.translate_response(provider_response)
        except Exception:
            logger.exception("Provider response translation failed")
            return 502, anthropic_error_body(502, "provider response translation failed")
        trace_provider_response(anthropic_response)
        return 200, json.dumps(anthropic_response).encode()

    # SSE-sync providers such as Codex/OpenAI return streamed deltas even for
    # non-streaming clients. Their completed output can be empty, so run the same
    # stream translation as the streaming path (which also captures reasoning
    # continuity) and fold the Anthropic SSE events into one Messages response.
    async def _single_chunk():
        yield raw_response

    try:
        events = [event async for event in provider.translate_stream(_single_chunk())]
    except Exception:
        logger.exception("Provider stream translation failed")
        return 502, anthropic_error_body(502, "provider response translation failed")

    anthropic_response = aggregate_stream_to_message(events)
    if anthropic_response is None:
        logger.error(
            "Provider stream carried no message_start: %s",
            provider_error_log_summary(raw_response),
        )
        return 502, anthropic_error_body(502, "could not parse provider response")

    trace_provider_response(anthropic_response)
    return 200, json.dumps(anthropic_response).encode()
