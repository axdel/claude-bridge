"""Async HTTP proxy server for LLM API requests — stdlib only.

Intercepts Anthropic Messages API traffic and routes it to either
the real Anthropic upstream (passthrough) or a configured provider
(direct/failover mode) with request/response translation.
"""

from __future__ import annotations

import asyncio
import json
import secrets
import time as _time
import urllib.error
import urllib.request

import claude_bridge.config as config
from claude_bridge.log import get_logger, is_trace_enabled, request_id_var, trace_event
from claude_bridge.provider import PROVIDERS, Provider
from claude_bridge.router import Router
from claude_bridge.stats import BridgeStats
from claude_bridge.stream import format_anthropic_sse

logger = get_logger("proxy")

_MAX_REQUEST_BODY = config.max_request_body()


def _get_timeout(default: int) -> int:
    """Return upstream timeout in seconds from UPSTREAM_TIMEOUT env var, or *default*."""

    def _warn_invalid(raw: str) -> None:
        logger.warning("Invalid UPSTREAM_TIMEOUT=%r, using default %ds", raw, default)

    return config.upstream_timeout(default, on_invalid=_warn_invalid)


_TRANSIENT_ERRORS = (urllib.error.URLError, TimeoutError, OSError)


def _retry_request(
    fn,
    *,
    retries: int = 1,
    backoff: float = 0.5,
) -> tuple[int, bytes]:
    """Call *fn* and retry on transient errors. Returns ``(status, body)``.

    *fn* must return ``(status, body)`` on success or raise an exception.
    HTTPError is not retried (server responded, just with an error status).
    """
    last_exc: Exception | None = None
    for attempt in range(1 + retries):
        try:
            return fn()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()
        except _TRANSIENT_ERRORS as exc:
            last_exc = exc
            if attempt < retries:
                logger.warning(
                    "Transient error (attempt %d/%d): %s", attempt + 1, retries + 1, exc
                )
                _time.sleep(backoff * (2**attempt))
    logger.error("All %d attempts failed: %s", retries + 1, last_exc)
    return 502, json.dumps({"error": "upstream unavailable"}).encode()


# Headers to forward from the client to the upstream API.
_FORWARD_HEADERS = ("x-api-key", "content-type", "anthropic-version")

# Upstream HTTP status codes that trigger failover.
_FAILOVER_STATUSES = {429, 500, 502, 503}

# SSE events too noisy for DEBUG — normal stream lifecycle, not interesting.
_RATELIMIT_HEADER_PREFIXES = ("x-ratelimit-", "anthropic-ratelimit-")
_RATELIMIT_EXACT_HEADERS = ("retry-after",)


def _extract_ratelimit_headers(headers) -> list[tuple[str, str]]:
    """Extract rate limit headers from an HTTP response headers object."""
    result = []
    for key, value in headers.items():
        lower_key = key.lower()
        is_ratelimit = any(lower_key.startswith(p) for p in _RATELIMIT_HEADER_PREFIXES)
        if is_ratelimit or lower_key in _RATELIMIT_EXACT_HEADERS:
            result.append((lower_key, value))
    return result


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
    upstream = upstream_url or config.anthropic_real_url()

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
            await _process_request(reader, writer, upstream_url, router, provider, stats)
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


# Approximate bytes-per-token ratio for mixed code/natural language traffic.
_BYTES_PER_TOKEN = 3.5


def estimate_input_tokens(request: dict) -> int:
    """Estimate input token count by walking the Anthropic request structure.

    Serializes system prompt, messages, and tool definitions to JSON, counts
    UTF-8 bytes, and divides by 3.5. Returns 0 for empty/malformed requests.
    Provider-agnostic — operates on Anthropic request format.
    """
    total_bytes = 0
    system = request.get("system")
    if system is not None:
        total_bytes += len(json.dumps(system).encode())
    for message in request.get("messages", []):
        total_bytes += len(json.dumps(message).encode())
    tools = request.get("tools")
    if tools:
        total_bytes += len(json.dumps(tools).encode())
    if total_bytes == 0:
        return 0
    return int(total_bytes / _BYTES_PER_TOKEN + 0.5)


def _estimate_tokens(body: bytes) -> int:
    """Estimate input tokens from raw request body bytes."""
    try:
        return estimate_input_tokens(json.loads(body))
    except (json.JSONDecodeError, ValueError):
        return 0


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


def _record_latency(stats: BridgeStats | None, request_start: float) -> None:
    """Record latency only (for streaming responses where tokens aren't easily available)."""
    if stats is None:
        return

    latency_ms = (_time.monotonic() - request_start) * 1000
    stats.record_response(200, latency_ms, 0, 0)


def _is_streaming(body: bytes) -> bool:
    """Return True if the request body has ``"stream": true``."""
    try:
        return json.loads(body).get("stream") is True
    except (json.JSONDecodeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Redacted compatibility trace — structural summaries + self-guarding hooks.
#
# The summarizers below are the redaction allowlist: each constructs a dict of
# explicitly named structural fields (counts, type names, tool names, lengths,
# token totals, stop reasons). They NEVER copy prompt text, tool arguments,
# tool results, reasoning payloads, request headers, or credentials into the
# trace. Redaction is enforced by construction here, not by discipline at the
# call sites. The hooks self-guard on ``is_trace_enabled()`` so the host
# functions carry zero added complexity and zero overhead when tracing is off.
# ---------------------------------------------------------------------------


def _block_type_counts(blocks: object) -> dict[str, int]:
    """Count content blocks by their ``type`` field — structure only, no content."""
    counts: dict[str, int] = {}
    if not isinstance(blocks, list):
        return counts
    for block in blocks:
        if isinstance(block, dict):
            block_type = str(block.get("type", "unknown"))
            counts[block_type] = counts.get(block_type, 0) + 1
    return counts


def _summarize_anthropic_request(request: dict) -> dict:
    """Structural-only summary of an inbound Anthropic request.

    Emits model, stream flag, message/tool counts, tool names, top-level block
    type counts, the tool_choice *type*, and the system prompt *length* — never
    any prompt text, tool argument, or tool result.
    """
    messages = request.get("messages")
    message_list = messages if isinstance(messages, list) else []
    block_types: dict[str, int] = {}
    for message in message_list:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            block_types["text"] = block_types.get("text", 0) + 1
        else:
            for block_type, count in _block_type_counts(content).items():
                block_types[block_type] = block_types.get(block_type, 0) + count
    tools = request.get("tools") or []
    tool_names = sorted(str(tool.get("name", "")) for tool in tools if isinstance(tool, dict))
    tool_choice = request.get("tool_choice")
    system = request.get("system")
    return {
        "model": str(request.get("model", "")),
        "stream": bool(request.get("stream")),
        "message_count": len(message_list),
        "system_chars": len(json.dumps(system)) if system is not None else 0,
        "block_types": block_types,
        "tool_count": len(tools),
        "tool_names": tool_names,
        "tool_choice": tool_choice.get("type") if isinstance(tool_choice, dict) else tool_choice,
    }


def _summarize_provider_request(translated: dict, warnings: list[str]) -> dict:
    """Structural-only summary of a translated provider request.

    Emits model, stream flag, input item count, tool count/names, the resolved
    tool_choice, reasoning effort, and the translation warnings — both the count
    and the sanitized warning strings, which name what was stripped — never any
    translated input content. The warning strings are neutralized at construction
    (see ``_safe_token``), so they are safe to persist to the trace.
    """
    tools = translated.get("tools") or []
    tool_names = sorted(str(tool.get("name", "")) for tool in tools if isinstance(tool, dict))
    tool_choice = translated.get("tool_choice")
    if isinstance(tool_choice, dict):
        tool_choice = f"{tool_choice.get('type')}:{tool_choice.get('name')}"
    reasoning = translated.get("reasoning")
    summary = {
        "model": str(translated.get("model", "")),
        "stream": bool(translated.get("stream")),
        "input_items": len(translated.get("input") or []),
        "tool_count": len(tools),
        "tool_names": tool_names,
        "tool_choice": tool_choice,
        "reasoning_effort": reasoning.get("effort") if isinstance(reasoning, dict) else None,
        "warning_count": len(warnings),
        "warnings": list(warnings),
    }
    if "parallel_tool_calls" in translated:
        summary["parallel_tool_calls"] = bool(translated.get("parallel_tool_calls"))
    return summary


def _summarize_anthropic_response(response: dict) -> dict:
    """Structural-only summary of an outbound Anthropic response.

    Emits model, stop_reason, block type counts, and token usage — never the
    response text or tool_use arguments.
    """
    usage = response.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    return {
        "model": str(response.get("model", "")),
        "stop_reason": response.get("stop_reason"),
        "block_types": _block_type_counts(response.get("content")),
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
    }


def _summarize_stream_event(event: dict) -> dict:
    """Structural-only summary of one translated Anthropic SSE event.

    Emits the event name, block index, block/delta *type*, stop_reason, and
    output token total — never the streamed text or partial tool-argument JSON.
    """
    data = event.get("data")
    data = data if isinstance(data, dict) else {}
    summary: dict = {"sse": event.get("event", "")}
    if "index" in data:
        summary["index"] = data.get("index")
    content_block = data.get("content_block")
    if isinstance(content_block, dict):
        summary["block_type"] = content_block.get("type")
    delta = data.get("delta")
    if isinstance(delta, dict):
        if "type" in delta:
            summary["delta_type"] = delta.get("type")
        if "stop_reason" in delta:
            summary["stop_reason"] = delta.get("stop_reason")
    usage = data.get("usage")
    if isinstance(usage, dict) and "output_tokens" in usage:
        summary["output_tokens"] = usage.get("output_tokens")
    return summary


def _trace_inbound_request(body: bytes) -> None:
    """Trace the structural shape of an inbound Anthropic request, if enabled."""
    if not is_trace_enabled():
        return
    try:
        trace_event("inbound_request", _summarize_anthropic_request(json.loads(body)))
    except Exception:
        logger.debug("inbound trace failed", exc_info=True)


def _trace_provider_request(translated: dict, warnings: list[str]) -> None:
    """Trace the structural shape of a translated provider request, if enabled."""
    if not is_trace_enabled():
        return
    try:
        trace_event("provider_request", _summarize_provider_request(translated, warnings))
    except Exception:
        logger.debug("provider request trace failed", exc_info=True)


def _emit_translation_warnings(warnings: list[str], translated: dict) -> None:
    """Surface translation warnings to every observer — the human log and the
    structural trace — from a single place.

    Both the streaming and non-streaming request paths route their warnings here so
    the logged warnings and the traced warnings can never drift out of lockstep.
    """
    for warning in warnings:
        logger.warning("Translation: %s", warning)
    _trace_provider_request(translated, warnings)


def _trace_provider_response(response: dict) -> None:
    """Trace the structural shape of a translated provider response, if enabled."""
    if not is_trace_enabled():
        return
    try:
        trace_event("provider_response", _summarize_anthropic_response(response))
    except Exception:
        logger.debug("provider response trace failed", exc_info=True)


def _trace_stream_event(event: dict) -> None:
    """Trace the structural shape of one translated SSE event, if enabled."""
    if not is_trace_enabled():
        return
    try:
        trace_event("stream_event", _summarize_stream_event(event))
    except Exception:
        logger.debug("stream event trace failed", exc_info=True)


async def _process_request(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    upstream_url: str,
    router: Router,
    provider: Provider | None = None,
    stats: BridgeStats | None = None,
) -> None:
    """Parse one HTTP request and proxy or reject it."""

    # Assign a short request ID for log correlation
    request_id_var.set(secrets.token_hex(4))
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
        _write_response(writer, 413, error_body)
        return
    if parsed is None:
        _write_response(writer, 400, b'{"error": "malformed request"}')
        return

    method, path, headers, body = parsed
    logger.info("%s %s (%dB)", method, path, len(body))

    # Strip query string for path matching (e.g. /v1/messages?beta=true → /v1/messages)
    base_path = path.split("?")[0]

    # Health check endpoint
    if base_path == "/health":
        _write_response(writer, 200, json.dumps({"status": "ok"}).encode())
        return

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
    _trace_inbound_request(body)
    if provider is not None:
        mode = "stream" if streaming else "sync"
        logger.info("-> DIRECT %s (%s) model=%s", provider.name, mode, request_model)
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
        status_code, response_body, rl_headers = await _auto_route(
            upstream_url, headers, body, router, stats
        )
        _write_response(writer, status_code, response_body, rl_headers)
        _record_sync_response(stats, request_start, status_code, response_body)


async def _try_failover(
    router: Router, body: bytes, stats: BridgeStats | None = None
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

    if not router.should_use_fallback():
        return None

    result = await _forward_via_provider(fallback, body)
    if stats:
        stats.record_failover()
    return result


async def _auto_route(
    upstream_url: str,
    headers: dict[str, str],
    body: bytes,
    router: Router,
    stats: BridgeStats | None = None,
) -> tuple[int, bytes, list[tuple[str, str]]]:
    """Auto mode: try Anthropic, failover on error."""
    # If circuit breaker is OPEN, try fallback first
    if router.should_use_fallback():
        result = await _try_failover(router, body, stats)
        if result is not None:
            return result[0], result[1], []

    # Try Anthropic upstream
    status_code, response_body, rl_headers = await asyncio.to_thread(
        _forward_request, upstream_url, body, headers
    )

    if status_code not in _FAILOVER_STATUSES:
        await router.record_success()
        return status_code, response_body, rl_headers

    # Anthropic failed — record and try failover
    await router.record_failure()
    result = await _try_failover(router, body, stats)
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


# HTTP status → Anthropic error type (docs.anthropic.com/en/api/errors). Anything
# unmapped falls back to ``api_error`` so a novel upstream status never crashes.
_ANTHROPIC_ERROR_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    413: "request_too_large",
    422: "invalid_request_error",
    429: "rate_limit_error",
    500: "api_error",
    529: "overloaded_error",
}


def _anthropic_error_body(status_code: int, message: str) -> bytes:
    """Build an Anthropic-shaped error envelope for a status code and message."""
    error_type = _ANTHROPIC_ERROR_TYPES.get(status_code, "api_error")
    return json.dumps(
        {"type": "error", "error": {"type": error_type, "message": message}}
    ).encode()


def _provider_error_message(raw_body: bytes) -> str:
    """Extract a human-readable message from an upstream provider error body.

    Understands the OpenAI ``{"error": {"message": ...}}`` shape and degrades to the
    decoded body (truncated) when the payload is not the expected JSON.
    """
    try:
        parsed = json.loads(raw_body)
    except (json.JSONDecodeError, ValueError):
        return raw_body.decode("utf-8", errors="replace")[:500]
    error = parsed.get("error") if isinstance(parsed, dict) else None
    if isinstance(error, dict):
        return error.get("message") or error.get("type") or json.dumps(error)[:500]
    if isinstance(error, str):
        return error
    if isinstance(parsed, dict) and parsed.get("message"):
        return parsed["message"]
    return raw_body.decode("utf-8", errors="replace")[:500]


def _provider_error_log_summary(raw_body: bytes) -> str:
    """Return an operator-safe provider error summary without raw body fallback."""
    try:
        parsed = json.loads(raw_body)
    except (json.JSONDecodeError, ValueError):
        return f"unparseable provider error body ({len(raw_body)}B)"
    error = parsed.get("error") if isinstance(parsed, dict) else None
    if isinstance(error, dict):
        message = error.get("message") or error.get("type")
        if message:
            return str(message)[:500]
    if isinstance(error, str):
        return error[:500]
    if isinstance(parsed, dict) and parsed.get("message"):
        return str(parsed["message"])[:500]
    return f"provider error body without message ({len(raw_body)}B)"


async def _forward_via_provider(provider: Provider, body: bytes) -> tuple[int, bytes]:
    """Authenticate, translate, forward to provider, translate back.

    The Codex endpoint always streams (even for non-streaming clients), so we read
    the whole SSE stream, translate it through the streaming path, and fold the
    Anthropic events into one Messages response — the ``response.completed`` event
    carries an empty ``output``, so the deltas are the only source of content.
    """
    request_dict = json.loads(body)
    auth_headers = await provider.authenticate()
    translated, warnings = provider.translate_request(request_dict)
    if not isinstance(translated, dict):
        logger.warning(
            "Provider %s translate_request returned %s, expected dict",
            provider.name,
            type(translated).__name__,
        )
        return 502, _anthropic_error_body(502, "Provider translation failed")
    _emit_translation_warnings(warnings, translated)

    # Open a streaming connection and collect the full response
    def _do_provider_request():
        data = json.dumps(translated).encode()
        req = urllib.request.Request(  # noqa: S310
            provider.endpoint, data=data, method="POST"
        )
        req.add_header("Content-Type", "application/json")
        for key, value in auth_headers.items():
            req.add_header(key, value)
        with urllib.request.urlopen(req, timeout=_get_timeout(120)) as resp:  # noqa: S310
            return resp.status, resp.read()

    status_code, raw_response = await asyncio.to_thread(_retry_request, _do_provider_request)
    logger.info("Provider response: %d (%dB)", status_code, len(raw_response))
    if status_code != 200:
        logger.error(
            "Provider HTTP %d: %s", status_code, _provider_error_log_summary(raw_response)
        )
        return status_code, _anthropic_error_body(
            status_code, _provider_error_message(raw_response)
        )

    # The Codex backend always streams (even for non-streaming clients) and its
    # response.completed.output is empty — the text/reasoning/tool-calls live only
    # in the delta events. So run the SAME stream translation as the streaming path
    # (which also captures reasoning continuity) and fold the Anthropic SSE events
    # into a single Messages response, rather than reading the empty completed output.
    async def _single_chunk():
        yield raw_response

    try:
        events = [event async for event in provider.translate_stream(_single_chunk())]
    except Exception:
        logger.exception("Provider stream translation failed")
        return 502, _anthropic_error_body(502, "could not parse provider response")

    anthropic_response = _aggregate_stream_to_message(events)
    if anthropic_response is None:
        logger.error("Provider stream carried no message_start: %s", raw_response[:200])
        return 502, _anthropic_error_body(502, "could not parse provider response")

    _trace_provider_response(anthropic_response)
    return 200, json.dumps(anthropic_response).encode()


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


def _aggregate_stream_to_message(events: list[dict]) -> dict | None:
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


def _forward_request(
    upstream_url: str, body: bytes, client_headers: dict[str, str]
) -> tuple[int, bytes, list[tuple[str, str]]]:
    """Synchronous HTTP POST to the upstream — called from asyncio.to_thread."""
    url = f"{upstream_url}/v1/messages"
    req = urllib.request.Request(url, data=body, method="POST")  # noqa: S310

    for key in _FORWARD_HEADERS:
        if key in client_headers:
            req.add_header(key, client_headers[key])

    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=_get_timeout(60)) as resp:  # noqa: S310
                return resp.status, resp.read(), _extract_ratelimit_headers(resp.headers)
        except urllib.error.HTTPError as exc:
            rl_headers = _extract_ratelimit_headers(exc.headers) if exc.headers else []
            return exc.code, exc.read(), rl_headers
        except _TRANSIENT_ERRORS as exc:
            last_exc = exc
            if attempt == 0:
                logger.warning("Upstream transient error, retrying: %s", exc)
                _time.sleep(0.5)
    logger.error("Upstream unavailable after retry: %s", last_exc)
    return 502, json.dumps({"error": "upstream unavailable"}).encode(), []


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
        req = urllib.request.Request(url, data=body, method="POST")  # noqa: S310
        for key in _FORWARD_HEADERS:
            if key in client_headers:
                req.add_header(key, client_headers[key])
        return urllib.request.urlopen(req, timeout=_get_timeout(120))  # noqa: S310

    try:
        resp = await asyncio.to_thread(_open_stream)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError):
        _write_response(writer, 502, json.dumps({"error": "upstream unavailable"}).encode())
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
    # Authenticate first — some providers need auth context before translation
    auth_headers = await provider.authenticate()
    translated, warnings = provider.translate_request(request_dict)
    if not isinstance(translated, dict):
        logger.warning(
            "Provider %s translate_request returned %s, expected dict",
            provider.name,
            type(translated).__name__,
        )
        _write_response(
            writer,
            502,
            json.dumps(
                {
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": "Provider translation failed",
                    },
                }
            ).encode(),
        )
        return
    _emit_translation_warnings(warnings, translated)

    # Enable streaming on the translated request (skip for providers that use URL-based streaming)
    if not getattr(provider, "stream_via_url", False):
        translated["stream"] = True

    def _open_stream():
        data = json.dumps(translated).encode()
        req = urllib.request.Request(  # noqa: S310
            provider.endpoint, data=data, method="POST"
        )
        req.add_header("Content-Type", "application/json")
        for key, value in auth_headers.items():
            req.add_header(key, value)
        return urllib.request.urlopen(req, timeout=_get_timeout(120))  # noqa: S310

    logger.debug(
        "Sending to provider: model=%s items=%d",
        translated.get("model"),
        len(translated.get("input", [])),
    )

    try:
        resp = await asyncio.to_thread(_open_stream)
    except urllib.error.HTTPError as exc:
        err_body = exc.read()
        logger.error("Provider HTTP %d: %s", exc.code, _provider_error_log_summary(err_body))
        _write_response(
            writer, exc.code, _anthropic_error_body(exc.code, _provider_error_message(err_body))
        )
        return
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        logger.error("Provider connection error: %s", exc)
        _write_response(writer, 502, _anthropic_error_body(502, "provider unavailable"))
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
            _trace_stream_event(anthropic_event)
            sse_bytes = format_anthropic_sse(anthropic_event["event"], anthropic_event["data"])
            event_name = anthropic_event["event"]
            if event_name not in _QUIET_SSE_EVENTS:
                logger.debug("SSE -> %s", event_name)
            await _safe_write(writer, sse_bytes)
    except _ClientDisconnected:
        logger.debug("Client disconnected during provider stream")
    except Exception:
        logger.exception("Unexpected error during provider stream")


def _write_response(
    writer: asyncio.StreamWriter,
    status: int,
    body: bytes,
    extra_headers: list[tuple[str, str]] | None = None,
) -> None:
    """Write a minimal HTTP/1.1 response."""
    reasons = {
        200: "OK",
        400: "Bad Request",
        404: "Not Found",
        413: "Payload Too Large",
        502: "Bad Gateway",
    }
    reason = reasons.get(status, "Error")
    writer.write(f"HTTP/1.1 {status} {reason}\r\n".encode())
    writer.write(b"Content-Type: application/json\r\n")
    writer.write(f"Content-Length: {len(body)}\r\n".encode())
    for key, value in extra_headers or []:
        writer.write(f"{key}: {value}\r\n".encode())
    writer.write(b"Connection: close\r\n")
    writer.write(b"\r\n")
    writer.write(body)
