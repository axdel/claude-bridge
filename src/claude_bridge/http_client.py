"""Low-level HTTP/2 transport for upstream and provider calls.

The proxy's orchestration (routing, streaming to the client, translation) lives in
``proxy.py``; this leaf owns the mechanics of *making the upstream call* over a shared,
event-loop-owned ``httpx.AsyncClient``: split connect/stream-idle timeouts, one
pre-body retry on transient transport errors, buffered POSTs, and streaming opens whose
status and headers are available before the body is read.

Only the DATA plane (Anthropic passthrough + provider messages/responses) runs on httpx
here. Token/OIDC refresh stays on stdlib urllib inside each provider's auth module — a
bounded, off-event-loop, SSRF-issuer-pinned control-plane POST that is not the streaming
path and is not the source of the 120s hang (D-TRANSPORT-001).
"""

from __future__ import annotations

import asyncio
import json
import random
import socket

import httpx

import claude_bridge.config as config
from claude_bridge.log import get_logger
from claude_bridge.wire import anthropic_error_body

logger = get_logger("http_client")

# Headers forwarded verbatim from the client to the Anthropic upstream on passthrough.
FORWARD_HEADERS = ("x-api-key", "content-type", "anthropic-version")

# Response header groups surfaced back to the client (rate-limit signalling).
_RATELIMIT_HEADER_PREFIXES = ("x-ratelimit-", "anthropic-ratelimit-")
_RATELIMIT_EXACT_HEADERS = ("retry-after",)

# Transient transport failures worth one retry BEFORE any downstream byte is written.
# ``httpx.TransportError`` covers connect/read/write/pool timeouts and protocol errors;
# it excludes HTTP status errors, which are returned as normal responses, never raised.
_TRANSIENT_ERRORS = (httpx.TransportError,)

_RETRY_BACKOFF_SECONDS = 0.5
# Resolver (getaddrinfo) failures get a longer pause: a short negative cache or
# mDNSResponder blip can clear in a couple of seconds; 0.5s always loses.
_RESOLVER_BACKOFF_SECONDS = 2.0

# Portable getaddrinfo failure codes — Darwin EAI_NONAME is 8, glibc is -2.
_RESOLVER_ERRNOS = frozenset(
    code
    for code in (
        getattr(socket, "EAI_NONAME", None),
        getattr(socket, "EAI_AGAIN", None),
        getattr(socket, "EAI_FAIL", None),
        getattr(socket, "EAI_NODATA", None),
    )
    if isinstance(code, int)
)
_RESOLVER_MESSAGE_MARKERS = (
    "nodename nor servname provided",  # Darwin
    "name or service not known",  # glibc EAI_NONAME
    "temporary failure in name resolution",  # glibc EAI_AGAIN
)


def _is_resolver_error(exc: BaseException) -> bool:
    """Return True when *exc* (or its cause chain) is a getaddrinfo failure.

    httpx wraps ``socket.gaierror`` in ``ConnectError``, so the classifier walks
    ``__cause__`` / ``__context__``. Errnos are the portable ``socket.EAI_*``
    constants; the Darwin log string is a fallback when the cause was dropped.
    """
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        errno = getattr(current, "errno", None)
        if errno in _RESOLVER_ERRNOS:
            return True
        message = str(current).lower()
        if any(marker in message for marker in _RESOLVER_MESSAGE_MARKERS):
            return True
        current = current.__cause__ or current.__context__
    return False


def _jittered_backoff(attempt: int = 1, *, base: float = _RETRY_BACKOFF_SECONDS) -> float:
    """Return the retry delay (seconds) for *attempt* (1-based) with decorrelating jitter.

    Base backoff scaled by the attempt number, times a random factor in [0.5, 1.5). Several
    bridge processes that trip the same transient upstream failure at once then retry at
    spread-out instants instead of waking together and colliding again — the multi-process
    deployment the cross-process refresh lock also serves. Not a hot-path optimization: it
    decorrelates the small number of concurrent retriers a local proxy can have, and never
    returns zero (the 0.5 floor keeps the backoff meaningful).
    """
    return base * attempt * (0.5 + random.random())  # noqa: S311  # nosec B311


def _backoff_for(exc: BaseException, attempt: int = 1) -> float:
    """Jittered delay: 2s base for resolver failures, 0.5s for other transients."""
    base = _RESOLVER_BACKOFF_SECONDS if _is_resolver_error(exc) else _RETRY_BACKOFF_SECONDS
    return _jittered_backoff(attempt, base=base)


async def _buffered_post_with_retry(
    client: httpx.AsyncClient,
    url: str,
    *,
    content: bytes,
    headers: dict[str, str],
    kind: str,
) -> httpx.Response | None:
    """POST *content* buffered, retrying once on a transient transport error.

    Returns the response for any HTTP status — the server answered, so an error status is
    the caller's to interpret and is never retried. Returns None when the transport fails
    on both the attempt and its one retry; the retry is safe because no downstream byte was
    ever written. *kind* labels the log lines (``"Upstream"`` / ``"Provider"``). Shared by
    the two buffered POST paths so retry, jittered backoff, and failure logging live once.
    """
    for attempt in range(2):
        try:
            return await client.post(url, content=content, headers=headers)
        except _TRANSIENT_ERRORS as exc:
            if attempt == 0:
                if _is_resolver_error(exc):
                    logger.warning("%s DNS resolver error, retrying: %s", kind, exc)
                else:
                    logger.warning("%s transient error, retrying: %s", kind, exc)
                await asyncio.sleep(_backoff_for(exc))
                continue
            if _is_resolver_error(exc):
                logger.error(
                    "%s DNS resolver failed after retry: %s; restart the bridge to recover",
                    kind,
                    exc,
                )
            else:
                logger.error("%s unavailable after retry: %s", kind, exc)
    return None


def _build_timeout() -> httpx.Timeout:
    """Build the split-phase timeout: fast connect, long per-chunk stream idle.

    A single urllib socket timeout fired on every recv and killed long-thinking
    grok-4.6 streams at ~120s. Splitting connect from read/write means a healthy stream
    runs as long as chunks keep arriving within the stream-idle window.
    """
    connect = config.connect_timeout()
    idle = config.stream_idle_timeout()
    return httpx.Timeout(connect=connect, read=idle, write=idle, pool=connect)


def create_client() -> httpx.AsyncClient:
    """Create the event-loop-owned async HTTP/2 client for all data-plane calls.

    HTTP/2 multiplexes concurrent requests over one connection; ``keepalive_expiry`` is
    raised from httpx's 5s default to the configured pool-idle so an idle keep-alive
    connection survives between turns instead of reconnecting constantly. Redirects are
    disabled — API endpoints must not redirect, and auto-following one is an
    SSRF/credential-leak vector.
    """
    return httpx.AsyncClient(
        http2=True,
        timeout=_build_timeout(),
        limits=httpx.Limits(keepalive_expiry=config.pool_idle()),
        follow_redirects=False,
    )


def select_forward_headers(client_headers: dict[str, str]) -> dict[str, str]:
    """Return only the client headers that are forwarded to the Anthropic upstream."""
    return {key: client_headers[key] for key in FORWARD_HEADERS if key in client_headers}


def _extract_ratelimit_headers(headers: httpx.Headers) -> list[tuple[str, str]]:
    """Extract rate-limit headers (lowercased) from a response headers object."""
    result: list[tuple[str, str]] = []
    for key, value in headers.items():
        lower_key = key.lower()
        is_ratelimit = any(lower_key.startswith(p) for p in _RATELIMIT_HEADER_PREFIXES)
        if is_ratelimit or lower_key in _RATELIMIT_EXACT_HEADERS:
            result.append((lower_key, value))
    return result


async def forward_request(
    client: httpx.AsyncClient,
    upstream_url: str,
    body: bytes,
    client_headers: dict[str, str],
) -> tuple[int, bytes, list[tuple[str, str]]]:
    """POST the raw Anthropic request to the passthrough upstream (buffered).

    Retries once on a transient transport error — safe because no downstream bytes have
    been written yet. An HTTP error *status* is returned as-is (never retried): the
    server responded, just unfavourably.
    """
    url = f"{upstream_url}/v1/messages"
    headers = select_forward_headers(client_headers)
    response = await _buffered_post_with_retry(
        client, url, content=body, headers=headers, kind="Upstream"
    )
    if response is None:
        return 502, anthropic_error_body(502, "upstream unavailable"), []
    return (
        response.status_code,
        response.content,
        _extract_ratelimit_headers(response.headers),
    )


async def post_provider(
    client: httpx.AsyncClient,
    endpoint: str,
    translated: dict,
    auth_headers: dict[str, str],
) -> tuple[int, bytes]:
    """POST a translated provider request and buffer the full response.

    Retries once on a transient transport error (no downstream bytes yet). An HTTP error
    status is returned as-is for the caller to translate.
    """
    headers = {"Content-Type": "application/json", **auth_headers}
    body = json.dumps(translated).encode()
    response = await _buffered_post_with_retry(
        client, endpoint, content=body, headers=headers, kind="Provider"
    )
    if response is None:
        return 502, anthropic_error_body(502, "upstream unavailable")
    return response.status_code, response.content


async def open_stream(
    client: httpx.AsyncClient,
    url: str,
    *,
    content: bytes,
    headers: dict[str, str],
    retries: int = 1,
) -> httpx.Response:
    """Open a streaming POST; return the Response with status+headers read, body not yet.

    ``client.send(request, stream=True)`` establishes the connection and reads the
    response headers (so ``status_code`` / ``headers`` are usable) WITHOUT consuming the
    body — the caller inspects the status and Content-Type, then either reads the error
    body or iterates ``aiter_bytes()``. The connect/header phase is retried once on a
    transient error because no downstream byte has been written; a partially read body is
    never retried (that would duplicate tool calls).

    The caller MUST ``await response.aclose()``.
    """
    send_headers = {"Accept": "text/event-stream", **headers}
    attempt = 0
    while True:
        try:
            request = client.build_request("POST", url, content=content, headers=send_headers)
            return await client.send(request, stream=True)
        except _TRANSIENT_ERRORS as exc:
            if attempt >= retries:
                if _is_resolver_error(exc):
                    logger.error(
                        "Stream connect DNS resolver failed after retry: %s; "
                        "restart the bridge to recover",
                        exc,
                    )
                raise
            attempt += 1
            if _is_resolver_error(exc):
                logger.warning(
                    "Stream connect DNS resolver error, retry %d/%d: %s",
                    attempt,
                    retries,
                    exc,
                )
            else:
                logger.warning(
                    "Stream connect transient error, retry %d/%d: %s", attempt, retries, exc
                )
            await asyncio.sleep(_backoff_for(exc, attempt))
