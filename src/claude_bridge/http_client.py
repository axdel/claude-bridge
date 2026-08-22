"""Low-level HTTP transport for upstream and provider calls — stdlib only.

The proxy's orchestration (routing, streaming to the client, translation) is kept
in ``proxy.py``; this leaf owns the mechanics of *making the upstream call*: the
timeout resolution, transient-error retry, provider request construction, the
buffered upstream POST, and rate-limit header extraction. Isolating the transport
behind a stable surface lets the underlying HTTP client be swapped without touching
the orchestration that calls it.
"""

from __future__ import annotations

import json
import time as _time
import urllib.error
import urllib.request

import claude_bridge.config as config
from claude_bridge.log import get_logger

logger = get_logger("http_client")


def get_timeout(default: int) -> int:
    """Return upstream timeout in seconds from UPSTREAM_TIMEOUT env var, or *default*."""

    def _warn_invalid(raw: str) -> None:
        logger.warning("Invalid UPSTREAM_TIMEOUT=%r, using default %ds", raw, default)

    return config.upstream_timeout(default, on_invalid=_warn_invalid)


_TRANSIENT_ERRORS = (urllib.error.URLError, TimeoutError, OSError)


def retry_request(
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
FORWARD_HEADERS = ("x-api-key", "content-type", "anthropic-version")

# Response header groups worth surfacing back to the client (rate-limit signalling).
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


def build_provider_request(
    endpoint: str, translated: dict, auth_headers: dict
) -> urllib.request.Request:
    """Build the POST request for a provider call: JSON body, content type, auth headers.

    Shared by the buffered (``_forward_via_provider``) and streaming
    (``_stream_via_provider``) paths, which diverge only in how they open the
    returned request — read-and-close versus keep-open for incremental reads.
    """
    data = json.dumps(translated).encode()
    req = urllib.request.Request(endpoint, data=data, method="POST")  # noqa: S310
    req.add_header("Content-Type", "application/json")
    for key, value in auth_headers.items():
        req.add_header(key, value)
    return req


def forward_request(
    upstream_url: str, body: bytes, client_headers: dict[str, str]
) -> tuple[int, bytes, list[tuple[str, str]]]:
    """Synchronous HTTP POST to the upstream — called from asyncio.to_thread."""
    url = f"{upstream_url}/v1/messages"
    req = urllib.request.Request(url, data=body, method="POST")  # noqa: S310

    for key in FORWARD_HEADERS:
        if key in client_headers:
            req.add_header(key, client_headers[key])

    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=get_timeout(60)) as resp:  # noqa: S310  # nosec B310
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
