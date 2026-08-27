"""Unit tests for the httpx HTTP/2 transport leaf.

Every expected value is an oracle independent of the implementation: HTTP status
semantics, the config split-timeout defaults, and the FORWARD_HEADERS / rate-limit
contracts. Upstream behaviour is faked with ``httpx.MockTransport`` — no sockets, no
real network — so a handler can deterministically return a status, raise a transport
error, or count its own invocations to prove retry behaviour.
"""

from __future__ import annotations

import errno
import json
import logging
import socket

import httpx
import pytest

from claude_bridge import http_client
from claude_bridge.config import (
    DEFAULT_CONNECT_TIMEOUT,
    DEFAULT_STREAM_IDLE_TIMEOUT,
)


def _resolver_connect_error() -> httpx.ConnectError:
    """httpx-wrapped Darwin/POSIX getaddrinfo failure (operator errno 8).

    Oracle: ``socket.EAI_NONAME`` is the portable constant; Darwin renders it as
    errno 8 with the exact log string the operator pasted. httpx wraps the
    ``gaierror`` in ``ConnectError``, so the cause chain is the production shape.
    """
    cause = socket.gaierror(socket.EAI_NONAME, "nodename nor servname provided, or not known")
    exc = httpx.ConnectError(str(cause))
    exc.__cause__ = cause
    return exc


def _refused_connect_error() -> httpx.ConnectError:
    """A connect failure that is NOT a DNS resolver problem."""
    cause = OSError(errno.ECONNREFUSED, "Connection refused")
    exc = httpx.ConnectError("refused")
    exc.__cause__ = cause
    return exc


class _LogCapture(logging.Handler):
    """Capture WARNING+ records on ``claude_bridge.http_client``.

    The bridge logger sets ``propagate=False`` once configured, so pytest's
    ``caplog`` never sees these records; a handler on the named logger is the
    oracle path (same pattern as ``test_proxy._WarningCapture``).
    """

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    @property
    def messages(self) -> list[str]:
        return [record.getMessage() for record in self.records]


def _client_with(handler) -> httpx.AsyncClient:
    """Build an AsyncClient whose transport is the given mock request handler."""
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# --- _build_timeout / create_client -------------------------------------------------


def test_build_timeout_splits_connect_from_stream_idle() -> None:
    """The timeout separates a fast connect from a long per-chunk stream idle."""
    timeout = http_client._build_timeout()
    assert timeout.connect == DEFAULT_CONNECT_TIMEOUT
    assert timeout.read == DEFAULT_STREAM_IDLE_TIMEOUT
    assert timeout.write == DEFAULT_STREAM_IDLE_TIMEOUT
    assert timeout.pool == DEFAULT_CONNECT_TIMEOUT


def test_build_timeout_honours_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit CONNECT_TIMEOUT / STREAM_IDLE_TIMEOUT env values flow into the timeout."""
    monkeypatch.setenv("CONNECT_TIMEOUT", "3")
    monkeypatch.setenv("STREAM_IDLE_TIMEOUT", "600")
    timeout = http_client._build_timeout()
    assert timeout.connect == 3.0
    assert timeout.read == 600.0


async def test_create_client_disables_redirects_with_split_timeout() -> None:
    """The shared client refuses redirects (SSRF guard) and carries the split timeout."""
    client = http_client.create_client()
    try:
        assert isinstance(client, httpx.AsyncClient)
        assert client.follow_redirects is False
        assert client.timeout.connect == DEFAULT_CONNECT_TIMEOUT
        assert client.timeout.read == DEFAULT_STREAM_IDLE_TIMEOUT
    finally:
        await client.aclose()


# --- select_forward_headers ---------------------------------------------------------


def test_select_forward_headers_keeps_only_the_forwarded_set() -> None:
    """Only x-api-key / content-type / anthropic-version cross to the upstream."""
    selected = http_client.select_forward_headers(
        {
            "x-api-key": "secret",
            "content-type": "application/json",
            "anthropic-version": "2023-06-01",
            "authorization": "leak",
            "x-internal": "drop",
        }
    )
    assert selected == {
        "x-api-key": "secret",
        "content-type": "application/json",
        "anthropic-version": "2023-06-01",
    }


def test_select_forward_headers_omits_absent_headers() -> None:
    """Headers not present on the request are simply absent, never None-filled."""
    assert http_client.select_forward_headers({"x-api-key": "k"}) == {"x-api-key": "k"}


# --- _extract_ratelimit_headers -----------------------------------------------------


def test_extract_ratelimit_headers_matches_prefixes_and_retry_after() -> None:
    """Prefix-matched rate-limit headers plus retry-after are surfaced, lowercased."""
    headers = httpx.Headers(
        {
            "X-RateLimit-Remaining": "5",
            "anthropic-ratelimit-requests-limit": "100",
            "Retry-After": "30",
            "Content-Type": "application/json",
        }
    )
    extracted = dict(http_client._extract_ratelimit_headers(headers))
    assert extracted == {
        "x-ratelimit-remaining": "5",
        "anthropic-ratelimit-requests-limit": "100",
        "retry-after": "30",
    }


# --- forward_request ----------------------------------------------------------------


async def test_forward_request_returns_status_body_and_ratelimit() -> None:
    """A 200 upstream yields its status, body, and rate-limit headers to the caller."""
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        return httpx.Response(200, content=b'{"ok":true}', headers={"x-ratelimit-remaining": "7"})

    client = _client_with(handler)
    try:
        status, body, ratelimit = await http_client.forward_request(
            client,
            "https://api.anthropic.com",
            b'{"model":"claude"}',
            {"x-api-key": "k", "authorization": "should-not-forward"},
        )
    finally:
        await client.aclose()

    assert status == 200
    assert body == b'{"ok":true}'
    assert ratelimit == [("x-ratelimit-remaining", "7")]
    assert seen["url"] == "https://api.anthropic.com/v1/messages"
    assert seen["headers"]["x-api-key"] == "k"  # type: ignore[index]
    assert "authorization" not in seen["headers"]  # type: ignore[operator]


async def test_forward_request_does_not_retry_error_status() -> None:
    """An HTTP 429 is a real answer — returned as-is, the upstream hit exactly once."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(429, content=b"slow down")

    client = _client_with(handler)
    try:
        status, body, ratelimit = await http_client.forward_request(
            client, "https://up", b"{}", {}
        )
    finally:
        await client.aclose()

    assert status == 429
    assert body == b"slow down"
    assert ratelimit == []
    assert calls["n"] == 1


async def test_forward_request_retries_once_then_succeeds() -> None:
    """A single transient transport error is retried and the second attempt returns."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("refused")
        return httpx.Response(200, content=b"ok")

    client = _client_with(handler)
    try:
        status, body, _ = await http_client.forward_request(client, "https://up", b"{}", {})
    finally:
        await client.aclose()

    assert status == 200
    assert body == b"ok"
    assert calls["n"] == 2


async def test_forward_request_returns_502_after_retry_exhausted() -> None:
    """Two transport failures exhaust the single retry and yield a synthetic 502."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ConnectError("down")

    client = _client_with(handler)
    try:
        status, body, ratelimit = await http_client.forward_request(
            client, "https://up", b"{}", {}
        )
    finally:
        await client.aclose()

    assert status == 502
    assert json.loads(body) == {
        "type": "error",
        "error": {"type": "api_error", "message": "upstream unavailable"},
    }
    assert ratelimit == []
    assert calls["n"] == 2


# --- post_provider ------------------------------------------------------------------


async def test_post_provider_returns_status_and_body() -> None:
    """A translated provider POST carries JSON content-type + auth and returns the body."""
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["headers"] = dict(request.headers)
        seen["content"] = request.content
        return httpx.Response(200, content=b'{"id":"resp_1"}')

    client = _client_with(handler)
    try:
        status, body = await http_client.post_provider(
            client,
            "https://provider/v1/responses",
            {"model": "grok-4.6", "input": []},
            {"Authorization": "Bearer t"},
        )
    finally:
        await client.aclose()

    assert status == 200
    assert body == b'{"id":"resp_1"}'
    assert seen["headers"]["content-type"] == "application/json"  # type: ignore[index]
    assert seen["headers"]["authorization"] == "Bearer t"  # type: ignore[index]
    assert json.loads(seen["content"]) == {"model": "grok-4.6", "input": []}  # type: ignore[arg-type]


async def test_post_provider_does_not_retry_error_status() -> None:
    """A 400 from the provider is returned unchanged and never retried."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, content=b"bad request")

    client = _client_with(handler)
    try:
        status, body = await http_client.post_provider(client, "https://p", {}, {})
    finally:
        await client.aclose()

    assert status == 400
    assert body == b"bad request"
    assert calls["n"] == 1


async def test_post_provider_returns_502_after_retry_exhausted() -> None:
    """Provider transport failure past the single retry yields a synthetic 502."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ConnectError("down")

    client = _client_with(handler)
    try:
        status, body = await http_client.post_provider(client, "https://p", {}, {})
    finally:
        await client.aclose()

    assert status == 502
    assert json.loads(body) == {
        "type": "error",
        "error": {"type": "api_error", "message": "upstream unavailable"},
    }
    assert calls["n"] == 2


# --- resolver-error classification (EAI_NONAME wedge) -------------------------------


def test_is_resolver_error_matches_eai_noname_cause() -> None:
    """A ConnectError wrapping socket.EAI_NONAME is a resolver failure.

    Oracle is the POSIX constant, not Darwin's errno 8: Linux uses -2 for the
    same name. The classifier must walk ``__cause__`` because httpx wraps the
    gaierror (the production log is ``ConnectError: [Errno 8] nodename...``).
    """
    assert http_client._is_resolver_error(_resolver_connect_error()) is True


def test_is_resolver_error_matches_message_without_cause() -> None:
    """The Darwin log string is classified even when httpx drops the gaierror cause."""
    exc = httpx.ConnectError("[Errno 8] nodename nor servname provided, or not known")
    assert http_client._is_resolver_error(exc) is True


def test_is_resolver_error_matches_eai_errno_without_message_marker() -> None:
    """Portable ``EAI_NONAME`` is classified even when the message is opaque.

    The production Darwin string also matches the message fallback, which would
    let an errno-check deletion survive. An opaque gaierror message forces the
    ``socket.EAI_*`` membership path — the Linux/portable half of the classifier.
    """
    cause = socket.gaierror(socket.EAI_NONAME, "unrelated opaque resolver failure")
    exc = httpx.ConnectError("connect failed")
    exc.__cause__ = cause
    assert http_client._is_resolver_error(exc) is True


def test_is_resolver_error_rejects_connection_refused() -> None:
    """ECONNREFUSED is a transport failure, not a resolver wedge."""
    assert http_client._is_resolver_error(_refused_connect_error()) is False


async def test_post_provider_resolver_error_uses_long_backoff_and_restart_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """EAI_NONAME spends the retry at ~2s and tells the operator to restart the bridge.

    Frozen ``random.random() = 0.5`` makes jitter a 1.0 multiplier, so the
    resolver base (2.0s) yields a 2.0s sleep — distinct from the ordinary 0.5s
    transient backoff. The ERROR line is the recovery instruction: a wedged
    libinfo/mDNSResponder client is process-local and only a restart clears it.
    """
    monkeypatch.setattr(http_client.random, "random", lambda: 0.5)
    slept: list[float] = []

    async def _fake_sleep(delay: float) -> None:
        slept.append(delay)

    monkeypatch.setattr(http_client.asyncio, "sleep", _fake_sleep)

    def handler(request: httpx.Request) -> httpx.Response:
        raise _resolver_connect_error()

    cap = _LogCapture()
    logger = logging.getLogger("claude_bridge.http_client")
    logger.addHandler(cap)
    logger.setLevel(logging.WARNING)
    client = _client_with(handler)
    try:
        status, body = await http_client.post_provider(client, "https://p", {}, {})
    finally:
        logger.removeHandler(cap)
        await client.aclose()

    assert status == 502
    assert json.loads(body)["error"]["message"] == "upstream unavailable"
    assert slept == [2.0]
    assert any("restart the bridge" in message for message in cap.messages)


async def test_post_provider_ordinary_connect_error_keeps_short_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Connection-refused stays on the 0.5s transient path and does not mention restart."""
    monkeypatch.setattr(http_client.random, "random", lambda: 0.5)
    slept: list[float] = []

    async def _fake_sleep(delay: float) -> None:
        slept.append(delay)

    monkeypatch.setattr(http_client.asyncio, "sleep", _fake_sleep)

    def handler(request: httpx.Request) -> httpx.Response:
        raise _refused_connect_error()

    cap = _LogCapture()
    logger = logging.getLogger("claude_bridge.http_client")
    logger.addHandler(cap)
    logger.setLevel(logging.WARNING)
    client = _client_with(handler)
    try:
        status, _body = await http_client.post_provider(client, "https://p", {}, {})
    finally:
        logger.removeHandler(cap)
        await client.aclose()

    assert status == 502
    assert slept == [0.5]
    assert any("unavailable after retry" in message for message in cap.messages)
    assert not any("restart the bridge" in message for message in cap.messages)


async def test_open_stream_resolver_error_logs_restart_after_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stream connect EAI_NONAME also names the restart recovery after the retry is spent."""
    monkeypatch.setattr(http_client.random, "random", lambda: 0.5)

    async def _fake_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr(http_client.asyncio, "sleep", _fake_sleep)

    def handler(request: httpx.Request) -> httpx.Response:
        raise _resolver_connect_error()

    cap = _LogCapture()
    logger = logging.getLogger("claude_bridge.http_client")
    logger.addHandler(cap)
    logger.setLevel(logging.WARNING)
    client = _client_with(handler)
    try:
        with pytest.raises(httpx.ConnectError):
            await http_client.open_stream(client, "https://up", content=b"{}", headers={})
    finally:
        logger.removeHandler(cap)
        await client.aclose()

    assert any("restart the bridge" in message for message in cap.messages)


# --- open_stream --------------------------------------------------------------------


async def test_open_stream_exposes_status_and_headers_before_body() -> None:
    """open_stream returns a response whose status/headers are readable pre-body, with
    Accept: text/event-stream sent upstream."""
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["accept"] = request.headers.get("accept")
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b"data: {}\n\n",
        )

    client = _client_with(handler)
    try:
        response = await http_client.open_stream(
            client, "https://up/v1/messages", content=b"{}", headers={"x-api-key": "k"}
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream"
        assert seen["accept"] == "text/event-stream"
        await response.aclose()
    finally:
        await client.aclose()


async def test_open_stream_retries_connect_error_then_succeeds() -> None:
    """A transient connect error before any body is retried once, then opens."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("refused stream")
        return httpx.Response(200, content=b"data: {}\n\n")

    client = _client_with(handler)
    try:
        response = await http_client.open_stream(client, "https://up", content=b"{}", headers={})
        assert response.status_code == 200
        assert calls["n"] == 2
        await response.aclose()
    finally:
        await client.aclose()


async def test_open_stream_raises_after_retries_exhausted() -> None:
    """When the connect keeps failing, the transport error propagates (no synthetic body
    is possible mid-stream) after the retry budget is spent."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ConnectError("persistently down")

    client = _client_with(handler)
    try:
        with pytest.raises(httpx.ConnectError):
            await http_client.open_stream(client, "https://up", content=b"{}", headers={})
        assert calls["n"] == 2
    finally:
        await client.aclose()


async def test_open_stream_does_not_retry_error_status() -> None:
    """An HTTP 500 is a real response, not a transport error — returned once, no retry."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(500, content=b"upstream boom")

    client = _client_with(handler)
    try:
        response = await http_client.open_stream(client, "https://up", content=b"{}", headers={})
        assert response.status_code == 500
        assert calls["n"] == 1
        await response.aclose()
    finally:
        await client.aclose()
