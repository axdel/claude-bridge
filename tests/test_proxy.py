"""Tests for the Claude Bridge proxy server."""

from __future__ import annotations

import asyncio
import io
import json
import socket
import urllib.error
import urllib.request
from collections.abc import AsyncIterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

import pytest

from claude_bridge.provider import PROVIDERS
from claude_bridge.proxy import start_proxy


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _MockUpstreamHandler(BaseHTTPRequestHandler):
    """Echoes back a canned Anthropic response."""

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        _body = self.rfile.read(length)
        resp = {
            "id": "msg_test",
            "type": "message",
            "content": [{"type": "text", "text": "hello from upstream"}],
            "model": "claude-sonnet-4-20250514",
        }
        payload = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        pass


@pytest.fixture()
def upstream_url():
    port = _find_free_port()
    server = HTTPServer(("127.0.0.1", port), _MockUpstreamHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


@pytest.fixture()
async def proxy_url(upstream_url: str):
    port = _find_free_port()
    server = await start_proxy(host="127.0.0.1", port=port, upstream_url=upstream_url)
    yield f"http://127.0.0.1:{port}"
    server.close()
    await server.wait_closed()


def _http_post(url: str, body: dict, headers: dict | None = None) -> tuple[int, dict]:
    """Stdlib HTTP POST helper — returns (status_code, json_body)."""
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _http_get(url: str) -> int:
    """Stdlib HTTP GET — returns status code only."""
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status
    except urllib.error.HTTPError as exc:
        return exc.code


@pytest.mark.asyncio
async def test_passthrough_forwards_to_upstream(proxy_url: str):
    """POST /v1/messages forwards to upstream and returns the response."""
    request_body = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
    }
    status, data = await asyncio.to_thread(
        _http_post,
        f"{proxy_url}/v1/messages",
        request_body,
        {"x-api-key": "test-key"},
    )
    assert status == 200
    assert data["type"] == "message"
    assert data["content"][0]["text"] == "hello from upstream"


@pytest.mark.asyncio
async def test_health_endpoint_returns_ok(proxy_url: str):
    """/health returns 200 with status ok."""
    status, data = await asyncio.to_thread(
        _http_post,
        f"{proxy_url}/health",
        {},
    )
    assert status == 200
    assert data == {"status": "ok"}


@pytest.mark.asyncio
async def test_stats_endpoint_returns_metrics(proxy_url: str):
    """GET /stats returns JSON with request metrics after a request."""
    # Make a request first so stats are non-zero
    request_body = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
    }
    await asyncio.to_thread(
        _http_post,
        f"{proxy_url}/v1/messages",
        request_body,
        {"x-api-key": "test-key"},
    )
    # Check stats (POST works — endpoint accepts any method)
    status, stats = await asyncio.to_thread(_http_post, f"{proxy_url}/stats", {})
    assert status == 200
    assert stats["requests_total"] >= 1
    assert "started_at" in stats
    assert "uptime_seconds" in stats
    assert stats["tokens_in"] >= 0


@pytest.mark.asyncio
async def test_wrong_path_returns_404(proxy_url: str):
    """GET to an unknown path returns 404."""
    status = await asyncio.to_thread(_http_get, f"{proxy_url}/v1/completions")
    assert status == 404


@pytest.mark.asyncio
async def test_malformed_content_length_returns_400(upstream_url: str):
    """Malformed Content-Length header returns 400 instead of crashing."""
    port = _find_free_port()
    server = await start_proxy(host="127.0.0.1", port=port, upstream_url=upstream_url)
    try:
        # Send a raw HTTP request with a bad Content-Length
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(b"POST /v1/messages HTTP/1.1\r\nContent-Length: not-a-number\r\n\r\n")
        await writer.drain()
        response = await asyncio.wait_for(reader.read(4096), timeout=2)
        writer.close()
        assert b"400" in response
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_upstream_unreachable_returns_502():
    """When upstream is unreachable, proxy returns 502."""
    port = _find_free_port()
    dead_upstream = f"http://127.0.0.1:{_find_free_port()}"
    server = await start_proxy(host="127.0.0.1", port=port, upstream_url=dead_upstream)
    try:
        status, data = await asyncio.to_thread(
            _http_post,
            f"http://127.0.0.1:{port}/v1/messages",
            {"model": "test", "messages": []},
        )
        assert status == 502
        assert "upstream unavailable" in data["error"]
    finally:
        server.close()
        await server.wait_closed()


# ---------------------------------------------------------------------------
# Failover + direct-mode tests
# ---------------------------------------------------------------------------


class _Mock500UpstreamHandler(BaseHTTPRequestHandler):
    """Returns 500 for all requests (simulates Anthropic outage)."""

    def do_POST(self):
        payload = json.dumps({"error": "internal server error"}).encode()
        self.send_response(500)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        pass


def _sse_bytes(events: list[tuple[str, dict]]) -> bytes:
    """Serialize ``(event_name, data)`` pairs as a provider SSE byte stream.

    Each data payload carries a ``"type"`` mirroring the event name, matching the
    real OpenAI Responses wire format (the type appears both on the ``event:``
    line and inside ``data``).
    """
    parts = [
        f"event: {name}\ndata: {json.dumps({'type': name, **data})}\n\n" for name, data in events
    ]
    return "".join(parts).encode()


def _codex_text_sse(
    text: str,
    *,
    resp_id: str = "resp_test123",
    model: str = "gpt-5.5",
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> bytes:
    """A faithful Codex Responses SSE stream for a text turn.

    Mirrors the real Codex backend: text arrives via ``output_text.delta`` events
    and ``response.completed.output`` is EMPTY — the bug that broke the
    non-streaming path. Splits ``text`` across two deltas so concatenation is
    exercised.
    """
    mid = len(text) // 2
    return _sse_bytes(
        [
            (
                "response.created",
                {
                    "response": {
                        "id": resp_id,
                        "model": model,
                        "usage": {"input_tokens": input_tokens, "output_tokens": 0},
                    }
                },
            ),
            ("response.content_part.added", {"content_index": 0}),
            ("response.output_text.delta", {"content_index": 0, "delta": text[:mid]}),
            ("response.output_text.delta", {"content_index": 0, "delta": text[mid:]}),
            ("response.output_text.done", {"content_index": 0}),
            (
                "response.completed",
                {
                    "response": {
                        "id": resp_id,
                        "status": "completed",
                        "model": model,
                        "output": [],
                        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
                    }
                },
            ),
        ]
    )


def _codex_tool_sse(
    *,
    call_id: str = "call_weather1",
    name: str = "get_weather",
    arg_fragments: tuple[str, ...] = ('{"city":', '"NYC"}'),
    resp_id: str = "resp_tool1",
    model: str = "gpt-5.5",
) -> bytes:
    """A faithful Codex Responses SSE stream for a tool-call turn.

    The function call is delivered entirely via stream deltas with an EMPTY
    ``response.completed.output`` — so ``tool_use`` stop_reason must come from the
    streamed ``content_block_start``, not the completed output.
    """
    events: list[tuple[str, dict]] = [
        (
            "response.created",
            {
                "response": {
                    "id": resp_id,
                    "model": model,
                    "usage": {"input_tokens": 12, "output_tokens": 0},
                }
            },
        ),
        (
            "response.output_item.added",
            {
                "output_index": 0,
                "item": {"type": "function_call", "call_id": call_id, "name": name},
            },
        ),
    ]
    events += [
        ("response.function_call_arguments.delta", {"output_index": 0, "delta": fragment})
        for fragment in arg_fragments
    ]
    events += [
        ("response.function_call_arguments.done", {"output_index": 0}),
        (
            "response.completed",
            {
                "response": {
                    "id": resp_id,
                    "status": "completed",
                    "model": model,
                    "output": [],
                    "usage": {"input_tokens": 12, "output_tokens": 8},
                }
            },
        ),
    ]
    return _sse_bytes(events)


class _MockOpenAIHandler(BaseHTTPRequestHandler):
    """Returns a faithful Codex Responses SSE stream (empty completed output)."""

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        _body = self.rfile.read(length)
        payload = _codex_text_sse("hello from openai fallback")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        pass


class _FakeOpenAIProvider:
    """Test provider that talks to a local mock OpenAI server."""

    name = "openai"

    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint

    async def authenticate(self) -> dict[str, str]:
        return {"Authorization": "Bearer fake-token"}

    def translate_request(self, anthropic_req: dict) -> tuple[dict, list[str]]:
        from claude_bridge.providers.openai import anthropic_to_openai

        return anthropic_to_openai(anthropic_req)

    def translate_response(self, provider_resp: dict) -> dict:
        from claude_bridge.providers.openai import openai_to_anthropic

        return openai_to_anthropic(provider_resp)

    async def translate_stream(self, raw_chunks):
        from claude_bridge.providers.openai import OpenAIProvider

        provider = OpenAIProvider()
        async for event in provider.translate_stream(raw_chunks):
            yield event


@pytest.fixture()
def _openai_mock_url():
    """Spin up a mock OpenAI endpoint and register a fake provider."""
    port = _find_free_port()
    server = HTTPServer(("127.0.0.1", port), _MockOpenAIHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{port}"

    # Register a fake provider class that points at the mock
    class _TestProvider(_FakeOpenAIProvider):
        def __init__(self):
            super().__init__(endpoint=f"{url}/v1/responses")

    old = PROVIDERS.get("openai")
    PROVIDERS["openai"] = _TestProvider
    # Clear the provider cache so _get_fallback_provider() picks up the mock
    from claude_bridge.proxy import _provider_cache

    _provider_cache.pop("openai", None)
    yield url
    _provider_cache.pop("openai", None)
    if old is not None:
        PROVIDERS["openai"] = old
    else:
        PROVIDERS.pop("openai", None)
    server.shutdown()


@pytest.mark.asyncio
async def test_failover_on_upstream_500(_openai_mock_url: str):
    """When Anthropic returns 500, proxy fails over to the fallback provider."""
    # Start a mock Anthropic that always returns 500
    anthropic_port = _find_free_port()
    anthropic_server = HTTPServer(("127.0.0.1", anthropic_port), _Mock500UpstreamHandler)
    anthropic_thread = Thread(target=anthropic_server.serve_forever, daemon=True)
    anthropic_thread.start()
    anthropic_url = f"http://127.0.0.1:{anthropic_port}"

    proxy_port = _find_free_port()
    server = await start_proxy(host="127.0.0.1", port=proxy_port, upstream_url=anthropic_url)
    try:
        request_body = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}],
        }
        # First request: Anthropic fails with 500 — records failure #1,
        # but circuit is still CLOSED (threshold=2), so returns 500
        status1, _data1 = await asyncio.to_thread(
            _http_post,
            f"http://127.0.0.1:{proxy_port}/v1/messages",
            request_body,
        )
        assert status1 == 500

        # Second request: Anthropic fails again — records failure #2,
        # circuit trips to OPEN, should_use_fallback() returns True → failover
        status2, data2 = await asyncio.to_thread(
            _http_post,
            f"http://127.0.0.1:{proxy_port}/v1/messages",
            request_body,
        )
        assert status2 == 200
        assert data2["type"] == "message"
        assert data2["content"][0]["text"] == "hello from openai fallback"
        assert data2["id"].startswith("msg_bridge_")
    finally:
        server.close()
        await server.wait_closed()
        anthropic_server.shutdown()


@pytest.mark.asyncio
async def test_direct_mode_skips_anthropic(_openai_mock_url: str):
    """With provider_name='openai', proxy never contacts Anthropic."""
    proxy_port = _find_free_port()
    # Use a dead upstream URL — if the proxy tries Anthropic, it will get 502
    dead_upstream = f"http://127.0.0.1:{_find_free_port()}"
    server = await start_proxy(
        host="127.0.0.1",
        port=proxy_port,
        upstream_url=dead_upstream,
        provider_name="openai",
    )
    try:
        request_body = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}],
        }
        status, data = await asyncio.to_thread(
            _http_post,
            f"http://127.0.0.1:{proxy_port}/v1/messages",
            request_body,
        )
        # Should get 200 from the OpenAI mock, translated back to Anthropic format
        assert status == 200
        assert data["type"] == "message"
        assert data["content"][0]["text"] == "hello from openai fallback"
        assert data["id"].startswith("msg_bridge_")
    finally:
        server.close()
        await server.wait_closed()


# ---------------------------------------------------------------------------
# Provider cache + configurable fallback tests
# ---------------------------------------------------------------------------


def test_provider_cache_returns_same_instance():
    """Cached provider instances are reused, not re-created."""
    from claude_bridge.proxy import _get_cached_provider, _provider_cache

    _provider_cache.clear()
    p1 = _get_cached_provider("openai")
    p2 = _get_cached_provider("openai")
    assert p1 is p2
    _provider_cache.clear()


def test_provider_cache_unknown_returns_none():
    """Unknown provider name returns None from cache."""
    from claude_bridge.proxy import _get_cached_provider

    assert _get_cached_provider("nonexistent") is None


@pytest.mark.asyncio
async def test_xai_stub_is_not_registered_for_cache_fallback_or_direct_mode(monkeypatch):
    """The discoverable xAI stub is not a routable runtime provider."""
    import importlib

    from claude_bridge.proxy import _get_cached_provider, _get_fallback_provider, _provider_cache

    PROVIDERS.pop("xai", None)
    _provider_cache.pop("xai", None)
    monkeypatch.setenv("LLM_BRIDGE_FALLBACK", "xai")

    try:
        xai_module = importlib.import_module("claude_bridge.providers.xai")
        importlib.reload(xai_module)
        assert "xai" not in PROVIDERS
        assert _get_cached_provider("xai") is None
        assert _get_fallback_provider() is None
        with pytest.raises(ValueError, match=r"Unknown provider 'xai'\. Available:"):
            await start_proxy(host="127.0.0.1", port=_find_free_port(), provider_name="xai")
    finally:
        PROVIDERS.pop("xai", None)
        _provider_cache.pop("xai", None)


def test_fallback_chain_from_env(monkeypatch):
    """LLM_BRIDGE_FALLBACK env var controls fallback order."""
    from claude_bridge.proxy import _get_fallback_chain

    monkeypatch.setenv("LLM_BRIDGE_FALLBACK", "openai,xai")
    chain = _get_fallback_chain()
    assert chain == ["openai", "xai"]


def test_fallback_chain_default(monkeypatch):
    """Without env var, fallback defaults to ['openai']."""
    from claude_bridge.proxy import _get_fallback_chain

    monkeypatch.delenv("LLM_BRIDGE_FALLBACK", raising=False)
    chain = _get_fallback_chain()
    assert chain == ["openai"]


def test_fallback_chain_empty_string(monkeypatch):
    """Empty LLM_BRIDGE_FALLBACK means no fallback available."""
    from claude_bridge.proxy import _get_fallback_chain

    monkeypatch.setenv("LLM_BRIDGE_FALLBACK", "")
    chain = _get_fallback_chain()
    assert chain == []


# ---------------------------------------------------------------------------
# Configurable timeout tests
# ---------------------------------------------------------------------------


def test_get_timeout_returns_default_when_unset(monkeypatch):
    """Without UPSTREAM_TIMEOUT env var, _get_timeout returns the provided default."""
    from claude_bridge.proxy import _get_timeout

    monkeypatch.delenv("UPSTREAM_TIMEOUT", raising=False)
    assert _get_timeout(60) == 60
    assert _get_timeout(120) == 120


def test_get_timeout_reads_env_var(monkeypatch):
    """UPSTREAM_TIMEOUT overrides the default for all callsites."""
    from claude_bridge.proxy import _get_timeout

    monkeypatch.setenv("UPSTREAM_TIMEOUT", "30")
    assert _get_timeout(60) == 30
    assert _get_timeout(120) == 30


def test_get_timeout_ignores_invalid_env_var(monkeypatch):
    """Non-numeric UPSTREAM_TIMEOUT falls back to default."""
    from claude_bridge.proxy import _get_timeout

    monkeypatch.setenv("UPSTREAM_TIMEOUT", "not-a-number")
    assert _get_timeout(120) == 120


def test_get_timeout_ignores_zero_and_negative(monkeypatch):
    """Zero or negative UPSTREAM_TIMEOUT falls back to default."""
    from claude_bridge.proxy import _get_timeout

    monkeypatch.setenv("UPSTREAM_TIMEOUT", "0")
    assert _get_timeout(120) == 120

    monkeypatch.setenv("UPSTREAM_TIMEOUT", "-5")
    assert _get_timeout(60) == 60


# ---------------------------------------------------------------------------
# Request body size limit tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_oversized_body_returns_413(upstream_url: str, monkeypatch):
    """Request with Content-Length exceeding MAX_REQUEST_BODY returns 413."""
    import claude_bridge.proxy as proxy_mod

    monkeypatch.setattr(proxy_mod, "_MAX_REQUEST_BODY", 100)

    port = _find_free_port()
    server = await start_proxy(host="127.0.0.1", port=port, upstream_url=upstream_url)
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(b"POST /v1/messages HTTP/1.1\r\nContent-Length: 200\r\n\r\n")
        await writer.drain()
        # Read until EOF — server sends Connection: close
        response = await asyncio.wait_for(reader.read(-1), timeout=2)
        writer.close()
        assert b"413" in response
        assert b"request_too_large" in response
    finally:
        server.close()
        await server.wait_closed()


# ---------------------------------------------------------------------------
# translate_request() validation tests
# ---------------------------------------------------------------------------


class _BrokenProvider:
    """Provider whose translate_request returns None (simulates a bug)."""

    name = "broken"
    endpoint = "http://127.0.0.1:1/unused"

    async def authenticate(self) -> dict[str, str]:
        return {}

    def translate_request(self, anthropic_req: dict) -> tuple[dict, list[str]]:
        # Deliberately violates the protocol (returns None where a dict is required)
        # to drive the proxy's translation-failure path — the 502 this test asserts.
        return None, []  # type: ignore[return-value]

    def translate_response(self, provider_resp: dict) -> dict:
        return {}

    async def translate_stream(self, raw_chunks: AsyncIterator[bytes]) -> AsyncIterator[dict]:
        # Unreachable — this provider fails at translate_request. Present only so the
        # class satisfies the Provider protocol structurally (it has a yield, making it
        # an async generator that yields nothing).
        for _ in ():
            yield {}


@pytest.mark.asyncio
async def test_translate_request_returns_none_gives_502():
    """Provider returning None from translate_request produces 502, not crash."""
    port = _find_free_port()
    provider = _BrokenProvider()
    server = await start_proxy(host="127.0.0.1", port=port, upstream_url="http://127.0.0.1:1")
    # We need to patch the handler's provider directly
    server.close()
    await server.wait_closed()

    from claude_bridge.proxy import _make_handler
    from claude_bridge.router import Router
    from claude_bridge.stats import BridgeStats

    handler = _make_handler("http://127.0.0.1:1", Router(), provider, BridgeStats())
    server = await asyncio.start_server(handler, "127.0.0.1", port)
    try:
        status, data = await asyncio.to_thread(
            _http_post,
            f"http://127.0.0.1:{port}/v1/messages",
            {"model": "test", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert status == 502
        assert data["type"] == "error"
        assert data["error"]["type"] == "api_error"
        assert "translation failed" in data["error"]["message"].lower()
    finally:
        server.close()
        await server.wait_closed()


# ---------------------------------------------------------------------------
# Rate limit header forwarding tests
# ---------------------------------------------------------------------------


class _RateLimitUpstreamHandler(BaseHTTPRequestHandler):
    """Returns rate limit headers alongside a normal response."""

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        _body = self.rfile.read(length)
        resp = {
            "id": "msg_test",
            "type": "message",
            "content": [{"type": "text", "text": "hello"}],
            "model": "claude-sonnet-4-20250514",
        }
        payload = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("anthropic-ratelimit-requests-limit", "1000")
        self.send_header("anthropic-ratelimit-requests-remaining", "999")
        self.send_header("retry-after", "30")
        self.send_header("x-ratelimit-limit-tokens", "50000")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        pass


@pytest.mark.asyncio
async def test_rate_limit_headers_forwarded():
    """Rate limit headers from upstream are forwarded to the client."""
    # Start upstream with rate limit headers
    upstream_port = _find_free_port()
    upstream_server = HTTPServer(("127.0.0.1", upstream_port), _RateLimitUpstreamHandler)
    upstream_thread = Thread(target=upstream_server.serve_forever, daemon=True)
    upstream_thread.start()
    upstream_url = f"http://127.0.0.1:{upstream_port}"

    proxy_port = _find_free_port()
    server = await start_proxy(host="127.0.0.1", port=proxy_port, upstream_url=upstream_url)
    try:

        def _check_headers():
            data = json.dumps(
                {"model": "test", "messages": [{"role": "user", "content": "hi"}]}
            ).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{proxy_port}/v1/messages",
                data=data,
                method="POST",
            )
            req.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return {k.lower(): v for k, v in resp.getheaders()}

        headers_dict = await asyncio.to_thread(_check_headers)
        assert "anthropic-ratelimit-requests-limit" in headers_dict
        assert headers_dict["anthropic-ratelimit-requests-limit"] == "1000"
        assert "retry-after" in headers_dict
        assert "x-ratelimit-limit-tokens" in headers_dict
    finally:
        server.close()
        await server.wait_closed()
        upstream_server.shutdown()


# ---------------------------------------------------------------------------
# count_tokens endpoint tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_count_tokens_returns_estimate(proxy_url: str):
    """POST /v1/messages/count_tokens returns a token count estimate."""
    request_body = {
        "model": "claude-sonnet-4-6",
        "messages": [{"role": "user", "content": "Hello, world!"}],
    }
    status, data = await asyncio.to_thread(
        _http_post,
        f"{proxy_url}/v1/messages/count_tokens",
        request_body,
    )
    assert status == 200
    assert "input_tokens" in data
    assert data["input_tokens"] > 0


@pytest.mark.asyncio
async def test_count_tokens_empty_messages(proxy_url: str):
    """count_tokens with empty messages list returns 0 (no content to count)."""
    request_body = {"model": "claude-sonnet-4-6", "messages": []}
    status, data = await asyncio.to_thread(
        _http_post,
        f"{proxy_url}/v1/messages/count_tokens",
        request_body,
    )
    assert status == 200
    assert data["input_tokens"] == 0


@pytest.mark.asyncio
async def test_count_tokens_with_tools(proxy_url: str):
    """count_tokens includes tool definitions in the estimate."""
    request_body = {
        "model": "claude-sonnet-4-6",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [
            {
                "name": "get_weather",
                "description": "Get the current weather",
                "input_schema": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            }
        ],
    }
    # With tools, estimate should be higher than without
    status_with, data_with = await asyncio.to_thread(
        _http_post,
        f"{proxy_url}/v1/messages/count_tokens",
        request_body,
    )
    _status_without, data_without = await asyncio.to_thread(
        _http_post,
        f"{proxy_url}/v1/messages/count_tokens",
        {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert status_with == 200
    assert data_with["input_tokens"] > data_without["input_tokens"]


@pytest.mark.asyncio
async def test_count_tokens_malformed_body(proxy_url: str):
    """count_tokens with non-JSON body returns 0 tokens (graceful fallback)."""
    # Send raw bytes that aren't valid JSON via raw connection
    port = int(proxy_url.rsplit(":", 1)[1])
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(
        b"POST /v1/messages/count_tokens HTTP/1.1\r\n"
        b"Content-Type: application/json\r\n"
        b"Content-Length: 11\r\n"
        b"\r\n"
        b"not-a-json!"
    )
    await writer.drain()
    response = await asyncio.wait_for(reader.read(-1), timeout=2)
    writer.close()
    assert b"200" in response
    assert b'"input_tokens": 0' in response


@pytest.mark.asyncio
async def test_normal_body_passes_size_check(proxy_url: str):
    """Request within MAX_REQUEST_BODY proceeds normally."""
    request_body = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
    }
    status, _data = await asyncio.to_thread(
        _http_post,
        f"{proxy_url}/v1/messages",
        request_body,
        {"x-api-key": "test-key"},
    )
    assert status == 200


# --- Streaming error path tests ---


class _Mock500StreamHandler(BaseHTTPRequestHandler):
    """Returns 500 for streaming requests."""

    def do_POST(self):
        payload = json.dumps(
            {
                "error": {"message": "provider stream failed", "type": "server_error"},
                "debug_body": "PLACEHOLDER_SECRET_STREAM_RAW_BODY",
            }
        ).encode()
        self.send_response(500)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        pass


def _register_provider_at(url: str):
    """Register a fake provider pointing at the given URL and return cleanup fn."""
    from claude_bridge.proxy import _provider_cache

    class _ErrorProvider(_FakeOpenAIProvider):
        def __init__(self):
            super().__init__(endpoint=f"{url}/v1/responses")

    old = PROVIDERS.get("openai")
    PROVIDERS["openai"] = _ErrorProvider
    _provider_cache.pop("openai", None)

    def _cleanup():
        _provider_cache.pop("openai", None)
        if old is not None:
            PROVIDERS["openai"] = old
        else:
            PROVIDERS.pop("openai", None)

    return _cleanup


@pytest.mark.asyncio
async def test_stream_via_provider_http_error_log_omits_raw_body_secret():
    """Streaming provider errors log status/message, never the raw provider body."""
    from claude_bridge.log import configure_logging

    stream = io.StringIO()
    configure_logging(level="ERROR", stream=stream)

    port = _find_free_port()
    server = HTTPServer(("127.0.0.1", port), _Mock500StreamHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    cleanup = _register_provider_at(f"http://127.0.0.1:{port}")
    try:
        proxy_port = _find_free_port()
        proxy_server = await start_proxy(
            host="127.0.0.1",
            port=proxy_port,
            upstream_url="http://127.0.0.1:1",  # unused — direct mode
            provider_name="openai",
        )
        try:
            request_body = {
                "model": "claude-sonnet-4-6",
                "max_tokens": 100,
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            }
            status, data = await asyncio.to_thread(
                _http_post,
                f"http://127.0.0.1:{proxy_port}/v1/messages",
                request_body,
            )
            assert status == 500
            # The provider error is translated to an Anthropic error envelope.
            assert data["type"] == "error"
            assert data["error"]["type"] == "api_error"
            assert data["error"]["message"] == "provider stream failed"

            logs = stream.getvalue()
            assert "Provider HTTP 500" in logs
            assert "provider stream failed" in logs
            assert "PLACEHOLDER_SECRET_STREAM_RAW_BODY" not in logs
            assert "debug_body" not in logs
        finally:
            proxy_server.close()
            await proxy_server.wait_closed()
    finally:
        cleanup()
        server.shutdown()


@pytest.mark.asyncio
async def test_stream_via_provider_connection_refused_returns_502():
    """Provider unreachable during streaming returns 502."""
    dead_port = _find_free_port()  # nothing listening
    cleanup = _register_provider_at(f"http://127.0.0.1:{dead_port}")
    try:
        proxy_port = _find_free_port()
        proxy_server = await start_proxy(
            host="127.0.0.1",
            port=proxy_port,
            upstream_url="http://127.0.0.1:1",
            provider_name="openai",
        )
        try:
            request_body = {
                "model": "claude-sonnet-4-6",
                "max_tokens": 100,
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            }
            status, _data = await asyncio.to_thread(
                _http_post,
                f"http://127.0.0.1:{proxy_port}/v1/messages",
                request_body,
            )
            assert status == 502
        finally:
            proxy_server.close()
            await proxy_server.wait_closed()
    finally:
        cleanup()


@pytest.mark.asyncio
async def test_stream_passthrough_upstream_unavailable_returns_502():
    """Upstream unreachable during streaming passthrough returns 502."""
    dead_port = _find_free_port()
    proxy_port = _find_free_port()
    proxy_server = await start_proxy(
        host="127.0.0.1",
        port=proxy_port,
        upstream_url=f"http://127.0.0.1:{dead_port}",
    )
    try:
        request_body = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 100,
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        }
        status, _data = await asyncio.to_thread(
            _http_post,
            f"http://127.0.0.1:{proxy_port}/v1/messages",
            request_body,
            {"x-api-key": "test-key"},
        )
        assert status == 502
    finally:
        proxy_server.close()
        await proxy_server.wait_closed()


# --- Retry logic tests ---


def test_retry_request_retries_on_transient_error():
    """_retry_request retries once on URLError, then succeeds."""
    import urllib.error

    from claude_bridge.proxy import _retry_request

    call_count = 0

    def flaky_fn():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise urllib.error.URLError("Connection reset")
        return 200, b"ok"

    status, body = _retry_request(flaky_fn, retries=1, backoff=0.0)
    assert status == 200
    assert body == b"ok"
    assert call_count == 2


def test_retry_request_gives_up_after_max_retries():
    """_retry_request returns error after exhausting retries."""
    import urllib.error

    from claude_bridge.proxy import _retry_request

    def always_fails():
        raise urllib.error.URLError("Connection refused")

    status, _body = _retry_request(always_fails, retries=1, backoff=0.0)
    assert status == 502


def test_retry_request_no_retry_on_http_error():
    """_retry_request does not retry on HTTPError (non-transient)."""
    import urllib.error

    from claude_bridge.proxy import _retry_request

    call_count = 0

    def http_error():
        nonlocal call_count
        call_count += 1
        raise urllib.error.HTTPError(
            "http://test",
            400,
            "Bad Request",
            {},  # type: ignore[arg-type]
            None,  # type: ignore[arg-type]
        )

    status, _body = _retry_request(http_error, retries=1, backoff=0.0)
    assert status == 400
    assert call_count == 1


# ---------------------------------------------------------------------------
# Non-streaming provider path: aggregate the Codex SSE stream into one Message
# (regression for the empty-content bug — Codex completed.output is always []).
# ---------------------------------------------------------------------------


def _provider_at(endpoint: str) -> _FakeOpenAIProvider:
    """A real-translation fake provider whose HTTP endpoint is a local mock."""
    return _FakeOpenAIProvider(endpoint=endpoint)


@pytest.fixture()
def _codex_sse_mock():
    """Mock provider endpoint that streams a faithful Codex SSE response.

    The handler echoes whichever stream the test installs via ``server.payload``,
    defaulting to a text turn. Yields the base URL.
    """
    payload_holder = {"bytes": _codex_text_sse("hello from codex deltas")}

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            body = payload_holder["bytes"]
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    port = _find_free_port()
    server = HTTPServer(("127.0.0.1", port), _Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}", payload_holder
    server.shutdown()


def _anthropic_request_bytes(text: str = "hi") -> bytes:
    return json.dumps(
        {
            "model": "claude-sonnet-4-6",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": text}],
        }
    ).encode()


def _text_stream_events() -> list[dict]:
    """Oracle-derived Anthropic SSE events for a two-delta text turn.

    Hand-written from the Anthropic Messages streaming spec — NOT produced by
    running the translator — so the fold's output can be checked against an
    independent oracle.
    """
    return [
        {
            "event": "message_start",
            "data": {
                "type": "message_start",
                "message": {
                    "id": "msg_bridge_resp_x",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": "gpt-5.5",
                    "stop_reason": None,
                    "usage": {"input_tokens": 10, "output_tokens": 0},
                },
            },
        },
        {"event": "ping", "data": {"type": "ping"}},
        {
            "event": "content_block_start",
            "data": {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        },
        {
            "event": "content_block_delta",
            "data": {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Hel"},
            },
        },
        {
            "event": "content_block_delta",
            "data": {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "lo"},
            },
        },
        {"event": "content_block_stop", "data": {"type": "content_block_stop", "index": 0}},
        {
            "event": "message_delta",
            "data": {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        },
        {"event": "message_stop", "data": {"type": "message_stop"}},
    ]


def test_aggregate_stream_to_message_concatenates_text_deltas():
    """Text deltas fold into a single text block; id/model/stop/usage preserved."""
    from claude_bridge.proxy import _aggregate_stream_to_message

    message = _aggregate_stream_to_message(_text_stream_events())
    assert message is not None

    # Oracle: "Hel" + "lo" = "Hello" by the streaming spec (deltas concatenate).
    assert message["content"] == [{"type": "text", "text": "Hello"}]
    assert message["id"] == "msg_bridge_resp_x"
    assert message["model"] == "gpt-5.5"
    assert message["role"] == "assistant"
    assert message["type"] == "message"
    assert message["stop_reason"] == "end_turn"
    # Oracle: the auto-compact signal — final usage comes from message_delta.
    assert message["usage"] == {"input_tokens": 10, "output_tokens": 5}


def test_aggregate_stream_to_message_parses_tool_use_json():
    """input_json_delta fragments fold into a parsed tool_use input dict."""
    from claude_bridge.proxy import _aggregate_stream_to_message

    events = [
        {
            "event": "message_start",
            "data": {
                "type": "message_start",
                "message": {
                    "id": "msg_bridge_t",
                    "model": "gpt-5.5",
                    "usage": {"input_tokens": 7, "output_tokens": 0},
                },
            },
        },
        {
            "event": "content_block_start",
            "data": {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "call_abc",
                    "name": "get_weather",
                    "input": {},
                },
            },
        },
        {
            "event": "content_block_delta",
            "data": {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '{"city":'},
            },
        },
        {
            "event": "content_block_delta",
            "data": {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '"NYC"}'},
            },
        },
        {"event": "content_block_stop", "data": {"type": "content_block_stop", "index": 0}},
        {
            "event": "message_delta",
            "data": {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
                "usage": {"input_tokens": 7, "output_tokens": 3},
            },
        },
        {"event": "message_stop", "data": {"type": "message_stop"}},
    ]

    message = _aggregate_stream_to_message(events)
    assert message is not None

    # Oracle: json.loads('{"city:' + '"NYC"}') == {"city": "NYC"} — hand-derivable.
    assert message["content"] == [
        {"type": "tool_use", "id": "call_abc", "name": "get_weather", "input": {"city": "NYC"}}
    ]
    assert message["stop_reason"] == "tool_use"


def test_aggregate_stream_to_message_preserves_block_order():
    """A text block followed by a tool_use block keeps arrival order."""
    from claude_bridge.proxy import _aggregate_stream_to_message

    events = [
        {
            "event": "message_start",
            "data": {
                "type": "message_start",
                "message": {
                    "id": "msg_bridge_o",
                    "model": "gpt-5.5",
                    "usage": {"input_tokens": 1, "output_tokens": 0},
                },
            },
        },
        {
            "event": "content_block_start",
            "data": {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        },
        {
            "event": "content_block_delta",
            "data": {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "look"},
            },
        },
        {"event": "content_block_stop", "data": {"type": "content_block_stop", "index": 0}},
        {
            "event": "content_block_start",
            "data": {
                "type": "content_block_start",
                "index": 1,
                "content_block": {
                    "type": "tool_use",
                    "id": "call_z",
                    "name": "search",
                    "input": {},
                },
            },
        },
        {
            "event": "content_block_delta",
            "data": {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": "{}"},
            },
        },
        {"event": "content_block_stop", "data": {"type": "content_block_stop", "index": 1}},
        {
            "event": "message_delta",
            "data": {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
                "usage": {"input_tokens": 1, "output_tokens": 2},
            },
        },
        {"event": "message_stop", "data": {"type": "message_stop"}},
    ]

    message = _aggregate_stream_to_message(events)
    assert message is not None

    assert [b["type"] for b in message["content"]] == ["text", "tool_use"]
    assert message["content"][0]["text"] == "look"
    assert message["content"][1]["id"] == "call_z"


def test_aggregate_stream_to_message_malformed_tool_json_keeps_raw():
    """Unparseable tool arguments fall back to {'_raw': ...} (never crash)."""
    from claude_bridge.proxy import _aggregate_stream_to_message

    events = [
        {
            "event": "message_start",
            "data": {
                "type": "message_start",
                "message": {
                    "id": "msg_bridge_m",
                    "model": "gpt-5.5",
                    "usage": {"input_tokens": 1, "output_tokens": 0},
                },
            },
        },
        {
            "event": "content_block_start",
            "data": {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "call_m", "name": "f", "input": {}},
            },
        },
        {
            "event": "content_block_delta",
            "data": {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": "{not json"},
            },
        },
        {"event": "content_block_stop", "data": {"type": "content_block_stop", "index": 0}},
        {
            "event": "message_delta",
            "data": {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        },
        {"event": "message_stop", "data": {"type": "message_stop"}},
    ]

    message = _aggregate_stream_to_message(events)
    assert message is not None

    assert message["content"][0]["input"] == {"_raw": "{not json"}


def test_aggregate_stream_to_message_no_message_start_returns_none():
    """A stream with no message_start (malformed/empty) is unusable → None."""
    from claude_bridge.proxy import _aggregate_stream_to_message

    assert _aggregate_stream_to_message([]) is None
    assert _aggregate_stream_to_message([{"event": "ping", "data": {"type": "ping"}}]) is None


@pytest.mark.asyncio
async def test_forward_via_provider_codex_empty_output_populates_text(_codex_sse_mock):
    """REGRESSION: Codex completed.output is [] — text must come from deltas.

    F2P: against the pre-fix code (_extract_completed_response reads the empty
    output) this returns content == [] ; the fold makes it the delta text.
    """
    from claude_bridge.proxy import _forward_via_provider

    url, _payload = _codex_sse_mock
    provider = _provider_at(f"{url}/v1/responses")

    status, body = await _forward_via_provider(provider, _anthropic_request_bytes())

    assert status == 200
    response = json.loads(body)
    # Oracle: the two deltas of "hello from codex deltas" reassemble verbatim.
    assert response["content"] == [{"type": "text", "text": "hello from codex deltas"}]
    assert response["stop_reason"] == "end_turn"
    assert response["model"] == "gpt-5.5"
    # The auto-compact signal must survive: input_tokens is the real context size.
    assert response["usage"]["input_tokens"] == 10
    assert response["usage"]["output_tokens"] == 5


@pytest.mark.asyncio
async def test_forward_via_provider_codex_tool_call_populates_tool_use(_codex_sse_mock):
    """A Codex tool turn (empty completed.output) folds into a tool_use block."""
    from claude_bridge.proxy import _forward_via_provider

    url, payload = _codex_sse_mock
    payload["bytes"] = _codex_tool_sse()
    provider = _provider_at(f"{url}/v1/responses")

    status, body = await _forward_via_provider(provider, _anthropic_request_bytes())

    assert status == 200
    response = json.loads(body)
    # Oracle: Claude Code requires toolu_ ids; call_weather1 -> toolu_weather1.
    assert response["content"] == [
        {
            "type": "tool_use",
            "id": "toolu_weather1",
            "name": "get_weather",
            "input": {"city": "NYC"},
        }
    ]
    # stop_reason comes from the streamed tool_use block, not the empty output.
    assert response["stop_reason"] == "tool_use"


@pytest.mark.asyncio
async def test_forward_via_provider_unparseable_stream_returns_502(_codex_sse_mock):
    """A non-SSE provider body (no message_start) is a 502, not a blank message."""
    from claude_bridge.proxy import _forward_via_provider

    url, payload = _codex_sse_mock
    payload["bytes"] = b"this is not an SSE stream at all"
    provider = _provider_at(f"{url}/v1/responses")

    status, _body = await _forward_via_provider(provider, _anthropic_request_bytes())

    assert status == 502


# ---------------------------------------------------------------------------
# Provider error translation: upstream OpenAI errors → Anthropic error envelope
# (so a Codex 400 context_length_exceeded reaches Claude Code cleanly, not as a
# raw OpenAI shape).
# ---------------------------------------------------------------------------


def test_anthropic_error_body_maps_status_to_type():
    """HTTP status maps to the Anthropic error type from the API spec table."""
    from claude_bridge.proxy import _anthropic_error_body

    # Oracle: Anthropic API error types (docs.anthropic.com/en/api/errors).
    cases = {
        400: "invalid_request_error",
        401: "authentication_error",
        403: "permission_error",
        404: "not_found_error",
        413: "request_too_large",
        429: "rate_limit_error",
        500: "api_error",
        529: "overloaded_error",
    }
    for status, expected_type in cases.items():
        body = json.loads(_anthropic_error_body(status, "boom"))
        assert body == {"type": "error", "error": {"type": expected_type, "message": "boom"}}


def test_anthropic_error_body_unknown_status_is_api_error():
    """An unmapped status defaults to api_error (never crashes)."""
    from claude_bridge.proxy import _anthropic_error_body

    body = json.loads(_anthropic_error_body(418, "teapot"))
    assert body["error"]["type"] == "api_error"
    assert body["error"]["message"] == "teapot"


def test_provider_error_message_extracts_openai_message():
    """The OpenAI {'error': {'message': ...}} shape yields just the message."""
    from claude_bridge.proxy import _provider_error_message

    raw = json.dumps(
        {
            "error": {
                "message": "Input tokens exceed the configured limit of 400000 tokens.",
                "type": "invalid_request_error",
                "code": "context_length_exceeded",
            }
        }
    ).encode()
    # Oracle: the human-readable string is .error.message, not the whole envelope.
    assert (
        _provider_error_message(raw)
        == "Input tokens exceed the configured limit of 400000 tokens."
    )


def test_provider_error_message_non_json_falls_back_to_body():
    """A non-JSON error body is surfaced verbatim (decoded, truncated)."""
    from claude_bridge.proxy import _provider_error_message

    assert _provider_error_message(b"502 Bad Gateway") == "502 Bad Gateway"


def test_provider_error_log_summary_non_json_omits_raw_body_secret():
    """Malformed provider bodies log only metadata, never raw body text."""
    from claude_bridge.proxy import _provider_error_log_summary

    summary = _provider_error_log_summary(b"PLACEHOLDER_SECRET_MALFORMED_RAW_BODY")

    assert summary == "unparseable provider error body (37B)"
    assert "PLACEHOLDER_SECRET_MALFORMED_RAW_BODY" not in summary


@pytest.mark.asyncio
async def test_forward_via_provider_error_log_omits_raw_body_secret():
    """Sync provider errors log status/message, never the raw provider body."""
    from claude_bridge.log import configure_logging
    from claude_bridge.proxy import _forward_via_provider

    safe_message = "Input tokens exceed the configured limit of 400000 tokens."
    secret_marker = "PLACEHOLDER_SECRET_SYNC_RAW_BODY"

    class _SecretErrorHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            payload = json.dumps(
                {
                    "error": {"message": safe_message, "type": "invalid_request_error"},
                    "debug_body": secret_marker,
                }
            ).encode()
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):
            pass

    stream = io.StringIO()
    configure_logging(level="ERROR", stream=stream)

    port = _find_free_port()
    server = HTTPServer(("127.0.0.1", port), _SecretErrorHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        provider = _provider_at(f"http://127.0.0.1:{port}/v1/responses")
        status, body = await _forward_via_provider(provider, _anthropic_request_bytes())
    finally:
        server.shutdown()

    assert status == 400
    response = json.loads(body)
    assert response["error"]["message"] == safe_message

    logs = stream.getvalue()
    assert "Provider HTTP 400" in logs
    assert safe_message in logs
    assert secret_marker not in logs
    assert "debug_body" not in logs


class _MockOverflowHandler(BaseHTTPRequestHandler):
    """Returns HTTP 400 with the OpenAI context_length_exceeded error shape."""

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        payload = json.dumps(
            {
                "error": {
                    "message": "Input tokens exceed the configured limit of 400000 tokens.",
                    "type": "invalid_request_error",
                    "code": "context_length_exceeded",
                }
            }
        ).encode()
        self.send_response(400)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        pass


@pytest.mark.asyncio
async def test_forward_via_provider_overflow_returns_anthropic_error():
    """A Codex 400 overflow is translated to an Anthropic error envelope (not raw)."""
    from claude_bridge.proxy import _forward_via_provider

    port = _find_free_port()
    server = HTTPServer(("127.0.0.1", port), _MockOverflowHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        provider = _provider_at(f"http://127.0.0.1:{port}/v1/responses")
        status, body = await _forward_via_provider(provider, _anthropic_request_bytes())
    finally:
        server.shutdown()

    assert status == 400
    response = json.loads(body)
    # Oracle: Anthropic error shape; 400 -> invalid_request_error; message preserved.
    assert response["type"] == "error"
    assert response["error"]["type"] == "invalid_request_error"
    assert "400000" in response["error"]["message"]
    # The raw OpenAI envelope must NOT leak through.
    assert "error" not in json.loads(body).get("error", {})
