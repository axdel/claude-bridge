"""Tests for the Claude Bridge proxy server."""

from __future__ import annotations

import asyncio
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


class _MockOpenAIHandler(BaseHTTPRequestHandler):
    """Returns a valid OpenAI Responses API response."""

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        _body = self.rfile.read(length)
        resp = {
            "id": "resp_test123",
            "status": "completed",
            "model": "gpt-5.5",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "hello from openai fallback"}],
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        payload = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
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

        provider = OpenAIProvider.__new__(OpenAIProvider)
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
        payload = json.dumps({"error": "server error"}).encode()
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
async def test_stream_via_provider_http_error_returns_error():
    """Provider returning HTTP 500 during streaming sends error to client."""
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
            status, _data = await asyncio.to_thread(
                _http_post,
                f"http://127.0.0.1:{proxy_port}/v1/messages",
                request_body,
            )
            assert status == 500
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
# Redacted compatibility trace — structural-only summaries must never leak
# prompt text, tool arguments, tool results, reasoning, or credential patterns.
# ---------------------------------------------------------------------------

# Every marker is planted in a content-bearing position of the fixtures below.
# The redaction oracle: none of these substrings may appear in any trace output.
# Values derived from the planted input — never from running the summarizer.
_SECRET_MARKERS = [
    "SUPER_SECRET_PROMPT",  # system prompt text
    "SECRET_TOOL_DESC",  # tool description
    "LEAKED_MESSAGE_TEXT",  # user message text
    "SECRET_REASONING",  # thinking block text
    "SECRET_SIGNATURE",  # thinking block signature (encrypted reasoning)
    "hunter2",  # tool_use input value
    "/etc/passwd",  # tool_use input value (file path)
    "LEAKED_TOOL_OUTPUT",  # tool_result content
    "sk-secretapikey123",  # api-key pattern embedded in content
    "Bearer tok_secret",  # bearer-token pattern embedded in content
]


def _secret_laden_request() -> dict:
    """An Anthropic request with a secret marker in every content-bearing field.

    Shape mirrors a Claude Code tool loop: system prompt, tool definitions,
    a user turn, an assistant thinking+tool_use turn, and a tool_result turn.
    """
    return {
        "model": "claude-opus-4-8",
        "stream": True,
        "system": [{"type": "text", "text": "SUPER_SECRET_PROMPT system instructions"}],
        "tools": [
            {
                "name": "Read",
                "description": "SECRET_TOOL_DESC read a file",
                "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
            {"name": "Bash", "description": "run a command", "input_schema": {"type": "object"}},
        ],
        "tool_choice": {"type": "tool", "name": "Read"},
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "LEAKED_MESSAGE_TEXT Bearer tok_secret sk-secretapikey123",
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "SECRET_REASONING let me read it",
                        "signature": "SECRET_SIGNATURE",
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_01",
                        "name": "Read",
                        "input": {"path": "/etc/passwd", "password": "hunter2"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_01",
                        "content": [{"type": "text", "text": "LEAKED_TOOL_OUTPUT contents"}],
                    }
                ],
            },
        ],
    }


def _secret_laden_response() -> dict:
    """An Anthropic response carrying secret markers in text and tool_use input."""
    return {
        "id": "msg_01",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-4-8",
        "stop_reason": "tool_use",
        "content": [
            {"type": "text", "text": "SECRET_RESPONSE_TEXT here is the answer"},
            {
                "type": "tool_use",
                "id": "toolu_02",
                "name": "Bash",
                "input": {"command": "echo SECRET_COMMAND"},
            },
        ],
        "usage": {"input_tokens": 1234, "output_tokens": 56},
    }


def _assert_no_secrets(summary: dict) -> None:
    """The redaction oracle — no planted secret may survive into the summary."""
    blob = json.dumps(summary)
    for marker in _SECRET_MARKERS:
        assert marker not in blob, f"secret leaked into trace: {marker!r}"


class TestSummarizeAnthropicRequest:
    """_summarize_anthropic_request emits structural shape only — counts, type
    names, tool names, lengths — and never the underlying prompt content."""

    def test_structural_fields_match_input(self):
        from claude_bridge.proxy import _summarize_anthropic_request

        summary = _summarize_anthropic_request(_secret_laden_request())
        assert summary["model"] == "claude-opus-4-8"
        assert summary["stream"] is True
        assert summary["message_count"] == 3
        assert summary["tool_count"] == 2
        assert summary["tool_names"] == ["Bash", "Read"]
        assert summary["tool_choice"] == "tool"
        assert summary["block_types"] == {
            "text": 1,
            "thinking": 1,
            "tool_use": 1,
            "tool_result": 1,
        }

    def test_system_chars_is_positive_length_not_content(self):
        from claude_bridge.proxy import _summarize_anthropic_request

        summary = _summarize_anthropic_request(_secret_laden_request())
        # A length, derived structurally — present and positive for a non-empty system.
        assert isinstance(summary["system_chars"], int)
        assert summary["system_chars"] > 0

    def test_no_secrets_leak(self):
        from claude_bridge.proxy import _summarize_anthropic_request

        _assert_no_secrets(_summarize_anthropic_request(_secret_laden_request()))

    def test_string_content_counts_as_text_block(self):
        from claude_bridge.proxy import _summarize_anthropic_request

        # Anthropic shorthand: content may be a bare string (one text block).
        request = {"model": "m", "messages": [{"role": "user", "content": "SECRET_STRING"}]}
        summary = _summarize_anthropic_request(request)
        assert summary["block_types"] == {"text": 1}
        assert "SECRET_STRING" not in json.dumps(summary)

    def test_absent_tools_and_choice_are_empty(self):
        from claude_bridge.proxy import _summarize_anthropic_request

        summary = _summarize_anthropic_request({"model": "m", "messages": []})
        assert summary["tool_count"] == 0
        assert summary["tool_names"] == []
        assert summary["tool_choice"] is None
        assert summary["system_chars"] == 0


class TestSummarizeProviderRequest:
    """_summarize_provider_request summarizes the translated provider request
    without leaking any translated input content."""

    def test_structural_fields_and_warning_count(self):
        from claude_bridge.providers.openai import anthropic_to_openai
        from claude_bridge.proxy import _summarize_provider_request

        translated, warnings = anthropic_to_openai(_secret_laden_request())
        summary = _summarize_provider_request(translated, warnings)
        assert summary["tool_count"] == 2
        assert summary["tool_names"] == ["Bash", "Read"]
        assert summary["input_items"] >= 3
        assert summary["warning_count"] == len(warnings)
        assert summary["stream"] is True

    def test_warning_strings_included_for_trace(self):
        # REQ1: the trace carries the sanitized warning *strings*, not just a count,
        # so a degraded translation is diagnosable from the trace alone (T-003 spec).
        from claude_bridge.proxy import _summarize_provider_request

        warnings = [
            "Stripped 'thinking' config (reasoning_mode=drop)",
            "Unsupported tool_choice type 'x', omitting tool_choice",
        ]
        summary = _summarize_provider_request({"model": "m", "input": []}, warnings)
        assert summary["warnings"] == warnings
        assert summary["warning_count"] == 2

    def test_forced_tool_choice_renders_structurally(self):
        from claude_bridge.proxy import _summarize_provider_request

        translated = {
            "model": "gpt-5.5",
            "input": [],
            "tools": [{"type": "function", "name": "Read"}],
            "tool_choice": {"type": "function", "name": "Read"},
        }
        summary = _summarize_provider_request(translated, [])
        assert summary["tool_choice"] == "function:Read"

    def test_parallel_flag_emitted_only_when_present(self):
        from claude_bridge.proxy import _summarize_provider_request

        with_flag = _summarize_provider_request(
            {"model": "m", "input": [], "parallel_tool_calls": False}, []
        )
        without_flag = _summarize_provider_request({"model": "m", "input": []}, [])
        assert with_flag["parallel_tool_calls"] is False
        assert "parallel_tool_calls" not in without_flag

    def test_no_secrets_leak(self):
        from claude_bridge.providers.openai import anthropic_to_openai
        from claude_bridge.proxy import _summarize_provider_request

        translated, warnings = anthropic_to_openai(_secret_laden_request())
        _assert_no_secrets(_summarize_provider_request(translated, warnings))

    def test_injected_encrypted_reasoning_never_leaks(self):
        # The provider echoes opaque encrypted reasoning into the outbound request
        # (T-005). The summarizer must count it as an input item, never serialize it.
        from claude_bridge.proxy import _summarize_provider_request

        translated = {
            "model": "gpt-5.5",
            "input": [
                {"type": "reasoning", "id": "rs_1", "encrypted_content": "SECRET_REASONING"},
                {"type": "function_call", "id": "fc_1", "call_id": "fc_1", "name": "Read"},
            ],
        }
        summary = _summarize_provider_request(translated, [])
        assert summary["input_items"] == 2
        assert "SECRET_REASONING" not in json.dumps(summary)


class TestSummarizeAnthropicResponse:
    """_summarize_anthropic_response emits stop_reason, block counts, and token
    usage — never response text or tool_use arguments."""

    def test_structural_fields_match_input(self):
        from claude_bridge.proxy import _summarize_anthropic_response

        summary = _summarize_anthropic_response(_secret_laden_response())
        assert summary["model"] == "claude-opus-4-8"
        assert summary["stop_reason"] == "tool_use"
        assert summary["block_types"] == {"text": 1, "tool_use": 1}
        assert summary["input_tokens"] == 1234
        assert summary["output_tokens"] == 56

    def test_no_secrets_leak(self):
        from claude_bridge.proxy import _summarize_anthropic_response

        summary = _summarize_anthropic_response(_secret_laden_response())
        assert "SECRET_RESPONSE_TEXT" not in json.dumps(summary)
        assert "SECRET_COMMAND" not in json.dumps(summary)


class TestSummarizeStreamEvent:
    """_summarize_stream_event classifies one SSE event by its structural fields
    (event name, block index, block/delta type, stop_reason) and drops content."""

    def test_content_block_start_emits_block_type(self):
        from claude_bridge.proxy import _summarize_stream_event

        event = {
            "event": "content_block_start",
            "data": {
                "index": 1,
                "content_block": {"type": "tool_use", "id": "toolu_03", "name": "Read"},
            },
        }
        summary = _summarize_stream_event(event)
        assert summary["sse"] == "content_block_start"
        assert summary["index"] == 1
        assert summary["block_type"] == "tool_use"

    def test_text_delta_drops_text_keeps_type(self):
        from claude_bridge.proxy import _summarize_stream_event

        event = {
            "event": "content_block_delta",
            "data": {"index": 0, "delta": {"type": "text_delta", "text": "SECRET_DELTA_TEXT"}},
        }
        summary = _summarize_stream_event(event)
        assert summary["delta_type"] == "text_delta"
        assert summary["index"] == 0
        assert "SECRET_DELTA_TEXT" not in json.dumps(summary)

    def test_input_json_delta_drops_partial_json(self):
        from claude_bridge.proxy import _summarize_stream_event

        event = {
            "event": "content_block_delta",
            "data": {
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": '{"path": "SECRET_PATH"}'},
            },
        }
        summary = _summarize_stream_event(event)
        assert summary["delta_type"] == "input_json_delta"
        assert "SECRET_PATH" not in json.dumps(summary)

    def test_message_delta_emits_stop_reason(self):
        from claude_bridge.proxy import _summarize_stream_event

        event = {
            "event": "message_delta",
            "data": {"delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 99}},
        }
        summary = _summarize_stream_event(event)
        assert summary["sse"] == "message_delta"
        assert summary["stop_reason"] == "end_turn"
        assert summary["output_tokens"] == 99


class TestTraceHooks:
    """The self-guarding trace hooks: off by default, redacted when on,
    and never raise — tracing can never break a request."""

    def test_inbound_disabled_writes_nothing(self, tmp_path, monkeypatch):
        from claude_bridge.proxy import _trace_inbound_request

        monkeypatch.delenv("CLAUDE_BRIDGE_TRACE_PATH", raising=False)
        target = tmp_path / "trace.jsonl"
        _trace_inbound_request(json.dumps(_secret_laden_request()).encode())
        assert not target.exists()

    def test_inbound_enabled_writes_redacted_structural_line(self, tmp_path, monkeypatch):
        from claude_bridge.proxy import _trace_inbound_request

        target = tmp_path / "trace.jsonl"
        monkeypatch.setenv("CLAUDE_BRIDGE_TRACE_PATH", str(target))
        _trace_inbound_request(json.dumps(_secret_laden_request()).encode())
        content = target.read_text(encoding="utf-8")
        for marker in _SECRET_MARKERS:
            assert marker not in content, f"secret leaked into trace file: {marker!r}"
        record = json.loads(content.splitlines()[0])
        assert record["event"] == "inbound_request"
        assert record["model"] == "claude-opus-4-8"
        assert record["message_count"] == 3

    def test_inbound_malformed_body_never_raises(self, tmp_path, monkeypatch):
        from claude_bridge.proxy import _trace_inbound_request

        target = tmp_path / "trace.jsonl"
        monkeypatch.setenv("CLAUDE_BRIDGE_TRACE_PATH", str(target))
        # Must swallow the JSON parse error — no file line, no exception.
        _trace_inbound_request(b"not json{{{")

    def test_provider_request_hook_redacts(self, tmp_path, monkeypatch):
        from claude_bridge.providers.openai import anthropic_to_openai
        from claude_bridge.proxy import _trace_provider_request

        target = tmp_path / "trace.jsonl"
        monkeypatch.setenv("CLAUDE_BRIDGE_TRACE_PATH", str(target))
        translated, warnings = anthropic_to_openai(_secret_laden_request())
        _trace_provider_request(translated, warnings)
        content = target.read_text(encoding="utf-8")
        for marker in _SECRET_MARKERS:
            assert marker not in content
        assert json.loads(content.splitlines()[0])["event"] == "provider_request"

    def test_provider_response_hook_redacts(self, tmp_path, monkeypatch):
        from claude_bridge.proxy import _trace_provider_response

        target = tmp_path / "trace.jsonl"
        monkeypatch.setenv("CLAUDE_BRIDGE_TRACE_PATH", str(target))
        _trace_provider_response(_secret_laden_response())
        content = target.read_text(encoding="utf-8")
        assert "SECRET_RESPONSE_TEXT" not in content
        assert "SECRET_COMMAND" not in content
        assert json.loads(content.splitlines()[0])["stop_reason"] == "tool_use"

    def test_stream_event_hook_redacts(self, tmp_path, monkeypatch):
        from claude_bridge.proxy import _trace_stream_event

        target = tmp_path / "trace.jsonl"
        monkeypatch.setenv("CLAUDE_BRIDGE_TRACE_PATH", str(target))
        _trace_stream_event(
            {
                "event": "content_block_delta",
                "data": {"index": 0, "delta": {"type": "text_delta", "text": "SECRET_DELTA_TEXT"}},
            }
        )
        content = target.read_text(encoding="utf-8")
        assert "SECRET_DELTA_TEXT" not in content
        assert json.loads(content.splitlines()[0])["delta_type"] == "text_delta"

    def test_stream_event_disabled_writes_nothing(self, tmp_path, monkeypatch):
        from claude_bridge.proxy import _trace_stream_event

        monkeypatch.delenv("CLAUDE_BRIDGE_TRACE_PATH", raising=False)
        target = tmp_path / "trace.jsonl"
        _trace_stream_event({"event": "message_stop", "data": {}})
        assert not target.exists()
