"""Tests for the Claude Bridge proxy server."""

from __future__ import annotations

import asyncio
import json
import socket
import urllib.request
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

    def do_POST(self):  # noqa: N802
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

    def log_message(self, format, *args):  # noqa: A002
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
        writer.write(
            b"POST /v1/messages HTTP/1.1\r\nContent-Length: not-a-number\r\n\r\n"
        )
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

    def do_POST(self):  # noqa: N802
        payload = json.dumps({"error": "internal server error"}).encode()
        self.send_response(500)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):  # noqa: A002
        pass


class _MockOpenAIHandler(BaseHTTPRequestHandler):
    """Returns a valid OpenAI Responses API response."""

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        _body = self.rfile.read(length)
        resp = {
            "id": "resp_test123",
            "status": "completed",
            "model": "gpt-5.4",
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "hello from openai fallback"}
                    ],
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

    def log_message(self, format, *args):  # noqa: A002
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
    anthropic_server = HTTPServer(
        ("127.0.0.1", anthropic_port), _Mock500UpstreamHandler
    )
    anthropic_thread = Thread(target=anthropic_server.serve_forever, daemon=True)
    anthropic_thread.start()
    anthropic_url = f"http://127.0.0.1:{anthropic_port}"

    proxy_port = _find_free_port()
    server = await start_proxy(
        host="127.0.0.1", port=proxy_port, upstream_url=anthropic_url
    )
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

    def translate_request(self, anthropic_req: dict) -> tuple[None, list[str]]:
        return None, []

    def translate_response(self, provider_resp: dict) -> dict:
        return {}


@pytest.mark.asyncio
async def test_translate_request_returns_none_gives_502():
    """Provider returning None from translate_request produces 502, not crash."""
    port = _find_free_port()
    provider = _BrokenProvider()
    server = await start_proxy(
        host="127.0.0.1", port=port, upstream_url="http://127.0.0.1:1"
    )
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

    def do_POST(self):  # noqa: N802
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

    def log_message(self, format, *args):  # noqa: A002
        pass


@pytest.mark.asyncio
async def test_rate_limit_headers_forwarded():
    """Rate limit headers from upstream are forwarded to the client."""
    # Start upstream with rate limit headers
    upstream_port = _find_free_port()
    upstream_server = HTTPServer(
        ("127.0.0.1", upstream_port), _RateLimitUpstreamHandler
    )
    upstream_thread = Thread(target=upstream_server.serve_forever, daemon=True)
    upstream_thread.start()
    upstream_url = f"http://127.0.0.1:{upstream_port}"

    proxy_port = _find_free_port()
    server = await start_proxy(
        host="127.0.0.1", port=proxy_port, upstream_url=upstream_url
    )
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


@pytest.mark.asyncio
async def test_normal_body_passes_size_check(proxy_url: str):
    """Request within MAX_REQUEST_BODY proceeds normally."""
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
