"""Tests for the module entry point's provider auth-routing and server wiring.

The routing logic (which constructor kwargs and log line each provider earns) is
deterministic and lives in pure helpers, tested here in isolation; a single wiring
smoke test drives ``main()`` end-to-end with the server bootstrap stubbed.
"""

from __future__ import annotations

import sys

import pytest


class TestDetectOpenAIAuthMode:
    def test_api_key_present_selects_api_key_mode(self, monkeypatch):
        import claude_bridge.__main__ as main_mod

        monkeypatch.setenv("OPENAI_API_KEY", "sk-live-placeholder")
        assert main_mod._detect_openai_auth_mode() == ("api_key", "sk-live-placeholder")

    def test_api_key_absent_selects_codex_oauth(self, monkeypatch):
        import claude_bridge.__main__ as main_mod

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert main_mod._detect_openai_auth_mode() == ("codex_oauth", None)

    def test_blank_api_key_selects_codex_oauth(self, monkeypatch):
        import claude_bridge.__main__ as main_mod

        monkeypatch.setenv("OPENAI_API_KEY", "   ")
        assert main_mod._detect_openai_auth_mode() == ("codex_oauth", None)


class TestBuildProviderKwargs:
    def test_xai_takes_no_kwargs(self, monkeypatch):
        import claude_bridge.__main__ as main_mod

        # Even with an OpenAI key present, xAI resolves its own subscription OAuth
        # from ~/.grok via a no-arg constructor — it must not inherit OpenAI kwargs.
        monkeypatch.setenv("OPENAI_API_KEY", "sk-live-placeholder")
        assert main_mod._build_provider_kwargs("xai") == {}

    def test_openai_with_key_carries_api_key_mode(self, monkeypatch):
        import claude_bridge.__main__ as main_mod

        monkeypatch.setenv("OPENAI_API_KEY", "sk-live-placeholder")
        assert main_mod._build_provider_kwargs("openai") == {
            "auth_mode": "api_key",
            "api_key": "sk-live-placeholder",
        }

    def test_openai_without_key_carries_codex_oauth_and_no_api_key(self, monkeypatch):
        import claude_bridge.__main__ as main_mod

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert main_mod._build_provider_kwargs("openai") == {"auth_mode": "codex_oauth"}

    def test_auto_mode_none_provider_carries_openai_kwargs(self, monkeypatch):
        import claude_bridge.__main__ as main_mod

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert main_mod._build_provider_kwargs(None) == {"auth_mode": "codex_oauth"}


class TestAuthModeLogMessage:
    def test_xai_names_grok_oauth(self):
        import claude_bridge.__main__ as main_mod

        msg = main_mod._auth_mode_log_message("xai", {})
        assert "grok_oauth" in msg

    def test_openai_api_key_names_api_key(self):
        import claude_bridge.__main__ as main_mod

        msg = main_mod._auth_mode_log_message("openai", {"auth_mode": "api_key"})
        assert "api_key" in msg

    def test_openai_codex_names_codex_oauth(self):
        import claude_bridge.__main__ as main_mod

        msg = main_mod._auth_mode_log_message("openai", {"auth_mode": "codex_oauth"})
        assert "codex_oauth" in msg

    def test_log_message_never_contains_the_credential(self):
        import claude_bridge.__main__ as main_mod

        # A credential in provider_kwargs must never reach the log line.
        kwargs = {"auth_mode": "api_key", "api_key": "sk-secret-value"}
        assert "sk-secret-value" not in main_mod._auth_mode_log_message("openai", kwargs)


class _StopServing(Exception):
    """Sentinel to break out of serve_forever() in the wiring smoke test."""


class _FakeServer:
    """Async-context-manager server stand-in whose serve loop exits immediately."""

    async def __aenter__(self) -> _FakeServer:
        return self

    async def __aexit__(self, *args: object) -> bool:
        return False

    async def serve_forever(self) -> None:
        raise _StopServing


class TestMainWiring:
    def test_main_wires_defaults_and_routes_xai_with_no_kwargs(self, monkeypatch):
        import claude_bridge.__main__ as main_mod

        captured: dict = {}

        async def _fake_start(**kwargs: object) -> _FakeServer:
            captured.update(kwargs)
            return _FakeServer()

        monkeypatch.setattr(main_mod, "start_proxy", _fake_start)
        monkeypatch.setattr(sys, "argv", ["claude-bridge", "--provider", "xai"])

        with pytest.raises(_StopServing):
            main_mod.main()

        # Oracle: argparse defaults (loopback host, port 9999) and xai routing (no kwargs).
        assert captured["host"] == "127.0.0.1"
        assert captured["port"] == 9999
        assert captured["provider_name"] == "xai"
        assert captured["provider_kwargs"] == {}

    def test_main_honors_explicit_host_and_port(self, monkeypatch):
        import claude_bridge.__main__ as main_mod

        captured: dict = {}

        async def _fake_start(**kwargs: object) -> _FakeServer:
            captured.update(kwargs)
            return _FakeServer()

        monkeypatch.setattr(main_mod, "start_proxy", _fake_start)
        # 192.0.2.1 is RFC 5737 TEST-NET-1 — a non-loopback value that proves the explicit
        # --host is threaded through, without the bind-all smell of 0.0.0.0.
        monkeypatch.setattr(
            sys,
            "argv",
            ["claude-bridge", "--provider", "xai", "--host", "192.0.2.1", "--port", "8080"],
        )

        with pytest.raises(_StopServing):
            main_mod.main()

        assert captured["host"] == "192.0.2.1"
        assert captured["port"] == 8080
