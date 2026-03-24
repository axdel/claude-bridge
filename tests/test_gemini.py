"""Tests for the Gemini provider — auth, translation, and streaming."""

from __future__ import annotations

import pytest

# --- Auth tests ---


class TestGeminiAuth:
    """GeminiProvider authentication via GEMINI_API_KEY."""

    def test_authenticate_returns_api_key_header(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key-placeholder")
        from claude_bridge.providers.gemini import GeminiProvider

        provider = GeminiProvider()
        import asyncio

        headers = asyncio.run(provider.authenticate())
        assert headers == {"x-goog-api-key": "test-gemini-key-placeholder"}

    def test_authenticate_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        from claude_bridge.providers.gemini import GeminiProvider

        provider = GeminiProvider()
        import asyncio

        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            asyncio.run(provider.authenticate())

    def test_provider_registered_in_providers(self):
        from claude_bridge.provider import PROVIDERS

        assert "gemini" in PROVIDERS

    def test_provider_has_correct_name(self):
        from claude_bridge.providers.gemini import GeminiProvider

        assert GeminiProvider.name == "gemini"

    def test_provider_endpoint_contains_model(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key-placeholder")
        from claude_bridge.providers.gemini import GeminiProvider

        provider = GeminiProvider()
        assert "generateContent" not in provider.endpoint
        assert "v1beta/models/" in provider.endpoint
