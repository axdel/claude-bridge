"""Tests for provider protocol capabilities declarations."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest


def test_provider_capabilities_are_frozen_protocol_owned_values():
    """ProviderCapabilities exposes immutable stream/sync mode declarations."""
    from claude_bridge.provider import ProviderCapabilities

    capabilities = ProviderCapabilities(
        stream_request_mode="body_parameter",
        sync_response_mode="sse",
    )

    assert capabilities.stream_request_mode == "body_parameter"
    assert capabilities.sync_response_mode == "sse"
    with pytest.raises(FrozenInstanceError):
        capabilities.stream_request_mode = "url"  # type: ignore[misc]


def test_provider_capabilities_reject_unknown_modes():
    """Invalid provider capability modes fail at declaration time."""
    from claude_bridge.provider import ProviderCapabilities

    with pytest.raises(ValueError, match="stream_request_mode"):
        ProviderCapabilities(stream_request_mode="query", sync_response_mode="sse")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="sync_response_mode"):
        ProviderCapabilities(stream_request_mode="url", sync_response_mode="xml")  # type: ignore[arg-type]


def test_openai_declares_body_parameter_streaming_and_sse_sync_response():
    """OpenAI exposes the proxy-visible transport modes it requires."""
    from claude_bridge.provider import ProviderCapabilities
    from claude_bridge.providers.openai import OpenAIProvider

    assert OpenAIProvider.capabilities == ProviderCapabilities(
        stream_request_mode="body_parameter",
        sync_response_mode="sse",
    )


def test_gemini_declares_url_streaming_and_sse_sync_response():
    """Gemini exposes URL-selected streaming and SSE sync-response folding."""
    from claude_bridge.provider import ProviderCapabilities
    from claude_bridge.providers.gemini import GeminiProvider

    assert GeminiProvider.capabilities == ProviderCapabilities(
        stream_request_mode="url",
        sync_response_mode="sse",
    )
