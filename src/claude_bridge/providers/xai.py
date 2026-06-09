"""xAI Grok provider placeholder for future implementation.

This module intentionally does not register ``XAIProvider`` in ``PROVIDERS``.
To enable xAI, implement authentication plus request/response/stream translation,
declare provider capabilities, then register and import the provider.
"""

from __future__ import annotations

from collections.abc import AsyncIterator


class XAIProvider:
    """xAI Grok provider — stub for extensibility proof."""

    name = "xai"
    endpoint = "https://api.x.ai/v1/chat/completions"

    async def authenticate(self) -> dict[str, str]:
        """Return xAI auth headers. Requires XAI_API_KEY env var."""
        raise NotImplementedError("xAI Grok provider not yet implemented")

    def translate_request(self, _anthropic_req: dict) -> tuple[dict, list[str]]:
        """Translate Anthropic request to xAI format."""
        raise NotImplementedError("xAI Grok provider not yet implemented")

    def translate_response(self, _provider_resp: dict) -> dict:
        """Translate xAI response back to Anthropic format."""
        raise NotImplementedError("xAI Grok provider not yet implemented")

    def translate_stream(self, _raw_chunks: AsyncIterator[bytes]) -> AsyncIterator[dict]:
        """Translate raw xAI byte chunks to Anthropic-format SSE events."""
        raise NotImplementedError("xAI Grok provider not yet implemented")
