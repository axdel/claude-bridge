"""xAI Grok provider stub — placeholder for future implementation.

Implements the Provider protocol with NotImplementedError stubs.
To complete: fill in authenticate(), translate_request(), translate_response(),
and translate_stream() with xAI API specifics.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from claude_bridge.provider import PROVIDERS


class XAIProvider:
    """xAI Grok provider — stub for extensibility proof."""

    name = "xai"
    endpoint = "https://api.x.ai/v1/chat/completions"

    async def authenticate(self) -> dict[str, str]:
        """Return xAI auth headers. Requires XAI_API_KEY env var."""
        raise NotImplementedError("xAI Grok provider not yet implemented")

    def translate_request(self, anthropic_req: dict) -> tuple[dict, list[str]]:
        """Translate Anthropic request to xAI format."""
        raise NotImplementedError("xAI Grok provider not yet implemented")

    def translate_response(self, provider_resp: dict) -> dict:
        """Translate xAI response back to Anthropic format."""
        raise NotImplementedError("xAI Grok provider not yet implemented")

    async def translate_stream(self, raw_chunks: AsyncIterator[bytes]) -> AsyncIterator[dict]:
        """Translate raw xAI byte chunks to Anthropic-format SSE events."""
        raise NotImplementedError("xAI Grok provider not yet implemented")


PROVIDERS["xai"] = XAIProvider
