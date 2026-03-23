"""Abstract provider interface for LLM API translation.

Every LLM provider (OpenAI, Gemini, xAI, etc.) implements the ``Provider``
protocol defined here. The proxy never imports provider-specific code —
it only uses this interface.

To add a new provider:

1. Create ``providers/<name>.py`` with a class implementing ``Provider``
2. ``translate_request``: Anthropic Messages -> your API format
3. ``translate_response``: your API format -> Anthropic Messages
4. ``translate_stream``: raw response bytes -> Anthropic SSE events
5. Register: ``PROVIDERS["<name>"] = YourProviderClass``
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable


@runtime_checkable
class Provider(Protocol):
    """Protocol that every LLM provider adapter must implement."""

    name: str

    async def authenticate(self) -> dict[str, str]:
        """Return headers required to authenticate with this provider."""
        ...

    def translate_request(self, anthropic_req: dict) -> tuple[dict, list[str]]:
        """Translate an Anthropic-format request to this provider's format.

        Returns (translated_request, warnings).
        """
        ...

    def translate_response(self, provider_resp: dict) -> dict:
        """Translate a provider response back to Anthropic format."""
        ...

    async def translate_stream(self, raw_chunks: AsyncIterator[bytes]) -> AsyncIterator[dict]:
        """Translate a stream of raw byte chunks to Anthropic-format SSE events.

        The proxy feeds raw HTTP response bytes; the provider owns SSE parsing
        and event translation.  Yields ``{event, data}`` dicts.
        """
        ...


# Provider registry — concrete implementations added in later tasks.
PROVIDERS: dict[str, type[Provider]] = {}
