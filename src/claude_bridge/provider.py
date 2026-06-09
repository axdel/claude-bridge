"""Abstract provider interface for LLM API translation.

Every registered LLM provider implements the ``Provider`` protocol defined here.
The proxy never imports provider-specific code — it only uses this interface.

To add a new provider:

1. Create ``providers/<name>.py`` with a class implementing ``Provider``
2. Declare ``capabilities`` for proxy-visible transport behavior
3. ``translate_request``: Anthropic Messages -> your API format
4. ``translate_response``: your API format -> Anthropic Messages
5. ``translate_stream``: raw response bytes -> Anthropic SSE events
6. Register: ``PROVIDERS["<name>"] = YourProviderClass``
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

StreamRequestMode = Literal["body_parameter", "url"]
SyncResponseMode = Literal["json", "sse"]
_STREAM_REQUEST_MODES = frozenset({"body_parameter", "url"})
_SYNC_RESPONSE_MODES = frozenset({"json", "sse"})


@dataclass(frozen=True)
class ProviderCapabilities:
    """Provider-declared transport and accounting behavior for proxy orchestration.

    ``stream_request_mode`` declares whether streaming is selected by request body
    or endpoint URL. ``sync_response_mode`` declares the response format returned
    for non-streaming client requests. ``token_count_multiplier`` lets providers
    tune reported usage totals for local compatibility without changing provider
    wire parsing.
    """

    stream_request_mode: StreamRequestMode
    sync_response_mode: SyncResponseMode
    token_count_multiplier: float = 1.0

    def __post_init__(self) -> None:
        """Validate mode and accounting values at runtime for dynamic providers."""
        if self.stream_request_mode not in _STREAM_REQUEST_MODES:
            msg = f"Unknown stream_request_mode: {self.stream_request_mode!r}"
            raise ValueError(msg)
        if self.sync_response_mode not in _SYNC_RESPONSE_MODES:
            msg = f"Unknown sync_response_mode: {self.sync_response_mode!r}"
            raise ValueError(msg)
        if self.token_count_multiplier <= 0:
            msg = f"Invalid token_count_multiplier: {self.token_count_multiplier!r}"
            raise ValueError(msg)


@runtime_checkable
class Provider(Protocol):
    """Protocol that every LLM provider adapter must implement."""

    name: str
    endpoint: str
    capabilities: ProviderCapabilities

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

    def translate_stream(self, raw_chunks: AsyncIterator[bytes]) -> AsyncIterator[dict]:
        """Translate a stream of raw byte chunks to Anthropic-format SSE events.

        The proxy feeds raw HTTP response bytes; the provider owns SSE parsing
        and event translation.  Yields ``{event, data}`` dicts.
        """
        ...


# Provider registry — concrete implementations added in later tasks.
PROVIDERS: dict[str, type[Provider]] = {}
