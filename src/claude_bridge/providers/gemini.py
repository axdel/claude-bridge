"""Google Gemini provider — API key auth + Anthropic/Gemini translation (stdlib only)."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

from claude_bridge.provider import PROVIDERS

_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

MODEL_MAP: dict[str, str] = {
    "claude-opus-4-6": "gemini-2.5-pro",
    "claude-sonnet-4-6": "gemini-2.5-flash",
    "claude-haiku-4-5-20251001": "gemini-2.5-flash",
}
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")


class GeminiProvider:
    """Google Gemini provider implementing the Provider protocol."""

    name = "gemini"

    def __init__(self) -> None:
        model = DEFAULT_MODEL
        self.endpoint = f"{_BASE_URL}/models/{model}"

    async def authenticate(self) -> dict[str, str]:
        """Return Gemini auth header. Requires GEMINI_API_KEY env var."""
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            msg = (
                "GEMINI_API_KEY environment variable is required "
                "for the Gemini provider but was not set or is empty."
            )
            raise ValueError(msg)
        return {"x-goog-api-key": api_key}

    def translate_request(self, anthropic_req: dict) -> tuple[dict, list[str]]:
        """Translate Anthropic Messages request to Gemini generateContent request."""
        raise NotImplementedError("Gemini request translation not yet implemented")

    def translate_response(self, provider_resp: dict) -> dict:
        """Translate Gemini response to Anthropic Messages response."""
        raise NotImplementedError("Gemini response translation not yet implemented")

    def translate_stream(self, raw_chunks: AsyncIterator[bytes]) -> AsyncIterator[dict]:
        """Translate Gemini SSE stream to Anthropic SSE events."""
        raise NotImplementedError("Gemini stream translation not yet implemented")


PROVIDERS["gemini"] = GeminiProvider
