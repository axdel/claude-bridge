"""OpenAI Codex provider — OAuth token management + Anthropic/OpenAI translation.

Split into cohesive submodules (auth, translate, stream, provider); this package
re-exports the public surface so ``claude_bridge.providers.openai`` keeps its
original import contract. Importing the package registers ``OpenAIProvider`` in
``PROVIDERS`` via the provider submodule.
"""

from __future__ import annotations

from claude_bridge.providers.openai.auth import (
    _DEFAULT_AUTH_PATH,
    get_bearer_token,
    read_codex_auth,
    refresh_access_token,
)
from claude_bridge.providers.openai.provider import _REASONING_CACHE_MAX, OpenAIProvider
from claude_bridge.providers.openai.stream import translate_openai_sse_event
from claude_bridge.providers.openai.translate import (
    DEFAULT_MODEL,
    GPT_TOKEN_COUNT_MULTIPLIER,
    _safe_token,
    _to_anthropic_id,
    _to_openai_id,
    anthropic_to_openai,
    openai_to_anthropic,
)

__all__ = [
    "DEFAULT_MODEL",
    "GPT_TOKEN_COUNT_MULTIPLIER",
    "_DEFAULT_AUTH_PATH",
    "_REASONING_CACHE_MAX",
    "OpenAIProvider",
    "_safe_token",
    "_to_anthropic_id",
    "_to_openai_id",
    "anthropic_to_openai",
    "get_bearer_token",
    "openai_to_anthropic",
    "read_codex_auth",
    "refresh_access_token",
    "translate_openai_sse_event",
]
