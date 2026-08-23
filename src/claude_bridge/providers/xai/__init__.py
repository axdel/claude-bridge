"""xAI Grok provider — subscription OAuth + Anthropic/xAI Responses translation.

Split into cohesive submodules (auth, translate, stream, provider); this package
re-exports the public surface so ``claude_bridge.providers.xai`` keeps its original
import contract. Importing the package registers ``XAIProvider`` in ``PROVIDERS`` via
the provider submodule. Duplicates the OpenAI provider's Responses translation by
design — cross-provider imports are forbidden (D-XAI-002).
"""

from __future__ import annotations

from claude_bridge.providers.xai.auth import (
    _XAI_CLIENT_IDENTIFIER,
    _iso_to_timestamp,
    _validated_bearer,
    _xai_token_expired,
    get_xai_bearer_token,
    read_xai_auth,
    refresh_xai_token,
)
from claude_bridge.providers.xai.provider import (
    _MAX_SSE_BUFFER,
    _REASONING_CACHE_MAX,
    _XAI_CAPABILITIES,
    _XAI_ENDPOINT,
    XAIProvider,
    _associate_reasoning_with_calls,
)
from claude_bridge.providers.xai.stream import translate_xai_sse_event
from claude_bridge.providers.xai.translate import (
    _XAI_TOKEN_COUNT_MULTIPLIER,
    _safe_document_filename,
    _tool_result_parts,
    _translate_document_block,
    _translate_image_block,
    anthropic_to_xai,
    xai_to_anthropic,
)

__all__ = [
    "_MAX_SSE_BUFFER",
    "_REASONING_CACHE_MAX",
    "_XAI_CAPABILITIES",
    "_XAI_CLIENT_IDENTIFIER",
    "_XAI_ENDPOINT",
    "_XAI_TOKEN_COUNT_MULTIPLIER",
    "XAIProvider",
    "_associate_reasoning_with_calls",
    "_iso_to_timestamp",
    "_safe_document_filename",
    "_tool_result_parts",
    "_translate_document_block",
    "_translate_image_block",
    "_validated_bearer",
    "_xai_token_expired",
    "anthropic_to_xai",
    "get_xai_bearer_token",
    "read_xai_auth",
    "refresh_xai_token",
    "translate_xai_sse_event",
    "xai_to_anthropic",
]
