"""XAIProvider — concrete Provider implementation for xAI Grok.

Wires subscription-OAuth auth, request/response/stream translation, and the grok
cli-chat-proxy gate headers into the Provider protocol, and owns the in-memory
encrypted-reasoning cache (D-XAI-004). Registers itself in ``PROVIDERS`` on import.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import claude_bridge.config as config
from claude_bridge.provider import PROVIDERS, ProviderCapabilities
from claude_bridge.providers.xai.auth import _XAI_CLIENT_IDENTIFIER, get_xai_bearer_token
from claude_bridge.providers.xai.stream import (
    _remap_block_index,
    _sse_synthetic_termination,
    translate_xai_sse_event,
)
from claude_bridge.providers.xai.translate import (
    _XAI_TOKEN_COUNT_MULTIPLIER,
    anthropic_to_xai,
    xai_to_anthropic,
)
from claude_bridge.stream import iter_sse_event_blobs, parse_sse_events

# Undrained SSE buffer ceiling. A well-formed provider event is far under 4 MiB, so a stream
# that grows past this without a ``\n\n`` terminator is malformed and is aborted rather than
# buffered unboundedly. Duplicated (not imported from the OpenAI provider) to keep this module
# self-contained — cross-provider imports are forbidden; cli-chat-proxy speaks the same
# Responses wire, so the bound is identical by coincidence of contract, not by shared code.
_MAX_SSE_BUFFER = 4 * 1024 * 1024


# Bound on the in-memory encrypted-reasoning cache — an upper limit on distinct in-flight tool
# calls whose reasoning continuation is retained. Oldest entry is evicted (LRU) once exceeded.
_REASONING_CACHE_MAX = 256


def _associate_reasoning_with_calls(output: list[dict]) -> dict[str, dict]:
    """Map each tool call's ``call_id`` to the encrypted reasoning item that immediately
    precedes it in a Responses ``output`` array.

    Walks the output once: a reasoning item carrying ``encrypted_content`` becomes the pending
    continuation state for the next function_call; once paired (or interrupted by any other
    item) it is consumed. Unlike the openai provider, keys are the call's ``call_id`` (falling
    back to ``id``) VERBATIM — xAI round-trips its own ``call-<uuid>-<idx>`` identity unchanged,
    so the next request's function_calls look up by the same key with no ``fc_`` rewrite.

    Pure function — no I/O, no state.
    """
    associations: dict[str, dict] = {}
    pending: dict | None = None
    for item in output:
        if not isinstance(item, dict):
            pending = None
            continue
        item_type = item.get("type")
        if item_type == "reasoning":
            pending = item if item.get("encrypted_content") else None
        elif item_type == "function_call":
            if pending is not None:
                key = item.get("call_id") or item.get("id", "")
                if key:
                    associations[key] = pending
            pending = None
        else:
            pending = None
    return associations


# The subscription-metered proxy's Responses endpoint — billed against the grok CLI login,
# not a separate api.x.ai API key. Chosen over ``api.x.ai/v1/responses`` (see plan decision).
_XAI_ENDPOINT = "https://cli-chat-proxy.grok.com/v1/responses"

# Full provider capabilities: text + image + document input, array-form tool output, and the
# identity token-count multiplier (subscription-metered, no OpenAI-compat scaling). References
# the single ``_XAI_TOKEN_COUNT_MULTIPLIER`` owner rather than re-encoding the literal.
_XAI_CAPABILITIES = ProviderCapabilities(
    stream_request_mode="body_parameter",
    sync_response_mode="sse",
    input_modalities=frozenset({"text", "image", "document"}),
    supports_tool_output_content_parts=True,
    token_count_multiplier=_XAI_TOKEN_COUNT_MULTIPLIER,
)


class XAIProvider:
    """xAI Grok provider — subscription-OAuth backend on cli-chat-proxy.

    Request, non-stream response, and streaming translation are implemented
    (``anthropic_to_xai``, ``xai_to_anthropic``, and ``translate_xai_sse_event`` above), plus
    encrypted-reasoning continuity: the reasoning item preceding each tool call is captured
    (from a response or a streamed terminal) and echoed back before its function_call on the
    next request, so Grok can resume its own chain of thought across tool turns. ``authenticate``
    resolves the grok subscription bearer from ``~/.grok/auth.json`` (refreshing via OIDC when
    expired) and pairs it with the ``x-grok-client-version`` / ``x-grok-client-identifier``
    headers the proxy gates on. Registered as ``PROVIDERS["xai"]`` at module import.
    """

    name = "xai"
    endpoint = _XAI_ENDPOINT
    capabilities = _XAI_CAPABILITIES

    def __init__(self, *, auth_path: Path | None = None) -> None:
        # Optional override of ~/.grok/auth.json for testing; the no-arg default resolves the
        # real subscription file, so the fallback path's ``provider_cls()`` construction works.
        self._auth_path = auth_path
        # Sticky prompt cache identity: a process-stable UUID (the proxy holds one provider per
        # process), invariant across this instance's requests yet distinct across launchers. Random
        # UUID — never hash(instructions), never logged, embeds no secret (INV-SEC-01, INV-SEC-06).
        self._prompt_cache_key = str(uuid.uuid4())
        # Encrypted reasoning items captured from each tool turn, keyed by the EXACT upstream
        # call_id, so they can be re-injected before their function_calls on the next request.
        # In-memory only — opaque, never persisted, never logged, never returned to Claude Code.
        self._reasoning_by_call_id: dict[str, dict] = {}
        self._reasoning_lock = threading.Lock()

    async def authenticate(self, *, force_refresh: bool = False) -> dict[str, str]:
        """Return the subscription bearer plus the grok client headers.

        Divergence from Codex (bearer-only): cli-chat-proxy rejects a request whose
        ``x-grok-client-version`` is older than its floor or that lacks a client identifier,
        so both accompany the bearer. The opaque bearer is never logged and rides only in the
        ``Authorization`` header.

        The ``x-grok-conv-id`` carries this instance's sticky prompt cache key so cli-chat-proxy
        routes the conversation to its cached prefix (grok-build resolves cache identity as
        ``key.or(conv_id)``, so key and header carry the same value). It is an opaque per-instance
        UUID — never a secret.

        Args:
            force_refresh: Force a bearer refresh regardless of proactive expiry — the
                reactive path after an upstream 401. Forwarded to ``get_xai_bearer_token``.

        Raises:
            FileNotFoundError / ValueError: Propagated from ``get_xai_bearer_token`` when the
                grok auth file is absent, malformed, or expired with no refresh token.
        """
        token = await get_xai_bearer_token(self._auth_path, force_refresh=force_refresh)
        return {
            "Authorization": f"Bearer {token}",
            "x-grok-client-version": config.xai_client_version(),
            "x-grok-client-identifier": _XAI_CLIENT_IDENTIFIER,
            "x-grok-conv-id": self._prompt_cache_key,
        }

    def _stash_reasoning(self, associations: dict[str, dict]) -> None:
        """Store captured reasoning blobs, refreshing recency and evicting the oldest entries
        once the cache exceeds its bound (LRU)."""
        if not associations:
            return
        with self._reasoning_lock:
            for call_id, reasoning in associations.items():
                self._reasoning_by_call_id.pop(call_id, None)
                self._reasoning_by_call_id[call_id] = reasoning
            while len(self._reasoning_by_call_id) > _REASONING_CACHE_MAX:
                oldest = next(iter(self._reasoning_by_call_id))
                del self._reasoning_by_call_id[oldest]

    def _inject_reasoning(self, translated: dict) -> None:
        """Insert each cached reasoning item immediately before the function_call it belongs to,
        in-place on ``translated['input']``.

        Each reasoning item is inserted at most once (dedup by its id), so parallel calls
        sharing one reasoning item get a single preceding copy. Keys match by the verbatim
        call_id — no ``fc_`` rewrite, mirroring the request-side function_call identity.
        """
        input_items = translated.get("input")
        if not isinstance(input_items, list):
            return
        with self._reasoning_lock:
            if not self._reasoning_by_call_id:
                return
            cache = dict(self._reasoning_by_call_id)
        new_input: list[dict] = []
        inserted: set = set()
        for item in input_items:
            if item.get("type") == "function_call":
                key = item.get("call_id") or item.get("id", "")
                reasoning = cache.get(key)
                if reasoning is not None:
                    dedup_key = reasoning.get("id") or id(reasoning)
                    if dedup_key not in inserted:
                        new_input.append(reasoning)
                        inserted.add(dedup_key)
            new_input.append(item)
        translated["input"] = new_input

    def translate_request(self, anthropic_req: dict) -> tuple[dict, list[str]]:
        """Translate an Anthropic Messages request to an xAI Responses request, stamping this
        instance's sticky prompt cache key and echoing any captured encrypted reasoning back
        before its function_calls.

        The cache key and reasoning echo are stamped here rather than in the pure
        ``anthropic_to_xai`` translator because both are per-instance state (D-REASON-001).
        """
        result, warnings = anthropic_to_xai(anthropic_req, self.capabilities)
        result["prompt_cache_key"] = self._prompt_cache_key
        self._inject_reasoning(result)
        return result, warnings

    def translate_response(self, provider_resp: dict) -> dict:
        """Translate an xAI Responses object back to Anthropic Messages format, capturing each
        function_call's preceding encrypted reasoning for the next request."""
        self._stash_reasoning(_associate_reasoning_with_calls(provider_resp.get("output", [])))
        return xai_to_anthropic(
            provider_resp,
            token_count_multiplier=self.capabilities.token_count_multiplier,
        )

    def _capture_stream_reasoning(self, parsed_event: dict) -> None:
        """Capture encrypted reasoning from a streamed terminal event.

        Both ``response.completed`` and ``response.incomplete`` carry the output array
        (reasoning items with ``encrypted_content`` included); an incomplete turn that still
        emitted a function_call needs its reasoning stashed too, or the next request's tool echo
        loses its continuation state.
        """
        if parsed_event.get("event") not in ("response.completed", "response.incomplete"):
            return
        response_obj = (parsed_event.get("data") or {}).get("response") or {}
        self._stash_reasoning(_associate_reasoning_with_calls(response_obj.get("output", [])))

    async def translate_stream(self, raw_chunks: AsyncIterator[bytes]) -> AsyncIterator[dict]:
        """Translate raw xAI byte chunks to Anthropic-format SSE events.

        Maintains a block-index counter so Anthropic indices are sequential from 0 (xAI
        output_index may have gaps from skipped reasoning items) and fixes stop_reason based
        on whether tool calls were emitted.

        Termination invariant: a stream that emits ``message_start`` is always closed by a
        terminator — a ``message_stop`` (success/incomplete) or an ``error`` event (failure).
        If the upstream drops without any terminal event, a ``message_stop`` is synthesized so
        Claude Code finalizes the turn instead of hanging.
        """
        block_index = 0
        index_map: dict[int, int] = {}
        has_tool_calls = False
        started = False
        terminated = False

        def _emit(event_bytes: bytes) -> list[dict]:
            """Translate one SSE blob, threading block-index and lifecycle state."""
            nonlocal block_index, has_tool_calls, started, terminated
            out: list[dict] = []
            for parsed_event in parse_sse_events(event_bytes):
                self._capture_stream_reasoning(parsed_event)
                for translated in translate_xai_sse_event(
                    parsed_event,
                    token_count_multiplier=self.capabilities.token_count_multiplier,
                ):
                    translated, block_index, has_tool_calls = _remap_block_index(
                        translated, index_map, block_index, has_tool_calls
                    )
                    name = translated.get("event")
                    if name == "message_start":
                        started = True
                    elif name in ("message_stop", "error"):
                        terminated = True
                    out.append(translated)
            return out

        async for event_bytes in iter_sse_event_blobs(raw_chunks, max_buffer=_MAX_SSE_BUFFER):
            for translated in _emit(event_bytes):
                yield translated

        if started and not terminated:
            for translated in _sse_synthetic_termination(has_tool_calls):
                yield translated


PROVIDERS["xai"] = XAIProvider
