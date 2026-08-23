"""OpenAIProvider — concrete Provider implementation.

Wires auth (bearer tokens), translation (request/response), and streaming into
the Provider protocol, and owns the in-memory encrypted-reasoning cache
(D-REASON-001 / D-CACHE-001). Registers itself in ``PROVIDERS`` on import.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

from claude_bridge.provider import PROVIDERS, ProviderCapabilities
from claude_bridge.providers.openai.auth import _validated_bearer, get_bearer_token
from claude_bridge.providers.openai.stream import (
    _remap_block_index,
    _sse_synthetic_termination,
    translate_openai_sse_event,
)
from claude_bridge.providers.openai.translate import (
    _API_KEY_CAPABILITIES,
    _CODEX_CAPABILITIES,
    GPT_TOKEN_COUNT_MULTIPLIER,
    _associate_reasoning_with_calls,
    _to_openai_id,
    anthropic_to_openai,
    openai_to_anthropic,
)
from claude_bridge.stream import iter_sse_event_blobs, parse_sse_events

_CODEX_ENDPOINT = "https://chatgpt.com/backend-api/codex/responses"
_API_KEY_ENDPOINT = "https://api.openai.com/v1/responses"

# Upper bound on the per-provider encrypted-reasoning cache (one entry per in-flight
# tool call). Bounds memory under long agentic sessions; oldest entries evict first.
_REASONING_CACHE_MAX = 256

# Upper bound on the undrained SSE buffer (one incomplete event). A well-formed
# stream drains on every "\n\n", so the buffer holds at most a single partial event.
# A provider that streams without event terminators would otherwise grow the buffer
# without limit (OOM) and make repeated concatenation quadratic; exceeding this cap
# means the stream is malformed, so we abort fast instead of accumulating.
_MAX_SSE_BUFFER = 4 * 1024 * 1024


class OpenAIProvider:
    """OpenAI provider implementing the Provider protocol.

    Supports two auth modes:
    - ``api_key``: uses an OpenAI API key (Bearer header to api.openai.com)
    - ``codex_oauth``: uses Codex OAuth flow (Bearer header to chatgpt.com)
    """

    name = "openai"
    capabilities = ProviderCapabilities(
        stream_request_mode="body_parameter",
        sync_response_mode="sse",
        token_count_multiplier=GPT_TOKEN_COUNT_MULTIPLIER,
    )

    def __init__(
        self,
        *,
        auth_mode: str = "codex_oauth",
        api_key: str | None = None,
        auth_path: Path | None = None,
    ) -> None:
        self.auth_mode = auth_mode
        self._api_key = api_key
        self._auth_path = auth_path
        # Sticky prompt cache identity: a process-stable UUID (the proxy holds one provider per
        # process), invariant across this instance's requests yet distinct across launchers. Random
        # UUID — never hash(instructions), never logged, embeds no secret (INV-SEC-01, INV-SEC-06).
        self._prompt_cache_key = str(uuid.uuid4())
        # Endpoint and input-content capabilities both vary by backend. The instance
        # attribute shadows the conservative class default so the proxy and
        # translate_request (which hold an instance) forward the modalities this
        # backend actually supports (D-MODALITY-001).
        if auth_mode == "api_key":
            self.endpoint = _API_KEY_ENDPOINT
            self.capabilities = _API_KEY_CAPABILITIES
        else:
            self.endpoint = _CODEX_ENDPOINT
            self.capabilities = _CODEX_CAPABILITIES
        # Encrypted reasoning blobs keyed by fc_ call id, captured from responses and
        # re-injected before their function_calls on the next request. In-memory only —
        # opaque, never persisted, never logged, never returned to Claude Code.
        self._reasoning_by_call_id: dict[str, dict] = {}
        self._reasoning_lock = threading.Lock()

    async def authenticate(self, *, force_refresh: bool = False) -> dict[str, str]:
        """Return Authorization header with a valid bearer token.

        Args:
            force_refresh: Force a token refresh regardless of proactive expiry — the
                reactive path after an upstream 401. A no-op in api_key mode (a static
                key has nothing to refresh); forwarded to ``get_bearer_token`` otherwise.
        """
        if self.auth_mode == "api_key":
            if not self._api_key:
                msg = (
                    "OPENAI_API_KEY environment variable is required for "
                    "api_key auth mode but was not set or is empty."
                )
                raise ValueError(msg)
            # Validate before header construction: a control-char-bearing key must not
            # reach an http.client "Invalid header value" error that echoes the secret
            # (CWE-20/CWE-113/CWE-532). The oauth branch is already validated at source
            # by get_bearer_token.
            return {"Authorization": f"Bearer {_validated_bearer(self._api_key)}"}
        token = await get_bearer_token(self._auth_path, force_refresh=force_refresh)
        return {"Authorization": f"Bearer {token}"}

    def _stash_reasoning(self, associations: dict[str, dict]) -> None:
        """Store captured reasoning blobs, refreshing recency and evicting the oldest
        entries once the cache exceeds its bound."""
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
        """Insert each cached reasoning item immediately before the function_call it
        belongs to, in-place on ``translated['input']``. Each reasoning item is inserted
        at most once (a duplicate item id would be rejected), so parallel calls sharing
        one reasoning item get a single preceding copy."""
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
                key = _to_openai_id(item.get("call_id") or item.get("id", ""))
                reasoning = cache.get(key)
                if reasoning is not None:
                    dedup_key = reasoning.get("id") or id(reasoning)
                    if dedup_key not in inserted:
                        new_input.append(reasoning)
                        inserted.add(dedup_key)
            new_input.append(item)
        translated["input"] = new_input

    def translate_request(self, anthropic_req: dict) -> tuple[dict, list[str]]:
        """Translate Anthropic Messages request to OpenAI Responses request, stamping this
        instance's sticky prompt cache key and echoing any captured encrypted reasoning back
        before its function_calls.

        The cache key and reasoning echo are stamped here rather than in the pure
        ``anthropic_to_openai`` translator because both are per-instance state (D-REASON-001).
        """
        result, warnings = anthropic_to_openai(anthropic_req, self.capabilities)
        result["prompt_cache_key"] = self._prompt_cache_key
        self._inject_reasoning(result)
        return result, warnings

    def translate_response(self, provider_resp: dict) -> dict:
        """Translate OpenAI Responses response to Anthropic Messages response, capturing
        each function_call's preceding encrypted reasoning for the next request."""
        self._stash_reasoning(_associate_reasoning_with_calls(provider_resp.get("output", [])))
        return openai_to_anthropic(
            provider_resp,
            token_count_multiplier=self.capabilities.token_count_multiplier,
        )

    def _capture_stream_reasoning(self, parsed_event: dict) -> None:
        """Capture encrypted reasoning from a streamed terminal event.

        Both ``response.completed`` and ``response.incomplete`` carry the output
        array (including reasoning items with ``encrypted_content``); an
        incomplete turn that still emitted a function_call needs its reasoning
        stashed too, or the next request's tool echo is rejected (D-REASON-001).
        """
        if parsed_event.get("event") not in ("response.completed", "response.incomplete"):
            return
        response_obj = (parsed_event.get("data") or {}).get("response") or {}
        self._stash_reasoning(_associate_reasoning_with_calls(response_obj.get("output", [])))

    async def translate_stream(self, raw_chunks: AsyncIterator[bytes]) -> AsyncIterator[dict]:
        """Translate raw provider byte chunks to Anthropic SSE events.

        Maintains a block index counter so Anthropic indices are sequential
        starting at 0 (OpenAI output_index may have gaps from skipped items) and
        fixes stop_reason based on whether tool calls were emitted.

        Termination invariant: a stream that emits ``message_start`` is always
        closed by a terminator — a ``message_stop`` (success/incomplete) or an
        ``error`` event (failure). If the upstream drops without any terminal
        event, a ``message_stop`` is synthesized so Claude Code finalizes the
        turn instead of hanging.
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
                for translated in translate_openai_sse_event(
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


PROVIDERS["openai"] = OpenAIProvider
