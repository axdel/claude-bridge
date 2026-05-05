"""OpenAI Codex provider — OAuth token management + Anthropic/OpenAI translation (stdlib only)."""

from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import AsyncIterator
from pathlib import Path

from claude_bridge.auth import is_token_expired
from claude_bridge.provider import PROVIDERS
from claude_bridge.stream import parse_sse_events

_CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
_TOKEN_URL = "https://auth.openai.com/oauth/token"  # noqa: S105
_DEFAULT_AUTH_PATH = Path.home() / ".codex" / "auth.json"


def read_codex_auth(path: Path | None = None) -> dict:
    """Read and validate Codex auth.json.

    Raises:
        FileNotFoundError: If the auth file does not exist (hint: run ``codex login``).
        ValueError: If ``auth_mode`` is not ``"chatgpt"``.
    """
    auth_path = path or _DEFAULT_AUTH_PATH
    if not auth_path.exists():
        msg = f"Codex auth file not found at {auth_path}. Run `codex login` to authenticate first."
        raise FileNotFoundError(msg)

    data: dict = json.loads(auth_path.read_text())

    if data.get("auth_mode") != "chatgpt":
        msg = (
            f"Unsupported auth_mode '{data.get('auth_mode')}' — "
            "only 'chatgpt' auth_mode is supported."
        )
        raise ValueError(msg)

    return data


_refresh_lock = asyncio.Lock()


async def get_bearer_token(auth_path: Path | None = None) -> str:
    """Return a valid access token, refreshing if expired.

    Uses an asyncio.Lock to prevent concurrent refresh stampede — multiple
    callers with expired tokens share a single refresh operation.
    """
    async with _refresh_lock:
        data = read_codex_auth(auth_path)
        tokens = data.get("tokens", data)  # support both nested and flat structures
        token = tokens["access_token"]

        if not is_token_expired(token):
            return token

        new_token = await refresh_access_token(tokens["refresh_token"], auth_path=auth_path)
        return new_token


async def refresh_access_token(refresh_token: str, auth_path: Path | None = None) -> str:
    """Exchange a refresh token for a new access token.

    POSTs to the OpenAI token endpoint, updates the local auth.json
    atomically, and returns the new access_token.
    """
    resolved_path = auth_path or _DEFAULT_AUTH_PATH

    def _do_refresh() -> str:
        body = urllib.parse.urlencode(
            {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": _CODEX_CLIENT_ID,
            }
        ).encode()
        req = urllib.request.Request(_TOKEN_URL, data=body, method="POST")  # noqa: S310
        req.add_header("Content-Type", "application/x-www-form-urlencoded")

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
                token_data: dict = json.loads(resp.read())
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            OSError,
        ) as exc:
            raise ValueError(f"Token refresh failed: {exc}") from exc

        try:
            new_access_token: str = token_data["access_token"]
        except KeyError as exc:
            raise ValueError("Token refresh failed: response missing 'access_token'") from exc
        new_refresh_token: str = token_data.get("refresh_token", refresh_token)

        # Atomic write: tmp file + os.replace
        current = json.loads(resolved_path.read_text()) if resolved_path.exists() else {}
        current["access_token"] = new_access_token
        current["refresh_token"] = new_refresh_token

        tmp_path = resolved_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(current, indent=2))
        os.replace(tmp_path, resolved_path)

        return new_access_token

    return await asyncio.to_thread(_do_refresh)


# ---------------------------------------------------------------------------
# Anthropic <-> OpenAI translation (pure functions, no I/O)
# ---------------------------------------------------------------------------

MODEL_MAP: dict[str, str] = {
    "claude-opus-4-6": "gpt-5.5",
    "claude-sonnet-4-6": "gpt-5.5",
    "claude-haiku-4-5-20251001": "gpt-5.5",
}
DEFAULT_MODEL = "gpt-5.5"

_STRIPPED_KEYS = ("output_config",)

# Reasoning mode: "passthrough" preserves thinking blocks, "drop" strips them.
_REASONING_MODE = os.environ.get("REASONING_MODE", "passthrough").lower()


def _to_openai_id(anthropic_id: str) -> str:
    """Convert Anthropic tool ID to OpenAI Responses API format.

    Anthropic uses ``toolu_xxx`` or ``call_xxx``; OpenAI Responses API requires ``fc_xxx``.
    """
    if not anthropic_id:
        return anthropic_id
    if anthropic_id.startswith("fc_"):
        return anthropic_id
    if anthropic_id.startswith("call_"):
        return "fc_" + anthropic_id[5:]
    if anthropic_id.startswith("toolu_"):
        return "fc_" + anthropic_id[6:]
    return "fc_" + anthropic_id


def _to_anthropic_id(openai_id: str) -> str:
    """Convert OpenAI Responses API tool ID back to Anthropic format.

    OpenAI uses ``fc_xxx``; Claude Code requires ``toolu_xxx`` prefix.
    """
    if not openai_id:
        return openai_id
    if openai_id.startswith("toolu_"):
        return openai_id
    if openai_id.startswith("fc_"):
        return "toolu_" + openai_id[3:]
    if openai_id.startswith("call_"):
        return "toolu_" + openai_id[5:]
    return "toolu_" + openai_id


def _translate_content_block(block: dict) -> tuple[dict, list[str]]:
    """Translate a single Anthropic content block to OpenAI Responses API format.

    Returns (translated_block, warnings). For tool_use / tool_result blocks,
    the translated block has a special ``_toplevel`` key set to True, signaling
    the caller to emit it as a top-level input item rather than nesting it
    inside a message's content array.

    OpenAI Responses API field names (from official docs + CLASP proxy):
    - function_call: {type, id, call_id, name, arguments} — BOTH id and call_id required
    - function_call_output: {type, call_id, output} — output is always a string, never null
    """
    warnings: list[str] = []
    block_type = block.get("type")

    if block_type == "text":
        translated = {"type": "input_text", "text": block["text"]}
        return translated, warnings

    if block_type == "thinking":
        if _REASONING_MODE == "drop":
            warnings.append("Stripped thinking block (reasoning_mode=drop)")
            return {"type": "input_text", "text": ""}, warnings
        # Passthrough: preserve as tagged text
        thinking_text = block.get("thinking", "")
        return {
            "type": "input_text",
            "text": f"[thinking]\n{thinking_text}\n[/thinking]",
        }, warnings

    if block_type == "tool_use":
        # Anthropic uses toolu_xxx or call_xxx; OpenAI requires fc_xxx prefix
        fc_id = _to_openai_id(block["id"])
        return {
            "_toplevel": True,
            "type": "function_call",
            "id": fc_id,
            "call_id": fc_id,
            "name": block["name"],
            "arguments": json.dumps(block["input"]),
        }, warnings

    if block_type == "tool_result":
        content = block.get("content", "")
        if isinstance(content, list):
            parts = []
            for b in content:
                if b.get("type") == "text":
                    parts.append(b.get("text", ""))
                elif b.get("type") == "image":
                    source = b.get("source", {})
                    if source.get("type") == "base64":
                        media = source.get("media_type", "application/octet-stream")
                        data = source.get("data", "")
                        parts.append(f"[image: data:{media};base64,{data}]")
                    elif source.get("type") == "url":
                        parts.append(f"[image: {source.get('url', '')}]")
            content = "\n".join(parts)
        output = str(content) if content else ""
        if block.get("is_error"):
            output = f"[Error] {output}"
        # tool_use_id from Anthropic needs fc_ prefix for OpenAI
        fc_id = _to_openai_id(block["tool_use_id"])
        return {
            "_toplevel": True,
            "type": "function_call_output",
            "call_id": fc_id,
            "output": output,
        }, warnings

    # Unknown block type — pass through as input_text with warning
    warnings.append(f"Unknown content block type '{block_type}', converted to input_text")
    return {"type": "input_text", "text": str(block)}, warnings


def _translate_message(message: dict) -> tuple[list[dict], list[str]]:
    """Translate one Anthropic message to a list of OpenAI Responses API input items.

    Anthropic puts everything in messages with content blocks. The Responses API
    uses a flat input array where:
    - User text → {role: "user", content: [{type: "input_text", text: "..."}]}
    - Assistant text → {role: "assistant", content: [{type: "output_text", text: "..."}]}
    - Tool use (assistant) → top-level {type: "function_call", ...} items
    - Tool result (user) → top-level {type: "function_call_output", ...} items
    """
    warnings: list[str] = []
    role = message.get("role", "user")
    content = message.get("content", [])

    # String shorthand → single text block
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]

    nested_content: list[dict] = []
    toplevel_items: list[dict] = []

    for block in content:
        translated, block_warnings = _translate_content_block(block)
        warnings.extend(block_warnings)

        if translated.pop("_toplevel", False):
            toplevel_items.append(translated)
        else:
            # For assistant messages, text blocks become output_text not input_text
            if role == "assistant" and translated.get("type") == "input_text":
                translated = {"type": "output_text", "text": translated["text"]}
            nested_content.append(translated)

    items: list[dict] = []

    # Emit a regular message if there's any nested content
    if nested_content:
        items.append(
            {
                "role": role,
                "content": nested_content,
            }
        )

    # Emit top-level items (function_call, function_call_output)
    items.extend(toplevel_items)

    return items, warnings


def _has_cache_control(request: dict) -> bool:
    """Return True if any part of the request contains cache_control hints."""
    system = request.get("system")
    if isinstance(system, list) and any("cache_control" in b for b in system):
        return True
    if any("cache_control" in t for t in request.get("tools", [])):
        return True
    for msg in request.get("messages", []):
        content = msg.get("content", [])
        if isinstance(content, list) and any("cache_control" in b for b in content):
            return True
    return False


def anthropic_to_openai(request: dict) -> tuple[dict, list[str]]:
    """Translate an Anthropic Messages API request to an OpenAI Responses API request.

    Returns ``(translated_request, warnings)`` where warnings lists any
    features that were stripped because they have no OpenAI equivalent.

    Pure function — no I/O.
    """
    warnings: list[str] = []

    # Strip unsupported top-level keys
    for key in _STRIPPED_KEYS:
        if key in request:
            warnings.append(f"Stripped unsupported key '{key}' from request")

    # Handle thinking config based on reasoning mode
    if "thinking" in request:
        if _REASONING_MODE == "drop":
            warnings.append("Stripped 'thinking' config (reasoning_mode=drop)")
        else:
            warnings.append("Thinking config passed through (reasoning_mode=passthrough)")

    # Model mapping
    model = request.get("model", "")
    translated_model = MODEL_MAP.get(model, DEFAULT_MODEL)

    # Build result — Codex endpoint requires stream: true
    result: dict = {
        "model": translated_model,
        "reasoning": {"effort": "xhigh"},
        "store": False,
        "stream": True,
    }

    # System prompt → instructions (required by Codex endpoint)
    system = request.get("system")
    if system is not None:
        if isinstance(system, str):
            result["instructions"] = system
        elif isinstance(system, list):
            result["instructions"] = "\n".join(block.get("text", "") for block in system)
    else:
        result["instructions"] = "You are a helpful assistant."

    # Note: Codex backend endpoint does not support max_output_tokens or temperature.
    # These are silently dropped. The model uses its own defaults.

    # Tools — Responses API uses flat structure (no function wrapper)
    # strict: false because Anthropic tool schemas mark ALL params as required
    # but Claude Code only provides values for truly needed params
    if "tools" in request:
        result["tools"] = [
            {
                "type": "function",
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {}),
                "strict": False,
            }
            for tool in request["tools"]
        ]

    # Messages → input
    input_items: list[dict] = []
    for message in request.get("messages", []):
        items, msg_warnings = _translate_message(message)
        input_items.extend(items)
        warnings.extend(msg_warnings)

    result["input"] = input_items

    if _has_cache_control(request):
        warnings.append(
            "Stripped cache_control hints (no provider equivalent — caching is automatic)"
        )

    return result, warnings


def openai_to_anthropic(response: dict) -> dict:
    """Translate an OpenAI Responses API response to an Anthropic Messages API response.

    Pure function — no I/O.
    """
    # Map status → stop_reason
    status = response.get("status", "completed")
    output_items = response.get("output", [])
    has_tool_calls = any(i.get("type") == "function_call" for i in output_items)
    if has_tool_calls:
        stop_reason = "tool_use"
    elif status == "incomplete":
        stop_reason = "max_tokens"
    else:
        stop_reason = "end_turn"

    # Translate output items → content blocks
    content: list[dict] = []
    for item in response.get("output", []):
        item_type = item.get("type")

        if item_type == "message":
            # Extract text from message content
            for block in item.get("content", []):
                if block.get("type") == "output_text":
                    content.append({"type": "text", "text": block["text"]})

        elif item_type == "function_call":
            raw_args = item.get("arguments", "{}")
            try:
                parsed_args = json.loads(raw_args)
            except (json.JSONDecodeError, ValueError):
                parsed_args = {"_raw": raw_args}
            # Convert fc_xxx back to call_xxx for Anthropic
            oai_id = item.get("call_id") or item.get("id", "")
            content.append(
                {
                    "type": "tool_use",
                    "id": _to_anthropic_id(oai_id),
                    "name": item["name"],
                    "input": parsed_args,
                }
            )

    # Map usage
    oai_usage = response.get("usage") or {}
    usage = {
        "input_tokens": oai_usage.get("input_tokens", 0),
        "output_tokens": oai_usage.get("output_tokens", 0),
    }

    return {
        "id": f"msg_bridge_{response.get('id', 'unknown')}",
        "type": "message",
        "role": "assistant",
        "model": response.get("model", ""),
        "stop_reason": stop_reason,
        "content": content,
        "usage": usage,
    }


# ---------------------------------------------------------------------------
# SSE event translation: OpenAI Responses API → Anthropic Messages API
# ---------------------------------------------------------------------------


def _sse_response_created(data: dict) -> list[dict]:
    """Translate response.created → message_start + ping."""
    resp = data.get("response", {})
    usage = resp.get("usage") or {}
    return [
        {
            "event": "message_start",
            "data": {
                "type": "message_start",
                "message": {
                    "id": f"msg_bridge_{resp.get('id', 'unknown')}",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": resp.get("model", ""),
                    "stop_reason": None,
                    "usage": {
                        "input_tokens": usage.get("input_tokens", 0),
                        "output_tokens": 0,
                    },
                },
            },
        },
        {"event": "ping", "data": {"type": "ping"}},
    ]


def _sse_output_item_added(data: dict) -> list[dict]:
    """Translate response.output_item.added → content_block_start for function_call items."""
    item = data.get("item", {})
    output_index = data.get("output_index", 0)
    if item.get("type") != "function_call":
        return []
    oai_id = item.get("call_id") or item.get("id", "")
    anthropic_id = _to_anthropic_id(oai_id) if oai_id else f"call_bridge_{output_index}"
    return [
        {
            "event": "content_block_start",
            "data": {
                "type": "content_block_start",
                "index": output_index,
                "content_block": {
                    "type": "tool_use",
                    "id": anthropic_id,
                    "name": item.get("name", ""),
                    "input": {},
                },
            },
        }
    ]


def _sse_response_completed(data: dict) -> list[dict]:
    """Translate response.completed → message_delta + message_stop."""
    resp = data.get("response", {})
    usage = resp.get("usage") or {}
    status = resp.get("status", "completed")
    output = resp.get("output", [])
    has_tool_calls = any(i.get("type") == "function_call" for i in output)
    if has_tool_calls:
        stop_reason = "tool_use"
    elif status == "incomplete":
        stop_reason = "max_tokens"
    else:
        stop_reason = "end_turn"
    return [
        {
            "event": "message_delta",
            "data": {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason},
                "usage": {
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                },
            },
        },
        {"event": "message_stop", "data": {"type": "message_stop"}},
    ]


# Events that are informational — no Anthropic equivalent.
_SKIPPED_SSE_EVENTS = frozenset(
    {
        "response.in_progress",
        "response.queued",
        "response.content_part.done",
        "response.output_item.done",
    }
)


def translate_openai_sse_event(event: dict) -> list[dict]:
    """Translate one OpenAI Responses API SSE event to Anthropic SSE events.

    Dispatches to sub-handlers by event type. Returns a list of ``{event, data}``
    dicts (may be 0, 1, or 2 items). Pure function — no I/O.
    """
    event_type = event.get("event", "")
    data = event.get("data", {})

    if event_type == "response.created":
        return _sse_response_created(data)

    if event_type == "response.content_part.added":
        return [
            {
                "event": "content_block_start",
                "data": {
                    "type": "content_block_start",
                    "index": data.get("content_index", 0),
                    "content_block": {"type": "text", "text": ""},
                },
            }
        ]

    if event_type == "response.output_text.delta":
        return [
            {
                "event": "content_block_delta",
                "data": {
                    "type": "content_block_delta",
                    "index": data.get("content_index", 0),
                    "delta": {"type": "text_delta", "text": data.get("delta", "")},
                },
            }
        ]

    if event_type in (
        "response.output_text.done",
        "response.function_call_arguments.done",
    ):
        return [
            {
                "event": "content_block_stop",
                "data": {
                    "type": "content_block_stop",
                    "index": data.get("content_index", data.get("output_index", 0)),
                },
            }
        ]

    if event_type == "response.output_item.added":
        return _sse_output_item_added(data)

    if event_type == "response.function_call_arguments.delta":
        return [
            {
                "event": "content_block_delta",
                "data": {
                    "type": "content_block_delta",
                    "index": data.get("output_index", 0),
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": data.get("delta", ""),
                    },
                },
            }
        ]

    if event_type == "response.completed":
        return _sse_response_completed(data)

    if event_type in _SKIPPED_SSE_EVENTS:
        return []

    return []


def _remap_block_index(
    event: dict,
    index_map: dict[int, int],
    next_index: int,
    has_tool_calls: bool,
) -> tuple[dict, int, bool]:
    """Remap OpenAI output_index to sequential Anthropic block indices.

    Returns (possibly-modified event, updated next_index, updated has_tool_calls).
    """
    data = event.get("data", {})

    if event.get("event") == "content_block_start":
        oai_index = data.get("index", 0)
        index_map[oai_index] = next_index
        data["index"] = next_index
        if data.get("content_block", {}).get("type") == "tool_use":
            has_tool_calls = True
        return event, next_index + 1, has_tool_calls

    if event.get("event") in ("content_block_delta", "content_block_stop"):
        oai_index = data.get("index", 0)
        data["index"] = index_map.get(oai_index, oai_index)
        return event, next_index, has_tool_calls

    if event.get("event") == "message_delta" and has_tool_calls:
        delta = data.get("delta", {})
        if delta.get("stop_reason") == "end_turn":
            delta["stop_reason"] = "tool_use"
        return event, next_index, has_tool_calls

    return event, next_index, has_tool_calls


# ---------------------------------------------------------------------------
# Concrete Provider implementation
# ---------------------------------------------------------------------------

_CODEX_ENDPOINT = "https://chatgpt.com/backend-api/codex/responses"
_API_KEY_ENDPOINT = "https://api.openai.com/v1/responses"


class OpenAIProvider:
    """OpenAI provider implementing the Provider protocol.

    Supports two auth modes:
    - ``api_key``: uses an OpenAI API key (Bearer header to api.openai.com)
    - ``codex_oauth``: uses Codex OAuth flow (Bearer header to chatgpt.com)
    """

    name = "openai"

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
        if auth_mode == "api_key":
            self.endpoint = _API_KEY_ENDPOINT
        else:
            self.endpoint = _CODEX_ENDPOINT

    async def authenticate(self) -> dict[str, str]:
        """Return Authorization header with a valid bearer token."""
        if self.auth_mode == "api_key":
            if not self._api_key:
                msg = (
                    "OPENAI_API_KEY environment variable is required for "
                    "api_key auth mode but was not set or is empty."
                )
                raise ValueError(msg)
            return {"Authorization": f"Bearer {self._api_key}"}
        token = await get_bearer_token(self._auth_path)
        return {"Authorization": f"Bearer {token}"}

    def translate_request(self, anthropic_req: dict) -> tuple[dict, list[str]]:
        """Translate Anthropic Messages request to OpenAI Responses request."""
        return anthropic_to_openai(anthropic_req)

    def translate_response(self, provider_resp: dict) -> dict:
        """Translate OpenAI Responses response to Anthropic Messages response."""
        return openai_to_anthropic(provider_resp)

    async def translate_stream(self, raw_chunks: AsyncIterator[bytes]) -> AsyncIterator[dict]:
        """Translate raw provider byte chunks to Anthropic SSE events.

        Maintains a block index counter so Anthropic indices are sequential
        starting at 0 (OpenAI output_index may have gaps from skipped items).
        Also fixes stop_reason based on whether tool calls were emitted.
        """
        buffer = b""
        block_index = 0
        index_map: dict[int, int] = {}
        has_tool_calls = False

        async for chunk in raw_chunks:
            buffer += chunk
            buffer = buffer.replace(b"\r\n", b"\n")
            while b"\n\n" in buffer:
                event_end = buffer.index(b"\n\n") + 2
                event_bytes = buffer[:event_end]
                buffer = buffer[event_end:]
                for parsed_event in parse_sse_events(event_bytes):
                    for translated in translate_openai_sse_event(parsed_event):
                        translated, block_index, has_tool_calls = _remap_block_index(
                            translated, index_map, block_index, has_tool_calls
                        )
                        yield translated
        if buffer.strip():
            for parsed_event in parse_sse_events(buffer):
                for translated in translate_openai_sse_event(parsed_event):
                    translated, block_index, has_tool_calls = _remap_block_index(
                        translated, index_map, block_index, has_tool_calls
                    )
                    yield translated


PROVIDERS["openai"] = OpenAIProvider
