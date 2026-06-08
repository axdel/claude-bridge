"""Google Gemini provider — OAuth + API key auth, Anthropic/Gemini translation (stdlib only)."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import AsyncIterator
from pathlib import Path

import claude_bridge.config as config
from claude_bridge.provider import PROVIDERS

_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
_CODE_ASSIST_URL = "https://cloudcode-pa.googleapis.com/v1internal"

_GEMINI_CLIENT_ID = "681255809395-oo8ft2oprdrnp9e3aqf6av3hmdib135j.apps.googleusercontent.com"
_GEMINI_CLIENT_SECRET = "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"  # noqa: S105  # gitleaks:allow (intentionally public — Google desktop app credential, same as Gemini CLI source)
_GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"  # noqa: S105
_DEFAULT_GEMINI_AUTH_PATH = Path.home() / ".gemini" / "oauth_creds.json"

MODEL_MAP: dict[str, str] = {
    "claude-opus-4-6": "gemini-2.5-pro",
    "claude-sonnet-4-6": "gemini-2.5-flash",
    "claude-haiku-4-5-20251001": "gemini-2.5-flash",
}
DEFAULT_MODEL = config.gemini_api_key_model()

_STRIPPED_KEYS = ("output_config",)

# Upper bound on user-controlled tokens embedded in warnings/placeholders.
_SAFE_TOKEN_MAX = 64


def _safe_token(value: object) -> str:
    """Return a printable, length-bounded token safe for warnings and traces."""
    cleaned = "".join(ch for ch in str(value) if ch.isprintable())
    if len(cleaned) > _SAFE_TOKEN_MAX:
        return cleaned[:_SAFE_TOKEN_MAX] + "..."
    return cleaned


# ---------------------------------------------------------------------------
# ID encoding — round-trip Gemini tool call IDs + optional thoughtSignature
# through Anthropic's opaque tool_use ID field.
#
# Format: "call_gemini_<id>" or "call_gemini_<id>:<base64_sig>" when signature present.
# ---------------------------------------------------------------------------

_GEMINI_ID_PREFIX = "toolu_gemini_"


def _encode_tool_id(gemini_id: str, thought_signature: str | None = None) -> str:
    """Encode a Gemini function call ID + optional thoughtSignature into an Anthropic tool ID."""
    base = f"{_GEMINI_ID_PREFIX}{gemini_id}"
    if thought_signature:
        sig_b64 = base64.urlsafe_b64encode(thought_signature.encode()).decode()
        return f"{base}:{sig_b64}"
    return base


_LEGACY_GEMINI_ID_PREFIX = "call_gemini_"


def _decode_tool_id(anthropic_id: str) -> tuple[str, str | None]:
    """Decode an Anthropic tool ID back to (gemini_id, thought_signature | None)."""
    if anthropic_id.startswith(_GEMINI_ID_PREFIX):
        remainder = anthropic_id[len(_GEMINI_ID_PREFIX) :]
    elif anthropic_id.startswith(_LEGACY_GEMINI_ID_PREFIX):
        remainder = anthropic_id[len(_LEGACY_GEMINI_ID_PREFIX) :]
    else:
        return anthropic_id, None
    if ":" in remainder:
        gemini_id, sig_b64 = remainder.split(":", 1)
        try:
            sig = base64.urlsafe_b64decode(sig_b64).decode()
        except Exception:
            sig = None
        return gemini_id, sig
    return remainder, None


# ---------------------------------------------------------------------------
# Anthropic → Gemini request translation (pure function, no I/O)
# ---------------------------------------------------------------------------


def _translate_block(
    block: dict,
    tool_id_to_name: dict[str, str],
    warnings: list[str],
) -> dict | None:
    """Translate a single Anthropic content block to a Gemini part. Returns None to skip."""
    block_type = block.get("type")

    if block_type == "text":
        return {"text": block["text"]}

    if block_type == "thinking":
        warnings.append("Stripped thinking block (no Gemini equivalent)")
        return None

    if block_type == "tool_use":
        tool_id_to_name[block["id"]] = block["name"]
        fc: dict = {"name": block["name"], "args": block.get("input", {})}
        _, sig = _decode_tool_id(block["id"])
        if sig:
            return {"functionCall": fc, "thoughtSignature": sig}
        return {"functionCall": fc}

    if block_type == "tool_result":
        return _translate_tool_result(block, tool_id_to_name)

    safe_type = _safe_token(block_type)
    warnings.append(
        f"Unsupported content block type '{safe_type}' replaced with a redacted "
        "placeholder (no provider equivalent)"
    )
    return {"text": f"[unsupported content block: {safe_type}]"}


def _translate_tool_result(block: dict, tool_id_to_name: dict[str, str]) -> dict:
    """Translate an Anthropic tool_result block to a Gemini functionResponse part."""
    tool_use_id = block.get("tool_use_id", "")
    name = tool_id_to_name.get(tool_use_id, "unknown")

    raw_content = block.get("content", "")
    if isinstance(raw_content, list):
        text_parts = [b.get("text", "") for b in raw_content if b.get("type") == "text"]
        raw_content = "\n".join(text_parts)
    raw_content = str(raw_content) if raw_content else ""

    try:
        response_obj = json.loads(raw_content)
    except (json.JSONDecodeError, ValueError):
        response_obj = {"result": raw_content}

    fr: dict = {"name": name, "response": response_obj}
    gemini_id, _ = _decode_tool_id(tool_use_id)
    if gemini_id != tool_use_id:
        fr["id"] = gemini_id
    return {"functionResponse": fr}


def _translate_messages(messages: list[dict], warnings: list[str]) -> list[dict]:
    """Translate Anthropic messages array to Gemini contents array."""
    tool_id_to_name: dict[str, str] = {}
    contents: list[dict] = []
    for message in messages:
        role = message.get("role", "user")
        gemini_role = "model" if role == "assistant" else "user"
        content = message.get("content", [])
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]

        parts = [
            p
            for block in content
            if (p := _translate_block(block, tool_id_to_name, warnings)) is not None
        ]
        if parts:
            contents.append({"role": gemini_role, "parts": parts})
    return contents


# Gemini accepts a strict subset of JSON Schema (OpenAPI 3.0.3 subset).
# Allowlist approach: only pass through known-supported keywords.
# Source: https://ai.google.dev/api/caching#Schema
_SUPPORTED_SCHEMA_KEYS = frozenset(
    {
        "type",
        "format",
        "title",
        "description",
        "nullable",
        "default",
        "example",
        "enum",
        "properties",
        "required",
        "minProperties",
        "maxProperties",
        "propertyOrdering",
        "items",
        "minItems",
        "maxItems",
        "minLength",
        "maxLength",
        "pattern",
        "minimum",
        "maximum",
        "anyOf",
    }
)


def _clean_schema(schema: dict) -> dict:
    """Recursively keep only Gemini-supported JSON Schema keywords.

    The `properties` value is a map of field names → sub-schemas, so we clean
    each sub-schema but preserve the field names (they aren't schema keywords).
    """
    cleaned: dict = {}
    for k, v in schema.items():
        if k not in _SUPPORTED_SCHEMA_KEYS:
            continue
        if k == "properties" and isinstance(v, dict):
            cleaned[k] = {prop: _clean_schema(sub) for prop, sub in v.items()}
        elif k == "anyOf" and isinstance(v, list):
            cleaned[k] = [_clean_schema(item) if isinstance(item, dict) else item for item in v]
        elif isinstance(v, dict):
            cleaned[k] = _clean_schema(v)
        elif isinstance(v, list):
            cleaned[k] = [_clean_schema(item) if isinstance(item, dict) else item for item in v]
        else:
            cleaned[k] = v
    return cleaned


def anthropic_to_gemini(request: dict) -> tuple[dict, list[str]]:
    """Translate an Anthropic Messages API request to a Gemini generateContent request.

    Returns (translated_request, warnings). Pure function — no I/O.
    """
    warnings: list[str] = []

    for key in _STRIPPED_KEYS:
        if key in request:
            warnings.append(f"Stripped unsupported key '{key}' from request")

    if "thinking" in request:
        warnings.append("Stripped 'thinking' config (no Gemini equivalent)")

    model = request.get("model", "")
    result: dict = {"model": MODEL_MAP.get(model, DEFAULT_MODEL)}

    # System prompt → system_instruction
    system = request.get("system")
    if isinstance(system, str):
        result["system_instruction"] = {"parts": [{"text": system}]}
    elif isinstance(system, list):
        joined = "\n".join(block.get("text", "") for block in system)
        result["system_instruction"] = {"parts": [{"text": joined}]}

    # Tools → function_declarations (strip $schema — Gemini rejects it)
    if "tools" in request:
        result["tools"] = [
            {
                "function_declarations": [
                    {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": _clean_schema(t.get("input_schema", {})),
                    }
                    for t in request["tools"]
                ]
            }
        ]

    result["contents"] = _translate_messages(request.get("messages", []), warnings)

    if _has_cache_control(request):
        warnings.append(
            "Stripped cache_control hints (no provider equivalent — caching is automatic)"
        )

    return result, warnings


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


# ---------------------------------------------------------------------------
# Gemini → Anthropic response translation (pure function, no I/O)
# ---------------------------------------------------------------------------


_SAFETY_REFUSAL = (
    "I cannot fulfill this request due to content safety policies. Please rephrase your request."
)


def gemini_to_anthropic(response: dict) -> dict:
    """Translate a Gemini generateContent response to Anthropic Messages format.

    Pure function — no I/O.
    """
    candidates = response.get("candidates", [])
    candidate = candidates[0] if candidates else {}
    finish_reason = candidate.get("finishReason", "STOP")
    parts = candidate.get("content", {}).get("parts", [])

    # Translate parts → Anthropic content blocks
    content: list[dict] = []
    has_tool_calls = False
    for idx, part in enumerate(parts):
        if "text" in part and part["text"] and not part.get("thought"):
            content.append({"type": "text", "text": part["text"]})
        elif "functionCall" in part:
            has_tool_calls = True
            fc = part["functionCall"]
            gemini_id = fc.get("id", f"gemini_{idx}")
            sig = part.get("thoughtSignature")
            anthropic_id = _encode_tool_id(gemini_id, sig)
            content.append(
                {
                    "type": "tool_use",
                    "id": anthropic_id,
                    "name": fc["name"],
                    "input": fc.get("args", {}),
                }
            )

    # Handle SAFETY finish with no content
    if finish_reason == "SAFETY" and not content:
        content.append({"type": "text", "text": _SAFETY_REFUSAL})

    # Map stop reason
    if has_tool_calls:
        stop_reason = "tool_use"
    elif finish_reason == "MAX_TOKENS":
        stop_reason = "max_tokens"
    else:
        stop_reason = "end_turn"

    # Usage
    usage_meta = response.get("usageMetadata", {})
    usage = {
        "input_tokens": usage_meta.get("promptTokenCount", 0),
        "output_tokens": usage_meta.get("candidatesTokenCount", 0),
    }

    resp_id = response.get("responseId", "unknown")
    return {
        "id": f"msg_bridge_{resp_id}",
        "type": "message",
        "role": "assistant",
        "model": response.get("modelVersion", ""),
        "stop_reason": stop_reason,
        "content": content,
        "usage": usage,
    }


# ---------------------------------------------------------------------------
# SSE stream translation: Gemini → Anthropic SSE events
# ---------------------------------------------------------------------------


def _sse_text_delta(index: int, text: str) -> dict:
    """Build an Anthropic content_block_delta event for text."""
    return {
        "event": "content_block_delta",
        "data": {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "text_delta", "text": text},
        },
    }


def _sse_tool_use_block(index: int, tool_id: str, name: str) -> list[dict]:
    """Build content_block_start + delta events for a function call."""
    return [
        {
            "event": "content_block_start",
            "data": {
                "type": "content_block_start",
                "index": index,
                "content_block": {
                    "type": "tool_use",
                    "id": tool_id,
                    "name": name,
                    "input": {},
                },
            },
        },
    ]


def _sse_text_block_start(index: int) -> dict:
    """Build an Anthropic content_block_start event for text."""
    return {
        "event": "content_block_start",
        "data": {
            "type": "content_block_start",
            "index": index,
            "content_block": {"type": "text", "text": ""},
        },
    }


def _sse_block_stop(index: int) -> dict:
    """Build an Anthropic content_block_stop event."""
    return {
        "event": "content_block_stop",
        "data": {"type": "content_block_stop", "index": index},
    }


def _sse_message_start(response_id: str, model: str, usage: dict) -> dict:
    """Build an Anthropic message_start event from Gemini response metadata."""
    return {
        "event": "message_start",
        "data": {
            "type": "message_start",
            "message": {
                "id": f"msg_bridge_{response_id}",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": None,
                "usage": {
                    "input_tokens": usage.get("promptTokenCount", 0),
                    "output_tokens": 0,
                },
            },
        },
    }


def _close_text_block(events: list[dict], state: dict, block_index: int, *, advance: bool) -> int:
    """Close an open text block and optionally advance to the next block index."""
    if not state.get("text_block_open"):
        return block_index
    events.append(_sse_block_stop(block_index))
    state["text_block_open"] = False
    return block_index + 1 if advance else block_index


def _append_text_part(events: list[dict], state: dict, block_index: int, text: str) -> None:
    """Append Gemini text as Anthropic text-block events."""
    if not state.get("text_block_open"):
        state["text_block_open"] = True
        events.append(_sse_text_block_start(block_index))
    events.append(_sse_text_delta(block_index, text))


def _append_tool_call_part(events: list[dict], part: dict, state: dict, block_index: int) -> int:
    """Append a Gemini functionCall as Anthropic tool_use events."""
    block_index = _close_text_block(events, state, block_index, advance=True)
    state["has_tool_calls"] = True
    function_call = part["functionCall"]
    gemini_id = function_call.get("id", f"gemini_{block_index}")
    tool_id = _encode_tool_id(gemini_id, part.get("thoughtSignature"))
    events.extend(_sse_tool_use_block(block_index, tool_id, function_call["name"]))
    events.append(
        {
            "event": "content_block_delta",
            "data": {
                "type": "content_block_delta",
                "index": block_index,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": json.dumps(function_call.get("args", {})),
                },
            },
        }
    )
    events.append(_sse_block_stop(block_index))
    return block_index + 1


def _append_part_events(events: list[dict], part: dict, state: dict, block_index: int) -> int:
    """Append Anthropic events for one Gemini part and return the next block index."""
    if "text" in part and not part.get("thought") and part["text"]:
        _append_text_part(events, state, block_index, part["text"])
        return block_index
    if "functionCall" in part:
        return _append_tool_call_part(events, part, state, block_index)
    return block_index


def _gemini_stop_reason(finish_reason: str, has_tools: bool) -> str:
    """Map Gemini finishReason to Anthropic stop_reason."""
    if finish_reason == "MAX_TOKENS":
        return "max_tokens"
    if has_tools:
        return "tool_use"
    return "end_turn"


def _append_safety_refusal(events: list[dict]) -> None:
    """Append a synthetic refusal text block for Gemini SAFETY-only responses."""
    events.append(_sse_text_block_start(0))
    events.append(_sse_text_delta(0, _SAFETY_REFUSAL))
    events.append(_sse_block_stop(0))


def _append_finish_events(
    events: list[dict],
    finish_reason: str,
    usage: dict,
    state: dict,
    block_index: int,
) -> None:
    """Append terminal Anthropic message events for a Gemini finish reason."""
    _close_text_block(events, state, block_index, advance=False)
    if finish_reason == "SAFETY" and block_index == 0:
        _append_safety_refusal(events)
    events.append(
        {
            "event": "message_delta",
            "data": {
                "type": "message_delta",
                "delta": {
                    "stop_reason": _gemini_stop_reason(
                        finish_reason,
                        bool(state.get("has_tool_calls", False)),
                    )
                },
                "usage": {
                    "input_tokens": usage.get("promptTokenCount", 0),
                    "output_tokens": usage.get("candidatesTokenCount", 0),
                },
            },
        }
    )
    events.append({"event": "message_stop", "data": {"type": "message_stop"}})


def translate_gemini_sse_chunk(chunk: dict, state: dict) -> list[dict]:
    """Translate one Gemini SSE chunk to Anthropic SSE events.

    *state* is a mutable dict tracking: first_chunk, block_index, response_id, model.
    Returns a list of {event, data} dicts.
    """
    events: list[dict] = []
    candidates = chunk.get("candidates", [])
    candidate = candidates[0] if candidates else {}
    parts = candidate.get("content", {}).get("parts", [])
    finish_reason = candidate.get("finishReason")
    usage = chunk.get("usageMetadata", {})

    if not state.get("started"):
        state["started"] = True
        response_id = chunk.get("responseId", "unknown")
        model = chunk.get("modelVersion", "")
        state["response_id"] = response_id
        state["model"] = model
        events.append(_sse_message_start(response_id, model, usage))
        events.append({"event": "ping", "data": {"type": "ping"}})

    block_index = state.get("block_index", 0)
    for part in parts:
        block_index = _append_part_events(events, part, state, block_index)
    state["block_index"] = block_index

    if finish_reason:
        _append_finish_events(events, finish_reason, usage, state, block_index)

    return events


def _parse_sse_data(raw: bytes, state: dict) -> list[dict]:
    """Parse SSE data lines from raw bytes and translate to Anthropic events."""
    events: list[dict] = []
    for line in raw.decode("utf-8", errors="replace").split("\n"):
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data_str = line[5:].strip()
        if not data_str or data_str == "[DONE]":
            continue
        try:
            parsed = json.loads(data_str)
        except (json.JSONDecodeError, ValueError):
            continue
        # Unwrap Code Assist envelope if present
        if "response" in parsed and "candidates" not in parsed:
            parsed = parsed["response"]
        events.extend(translate_gemini_sse_chunk(parsed, state))
    return events


# ---------------------------------------------------------------------------
# OAuth token management — reads ~/.gemini/oauth_creds.json (same pattern as OpenAI Codex)
# ---------------------------------------------------------------------------


def read_gemini_auth(path: Path | None = None) -> dict:
    """Read Gemini CLI OAuth credentials.

    Raises:
        FileNotFoundError: If oauth_creds.json doesn't exist (hint: run ``gemini login``).
    """
    auth_path = path or _DEFAULT_GEMINI_AUTH_PATH
    if not auth_path.exists():
        msg = (
            f"Gemini auth file not found at {auth_path}. Run `gemini login` to authenticate first."
        )
        raise FileNotFoundError(msg)
    return json.loads(auth_path.read_text())


def _is_gemini_token_expired(creds: dict, margin_ms: int = 30_000) -> bool:
    """Check if the Gemini OAuth token is expired using the expiry_date field.

    Gemini CLI stores ``expiry_date`` as a Unix timestamp in milliseconds.
    Returns True if expired or expiry_date is missing.
    """
    import time

    expiry_ms = creds.get("expiry_date")
    if expiry_ms is None:
        return True
    now_ms = int(time.time() * 1000)
    return now_ms + margin_ms >= expiry_ms


_gemini_refresh_lock = asyncio.Lock()


async def get_gemini_bearer_token(auth_path: Path | None = None) -> str:
    """Return a valid Gemini access token, refreshing if expired.

    Uses an asyncio.Lock to prevent concurrent refresh stampede.
    """
    async with _gemini_refresh_lock:
        creds = read_gemini_auth(auth_path)
        token = creds.get("access_token", "")

        if not _is_gemini_token_expired(creds):
            return token

        refresh_token = creds.get("refresh_token", "")
        if not refresh_token:
            msg = "Gemini auth missing refresh_token — run `gemini login` to re-authenticate."
            raise ValueError(msg)

        new_token = await refresh_gemini_token(refresh_token, auth_path=auth_path)
        return new_token


async def refresh_gemini_token(refresh_token: str, auth_path: Path | None = None) -> str:
    """Exchange a refresh token for a new access token via Google OAuth2.

    Updates the local oauth_creds.json atomically and returns the new access_token.
    """
    resolved_path = auth_path or _DEFAULT_GEMINI_AUTH_PATH

    def _do_refresh() -> str:
        body = urllib.parse.urlencode(
            {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": _GEMINI_CLIENT_ID,
                "client_secret": _GEMINI_CLIENT_SECRET,
            }
        ).encode()
        req = urllib.request.Request(  # noqa: S310
            _GOOGLE_TOKEN_URL, data=body, method="POST"
        )
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
            raise ValueError(f"Gemini token refresh failed: {exc}") from exc

        try:
            new_access_token: str = token_data["access_token"]
        except KeyError as exc:
            raise ValueError(
                "Gemini token refresh failed: response missing 'access_token'"
            ) from exc

        # Update expiry — Google returns expires_in (seconds)
        import time

        expires_in = token_data.get("expires_in", 3600)
        new_expiry_ms = int(time.time() * 1000) + (expires_in * 1000)

        new_refresh = token_data.get("refresh_token", refresh_token)

        # Atomic write
        current = json.loads(resolved_path.read_text()) if resolved_path.exists() else {}
        current["access_token"] = new_access_token
        current["refresh_token"] = new_refresh
        current["expiry_date"] = new_expiry_ms

        tmp_path = resolved_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(current, indent=2))
        os.replace(tmp_path, resolved_path)

        return new_access_token

    return await asyncio.to_thread(_do_refresh)


# ---------------------------------------------------------------------------
# Code Assist envelope — wraps/unwraps requests for the OAuth endpoint
# ---------------------------------------------------------------------------

_LOAD_CODE_ASSIST_URL = f"{_CODE_ASSIST_URL}:loadCodeAssist"

_cached_project: str | None = None


def _get_code_assist_project(auth_headers: dict[str, str]) -> str:
    """Fetch the cloudaicompanionProject from the loadCodeAssist endpoint.

    Caches the result — the project ID doesn't change within a session.
    """
    global _cached_project
    if _cached_project is not None:
        return _cached_project

    req = urllib.request.Request(  # noqa: S310
        _LOAD_CODE_ASSIST_URL, data=b"{}", method="POST"
    )
    req.add_header("Content-Type", "application/json")
    for key, value in auth_headers.items():
        req.add_header(key, value)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
            data = json.loads(resp.read())
            project = data.get("cloudaicompanionProject", "")
            if not project:
                raise ValueError(
                    "loadCodeAssist returned no cloudaicompanionProject. "
                    f"Response keys: {list(data.keys())}"
                )
            _cached_project = project
            return project
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ValueError(f"Failed to load Code Assist project: {exc}") from exc


def _wrap_code_assist_request(gemini_req: dict, model: str, project: str, session_id: str) -> dict:
    """Wrap a standard Gemini request in the Code Assist envelope."""
    inner: dict = {"contents": gemini_req.get("contents", [])}
    # Only include non-null optional fields — Gemini rejects null values
    if gemini_req.get("system_instruction"):
        inner["systemInstruction"] = gemini_req["system_instruction"]
    if gemini_req.get("tools"):
        inner["tools"] = gemini_req["tools"]
    if gemini_req.get("generationConfig"):
        inner["generationConfig"] = gemini_req["generationConfig"]
    return {
        "model": model,
        "project": project,
        "user_prompt_id": session_id,
        "request": inner,
    }


def _unwrap_code_assist_response(envelope: dict) -> dict:
    """Unwrap a Code Assist envelope to a standard Gemini response."""
    inner = envelope.get("response", envelope)
    return inner


# ---------------------------------------------------------------------------
# Concrete Provider implementation
# ---------------------------------------------------------------------------


_OAUTH_DEFAULT_MODEL = config.gemini_oauth_model()


class GeminiProvider:
    """Google Gemini provider implementing the Provider protocol.

    Supports two auth modes:
    - ``api_key``: uses GEMINI_API_KEY env var (x-goog-api-key header to public API)
    - ``gemini_oauth``: uses Gemini CLI OAuth tokens (Bearer header to Code Assist endpoint)
    """

    name = "gemini"
    stream_via_url = True

    def __init__(
        self,
        *,
        auth_mode: str = "gemini_oauth",
        api_key: str | None = None,
        auth_path: Path | None = None,
        **_kwargs,
    ) -> None:
        self.auth_mode = auth_mode
        self._api_key = api_key
        self._auth_path = auth_path
        self._project: str = ""
        import uuid

        self._session_id = str(uuid.uuid4())
        if auth_mode == "api_key":
            model = config.gemini_api_key_model()
            self.endpoint = f"{_BASE_URL}/models/{model}:streamGenerateContent?alt=sse"
        else:
            model = config.gemini_oauth_model()
            self.endpoint = f"{_CODE_ASSIST_URL}:streamGenerateContent?alt=sse"
        self._model = model

    async def authenticate(self) -> dict[str, str]:
        """Return auth headers for the configured mode.

        For OAuth mode, also resolves the Code Assist project on first call.
        """
        if self.auth_mode == "api_key":
            api_key = self._api_key or config.gemini_api_key()
            if not api_key:
                msg = (
                    "GEMINI_API_KEY environment variable is required for "
                    "api_key auth mode but was not set or is empty."
                )
                raise ValueError(msg)
            return {"x-goog-api-key": api_key}
        token = await get_gemini_bearer_token(self._auth_path)
        headers = {
            "Authorization": f"Bearer {token}",
            "User-Agent": f"GeminiCLI/claude-bridge ({os.uname().sysname})",
        }
        # Resolve project on first auth (needs the Bearer token)
        if not self._project:
            self._project = await asyncio.to_thread(_get_code_assist_project, headers)
        return headers

    def translate_request(self, anthropic_req: dict) -> tuple[dict, list[str]]:
        """Translate Anthropic Messages request to Gemini format."""
        result, warnings = anthropic_to_gemini(anthropic_req)
        # Gemini controls streaming via URL, not body — strip the field
        result.pop("stream", None)
        if self.auth_mode == "gemini_oauth":
            result = _wrap_code_assist_request(
                result, self._model, self._project, self._session_id
            )
        return result, warnings

    def translate_response(self, provider_resp: dict) -> dict:
        """Translate Gemini response to Anthropic Messages format."""
        if self.auth_mode == "gemini_oauth":
            provider_resp = _unwrap_code_assist_response(provider_resp)
        return gemini_to_anthropic(provider_resp)

    async def translate_stream(self, raw_chunks: AsyncIterator[bytes]) -> AsyncIterator[dict]:  # type: ignore[override]
        """Translate Gemini SSE stream to Anthropic SSE events.

        Gemini SSE format: ``data: <json>\\n\\n`` with complete JSON per chunk.
        No typed event names, no ``[DONE]`` sentinel.
        """
        buffer = b""
        state: dict = {}
        async for chunk in raw_chunks:
            buffer += chunk
            buffer = buffer.replace(b"\r\n", b"\n")
            while b"\n\n" in buffer:
                event_end = buffer.index(b"\n\n") + 2
                event_bytes = buffer[:event_end]
                buffer = buffer[event_end:]
                for event in _parse_sse_data(event_bytes, state):
                    yield event
        if buffer.strip():
            for event in _parse_sse_data(buffer, state):
                yield event


PROVIDERS["gemini"] = GeminiProvider
