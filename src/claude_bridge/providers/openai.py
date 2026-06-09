"""OpenAI Codex provider — OAuth token management + Anthropic/OpenAI translation (stdlib only)."""

from __future__ import annotations

import asyncio
import json
import os
import threading
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import AsyncIterator
from pathlib import Path

import claude_bridge.config as config
from claude_bridge.auth import is_token_expired
from claude_bridge.content import MediaSource, parse_media_source
from claude_bridge.provider import PROVIDERS, ProviderCapabilities
from claude_bridge.stream import parse_sse_events

_CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
_TOKEN_URL = "https://auth.openai.com/oauth/token"  # noqa: S105  # nosec B105
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
            with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310  # nosec B310
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
_REASONING_MODE = config.reasoning_mode()

# Upper bound on the per-provider encrypted-reasoning cache (one entry per in-flight
# tool call). Bounds memory under long agentic sessions; oldest entries evict first.
_REASONING_CACHE_MAX = 256

# Upper bound on the undrained SSE buffer (one incomplete event). A well-formed
# stream drains on every "\n\n", so the buffer holds at most a single partial event.
# A provider that streams without event terminators would otherwise grow the buffer
# without limit (OOM) and make repeated concatenation quadratic; exceeding this cap
# means the stream is malformed, so we abort fast instead of accumulating.
_MAX_SSE_BUFFER = 4 * 1024 * 1024

# Upper bound on a user-controlled token (block/tool_choice ``type``) embedded in a
# translation warning. Caps log/trace line length against a hostile oversized type.
_SAFE_TOKEN_MAX = 64

# Image MIME types the Responses API ``input_image`` part accepts. A base64 image
# whose media_type is outside this set degrades to a placeholder rather than risking
# an upstream 400 — the set is the contract, not a guess.
_IMAGE_MIME_ALLOWLIST = frozenset({"image/jpeg", "image/png", "image/gif", "image/webp"})

# Fallback filename for a document with no Anthropic ``title`` — the Responses
# ``input_file`` part requires a filename.
_DEFAULT_DOCUMENT_FILENAME = "document.pdf"

# Conservative default for callers that don't pass a provider's capabilities (e.g.
# direct translation in tests): text-only input, string tool output — the pre-media
# behavior. The real path threads ``OpenAIProvider.capabilities`` from translate_request.
_TEXT_ONLY_CAPABILITIES = ProviderCapabilities(
    stream_request_mode="body_parameter", sync_response_mode="sse"
)


def _safe_token(value: object) -> str:
    """Neutralize an attacker-controlled token for safe embedding in a log line or trace.

    A block / tool_choice ``type`` comes straight from the client request and is
    interpolated into a translation warning that reaches the human log and the
    structural trace. Strips non-printable characters (newline, carriage return,
    tab, ANSI escapes — CWE-117 log injection) and caps the result at
    ``_SAFE_TOKEN_MAX`` so a hostile type cannot forge log records or flood the trace.
    """
    cleaned = "".join(ch for ch in str(value) if ch.isprintable())
    if len(cleaned) > _SAFE_TOKEN_MAX:
        return cleaned[:_SAFE_TOKEN_MAX] + "..."
    return cleaned


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


def _translate_content_block(
    block: dict, capabilities: ProviderCapabilities
) -> tuple[dict, list[str]]:
    """Translate a single Anthropic content block to OpenAI Responses API format.

    Returns ``(translated_block, warnings)``. For tool_use / tool_result blocks the
    translated block has a special ``_toplevel`` key set to True, signaling the caller
    to emit it as a top-level input item rather than nesting it inside a message's
    content array. Media blocks (image/document) become real Responses content parts
    when ``capabilities.input_modalities`` allows, and degrade to a redacted
    placeholder (never echoing base64) otherwise.

    Thin dispatcher: each block type delegates to a dedicated helper so this function
    stays well under the CCN ceiling as block types grow.
    """
    block_type = block.get("type", "unknown")
    if block_type == "text":
        return {"type": "input_text", "text": block["text"]}, []
    if block_type == "thinking":
        return _translate_thinking_block(block)
    if block_type == "image":
        return _translate_image_block(parse_media_source(block), capabilities)
    if block_type == "document":
        return _translate_document_block(parse_media_source(block), capabilities)
    if block_type == "tool_use":
        return _translate_tool_use_block(block), []
    if block_type == "tool_result":
        return _translate_tool_result_block(block, capabilities)
    return _translate_unsupported_block(block_type)


def _translate_thinking_block(block: dict) -> tuple[dict, list[str]]:
    """Translate an Anthropic thinking block per the configured reasoning mode."""
    if _REASONING_MODE == "drop":
        return {"type": "input_text", "text": ""}, [
            "Stripped thinking block (reasoning_mode=drop)"
        ]
    thinking_text = block.get("thinking", "")
    return {"type": "input_text", "text": f"[thinking]\n{thinking_text}\n[/thinking]"}, []


def _media_placeholder(kind: str, reason: str) -> tuple[dict, list[str]]:
    """Build a redacted placeholder + warning for media that can't be forwarded.

    ``reason`` must name only a safe token (media_type, source kind, modality) — never
    the base64 payload, which must never reach the placeholder text or the warning.
    """
    safe_reason = _safe_token(reason)
    return (
        {"type": "input_text", "text": f"[unsupported {kind}: {safe_reason}]"},
        [f"{kind} input degraded to placeholder: {safe_reason}"],
    )


def _translate_image_block(
    source: MediaSource, capabilities: ProviderCapabilities
) -> tuple[dict, list[str]]:
    """Forward an image as a Responses ``input_image`` part, or degrade if unforwardable.

    Spec: ``input_image.image_url`` is a STRING — a ``data:`` URL for base64 or the
    source URL directly. Base64 outside the MIME allowlist, or a file/unknown source
    (no bytes), degrades to a redacted placeholder (never echoes the payload).
    """
    if "image" not in capabilities.input_modalities:
        return _media_placeholder("image", "not supported by this provider/auth mode")
    if source.source_kind == "url":
        return {"type": "input_image", "image_url": source.url}, []
    if source.source_kind == "base64":
        if source.media_type not in _IMAGE_MIME_ALLOWLIST:
            return _media_placeholder("image", source.media_type)
        return {
            "type": "input_image",
            "image_url": f"data:{source.media_type};base64,{source.data}",
        }, []
    return _media_placeholder("image", source.source_kind)


def _translate_document_block(
    source: MediaSource, capabilities: ProviderCapabilities
) -> tuple[dict, list[str]]:
    """Forward a document as a Responses ``input_file`` part, or degrade if unforwardable.

    Spec: ``input_file`` carries ``filename``+``file_data`` (a ``data:`` URL) for base64,
    or ``file_url`` for a URL source. A file/unknown source (no bytes) degrades to a
    redacted placeholder.
    """
    if "document" not in capabilities.input_modalities:
        return _media_placeholder("document", "not supported by this provider/auth mode")
    if source.source_kind == "url":
        return {"type": "input_file", "file_url": source.url}, []
    if source.source_kind == "base64":
        return {
            "type": "input_file",
            "filename": source.filename or _DEFAULT_DOCUMENT_FILENAME,
            "file_data": f"data:{source.media_type};base64,{source.data}",
        }, []
    return _media_placeholder("document", source.source_kind)


def _translate_tool_use_block(block: dict) -> dict:
    """Translate an Anthropic tool_use block to a top-level function_call item.

    Anthropic uses ``toolu_xxx``/``call_xxx``; OpenAI Responses requires ``fc_xxx``,
    and both ``id`` and ``call_id`` are required.
    """
    fc_id = _to_openai_id(block["id"])
    return {
        "_toplevel": True,
        "type": "function_call",
        "id": fc_id,
        "call_id": fc_id,
        "name": block["name"],
        "arguments": json.dumps(block["input"]),
    }


_TOOL_RESULT_MEDIA_TYPES = frozenset({"image", "document"})


def _tool_result_has_media(content: object) -> bool:
    """Report whether tool_result content carries an image/document block."""
    if not isinstance(content, list):
        return False
    return any(b.get("type") in _TOOL_RESULT_MEDIA_TYPES for b in content)


def _tool_result_string(content: object, is_error: bool) -> str:
    """Flatten tool_result content to a string, redacting media (never base64).

    Media blocks degrade to a bounded ``[media omitted: <kind>/<media_type> — …]``
    placeholder: a string-only backend cannot carry the bytes, and echoing the base64
    payload would both be useless to the model and leak the tool's output.
    """
    if isinstance(content, list):
        rendered = []
        for b in content:
            if b.get("type") == "text":
                rendered.append(b.get("text", ""))
            elif b.get("type") in _TOOL_RESULT_MEDIA_TYPES:
                src = parse_media_source(b)
                rendered.append(
                    f"[media omitted: {src.kind}/{src.media_type} — "
                    "provider/auth mode does not support tool-output media]"
                )
        text = "\n".join(rendered)
    else:
        text = str(content) if content else ""
    return f"[Error] {text}" if is_error else text


def _tool_result_parts(
    content: list, capabilities: ProviderCapabilities, is_error: bool
) -> tuple[list[dict], list[str]]:
    """Build the array form of tool_result output: real Responses content parts.

    Text becomes ``input_text``; image/document delegate to the same media helpers as
    the message path (so an unforwardable modality degrades to a redacted part). An
    error result is prefixed with a leading ``[Error]`` marker part so the model can
    distinguish failure from success carrying the same parts.
    """
    parts: list[dict] = []
    warnings: list[str] = []
    for b in content:
        btype = b.get("type")
        if btype == "text":
            parts.append({"type": "input_text", "text": b.get("text", "")})
        elif btype == "image":
            part, warns = _translate_image_block(parse_media_source(b), capabilities)
            parts.append(part)
            warnings.extend(warns)
        elif btype == "document":
            part, warns = _translate_document_block(parse_media_source(b), capabilities)
            parts.append(part)
            warnings.extend(warns)
    if is_error:
        parts.insert(0, {"type": "input_text", "text": "[Error]"})
    return parts, warnings


def _translate_tool_result_block(
    block: dict, capabilities: ProviderCapabilities
) -> tuple[dict, list[str]]:
    """Translate an Anthropic tool_result block to a top-level function_call_output item.

    ``output`` is ``str | list[dict]``. When the content carries media AND the provider
    declares ``supports_tool_output_content_parts``, the output is an ARRAY of real
    content parts (so tool-returned screenshots/PDFs reach a vision model). Otherwise it
    is a string: text-only results keep their original string shape, and media in a
    string-only backend is redacted (never base64).
    """
    content = block.get("content", "")
    is_error = bool(block.get("is_error"))
    fc_id = _to_openai_id(block["tool_use_id"])
    if (
        isinstance(content, list)
        and _tool_result_has_media(content)
        and capabilities.supports_tool_output_content_parts
    ):
        output, warnings = _tool_result_parts(content, capabilities, is_error)
    else:
        output, warnings = _tool_result_string(content, is_error), []
    return {
        "_toplevel": True,
        "type": "function_call_output",
        "call_id": fc_id,
        "output": output,
    }, warnings


def _translate_unsupported_block(block_type: str) -> tuple[dict, list[str]]:
    """Degrade an unsupported block to a redacted, type-named placeholder.

    Unsupported / special blocks (server_tool_use, web_search_tool_result, ...) have no
    OpenAI Responses route (D-SRVTOOL-001). The placeholder NEVER echoes the block's
    nested content: a raw str(block) would pollute the request AND leak tool inputs.
    """
    safe_type = _safe_token(block_type)
    warning = (
        f"Unsupported content block type '{safe_type}' replaced with a redacted "
        "placeholder (no provider equivalent)"
    )
    return {"type": "input_text", "text": f"[unsupported content block: {safe_type}]"}, [warning]


def _translate_message(
    message: dict, capabilities: ProviderCapabilities
) -> tuple[list[dict], list[str]]:
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
        translated, block_warnings = _translate_content_block(block, capabilities)
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


def _translate_tool_choice(tool_choice: dict) -> tuple[dict, list[str]]:
    """Map an Anthropic ``tool_choice`` to OpenAI Responses request fields.

    Returns ``(fields, warnings)`` where ``fields`` carries the keys to merge into
    the translated request: ``tool_choice`` (``"auto"``/``"none"``/``"required"`` or a
    forced ``{"type": "function", "name": ...}`` object) and ``parallel_tool_calls``
    when Anthropic's ``disable_parallel_tool_use`` is set. Unsupported choice types
    are omitted with a warning rather than guessed.
    """
    fields: dict = {}
    warnings: list[str] = []
    choice_type = tool_choice.get("type")
    if choice_type == "auto":
        fields["tool_choice"] = "auto"
    elif choice_type == "none":
        fields["tool_choice"] = "none"
    elif choice_type == "any":
        fields["tool_choice"] = "required"
    elif choice_type == "tool":
        fields["tool_choice"] = {"type": "function", "name": tool_choice["name"]}
    else:
        warnings.append(
            f"Unsupported tool_choice type '{_safe_token(choice_type)}', omitting tool_choice"
        )
    if tool_choice.get("disable_parallel_tool_use"):
        fields["parallel_tool_calls"] = False
    return fields, warnings


def anthropic_to_openai(
    request: dict, capabilities: ProviderCapabilities = _TEXT_ONLY_CAPABILITIES
) -> tuple[dict, list[str]]:
    """Translate an Anthropic Messages API request to an OpenAI Responses API request.

    Returns ``(translated_request, warnings)`` where warnings lists any features that
    were stripped or degraded because they have no OpenAI equivalent. ``capabilities``
    declares which input modalities to forward; it defaults to text-only so direct
    callers keep the pre-media behavior, while the provider passes its real
    capabilities via ``translate_request``.

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

    # Build result — Codex endpoint requires stream: true.
    # include=reasoning.encrypted_content: with store:false the model is stateless,
    # so it returns each reasoning item's encrypted continuation blob. The provider
    # echoes these back before their function_calls on the next turn (see
    # _associate_reasoning_with_calls / OpenAIProvider._inject_reasoning); without it
    # gpt-5-class models reject the follow-up with "function_call was provided without
    # its required reasoning item".
    result: dict = {
        "model": translated_model,
        "reasoning": {"effort": "xhigh"},
        "store": False,
        "stream": True,
        "include": ["reasoning.encrypted_content"],
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

    # tool_choice / parallel controls — preserve Claude Code's requested tool policy
    tool_choice = request.get("tool_choice")
    if tool_choice is not None:
        tc_fields, tc_warnings = _translate_tool_choice(tool_choice)
        result.update(tc_fields)
        warnings.extend(tc_warnings)

    # Messages → input
    input_items: list[dict] = []
    for message in request.get("messages", []):
        items, msg_warnings = _translate_message(message, capabilities)
        input_items.extend(items)
        warnings.extend(msg_warnings)

    result["input"] = input_items

    if _has_cache_control(request):
        warnings.append(
            "Stripped cache_control hints (no provider equivalent — caching is automatic)"
        )

    return result, warnings


# OpenAI Responses ``incomplete_details.reason`` that signals a moderation block, not
# token-budget exhaustion. Disambiguating the two is the whole point of T-006: mapping
# a content-filtered turn to ``max_tokens`` makes Claude Code auto-compact a context
# nowhere near full and retry endlessly.
_CONTENT_FILTER_REASON = "content_filter"

# Surfaced to Claude Code when a turn is content-filtered with no model text, so the turn
# renders as a visible refusal rather than a blank assistant message. Mirrors the Gemini
# provider's ``_SAFETY_REFUSAL``.
_CONTENT_FILTER_REFUSAL = (
    "I cannot complete this response because it was blocked by content safety filters. "
    "Please rephrase your request."
)


def _coerce_token_count(value: object) -> int:
    """Coerce a provider token count to a non-negative int.

    Provider usage may carry floats or nulls; Anthropic's usage fields are integers
    that Claude Code's ``/context`` math divides by. Non-numeric values default to 0.
    """
    if isinstance(value, (int, float)):
        return max(0, int(value))
    return 0


def _anthropic_usage(oai_usage: object) -> dict:
    """Project OpenAI Responses usage onto Anthropic's flat integer shape.

    OpenAI ``input_tokens`` already includes cached tokens and ``output_tokens`` already
    includes reasoning tokens (both are subsets, per the Responses contract), so each maps
    straight to Anthropic's corresponding total. Cached tokens are deliberately NOT split
    into ``cache_read_input_tokens`` — Anthropic's totals are non-overlapping, so doing so
    would double-count. Missing or non-numeric fields default to 0. See D-USAGE-001.
    """
    usage = oai_usage if isinstance(oai_usage, dict) else {}
    return {
        "input_tokens": _coerce_token_count(usage.get("input_tokens", 0)),
        "output_tokens": _coerce_token_count(usage.get("output_tokens", 0)),
    }


def _incomplete_reason(response: dict) -> str:
    """Return ``incomplete_details.reason`` from a Responses object, or ``""`` if absent.

    GPT-5 sometimes returns ``status: "incomplete"`` with a null ``incomplete_details``;
    that absence reads as token exhaustion (the conservative default).
    """
    details = response.get("incomplete_details")
    return details.get("reason", "") if isinstance(details, dict) else ""


def _stop_reason(status: str, has_tool_calls: bool, incomplete_reason: str) -> str:
    """Map an OpenAI Responses terminal status to an Anthropic ``stop_reason``.

    Tool calls win — Claude Code must run the tool rather than compact. A content-filtered
    completion ends the turn cleanly (``end_turn``); any other ``incomplete`` is treated as
    output-token exhaustion (``max_tokens``), the signal Claude Code auto-compacts on.
    """
    if has_tool_calls:
        return "tool_use"
    if status == "incomplete":
        return "end_turn" if incomplete_reason == _CONTENT_FILTER_REASON else "max_tokens"
    return "end_turn"


def openai_to_anthropic(response: dict) -> dict:
    """Translate an OpenAI Responses API response to an Anthropic Messages API response.

    Pure function — no I/O.
    """
    # Map status → stop_reason (disambiguating content_filter from token exhaustion)
    status = response.get("status", "completed")
    output_items = response.get("output", [])
    has_tool_calls = any(i.get("type") == "function_call" for i in output_items)
    incomplete_reason = _incomplete_reason(response)
    stop_reason = _stop_reason(status, has_tool_calls, incomplete_reason)

    # Translate output items → content blocks
    content: list[dict] = []
    for item in output_items:
        item_type = item.get("type")

        if item_type == "message":
            # Extract text from message content
            for block in item.get("content", []):
                if block.get("type") == "output_text":
                    content.append({"type": "text", "text": block["text"]})

        elif item_type == "refusal":
            # A model refusal carries human-readable text — surface it, don't drop it.
            content.append({"type": "text", "text": item.get("refusal", "")})

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

    # Content-filtered turn with no model text → synthesize a visible refusal so the
    # turn never renders as a blank assistant message.
    has_text = any(b.get("type") == "text" and b.get("text") for b in content)
    if incomplete_reason == _CONTENT_FILTER_REASON and not has_text:
        content.append({"type": "text", "text": _CONTENT_FILTER_REFUSAL})

    usage = _anthropic_usage(response.get("usage"))

    return {
        "id": f"msg_bridge_{response.get('id', 'unknown')}",
        "type": "message",
        "role": "assistant",
        "model": response.get("model", ""),
        "stop_reason": stop_reason,
        "content": content,
        "usage": usage,
    }


def _associate_reasoning_with_calls(output: list[dict]) -> dict[str, dict]:
    """Map each tool call's id to the reasoning item that immediately precedes it.

    Walks the Responses ``output`` once: a reasoning item carrying
    ``encrypted_content`` becomes the pending continuation state for the next
    function_call; once paired (or interrupted by any other item) it is consumed.
    Keys use the call's ``call_id`` (falling back to ``id``) normalized to ``fc_``
    form — the same identity ``openai_to_anthropic`` exposes to Claude Code — so the
    next request's function_calls look up by the matching key.

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
                key = _to_openai_id(item.get("call_id") or item.get("id", ""))
                if key:
                    associations[key] = pending
            pending = None
        else:
            pending = None
    return associations


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
                        "input_tokens": _coerce_token_count(usage.get("input_tokens", 0)),
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


def _synthesize_refusal_block(text: str) -> list[dict]:
    """Build start/delta/stop SSE events for a synthetic refusal text block.

    Emitted when a streamed turn is content-filtered with no model text, so the stream
    does not end on an empty assistant message. The placeholder index 0 is reassigned to
    the next sequential Anthropic block index by ``_remap_block_index``.
    """
    return [
        {
            "event": "content_block_start",
            "data": {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        },
        {
            "event": "content_block_delta",
            "data": {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": text},
            },
        },
        {"event": "content_block_stop", "data": {"type": "content_block_stop", "index": 0}},
    ]


def _sse_response_completed(data: dict) -> list[dict]:
    """Translate response.completed → [refusal block?] + message_delta + message_stop.

    A content-filtered turn is prefixed with a synthesized refusal text block and ends
    with ``end_turn``; any other ``incomplete`` maps to ``max_tokens``.
    """
    resp = data.get("response", {})
    status = resp.get("status", "completed")
    output = resp.get("output", [])
    has_tool_calls = any(i.get("type") == "function_call" for i in output)
    incomplete_reason = _incomplete_reason(resp)
    stop_reason = _stop_reason(status, has_tool_calls, incomplete_reason)

    events: list[dict] = []
    if incomplete_reason == _CONTENT_FILTER_REASON:
        events.extend(_synthesize_refusal_block(_CONTENT_FILTER_REFUSAL))
    events.append(
        {
            "event": "message_delta",
            "data": {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason},
                "usage": _anthropic_usage(resp.get("usage")),
            },
        }
    )
    events.append({"event": "message_stop", "data": {"type": "message_stop"}})
    return events


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
    capabilities = ProviderCapabilities(
        stream_request_mode="body_parameter",
        sync_response_mode="sse",
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
        if auth_mode == "api_key":
            self.endpoint = _API_KEY_ENDPOINT
        else:
            self.endpoint = _CODEX_ENDPOINT
        # Encrypted reasoning blobs keyed by fc_ call id, captured from responses and
        # re-injected before their function_calls on the next request. In-memory only —
        # opaque, never persisted, never logged, never returned to Claude Code.
        self._reasoning_by_call_id: dict[str, dict] = {}
        self._reasoning_lock = threading.Lock()

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
        """Translate Anthropic Messages request to OpenAI Responses request, echoing any
        captured encrypted reasoning back before its function_calls."""
        result, warnings = anthropic_to_openai(anthropic_req, self.capabilities)
        self._inject_reasoning(result)
        return result, warnings

    def translate_response(self, provider_resp: dict) -> dict:
        """Translate OpenAI Responses response to Anthropic Messages response, capturing
        each function_call's preceding encrypted reasoning for the next request."""
        self._stash_reasoning(_associate_reasoning_with_calls(provider_resp.get("output", [])))
        return openai_to_anthropic(provider_resp)

    def _capture_stream_reasoning(self, parsed_event: dict) -> None:
        """Capture encrypted reasoning from a streamed response.completed event."""
        if parsed_event.get("event") != "response.completed":
            return
        response_obj = (parsed_event.get("data") or {}).get("response") or {}
        self._stash_reasoning(_associate_reasoning_with_calls(response_obj.get("output", [])))

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
                    self._capture_stream_reasoning(parsed_event)
                    for translated in translate_openai_sse_event(parsed_event):
                        translated, block_index, has_tool_calls = _remap_block_index(
                            translated, index_map, block_index, has_tool_calls
                        )
                        yield translated
            if len(buffer) > _MAX_SSE_BUFFER:
                raise RuntimeError(
                    f"Provider SSE stream exceeded {_MAX_SSE_BUFFER} bytes without an "
                    "event terminator; aborting malformed stream"
                )
        if buffer.strip():
            for parsed_event in parse_sse_events(buffer):
                self._capture_stream_reasoning(parsed_event)
                for translated in translate_openai_sse_event(parsed_event):
                    translated, block_index, has_tool_calls = _remap_block_index(
                        translated, index_map, block_index, has_tool_calls
                    )
                    yield translated


PROVIDERS["openai"] = OpenAIProvider
