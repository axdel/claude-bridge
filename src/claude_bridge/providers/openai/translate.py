"""Anthropic <-> OpenAI Responses API translation — pure functions, no I/O.

Owns the shared token/id helpers, content-filter constants, per-backend
capabilities, and reasoning-mode config that the stream and provider submodules
derive from. A leaf within the package: imports only config/content/provider.
"""

from __future__ import annotations

import json

import claude_bridge.config as config
from claude_bridge.content import MediaSource, parse_media_source
from claude_bridge.provider import ProviderCapabilities

# Every Anthropic model Claude Code sends is routed to one upstream OpenAI model.
# There are no per-model overrides, so this map is empty and DEFAULT_MODEL applies
# to every request — the routing is not keyed on exact opus/sonnet/haiku versions,
# so new Claude releases need no change here. (Add an entry only to override a
# specific model to a different upstream target.)
MODEL_MAP: dict[str, str] = {}
DEFAULT_MODEL = "gpt-5.6-sol"
GPT_TOKEN_COUNT_MULTIPLIER = 1.1

# Anthropic output_config.effort vocabulary (Opus 4.6: low/medium/high/max; 4.7 adds xhigh).
# GPT-5.6's reasoning.effort accepts the SAME set, so a recognized caller effort maps 1:1.
_ANTHROPIC_EFFORT_VALUES = frozenset({"low", "medium", "high", "xhigh", "max"})
# Default when the caller sends no effort — max preserves the prior always-max behavior and
# matches "max everywhere". Claude Code always sends output_config.effort, so this is the
# non-Claude-Code caller path only.
_DEFAULT_OPENAI_EFFORT = "max"

# Reasoning mode: "passthrough" preserves thinking blocks, "drop" strips them.
_REASONING_MODE = config.reasoning_mode()

# Upper bound on a user-controlled token (block/tool_choice ``type``) embedded in a
# translation warning. Caps log/trace line length against a hostile oversized type.
_SAFE_TOKEN_MAX = 64

# Image MIME types the Responses API ``input_image`` part accepts. A base64 image
# whose media_type is outside this set degrades to a placeholder rather than risking
# an upstream 400 — the set is the contract, not a guess.
_IMAGE_MIME_ALLOWLIST = frozenset({"image/jpeg", "image/png", "image/gif", "image/webp"})

# Document MIME types the Responses ``input_file`` part accepts as base64 ``file_data``.
# A base64 document whose media_type is outside this set degrades to a placeholder
# rather than interpolating an unvalidated, client-controlled media_type into a data:
# URL — the same allowlist discipline the image path applies (the set is the contract).
_DOCUMENT_MIME_ALLOWLIST = frozenset({"application/pdf"})

# Fallback filename for a document with no Anthropic ``title`` — the Responses
# ``input_file`` part requires a filename.
_DEFAULT_DOCUMENT_FILENAME = "document.pdf"

# Upper bound on a forwarded filename. The Anthropic ``title`` is client-controlled
# and reaches the provider as metadata; cap it so a hostile title cannot bloat the body.
_FILENAME_MAX = 255

# Conservative default for callers that don't pass a provider's capabilities (e.g.
# direct translation in tests): text-only input, string tool output — the pre-media
# behavior. The real path threads ``OpenAIProvider.capabilities`` from translate_request.
_TEXT_ONLY_CAPABILITIES = ProviderCapabilities(
    stream_request_mode="body_parameter", sync_response_mode="sse"
)

# Per-backend capabilities (D-MODALITY-001). api.openai.com documents image+document
# input and array-form tool output. chatgpt.com (Codex) returned HTTP 200 for input_image
# and input_file in the T-001 probe, but array-form function_call_output was NOT probed,
# so tool-output arrays stay disabled there (media degrades observably) until a tool-loop
# probe confirms support. Selected per auth mode on the instance in ``__init__``.
_API_KEY_CAPABILITIES = ProviderCapabilities(
    stream_request_mode="body_parameter",
    sync_response_mode="sse",
    input_modalities=frozenset({"text", "image", "document"}),
    supports_tool_output_content_parts=True,
    token_count_multiplier=GPT_TOKEN_COUNT_MULTIPLIER,
)
_CODEX_CAPABILITIES = ProviderCapabilities(
    stream_request_mode="body_parameter",
    sync_response_mode="sse",
    input_modalities=frozenset({"text", "image", "document"}),
    supports_tool_output_content_parts=False,
    token_count_multiplier=GPT_TOKEN_COUNT_MULTIPLIER,
)


def _safe_token(value: object) -> str:
    """Neutralize an attacker-controlled token for safe embedding in a log line or trace.

    A block / tool_choice ``type`` comes straight from the client request and is
    interpolated into a translation warning that reaches the human log and the
    structural trace. Strips non-printable characters (newline, carriage return,
    tab, ANSI escapes — CWE-117 log injection) and caps the result at
    ``_SAFE_TOKEN_MAX`` so a hostile type cannot forge log records or flood the trace.

    A container value (a malformed dict/list effort) is reduced to a bare ``<type>``
    tag: ``str()`` on a container would copy its contents — which may nest a client
    secret — into the log and trace, so only its type is surfaced, never its value.
    Scalars coerce to their short, bounded ``str()``: a missing ``type`` key
    legitimately arrives as ``None`` and must render as ``None`` for diagnosis.
    """
    if isinstance(value, (dict, list, tuple, set)):
        return f"<{type(value).__name__}>"
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


def _safe_document_filename(title: object) -> str:
    """Sanitize a client-controlled document title into a safe input_file filename.

    The Anthropic ``title`` is untrusted and forwarded to the provider as metadata.
    Reduces it to a basename (drops POSIX and Windows path separators), strips
    non-printable characters, and caps length; an empty or fully-stripped title
    falls back to ``_DEFAULT_DOCUMENT_FILENAME``.
    """
    if not title:
        return _DEFAULT_DOCUMENT_FILENAME
    basename = str(title).replace("\\", "/").rsplit("/", 1)[-1]
    cleaned = "".join(ch for ch in basename if ch.isprintable()).strip()
    return cleaned[:_FILENAME_MAX] or _DEFAULT_DOCUMENT_FILENAME


def _translate_document_block(
    source: MediaSource, capabilities: ProviderCapabilities
) -> tuple[dict, list[str]]:
    """Forward a document as a Responses ``input_file`` part, or degrade if unforwardable.

    Spec: ``input_file`` carries ``filename``+``file_data`` (a ``data:`` URL) for base64,
    or ``file_url`` for a URL source. Base64 outside the document MIME allowlist, or a
    file/unknown source (no bytes), degrades to a redacted placeholder (never echoes the
    payload). The forwarded filename is sanitized — the title is client-controlled.
    """
    if "document" not in capabilities.input_modalities:
        return _media_placeholder("document", "not supported by this provider/auth mode")
    if source.source_kind == "url":
        return {"type": "input_file", "file_url": source.url}, []
    if source.source_kind == "base64":
        if source.media_type not in _DOCUMENT_MIME_ALLOWLIST:
            return _media_placeholder("document", source.media_type)
        return {
            "type": "input_file",
            "filename": _safe_document_filename(source.filename),
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
        if isinstance(content, list) and _tool_result_has_media(content):
            warnings = [
                "tool_result media redacted to string "
                "(provider/auth mode does not support tool-output content arrays)"
            ]
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


def _resolve_openai_effort(request: dict, warnings: list[str]) -> str:
    """Resolve ``reasoning.effort`` from the caller's ``output_config``, defaulting to max.

    GPT-5.6 shares Anthropic's effort vocabulary, so a recognized ``output_config.effort``
    maps 1:1. Any other ``output_config`` subkey (e.g. structured-output ``format``) has no
    Responses equivalent and surfaces a warning; an unrecognized effort value defaults to
    max with a warning naming it. A safe-token wrapper neutralizes the client-controlled
    value before it reaches a log line or trace (CWE-117).
    """
    output_config = request.get("output_config")
    if not isinstance(output_config, dict):
        return _DEFAULT_OPENAI_EFFORT
    for key in sorted(output_config):
        if key != "effort":
            warnings.append(f"Dropped unsupported output_config.{_safe_token(key)}")
    effort = output_config.get("effort")
    if effort is None:
        return _DEFAULT_OPENAI_EFFORT
    if isinstance(effort, str) and effort in _ANTHROPIC_EFFORT_VALUES:
        return effort
    warnings.append(
        f"Unrecognized output_config.effort '{_safe_token(effort)}', "
        f"using default '{_DEFAULT_OPENAI_EFFORT}'"
    )
    return _DEFAULT_OPENAI_EFFORT


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

    # reasoning.effort — honor the caller's output_config.effort (1:1 with GPT-5.6's
    # vocabulary), defaulting to max. Replaces the old blanket strip of output_config: the
    # caller's per-request effort is a real instruction, not an unsupported key to discard.
    effort = _resolve_openai_effort(request, warnings)

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
        "reasoning": {"effort": effort},
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

    return result, warnings


# OpenAI Responses ``incomplete_details.reason`` that signals a moderation block, not
# token-budget exhaustion. Disambiguating the two is the whole point of T-006: mapping
# a content-filtered turn to ``max_tokens`` makes Claude Code auto-compact a context
# nowhere near full and retry endlessly.
_CONTENT_FILTER_REASON = "content_filter"

# Surfaced to Claude Code when a turn is content-filtered with no model text, so the turn
# renders as a visible refusal rather than a blank assistant message.
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


def _scale_token_count(value: object, multiplier: float) -> int:
    """Return a non-negative token count adjusted by the provider multiplier."""
    return int(_coerce_token_count(value) * multiplier + 0.5)


def _anthropic_usage(
    oai_usage: object,
    *,
    token_count_multiplier: float = GPT_TOKEN_COUNT_MULTIPLIER,
) -> dict:
    """Project OpenAI Responses usage onto Anthropic's flat integer shape.

    OpenAI ``input_tokens`` already includes cached tokens and ``output_tokens`` already
    includes reasoning tokens (both are subsets, per the Responses contract), so each maps
    to Anthropic's corresponding total before applying the OpenAI compatibility multiplier.
    Cached tokens are deliberately NOT split into ``cache_read_input_tokens`` — Anthropic's
    totals are non-overlapping, so doing so would double-count. Missing or non-numeric
    fields default to 0. See D-USAGE-001 and D-USAGE-003.
    """
    usage = oai_usage if isinstance(oai_usage, dict) else {}
    return {
        "input_tokens": _scale_token_count(usage.get("input_tokens", 0), token_count_multiplier),
        "output_tokens": _scale_token_count(usage.get("output_tokens", 0), token_count_multiplier),
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


def openai_to_anthropic(
    response: dict,
    *,
    token_count_multiplier: float = GPT_TOKEN_COUNT_MULTIPLIER,
) -> dict:
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

    usage = _anthropic_usage(response.get("usage"), token_count_multiplier=token_count_multiplier)

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
