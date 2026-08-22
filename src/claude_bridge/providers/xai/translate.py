"""Anthropic Messages -> xAI (Grok) Responses translation — pure functions, no I/O.

Owns the shared token/usage helpers, content-filter constants, capabilities, and the
reasoning-mode config the stream and provider submodules derive from. A leaf within the
package (imports only config/content/provider). Duplicates the OpenAI provider's
Responses translation by design — cross-provider imports are forbidden (D-XAI-002).
"""

from __future__ import annotations

import json
import re

import claude_bridge.config as config
from claude_bridge.content import MediaSource, parse_media_source
from claude_bridge.provider import ProviderCapabilities

# Reasoning mode (shared REASONING_MODE env): "passthrough" keeps a replayed
# Anthropic thinking block as bracketed text, "drop" strips it. xAI reasoning
# *continuity* rides on encrypted reasoning items (include=reasoning.encrypted_content),
# replayed by the provider — not this text path. Read once at import, like openai.py.
_XAI_REASONING_MODE = config.reasoning_mode()

# Top-level Anthropic request keys with no xAI Responses equivalent.
_XAI_STRIPPED_KEYS = ("output_config",)

# Content-block types that carry media inside a tool_result. When the provider declares
# ``supports_tool_output_content_parts`` the media is forwarded as real Responses content
# parts; otherwise it degrades to a redacted string (never base64).
_TOOL_RESULT_MEDIA_TYPES = frozenset({"image", "document"})

# Image MIME types the Responses ``input_image`` part accepts. A base64 image whose
# media_type is outside this set degrades to a placeholder rather than risking an upstream
# 400 — the set is the contract, not a guess. Proven accepted by image_input.json.
_IMAGE_MIME_ALLOWLIST = frozenset({"image/jpeg", "image/png", "image/gif", "image/webp"})

# Document MIME types the Responses ``input_file`` part accepts as base64 ``file_data``.
# A base64 document outside this set degrades rather than interpolating an unvalidated,
# client-controlled media_type into a data: URL — same allowlist discipline as images.
_DOCUMENT_MIME_ALLOWLIST = frozenset({"application/pdf"})

# Fallback filename for a document with no Anthropic ``title`` — the Responses
# ``input_file`` part requires a filename.
_DEFAULT_DOCUMENT_FILENAME = "document.pdf"

# Upper bound on a forwarded filename. The Anthropic ``title`` is client-controlled and
# reaches the provider as metadata; cap it so a hostile title cannot bloat the body.
_FILENAME_MAX = 255

# Upper bound on a user-controlled token (block / tool_choice ``type``) embedded in a
# translation warning — caps log/trace line length against a hostile oversized type.
_SAFE_TOKEN_MAX = 64

# Instruction sent to xAI when the Anthropic request carries no system prompt (the
# cli-chat-proxy Responses endpoint expects an instructions string).
_DEFAULT_XAI_INSTRUCTIONS = "You are a helpful assistant."

# Text-only capabilities for the provider instance. B4 adds image/document input
# modalities and array-form tool output; B8 finalizes the capability set and the
# token-count multiplier. Declared here so ``XAIProvider`` satisfies the Provider
# protocol and the proxy can read its stream/response modes.
_XAI_TEXT_ONLY_CAPABILITIES = ProviderCapabilities(
    stream_request_mode="body_parameter", sync_response_mode="sse"
)


def _safe_token(value: object) -> str:
    """Neutralize an attacker-controlled token for safe embedding in a log line or trace.

    A block / tool_choice ``type`` comes straight from the client request and is
    interpolated into a translation warning that reaches the human log and the
    structural trace. Strips non-printable characters (CWE-117 log injection) and caps
    the length so a hostile type cannot forge log records or flood the trace.
    """
    cleaned = "".join(ch for ch in str(value) if ch.isprintable())
    if len(cleaned) > _SAFE_TOKEN_MAX:
        return cleaned[:_SAFE_TOKEN_MAX] + "..."
    return cleaned


def _translate_thinking_block(block: dict) -> tuple[dict, list[str]]:
    """Translate an Anthropic thinking block per the configured reasoning mode."""
    if _XAI_REASONING_MODE == "drop":
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

    Spec: ``input_image.image_url`` is a STRING — a ``data:`` URL for base64 (proven
    accepted by image_input.json) or the source URL directly. Base64 outside the MIME
    allowlist, or a file/unknown source (no bytes), degrades to a redacted placeholder
    (never echoes the payload).
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
    """Translate an Anthropic tool_use block to a top-level xAI ``function_call`` item.

    xAI's cli-chat-proxy links a call to its result by ``call_id`` ALONE and accepts
    the id VERBATIM (proved by tool_result_replay_exact.json). So — unlike the OpenAI
    path, which rewrites ids to the ``fc_`` form — the Anthropic tool id is forwarded
    unchanged and NO separate item ``id`` is synthesized: the id surfaced on the
    response IS xAI's ``call_id``, and it must round-trip byte-for-byte.
    """
    return {
        "_toplevel": True,
        "type": "function_call",
        "call_id": block["id"],
        "name": block["name"],
        "arguments": json.dumps(block["input"]),
    }


def _tool_result_has_media(content: object) -> bool:
    """Report whether tool_result content carries an image/document block."""
    if not isinstance(content, list):
        return False
    return any(b.get("type") in _TOOL_RESULT_MEDIA_TYPES for b in content)


def _tool_result_string(content: object, is_error: bool) -> str:
    """Flatten tool_result content to a string, redacting media by type (never base64).

    A media block degrades to a bounded ``[media omitted: <type> — …]`` placeholder
    naming only the block type: B3's text-only backend cannot carry the bytes, and
    echoing the base64 payload would leak the tool's output. Real tool-output media
    forwarding lands in B4.
    """
    if isinstance(content, list):
        rendered = []
        for b in content:
            if b.get("type") == "text":
                rendered.append(b.get("text", ""))
            elif b.get("type") in _TOOL_RESULT_MEDIA_TYPES:
                rendered.append(
                    f"[media omitted: {_safe_token(b.get('type'))} — "
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
    """Translate an Anthropic tool_result to a top-level ``function_call_output`` item.

    ``call_id`` is the Anthropic ``tool_use_id`` VERBATIM — the same exact-linkage rule
    as ``_translate_tool_use_block``, and the property tool_result_replay_exact.json
    proves the model consumes. ``output`` is ``str | list[dict]``: when the content
    carries media AND the provider declares ``supports_tool_output_content_parts``, it is
    an ARRAY of real content parts (so tool-returned screenshots/PDFs reach a vision
    model); otherwise it is a string, and media in a string-only backend is redacted
    (never base64).
    """
    content = block.get("content", "")
    is_error = bool(block.get("is_error"))
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
        "call_id": block["tool_use_id"],
        "output": output,
    }, warnings


def _translate_unsupported_block(block_type: str) -> tuple[dict, list[str]]:
    """Degrade an unsupported block to a redacted, type-named placeholder.

    B3 is text+tools only, so image/document (until B4) and any special block
    (server_tool_use, ...) land here. The placeholder NEVER echoes the block's nested
    content: a raw ``str(block)`` would pollute the request AND leak media/tool inputs.
    """
    safe_type = _safe_token(block_type)
    warning = (
        f"Unsupported content block type '{safe_type}' replaced with a redacted "
        "placeholder (no provider equivalent)"
    )
    return {"type": "input_text", "text": f"[unsupported content block: {safe_type}]"}, [warning]


def _translate_content_block(
    block: dict, capabilities: ProviderCapabilities
) -> tuple[dict, list[str]]:
    """Translate one Anthropic content block to an xAI Responses input block.

    Returns ``(translated_block, warnings)``. tool_use / tool_result carry a special
    ``_toplevel`` key signaling the caller to emit them as top-level input items rather
    than nesting them inside a message's content array. Media blocks (image/document)
    become real Responses content parts when ``capabilities.input_modalities`` allows,
    and degrade to a redacted placeholder (never echoing base64) otherwise. Thin
    dispatcher — each type delegates to a helper so this stays under the CCN ceiling.
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


def _translate_message(
    message: dict, capabilities: ProviderCapabilities
) -> tuple[list[dict], list[str]]:
    """Translate one Anthropic message to a list of xAI Responses input items.

    Anthropic nests everything in messages with content blocks; the Responses API uses
    a flat input array where user text → ``input_text``, assistant text → ``output_text``,
    and tool_use / tool_result become TOP-LEVEL ``function_call`` / ``function_call_output``
    items sitting outside any message wrapper.
    """
    warnings: list[str] = []
    role = message.get("role", "user")
    content = message.get("content", [])

    # String shorthand → a single text block.
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
            # Assistant text is output_text, not input_text.
            if role == "assistant" and translated.get("type") == "input_text":
                translated = {"type": "output_text", "text": translated["text"]}
            nested_content.append(translated)

    items: list[dict] = []
    if nested_content:
        items.append({"role": role, "content": nested_content})
    items.extend(toplevel_items)
    return items, warnings


def _translate_tool_choice(tool_choice: dict) -> tuple[dict, list[str]]:
    """Map an Anthropic ``tool_choice`` to xAI Responses request fields.

    Returns ``(fields, warnings)``: ``tool_choice`` becomes ``"auto"`` / ``"none"`` /
    ``"required"`` or a forced ``{"type": "function", "name": ...}`` object, and
    ``parallel_tool_calls`` is set False when Anthropic's ``disable_parallel_tool_use``
    is on. An unknown choice type is omitted with a warning rather than guessed.
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


# grok-4.6+ accept a ``reasoning.effort`` parameter (U-EFFORT: omit/low/medium/high all HTTP
# 200); grok-4.20 and earlier 400 on it (field_effort_low.json). The model id is compared as a
# DECIMAL, not a version tuple: ``grok-4.20`` is 4.2 (older than 4.6), which is what makes the
# fixture consistent — 4.20 lacks the field, 4.6 has it, so 4.20 must rank older; a tuple would
# rank (4,20) newer than (4,6). config._version_tuple is deliberately NOT reused for this reason
# (client versions there are genuine semver tuples).
_XAI_EFFORT_MIN_VERSION = 4.6


def _parse_grok_version(model: str) -> float | None:
    """Return the decimal version in a ``grok-<major>.<minor>`` id, or None if it has none.

    ``grok-4.6`` -> 4.6, ``grok-4.20`` -> 4.2 (decimal), ``grok-build`` -> None.
    """
    match = re.search(r"grok-(\d+(?:\.\d+)?)", model)
    return float(match.group(1)) if match else None


def _model_accepts_reasoning_effort(model: str) -> bool:
    """Whether the resolved model accepts a ``reasoning.effort`` parameter.

    True for grok-4.6+ and for any non-versioned alias such as ``grok-build`` (the rolling
    latest-coding model, assumed modern); False only for a model that parses to a version
    below 4.6, which the proxy 400s on (field_effort_low.json). Gating on "provably old"
    rather than "provably new" keeps effort working on supported models while still refusing
    the one known-unsupported case, so an ``XAI_MODEL`` override cannot recreate the 4.20 400.
    """
    version = _parse_grok_version(model)
    return version is None or version >= _XAI_EFFORT_MIN_VERSION


def anthropic_to_xai(
    request: dict, capabilities: ProviderCapabilities = _XAI_TEXT_ONLY_CAPABILITIES
) -> tuple[dict, list[str]]:
    """Translate an Anthropic Messages request to an xAI (Grok) Responses request.

    Returns ``(translated_request, warnings)`` where warnings lists features stripped
    or degraded because they have no xAI equivalent. Pure function — no I/O.
    ``capabilities`` gates media forwarding: image/document blocks forward as Responses
    parts only when the modality is declared, else degrade to a redacted placeholder.
    Defaults to the conservative text-only set; the real path threads
    ``XAIProvider.capabilities`` via ``translate_request``.

    Two divergences from the OpenAI path are load-bearing: (1) tool ids are forwarded
    VERBATIM (no ``fc_`` rewrite, no synthesized item id) because xAI links by
    ``call_id`` alone; (2) ``reasoning.effort`` is model-gated — sent to grok-4.6+ (which
    accept it) and omitted for pre-4.6 models that 400 on it (field_effort_low.json), while
    encrypted reasoning continuity always rides ``include=reasoning.encrypted_content`` with
    ``store=false``.
    """
    warnings: list[str] = []

    for key in _XAI_STRIPPED_KEYS:
        if key in request:
            warnings.append(f"Stripped unsupported key '{key}' from request")

    if "thinking" in request:
        if _XAI_REASONING_MODE == "drop":
            warnings.append("Stripped 'thinking' config (reasoning_mode=drop)")
        else:
            warnings.append("Thinking config passed through (reasoning_mode=passthrough)")

    # include=reasoning.encrypted_content + store=false: the stateless model returns each
    # reasoning item's encrypted continuation blob, replayed before its function_call on the
    # next turn (reasoning continuity — B7). Orthogonal to reasoning.effort below — include
    # controls whether the encrypted blob is returned, effort controls how much the model thinks.
    model = config.xai_model()
    result: dict = {
        "model": model,
        "store": False,
        "stream": True,
        "include": ["reasoning.encrypted_content"],
    }

    # reasoning.effort tunes grok-4.6+ thinking length (low by default — a latency choice).
    # Model-gated: pre-4.6 models 400 on the field (field_effort_low.json).
    if _model_accepts_reasoning_effort(model):
        result["reasoning"] = {"effort": config.xai_reasoning_effort()}

    # Anthropic max_tokens -> Responses max_output_tokens. grok-4.6 has no fixed text cap, so
    # forwarding Claude's limit keeps slow turns bounded; omitted when absent or non-positive.
    max_tokens = request.get("max_tokens")
    if isinstance(max_tokens, int) and max_tokens > 0:
        result["max_output_tokens"] = max_tokens

    system = request.get("system")
    if isinstance(system, str):
        result["instructions"] = system
    elif isinstance(system, list):
        result["instructions"] = "\n".join(block.get("text", "") for block in system)
    else:
        result["instructions"] = _DEFAULT_XAI_INSTRUCTIONS

    # Tools — flat Responses structure. strict:false because Anthropic marks ALL params
    # required but Claude Code only fills the truly needed ones (single_tool_call.json).
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

    tool_choice = request.get("tool_choice")
    if tool_choice is not None:
        tc_fields, tc_warnings = _translate_tool_choice(tool_choice)
        result.update(tc_fields)
        warnings.extend(tc_warnings)

    input_items: list[dict] = []
    for message in request.get("messages", []):
        items, msg_warnings = _translate_message(message, capabilities)
        input_items.extend(items)
        warnings.extend(msg_warnings)
    result["input"] = input_items

    return result, warnings


# xAI Grok is billed through the subscription-metered cli-chat-proxy, so its token counts
# need no OpenAI-compat scaling — the multiplier is the identity. Kept as a named constant
# (not a bare 1.0) so B8's capability declaration references one owner. See D-XAI-005.
_XAI_TOKEN_COUNT_MULTIPLIER = 1.0

# Responses ``incomplete_details.reason`` that signals a moderation block, not token-budget
# exhaustion. Mapping a content-filtered turn to ``max_tokens`` makes Claude Code auto-compact
# a context nowhere near full and retry endlessly, so the two are disambiguated here.
_CONTENT_FILTER_REASON = "content_filter"

# Surfaced to Claude Code when a turn is content-filtered with no model text, so the turn
# renders as a visible refusal rather than a blank assistant message.
_CONTENT_FILTER_REFUSAL = (
    "I cannot complete this response because it was blocked by content safety filters. "
    "Please rephrase your request."
)


def _coerce_token_count(value: object) -> int:
    """Coerce a provider token count to a non-negative int.

    Provider usage may carry floats or nulls; Anthropic's usage fields are integers that
    Claude Code's ``/context`` math divides by. Non-numeric values default to 0.
    """
    if isinstance(value, (int, float)):
        return max(0, int(value))
    return 0


def _scale_token_count(value: object, multiplier: float) -> int:
    """Return a non-negative token count adjusted by the provider multiplier (half-up)."""
    return int(_coerce_token_count(value) * multiplier + 0.5)


def _anthropic_usage(
    xai_usage: object,
    *,
    token_count_multiplier: float = _XAI_TOKEN_COUNT_MULTIPLIER,
) -> dict:
    """Project xAI Responses usage onto Anthropic's flat integer shape.

    ``input_tokens`` already includes cached tokens and ``output_tokens`` already includes
    reasoning tokens (both subsets, per the Responses contract), so each maps to Anthropic's
    corresponding total. Cached tokens are deliberately NOT split into
    ``cache_read_input_tokens`` — Anthropic's totals are non-overlapping, so splitting would
    double-count. Missing or non-numeric fields default to 0.
    """
    usage = xai_usage if isinstance(xai_usage, dict) else {}
    return {
        "input_tokens": _scale_token_count(usage.get("input_tokens", 0), token_count_multiplier),
        "output_tokens": _scale_token_count(usage.get("output_tokens", 0), token_count_multiplier),
    }


def _incomplete_reason(response: dict) -> str:
    """Return ``incomplete_details.reason`` from a Responses object, or ``""`` if absent.

    A ``status: "incomplete"`` with a null ``incomplete_details`` reads as token exhaustion
    (the conservative default).
    """
    details = response.get("incomplete_details")
    return details.get("reason", "") if isinstance(details, dict) else ""


def _stop_reason(status: str, has_tool_calls: bool, incomplete_reason: str) -> str:
    """Map an xAI Responses terminal status to an Anthropic ``stop_reason``.

    Tool calls win — Claude Code must run the tool rather than compact. A content-filtered
    completion ends the turn cleanly (``end_turn``); any other ``incomplete`` is treated as
    output-token exhaustion (``max_tokens``), the signal Claude Code auto-compacts on.
    """
    if has_tool_calls:
        return "tool_use"
    if status == "incomplete":
        return "end_turn" if incomplete_reason == _CONTENT_FILTER_REASON else "max_tokens"
    return "end_turn"


def xai_to_anthropic(
    response: dict,
    *,
    token_count_multiplier: float = _XAI_TOKEN_COUNT_MULTIPLIER,
) -> dict:
    """Translate an xAI Responses API response to an Anthropic Messages API response.

    Pure function — no I/O. Diverges from the OpenAI provider on one point: a tool_use ``id``
    is the upstream ``call_id`` VERBATIM (no ``fc_``/``call_`` rewrite), so it round-trips to
    the request-side ``function_call_output`` call_id unchanged. Reasoning-continuity capture
    is layered on separately (that is the ``translate_response`` wrapper's job), not here.
    """
    status = response.get("status", "completed")
    output_items = response.get("output", [])
    has_tool_calls = any(i.get("type") == "function_call" for i in output_items)
    incomplete_reason = _incomplete_reason(response)
    stop_reason = _stop_reason(status, has_tool_calls, incomplete_reason)

    content: list[dict] = []
    for item in output_items:
        item_type = item.get("type")

        if item_type == "message":
            for block in item.get("content", []):
                if block.get("type") == "output_text":
                    content.append({"type": "text", "text": block["text"]})

        elif item_type == "refusal":
            content.append({"type": "text", "text": item.get("refusal", "")})

        elif item_type == "function_call":
            raw_args = item.get("arguments", "{}")
            try:
                parsed_args = json.loads(raw_args)
            except (json.JSONDecodeError, ValueError):
                parsed_args = {"_raw": raw_args}
            # xAI: expose the call_id EXACTLY as captured (no rewrite) so it round-trips.
            call_id = item.get("call_id") or item.get("id", "")
            content.append(
                {
                    "type": "tool_use",
                    "id": call_id,
                    "name": item["name"],
                    "input": parsed_args,
                }
            )

    # Content-filtered turn with no model text -> synthesize a visible refusal so the turn
    # never renders as a blank assistant message.
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
