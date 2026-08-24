"""Content-free structural views of Anthropic requests and responses — stdlib only.

Two derivations that read only the *shape* of a request/response, never its
content: token estimation for the ``count_tokens`` endpoint, and redacted
structural summaries for the compatibility trace. They are kept together because
both build media descriptors from the same blocks via the shared ``_media_descriptor``
helper, so splitting them would push that connascence across a module boundary. Nothing
here copies prompt text, tool arguments, tool results, reasoning payloads, or base64
media into its output; redaction is enforced by construction.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

from claude_bridge.content import parse_media_source
from claude_bridge.log import get_logger, is_trace_enabled, trace_event

logger = get_logger("request_view")

# Approximate bytes-per-token ratio for mixed code/natural language traffic.
_BYTES_PER_TOKEN = 3.5

# Flat per-modality token budgets for media blocks. A model's cost for an image
# or document is dominated by fixed per-item processing — OpenAI bills vision by
# 512px-tile count and files by per-page render+extract — NOT by the base64 byte
# length. So media contributes a flat budget, never its encoded payload size: a
# 200 KB pasted image counted as text is ~57k phantom tokens, which wrecks Claude
# Code's auto-compact signal. Values are representative (a high-detail OpenAI
# vision image is ~765-1445 tokens; a multi-page document is larger) and bias
# toward over-counting so auto-compact trips early, not late. Limitation: the
# document budget does not scale with page count — refine with the
# token_count_multiplier follow-up.
_IMAGE_TOKEN_ESTIMATE = 1200
_DOCUMENT_TOKEN_ESTIMATE = 3000
_MEDIA_TOKEN_ESTIMATES = {"image": _IMAGE_TOKEN_ESTIMATE, "document": _DOCUMENT_TOKEN_ESTIMATE}

# A single media block whose decoded payload exceeds this is logged. No hard cap —
# _MAX_REQUEST_BODY already bounds the whole request body.
_OVERSIZED_MEDIA_BYTES = 5 * 1024 * 1024  # 5 MiB decoded


def _approx_decoded_bytes(data: str) -> int:
    """Approximate the decoded size of a base64 string without decoding it.

    base64 encodes 3 bytes per 4 characters, so the decoded size is ~len*3/4.
    Used for trace size summaries and the oversized-media warning only — the
    payload itself is never decoded, logged, or copied into a trace.
    """
    return len(data) * 3 // 4


def _media_descriptor(block: dict) -> dict:
    """Structural ``{kind, media_type, approx_bytes}`` for one media block.

    Derives the block shape from ``content.parse_media_source`` — the single owner
    of Anthropic media-block parsing (INV-MEDIA-01) — rather than re-reading
    ``source["media_type"]`` / ``source["data"]`` here, so this trace view cannot
    drift from the shape the providers forward. Only the normalized media type and
    the base64 length reach the result — never the payload — so it is safe to
    persist to a trace. ``approx_bytes`` is non-zero only for an inline base64
    source (``data`` populated); a url/file source carries no bytes to size.
    """
    media = parse_media_source(block)
    return {
        "kind": media.kind,
        "media_type": media.media_type,
        "approx_bytes": _approx_decoded_bytes(media.data) if media.data is not None else 0,
    }


def _iter_media_blocks(content: object) -> Iterator[dict]:
    """Yield image/document blocks in a content value, descending into tool_result.

    Media may sit at the top level of a message or nested inside tool_result
    content (T-005); both reach the provider, so both are surfaced here.
    """
    if not isinstance(content, list):
        return
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type in _MEDIA_TOKEN_ESTIMATES:
            yield block
        elif block_type == "tool_result":
            yield from _iter_media_blocks(block.get("content"))


def _oversized_media(descriptors: list[dict]) -> list[dict]:
    """Descriptors whose decoded payload exceeds ``_OVERSIZED_MEDIA_BYTES``."""
    return [d for d in descriptors if d["approx_bytes"] > _OVERSIZED_MEDIA_BYTES]


def _content_token_units(content: object) -> tuple[int, int, list[dict]]:
    """Return ``(text_bytes, media_tokens, media_descriptors)`` for a content value.

    Image/document blocks contribute a flat per-modality token budget AND a
    structural descriptor (surfaced from this single walk so the caller need not
    re-traverse for the oversized-media scan); tool_result content is walked so
    nested media is budgeted and described identically; every other block is
    counted by JSON byte size, matching the pre-media estimate.
    """
    if not isinstance(content, list):
        return len(json.dumps(content).encode()), 0, []
    text_bytes = 0
    media_tokens = 0
    descriptors: list[dict] = []
    for block in content:
        block_type = block.get("type") if isinstance(block, dict) else None
        if block_type in _MEDIA_TOKEN_ESTIMATES:
            media_tokens += _MEDIA_TOKEN_ESTIMATES[block_type]
            descriptors.append(_media_descriptor(block))
        elif block_type == "tool_result":
            wrapper = {k: v for k, v in block.items() if k != "content"}
            nested_bytes, nested_media, nested_descriptors = _content_token_units(
                block.get("content")
            )
            text_bytes += len(json.dumps(wrapper).encode()) + nested_bytes
            media_tokens += nested_media
            descriptors.extend(nested_descriptors)
        else:
            text_bytes += len(json.dumps(block).encode())
    return text_bytes, media_tokens, descriptors


def _message_token_units(message: object) -> tuple[int, int, list[dict]]:
    """Return ``(text_bytes, media_tokens, media_descriptors)`` for one message.

    String/non-list content is counted whole, byte-identical to the pre-media
    estimate; list content has its media blocks budgeted flatly (see
    ``_content_token_units``) while the role wrapper is still counted by bytes.
    """
    if not isinstance(message, dict) or not isinstance(message.get("content"), list):
        return len(json.dumps(message).encode()), 0, []
    wrapper = {k: v for k, v in message.items() if k != "content"}
    text_bytes, media_tokens, descriptors = _content_token_units(message["content"])
    return len(json.dumps(wrapper).encode()) + text_bytes, media_tokens, descriptors


def estimate_input_tokens(request: dict) -> int:
    """Estimate input token count by walking the Anthropic request structure.

    Counts UTF-8 JSON bytes of text content (system, message text/tool blocks,
    tool definitions) at ~3.5 bytes/token. Image and document blocks — top-level
    or nested in tool_result — instead contribute a FLAT per-modality budget
    (``_IMAGE_TOKEN_ESTIMATE`` / ``_DOCUMENT_TOKEN_ESTIMATE``), because a model's
    media cost is dominated by fixed per-item processing, not the base64 byte
    length. Logs a warning for any single media block whose decoded payload
    exceeds ``_OVERSIZED_MEDIA_BYTES`` (no hard cap — ``_MAX_REQUEST_BODY`` bounds
    the request). Returns 0 for empty/malformed requests. Provider-agnostic —
    operates on the Anthropic request format.
    """
    text_bytes = 0
    media_tokens = 0
    media_descriptors: list[dict] = []
    system = request.get("system")
    if system is not None:
        text_bytes += len(json.dumps(system).encode())
    for message in request.get("messages", []):
        message_bytes, message_media, message_descriptors = _message_token_units(message)
        text_bytes += message_bytes
        media_tokens += message_media
        media_descriptors.extend(message_descriptors)
    tools = request.get("tools")
    if tools:
        text_bytes += len(json.dumps(tools).encode())
    for descriptor in _oversized_media(media_descriptors):
        logger.warning(
            "Oversized %s media (~%d bytes) forwarded without a hard cap",
            descriptor["kind"],
            descriptor["approx_bytes"],
        )
    if text_bytes == 0 and media_tokens == 0:
        return 0
    return int(text_bytes / _BYTES_PER_TOKEN + 0.5) + media_tokens


def estimate_tokens(body: bytes) -> int:
    """Estimate input tokens from raw request body bytes."""
    try:
        return estimate_input_tokens(json.loads(body))
    except (json.JSONDecodeError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# Redacted compatibility trace — structural summaries + self-guarding hooks.
#
# The summarizers below are the redaction allowlist: each constructs a dict of
# explicitly named structural fields (counts, type names, tool names, lengths,
# token totals, stop reasons). They NEVER copy prompt text, tool arguments,
# tool results, reasoning payloads, request headers, or credentials into the
# trace. Redaction is enforced by construction here, not by discipline at the
# call sites. The hooks self-guard on ``is_trace_enabled()`` so the host
# functions carry zero added complexity and zero overhead when tracing is off.
# ---------------------------------------------------------------------------


def _block_type_counts(blocks: object) -> dict[str, int]:
    """Count content blocks by their ``type`` field — structure only, no content."""
    counts: dict[str, int] = {}
    if not isinstance(blocks, list):
        return counts
    for block in blocks:
        if isinstance(block, dict):
            block_type = str(block.get("type", "unknown"))
            counts[block_type] = counts.get(block_type, 0) + 1
    return counts


def _summarize_anthropic_request(request: dict) -> dict:
    """Structural-only summary of an inbound Anthropic request.

    Emits model, stream flag, message/tool counts, tool names, top-level block
    type counts, the tool_choice *type*, the system prompt *length*, and a media
    list of ``{kind, media_type, approx_bytes}`` for every image/document block
    (top-level or tool_result-nested) — never any prompt text, tool argument,
    tool result, or base64 media payload.
    """
    messages = request.get("messages")
    message_list = messages if isinstance(messages, list) else []
    block_types: dict[str, int] = {}
    media_descriptors: list[dict] = []
    for message in message_list:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            block_types["text"] = block_types.get("text", 0) + 1
        else:
            for block_type, count in _block_type_counts(content).items():
                block_types[block_type] = block_types.get(block_type, 0) + count
        media_descriptors.extend(_media_descriptor(block) for block in _iter_media_blocks(content))
    tools = request.get("tools") or []
    tool_names = sorted(str(tool.get("name", "")) for tool in tools if isinstance(tool, dict))
    tool_choice = request.get("tool_choice")
    system = request.get("system")
    return {
        "model": str(request.get("model", "")),
        "stream": bool(request.get("stream")),
        "message_count": len(message_list),
        "system_chars": len(json.dumps(system)) if system is not None else 0,
        "block_types": block_types,
        "tool_count": len(tools),
        "tool_names": tool_names,
        "tool_choice": tool_choice.get("type") if isinstance(tool_choice, dict) else tool_choice,
        "media": media_descriptors,
    }


def _summarize_provider_request(translated: dict, warnings: list[str]) -> dict:
    """Structural-only summary of a translated provider request.

    Emits model, stream flag, input item count, tool count/names, the resolved
    tool_choice, reasoning effort, and the translation warnings — both the count
    and the sanitized warning strings, which name what was stripped — never any
    translated input content. The warning strings are neutralized at construction
    (see ``_safe_token``), so they are safe to persist to the trace.
    """
    tools = translated.get("tools") or []
    tool_names = sorted(str(tool.get("name", "")) for tool in tools if isinstance(tool, dict))
    tool_choice = translated.get("tool_choice")
    if isinstance(tool_choice, dict):
        tool_choice = f"{tool_choice.get('type')}:{tool_choice.get('name')}"
    reasoning = translated.get("reasoning")
    summary = {
        "model": str(translated.get("model", "")),
        "stream": bool(translated.get("stream")),
        "input_items": len(translated.get("input") or []),
        "tool_count": len(tools),
        "tool_names": tool_names,
        "tool_choice": tool_choice,
        "reasoning_effort": reasoning.get("effort") if isinstance(reasoning, dict) else None,
        "warning_count": len(warnings),
        "warnings": list(warnings),
    }
    if "parallel_tool_calls" in translated:
        summary["parallel_tool_calls"] = bool(translated.get("parallel_tool_calls"))
    return summary


def _summarize_anthropic_response(response: dict) -> dict:
    """Structural-only summary of an outbound Anthropic response.

    Emits model, stop_reason, block type counts, and token usage — never the
    response text or tool_use arguments.
    """
    usage = response.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    return {
        "model": str(response.get("model", "")),
        "stop_reason": response.get("stop_reason"),
        "block_types": _block_type_counts(response.get("content")),
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
    }


def _summarize_stream_event(event: dict) -> dict:
    """Structural-only summary of one translated Anthropic SSE event.

    Emits the event name, block index, block/delta *type*, stop_reason, and
    output token total — never the streamed text or partial tool-argument JSON.
    """
    data = event.get("data")
    data = data if isinstance(data, dict) else {}
    summary: dict = {"sse": event.get("event", "")}
    if "index" in data:
        summary["index"] = data.get("index")
    content_block = data.get("content_block")
    if isinstance(content_block, dict):
        summary["block_type"] = content_block.get("type")
    delta = data.get("delta")
    if isinstance(delta, dict):
        if "type" in delta:
            summary["delta_type"] = delta.get("type")
        if "stop_reason" in delta:
            summary["stop_reason"] = delta.get("stop_reason")
    usage = data.get("usage")
    if isinstance(usage, dict) and "output_tokens" in usage:
        summary["output_tokens"] = usage.get("output_tokens")
    return summary


def trace_inbound_request(body: bytes) -> None:
    """Trace the structural shape of an inbound Anthropic request, if enabled."""
    if not is_trace_enabled():
        return
    try:
        trace_event("inbound_request", _summarize_anthropic_request(json.loads(body)))
    except Exception:
        logger.debug("inbound trace failed", exc_info=True)


def _trace_provider_request(translated: dict, warnings: list[str]) -> None:
    """Trace the structural shape of a translated provider request, if enabled."""
    if not is_trace_enabled():
        return
    try:
        trace_event("provider_request", _summarize_provider_request(translated, warnings))
    except Exception:
        logger.debug("provider request trace failed", exc_info=True)


# Routine, expected-every-request translation notices — logged at DEBUG, not WARNING, because
# the launchers share the bridge's stderr with the Claude Code TUI and these fire on normal
# traffic: thinking is on essentially every request, grok clamps the caller's max on every
# request, and Claude Code sends output_config.format (a structured-output hint the Responses
# API path does not take) on every request. Dropping an unsupported output_config *hint* is
# expected, non-actionable bridge behavior — the trace still records the full notice — so the
# whole subkey-drop class is routine (amends D-EFFORT-001, which first kept it a WARNING).
#
# Matched by ANCHORED PREFIX, not substring: each prefix is FIXED text, and every notice
# interpolates client-controlled text (an effort value, a subkey name) only AFTER its fixed
# prefix — so startswith tests the fixed part alone and a crafted value cannot forge or escape
# a classification. The genuinely-lossy notices — "Unrecognized" effort, "Unsupported"
# tool_choice, "Invalid" override, and content/tool drops phrased "Dropped <content> …" —
# begin with words that are NOT prefixes here, so they stay loud by default. The one "Dropped"
# prefix below is specific to "Dropped unsupported output_config." and never matches a
# content/media drop (guarded by test_non_output_config_dropped_notice_stays_warning).
#
# This is an ALLOWLIST: a notice matching no prefix logs at WARNING, so a genuinely-lossy or
# unforeseen notice stays loud by default. A translator rewording one of these prefixes without
# updating this tuple is caught by the request_view coupling tests, not silently re-flooded.
_ROUTINE_TRANSLATION_PREFIXES = (
    "Thinking config passed through",  # thinking passed through — every request
    "Stripped 'thinking' config",  # thinking dropped — every request in drop mode
    "output_config.effort '",  # grok effort clamp (max/xhigh -> high) — every request
    "Dropped unsupported output_config.",  # unsupported output_config hint, e.g. format
)


def _is_routine_translation(message: str) -> bool:
    """True if a translation notice is routine/expected — logged at DEBUG, not WARNING."""
    return any(message.startswith(prefix) for prefix in _ROUTINE_TRANSLATION_PREFIXES)


def emit_translation_warnings(warnings: list[str], translated: dict) -> None:
    """Surface translation warnings to every observer — the human log and the
    structural trace — from a single place.

    Both the streaming and non-streaming request paths route their warnings here so the
    logged warnings and the traced warnings can never drift out of lockstep. Routine notices
    (thinking passthrough/drop, the grok effort clamp, dropped output_config hints) log at
    DEBUG to keep the shared TUI quiet; genuinely-lossy notices (content/tool drops,
    unrecognized values) log at WARNING. The trace always records the full list, so
    diagnosability is unchanged regardless of log level.
    """
    for warning in warnings:
        if _is_routine_translation(warning):
            logger.debug("Translation: %s", warning)
        else:
            logger.warning("Translation: %s", warning)
    _trace_provider_request(translated, warnings)


def trace_provider_response(response: dict) -> None:
    """Trace the structural shape of a translated provider response, if enabled."""
    if not is_trace_enabled():
        return
    try:
        trace_event("provider_response", _summarize_anthropic_response(response))
    except Exception:
        logger.debug("provider response trace failed", exc_info=True)


def trace_stream_event(event: dict) -> None:
    """Trace the structural shape of one translated SSE event, if enabled."""
    if not is_trace_enabled():
        return
    try:
        trace_event("stream_event", _summarize_stream_event(event))
    except Exception:
        logger.debug("stream event trace failed", exc_info=True)
