"""xAI Grok provider — subscription OAuth token management + translation (stdlib only).

Auth mirrors the Codex OAuth mechanics (``providers/openai.py``) but reads the
grok CLI's own credential file, ``~/.grok/auth.json``: a single
``https://auth.x.ai::<client_id>`` keyed entry whose bearer JWT lives in ``key``
(not ``access_token``), sitting alongside profile sibling fields that must survive
a refresh. There is no API-key mode — the only credential source is the grok
subscription login.

Request translation (Anthropic Messages -> xAI Responses) is implemented below;
response and stream translation are still stubs (see ``XAIProvider``). This module
intentionally does not register ``XAIProvider`` in ``PROVIDERS`` yet.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime
import json
import os
import stat
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import AsyncIterator
from pathlib import Path

import claude_bridge.config as config
from claude_bridge.auth import decode_jwt_exp
from claude_bridge.content import MediaSource, parse_media_source
from claude_bridge.provider import ProviderCapabilities

# The grok CLI's OAuth client is a *public* OIDC client identifier (like Codex's),
# not a secret. The token endpoint is derived from the entry's ``oidc_issuer``;
# this value is the guaranteed fallback matching the grok CLI's issuer.
_XAI_ISSUER = "https://auth.x.ai"
_XAI_AUTH_KEY_PREFIX = "https://auth.x.ai::"
_XAI_TOKEN_PATH = "/oauth2/token"  # noqa: S105  # nosec B105  # URL path, not a secret
_DEFAULT_XAI_AUTH_PATH = Path.home() / ".grok" / "auth.json"


def _iso_to_timestamp(value: str) -> float:
    """Parse an ISO-8601 timestamp (optional trailing ``Z``) to epoch seconds.

    Raises:
        ValueError: If the string is not a valid ISO-8601 timestamp.
    """
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    return datetime.datetime.fromisoformat(normalized).timestamp()


def _timestamp_to_iso(ts: float) -> str:
    """Format epoch seconds as the ISO-8601 'Z' form xAI uses in ``expires_at``."""
    return datetime.datetime.fromtimestamp(ts, datetime.UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def read_xai_auth(path: Path | None = None) -> tuple[str, dict]:
    """Read ``~/.grok/auth.json`` and return the single xAI OIDC entry.

    Returns:
        A ``(entry_key, entry)`` tuple where ``entry_key`` is the
        ``https://auth.x.ai::<client_id>`` top-level key and ``entry`` is its dict.

    Raises:
        FileNotFoundError: If the auth file does not exist (hint: run ``grok``).
        ValueError: If no xAI entry is present, or more than one is (ambiguous).
    """
    auth_path = path or _DEFAULT_XAI_AUTH_PATH
    if not auth_path.exists():
        msg = (
            f"Grok auth file not found at {auth_path}. "
            "Run `grok` and sign in to authenticate first."
        )
        raise FileNotFoundError(msg)

    data: dict = json.loads(auth_path.read_text())
    xai_entries = [
        (k, v)
        for k, v in data.items()
        if k.startswith(_XAI_AUTH_KEY_PREFIX) and isinstance(v, dict)
    ]
    if not xai_entries:
        msg = (
            f"No xAI OAuth entry (top-level key prefixed '{_XAI_AUTH_KEY_PREFIX}') "
            f"found in {auth_path}. Run `grok` and sign in first."
        )
        raise ValueError(msg)
    if len(xai_entries) > 1:
        keys = ", ".join(k for k, _ in xai_entries)
        msg = f"Multiple xAI OAuth entries in {auth_path}; cannot disambiguate ({keys})."
        raise ValueError(msg)
    return xai_entries[0]


def _xai_token_expired(entry: dict, margin_seconds: int = 30) -> bool:
    """Return True if the entry's bearer expires within *margin_seconds*.

    Expiry is the *earlier* of the JWT ``exp`` claim and the ``expires_at``
    bookkeeping field — whichever fires first governs. If neither is parseable
    the token is treated as expired, forcing a refresh (safe default).
    """
    candidates: list[float] = []
    token = entry.get("key")
    if isinstance(token, str):
        with contextlib.suppress(ValueError):
            candidates.append(decode_jwt_exp(token))
    expires_at = entry.get("expires_at")
    if isinstance(expires_at, str):
        with contextlib.suppress(ValueError):
            candidates.append(_iso_to_timestamp(expires_at))
    if not candidates:
        return True
    return time.time() + margin_seconds >= min(candidates)


_xai_refresh_lock = asyncio.Lock()


async def get_xai_bearer_token(auth_path: Path | None = None) -> str:
    """Return a valid xAI bearer, refreshing via OIDC if expired.

    Uses an ``asyncio.Lock`` to prevent a concurrent refresh stampede — multiple
    callers with an expired token share one refresh.

    Note: refresh is *proactive* (expiry checked before each request). There is
    no reactive refresh-on-401 — a token expiring mid-flight surfaces as an
    upstream auth error, recovered on the next request cycle.

    Raises:
        FileNotFoundError / ValueError: Propagated from ``read_xai_auth`` /
            ``refresh_xai_token``; also raised if the token is expired and no
            refresh token is present.
    """
    async with _xai_refresh_lock:
        entry_key, entry = read_xai_auth(auth_path)
        token = entry.get("key")
        if isinstance(token, str) and not _xai_token_expired(entry):
            return token

        refresh_token = entry.get("refresh_token")
        if not refresh_token:
            msg = (
                f"xAI token expired and no refresh_token is present in the "
                f"'{entry_key}' entry; run `grok` to re-authenticate."
            )
            raise ValueError(msg)

        client_id = entry.get("oidc_client_id") or entry_key.split("::", 1)[-1]
        issuer = entry.get("oidc_issuer") or _XAI_ISSUER
        return await refresh_xai_token(
            entry_key, refresh_token, client_id, issuer=issuer, auth_path=auth_path
        )


async def refresh_xai_token(
    entry_key: str,
    refresh_token: str,
    client_id: str,
    *,
    issuer: str = _XAI_ISSUER,
    auth_path: Path | None = None,
) -> str:
    """Exchange a refresh token for a new bearer and persist it atomically.

    POSTs ``grant_type=refresh_token`` to ``<issuer>/oauth2/token`` (RFC 6749
    §6), then rewrites ``~/.grok/auth.json`` updating ONLY the selected entry's
    ``key`` / ``refresh_token`` / ``expires_at`` — preserving every sibling
    entry, profile field, unknown field, and the file's permission bits.

    Returns:
        The new bearer JWT.

    Raises:
        ValueError: On network failure or a response missing ``access_token``.
            The on-disk file is left byte-for-byte unchanged on any failure.
    """
    resolved_path = auth_path or _DEFAULT_XAI_AUTH_PATH
    token_url = f"{issuer.rstrip('/')}{_XAI_TOKEN_PATH}"

    def _do_refresh() -> str:
        body = urllib.parse.urlencode(
            {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": client_id,
            }
        ).encode()
        req = urllib.request.Request(token_url, data=body, method="POST")  # noqa: S310
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

        # RFC 6749 §5.1 token response — field names confirmed against the grok
        # CLI binary's own strings (access_token / refresh_token / expires_in).
        try:
            new_key: str = token_data["access_token"]
        except KeyError as exc:
            raise ValueError("Token refresh failed: response missing 'access_token'") from exc
        new_refresh_token: str = token_data.get("refresh_token", refresh_token)

        # Re-read before write so we never clobber a concurrent grok-CLI update,
        # and preserve every field we do not own.
        current: dict = json.loads(resolved_path.read_text()) if resolved_path.exists() else {}
        entry = current.get(entry_key)
        if not isinstance(entry, dict):
            entry = {}
        entry["key"] = new_key
        entry["refresh_token"] = new_refresh_token
        # Keep expires_at in sync with the new JWT's exp so the earlier-of check
        # does not immediately re-expire on stale bookkeeping (refresh loop).
        try:
            entry["expires_at"] = _timestamp_to_iso(decode_jwt_exp(new_key))
        except ValueError:
            expires_in = token_data.get("expires_in")
            if isinstance(expires_in, (int, float)):
                entry["expires_at"] = _timestamp_to_iso(time.time() + expires_in)
            else:
                entry.pop("expires_at", None)
        current[entry_key] = entry

        # Preserve the secret file's permission bits across the atomic replace.
        try:
            mode = stat.S_IMODE(os.stat(resolved_path).st_mode)
        except OSError:
            mode = 0o600
        tmp_path = resolved_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(current, indent=2))
        os.chmod(tmp_path, mode)
        os.replace(tmp_path, resolved_path)

        return new_key

    return await asyncio.to_thread(_do_refresh)


# ---------------------------------------------------------------------------
# Anthropic -> xAI (Grok) request translation (pure functions, no I/O)
# ---------------------------------------------------------------------------

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
_XAI_SAFE_TOKEN_MAX = 64

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
    if len(cleaned) > _XAI_SAFE_TOKEN_MAX:
        return cleaned[:_XAI_SAFE_TOKEN_MAX] + "..."
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


def _has_cache_control(request: dict) -> bool:
    """Return True if any part of the request carries cache_control hints."""
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
    ``call_id`` alone; (2) NO ``reasoning`` key is sent — cli-chat-proxy 400s on
    ``reasoning.effort`` (field_effort_low.json), and encrypted reasoning arrives via
    ``include=reasoning.encrypted_content`` with ``store=false`` instead.
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

    # include=reasoning.encrypted_content + store=false: the stateless model returns
    # each reasoning item's encrypted continuation blob, replayed before its
    # function_call on the next turn (reasoning continuity — B7). No ``reasoning`` key:
    # ``reasoning.effort`` 400s (field_effort_low.json); xAI applies its own default.
    result: dict = {
        "model": config.xai_model(),
        "store": False,
        "stream": True,
        "include": ["reasoning.encrypted_content"],
    }

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

    if _has_cache_control(request):
        warnings.append(
            "Stripped cache_control hints (no provider equivalent — caching is automatic)"
        )

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


class XAIProvider:
    """xAI Grok provider — subscription-OAuth backend on cli-chat-proxy.

    Request and non-stream response translation are implemented (``anthropic_to_xai`` and
    ``xai_to_anthropic`` above). Auth headers and stream translation are still stubs; this
    module does not yet register ``XAIProvider`` in ``PROVIDERS``.
    """

    name = "xai"
    endpoint = "https://api.x.ai/v1/chat/completions"
    capabilities = _XAI_TEXT_ONLY_CAPABILITIES

    async def authenticate(self) -> dict[str, str]:
        """Return xAI auth headers. Requires XAI_API_KEY env var."""
        raise NotImplementedError("xAI Grok provider not yet implemented")

    def translate_request(self, anthropic_req: dict) -> tuple[dict, list[str]]:
        """Translate an Anthropic Messages request to an xAI Responses request."""
        return anthropic_to_xai(anthropic_req, self.capabilities)

    def translate_response(self, provider_resp: dict) -> dict:
        """Translate an xAI Responses object back to Anthropic Messages format."""
        return xai_to_anthropic(
            provider_resp,
            token_count_multiplier=self.capabilities.token_count_multiplier,
        )

    def translate_stream(self, _raw_chunks: AsyncIterator[bytes]) -> AsyncIterator[dict]:
        """Translate raw xAI byte chunks to Anthropic-format SSE events."""
        raise NotImplementedError("xAI Grok provider not yet implemented")
