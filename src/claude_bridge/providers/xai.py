"""xAI Grok provider — subscription OAuth token management + translation (stdlib only).

Auth mirrors the Codex OAuth mechanics (``providers/openai.py``) but reads the
grok CLI's own credential file, ``~/.grok/auth.json``: a single
``https://auth.x.ai::<client_id>`` keyed entry whose bearer JWT lives in ``key``
(not ``access_token``), sitting alongside profile sibling fields that must survive
a refresh. There is no API-key mode — the only credential source is the grok
subscription login.

Request, response, and stream translation (Anthropic Messages <-> xAI Responses)
are all implemented on ``XAIProvider``, which registers itself in ``PROVIDERS``
under ``"xai"`` at import time. The backend is the grok CLI's subscription-metered
proxy (``cli-chat-proxy.grok.com``), reached with the subscription bearer plus the
``x-grok-client-version`` / ``x-grok-client-identifier`` gate headers.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime
import json
import os
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import AsyncIterator
from pathlib import Path

import claude_bridge.config as config
from claude_bridge.auth import decode_jwt_exp
from claude_bridge.content import MediaSource, parse_media_source
from claude_bridge.provider import PROVIDERS, ProviderCapabilities
from claude_bridge.stream import iter_sse_event_blobs, parse_sse_events

# The grok CLI's OAuth client is a *public* OIDC client identifier (like Codex's),
# not a secret. The refresh token endpoint is PINNED to the issuer below — never
# taken from the (attacker-controllable) auth.json — so a poisoned credential file
# cannot redirect the refresh_token POST to an arbitrary or internal host
# (SSRF / credential exfiltration, CWE-918/CWE-200).
_XAI_ISSUER = "https://auth.x.ai"
_XAI_AUTH_KEY_PREFIX = "https://auth.x.ai::"
_XAI_TOKEN_PATH = "/oauth2/token"  # noqa: S105  # nosec B105  # URL path, not a secret
# The one host the refresh_token may ever be POSTed to (derived from the pinned issuer).
_TRUSTED_ISSUER_HOST = urllib.parse.urlsplit(_XAI_ISSUER).hostname
_DEFAULT_XAI_AUTH_PATH = Path.home() / ".grok" / "auth.json"

# cli-chat-proxy gates each request on a client-identifier header alongside the version
# header. This is the grok CLI's own public client name, not a secret. Divergence from
# Codex (bearer-only) is documented in D-XAI-001 / D-XAI-002.
_XAI_CLIENT_IDENTIFIER = "grok-cli"


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


def _validated_bearer(token: str) -> str:
    """Return *token* iff it is safe to place in an ``Authorization`` header.

    A bearer JWT is ``token68`` (RFC 7235); we enforce the header-safety subset of
    that grammar — printable ASCII with no control characters (``isprintable()`` also
    admits a bare space, which no JWT contains and which cannot inject a header).
    Rejecting anything else BEFORE header construction stops a
    CR/LF- or control-char-bearing token (corrupt or poisoned ``auth.json``) from
    surfacing the secret inside an ``http.client`` "Invalid header value"
    ``ValueError`` and its traceback (credential leak, CWE-20/CWE-532).

    Raises:
        ValueError: If the token is empty or carries non-ASCII / non-printable
            characters. The message never contains the token value.
    """
    if not token or not token.isascii() or not token.isprintable():
        raise ValueError(
            "xAI bearer token is malformed (non-printable characters); "
            "run `grok` to re-authenticate."
        )
    return token


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
            return _validated_bearer(token)

        refresh_token = entry.get("refresh_token")
        if not refresh_token:
            msg = (
                f"xAI token expired and no refresh_token is present in the "
                f"'{entry_key}' entry; run `grok` to re-authenticate."
            )
            raise ValueError(msg)

        client_id = entry.get("oidc_client_id") or entry_key.split("::", 1)[-1]
        # Issuer is PINNED to _XAI_ISSUER (never entry.get("oidc_issuer")): the refresh
        # POST carries the refresh_token, so a poisoned auth.json must not choose its
        # destination host.
        return _validated_bearer(
            await refresh_xai_token(entry_key, refresh_token, client_id, auth_path=auth_path)
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
    entry, profile field, and unknown field, and rewriting at owner-only ``0600``
    perms (D-XAI-003), never the source file's possibly-broad bits.

    Returns:
        The new bearer JWT.

    Raises:
        ValueError: On network failure or a response missing ``access_token``.
            The on-disk file is left byte-for-byte unchanged on any failure.
    """
    resolved_path = auth_path or _DEFAULT_XAI_AUTH_PATH
    token_url = f"{issuer.rstrip('/')}{_XAI_TOKEN_PATH}"
    # Defense in depth: callers already pin the issuer, but refuse outright to POST the
    # refresh_token anywhere but HTTPS on the trusted xAI host — a standing guard against
    # any future reintroduction of a dynamic issuer (SSRF, CWE-918).
    _parts = urllib.parse.urlsplit(token_url)
    if _parts.scheme != "https" or _parts.hostname != _TRUSTED_ISSUER_HOST:
        raise ValueError("Refusing to refresh against an untrusted xAI token endpoint.")

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

        # Write the rotated secret through a uniquely-named 0600 descriptor in the
        # same directory: mkstemp opens O_EXCL (never follows a symlink, never reuses
        # a predictable name) and the file is owner-only from the instant it exists,
        # so there is no world-readable umask window (CWE-377/CWE-732). os.replace
        # then atomically swaps it in, carrying the 0600 mode onto auth.json —
        # matching D-XAI-003, not the source file's possibly-broad bits.
        fd, tmp_name = tempfile.mkstemp(dir=resolved_path.parent, prefix=".auth-", suffix=".tmp")
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w") as fh:
                fh.write(json.dumps(current, indent=2))
            os.replace(tmp_path, resolved_path)
        finally:
            # os.replace consumes tmp_path on success; on any failure before it, drop
            # the partial file rather than leaking a token-bearing temp into ~/.grok.
            tmp_path.unlink(missing_ok=True)

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


# Undrained SSE buffer ceiling. A well-formed provider event is far under 4 MiB, so a stream
# that grows past this without a ``\n\n`` terminator is malformed and is aborted rather than
# buffered unboundedly. Duplicated (not imported from the OpenAI provider) to keep this module
# self-contained — cross-provider imports are forbidden; cli-chat-proxy speaks the same
# Responses wire, so the bound is identical by coincidence of contract, not by shared code.
_MAX_SSE_BUFFER = 4 * 1024 * 1024

# Upper bound on a provider-controlled error message surfaced in an Anthropic error event.
# json.dumps escapes control characters on the wire, so this only guards against a hostile or
# huge message bloating the stream — not log injection.
_ERROR_MESSAGE_MAX = 500


def _sse_response_created(data: dict, *, token_count_multiplier: float) -> list[dict]:
    """Translate ``response.created`` → message_start + ping.

    ``response.created`` carries no output yet; input usage (if present this early) is scaled
    and output is reported as 0 until the terminal event supplies the final counts.
    """
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
                        "input_tokens": _scale_token_count(
                            usage.get("input_tokens", 0), token_count_multiplier
                        ),
                        "output_tokens": 0,
                    },
                },
            },
        },
        {"event": "ping", "data": {"type": "ping"}},
    ]


def _sse_output_item_added(data: dict) -> list[dict]:
    """Translate ``response.output_item.added`` → content_block_start for function_call items.

    xAI divergence: the tool_use ``id`` is the upstream ``call_id`` VERBATIM (no ``fc_``/
    ``call_`` rewrite), identical to the non-stream ``xai_to_anthropic`` path, so a tool call
    streamed and the same call replayed as a ``function_call_output`` share one id.
    Non-function_call items (reasoning, message) are surfaced via their own events, not here.
    """
    item = data.get("item", {})
    output_index = data.get("output_index", 0)
    if item.get("type") != "function_call":
        return []
    call_id = item.get("call_id") or item.get("id", "")
    return [
        {
            "event": "content_block_start",
            "data": {
                "type": "content_block_start",
                "index": output_index,
                "content_block": {
                    "type": "tool_use",
                    "id": call_id,
                    "name": item.get("name", ""),
                    "input": {},
                },
            },
        }
    ]


def _synthesize_refusal_block(text: str) -> list[dict]:
    """Build start/delta/stop SSE events for a synthetic refusal text block.

    Emitted when a streamed turn is content-filtered with no model text, so the stream does
    not end on an empty assistant message. The placeholder index 0 is reassigned to the next
    sequential Anthropic block index by ``_remap_block_index``.
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


def _sse_terminal_response(data: dict, *, token_count_multiplier: float) -> list[dict]:
    """Translate a terminal Responses event (``response.completed`` / ``response.incomplete``)
    → [refusal block?] + message_delta + message_stop.

    ``status`` and ``incomplete_details`` drive the stop_reason via the shared ``_stop_reason``:
    ``completed`` → ``end_turn`` (``tool_use`` when tool calls were emitted); ``incomplete`` →
    ``max_tokens`` unless the reason is ``content_filter``, which ends the turn cleanly
    (``end_turn``) and is prefixed with a synthesized refusal text block.
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
                "usage": _anthropic_usage(
                    resp.get("usage"), token_count_multiplier=token_count_multiplier
                ),
            },
        }
    )
    events.append({"event": "message_stop", "data": {"type": "message_stop"}})
    return events


def _sse_error_event(message: str) -> list[dict]:
    """Build an Anthropic ``error`` SSE event that terminates the stream.

    A failed or errored upstream response is an API error, not assistant output; the Anthropic
    streaming protocol ends such a stream with an ``error`` event rather than a
    ``message_stop``. The provider-controlled message is length-bounded.
    """
    text = (message or "Provider stream error").strip()[:_ERROR_MESSAGE_MAX]
    return [
        {
            "event": "error",
            "data": {"type": "error", "error": {"type": "api_error", "message": text}},
        }
    ]


def _sse_response_failed(data: dict) -> list[dict]:
    """Translate a ``response.failed`` event to an Anthropic error event.

    The ``response.error`` object carries a ``code`` (``server_error``,
    ``rate_limit_exceeded``, ...) and a human-readable ``message``.
    """
    resp = data.get("response")
    resp = resp if isinstance(resp, dict) else {}
    error = resp.get("error")
    error = error if isinstance(error, dict) else {}
    message = error.get("message") or error.get("code") or "Provider response failed"
    return _sse_error_event(message)


def _sse_top_level_error(data: dict) -> list[dict]:
    """Translate a bare top-level Responses ``error`` stream event (emitted on a mid-stream
    server failure) to an Anthropic error event."""
    error = data.get("error")
    error = error if isinstance(error, dict) else {}
    message = data.get("message") or error.get("message") or data.get("code")
    return _sse_error_event(message or "Provider stream error")


def _sse_synthetic_termination(has_tool_calls: bool) -> list[dict]:
    """Build the message_delta + message_stop for a stream that emitted a message_start but
    never received a terminal event (e.g. a dropped upstream connection).

    stop_reason is ``tool_use`` when tool calls were already emitted (Claude Code must run
    them), else ``end_turn`` — a clean stop that does NOT masquerade as token exhaustion and
    trigger an auto-compact retry loop. Usage is reported as zero output tokens since the true
    terminal usage never arrived.
    """
    stop_reason = "tool_use" if has_tool_calls else "end_turn"
    return [
        {
            "event": "message_delta",
            "data": {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason},
                "usage": {"output_tokens": 0},
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


def translate_xai_sse_event(
    event: dict,
    *,
    token_count_multiplier: float = _XAI_TOKEN_COUNT_MULTIPLIER,
) -> list[dict]:
    """Translate one xAI Responses API SSE event to Anthropic SSE events.

    Dispatches to sub-handlers by event type. Returns a list of ``{event, data}`` dicts (0, 1,
    or more). Pure function — no I/O, no reasoning capture (that is the ``translate_stream``
    wrapper's job in a later layer).
    """
    event_type = event.get("event", "")
    data = event.get("data", {})

    if event_type == "response.created":
        return _sse_response_created(data, token_count_multiplier=token_count_multiplier)

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

    if event_type in ("response.completed", "response.incomplete"):
        return _sse_terminal_response(data, token_count_multiplier=token_count_multiplier)

    if event_type == "response.failed":
        return _sse_response_failed(data)

    if event_type == "error":
        return _sse_top_level_error(data)

    if event_type in _SKIPPED_SSE_EVENTS:
        return []

    return []


def _remap_block_index(
    event: dict,
    index_map: dict[int, int],
    next_index: int,
    has_tool_calls: bool,
) -> tuple[dict, int, bool]:
    """Remap xAI output_index to sequential Anthropic block indices.

    xAI ``output_index`` may have gaps (skipped reasoning items), while Anthropic indices must
    be sequential from 0. Returns (possibly-modified event, updated next_index, updated
    has_tool_calls).
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
        # Encrypted reasoning items captured from each tool turn, keyed by the EXACT upstream
        # call_id, so they can be re-injected before their function_calls on the next request.
        # In-memory only — opaque, never persisted, never logged, never returned to Claude Code.
        self._reasoning_by_call_id: dict[str, dict] = {}
        self._reasoning_lock = threading.Lock()

    async def authenticate(self) -> dict[str, str]:
        """Return the subscription bearer plus the grok client headers.

        Divergence from Codex (bearer-only): cli-chat-proxy rejects a request whose
        ``x-grok-client-version`` is older than its floor or that lacks a client identifier,
        so both accompany the bearer. The opaque bearer is never logged and rides only in the
        ``Authorization`` header.

        Raises:
            FileNotFoundError / ValueError: Propagated from ``get_xai_bearer_token`` when the
                grok auth file is absent, malformed, or expired with no refresh token.
        """
        token = await get_xai_bearer_token(self._auth_path)
        return {
            "Authorization": f"Bearer {token}",
            "x-grok-client-version": config.xai_client_version(),
            "x-grok-client-identifier": _XAI_CLIENT_IDENTIFIER,
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
        """Translate an Anthropic Messages request to an xAI Responses request, echoing any
        captured encrypted reasoning back before its function_calls."""
        result, warnings = anthropic_to_xai(anthropic_req, self.capabilities)
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
