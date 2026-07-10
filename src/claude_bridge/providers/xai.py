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

# Content-block types that carry media inside a tool_result. B3 is text+tools only,
# so a media block here degrades to a redacted string (never base64); real tool-output
# media forwarding lands in B4.
_TOOL_RESULT_MEDIA_TYPES = frozenset({"image", "document"})

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


def _translate_tool_result_block(block: dict) -> tuple[dict, list[str]]:
    """Translate an Anthropic tool_result to a top-level ``function_call_output`` item.

    ``call_id`` is the Anthropic ``tool_use_id`` VERBATIM — the same exact-linkage rule
    as ``_translate_tool_use_block``, and the property tool_result_replay_exact.json
    proves the model consumes. ``output`` is a string; media in the content degrades to
    a redacted placeholder (never base64) with a warning.
    """
    content = block.get("content", "")
    is_error = bool(block.get("is_error"))
    output = _tool_result_string(content, is_error)
    warnings: list[str] = []
    if _tool_result_has_media(content):
        warnings = [
            "tool_result media redacted to string "
            "(provider/auth mode does not support tool-output media)"
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


def _translate_content_block(block: dict) -> tuple[dict, list[str]]:
    """Translate one Anthropic content block to an xAI Responses input block.

    Returns ``(translated_block, warnings)``. tool_use / tool_result carry a special
    ``_toplevel`` key signaling the caller to emit them as top-level input items rather
    than nesting them inside a message's content array. Thin dispatcher — each type
    delegates to a helper so this stays under the CCN ceiling as types grow (B4 adds
    the image/document cases here).
    """
    block_type = block.get("type", "unknown")
    if block_type == "text":
        return {"type": "input_text", "text": block["text"]}, []
    if block_type == "thinking":
        return _translate_thinking_block(block)
    if block_type == "tool_use":
        return _translate_tool_use_block(block), []
    if block_type == "tool_result":
        return _translate_tool_result_block(block)
    return _translate_unsupported_block(block_type)


def _translate_message(message: dict) -> tuple[list[dict], list[str]]:
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
        translated, block_warnings = _translate_content_block(block)
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


def anthropic_to_xai(request: dict) -> tuple[dict, list[str]]:
    """Translate an Anthropic Messages request to an xAI (Grok) Responses request.

    Returns ``(translated_request, warnings)`` where warnings lists features stripped
    or degraded because they have no xAI equivalent. Pure function — no I/O.

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
        items, msg_warnings = _translate_message(message)
        input_items.extend(items)
        warnings.extend(msg_warnings)
    result["input"] = input_items

    if _has_cache_control(request):
        warnings.append(
            "Stripped cache_control hints (no provider equivalent — caching is automatic)"
        )

    return result, warnings


class XAIProvider:
    """xAI Grok provider — subscription-OAuth backend on cli-chat-proxy.

    Request translation is implemented (``anthropic_to_xai`` above). Auth headers,
    response, and stream translation are still stubs; this module does not yet register
    ``XAIProvider`` in ``PROVIDERS``.
    """

    name = "xai"
    endpoint = "https://api.x.ai/v1/chat/completions"
    capabilities = _XAI_TEXT_ONLY_CAPABILITIES

    async def authenticate(self) -> dict[str, str]:
        """Return xAI auth headers. Requires XAI_API_KEY env var."""
        raise NotImplementedError("xAI Grok provider not yet implemented")

    def translate_request(self, anthropic_req: dict) -> tuple[dict, list[str]]:
        """Translate an Anthropic Messages request to an xAI Responses request."""
        return anthropic_to_xai(anthropic_req)

    def translate_response(self, _provider_resp: dict) -> dict:
        """Translate xAI response back to Anthropic format."""
        raise NotImplementedError("xAI Grok provider not yet implemented")

    def translate_stream(self, _raw_chunks: AsyncIterator[bytes]) -> AsyncIterator[dict]:
        """Translate raw xAI byte chunks to Anthropic-format SSE events."""
        raise NotImplementedError("xAI Grok provider not yet implemented")
