"""xAI Grok provider — subscription OAuth token management + translation (stdlib only).

Auth mirrors the Codex OAuth mechanics (``providers/openai.py``) but reads the
grok CLI's own credential file, ``~/.grok/auth.json``: a single
``https://auth.x.ai::<client_id>`` keyed entry whose bearer JWT lives in ``key``
(not ``access_token``), sitting alongside profile sibling fields that must survive
a refresh. There is no API-key mode — the only credential source is the grok
subscription login.

Request/response/stream translation is still a stub (see ``XAIProvider`` below);
this module intentionally does not register ``XAIProvider`` in ``PROVIDERS`` yet.
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

from claude_bridge.auth import decode_jwt_exp

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


class XAIProvider:
    """xAI Grok provider — stub for extensibility proof."""

    name = "xai"
    endpoint = "https://api.x.ai/v1/chat/completions"

    async def authenticate(self) -> dict[str, str]:
        """Return xAI auth headers. Requires XAI_API_KEY env var."""
        raise NotImplementedError("xAI Grok provider not yet implemented")

    def translate_request(self, _anthropic_req: dict) -> tuple[dict, list[str]]:
        """Translate Anthropic request to xAI format."""
        raise NotImplementedError("xAI Grok provider not yet implemented")

    def translate_response(self, _provider_resp: dict) -> dict:
        """Translate xAI response back to Anthropic format."""
        raise NotImplementedError("xAI Grok provider not yet implemented")

    def translate_stream(self, _raw_chunks: AsyncIterator[bytes]) -> AsyncIterator[dict]:
        """Translate raw xAI byte chunks to Anthropic-format SSE events."""
        raise NotImplementedError("xAI Grok provider not yet implemented")
