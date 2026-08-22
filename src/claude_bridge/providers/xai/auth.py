"""xAI Grok subscription OAuth — read ~/.grok/auth.json, validate + refresh the bearer.

Reads the grok CLI's own credential file: a single ``https://auth.x.ai::<client_id>``
keyed entry whose bearer JWT lives in ``key`` (not ``access_token``). No API-key mode —
the only credential source is the grok subscription login. The refresh endpoint is
pinned to the issuer (SSRF defense, D-XAI-007). Stdlib only; self-contained leaf.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime
import json
import os
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX (Windows); flock is POSIX-only
    fcntl = None  # type: ignore[assignment]

from claude_bridge.auth import decode_jwt_exp

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
# A dedicated sibling lock file — never auth.json itself, whose inode os.replace swaps out
# from under any lock held on it. Its own inode is stable, so an flock on it is honored by
# every process that opens the same path.
_XAI_REFRESH_LOCK_FILENAME = ".xai-refresh.lock"

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


@contextlib.contextmanager
def _cross_process_refresh_lock(auth_dir: Path):
    """Serialize the OAuth refresh across separate bridge processes via ``flock``.

    The module-level ``asyncio.Lock`` guards a single event loop, but three
    concurrent ``claude-grok`` processes each hold their own — so without an
    OS-level lock they stampede the refresh endpoint, the first consumes the
    single-use refresh_token, and the losers ``400`` on the dead one. A blocking
    *exclusive* advisory lock on a dedicated sibling file (see
    ``_XAI_REFRESH_LOCK_FILENAME``) makes the refresh section mutually exclusive
    across processes; the winner refreshes and the losers re-read (double-check)
    the now-valid token and skip their own POST.

    On a platform without ``fcntl`` (Windows) this degrades to a no-op: the
    single-process case is already covered by the ``asyncio.Lock``, and there is
    no multi-process story to coordinate.
    """
    if fcntl is None:  # pragma: no cover - non-POSIX; asyncio.Lock covers single-process
        yield
        return
    lock_path = auth_dir / _XAI_REFRESH_LOCK_FILENAME
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        # Closing the descriptor releases the advisory lock (POSIX); no explicit
        # LOCK_UN needed, and the release is immediate since close follows the yield.
        os.close(fd)


async def get_xai_bearer_token(auth_path: Path | None = None) -> str:
    """Return a valid xAI bearer, refreshing via OIDC if expired.

    Serializes concurrent refreshes at two levels: an ``asyncio.Lock`` within one
    event loop, and a cross-process ``flock`` (see ``refresh_xai_token``) across
    separate ``claude-grok`` processes — so callers with an expired token share a
    single refresh instead of stampeding the endpoint and burning the single-use
    refresh_token.

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

    Acquires a cross-process advisory lock (``flock`` on a dedicated sibling file)
    for the whole check-refresh-write section, then double-checks the on-disk
    token: if a peer ``claude-grok`` process already rotated it while this caller
    waited for the lock, the now-valid token is returned and no POST is made — the
    caller's own refresh_token may be the single-use one that peer already spent.

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

    def _refresh_critical() -> str:
        # Double-check under the lock: another bridge process may have refreshed
        # while we waited to acquire it. If the on-disk token is now valid, use it
        # and skip the POST — our captured refresh_token may be the consumed one.
        snapshot: dict = json.loads(resolved_path.read_text()) if resolved_path.exists() else {}
        existing = snapshot.get(entry_key)
        if not isinstance(existing, dict):
            existing = {}
        existing_key = existing.get("key")
        if isinstance(existing_key, str) and not _xai_token_expired(existing):
            return existing_key

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

        # Re-read immediately before the write — as late as possible, after the POST —
        # so a concurrent grok-CLI update (external; our flock does not serialize it)
        # is not clobbered, and preserve every field we do not own.
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

    def _do_refresh() -> str:
        # Acquire the cross-process lock, then run the critical section. flock is a
        # blocking syscall, so it must be held inside the worker thread — never
        # across an await, which would block the event loop.
        with _cross_process_refresh_lock(resolved_path.parent):
            return _refresh_critical()

    return await asyncio.to_thread(_do_refresh)
