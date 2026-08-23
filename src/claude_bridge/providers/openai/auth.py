"""OpenAI Codex OAuth token management — read auth.json, refresh access tokens.

Stdlib only. Self-contained: no dependency on the translation, stream, or
provider-class submodules, and — per D-XAI-002 — no cross-provider imports; the
credential-hardening helpers here intentionally duplicate the xAI ones rather than
share a module, so each provider's auth stays independently auditable.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX (Windows); flock is POSIX-only
    fcntl = None  # type: ignore[assignment]

from claude_bridge.auth import is_token_expired

_CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
_TOKEN_URL = "https://auth.openai.com/oauth/token"  # noqa: S105  # nosec B105
# The one host the refresh_token may ever be POSTed to (derived from the pinned URL
# above). Unlike xAI, the Codex token URL is a fixed constant — never read from the
# (attacker-controllable) auth.json — so it cannot be redirected by a poisoned file;
# the host check below is defense-in-depth against a future dynamic endpoint.
_TRUSTED_TOKEN_HOST = urllib.parse.urlsplit(_TOKEN_URL).hostname
_DEFAULT_AUTH_PATH = Path.home() / ".codex" / "auth.json"
# A dedicated sibling lock file — never auth.json itself, whose inode os.replace swaps
# out from under any lock held on it. Its own inode is stable, so an flock on it is
# honored by every process that opens the same path.
_CODEX_REFRESH_LOCK_FILENAME = ".codex-refresh.lock"


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """A redirect handler that refuses every redirect from the token endpoint.

    Default urllib follows 3xx automatically, so a provider-controlled
    ``Location: http://127.0.0.1:...`` on the token POST would drive an arbitrary
    internal request whose response is then parsed as token JSON (SSRF, CWE-918).
    A token endpoint never legitimately redirects; refusing outright removes the
    vector. The raised error carries no redirect URL (never echo the attacker's
    target).
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.URLError("refusing to follow a redirect from the token endpoint")


# One opener reused for every refresh: no-redirect, no proxy inference from the
# environment (an empty ProxyHandler ignores http(s)_proxy so the pinned host is
# contacted directly, not via an attacker-set proxy).
_TOKEN_OPENER = urllib.request.build_opener(_NoRedirectHandler, urllib.request.ProxyHandler({}))


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


def _validated_bearer(token: str) -> str:
    """Return *token* iff it is safe to place in an ``Authorization`` header.

    A bearer JWT is ``token68`` (RFC 7235); we enforce the header-safety subset of
    that grammar — printable ASCII with no control characters (``isprintable()`` also
    admits a bare space, which no JWT contains and which cannot inject a header).
    Rejecting anything else BEFORE header construction stops a CR/LF- or
    control-char-bearing credential (corrupt/poisoned ``auth.json`` or a malformed
    ``OPENAI_API_KEY``) from surfacing the secret inside an ``http.client``
    "Invalid header value" ``ValueError`` and its traceback, or in a transport
    exception string later logged (credential leak, CWE-20/CWE-113/CWE-532).

    Raises:
        ValueError: If the token is empty or carries non-ASCII / non-printable
            characters. The message never contains the token value.
    """
    if not token or not token.isascii() or not token.isprintable():
        raise ValueError(
            "Codex credential is malformed (non-printable characters); "
            "run `codex login` to re-authenticate."
        )
    return token


_refresh_lock = asyncio.Lock()


@contextlib.contextmanager
def _cross_process_refresh_lock(auth_dir: Path):
    """Serialize the OAuth refresh across separate bridge processes via ``flock``.

    The module-level ``asyncio.Lock`` guards a single event loop, but multiple
    concurrent ``claude-codex`` processes each hold their own — so without an
    OS-level lock they stampede the refresh endpoint, the first consumes the
    single-use refresh_token, and the losers fail on the dead one. A blocking
    *exclusive* advisory lock on a dedicated sibling file (see
    ``_CODEX_REFRESH_LOCK_FILENAME``) makes the refresh section mutually exclusive
    across processes; the winner refreshes and the losers re-read (double-check)
    the now-valid token and skip their own POST.

    On a platform without ``fcntl`` (Windows) this degrades to a no-op: the
    single-process case is already covered by the ``asyncio.Lock``, and there is
    no multi-process story to coordinate.
    """
    if fcntl is None:  # pragma: no cover - non-POSIX; asyncio.Lock covers single-process
        yield
        return
    lock_path = auth_dir / _CODEX_REFRESH_LOCK_FILENAME
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        # Closing the descriptor releases the advisory lock (POSIX); no explicit
        # LOCK_UN needed, and the release is immediate since close follows the yield.
        os.close(fd)


async def get_bearer_token(auth_path: Path | None = None, *, force_refresh: bool = False) -> str:
    """Return a valid access token, refreshing if expired.

    Serializes concurrent refreshes at two levels: an ``asyncio.Lock`` within one
    event loop, and a cross-process ``flock`` (see ``refresh_access_token``) across
    separate ``claude-codex`` processes — so callers with an expired token share a
    single refresh instead of stampeding the endpoint and burning the single-use
    refresh_token. The returned token is header-safety validated (never a
    control-char-bearing credential).

    Args:
        force_refresh: Force a refresh regardless of the proactive expiry check —
            the reactive path after an upstream 401 rejects a token that still looks
            unexpired, threading it through as ``stale_token`` so a peer's freshly
            rotated token is honored but the rejected one is never returned. Default
            False keeps proactive behavior (refresh only on expiry).
    """
    async with _refresh_lock:
        data = read_codex_auth(auth_path)
        tokens = data.get("tokens", data)  # support both nested and flat structures
        token = tokens["access_token"]

        if not force_refresh and not is_token_expired(token):
            return _validated_bearer(token)

        # A forced refresh passes the (still-valid-looking but upstream-rejected) token
        # as stale_token so the double-check under the lock does not hand it back.
        stale_token = token if force_refresh else None
        new_token = await refresh_access_token(
            tokens["refresh_token"], auth_path=auth_path, stale_token=stale_token
        )
        return _validated_bearer(new_token)


def _on_disk_access_token(resolved_path: Path) -> str | None:
    """Return the on-disk access_token if present, else None (best-effort re-read)."""
    if not resolved_path.exists():
        return None
    try:
        snapshot: dict = json.loads(resolved_path.read_text())
    except (OSError, ValueError):
        return None
    tokens = snapshot.get("tokens", snapshot) if isinstance(snapshot, dict) else {}
    token = tokens.get("access_token") if isinstance(tokens, dict) else None
    return token if isinstance(token, str) else None


async def refresh_access_token(
    refresh_token: str,
    auth_path: Path | None = None,
    *,
    stale_token: str | None = None,
) -> str:
    """Exchange a refresh token for a new access token.

    POSTs to the pinned OpenAI token endpoint (no redirects followed), updates the
    local auth.json atomically at owner-only ``0600`` perms, and returns the new
    access_token.

    Acquires a cross-process advisory lock (``flock`` on a dedicated sibling file)
    for the whole check-refresh-write section, then double-checks the on-disk token:
    if a peer ``claude-codex`` process already rotated it while this caller waited
    for the lock, the now-valid token is returned and no POST is made — the caller's
    own refresh_token may be the single-use one that peer already spent.

    Args:
        stale_token: The just-rejected access_token on the reactive (post-401) path.
            The double-check skips only if a *different* valid token is on disk (a
            peer rotated it), never if the on-disk token is still the rejected one.
            ``None`` (proactive default) reduces the double-check to skip-if-valid.

    Raises:
        ValueError: On network failure, a refused redirect, or a response missing
            ``access_token``. The on-disk file is left unchanged on any failure.
    """
    resolved_path = auth_path or _DEFAULT_AUTH_PATH

    # Defense in depth: the endpoint is a fixed HTTPS constant, but refuse outright to
    # POST the refresh_token anywhere but HTTPS on the trusted host — a standing guard
    # against any future reintroduction of a dynamic endpoint (SSRF, CWE-918).
    _parts = urllib.parse.urlsplit(_TOKEN_URL)
    if _parts.scheme != "https" or _parts.hostname != _TRUSTED_TOKEN_HOST:
        raise ValueError("Refusing to refresh against an untrusted Codex token endpoint.")

    def _refresh_critical() -> str:
        # Double-check under the lock: another bridge process may have refreshed while
        # we waited. If a *different* valid token is on disk, use it and skip the POST —
        # our captured refresh_token may be the consumed one. In reactive mode the
        # just-rejected token is stale_token, so a still-on-disk rejected token never
        # short-circuits the refresh.
        existing = _on_disk_access_token(resolved_path)
        if existing is not None and existing != stale_token and not is_token_expired(existing):
            return existing

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
            with _TOKEN_OPENER.open(req, timeout=30) as resp:
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

        # Re-read immediately before the write (as late as possible, after the POST) so a
        # concurrent external update is not clobbered, preserving every field we do not
        # own and the original nested/flat token layout.
        current: dict = json.loads(resolved_path.read_text()) if resolved_path.exists() else {}
        nested = current.get("tokens")
        target = nested if isinstance(nested, dict) else current
        target["access_token"] = new_access_token
        target["refresh_token"] = new_refresh_token

        # Write the rotated secret through a uniquely-named 0600 descriptor in the same
        # directory: mkstemp opens O_EXCL (never follows a symlink, never reuses a
        # predictable name) and the file is owner-only from the instant it exists, so
        # there is no world-readable umask window (CWE-377/CWE-732/CWE-59). os.replace
        # then atomically swaps it in, carrying the 0600 mode onto auth.json.
        fd, tmp_name = tempfile.mkstemp(dir=resolved_path.parent, prefix=".auth-", suffix=".tmp")
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w") as fh:
                fh.write(json.dumps(current, indent=2))
            os.replace(tmp_path, resolved_path)
        finally:
            # os.replace consumes tmp_path on success; on any failure before it, drop the
            # partial file rather than leaking a token-bearing temp into ~/.codex.
            tmp_path.unlink(missing_ok=True)

        return new_access_token

    def _do_refresh() -> str:
        # Acquire the cross-process lock, then run the critical section. flock is a
        # blocking syscall, so it must be held inside the worker thread — never across
        # an await, which would block the event loop.
        with _cross_process_refresh_lock(resolved_path.parent):
            return _refresh_critical()

    return await asyncio.to_thread(_do_refresh)
