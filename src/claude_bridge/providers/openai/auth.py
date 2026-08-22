"""OpenAI Codex OAuth token management — read auth.json, refresh access tokens.

Stdlib only. Self-contained: no dependency on the translation, stream, or
provider-class submodules.
"""

from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from claude_bridge.auth import is_token_expired

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


async def get_bearer_token(auth_path: Path | None = None, *, force_refresh: bool = False) -> str:
    """Return a valid access token, refreshing if expired.

    Uses an asyncio.Lock to prevent concurrent refresh stampede — multiple
    callers with expired tokens share a single refresh operation.

    Args:
        force_refresh: Force a refresh regardless of the proactive expiry check —
            the reactive path after an upstream 401 rejects a token that still looks
            unexpired. Default False keeps proactive behavior (refresh only on expiry).
    """
    async with _refresh_lock:
        data = read_codex_auth(auth_path)
        tokens = data.get("tokens", data)  # support both nested and flat structures
        token = tokens["access_token"]

        if not force_refresh and not is_token_expired(token):
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
