"""Tests for auth utilities and Codex OAuth provider auth."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import stat
import time
from pathlib import Path

import pytest

from claude_bridge.auth import decode_jwt_exp, is_token_expired
from claude_bridge.providers.openai import (
    OpenAIProvider,
    get_bearer_token,
    read_codex_auth,
    refresh_access_token,
)
from claude_bridge.providers.openai.auth import (
    _TOKEN_OPENER,
    _TOKEN_URL,
    _NoRedirectHandler,
    _on_disk_access_token,
    _validated_bearer,
)


def _make_jwt(payload: dict) -> str:
    """Build a fake JWT with the given payload (no crypto verification needed)."""
    header = base64.urlsafe_b64encode(json.dumps({"alg": "RS256"}).encode()).rstrip(b"=")
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=")
    signature = base64.urlsafe_b64encode(b"fakesig").rstrip(b"=")
    return f"{header.decode()}.{body.decode()}.{signature.decode()}"


class _FakeTokenResp:
    """Minimal _TOKEN_OPENER.open context-manager stand-in returning a fixed JSON body."""

    def __init__(self, body: dict):
        self._data = json.dumps(body).encode()

    def read(self) -> bytes:
        return self._data

    def __enter__(self) -> _FakeTokenResp:
        return self

    def __exit__(self, *args: object) -> bool:
        return False


# --- decode_jwt_exp ---


class TestDecodeJwtExp:
    def test_extracts_exp_from_valid_token(self):
        token = _make_jwt({"exp": 1700000000, "sub": "user"})
        assert decode_jwt_exp(token) == 1700000000

    def test_extracts_float_exp(self):
        token = _make_jwt({"exp": 1700000000.5})
        assert decode_jwt_exp(token) == 1700000000.5

    def test_raises_on_missing_exp(self):
        token = _make_jwt({"sub": "user"})
        with pytest.raises(ValueError, match="missing 'exp' claim"):
            decode_jwt_exp(token)

    def test_raises_on_malformed_token_no_dots(self):
        with pytest.raises(ValueError, match="missing payload segment"):
            decode_jwt_exp("not-a-jwt")

    def test_raises_on_bad_base64_payload(self, monkeypatch):
        import binascii

        def _bad_decode(s):
            raise binascii.Error("Invalid base64")

        monkeypatch.setattr(base64, "urlsafe_b64decode", _bad_decode)
        with pytest.raises(ValueError, match="not valid base64"):
            decode_jwt_exp("header.payload.signature")

    def test_raises_on_non_json_payload(self):
        payload = base64.urlsafe_b64encode(b"not json at all").rstrip(b"=").decode()
        with pytest.raises(ValueError, match="not valid JSON"):
            decode_jwt_exp(f"header.{payload}.signature")


# --- is_token_expired ---


class TestIsTokenExpired:
    def test_expired_token_returns_true(self):
        expired_time = time.time() - 100
        token = _make_jwt({"exp": expired_time})
        assert is_token_expired(token) is True

    def test_future_token_returns_false(self):
        future_time = time.time() + 3600
        token = _make_jwt({"exp": future_time})
        assert is_token_expired(token) is False

    def test_margin_makes_near_future_expired(self):
        """Token expiring in 20s is considered expired with 30s margin."""
        near_future = time.time() + 20
        token = _make_jwt({"exp": near_future})
        assert is_token_expired(token, margin_seconds=30) is True

    def test_custom_margin_zero(self):
        """With zero margin, only truly expired tokens count."""
        near_future = time.time() + 5
        token = _make_jwt({"exp": near_future})
        assert is_token_expired(token, margin_seconds=0) is False

    def test_malformed_token_raises_with_context(self):
        with pytest.raises(ValueError, match="Cannot check token expiry"):
            is_token_expired("not-a-jwt")


# --- read_codex_auth ---


class TestReadCodexAuth:
    def test_reads_valid_auth_file(self, tmp_path: Path):
        auth_data = {
            "auth_mode": "chatgpt",
            "access_token": "tok_abc",
            "refresh_token": "ref_xyz",
        }
        auth_file = tmp_path / ".codex" / "auth.json"
        auth_file.parent.mkdir(parents=True)
        auth_file.write_text(json.dumps(auth_data))
        result = read_codex_auth(auth_file)
        assert result == auth_data

    def test_missing_file_raises_file_not_found(self, tmp_path: Path):
        missing = tmp_path / ".codex" / "auth.json"
        with pytest.raises(FileNotFoundError, match="codex login"):
            read_codex_auth(missing)

    def test_wrong_auth_mode_raises_value_error(self, tmp_path: Path):
        auth_data = {
            "auth_mode": "api_key",
            "access_token": "tok_abc",
        }
        auth_file = tmp_path / ".codex" / "auth.json"
        auth_file.parent.mkdir(parents=True)
        auth_file.write_text(json.dumps(auth_data))
        with pytest.raises(ValueError, match="chatgpt"):
            read_codex_auth(auth_file)


# --- get_bearer_token ---


class TestGetBearerToken:
    @pytest.mark.asyncio
    async def test_returns_valid_token_without_refresh(self, tmp_path: Path):
        """When token is not expired, returns it directly."""
        future_exp = time.time() + 3600
        token = _make_jwt({"exp": future_exp})
        auth_data = {
            "auth_mode": "chatgpt",
            "access_token": token,
            "refresh_token": "ref_xyz",
        }
        auth_file = tmp_path / ".codex" / "auth.json"
        auth_file.parent.mkdir(parents=True)
        auth_file.write_text(json.dumps(auth_data))
        result = await get_bearer_token(auth_file)
        assert result == token

    @pytest.mark.asyncio
    async def test_force_refresh_refreshes_even_a_valid_token(self, monkeypatch, tmp_path: Path):
        """Reactive 401 parity with xAI: a proactively-valid Codex token that upstream
        rejected must be force-refreshed, not returned as-is. Oracle: the mocked refresh
        returns a new access_token; force yields it, a no-op returns the old one."""
        valid_token = _make_jwt({"exp": time.time() + 3600})
        new_token = _make_jwt({"exp": time.time() + 7200})
        auth_data = {
            "auth_mode": "chatgpt",
            "access_token": valid_token,
            "refresh_token": "ref_xyz",
        }
        auth_file = tmp_path / ".codex" / "auth.json"
        auth_file.parent.mkdir(parents=True)
        auth_file.write_text(json.dumps(auth_data))

        class _FakeResp:
            def __init__(self) -> None:
                self._data = json.dumps({"access_token": new_token}).encode()

            def read(self) -> bytes:
                return self._data

            def __enter__(self) -> _FakeResp:
                return self

            def __exit__(self, *args: object) -> None:
                return None

        monkeypatch.setattr(
            "claude_bridge.providers.openai.auth._TOKEN_OPENER.open", lambda *a, **kw: _FakeResp()
        )
        result = await get_bearer_token(auth_file, force_refresh=True)
        assert result == new_token

    @pytest.mark.asyncio
    async def test_malformed_stored_token_raises_value_error(self, tmp_path: Path):
        """Malformed access_token in auth.json raises ValueError from is_token_expired."""
        auth_data = {
            "auth_mode": "chatgpt",
            "access_token": "not-a-jwt",
            "refresh_token": "ref_xyz",
        }
        auth_file = tmp_path / ".codex" / "auth.json"
        auth_file.parent.mkdir(parents=True)
        auth_file.write_text(json.dumps(auth_data))
        with pytest.raises(ValueError, match="Cannot check token expiry"):
            await get_bearer_token(auth_file)

    @pytest.mark.asyncio
    async def test_expired_token_refresh_failure_raises(self, monkeypatch, tmp_path: Path):
        """Expired token + refresh network error surfaces as ValueError."""
        expired_token = _make_jwt({"exp": time.time() - 100})
        auth_data = {
            "auth_mode": "chatgpt",
            "access_token": expired_token,
            "refresh_token": "ref_xyz",
        }
        auth_file = tmp_path / ".codex" / "auth.json"
        auth_file.parent.mkdir(parents=True)
        auth_file.write_text(json.dumps(auth_data))

        def _raise_timeout(*args, **kwargs):
            raise TimeoutError("Connection timed out")

        monkeypatch.setattr(
            "claude_bridge.providers.openai.auth._TOKEN_OPENER.open", _raise_timeout
        )
        with pytest.raises(ValueError, match="Token refresh failed"):
            await get_bearer_token(auth_file)

    @pytest.mark.asyncio
    async def test_token_without_exp_claim_raises(self, tmp_path: Path):
        """Token with valid JWT structure but no exp claim raises ValueError."""
        no_exp_token = _make_jwt({"sub": "user", "iat": 1700000000})
        auth_data = {
            "auth_mode": "chatgpt",
            "access_token": no_exp_token,
            "refresh_token": "ref_xyz",
        }
        auth_file = tmp_path / ".codex" / "auth.json"
        auth_file.parent.mkdir(parents=True)
        auth_file.write_text(json.dumps(auth_data))
        with pytest.raises(ValueError, match="Cannot check token expiry"):
            await get_bearer_token(auth_file)


class TestRefreshLock:
    """Auth refresh lock prevents concurrent stampede."""

    @pytest.mark.asyncio
    async def test_concurrent_refresh_uses_lock(self, tmp_path: Path):
        """Multiple concurrent get_bearer_token calls share one refresh."""
        future_exp = time.time() + 3600
        token = _make_jwt({"exp": future_exp})
        auth_data = {
            "auth_mode": "chatgpt",
            "access_token": token,
            "refresh_token": "ref_xyz",
        }
        auth_file = tmp_path / ".codex" / "auth.json"
        auth_file.parent.mkdir(parents=True)
        auth_file.write_text(json.dumps(auth_data))

        # Run 5 concurrent calls — all should return the same token
        results = await asyncio.gather(*[get_bearer_token(auth_file) for _ in range(5)])
        assert all(r == token for r in results)


# --- refresh_access_token error handling ---


class TestRefreshAccessTokenErrors:
    """Token refresh raises ValueError on network and response errors."""

    @pytest.mark.asyncio
    async def test_http_error_raises_value_error(self, monkeypatch, tmp_path: Path):
        import http.client
        import urllib.error

        def _raise_http_error(*args, **kwargs):
            raise urllib.error.HTTPError(
                "https://auth.openai.com/oauth/token",
                401,
                "Unauthorized",
                http.client.HTTPMessage(),
                None,
            )

        monkeypatch.setattr(
            "claude_bridge.providers.openai.auth._TOKEN_OPENER.open", _raise_http_error
        )
        auth_file = tmp_path / "auth.json"
        auth_file.write_text("{}")
        with pytest.raises(ValueError, match="Token refresh failed"):
            await refresh_access_token("fake-refresh-token", auth_path=auth_file)

    @pytest.mark.asyncio
    async def test_timeout_error_raises_value_error(self, monkeypatch, tmp_path: Path):
        def _raise_timeout(*args, **kwargs):
            raise TimeoutError("Connection timed out")

        monkeypatch.setattr(
            "claude_bridge.providers.openai.auth._TOKEN_OPENER.open", _raise_timeout
        )
        auth_file = tmp_path / "auth.json"
        auth_file.write_text("{}")
        with pytest.raises(ValueError, match="Token refresh failed"):
            await refresh_access_token("fake-refresh-token", auth_path=auth_file)

    @pytest.mark.asyncio
    async def test_missing_access_token_raises_value_error(self, monkeypatch, tmp_path: Path):

        class _FakeResp:
            def __init__(self):
                self._data = json.dumps({"refresh_token": "new-ref"}).encode()

            def read(self):
                return self._data

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        monkeypatch.setattr(
            "claude_bridge.providers.openai.auth._TOKEN_OPENER.open", lambda *a, **kw: _FakeResp()
        )
        auth_file = tmp_path / "auth.json"
        auth_file.write_text("{}")
        with pytest.raises(ValueError, match="missing 'access_token'"):
            await refresh_access_token("fake-refresh-token", auth_path=auth_file)


# --- _validated_bearer: header-safety gate for the outbound bearer ---


class TestValidatedBearer:
    """_validated_bearer — the header-safety gate for the outbound Codex bearer.

    Oracle: RFC 7235 defines a bearer credential as ``token68`` — printable ASCII
    with no control characters. Every expected verdict derives from that grammar,
    never from running the validator. Security invariant: a rejection message never
    contains the token value (CWE-532).
    """

    def test_clean_jwt_passes_through_unchanged(self):
        token = _make_jwt({"exp": time.time() + 3600})
        assert _validated_bearer(token) == token

    def test_empty_token_rejected(self):
        with pytest.raises(ValueError, match="malformed"):
            _validated_bearer("")

    def test_carriage_return_rejected(self):
        with pytest.raises(ValueError, match="malformed"):
            _validated_bearer("good\rX-Injected: evil")

    def test_newline_rejected(self):
        with pytest.raises(ValueError, match="malformed"):
            _validated_bearer("good\nX-Injected: evil")

    def test_tab_rejected(self):
        with pytest.raises(ValueError, match="malformed"):
            _validated_bearer("good\tinjected")

    def test_non_ascii_rejected(self):
        with pytest.raises(ValueError, match="malformed"):
            _validated_bearer("tökén-with-unicode")

    def test_message_never_contains_the_token_value(self):
        secret = "SUPER-SECRET-BEARER\r\ninjected"
        with pytest.raises(ValueError) as excinfo:
            _validated_bearer(secret)
        assert "SUPER-SECRET-BEARER" not in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_control_char_bearer_rejected_end_to_end(self, tmp_path: Path):
        # A fresh (valid-exp) but CR/LF-poisoned bearer must be rejected BEFORE it
        # reaches an Authorization header — else http.client echoes the secret into an
        # "Invalid header value" ValueError (CWE-532). Oracle: RFC 7235 forbids control
        # chars, and the raised message must not contain the token. Fails against code
        # that skips validation (it would return the poisoned token).
        valid = _make_jwt({"exp": time.time() + 3600})
        poisoned = valid + "\r\nX-Injected: evil"
        auth_data = {"auth_mode": "chatgpt", "access_token": poisoned, "refresh_token": "ref"}
        auth_file = tmp_path / ".codex" / "auth.json"
        auth_file.parent.mkdir(parents=True)
        auth_file.write_text(json.dumps(auth_data))
        with pytest.raises(ValueError, match="malformed") as excinfo:
            await get_bearer_token(auth_file)
        assert poisoned not in str(excinfo.value)
        assert "evil" not in str(excinfo.value)


# --- _NoRedirectHandler: the token endpoint never redirects (SSRF, CWE-918) ---


class TestNoRedirectOpener:
    """The refresh opener refuses every 3xx from the token endpoint.

    Oracle: a token endpoint never legitimately redirects; following a
    provider-controlled ``Location`` would drive an arbitrary internal request whose
    response is then parsed as token JSON (SSRF). The refusal must carry no redirect URL.
    """

    def test_handler_is_wired_into_the_refresh_opener(self):
        # Kills a mutant that drops _NoRedirectHandler from build_opener (redirects
        # would then be followed automatically). OpenerDirector.handlers is a real
        # documented CPython attribute the typeshed stub omits, hence the ignore.
        handlers = _TOKEN_OPENER.handlers  # type: ignore[attr-defined]
        assert any(isinstance(h, _NoRedirectHandler) for h in handlers)

    def test_redirect_refused_without_echoing_target(self):
        import http.client
        import io
        import urllib.error
        import urllib.request

        handler = _NoRedirectHandler()
        target = "http://169.254.169.254/latest/meta-data/"
        req = urllib.request.Request(_TOKEN_URL)
        with pytest.raises(urllib.error.URLError) as excinfo:
            handler.redirect_request(
                req, io.BytesIO(), 302, "Found", http.client.HTTPMessage(), target
            )
        assert target not in str(excinfo.value)
        assert "169.254.169.254" not in str(excinfo.value)


# --- _on_disk_access_token: best-effort re-read for the cross-process double-check ---


class TestOnDiskAccessToken:
    def test_returns_none_when_file_absent(self, tmp_path: Path):
        assert _on_disk_access_token(tmp_path / "nope.json") is None

    def test_returns_none_on_corrupt_json(self, tmp_path: Path):
        f = tmp_path / "auth.json"
        f.write_text("{ not json")
        assert _on_disk_access_token(f) is None

    def test_reads_flat_access_token(self, tmp_path: Path):
        f = tmp_path / "auth.json"
        f.write_text(json.dumps({"access_token": "tok-flat"}))
        assert _on_disk_access_token(f) == "tok-flat"

    def test_reads_nested_access_token(self, tmp_path: Path):
        f = tmp_path / "auth.json"
        f.write_text(json.dumps({"tokens": {"access_token": "tok-nested"}}))
        assert _on_disk_access_token(f) == "tok-nested"


# --- refresh_access_token: persistence, permissions, endpoint pinning ---


class TestRefreshAccessTokenPersistence:
    """The refresh persists the rotated secret atomically at 0600 and pins the endpoint."""

    def _expired_file(self, tmp_path: Path, **overrides) -> Path:
        auth_data = {
            "auth_mode": "chatgpt",
            "access_token": _make_jwt({"exp": time.time() - 100}),
            "refresh_token": "ref-original",
        }
        auth_data.update(overrides)
        auth_file = tmp_path / ".codex" / "auth.json"
        auth_file.parent.mkdir(parents=True)
        auth_file.write_text(json.dumps(auth_data))
        return auth_file

    @pytest.mark.asyncio
    async def test_persists_new_access_token_and_rotates_refresh(
        self, monkeypatch, tmp_path: Path
    ):
        # Oracle: the mocked response carries a new access_token AND refresh_token; both
        # must land in auth.json (RFC 6749 §6 single-use rotation). Fails against code that
        # returns the token without persisting, or drops the rotated refresh_token.
        new_access = _make_jwt({"exp": time.time() + 3600})
        auth_file = self._expired_file(tmp_path)
        monkeypatch.setattr(
            "claude_bridge.providers.openai.auth._TOKEN_OPENER.open",
            lambda *a, **kw: _FakeTokenResp(
                {"access_token": new_access, "refresh_token": "ref-rotated"}
            ),
        )
        returned = await refresh_access_token("ref-original", auth_path=auth_file)
        assert returned == new_access
        persisted = json.loads(auth_file.read_text())
        assert persisted["access_token"] == new_access
        assert persisted["refresh_token"] == "ref-rotated"

    @pytest.mark.asyncio
    async def test_keeps_refresh_token_when_response_omits_it(self, monkeypatch, tmp_path: Path):
        # RFC 6749 §6: a token response MAY omit refresh_token, meaning "keep the current
        # one". Fails against code that overwrites it with an empty/None value.
        new_access = _make_jwt({"exp": time.time() + 3600})
        auth_file = self._expired_file(tmp_path)
        monkeypatch.setattr(
            "claude_bridge.providers.openai.auth._TOKEN_OPENER.open",
            lambda *a, **kw: _FakeTokenResp({"access_token": new_access}),
        )
        await refresh_access_token("ref-original", auth_path=auth_file)
        assert json.loads(auth_file.read_text())["refresh_token"] == "ref-original"

    @pytest.mark.asyncio
    async def test_preserves_sibling_fields(self, monkeypatch, tmp_path: Path):
        # Fields the bridge does not own (account_id, auth_mode) survive the rewrite —
        # the whole document is re-serialized, only the two token fields change.
        new_access = _make_jwt({"exp": time.time() + 3600})
        auth_file = self._expired_file(tmp_path, account_id="acct-1")
        monkeypatch.setattr(
            "claude_bridge.providers.openai.auth._TOKEN_OPENER.open",
            lambda *a, **kw: _FakeTokenResp({"access_token": new_access}),
        )
        await refresh_access_token("ref-original", auth_path=auth_file)
        persisted = json.loads(auth_file.read_text())
        assert persisted["account_id"] == "acct-1"
        assert persisted["auth_mode"] == "chatgpt"

    @pytest.mark.asyncio
    async def test_tightens_permissions_to_owner_only(self, monkeypatch, tmp_path: Path):
        # The rotated secret is ALWAYS rewritten owner-only (0600, CWE-732). Even when the
        # prior file was world/group-readable, the refresh must tighten — never carry broad
        # bits forward (0644 in must become 0600 out). Fails against mode-preserving code.
        new_access = _make_jwt({"exp": time.time() + 3600})
        auth_file = self._expired_file(tmp_path)
        auth_file.chmod(0o644)  # start broad — a world-readable secret
        monkeypatch.setattr(
            "claude_bridge.providers.openai.auth._TOKEN_OPENER.open",
            lambda *a, **kw: _FakeTokenResp({"access_token": new_access}),
        )
        await refresh_access_token("ref-original", auth_path=auth_file)
        assert stat.S_IMODE(os.stat(auth_file).st_mode) == 0o600

    @pytest.mark.asyncio
    async def test_failure_leaves_file_byte_identical(self, monkeypatch, tmp_path: Path):
        # A failed refresh must not partially rewrite auth.json — the mkstemp temp is
        # unlinked and the original is untouched. Fails against code that truncates the
        # file before the POST.
        auth_file = self._expired_file(tmp_path)
        before = auth_file.read_bytes()

        def _boom(*a, **kw):
            raise TimeoutError("boom")

        monkeypatch.setattr("claude_bridge.providers.openai.auth._TOKEN_OPENER.open", _boom)
        with pytest.raises(ValueError, match="Token refresh failed"):
            await refresh_access_token("ref-original", auth_path=auth_file)
        assert auth_file.read_bytes() == before

    @pytest.mark.asyncio
    async def test_untrusted_endpoint_refused(self, monkeypatch, tmp_path: Path):
        # Defense in depth (CWE-918): the refresh_token is POSTed only to HTTPS on the
        # pinned host. Point _TOKEN_URL at an attacker host (the derived _TRUSTED_TOKEN_HOST
        # stays pinned to auth.openai.com) and the refresh must refuse before any network
        # call. Oracle: the endpoint-mismatch guard raises; _boom proves no POST fired.
        auth_file = self._expired_file(tmp_path)
        monkeypatch.setattr(
            "claude_bridge.providers.openai.auth._TOKEN_URL",
            "https://attacker.example/oauth/token",
        )

        def _boom(*a, **kw):
            raise AssertionError("must refuse before contacting an untrusted endpoint")

        monkeypatch.setattr("claude_bridge.providers.openai.auth._TOKEN_OPENER.open", _boom)
        with pytest.raises(ValueError, match="untrusted Codex token endpoint"):
            await refresh_access_token("ref-original", auth_path=auth_file)


class TestCrossProcessRefreshLock:
    """Refresh serializes across separate ``claude-codex`` processes and skips a
    now-redundant POST — the cross-process advisory lock plus its double-check.

    Codex shares the single-use refresh_token stampede risk (RFC 6749 §6): the
    process-local ``asyncio.Lock`` guards one event loop, so without an OS-level lock
    three separate processes each hold their own and stampede the endpoint, the first
    consumes the refresh_token, and the losers fail on the dead one. Oracle: a loser
    that re-reads a freshly-rotated valid token MUST use it and skip its own POST.
    """

    def _expired_file(self, tmp_path: Path) -> Path:
        auth_data = {
            "auth_mode": "chatgpt",
            "access_token": _make_jwt({"exp": time.time() - 100}),
            "refresh_token": "ref-original",
        }
        auth_file = tmp_path / ".codex" / "auth.json"
        auth_file.parent.mkdir(parents=True)
        auth_file.write_text(json.dumps(auth_data))
        return auth_file

    def _valid_file(self, tmp_path: Path, token: str) -> Path:
        auth_data = {"auth_mode": "chatgpt", "access_token": token, "refresh_token": "ref"}
        auth_file = tmp_path / ".codex" / "auth.json"
        auth_file.parent.mkdir(parents=True)
        auth_file.write_text(json.dumps(auth_data))
        return auth_file

    @pytest.mark.asyncio
    async def test_refresh_skips_network_when_token_already_valid(
        self, monkeypatch, tmp_path: Path
    ):
        # Double-check under the lock: a valid on-disk token (a peer rotated it while we
        # waited) short-circuits — the POST is skipped and the fresh token returned. Fails
        # against unconditional-POST code (which would call _boom and raise).
        valid = _make_jwt({"exp": time.time() + 3600})
        auth_file = self._valid_file(tmp_path, valid)

        def _boom(*a, **kw):
            raise AssertionError("network refresh must not run when the token is already valid")

        monkeypatch.setattr("claude_bridge.providers.openai.auth._TOKEN_OPENER.open", _boom)
        returned = await refresh_access_token("ref", auth_path=auth_file)
        assert returned == valid

    @pytest.mark.asyncio
    async def test_reactive_refresh_posts_when_ondisk_is_the_rejected_token(
        self, monkeypatch, tmp_path: Path
    ):
        # Reactive 401 path: the on-disk token passes the proactive expiry check yet
        # upstream just rejected it. Passing it as stale_token must force the POST — the
        # double-check may NOT short-circuit on "looks valid" and hand the rejected
        # credential back. Oracle: the mocked refresh returns a new token; a correct force
        # yields it. Kills a dropped ``!= stale_token`` clause and its flip.
        rejected = _make_jwt({"exp": time.time() + 3600})
        new_token = _make_jwt({"exp": time.time() + 7200})
        auth_file = self._valid_file(tmp_path, rejected)
        monkeypatch.setattr(
            "claude_bridge.providers.openai.auth._TOKEN_OPENER.open",
            lambda *a, **kw: _FakeTokenResp({"access_token": new_token}),
        )
        returned = await refresh_access_token("ref", auth_path=auth_file, stale_token=rejected)
        assert returned == new_token

    @pytest.mark.asyncio
    async def test_reactive_refresh_skips_when_a_peer_already_rotated(
        self, monkeypatch, tmp_path: Path
    ):
        # A peer refreshed while we waited: the on-disk token DIFFERS from the one we
        # rejected and is valid. The double-check must return the peer's fresh token and
        # skip the POST — our own refresh_token may be the single-use one the peer spent.
        # Oracle: the opener must never fire; the returned token is the peer's.
        rejected = _make_jwt({"exp": time.time() + 3600})
        peer_token = _make_jwt({"exp": time.time() + 7200})
        auth_file = self._valid_file(tmp_path, peer_token)

        def _boom(*a, **kw):
            raise AssertionError("refresh POST fired though a peer already rotated the token")

        monkeypatch.setattr("claude_bridge.providers.openai.auth._TOKEN_OPENER.open", _boom)
        returned = await refresh_access_token("ref", auth_path=auth_file, stale_token=rejected)
        assert returned == peer_token

    @pytest.mark.asyncio
    async def test_lock_released_after_successful_refresh(self, monkeypatch, tmp_path: Path):
        # After a successful refresh the lock is released — a subsequent non-blocking
        # exclusive acquire succeeds (no BlockingIOError). Kills a mutant that leaks the
        # fd/lock (dropped finally), which would deadlock every later refresh. The lock
        # file is owner-only (0600), consistent with ~/.codex hygiene.
        import fcntl

        auth_file = self._expired_file(tmp_path)
        new_token = _make_jwt({"exp": time.time() + 3600})
        monkeypatch.setattr(
            "claude_bridge.providers.openai.auth._TOKEN_OPENER.open",
            lambda *a, **kw: _FakeTokenResp({"access_token": new_token}),
        )
        await refresh_access_token("ref-original", auth_path=auth_file)

        lock_path = auth_file.parent / ".codex-refresh.lock"
        assert stat.S_IMODE(os.stat(lock_path).st_mode) == 0o600
        probe_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(probe_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(probe_fd, fcntl.LOCK_UN)
        finally:
            os.close(probe_fd)


# --- OpenAIProvider auth modes ---


class TestOpenAIProviderApiKeyAuth:
    """API key auth mode uses OPENAI_API_KEY env var."""

    def test_api_key_mode_sets_correct_endpoint(self):
        provider = OpenAIProvider(auth_mode="api_key", api_key="sk-test-123")
        assert provider.endpoint == "https://api.openai.com/v1/responses"

    @pytest.mark.asyncio
    async def test_api_key_mode_returns_bearer_header(self):
        provider = OpenAIProvider(auth_mode="api_key", api_key="sk-test-123")
        headers = await provider.authenticate()
        assert headers == {"Authorization": "Bearer sk-test-123"}

    @pytest.mark.asyncio
    async def test_api_key_mode_missing_key_raises(self):
        """API key mode with no key raises a clear error."""
        provider = OpenAIProvider(auth_mode="api_key", api_key=None)
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            await provider.authenticate()


class TestOpenAIProviderCodexOAuth:
    """Codex OAuth mode uses existing Codex auth flow."""

    def test_codex_oauth_mode_sets_correct_endpoint(self):
        provider = OpenAIProvider(auth_mode="codex_oauth")
        assert provider.endpoint == "https://chatgpt.com/backend-api/codex/responses"

    @pytest.mark.asyncio
    async def test_codex_oauth_mode_uses_bearer_token(self, tmp_path: Path):
        """Codex OAuth mode delegates to get_bearer_token."""
        future_exp = time.time() + 3600
        token = _make_jwt({"exp": future_exp})
        auth_data = {
            "auth_mode": "chatgpt",
            "access_token": token,
            "refresh_token": "ref_xyz",
        }
        auth_file = tmp_path / ".codex" / "auth.json"
        auth_file.parent.mkdir(parents=True)
        auth_file.write_text(json.dumps(auth_data))

        provider = OpenAIProvider(auth_mode="codex_oauth", auth_path=auth_file)
        headers = await provider.authenticate()
        assert headers == {"Authorization": f"Bearer {token}"}


class TestOpenAIProviderDefaults:
    """Default constructor behavior preserves backward compatibility."""

    def test_default_auth_mode_is_api_key(self):
        provider = OpenAIProvider(auth_mode="api_key", api_key="sk-test")
        assert provider.auth_mode == "api_key"

    def test_codex_oauth_mode_stored(self):
        provider = OpenAIProvider(auth_mode="codex_oauth")
        assert provider.auth_mode == "codex_oauth"


class TestDetectAuthMode:
    """Auth mode detection from environment in __main__.py."""

    def test_api_key_present_selects_api_key_mode(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-placeholder")
        from claude_bridge.__main__ import _detect_openai_auth_mode

        mode, key = _detect_openai_auth_mode()
        assert mode == "api_key"
        assert key == "test-key-placeholder"

    def test_empty_api_key_selects_codex_oauth(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "")
        from claude_bridge.__main__ import _detect_openai_auth_mode

        mode, key = _detect_openai_auth_mode()
        assert mode == "codex_oauth"
        assert key is None

    def test_missing_api_key_selects_codex_oauth(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from claude_bridge.__main__ import _detect_openai_auth_mode

        mode, key = _detect_openai_auth_mode()
        assert mode == "codex_oauth"
        assert key is None
