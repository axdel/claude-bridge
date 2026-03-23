"""Tests for auth utilities and Codex OAuth provider auth."""

from __future__ import annotations

import asyncio
import base64
import json
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


def _make_jwt(payload: dict) -> str:
    """Build a fake JWT with the given payload (no crypto verification needed)."""
    header = base64.urlsafe_b64encode(json.dumps({"alg": "RS256"}).encode()).rstrip(
        b"="
    )
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=")
    signature = base64.urlsafe_b64encode(b"fakesig").rstrip(b"=")
    return f"{header.decode()}.{body.decode()}.{signature.decode()}"


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
    async def test_expired_token_refresh_failure_raises(
        self, monkeypatch, tmp_path: Path
    ):
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

        monkeypatch.setattr("urllib.request.urlopen", _raise_timeout)
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

        monkeypatch.setattr("urllib.request.urlopen", _raise_http_error)
        auth_file = tmp_path / "auth.json"
        auth_file.write_text("{}")
        with pytest.raises(ValueError, match="Token refresh failed"):
            await refresh_access_token("fake-refresh-token", auth_path=auth_file)

    @pytest.mark.asyncio
    async def test_timeout_error_raises_value_error(self, monkeypatch, tmp_path: Path):
        def _raise_timeout(*args, **kwargs):
            raise TimeoutError("Connection timed out")

        monkeypatch.setattr("urllib.request.urlopen", _raise_timeout)
        auth_file = tmp_path / "auth.json"
        auth_file.write_text("{}")
        with pytest.raises(ValueError, match="Token refresh failed"):
            await refresh_access_token("fake-refresh-token", auth_path=auth_file)

    @pytest.mark.asyncio
    async def test_missing_access_token_raises_value_error(
        self, monkeypatch, tmp_path: Path
    ):

        class _FakeResp:
            def __init__(self):
                self._data = json.dumps({"refresh_token": "new-ref"}).encode()

            def read(self):
                return self._data

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: _FakeResp())
        auth_file = tmp_path / "auth.json"
        auth_file.write_text("{}")
        with pytest.raises(ValueError, match="missing 'access_token'"):
            await refresh_access_token("fake-refresh-token", auth_path=auth_file)


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
        from claude_bridge.__main__ import _detect_auth_mode

        mode, key = _detect_auth_mode()
        assert mode == "api_key"
        assert key == "test-key-placeholder"

    def test_empty_api_key_selects_codex_oauth(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "")
        from claude_bridge.__main__ import _detect_auth_mode

        mode, key = _detect_auth_mode()
        assert mode == "codex_oauth"
        assert key is None

    def test_missing_api_key_selects_codex_oauth(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from claude_bridge.__main__ import _detect_auth_mode

        mode, key = _detect_auth_mode()
        assert mode == "codex_oauth"
        assert key is None
