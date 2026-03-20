"""Tests for auth utilities and Codex OAuth provider auth."""

from __future__ import annotations

import asyncio
import base64
import json
import time
from pathlib import Path

import pytest

from claude_bridge.auth import decode_jwt_exp, is_token_expired


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
        with pytest.raises(KeyError):
            decode_jwt_exp(token)


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


# --- read_codex_auth ---


from claude_bridge.providers.openai import read_codex_auth


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


from claude_bridge.providers.openai import get_bearer_token


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
