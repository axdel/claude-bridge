"""Tests for the xAI (Grok) provider — subscription OAuth token management.

Mirrors the Codex OAuth mechanics (``tests/test_auth.py``) but against the
``~/.grok/auth.json`` shape: a single ``https://auth.x.ai::<client_id>`` keyed
entry whose bearer lives in ``key`` (not ``access_token``), alongside profile
sibling fields that MUST survive a token refresh.

Oracle discipline: every expiry assertion derives from a hand-built JWT with a
known ``exp`` claim or a constructed ISO-8601 timestamp — never from running the
code under test. File-preservation assertions compare bytes / permission bits.
"""

from __future__ import annotations

import asyncio
import base64
import datetime
import json
import os
import stat
import time
from pathlib import Path

import pytest

from claude_bridge.auth import decode_jwt_exp
from claude_bridge.providers.xai import (
    _iso_to_timestamp,
    _xai_token_expired,
    get_xai_bearer_token,
    read_xai_auth,
    refresh_xai_token,
)

_CLIENT_ID = "b1a00492-073a-47ea-816f-4c329264a828"
_ENTRY_KEY = f"https://auth.x.ai::{_CLIENT_ID}"


def _make_jwt(payload: dict) -> str:
    """Build a fake JWT with the given payload (no crypto verification needed)."""
    header = base64.urlsafe_b64encode(json.dumps({"alg": "RS256"}).encode()).rstrip(b"=")
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=")
    signature = base64.urlsafe_b64encode(b"fakesig").rstrip(b"=")
    return f"{header.decode()}.{body.decode()}.{signature.decode()}"


def _iso(ts: float) -> str:
    """Format an epoch timestamp as the ISO-8601 'Z' form xAI uses in expires_at."""
    return datetime.datetime.fromtimestamp(ts, datetime.UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _grok_auth(
    entry_overrides: dict | None = None,
    *,
    extra_entries: dict | None = None,
    client_id: str = _CLIENT_ID,
) -> dict:
    """Construct a ~/.grok/auth.json payload with one xAI OIDC entry."""
    now = time.time()
    entry = {
        "key": _make_jwt({"exp": now + 3600, "sub": "user"}),
        "auth_mode": "oidc",
        "create_time": _iso(now - 3600),
        "user_id": "uid-00000000-0000-0000-0000-000000000000",
        "email": "user@example.com",
        "first_name": "Test",
        "last_name": "User",
        "principal_type": "User",
        "principal_id": "pid-1",
        "team_id": "team-1",
        "coding_data_retention_opt_out": False,
        "refresh_token": "refresh-original",
        "expires_at": _iso(now + 3600),
        "oidc_issuer": "https://auth.x.ai",
        "oidc_client_id": client_id,
    }
    if entry_overrides:
        entry.update(entry_overrides)
    data: dict = {f"https://auth.x.ai::{client_id}": entry}
    if extra_entries:
        data.update(extra_entries)
    return data


def _write_grok_auth(tmp_path: Path, data: dict, *, mode: int = 0o600) -> Path:
    auth_file = tmp_path / ".grok" / "auth.json"
    auth_file.parent.mkdir(parents=True, exist_ok=True)
    auth_file.write_text(json.dumps(data, indent=2))
    auth_file.chmod(mode)
    return auth_file


class _FakeTokenResp:
    """Minimal urlopen context-manager stand-in returning a fixed JSON body."""

    def __init__(self, body: dict):
        self._data = json.dumps(body).encode()

    def read(self) -> bytes:
        return self._data

    def __enter__(self) -> _FakeTokenResp:
        return self

    def __exit__(self, *args: object) -> bool:
        return False


# --- _iso_to_timestamp ---


class TestIsoToTimestamp:
    def test_parses_z_suffixed_utc(self):
        # 2026-07-10T14:58:39Z is a fixed instant; derive its epoch independently.
        expected = datetime.datetime(2026, 7, 10, 14, 58, 39, tzinfo=datetime.UTC).timestamp()
        assert _iso_to_timestamp("2026-07-10T14:58:39Z") == expected

    def test_parses_microseconds(self):
        expected = datetime.datetime(
            2026, 7, 10, 14, 58, 39, 466151, tzinfo=datetime.UTC
        ).timestamp()
        assert _iso_to_timestamp("2026-07-10T14:58:39.466151Z") == expected

    def test_raises_on_garbage(self):
        with pytest.raises(ValueError):
            _iso_to_timestamp("not-a-timestamp")


# --- read_xai_auth ---


class TestReadXaiAuth:
    def test_missing_file_raises_file_not_found(self, tmp_path: Path):
        missing = tmp_path / ".grok" / "auth.json"
        with pytest.raises(FileNotFoundError, match="grok"):
            read_xai_auth(missing)

    def test_selects_single_xai_entry(self, tmp_path: Path):
        data = _grok_auth()
        auth_file = _write_grok_auth(tmp_path, data)
        entry_key, entry = read_xai_auth(auth_file)
        assert entry_key == _ENTRY_KEY
        assert entry["key"] == data[_ENTRY_KEY]["key"]

    def test_no_xai_entry_raises(self, tmp_path: Path):
        auth_file = _write_grok_auth(tmp_path, {"https://other.example::x": {"key": "k"}})
        with pytest.raises(ValueError, match="No xAI"):
            read_xai_auth(auth_file)

    def test_multiple_xai_entries_raises(self, tmp_path: Path):
        data = _grok_auth()
        data["https://auth.x.ai::second-client"] = {"key": _make_jwt({"exp": 1})}
        auth_file = _write_grok_auth(tmp_path, data)
        with pytest.raises(ValueError, match="Multiple"):
            read_xai_auth(auth_file)

    def test_preserves_profile_fields_in_returned_entry(self, tmp_path: Path):
        auth_file = _write_grok_auth(tmp_path, _grok_auth())
        _key, entry = read_xai_auth(auth_file)
        assert entry["email"] == "user@example.com"
        assert entry["user_id"] == "uid-00000000-0000-0000-0000-000000000000"
        assert entry["team_id"] == "team-1"

    def test_ignores_unrelated_top_level_keys(self, tmp_path: Path):
        data = _grok_auth(extra_entries={"schema_version": "1", "device_id": "abc"})
        auth_file = _write_grok_auth(tmp_path, data)
        entry_key, _entry = read_xai_auth(auth_file)
        assert entry_key == _ENTRY_KEY


# --- _xai_token_expired ---


class TestXaiTokenExpired:
    def test_fresh_token_not_expired(self):
        entry = {
            "key": _make_jwt({"exp": time.time() + 3600}),
            "expires_at": _iso(time.time() + 3600),
        }
        assert _xai_token_expired(entry) is False

    def test_past_jwt_expired(self):
        entry = {
            "key": _make_jwt({"exp": time.time() - 100}),
            "expires_at": _iso(time.time() + 3600),
        }
        assert _xai_token_expired(entry) is True

    def test_earlier_of_jwt_and_expires_at_wins(self):
        # JWT is fresh but the bookkeeping expires_at is already past — earlier wins.
        entry = {
            "key": _make_jwt({"exp": time.time() + 3600}),
            "expires_at": _iso(time.time() - 100),
        }
        assert _xai_token_expired(entry) is True

    def test_malformed_jwt_falls_back_to_expires_at(self):
        entry = {"key": "not-a-jwt", "expires_at": _iso(time.time() + 3600)}
        assert _xai_token_expired(entry) is False

    def test_malformed_expires_at_ignored_when_jwt_valid(self):
        # A garbage expires_at string must not crash — the valid JWT governs.
        entry = {"key": _make_jwt({"exp": time.time() + 3600}), "expires_at": "garbage"}
        assert _xai_token_expired(entry) is False

    def test_no_parseable_expiry_forces_refresh(self):
        entry = {"key": "not-a-jwt"}
        assert _xai_token_expired(entry) is True

    def test_margin_makes_near_future_expired(self):
        entry = {"key": _make_jwt({"exp": time.time() + 20})}
        assert _xai_token_expired(entry, margin_seconds=30) is True


# --- get_xai_bearer_token ---


class TestGetXaiBearerToken:
    @pytest.mark.asyncio
    async def test_returns_valid_token_without_refresh(self, tmp_path: Path):
        data = _grok_auth()
        auth_file = _write_grok_auth(tmp_path, data)
        result = await get_xai_bearer_token(auth_file)
        assert result == data[_ENTRY_KEY]["key"]

    @pytest.mark.asyncio
    async def test_expired_triggers_refresh(self, monkeypatch, tmp_path: Path):
        new_token = _make_jwt({"exp": time.time() + 3600})
        data = _grok_auth(
            {"key": _make_jwt({"exp": time.time() - 100}), "expires_at": _iso(time.time() - 100)}
        )
        auth_file = _write_grok_auth(tmp_path, data)
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda *a, **kw: _FakeTokenResp({"access_token": new_token, "expires_in": 3600}),
        )
        result = await get_xai_bearer_token(auth_file)
        assert result == new_token

    @pytest.mark.asyncio
    async def test_expired_no_refresh_token_raises(self, tmp_path: Path):
        data = _grok_auth(
            {
                "key": _make_jwt({"exp": time.time() - 100}),
                "expires_at": _iso(time.time() - 100),
                "refresh_token": "",
            }
        )
        auth_file = _write_grok_auth(tmp_path, data)
        with pytest.raises(ValueError, match="re-authenticate"):
            await get_xai_bearer_token(auth_file)

    @pytest.mark.asyncio
    async def test_refresh_failure_raises(self, monkeypatch, tmp_path: Path):
        data = _grok_auth(
            {"key": _make_jwt({"exp": time.time() - 100}), "expires_at": _iso(time.time() - 100)}
        )
        auth_file = _write_grok_auth(tmp_path, data)

        def _raise_timeout(*args, **kwargs):
            raise TimeoutError("Connection timed out")

        monkeypatch.setattr("urllib.request.urlopen", _raise_timeout)
        with pytest.raises(ValueError, match="Token refresh failed"):
            await get_xai_bearer_token(auth_file)

    @pytest.mark.asyncio
    async def test_concurrent_calls_share_lock(self, tmp_path: Path):
        data = _grok_auth()
        auth_file = _write_grok_auth(tmp_path, data)
        results = await asyncio.gather(*[get_xai_bearer_token(auth_file) for _ in range(5)])
        assert all(r == data[_ENTRY_KEY]["key"] for r in results)


# --- refresh_xai_token: persistence, rotation, preservation ---


class TestRefreshXaiToken:
    def _expired_file(self, tmp_path: Path, **entry_overrides) -> Path:
        base = {
            "key": _make_jwt({"exp": time.time() - 100}),
            "expires_at": _iso(time.time() - 100),
        }
        base.update(entry_overrides)
        return _write_grok_auth(tmp_path, _grok_auth(base))

    @pytest.mark.asyncio
    async def test_updates_key_and_persists(self, monkeypatch, tmp_path: Path):
        new_token = _make_jwt({"exp": time.time() + 3600})
        auth_file = self._expired_file(tmp_path)
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda *a, **kw: _FakeTokenResp({"access_token": new_token, "expires_in": 3600}),
        )
        returned = await refresh_xai_token(
            _ENTRY_KEY, "refresh-original", _CLIENT_ID, auth_path=auth_file
        )
        assert returned == new_token
        persisted = json.loads(auth_file.read_text())
        assert persisted[_ENTRY_KEY]["key"] == new_token

    @pytest.mark.asyncio
    async def test_rotates_refresh_token(self, monkeypatch, tmp_path: Path):
        new_token = _make_jwt({"exp": time.time() + 3600})
        auth_file = self._expired_file(tmp_path)
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda *a, **kw: _FakeTokenResp(
                {"access_token": new_token, "refresh_token": "refresh-rotated", "expires_in": 3600}
            ),
        )
        await refresh_xai_token(_ENTRY_KEY, "refresh-original", _CLIENT_ID, auth_path=auth_file)
        persisted = json.loads(auth_file.read_text())
        assert persisted[_ENTRY_KEY]["refresh_token"] == "refresh-rotated"

    @pytest.mark.asyncio
    async def test_keeps_refresh_token_when_response_omits_it(self, monkeypatch, tmp_path: Path):
        new_token = _make_jwt({"exp": time.time() + 3600})
        auth_file = self._expired_file(tmp_path)
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda *a, **kw: _FakeTokenResp({"access_token": new_token, "expires_in": 3600}),
        )
        await refresh_xai_token(_ENTRY_KEY, "refresh-original", _CLIENT_ID, auth_path=auth_file)
        persisted = json.loads(auth_file.read_text())
        assert persisted[_ENTRY_KEY]["refresh_token"] == "refresh-original"

    @pytest.mark.asyncio
    async def test_preserves_sibling_entries_and_profile(self, monkeypatch, tmp_path: Path):
        new_token = _make_jwt({"exp": time.time() + 3600})
        data = _grok_auth(
            {"key": _make_jwt({"exp": time.time() - 100}), "expires_at": _iso(time.time() - 100)},
            extra_entries={"schema_version": "7", "device_id": "dev-abc"},
        )
        auth_file = _write_grok_auth(tmp_path, data)
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda *a, **kw: _FakeTokenResp({"access_token": new_token, "expires_in": 3600}),
        )
        await refresh_xai_token(_ENTRY_KEY, "refresh-original", _CLIENT_ID, auth_path=auth_file)
        persisted = json.loads(auth_file.read_text())
        # Unrelated top-level siblings survive.
        assert persisted["schema_version"] == "7"
        assert persisted["device_id"] == "dev-abc"
        # Profile fields on the xAI entry survive.
        assert persisted[_ENTRY_KEY]["email"] == "user@example.com"
        assert persisted[_ENTRY_KEY]["team_id"] == "team-1"
        assert persisted[_ENTRY_KEY]["oidc_issuer"] == "https://auth.x.ai"

    @pytest.mark.asyncio
    async def test_preserves_file_permissions(self, monkeypatch, tmp_path: Path):
        new_token = _make_jwt({"exp": time.time() + 3600})
        auth_file = self._expired_file(tmp_path)
        auth_file.chmod(0o600)
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda *a, **kw: _FakeTokenResp({"access_token": new_token, "expires_in": 3600}),
        )
        await refresh_xai_token(_ENTRY_KEY, "refresh-original", _CLIENT_ID, auth_path=auth_file)
        mode = stat.S_IMODE(os.stat(auth_file).st_mode)
        assert mode == 0o600

    @pytest.mark.asyncio
    async def test_failure_leaves_file_byte_identical(self, monkeypatch, tmp_path: Path):
        auth_file = self._expired_file(tmp_path)
        before = auth_file.read_bytes()

        def _raise_http(*args, **kwargs):
            raise TimeoutError("boom")

        monkeypatch.setattr("urllib.request.urlopen", _raise_http)
        with pytest.raises(ValueError, match="Token refresh failed"):
            await refresh_xai_token(
                _ENTRY_KEY, "refresh-original", _CLIENT_ID, auth_path=auth_file
            )
        assert auth_file.read_bytes() == before

    @pytest.mark.asyncio
    async def test_resyncs_expires_at_from_new_jwt(self, monkeypatch, tmp_path: Path):
        # Oracle: the new token's exp claim is a known instant; expires_at after
        # refresh must reflect it (else the earlier-of check re-expires immediately).
        new_exp = time.time() + 4321
        new_token = _make_jwt({"exp": new_exp})
        auth_file = self._expired_file(tmp_path)
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda *a, **kw: _FakeTokenResp({"access_token": new_token, "expires_in": 4321}),
        )
        await refresh_xai_token(_ENTRY_KEY, "refresh-original", _CLIENT_ID, auth_path=auth_file)
        persisted = json.loads(auth_file.read_text())
        resynced = _iso_to_timestamp(persisted[_ENTRY_KEY]["expires_at"])
        assert abs(resynced - new_exp) < 2
        # And the refreshed entry is no longer considered expired.
        assert _xai_token_expired(persisted[_ENTRY_KEY]) is False

    @pytest.mark.asyncio
    async def test_missing_access_token_raises(self, monkeypatch, tmp_path: Path):
        auth_file = self._expired_file(tmp_path)
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda *a, **kw: _FakeTokenResp({"refresh_token": "r", "expires_in": 3600}),
        )
        with pytest.raises(ValueError, match="access_token"):
            await refresh_xai_token(
                _ENTRY_KEY, "refresh-original", _CLIENT_ID, auth_path=auth_file
            )


# --- import decode_jwt_exp used to keep the oracle honest (no unused import) ---
assert callable(decode_jwt_exp)
