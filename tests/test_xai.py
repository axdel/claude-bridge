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
    XAIProvider,
    _iso_to_timestamp,
    _xai_token_expired,
    anthropic_to_xai,
    get_xai_bearer_token,
    read_xai_auth,
    refresh_xai_token,
)

_CLIENT_ID = "b1a00492-073a-47ea-816f-4c329264a828"
_ENTRY_KEY = f"https://auth.x.ai::{_CLIENT_ID}"

_FIXTURES = Path(__file__).parent / "fixtures" / "xai"


def _load_fixture(name: str) -> dict:
    """Load a golden xAI wire-capture fixture (see tests/fixtures/xai/README.md)."""
    return json.loads((_FIXTURES / name).read_text())


def _input_items(result: dict, item_type: str) -> list[dict]:
    """Return the translated input items of a given Responses ``type``."""
    return [item for item in result["input"] if item.get("type") == item_type]


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

    @pytest.mark.asyncio
    async def test_derives_expires_at_from_expires_in_when_new_token_opaque(
        self, monkeypatch, tmp_path: Path
    ):
        # Oracle: an opaque (non-JWT) access_token yields no exp claim, so RFC 6749
        # §5.1 expires_in seconds must drive expires_at (now + expires_in).
        auth_file = self._expired_file(tmp_path)
        before = time.time()
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda *a, **kw: _FakeTokenResp(
                {"access_token": "opaque-not-a-jwt", "expires_in": 4321}
            ),
        )
        await refresh_xai_token(_ENTRY_KEY, "refresh-original", _CLIENT_ID, auth_path=auth_file)
        persisted = json.loads(auth_file.read_text())
        assert persisted[_ENTRY_KEY]["key"] == "opaque-not-a-jwt"
        delta = _iso_to_timestamp(persisted[_ENTRY_KEY]["expires_at"]) - before
        assert 4321 - 5 <= delta <= 4321 + 5

    @pytest.mark.asyncio
    async def test_removes_stale_expires_at_when_opaque_token_and_no_expires_in(
        self, monkeypatch, tmp_path: Path
    ):
        # Oracle: with neither a decodable exp claim nor an expires_in, keeping the
        # old expires_at would wrongly gate the fresh bearer as expired — so it is
        # dropped, not preserved.
        auth_file = self._expired_file(tmp_path)  # entry starts WITH an expires_at
        assert "expires_at" in json.loads(auth_file.read_text())[_ENTRY_KEY]
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda *a, **kw: _FakeTokenResp({"access_token": "opaque-not-a-jwt"}),
        )
        await refresh_xai_token(_ENTRY_KEY, "refresh-original", _CLIENT_ID, auth_path=auth_file)
        persisted = json.loads(auth_file.read_text())
        assert persisted[_ENTRY_KEY]["key"] == "opaque-not-a-jwt"
        assert "expires_at" not in persisted[_ENTRY_KEY]

    @pytest.mark.asyncio
    async def test_replaces_non_dict_entry_with_fresh_entry(self, monkeypatch, tmp_path: Path):
        # A corrupted entry (a bare string where a dict is expected) must not crash
        # the refresh — it is replaced by a fresh entry carrying the new bearer, and
        # unrelated siblings survive.
        new_token = _make_jwt({"exp": time.time() + 3600})
        auth_file = _write_grok_auth(
            tmp_path, {_ENTRY_KEY: "corrupted-non-dict", "sibling": {"k": 1}}
        )
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
        assert persisted[_ENTRY_KEY]["refresh_token"] == "refresh-original"
        assert persisted["sibling"] == {"k": 1}

    @pytest.mark.asyncio
    async def test_creates_file_with_default_mode_when_absent(self, monkeypatch, tmp_path: Path):
        # If the auth file vanished between read and write, stat() fails; the bridge
        # must fall back to 0o600 (never a world-readable secret) and still persist.
        new_token = _make_jwt({"exp": time.time() + 3600})
        auth_file = tmp_path / ".grok" / "auth.json"
        auth_file.parent.mkdir(parents=True, exist_ok=True)  # dir exists, file does not
        assert not auth_file.exists()
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda *a, **kw: _FakeTokenResp({"access_token": new_token, "expires_in": 3600}),
        )
        await refresh_xai_token(_ENTRY_KEY, "refresh-original", _CLIENT_ID, auth_path=auth_file)
        assert auth_file.exists()
        assert stat.S_IMODE(os.stat(auth_file).st_mode) == 0o600
        assert json.loads(auth_file.read_text())[_ENTRY_KEY]["key"] == new_token


# --- import decode_jwt_exp used to keep the oracle honest (no unused import) ---
assert callable(decode_jwt_exp)


class TestRequestTranslation:
    """Anthropic Messages request -> xAI Responses request (``anthropic_to_xai``).

    Oracle discipline: every expected value derives from the golden wire captures in
    tests/fixtures/xai/ (real cli-chat-proxy bytes) or from a spec-level invariant —
    never from running the translator. The load-bearing divergence from the OpenAI
    path is proven here: xAI links a tool call to its result by ``call_id`` ALONE and
    accepts it verbatim, so the Anthropic tool id is forwarded unchanged (NO ``fc_``
    rewrite, NO synthesized ``id`` field), and ``reasoning.effort`` is never sent
    (field_effort_low.json shows it 400s).
    """

    # -- request envelope -----------------------------------------------------

    def test_envelope_core_fields(self, monkeypatch):
        """The base envelope pins model, store, stream, and the encrypted-reasoning include."""
        monkeypatch.delenv("XAI_MODEL", raising=False)
        result, _ = anthropic_to_xai({"messages": []})
        assert result["model"] == "grok-4.20"
        assert result["store"] is False
        assert result["stream"] is True
        assert result["include"] == ["reasoning.encrypted_content"]
        assert result["input"] == []

    def test_envelope_honors_configured_model(self, monkeypatch):
        """The model comes from config.xai_model(), so XAI_MODEL overrides it."""
        monkeypatch.setenv("XAI_MODEL", "grok-3-mini")
        result, _ = anthropic_to_xai({"messages": []})
        assert result["model"] == "grok-3-mini"

    def test_request_omits_reasoning_key_entirely(self):
        """xAI 400s on reasoning.effort (field_effort_low.json) — the key must be absent."""
        result, _ = anthropic_to_xai({"messages": []})
        assert "reasoning" not in result

    def test_thinking_config_warns_without_adding_reasoning_effort(self):
        """A thinking config is acknowledged in warnings but never becomes reasoning.effort."""
        result, warnings = anthropic_to_xai(
            {"messages": [], "thinking": {"type": "enabled", "budget_tokens": 1024}}
        )
        assert "reasoning" not in result
        assert any("thinking" in w.lower() for w in warnings)

    # -- system / instructions ------------------------------------------------

    def test_system_string_becomes_instructions(self):
        result, _ = anthropic_to_xai({"system": "Be terse.", "messages": []})
        assert result["instructions"] == "Be terse."

    def test_system_list_joined_into_instructions(self):
        result, _ = anthropic_to_xai(
            {
                "system": [
                    {"type": "text", "text": "Line A"},
                    {"type": "text", "text": "Line B"},
                ],
                "messages": [],
            }
        )
        assert result["instructions"] == "Line A\nLine B"

    def test_no_system_uses_default_instruction(self):
        result, _ = anthropic_to_xai({"messages": []})
        assert result["instructions"] == "You are a helpful assistant."

    # -- tools ----------------------------------------------------------------

    def test_tools_flattened_to_function_entries(self):
        """Tool shape mirrors single_tool_call.json's ``tools`` array exactly."""
        schema = {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        }
        result, _ = anthropic_to_xai(
            {
                "messages": [],
                "tools": [
                    {
                        "name": "get_weather",
                        "description": "Get the current weather for a city.",
                        "input_schema": schema,
                    }
                ],
            }
        )
        assert result["tools"] == [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get the current weather for a city.",
                "parameters": schema,
                "strict": False,
            }
        ]

    # -- tool_choice ----------------------------------------------------------

    def test_tool_choice_auto(self):
        result, _ = anthropic_to_xai({"messages": [], "tool_choice": {"type": "auto"}})
        assert result["tool_choice"] == "auto"

    def test_tool_choice_none(self):
        result, _ = anthropic_to_xai({"messages": [], "tool_choice": {"type": "none"}})
        assert result["tool_choice"] == "none"

    def test_tool_choice_any_maps_to_required(self):
        result, _ = anthropic_to_xai({"messages": [], "tool_choice": {"type": "any"}})
        assert result["tool_choice"] == "required"

    def test_tool_choice_tool_forces_named_function(self):
        result, _ = anthropic_to_xai(
            {"messages": [], "tool_choice": {"type": "tool", "name": "get_weather"}}
        )
        assert result["tool_choice"] == {"type": "function", "name": "get_weather"}

    def test_tool_choice_disable_parallel_sets_flag(self):
        result, _ = anthropic_to_xai(
            {
                "messages": [],
                "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
            }
        )
        assert result["parallel_tool_calls"] is False

    def test_tool_choice_unknown_type_omitted_with_warning(self):
        result, warnings = anthropic_to_xai({"messages": [], "tool_choice": {"type": "bogus"}})
        assert "tool_choice" not in result
        assert any("tool_choice" in w for w in warnings)

    # -- tool_use: call_id VERBATIM, no fc_ transform, no id ------------------

    def test_tool_use_forwards_anthropic_id_as_call_id_verbatim(self):
        """A ``toolu_``-prefixed id is surfaced as ``call_id`` VERBATIM — NOT rewritten to fc_."""
        result, _ = anthropic_to_xai(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_abc123",
                                "name": "get_weather",
                                "input": {"city": "Paris"},
                            }
                        ],
                    }
                ]
            }
        )
        calls = _input_items(result, "function_call")
        assert len(calls) == 1
        call = calls[0]
        # The whole point of B3: no fc_ rewrite, no synthesized item id.
        assert call["call_id"] == "toolu_abc123"
        assert call["call_id"] != "fc_abc123"
        assert "id" not in call
        assert call["name"] == "get_weather"
        assert json.loads(call["arguments"]) == {"city": "Paris"}

    def test_tool_use_roundtrips_xai_native_call_id_exactly(self):
        """xAI's own ``call-<uuid>-<idx>`` id round-trips byte-for-byte (single_tool_call.json)."""
        native_call_id = _load_fixture("single_tool_call.json")["output"][1]["call_id"]
        result, _ = anthropic_to_xai(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": native_call_id,
                                "name": "get_weather",
                                "input": {"city": "Paris"},
                            }
                        ],
                    }
                ]
            }
        )
        call = _input_items(result, "function_call")[0]
        assert call["call_id"] == native_call_id
        assert call["call_id"] == "call-bb020360-5c8a-4760-a7e6-b9cff0c2e700-0"

    def test_tool_use_arguments_are_a_json_string(self):
        """Responses ``arguments`` is a JSON STRING (e.g. '{\"city\":\"Paris\"}'), not a dict."""
        result, _ = anthropic_to_xai(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_x",
                                "name": "get_weather",
                                "input": {"city": "Paris"},
                            }
                        ],
                    }
                ]
            }
        )
        call = _input_items(result, "function_call")[0]
        assert isinstance(call["arguments"], str)
        assert call["arguments"] == '{"city": "Paris"}'

    # -- tool_result: call_id VERBATIM, string output ------------------------

    def test_tool_result_forwards_call_id_verbatim_string_output(self):
        """function_call_output keeps the id verbatim (tool_result_replay_exact.json)."""
        result, _ = anthropic_to_xai(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "call-bb020360-5c8a-4760-a7e6-b9cff0c2e700-0",
                                "content": "18°C and sunny",
                            }
                        ],
                    }
                ]
            }
        )
        outputs = _input_items(result, "function_call_output")
        assert len(outputs) == 1
        assert outputs[0]["call_id"] == "call-bb020360-5c8a-4760-a7e6-b9cff0c2e700-0"
        assert outputs[0]["output"] == "18°C and sunny"

    def test_tool_result_error_is_marked(self):
        result, _ = anthropic_to_xai(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_x",
                                "content": "boom",
                                "is_error": True,
                            }
                        ],
                    }
                ]
            }
        )
        assert _input_items(result, "function_call_output")[0]["output"] == "[Error] boom"

    def test_tool_result_text_block_list_flattened(self):
        result, _ = anthropic_to_xai(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_x",
                                "content": [
                                    {"type": "text", "text": "line1"},
                                    {"type": "text", "text": "line2"},
                                ],
                            }
                        ],
                    }
                ]
            }
        )
        assert _input_items(result, "function_call_output")[0]["output"] == "line1\nline2"

    def test_tool_result_media_redacted_without_leaking_base64(self):
        """A media block in a tool_result degrades to a redacted string — base64 NEVER leaks."""
        result, warnings = anthropic_to_xai(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_x",
                                "content": [
                                    {"type": "text", "text": "see:"},
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": "image/png",
                                            "data": "SECRETBASE64PAYLOAD==",
                                        },
                                    },
                                ],
                            }
                        ],
                    }
                ]
            }
        )
        output = _input_items(result, "function_call_output")[0]["output"]
        assert "SECRETBASE64PAYLOAD" not in output
        assert "image" in output
        assert any("media" in w.lower() for w in warnings)

    # -- message text ---------------------------------------------------------

    def test_user_text_becomes_input_text(self):
        result, _ = anthropic_to_xai(
            {"messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]}
        )
        assert result["input"] == [
            {"role": "user", "content": [{"type": "input_text", "text": "hello"}]}
        ]

    def test_assistant_text_becomes_output_text(self):
        result, _ = anthropic_to_xai(
            {
                "messages": [
                    {"role": "assistant", "content": [{"type": "text", "text": "hi there"}]}
                ]
            }
        )
        assert result["input"] == [
            {"role": "assistant", "content": [{"type": "output_text", "text": "hi there"}]}
        ]

    def test_string_content_shorthand_becomes_input_text(self):
        result, _ = anthropic_to_xai({"messages": [{"role": "user", "content": "plain"}]})
        assert result["input"] == [
            {"role": "user", "content": [{"type": "input_text", "text": "plain"}]}
        ]

    # -- thinking-block reasoning modes (module-attr patched) -----------------

    def test_thinking_block_passthrough_wraps_text(self):
        """Default reasoning mode keeps a thinking block as bracketed text (→ output_text)."""
        result, _ = anthropic_to_xai(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [{"type": "thinking", "thinking": "pondering"}],
                    }
                ]
            }
        )
        block = result["input"][0]["content"][0]
        assert block["type"] == "output_text"
        assert "[thinking]" in block["text"]
        assert "pondering" in block["text"]

    def test_thinking_block_dropped_in_drop_mode(self, monkeypatch):
        """reasoning_mode=drop empties the thinking block — the text never survives."""
        import claude_bridge.providers.xai as xai

        monkeypatch.setattr(xai, "_XAI_REASONING_MODE", "drop")
        result, warnings = xai.anthropic_to_xai(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [{"type": "thinking", "thinking": "secret-cot"}],
                    }
                ]
            }
        )
        assert "secret-cot" not in json.dumps(result)
        assert result["input"][0]["content"][0]["text"] == ""
        assert any("drop" in w.lower() for w in warnings)

    # -- unsupported blocks (B3 is text+tools only) --------------------------

    def test_unsupported_message_block_degrades_without_leaking_content(self):
        """An image block at message level (B3 is text-only) → redacted placeholder, no base64."""
        result, warnings = anthropic_to_xai(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": "SECRETPNGPAYLOAD==",
                                },
                            }
                        ],
                    }
                ]
            }
        )
        block = result["input"][0]["content"][0]
        assert block["type"] == "input_text"
        assert block["text"] == "[unsupported content block: image]"
        assert "SECRETPNGPAYLOAD" not in json.dumps(result)
        assert any("image" in w for w in warnings)

    def test_unsupported_block_type_is_truncated_when_oversized(self):
        """A hostile oversized block type is capped at 64 chars (CWE-117 / log-flood defense)."""
        long_type = "z" * 200
        result, _ = anthropic_to_xai(
            {"messages": [{"role": "user", "content": [{"type": long_type}]}]}
        )
        placeholder = result["input"][0]["content"][0]["text"]
        assert placeholder.endswith("...]")
        assert long_type not in placeholder  # the full 200-char type is never echoed
        assert "z" * 64 in placeholder  # the capped 64-char prefix survives

    # -- stripped keys / cache_control ---------------------------------------

    def test_output_config_stripped_with_warning(self):
        _, warnings = anthropic_to_xai({"messages": [], "output_config": {"x": 1}})
        assert any("output_config" in w for w in warnings)

    def test_thinking_config_drop_mode_warns_stripped(self, monkeypatch):
        """In drop mode a top-level thinking config is reported as stripped."""
        import claude_bridge.providers.xai as xai

        monkeypatch.setattr(xai, "_XAI_REASONING_MODE", "drop")
        _, warnings = xai.anthropic_to_xai(
            {"messages": [], "thinking": {"type": "enabled", "budget_tokens": 5}}
        )
        assert any("drop" in w.lower() for w in warnings)

    def test_cache_control_on_system_warned(self):
        _, warnings = anthropic_to_xai(
            {
                "system": [{"type": "text", "text": "s", "cache_control": {"type": "ephemeral"}}],
                "messages": [],
            }
        )
        assert any("cache_control" in w for w in warnings)

    def test_cache_control_on_tool_warned(self):
        _, warnings = anthropic_to_xai(
            {
                "messages": [],
                "tools": [
                    {
                        "name": "t",
                        "input_schema": {},
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        )
        assert any("cache_control" in w for w in warnings)

    def test_cache_control_hints_warned(self):
        _, warnings = anthropic_to_xai(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "hi",
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    }
                ]
            }
        )
        assert any("cache_control" in w for w in warnings)

    # -- provider delegation --------------------------------------------------

    def test_provider_translate_request_delegates(self, monkeypatch):
        """XAIProvider.translate_request routes through anthropic_to_xai."""
        monkeypatch.delenv("XAI_MODEL", raising=False)
        result, _ = XAIProvider().translate_request(
            {"messages": [{"role": "user", "content": "hi"}]}
        )
        assert result["model"] == "grok-4.20"
        assert result["input"] == [
            {"role": "user", "content": [{"type": "input_text", "text": "hi"}]}
        ]
