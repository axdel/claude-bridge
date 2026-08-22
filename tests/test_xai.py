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
from claude_bridge.content import parse_media_source
from claude_bridge.provider import PROVIDERS, ProviderCapabilities
from claude_bridge.providers.xai import (
    _MAX_SSE_BUFFER,
    _REASONING_CACHE_MAX,
    _XAI_CAPABILITIES,
    _XAI_CLIENT_IDENTIFIER,
    _XAI_ENDPOINT,
    XAIProvider,
    _associate_reasoning_with_calls,
    _iso_to_timestamp,
    _safe_document_filename,
    _tool_result_parts,
    _translate_document_block,
    _translate_image_block,
    _validated_bearer,
    _xai_token_expired,
    anthropic_to_xai,
    get_xai_bearer_token,
    read_xai_auth,
    refresh_xai_token,
    translate_xai_sse_event,
    xai_to_anthropic,
)

_CLIENT_ID = "b1a00492-073a-47ea-816f-4c329264a828"
_ENTRY_KEY = f"https://auth.x.ai::{_CLIENT_ID}"

_FIXTURES = Path(__file__).parent / "fixtures" / "xai"

# Media-capable capability objects for exercising the forward-vs-degrade seam directly.
# xAI's PROVIDER instance declares its own (conservative until B8); these drive the
# translation functions in isolation, mirroring how the openai media tests work.
_MEDIA_CAPS = ProviderCapabilities(
    stream_request_mode="body_parameter",
    sync_response_mode="sse",
    input_modalities=frozenset({"text", "image", "document"}),
)
_ARRAY_TOOL_CAPS = ProviderCapabilities(
    stream_request_mode="body_parameter",
    sync_response_mode="sse",
    input_modalities=frozenset({"text", "image", "document"}),
    supports_tool_output_content_parts=True,
)


def _load_fixture(name: str) -> dict:
    """Load a golden xAI wire-capture fixture (see tests/fixtures/xai/README.md)."""
    return json.loads((_FIXTURES / name).read_text())


# The only value a committed fixture may carry in an ``encrypted_content`` field. A real
# provider-issued reasoning continuation is opaque, replayable state that the invariant
# (xai.py: "In-memory only ... never persisted") forbids checking into the repo.
_SYNTHETIC_ENCRYPTED_SENTINEL = "SYNTHETIC-TEST-REASONING-BLOB-not-a-real-grok-continuation"


def _iter_encrypted_content(node: object) -> list[str]:
    """Recursively collect every ``encrypted_content`` value in a decoded JSON tree."""
    found: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "encrypted_content" and isinstance(value, str):
                found.append(value)
            else:
                found.extend(_iter_encrypted_content(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(_iter_encrypted_content(item))
    return found


def test_no_fixture_persists_a_real_encrypted_reasoning_blob():
    """Enforce the in-memory-only reasoning invariant at the fixture layer: no committed
    fixture may hold a real captured ``encrypted_content`` blob — only the synthetic
    sentinel. Guards against a future re-capture silently persisting replayable state."""
    offenders: dict[str, list[str]] = {}
    for fixture in _FIXTURES.glob("*.json"):
        blobs = _iter_encrypted_content(json.loads(fixture.read_text()))
        bad = [b for b in blobs if b != _SYNTHETIC_ENCRYPTED_SENTINEL]
        if bad:
            offenders[fixture.name] = bad
    assert not offenders, (
        f"Fixtures persist real encrypted_content (must be the synthetic sentinel): "
        f"{sorted(offenders)}"
    )


def _input_items(result: dict, item_type: str) -> list[dict]:
    """Return the translated input items of a given Responses ``type``."""
    return [item for item in result["input"] if item.get("type") == item_type]


def _user_content_parts(result: dict) -> list[dict]:
    """Return the nested content parts of the first user message input item."""
    for item in result["input"]:
        if item.get("role") == "user":
            return item["content"]
    return []


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

    @pytest.mark.asyncio
    async def test_poisoned_issuer_ignored_on_refresh(self, monkeypatch, tmp_path: Path):
        # Confused-deputy pin: the refresh POST carries the refresh_token, so a
        # poisoned auth.json must NOT choose its destination host. Oracle: the
        # endpoint is pinned to _XAI_ISSUER + the RFC 6749 token path —
        # https://auth.x.ai/oauth2/token — regardless of the attacker-controlled
        # oidc_issuer field. Fails against issuer-derived-from-file code (which
        # would POST to http://attacker.example/oauth2/token).
        new_token = _make_jwt({"exp": time.time() + 3600})
        data = _grok_auth(
            {
                "key": _make_jwt({"exp": time.time() - 100}),
                "expires_at": _iso(time.time() - 100),
                "oidc_issuer": "http://attacker.example",
            }
        )
        auth_file = _write_grok_auth(tmp_path, data)
        captured: dict = {}

        def _capture(req, *a, **kw):
            captured["url"] = req.full_url
            return _FakeTokenResp({"access_token": new_token, "expires_in": 3600})

        monkeypatch.setattr("urllib.request.urlopen", _capture)
        result = await get_xai_bearer_token(auth_file)
        assert result == new_token
        assert captured["url"] == "https://auth.x.ai/oauth2/token"
        assert "attacker.example" not in captured["url"]

    @pytest.mark.asyncio
    async def test_control_char_bearer_rejected_without_leaking_secret(self, tmp_path: Path):
        # A fresh (valid-exp) but CR/LF-poisoned bearer must be rejected BEFORE it
        # reaches an Authorization header — else http.client echoes the secret into
        # an "Invalid header value" ValueError (CWE-532). Oracle: RFC 7235 token68
        # forbids control chars, and the raised message must NOT contain the token.
        # Fails against code that skips validation (it would return the poisoned token).
        valid = _make_jwt({"exp": time.time() + 3600})
        poisoned = valid + "\r\nX-Injected: evil"
        data = _grok_auth({"key": poisoned, "expires_at": _iso(time.time() + 3600)})
        auth_file = _write_grok_auth(tmp_path, data)
        with pytest.raises(ValueError, match="malformed") as excinfo:
            await get_xai_bearer_token(auth_file)
        assert poisoned not in str(excinfo.value)
        assert "evil" not in str(excinfo.value)


# --- _validated_bearer: header-safety gate for the outbound bearer ---


class TestValidatedBearer:
    """_validated_bearer — the header-safety gate for the outbound bearer.

    Oracle: RFC 7235 defines a bearer credential as ``token68`` — printable ASCII
    with no control characters. Every expected verdict derives from that grammar,
    never from running the validator. The security invariant: a rejection message
    never contains the token value (CWE-532).
    """

    def test_clean_jwt_passes_through_unchanged(self):
        token = _make_jwt({"exp": time.time() + 3600})
        assert _validated_bearer(token) == token

    def test_empty_token_rejected(self):
        with pytest.raises(ValueError, match="malformed"):
            _validated_bearer("")

    def test_carriage_return_rejected(self):
        with pytest.raises(ValueError, match="malformed"):
            _validated_bearer("Bearer-good\rX-Injected: evil")

    def test_newline_rejected(self):
        with pytest.raises(ValueError, match="malformed"):
            _validated_bearer("Bearer-good\nX-Injected: evil")

    def test_tab_rejected(self):
        with pytest.raises(ValueError, match="malformed"):
            _validated_bearer("Bearer-good\tinjected")

    def test_non_ascii_rejected(self):
        with pytest.raises(ValueError, match="malformed"):
            _validated_bearer("tökén-with-unicode")

    def test_message_never_contains_the_token_value(self):
        secret = "SUPER-SECRET-BEARER\r\ninjected"
        with pytest.raises(ValueError) as excinfo:
            _validated_bearer(secret)
        assert "SUPER-SECRET-BEARER" not in str(excinfo.value)


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
    async def test_refresh_tightens_permissions_to_owner_only(self, monkeypatch, tmp_path: Path):
        # The rotated secret is ALWAYS rewritten owner-only (0600, D-XAI-003). Even when the
        # prior file was world/group-readable, the refresh must tighten — never carry broad
        # bits forward (0644 in must become 0600 out). Fails against mode-preserving code.
        new_token = _make_jwt({"exp": time.time() + 3600})
        auth_file = self._expired_file(tmp_path)
        auth_file.chmod(0o644)  # start broad — a world-readable secret
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
        # If the auth file does not yet exist, the refresh still persists it owner-only
        # (0600, never a world-readable secret) via the freshly-created 0600 temp descriptor.
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

    @pytest.mark.asyncio
    async def test_non_https_issuer_refused_without_posting(self, monkeypatch, tmp_path: Path):
        # Standing defense-in-depth guard (CWE-918): refuse to POST the refresh_token
        # to a non-HTTPS endpoint even if a caller passes a downgraded issuer. Oracle:
        # the security requirement forbids sending the secret over http. The secret
        # must never leave the process — urlopen is never reached, file byte-identical.
        auth_file = self._expired_file(tmp_path)
        before = auth_file.read_bytes()
        calls: list[int] = []
        monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: calls.append(1))
        with pytest.raises(ValueError, match="untrusted"):
            await refresh_xai_token(
                _ENTRY_KEY,
                "refresh-original",
                _CLIENT_ID,
                issuer="http://auth.x.ai",
                auth_path=auth_file,
            )
        assert calls == []  # the refresh_token never left the process
        assert auth_file.read_bytes() == before

    @pytest.mark.asyncio
    async def test_cross_host_https_issuer_refused_without_posting(
        self, monkeypatch, tmp_path: Path
    ):
        # HTTPS is necessary but not sufficient: the host must be the pinned xAI
        # issuer host. A cross-origin https issuer (SSRF pivot to an internal service)
        # is refused before the POST. Oracle: only _TRUSTED_ISSUER_HOST is permitted.
        auth_file = self._expired_file(tmp_path)
        before = auth_file.read_bytes()
        calls: list[int] = []
        monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: calls.append(1))
        with pytest.raises(ValueError, match="untrusted"):
            await refresh_xai_token(
                _ENTRY_KEY,
                "refresh-original",
                _CLIENT_ID,
                issuer="https://evil.example",
                auth_path=auth_file,
            )
        assert calls == []
        assert auth_file.read_bytes() == before


# --- import decode_jwt_exp used to keep the oracle honest (no unused import) ---
assert callable(decode_jwt_exp)


class TestRequestTranslation:
    """Anthropic Messages request -> xAI Responses request (``anthropic_to_xai``).

    Oracle discipline: every expected value derives from the golden wire captures in
    tests/fixtures/xai/ (real cli-chat-proxy bytes) or from a spec-level invariant —
    never from running the translator. The load-bearing divergence from the OpenAI
    path is proven here: xAI links a tool call to its result by ``call_id`` ALONE and
    accepts it verbatim, so the Anthropic tool id is forwarded unchanged (NO ``fc_``
    rewrite, NO synthesized ``id`` field), and ``reasoning.effort`` is sent only to
    models that accept it (grok-4.6+; grok-4.20 and earlier 400 — field_effort_low.json).
    """

    # -- request envelope -----------------------------------------------------

    def test_envelope_core_fields(self, monkeypatch):
        """The base envelope pins model, store, stream, and the encrypted-reasoning include."""
        monkeypatch.delenv("XAI_MODEL", raising=False)
        result, _ = anthropic_to_xai({"messages": []})
        assert result["model"] == "grok-4.6"
        assert result["store"] is False
        assert result["stream"] is True
        assert result["include"] == ["reasoning.encrypted_content"]
        assert result["input"] == []

    def test_envelope_honors_configured_model(self, monkeypatch):
        """The model comes from config.xai_model(), so XAI_MODEL overrides it."""
        monkeypatch.setenv("XAI_MODEL", "grok-3-mini")
        result, _ = anthropic_to_xai({"messages": []})
        assert result["model"] == "grok-3-mini"

    def test_grok46_default_sends_reasoning_effort_low(self, monkeypatch):
        """grok-4.6 accepts reasoning.effort (U-EFFORT: omit/low/medium/high all HTTP 200),
        so the default request carries the config effort ``low`` (63 vs 146 reasoning tokens
        at omit — a latency choice, not native-high parity)."""
        monkeypatch.delenv("XAI_MODEL", raising=False)
        monkeypatch.delenv("XAI_REASONING_EFFORT", raising=False)
        result, _ = anthropic_to_xai({"messages": []})
        assert result["reasoning"] == {"effort": "low"}

    def test_gated_older_model_omits_reasoning_effort(self, monkeypatch):
        """grok-4.20 (a pre-4.6 model) 400s on reasoning.effort (field_effort_low.json), so the
        version gate omits the key. Decimal compare, not tuple: 4.20 == 4.2 < 4.6."""
        monkeypatch.setenv("XAI_MODEL", "grok-4.20")
        result, _ = anthropic_to_xai({"messages": []})
        assert "reasoning" not in result

    def test_lower_boundary_model_omits_reasoning_effort(self, monkeypatch):
        """A model just below the 4.6 floor is gated out — pins the >= 4.6 threshold."""
        monkeypatch.setenv("XAI_MODEL", "grok-4.5")
        result, _ = anthropic_to_xai({"messages": []})
        assert "reasoning" not in result

    def test_grok_build_alias_sends_reasoning_effort(self, monkeypatch):
        """grok-build (the rolling latest-coding alias, no parseable version) is assumed modern
        and carries reasoning.effort — only a model that parses below 4.6 is gated out, so an
        XAI_MODEL override cannot silently lose effort on a supported model."""
        monkeypatch.setenv("XAI_MODEL", "grok-build")
        monkeypatch.delenv("XAI_REASONING_EFFORT", raising=False)
        result, _ = anthropic_to_xai({"messages": []})
        assert result["reasoning"] == {"effort": "low"}

    def test_reasoning_effort_honors_env_override(self, monkeypatch):
        """XAI_REASONING_EFFORT overrides the default effort on an accepting model."""
        monkeypatch.delenv("XAI_MODEL", raising=False)
        monkeypatch.setenv("XAI_REASONING_EFFORT", "medium")
        result, _ = anthropic_to_xai({"messages": []})
        assert result["reasoning"] == {"effort": "medium"}

    def test_thinking_config_does_not_drive_reasoning_effort(self, monkeypatch):
        """A thinking config is acknowledged in warnings, but reasoning.effort comes from config,
        never from the thinking budget (budget_tokens never becomes the effort value)."""
        monkeypatch.delenv("XAI_MODEL", raising=False)
        monkeypatch.delenv("XAI_REASONING_EFFORT", raising=False)
        result, warnings = anthropic_to_xai(
            {"messages": [], "thinking": {"type": "enabled", "budget_tokens": 1024}}
        )
        assert result["reasoning"] == {"effort": "low"}
        assert any("thinking" in w.lower() for w in warnings)

    # -- max output tokens ----------------------------------------------------

    def test_max_tokens_maps_to_max_output_tokens(self):
        """Anthropic max_tokens becomes Responses max_output_tokens: grok-4.6 has no fixed text
        cap, so forwarding Claude's limit keeps slow turns bounded. The Anthropic key name
        does not survive."""
        result, _ = anthropic_to_xai({"messages": [], "max_tokens": 4096})
        assert result["max_output_tokens"] == 4096
        assert "max_tokens" not in result

    def test_absent_max_tokens_omits_max_output_tokens(self):
        """No Anthropic max_tokens -> no cap forwarded; grok applies its own default."""
        result, _ = anthropic_to_xai({"messages": []})
        assert "max_output_tokens" not in result

    def test_nonpositive_max_tokens_omits_max_output_tokens(self):
        """A malformed non-positive max_tokens is ignored rather than forwarded as a cap."""
        result, _ = anthropic_to_xai({"messages": [], "max_tokens": 0})
        assert "max_output_tokens" not in result

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
        import claude_bridge.providers.xai.translate as xai_translate

        monkeypatch.setattr(xai_translate, "_XAI_REASONING_MODE", "drop")
        result, warnings = xai_translate.anthropic_to_xai(
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
        """A block type with no xAI route (server_tool_use) → redacted placeholder, no leak.

        The placeholder NEVER echoes the block's nested fields — a raw str(block) would
        both pollute the request and leak whatever the special block carried.
        """
        result, warnings = anthropic_to_xai(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "server_tool_use",
                                "id": "srvtoolu_x",
                                "name": "web_search",
                                "input": {"query": "SECRETQUERYPAYLOAD"},
                            }
                        ],
                    }
                ]
            }
        )
        block = result["input"][0]["content"][0]
        assert block["type"] == "input_text"
        assert block["text"] == "[unsupported content block: server_tool_use]"
        assert "SECRETQUERYPAYLOAD" not in json.dumps(result)
        assert any("server_tool_use" in w for w in warnings)

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
        import claude_bridge.providers.xai.translate as xai_translate

        monkeypatch.setattr(xai_translate, "_XAI_REASONING_MODE", "drop")
        _, warnings = xai_translate.anthropic_to_xai(
            {"messages": [], "thinking": {"type": "enabled", "budget_tokens": 5}}
        )
        assert any("drop" in w.lower() for w in warnings)

    def test_cache_control_on_system_stripped_without_warning(self):
        result, warnings = anthropic_to_xai(
            {
                "system": [{"type": "text", "text": "s", "cache_control": {"type": "ephemeral"}}],
                "messages": [],
            }
        )
        # The Anthropic per-block marker has no Responses equivalent, so it is dropped
        # from the outbound request. Sticky caching now rides an explicit prompt_cache_key,
        # so the old "caching is automatic" advisory is no longer emitted (noise on every turn).
        assert "cache_control" not in json.dumps(result)
        assert not any("cache_control" in w.lower() for w in warnings)

    def test_cache_control_on_tool_stripped_without_warning(self):
        result, warnings = anthropic_to_xai(
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
        assert "cache_control" not in json.dumps(result)
        assert not any("cache_control" in w.lower() for w in warnings)

    def test_cache_control_on_content_stripped_without_warning(self):
        result, warnings = anthropic_to_xai(
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
        assert "cache_control" not in json.dumps(result)
        assert not any("cache_control" in w.lower() for w in warnings)

    # -- provider delegation --------------------------------------------------

    def test_provider_translate_request_delegates(self, monkeypatch):
        """XAIProvider.translate_request routes through anthropic_to_xai."""
        monkeypatch.delenv("XAI_MODEL", raising=False)
        result, _ = XAIProvider().translate_request(
            {"messages": [{"role": "user", "content": "hi"}]}
        )
        assert result["model"] == "grok-4.6"


class TestMediaForwarding:
    """Anthropic image/document blocks -> xAI Responses input_image/input_file parts.

    Oracle: the input_image ``data:`` URL shape proven accepted by image_input.json, and
    the Responses input_image/input_file spec. Every degrade path asserts the base64
    payload never leaks into the forwarded part or the warning.
    """

    _PNG_1PX = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCA',placeholder"

    def _image_block(self, media_type: str = "image/png", data: str = "IMGDATA") -> dict:
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        }

    def _doc_block(
        self, media_type: str = "application/pdf", data: str = "DOCDATA", **extra
    ) -> dict:
        block: dict = {
            "type": "document",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        }
        block.update(extra)
        return block

    def _user_msg(self, *blocks: dict) -> dict:
        return {"messages": [{"role": "user", "content": list(blocks)}]}

    # -- image message path ---------------------------------------------------

    def test_base64_png_forwarded_as_input_image_data_uri(self):
        # Oracle: input_image.image_url is a data: URL (image_input.json proves the shape).
        result, warnings = anthropic_to_xai(self._user_msg(self._image_block()), _MEDIA_CAPS)
        assert _user_content_parts(result) == [
            {"type": "input_image", "image_url": "data:image/png;base64,IMGDATA"}
        ]
        assert warnings == []

    def test_url_image_forwarded_as_input_image_url(self):
        block = {"type": "image", "source": {"type": "url", "url": "https://ex.com/a.png"}}
        result, _ = anthropic_to_xai(self._user_msg(block), _MEDIA_CAPS)
        assert _user_content_parts(result) == [
            {"type": "input_image", "image_url": "https://ex.com/a.png"}
        ]

    def test_base64_image_outside_mime_allowlist_degrades_without_leaking_base64(self):
        # image/tiff is not a Responses input_image type -> redacted placeholder, never bytes.
        block = self._image_block(media_type="image/tiff", data="SECRETIMGBYTES")
        result, warnings = anthropic_to_xai(self._user_msg(block), _MEDIA_CAPS)
        parts = _user_content_parts(result)
        assert parts == [{"type": "input_text", "text": "[unsupported image: image/tiff]"}]
        assert "SECRETIMGBYTES" not in json.dumps(parts)
        assert "SECRETIMGBYTES" not in json.dumps(warnings)

    def test_image_degrades_when_provider_lacks_image_modality(self):
        # Default text-only capabilities -> no image forwarding.
        block = self._image_block(data="SECRETIMGBYTES")
        result, warnings = anthropic_to_xai(self._user_msg(block))
        parts = _user_content_parts(result)
        assert parts[0]["type"] == "input_text"
        assert parts[0]["text"].startswith("[unsupported image:")
        assert "SECRETIMGBYTES" not in json.dumps(parts) + json.dumps(warnings)

    def test_file_source_image_degrades_no_bytes(self):
        block = {"type": "image", "source": {"type": "file", "file_id": "file_123"}}
        src = parse_media_source(block)
        part, warnings = _translate_image_block(src, _MEDIA_CAPS)
        assert part == {"type": "input_text", "text": "[unsupported image: file]"}
        assert warnings

    # -- document message path ------------------------------------------------

    def test_base64_pdf_forwarded_as_input_file_data_uri(self):
        block = self._doc_block(title="report.pdf")
        result, warnings = anthropic_to_xai(self._user_msg(block), _MEDIA_CAPS)
        assert _user_content_parts(result) == [
            {
                "type": "input_file",
                "filename": "report.pdf",
                "file_data": "data:application/pdf;base64,DOCDATA",
            }
        ]
        assert warnings == []

    def test_url_document_forwarded_as_input_file_url(self):
        block = {"type": "document", "source": {"type": "url", "url": "https://ex.com/a.pdf"}}
        result, _ = anthropic_to_xai(self._user_msg(block), _MEDIA_CAPS)
        assert _user_content_parts(result) == [
            {"type": "input_file", "file_url": "https://ex.com/a.pdf"}
        ]

    def test_base64_document_outside_mime_allowlist_degrades(self):
        block = self._doc_block(media_type="text/csv", data="SECRETDOCBYTES")
        part, warnings = _translate_document_block(parse_media_source(block), _MEDIA_CAPS)
        assert part == {"type": "input_text", "text": "[unsupported document: text/csv]"}
        assert "SECRETDOCBYTES" not in json.dumps(part) + json.dumps(warnings)

    def test_document_filename_sanitized_strips_path(self):
        # A client-controlled title with path separators is reduced to a basename.
        assert _safe_document_filename("../../etc/passwd") == "passwd"
        assert _safe_document_filename("C:\\Windows\\secret.pdf") == "secret.pdf"

    def test_document_default_filename_when_title_absent(self):
        assert _safe_document_filename(None) == "document.pdf"
        assert _safe_document_filename("") == "document.pdf"

    def test_document_degrades_when_provider_lacks_document_modality(self):
        block = self._doc_block(data="SECRETDOCBYTES")
        result, warnings = anthropic_to_xai(self._user_msg(block))  # text-only default
        parts = _user_content_parts(result)
        assert parts[0]["type"] == "input_text"
        assert parts[0]["text"].startswith("[unsupported document:")
        assert "SECRETDOCBYTES" not in json.dumps(parts) + json.dumps(warnings)

    # -- tool_result media: array form (capability-gated) ---------------------

    def test_tool_result_media_forwarded_as_content_part_array_when_supported(self):
        # array-capable provider -> function_call_output.output is a LIST of real parts.
        block = {
            "type": "tool_result",
            "tool_use_id": "toolu_media_1",
            "content": [
                {"type": "text", "text": "here"},
                self._image_block(),
            ],
        }
        result, warnings = anthropic_to_xai(
            {"messages": [{"role": "user", "content": [block]}]}, _ARRAY_TOOL_CAPS
        )
        outputs = _input_items(result, "function_call_output")
        assert len(outputs) == 1
        assert outputs[0]["output"] == [
            {"type": "input_text", "text": "here"},
            {"type": "input_image", "image_url": "data:image/png;base64,IMGDATA"},
        ]
        # The verbatim-call_id divergence is preserved in the array path.
        assert outputs[0]["call_id"] == "toolu_media_1"
        assert warnings == []

    def test_tool_result_error_prepends_error_marker_part(self):
        content = [{"type": "text", "text": "boom"}]
        parts, _ = _tool_result_parts(content, _ARRAY_TOOL_CAPS, is_error=True)
        assert parts[0] == {"type": "input_text", "text": "[Error]"}
        assert parts[1] == {"type": "input_text", "text": "boom"}

    def test_tool_result_media_degrades_to_string_when_array_unsupported(self):
        # img/doc INPUT supported but tool-output arrays NOT -> media redacted to a string,
        # never base64; call_id still verbatim.
        block = {
            "type": "tool_result",
            "tool_use_id": "toolu_media_2",
            "content": [{"type": "text", "text": "cap"}, self._image_block(data="SECRETIMGBYTES")],
        }
        result, warnings = anthropic_to_xai(
            {"messages": [{"role": "user", "content": [block]}]}, _MEDIA_CAPS
        )
        outputs = _input_items(result, "function_call_output")
        assert isinstance(outputs[0]["output"], str)
        assert "SECRETIMGBYTES" not in outputs[0]["output"]
        assert outputs[0]["call_id"] == "toolu_media_2"
        assert any("redacted" in w for w in warnings)

    # -- capability threading -------------------------------------------------

    def test_anthropic_to_xai_defaults_to_conservative_text_only(self):
        # No capabilities argument => the pre-media conservative default (degrade image).
        result, _ = anthropic_to_xai(self._user_msg(self._image_block()))
        assert _user_content_parts(result)[0]["type"] == "input_text"

    def test_translate_request_honors_provider_declared_capabilities(self):
        # translate_request must thread self.capabilities: media forwards iff the provider
        # declares the image modality (text-only until B8 wires media caps).
        result, _ = XAIProvider().translate_request(self._user_msg(self._image_block()))
        part = _user_content_parts(result)[0]
        if "image" in XAIProvider.capabilities.input_modalities:
            assert part == {"type": "input_image", "image_url": "data:image/png;base64,IMGDATA"}
        else:
            assert part["type"] == "input_text"


class TestResponseTranslation:
    """xai_to_anthropic: Responses object -> Anthropic Messages response (B5).

    Oracles are the golden wire captures (text_nonstream / single_tool_call /
    incomplete_max_tokens) for structure, plus the published Responses contract for
    content_filter, refusal, and malformed-argument handling (cli-chat-proxy mirrors the
    Responses API identically — the same source the openai provider translates). Every
    expected value is derived from the fixture bytes or the contract, never from running
    the translator. The load-bearing divergence from the openai provider: a tool_use `id`
    is the upstream ``call_id`` VERBATIM (no ``fc_``/``call_`` rewrite), so it round-trips
    to the request-side ``function_call_output`` call_id unchanged.
    """

    # -- completed text -------------------------------------------------------

    def test_completed_text_maps_to_single_text_block_end_turn(self):
        # Golden completed response -> one text block "pong"; the reasoning item is dropped
        # (only message output_text becomes content), status completed -> end_turn.
        result = xai_to_anthropic(_load_fixture("text_nonstream.json"))
        assert result["stop_reason"] == "end_turn"
        assert result["content"] == [{"type": "text", "text": "pong"}]

    def test_completed_text_envelope_id_role_model(self):
        result = xai_to_anthropic(_load_fixture("text_nonstream.json"))
        assert result["id"] == "msg_bridge_b9c3e6c4-1384-9c9e-8141-d292fdd48011"
        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert result["model"] == "grok-4.20-0309-reasoning"

    def test_completed_text_usage_flat_mapped_multiplier_one(self):
        # usage.input_tokens=192, output_tokens=309 in the fixture; xAI multiplier is 1.0,
        # so they pass through unscaled. Cached tokens (128) are NOT split out (Anthropic
        # totals are non-overlapping).
        result = xai_to_anthropic(_load_fixture("text_nonstream.json"))
        assert result["usage"] == {"input_tokens": 192, "output_tokens": 309}

    def test_usage_multiplier_scales_and_rounds_half_up(self):
        # A non-default multiplier proves the scale threads AND the +0.5 half-up rounding:
        # 192*1.5 = 288.0 -> 288; 309*1.5 = 463.5 -> 464 (a truncating int() would give 463).
        result = xai_to_anthropic(_load_fixture("text_nonstream.json"), token_count_multiplier=1.5)
        assert result["usage"] == {"input_tokens": 288, "output_tokens": 464}

    def test_usage_coerces_float_down_and_clamps_negative_to_zero(self):
        # 10.9 -> int 10 (truncate before scale); -5 -> max(0, -5) = 0. A missing max(0,...)
        # would surface a negative token count that Claude Code's /context math divides by.
        response = {
            "status": "completed",
            "id": "x",
            "output": [],
            "usage": {"input_tokens": 10.9, "output_tokens": -5},
        }
        assert xai_to_anthropic(response)["usage"] == {"input_tokens": 10, "output_tokens": 0}

    def test_usage_non_numeric_defaults_to_zero(self):
        response = {"status": "completed", "id": "x", "output": [], "usage": {"input_tokens": "?"}}
        assert xai_to_anthropic(response)["usage"] == {"input_tokens": 0, "output_tokens": 0}

    def test_missing_usage_defaults_to_zero(self):
        response = {"status": "completed", "id": "x", "output": []}
        assert xai_to_anthropic(response)["usage"] == {"input_tokens": 0, "output_tokens": 0}

    # -- tool calls -----------------------------------------------------------

    def test_completed_tool_call_maps_to_tool_use_and_stop_reason(self):
        result = xai_to_anthropic(_load_fixture("single_tool_call.json"))
        assert result["stop_reason"] == "tool_use"
        tool_uses = [b for b in result["content"] if b["type"] == "tool_use"]
        assert len(tool_uses) == 1

    def test_tool_use_id_is_upstream_call_id_verbatim(self):
        # THE xAI divergence: expose the call_id EXACTLY as captured, not the fc_ item id
        # and not any rewritten form. This is what makes the request-side call_id round-trip.
        result = xai_to_anthropic(_load_fixture("single_tool_call.json"))
        tool_use = next(b for b in result["content"] if b["type"] == "tool_use")
        assert tool_use["id"] == "call-bb020360-5c8a-4760-a7e6-b9cff0c2e700-0"

    def test_tool_use_name_and_parsed_input(self):
        result = xai_to_anthropic(_load_fixture("single_tool_call.json"))
        tool_use = next(b for b in result["content"] if b["type"] == "tool_use")
        assert tool_use["name"] == "get_weather"
        assert tool_use["input"] == {"city": "Paris"}

    def test_tool_call_id_falls_back_to_item_id_when_call_id_absent(self):
        # No call_id on the item -> fall back to the item id, still verbatim.
        response = {
            "status": "completed",
            "id": "r",
            "output": [
                {"type": "function_call", "id": "fc_solo", "name": "ping", "arguments": "{}"}
            ],
        }
        tool_use = next(
            b for b in xai_to_anthropic(response)["content"] if b["type"] == "tool_use"
        )
        assert tool_use["id"] == "fc_solo"

    def test_malformed_tool_arguments_preserved_as_raw(self):
        # Unparseable arguments must not crash or be silently dropped; the raw string is
        # preserved under _raw so nothing is lost.
        response = {
            "status": "completed",
            "id": "r",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call-x",
                    "name": "f",
                    "arguments": "not json{",
                }
            ],
        }
        tool_use = next(
            b for b in xai_to_anthropic(response)["content"] if b["type"] == "tool_use"
        )
        assert tool_use["input"] == {"_raw": "not json{"}

    def test_tool_call_wins_over_incomplete_status(self):
        # has_tool_calls outranks an incomplete terminal: Claude Code must run the tool,
        # not treat the turn as truncated.
        response = {
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "id": "r",
            "output": [{"type": "function_call", "call_id": "c", "name": "f", "arguments": "{}"}],
        }
        assert xai_to_anthropic(response)["stop_reason"] == "tool_use"

    # -- incomplete / stop_reason mapping ------------------------------------

    def test_incomplete_max_tokens_maps_to_max_tokens_with_partial_text(self):
        result = xai_to_anthropic(_load_fixture("incomplete_max_tokens.json"))
        assert result["stop_reason"] == "max_tokens"
        assert result["content"] == [
            {"type": "text", "text": "**Quantum Chromodynamics (QCD)** is the SU(3) gauge theory"}
        ]

    def test_incomplete_null_details_reads_as_max_tokens(self):
        # status incomplete with a null incomplete_details (the GPT-5 quirk) -> the
        # conservative token-exhaustion default, not content_filter.
        response = {"status": "incomplete", "incomplete_details": None, "id": "r", "output": []}
        assert xai_to_anthropic(response)["stop_reason"] == "max_tokens"

    def test_missing_status_defaults_to_completed_end_turn(self):
        response = {
            "id": "r",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "hi"}]}],
        }
        assert xai_to_anthropic(response)["stop_reason"] == "end_turn"

    # -- content_filter (Responses contract, spec-derived) -------------------

    def test_content_filter_without_text_synthesizes_visible_refusal(self):
        # A content-filtered turn with no model text ends cleanly (end_turn, NOT max_tokens
        # -> Claude Code must not auto-compact and retry) and renders a visible refusal
        # rather than a blank assistant message.
        response = {
            "status": "incomplete",
            "incomplete_details": {"reason": "content_filter"},
            "id": "r",
            "output": [],
        }
        result = xai_to_anthropic(response)
        assert result["stop_reason"] == "end_turn"
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "text"
        assert "content safety filters" in result["content"][0]["text"]

    def test_content_filter_with_text_does_not_synthesize_refusal(self):
        # When the filtered turn still produced text, surface only that text (no synthetic
        # refusal appended) — the has_text guard.
        response = {
            "status": "incomplete",
            "incomplete_details": {"reason": "content_filter"},
            "id": "r",
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "partial"}]}
            ],
        }
        result = xai_to_anthropic(response)
        assert result["stop_reason"] == "end_turn"
        assert result["content"] == [{"type": "text", "text": "partial"}]

    # -- other output item kinds ---------------------------------------------

    def test_refusal_item_surfaced_as_text(self):
        # A Responses ``refusal`` output item carries human-readable text — surface it.
        response = {
            "status": "completed",
            "id": "r",
            "output": [{"type": "refusal", "refusal": "I won't do that."}],
        }
        assert xai_to_anthropic(response)["content"] == [
            {"type": "text", "text": "I won't do that."}
        ]

    def test_empty_completed_output_yields_empty_content(self):
        response = {"status": "completed", "id": "r", "output": []}
        result = xai_to_anthropic(response)
        assert result["content"] == []
        assert result["stop_reason"] == "end_turn"

    # -- provider delegation --------------------------------------------------

    def test_provider_translate_response_delegates_to_xai_to_anthropic(self):
        # XAIProvider.translate_response threads self.capabilities.token_count_multiplier
        # (1.0) and returns the same translation. Asserts the stable outcome, not object
        # identity, so it survives B7 adding reasoning stashing on top.
        result = XAIProvider().translate_response(_load_fixture("text_nonstream.json"))
        assert result["stop_reason"] == "end_turn"
        assert result["content"] == [{"type": "text", "text": "pong"}]
        assert result["usage"] == {"input_tokens": 192, "output_tokens": 309}


# --- B6: SSE stream translation helpers ---


async def _aiter_bytes(chunks: list[bytes]):
    """Yield raw byte chunks as an async iterator (a fake provider stream)."""
    for chunk in chunks:
        yield chunk


def _run_stream(chunks: list[bytes]) -> list[dict]:
    """Drive XAIProvider.translate_stream over byte chunks, collecting Anthropic events."""

    async def _collect() -> list[dict]:
        return [event async for event in XAIProvider().translate_stream(_aiter_bytes(chunks))]

    return asyncio.run(_collect())


def _event_names(events: list[dict]) -> list[str]:
    return [e["event"] for e in events]


def _sse_blob(event_type: str, data: dict) -> bytes:
    """Serialize one provider SSE event to wire bytes for the stream driver."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n".encode()


def _terminal_blob(event_type: str, response: dict) -> bytes:
    """Wrap a full Responses object in a terminal SSE event (completed/incomplete)."""
    return _sse_blob(event_type, {"type": event_type, "response": response})


_CREATED_BLOB = _terminal_blob(
    "response.created", {"id": "r", "model": "grok-4.20-0309-reasoning", "status": "in_progress"}
)


class TestSseEventTranslation:
    """translate_xai_sse_event: one Responses SSE event -> Anthropic SSE events (B6).

    Oracles are the golden stream capture (text_stream.txt) event shapes plus the published
    Responses streaming contract (cli-chat-proxy mirrors it identically). The divergence from
    the openai provider: a streamed tool_use id is the upstream call_id VERBATIM, matching the
    non-stream path so a call streamed and a call replayed share one id.
    """

    def test_response_created_yields_message_start_and_ping(self):
        events = translate_xai_sse_event(
            {
                "event": "response.created",
                "data": {"response": {"id": "abc", "model": "grok-4.20", "usage": None}},
            }
        )
        assert _event_names(events) == ["message_start", "ping"]
        message = events[0]["data"]["message"]
        assert message["id"] == "msg_bridge_abc"
        assert message["model"] == "grok-4.20"
        assert message["stop_reason"] is None
        # created carries no usage yet -> zero, output always 0 at start.
        assert message["usage"] == {"input_tokens": 0, "output_tokens": 0}

    def test_response_created_scales_input_usage_when_present(self):
        events = translate_xai_sse_event(
            {
                "event": "response.created",
                "data": {"response": {"id": "a", "usage": {"input_tokens": 10}}},
            },
            token_count_multiplier=2.0,
        )
        assert events[0]["data"]["message"]["usage"] == {"input_tokens": 20, "output_tokens": 0}

    def test_content_part_added_opens_text_block(self):
        events = translate_xai_sse_event(
            {"event": "response.content_part.added", "data": {"content_index": 0}}
        )
        assert events == [
            {
                "event": "content_block_start",
                "data": {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            }
        ]

    def test_output_text_delta_maps_to_text_delta(self):
        events = translate_xai_sse_event(
            {"event": "response.output_text.delta", "data": {"content_index": 0, "delta": "hi"}}
        )
        assert events[0]["data"]["delta"] == {"type": "text_delta", "text": "hi"}

    def test_output_text_done_closes_block(self):
        events = translate_xai_sse_event(
            {"event": "response.output_text.done", "data": {"content_index": 0}}
        )
        assert events[0]["event"] == "content_block_stop"
        assert events[0]["data"]["index"] == 0

    def test_function_call_item_opens_tool_use_with_verbatim_call_id(self):
        # THE divergence, in the stream path: the tool_use id is the call_id EXACTLY, not the
        # fc_ item id and not a rewritten form.
        events = translate_xai_sse_event(
            {
                "event": "response.output_item.added",
                "data": {
                    "output_index": 0,
                    "item": {
                        "type": "function_call",
                        "call_id": "call-xyz-0",
                        "id": "fc_r_0",
                        "name": "get_weather",
                    },
                },
            }
        )
        block = events[0]["data"]["content_block"]
        assert block == {
            "type": "tool_use",
            "id": "call-xyz-0",
            "name": "get_weather",
            "input": {},
        }

    def test_reasoning_output_item_added_is_dropped(self):
        # A reasoning item is not surfaced as an Anthropic content block in the stream.
        events = translate_xai_sse_event(
            {
                "event": "response.output_item.added",
                "data": {"output_index": 0, "item": {"type": "reasoning", "id": "rs_r"}},
            }
        )
        assert events == []

    def test_function_call_arguments_delta_maps_to_input_json_delta(self):
        events = translate_xai_sse_event(
            {
                "event": "response.function_call_arguments.delta",
                "data": {"output_index": 0, "delta": '{"city":'},
            }
        )
        assert events[0]["data"]["delta"] == {
            "type": "input_json_delta",
            "partial_json": '{"city":',
        }

    def test_function_call_arguments_done_closes_block(self):
        events = translate_xai_sse_event(
            {"event": "response.function_call_arguments.done", "data": {"output_index": 2}}
        )
        assert events[0]["event"] == "content_block_stop"
        assert events[0]["data"]["index"] == 2

    def test_completed_terminal_yields_message_delta_and_stop(self):
        events = translate_xai_sse_event(
            {
                "event": "response.completed",
                "data": {
                    "response": {
                        "status": "completed",
                        "output": [],
                        "usage": {"output_tokens": 7},
                    }
                },
            }
        )
        assert _event_names(events) == ["message_delta", "message_stop"]
        assert events[0]["data"]["delta"] == {"stop_reason": "end_turn"}
        assert events[0]["data"]["usage"]["output_tokens"] == 7

    def test_incomplete_max_tokens_terminal_maps_to_max_tokens(self):
        events = translate_xai_sse_event(
            {
                "event": "response.incomplete",
                "data": {
                    "response": {
                        "status": "incomplete",
                        "incomplete_details": {"reason": "max_output_tokens"},
                        "output": [],
                    }
                },
            }
        )
        assert events[0]["data"]["delta"] == {"stop_reason": "max_tokens"}
        assert events[-1]["event"] == "message_stop"

    def test_incomplete_content_filter_terminal_synthesizes_refusal_block(self):
        events = translate_xai_sse_event(
            {
                "event": "response.incomplete",
                "data": {
                    "response": {
                        "status": "incomplete",
                        "incomplete_details": {"reason": "content_filter"},
                        "output": [],
                    }
                },
            }
        )
        # A refusal text block is synthesized before the terminal; stop_reason is end_turn.
        assert _event_names(events)[:3] == [
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
        ]
        refusal = events[1]["data"]["delta"]["text"]
        assert "content safety filters" in refusal
        message_delta = next(e for e in events if e["event"] == "message_delta")
        assert message_delta["data"]["delta"] == {"stop_reason": "end_turn"}
        assert events[-1]["event"] == "message_stop"

    def test_response_failed_maps_to_error_event(self):
        events = translate_xai_sse_event(
            {
                "event": "response.failed",
                "data": {"response": {"error": {"code": "server_error", "message": "boom"}}},
            }
        )
        assert events[0]["event"] == "error"
        assert events[0]["data"]["error"]["message"] == "boom"
        assert events[0]["data"]["error"]["type"] == "api_error"

    def test_top_level_error_maps_to_error_event(self):
        events = translate_xai_sse_event({"event": "error", "data": {"message": "kaboom"}})
        assert events[0]["event"] == "error"
        assert events[0]["data"]["error"]["message"] == "kaboom"

    def test_error_message_is_length_bounded(self):
        events = translate_xai_sse_event({"event": "error", "data": {"message": "x" * 1000}})
        assert len(events[0]["data"]["error"]["message"]) == 500

    def test_skipped_events_yield_nothing(self):
        for event_type in (
            "response.in_progress",
            "response.queued",
            "response.content_part.done",
            "response.output_item.done",
        ):
            assert translate_xai_sse_event({"event": event_type, "data": {}}) == []

    def test_unknown_event_yields_nothing(self):
        assert (
            translate_xai_sse_event({"event": "response.reasoning_summary_text.delta", "data": {}})
            == []
        )


class TestStreamLifecycle:
    """XAIProvider.translate_stream: full lifecycle over the golden capture + block remapping."""

    def test_text_stream_full_lifecycle_matches_golden_capture(self):
        raw = (_FIXTURES / "text_stream.txt").read_bytes()
        events = _run_stream([raw])
        assert _event_names(events) == [
            "message_start",
            "ping",
            "content_block_start",
            "content_block_delta",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]

    def test_text_stream_reconstructs_the_message(self):
        raw = (_FIXTURES / "text_stream.txt").read_bytes()
        events = _run_stream([raw])
        deltas = [
            e["data"]["delta"]["text"] for e in events if e["event"] == "content_block_delta"
        ]
        assert "".join(deltas) == "hi there"

    def test_text_stream_terminal_carries_end_turn_and_full_usage(self):
        raw = (_FIXTURES / "text_stream.txt").read_bytes()
        events = _run_stream([raw])
        message_delta = next(e for e in events if e["event"] == "message_delta")
        assert message_delta["data"]["delta"]["stop_reason"] == "end_turn"
        assert message_delta["data"]["usage"] == {"input_tokens": 189, "output_tokens": 333}

    def test_message_start_id_and_model_from_created(self):
        raw = (_FIXTURES / "text_stream.txt").read_bytes()
        events = _run_stream([raw])
        message = events[0]["data"]["message"]
        assert message["id"] == "msg_bridge_322daa90-f2da-959b-88b8-4a968a6b4d54"
        assert message["model"] == "grok-4.20-0309-reasoning"

    def test_block_indices_are_sequential_from_zero(self):
        raw = (_FIXTURES / "text_stream.txt").read_bytes()
        events = _run_stream([raw])
        block_start = next(e for e in events if e["event"] == "content_block_start")
        assert block_start["data"]["index"] == 0
        for e in events:
            if e["event"] in ("content_block_delta", "content_block_stop"):
                assert e["data"]["index"] == 0

    def test_stream_survives_chunk_boundaries_mid_event(self):
        # Splitting the raw bytes at an arbitrary offset (mid-event) must not change the
        # translation — iter_sse_event_blobs reframes on \n\n regardless of chunk seams.
        raw = (_FIXTURES / "text_stream.txt").read_bytes()
        mid = len(raw) // 2
        whole = _run_stream([raw])
        split = _run_stream([raw[:mid], raw[mid:]])
        assert _event_names(split) == _event_names(whole)

    def test_streamed_tool_call_opens_tool_use_and_ends_tool_use(self):
        # Spec-derived streaming tool-call sequence (Responses streaming contract). The
        # tool_use id is the call_id verbatim; the terminal stop_reason is tool_use.
        events = _run_stream(
            [
                _CREATED_BLOB,
                _sse_blob(
                    "response.output_item.added",
                    {
                        "output_index": 0,
                        "item": {
                            "type": "function_call",
                            "call_id": "call-abc-0",
                            "id": "fc_r_0",
                            "name": "get_weather",
                        },
                    },
                ),
                _sse_blob(
                    "response.function_call_arguments.delta",
                    {"output_index": 0, "delta": '{"city":"Paris"}'},
                ),
                _sse_blob("response.function_call_arguments.done", {"output_index": 0}),
                _terminal_blob(
                    "response.completed",
                    {
                        "status": "completed",
                        "output": [
                            {
                                "type": "function_call",
                                "call_id": "call-abc-0",
                                "name": "get_weather",
                            }
                        ],
                        "usage": {"input_tokens": 5, "output_tokens": 6},
                    },
                ),
            ]
        )
        tool_start = next(
            e
            for e in events
            if e["event"] == "content_block_start"
            and e["data"]["content_block"]["type"] == "tool_use"
        )
        assert tool_start["data"]["content_block"]["id"] == "call-abc-0"
        json_delta = next(e for e in events if e["event"] == "content_block_delta")
        assert json_delta["data"]["delta"]["partial_json"] == '{"city":"Paris"}'
        message_delta = next(e for e in events if e["event"] == "message_delta")
        assert message_delta["data"]["delta"]["stop_reason"] == "tool_use"

    def test_streamed_tool_call_absent_from_terminal_output_still_stops_tool_use(self):
        # Safety net: the stream ANNOUNCES a tool via output_item.added, but the terminal
        # response.completed omits it from output[] (a stream/terminal mismatch). _stop_reason
        # computes end_turn from the empty terminal output, but the driver remembers a tool
        # block was streamed and upgrades the stop_reason to tool_use so Claude Code runs it.
        events = _run_stream(
            [
                _CREATED_BLOB,
                _sse_blob(
                    "response.output_item.added",
                    {
                        "output_index": 0,
                        "item": {"type": "function_call", "call_id": "call-q-0", "name": "f"},
                    },
                ),
                _sse_blob("response.function_call_arguments.done", {"output_index": 0}),
                _terminal_blob(
                    "response.completed",
                    {"status": "completed", "output": [], "usage": {"output_tokens": 1}},
                ),
            ]
        )
        message_delta = next(e for e in events if e["event"] == "message_delta")
        assert message_delta["data"]["delta"]["stop_reason"] == "tool_use"
        assert _event_names(events)[-1] == "message_stop"


class TestStreamTerminatorInvariant:
    """Lifecycle invariant: every stream that emits message_start ends in a terminator.

    Universally quantified over the terminal-event enum the upstream can emit — a table
    test, because an example cannot establish a 'for all terminals' property. Run against a
    translate_stream that forgot any terminal, the missing row goes red.
    """

    @pytest.mark.parametrize(
        ("terminal_blob", "expected_terminator"),
        [
            (
                _terminal_blob("response.completed", {"status": "completed", "output": []}),
                "message_stop",
            ),
            (
                _terminal_blob(
                    "response.incomplete",
                    {
                        "status": "incomplete",
                        "incomplete_details": {"reason": "max_output_tokens"},
                        "output": [],
                    },
                ),
                "message_stop",
            ),
            (
                _terminal_blob(
                    "response.incomplete",
                    {
                        "status": "incomplete",
                        "incomplete_details": {"reason": "content_filter"},
                        "output": [],
                    },
                ),
                "message_stop",
            ),
            (
                _sse_blob(
                    "response.failed",
                    {"response": {"error": {"code": "server_error", "message": "boom"}}},
                ),
                "error",
            ),
            (_sse_blob("error", {"message": "kaboom"}), "error"),
        ],
    )
    def test_every_terminal_event_closes_the_stream(self, terminal_blob, expected_terminator):
        events = _run_stream([_CREATED_BLOB, terminal_blob])
        assert _event_names(events)[0] == "message_start"
        assert _event_names(events)[-1] == expected_terminator

    def test_dropped_stream_after_start_synthesizes_message_stop(self):
        # message_start emitted but the upstream dropped before any terminal -> a synthetic
        # message_stop is appended so Claude Code finalizes the turn instead of hanging.
        events = _run_stream([_CREATED_BLOB])
        assert _event_names(events)[0] == "message_start"
        assert _event_names(events)[-1] == "message_stop"

    def test_dropped_stream_after_tool_call_synthesizes_tool_use_stop(self):
        # A drop after a tool_use block must stop as tool_use (Claude Code must run the tool),
        # never end_turn masquerading as completion nor max_tokens triggering a retry loop.
        events = _run_stream(
            [
                _CREATED_BLOB,
                _sse_blob(
                    "response.output_item.added",
                    {
                        "output_index": 0,
                        "item": {"type": "function_call", "call_id": "call-z-0", "name": "f"},
                    },
                ),
            ]
        )
        message_delta = next(e for e in events if e["event"] == "message_delta")
        assert message_delta["data"]["delta"]["stop_reason"] == "tool_use"
        assert _event_names(events)[-1] == "message_stop"

    def test_stream_that_never_started_gets_no_synthetic_terminator(self):
        # A stream of only skipped events never emits message_start, so no fake terminator is
        # invented — the invariant is guarded on 'started', not applied unconditionally.
        events = _run_stream([_sse_blob("response.in_progress", {})])
        assert events == []


class TestStreamBufferCap:
    """The 4MiB undrained-buffer abort — a terminator-less stream must not OOM."""

    def test_max_sse_buffer_is_four_mebibytes(self):
        # The cap is the spec value, derived from the doctrine (one partial event fits well
        # under 4MiB); a terminator-less stream is malformed past it.
        assert _MAX_SSE_BUFFER == 4 * 1024 * 1024

    def test_oversized_terminatorless_stream_aborts(self):
        oversized = b"x" * (_MAX_SSE_BUFFER + 1)  # no \n\n terminator, ever
        with pytest.raises(RuntimeError, match="terminator"):
            _run_stream([oversized])


# ---------------------------------------------------------------------------
# B7 — reasoning / tool continuity (encrypted-reasoning echo across turns)
# ---------------------------------------------------------------------------

_CID = "call-bb020360-5c8a-4760-a7e6-b9cff0c2e700-0"


def _encrypted_reasoning_item() -> dict:
    """The captured reasoning item structure carrying an ``encrypted_content`` field
    (reasoning_encrypted.json output[0]) — the continuation state B7 must round-trip.

    The opaque blob itself is a synthetic sentinel: the real provider-issued
    continuation is never persisted to the repo (in-memory-only invariant; see
    ``test_no_fixture_persists_a_real_encrypted_reasoning_blob``)."""
    return _load_fixture("reasoning_encrypted.json")["output"][0]


def _real_function_call(call_id: str) -> dict:
    """A real captured function_call item (single_tool_call.json output[1]), rekeyed to
    ``call_id`` — its own item id stays the fixture's ``fc_...`` form."""
    fc = dict(_load_fixture("single_tool_call.json")["output"][1])
    fc["call_id"] = call_id
    return fc


def _encrypted_reasoning_tool_response(call_id: str = _CID, *, status: str = "completed") -> dict:
    """Compose a store:false tool turn: a captured encrypted reasoning item (its opaque blob
    replaced by a synthetic sentinel) followed by a real function_call.

    xAI was captured emitting encrypted reasoning (reasoning_encrypted.json — a text turn) and
    tool calls (single_tool_call.json — captured ``store:true``, so no encrypted blob) in
    separate responses; under ``store:false`` + ``include:[reasoning.encrypted_content]`` a tool
    turn carries both. This composes the two REAL captured pieces rather than inventing a wire
    shape (Boundary Fixture Fidelity — schema-faithful composition of captured parts).
    """
    return {
        "id": "resp_compose",
        "model": "grok-4.20-0309-reasoning",
        "status": status,
        "output": [_encrypted_reasoning_item(), _real_function_call(call_id)],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def _anthropic_tool_use_request(*call_ids: str) -> dict:
    """An Anthropic request whose assistant turn issued the given ``tool_use`` call ids."""
    return {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": cid,
                        "name": "get_weather",
                        "input": {"city": "Paris"},
                    }
                    for cid in call_ids
                ],
            }
        ]
    }


def _run_stream_on(provider: XAIProvider, chunks: list[bytes]) -> list[dict]:
    """Drive translate_stream on a GIVEN provider instance (so cache state persists)."""

    async def _collect() -> list[dict]:
        return [event async for event in provider.translate_stream(_aiter_bytes(chunks))]

    return asyncio.run(_collect())


def _reasoning_items(result: dict) -> list[dict]:
    return [item for item in result["input"] if item.get("type") == "reasoning"]


class TestReasoningAssociation:
    """_associate_reasoning_with_calls: pair each function_call with its immediately-preceding
    ENCRYPTED reasoning item, keyed by the VERBATIM call_id (B7).

    Oracles: the real captures give the negative case (no ``encrypted_content`` → no
    continuation state — single_tool_call/parallel were captured ``store:true``); the positive
    case composes the real encrypted reasoning item with a real function_call.
    """

    def test_reasoning_then_message_yields_no_association(self):
        # reasoning_encrypted.json: encrypted reasoning followed by a message, no tool call.
        output = _load_fixture("reasoning_encrypted.json")["output"]
        assert _associate_reasoning_with_calls(output) == {}

    def test_tool_call_without_encrypted_reasoning_yields_no_association(self):
        # single_tool_call.json (store:true) → its reasoning carries no encrypted_content.
        output = _load_fixture("single_tool_call.json")["output"]
        assert _associate_reasoning_with_calls(output) == {}

    def test_parallel_tool_calls_without_encrypted_reasoning_yield_no_association(self):
        output = _load_fixture("parallel_tool_calls.json")["output"]
        assert _associate_reasoning_with_calls(output) == {}

    def test_encrypted_reasoning_before_call_binds_by_verbatim_call_id(self):
        reasoning = _encrypted_reasoning_item()
        assoc = _associate_reasoning_with_calls([reasoning, _real_function_call(_CID)])
        assert assoc == {_CID: reasoning}

    def test_association_key_is_call_id_not_the_fc_item_id(self):
        # The B7 divergence from the openai provider: the key is the call_id VERBATIM, never the
        # function_call's own ``fc_...`` item id and never an ``fc_``-rewritten form.
        reasoning = _encrypted_reasoning_item()
        fc = _real_function_call(_CID)  # its item id is fc_c4de0fe3-..._0
        assoc = _associate_reasoning_with_calls([reasoning, fc])
        assert _CID in assoc
        assert fc["id"] not in assoc
        assert not any(key.startswith("fc_") for key in assoc)

    def test_single_reasoning_binds_only_the_first_of_parallel_calls(self):
        # One encrypted reasoning item is consumed by the first call; the second is unpaired.
        reasoning = _encrypted_reasoning_item()
        cid0 = "call-9b62d611-9755-4457-a20c-a1cb51b96c96-0"
        cid1 = "call-9b62d611-9755-4457-a20c-a1cb51b96c96-1"
        assoc = _associate_reasoning_with_calls(
            [reasoning, _real_function_call(cid0), _real_function_call(cid1)]
        )
        assert assoc == {cid0: reasoning}

    def test_intervening_message_clears_pending_reasoning(self):
        # A non-call item between reasoning and the call breaks the pairing.
        reasoning = _encrypted_reasoning_item()
        message = _load_fixture("reasoning_encrypted.json")["output"][1]
        assoc = _associate_reasoning_with_calls([reasoning, message, _real_function_call(_CID)])
        assert assoc == {}

    def test_non_dict_output_item_is_tolerated_and_clears_pending(self):
        reasoning = _encrypted_reasoning_item()
        # Malformed upstream output (a non-dict element) must not crash the single-pass walk.
        malformed: list = [reasoning, "garbage", _real_function_call(_CID)]
        assert _associate_reasoning_with_calls(malformed) == {}


class TestReasoningRoundTrip:
    """Encrypted reasoning captured from one response is echoed back before the matching
    function_call on the NEXT request — keyed by the verbatim call_id, per provider instance,
    and NEVER surfaced to Claude Code (B7)."""

    def test_captured_reasoning_is_injected_before_its_function_call(self):
        provider = XAIProvider()
        reasoning = _encrypted_reasoning_item()
        provider.translate_response(_encrypted_reasoning_tool_response(_CID))

        result, _ = provider.translate_request(_anthropic_tool_use_request(_CID))
        items = result["input"]
        fc_index = next(i for i, it in enumerate(items) if it.get("type") == "function_call")
        assert items[fc_index - 1] == reasoning
        assert items[fc_index - 1].get("encrypted_content")

    def test_no_prior_capture_leaves_request_input_unchanged(self):
        provider = XAIProvider()
        result, _ = provider.translate_request(_anthropic_tool_use_request(_CID))
        assert _reasoning_items(result) == []

    def test_unmatched_call_id_injects_nothing(self):
        provider = XAIProvider()
        provider.translate_response(_encrypted_reasoning_tool_response(_CID))
        other = "call-ffffffff-0000-0000-0000-000000000000-0"
        result, _ = provider.translate_request(_anthropic_tool_use_request(other))
        assert _reasoning_items(result) == []

    def test_capture_is_instance_scoped_not_shared_across_providers(self):
        capturer = XAIProvider()
        capturer.translate_response(_encrypted_reasoning_tool_response(_CID))
        fresh = XAIProvider()
        result, _ = fresh.translate_request(_anthropic_tool_use_request(_CID))
        assert _reasoning_items(result) == []

    def test_encrypted_blob_is_never_returned_to_claude_code(self):
        # The opaque continuation blob is in-memory only: it must never appear in the Anthropic
        # response handed back to Claude Code.
        provider = XAIProvider()
        blob = _encrypted_reasoning_item()["encrypted_content"]
        anthropic = provider.translate_response(_encrypted_reasoning_tool_response(_CID))
        assert blob not in json.dumps(anthropic)

    def test_injection_tolerates_a_request_with_no_input_list(self):
        # Defensive guard: a translated request lacking a list ``input`` is left untouched
        # rather than crashing, even with a populated cache.
        provider = XAIProvider()
        provider.translate_response(_encrypted_reasoning_tool_response(_CID))
        payload = {"model": "grok-4.20"}
        provider._inject_reasoning(payload)
        assert "input" not in payload


class TestReasoningInjectionDedup:
    """A reasoning item cached under multiple call_ids is echoed at most once, so parallel tool
    calls that share one reasoning item never get a duplicate preceding copy (B7)."""

    def test_shared_reasoning_injected_once_for_parallel_calls(self):
        provider = XAIProvider()
        cid0 = "call-9b62d611-9755-4457-a20c-a1cb51b96c96-0"
        cid1 = "call-9b62d611-9755-4457-a20c-a1cb51b96c96-1"
        # Two turns whose encrypted reasoning shares one id, one cached per parallel call.
        provider.translate_response(_encrypted_reasoning_tool_response(cid0))
        provider.translate_response(_encrypted_reasoning_tool_response(cid1))
        result, _ = provider.translate_request(_anthropic_tool_use_request(cid0, cid1))
        items = _reasoning_items(result)
        assert len(items) == 1
        assert items[0].get("id") == _encrypted_reasoning_item()["id"]


class TestReasoningStreamCapture:
    """Streamed terminal events capture encrypted reasoning for the next request — completed AND
    incomplete, since an incomplete turn that still emitted a tool call needs its reasoning
    echoed too; non-terminal events capture nothing (B7)."""

    def _streamed_then_request(self, terminal_event: str, status: str) -> dict:
        provider = XAIProvider()
        response = _encrypted_reasoning_tool_response(_CID, status=status)
        _run_stream_on(provider, [_CREATED_BLOB, _terminal_blob(terminal_event, response)])
        result, _ = provider.translate_request(_anthropic_tool_use_request(_CID))
        return result

    def test_completed_stream_captures_reasoning_for_next_request(self):
        result = self._streamed_then_request("response.completed", "completed")
        assert len(_reasoning_items(result)) == 1

    def test_incomplete_stream_also_captures_reasoning(self):
        result = self._streamed_then_request("response.incomplete", "incomplete")
        assert len(_reasoning_items(result)) == 1

    def test_non_terminal_stream_events_capture_nothing(self):
        provider = XAIProvider()
        _run_stream_on(
            provider,
            [
                _CREATED_BLOB,
                _sse_blob(
                    "response.output_text.delta",
                    {"type": "response.output_text.delta", "delta": "hi", "content_index": 0},
                ),
            ],
        )
        result, _ = provider.translate_request(_anthropic_tool_use_request(_CID))
        assert _reasoning_items(result) == []


class TestReasoningCacheBound:
    """The reasoning cache is a bounded LRU: it never grows past _REASONING_CACHE_MAX, evicts the
    oldest entry first, and re-stashing an existing key refreshes its recency (B7)."""

    @staticmethod
    def _entry(index: int) -> dict:
        return {"type": "reasoning", "id": f"rs_{index}", "encrypted_content": "x"}

    def _fill(self, provider: XAIProvider, count: int) -> None:
        for i in range(count):
            provider._stash_reasoning({f"call-{i}": self._entry(i)})

    def test_cache_max_is_256(self):
        assert _REASONING_CACHE_MAX == 256

    def test_cache_never_exceeds_the_bound(self):
        provider = XAIProvider()
        self._fill(provider, _REASONING_CACHE_MAX + 50)
        assert len(provider._reasoning_by_call_id) == _REASONING_CACHE_MAX

    def test_oldest_entry_is_evicted_first(self):
        provider = XAIProvider()
        self._fill(provider, _REASONING_CACHE_MAX)
        provider._stash_reasoning({"call-new": self._entry(9999)})
        assert "call-0" not in provider._reasoning_by_call_id
        assert "call-new" in provider._reasoning_by_call_id
        assert "call-1" in provider._reasoning_by_call_id

    def test_restash_refreshes_recency_so_touched_entry_survives(self):
        provider = XAIProvider()
        self._fill(provider, _REASONING_CACHE_MAX)
        # Touch call-0 → most-recent; the next insert must now evict call-1, not call-0.
        provider._stash_reasoning({"call-0": self._entry(1000)})
        provider._stash_reasoning({"call-new": self._entry(9999)})
        assert "call-0" in provider._reasoning_by_call_id
        assert "call-1" not in provider._reasoning_by_call_id


# --- authenticate (subscription OAuth headers) ---


class TestAuthenticate:
    """authenticate() assembles the subscription bearer plus the grok client headers.

    The bearer is resolved from a real ~/.grok/auth.json (a fresh, far-future-exp fake
    JWT, so no refresh and no network fires) — in-process code is exercised directly,
    never mocked (Mock Decision Framework).
    """

    @pytest.mark.asyncio
    async def test_returns_bearer_version_and_identifier_headers(self, tmp_path, monkeypatch):
        data = _grok_auth()
        auth_file = _write_grok_auth(tmp_path, data)
        monkeypatch.setenv("XAI_CLIENT_VERSION", "1.2.3")

        provider = XAIProvider(auth_path=auth_file)
        headers = await provider.authenticate()

        # Oracle: token is the exact bearer we wrote; version is the env override
        # (config resolver returns it verbatim); identifier is the fixed client string;
        # conv-id is this instance's sticky prompt cache key. The exact-dict match also
        # proves no extra header leaks in.
        token = data[_ENTRY_KEY]["key"]
        assert headers == {
            "Authorization": f"Bearer {token}",
            "x-grok-client-version": "1.2.3",
            "x-grok-client-identifier": _XAI_CLIENT_IDENTIFIER,
            "x-grok-conv-id": provider._prompt_cache_key,
        }

    @pytest.mark.asyncio
    async def test_authorization_carries_the_auth_file_bearer(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XAI_CLIENT_VERSION", "9.9.9")
        data = _grok_auth()
        auth_file = _write_grok_auth(tmp_path, data)

        headers = await XAIProvider(auth_path=auth_file).authenticate()

        assert headers["Authorization"] == f"Bearer {data[_ENTRY_KEY]['key']}"

    @pytest.mark.asyncio
    async def test_client_identifier_is_grok_cli(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XAI_CLIENT_VERSION", "9.9.9")
        auth_file = _write_grok_auth(tmp_path, _grok_auth())

        headers = await XAIProvider(auth_path=auth_file).authenticate()

        # Oracle: the plan's documented divergence from Codex (bearer-only) — the
        # fixed grok CLI client identifier string, not a secret.
        assert headers["x-grok-client-identifier"] == "grok-cli"

    @pytest.mark.asyncio
    async def test_client_version_reflects_config_resolver(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XAI_CLIENT_VERSION", "0.2.93")
        auth_file = _write_grok_auth(tmp_path, _grok_auth())

        headers = await XAIProvider(auth_path=auth_file).authenticate()

        assert headers["x-grok-client-version"] == "0.2.93"

    @pytest.mark.asyncio
    async def test_bearer_never_leaks_outside_authorization_header(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XAI_CLIENT_VERSION", "1.2.3")
        data = _grok_auth()
        auth_file = _write_grok_auth(tmp_path, data)

        headers = await XAIProvider(auth_path=auth_file).authenticate()

        # Security oracle: the opaque bearer must appear ONLY in Authorization —
        # never smuggled into the version or identifier header.
        token = data[_ENTRY_KEY]["key"]
        assert token not in headers["x-grok-client-version"]
        assert token not in headers["x-grok-client-identifier"]


# --- prompt cache identity (sticky routing key) ---


class TestPromptCacheIdentity:
    """The prompt cache key is a per-instance sticky-routing identity: stable across
    every request on one provider (so grok-4.6 reuses the cached prefix), distinct across
    instances (so three concurrent launchers never collide), and never derived from request
    content. It rides both the body ``prompt_cache_key`` and the ``x-grok-conv-id`` header
    as the same value, because grok-build resolves cache identity as ``key.or(conv_id)``.
    """

    @staticmethod
    def _request(text: str) -> dict:
        return {"messages": [{"role": "user", "content": text}]}

    def test_prompt_cache_key_present_and_stable_across_requests_on_one_instance(self):
        provider = XAIProvider()
        first, _ = provider.translate_request(self._request("alpha"))
        second, _ = provider.translate_request(self._request("beta"))
        # Oracle: sticky routing requires ONE identity across the process, so the key is
        # invariant across requests on a single instance despite differing content.
        assert first["prompt_cache_key"]
        assert first["prompt_cache_key"] == second["prompt_cache_key"]

    def test_prompt_cache_key_differs_across_instances_for_identical_input(self):
        # Identical input to both instances: same content, different key proves the key is
        # instance identity, NOT hash(instructions) — the security constraint (INV-SEC-01/06).
        a, _ = XAIProvider().translate_request(self._request("same"))
        b, _ = XAIProvider().translate_request(self._request("same"))
        assert a["prompt_cache_key"] != b["prompt_cache_key"]

    @pytest.mark.asyncio
    async def test_conv_id_header_equals_body_prompt_cache_key(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XAI_CLIENT_VERSION", "1.2.3")
        auth_file = _write_grok_auth(tmp_path, _grok_auth())
        provider = XAIProvider(auth_path=auth_file)

        headers = await provider.authenticate()
        body, _ = provider.translate_request(self._request("hi"))

        # Oracle (plan sub-item 3): the one sticky key rides BOTH the header and the body,
        # as the SAME value — cli-chat-proxy accepts either (grok-build is key.or(conv_id)).
        assert headers["x-grok-conv-id"] == body["prompt_cache_key"]
        assert body["prompt_cache_key"]


# --- provider contract: capabilities, endpoint, registration ---


class TestProviderContract:
    def test_capabilities_declare_image_and_document_modalities(self):
        caps = XAIProvider().capabilities
        assert caps.input_modalities == frozenset({"text", "image", "document"})

    def test_capabilities_declare_array_form_tool_output(self):
        assert XAIProvider().capabilities.supports_tool_output_content_parts is True

    def test_capabilities_token_multiplier_is_identity(self):
        # Subscription-metered proxy needs no OpenAI-compat token scaling (D-XAI-005).
        assert XAIProvider().capabilities.token_count_multiplier == 1.0

    def test_capabilities_transport_modes(self):
        caps = XAIProvider().capabilities
        assert caps.stream_request_mode == "body_parameter"
        assert caps.sync_response_mode == "sse"

    def test_endpoint_is_cli_chat_proxy_responses(self):
        # Oracle: the subscription-metered proxy's Responses endpoint (plan decision).
        assert XAIProvider().endpoint == "https://cli-chat-proxy.grok.com/v1/responses"
        assert XAIProvider.endpoint == _XAI_ENDPOINT

    def test_provider_registered_in_registry(self):
        assert PROVIDERS["xai"] is XAIProvider

    def test_provider_capabilities_are_the_full_declaration(self):
        assert XAIProvider().capabilities is _XAI_CAPABILITIES
