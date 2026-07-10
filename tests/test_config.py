"""Tests for the bridge runtime configuration owner."""

from __future__ import annotations


def test_upstream_timeout_defaults_and_env_override(monkeypatch):
    """UPSTREAM_TIMEOUT is read at call time and falls back for invalid values."""
    import claude_bridge.config as config

    monkeypatch.delenv(config.UPSTREAM_TIMEOUT_ENV, raising=False)
    assert config.upstream_timeout(60) == 60
    assert config.upstream_timeout(120) == 120

    monkeypatch.setenv(config.UPSTREAM_TIMEOUT_ENV, "30")
    assert config.upstream_timeout(60) == 30
    assert config.upstream_timeout(120) == 30

    invalid_values: list[str] = []
    monkeypatch.setenv(config.UPSTREAM_TIMEOUT_ENV, "not-a-number")
    assert config.upstream_timeout(120, on_invalid=invalid_values.append) == 120
    assert invalid_values == ["not-a-number"]

    monkeypatch.setenv(config.UPSTREAM_TIMEOUT_ENV, "0")
    assert config.upstream_timeout(120, on_invalid=invalid_values.append) == 120
    assert invalid_values == ["not-a-number", "0"]


def test_max_request_body_default_and_override(monkeypatch):
    """MAX_REQUEST_BODY owns the import-time proxy body limit default."""
    import claude_bridge.config as config

    monkeypatch.delenv(config.MAX_REQUEST_BODY_ENV, raising=False)
    assert config.max_request_body() == 10_485_760

    monkeypatch.setenv(config.MAX_REQUEST_BODY_ENV, "2048")
    assert config.max_request_body() == 2048

    invalid_values: list[str] = []
    monkeypatch.setenv(config.MAX_REQUEST_BODY_ENV, "not-a-number")
    assert config.max_request_body(on_invalid=invalid_values.append) == 10_485_760
    assert invalid_values == ["not-a-number"]

    monkeypatch.setenv(config.MAX_REQUEST_BODY_ENV, "0")
    assert config.max_request_body(on_invalid=invalid_values.append) == 10_485_760
    assert invalid_values == ["not-a-number", "0"]


def test_fallback_chain_default_blank_and_csv(monkeypatch):
    """LLM_BRIDGE_FALLBACK produces the ordered registered-provider preference list."""
    import claude_bridge.config as config

    monkeypatch.delenv(config.LLM_BRIDGE_FALLBACK_ENV, raising=False)
    assert config.fallback_chain() == ["openai"]

    monkeypatch.setenv(config.LLM_BRIDGE_FALLBACK_ENV, " xai, openai,,")
    assert config.fallback_chain() == ["xai", "openai"]

    monkeypatch.setenv(config.LLM_BRIDGE_FALLBACK_ENV, "")
    assert config.fallback_chain() == []


def test_optional_path_and_api_key_accessors_trim_empty_values(monkeypatch):
    """Trace path and provider API keys normalize missing/blank values to None."""
    import claude_bridge.config as config

    monkeypatch.delenv(config.CLAUDE_BRIDGE_TRACE_PATH_ENV, raising=False)
    monkeypatch.delenv(config.OPENAI_API_KEY_ENV, raising=False)
    assert config.trace_path() is None
    assert config.openai_api_key() is None

    monkeypatch.setenv(config.CLAUDE_BRIDGE_TRACE_PATH_ENV, "/tmp/trace.jsonl")
    monkeypatch.setenv(config.OPENAI_API_KEY_ENV, "  sk-test-placeholder  ")
    assert config.trace_path() == "/tmp/trace.jsonl"
    assert config.openai_api_key() == "sk-test-placeholder"

    monkeypatch.setenv(config.CLAUDE_BRIDGE_TRACE_PATH_ENV, "")
    monkeypatch.setenv(config.OPENAI_API_KEY_ENV, "   ")
    assert config.trace_path() is None
    assert config.openai_api_key() is None


def test_reasoning_mode_default_and_lowercase_override(monkeypatch):
    """REASONING_MODE preserves the OpenAI import-time lowercase behavior."""
    import claude_bridge.config as config

    monkeypatch.delenv(config.REASONING_MODE_ENV, raising=False)
    assert config.reasoning_mode() == "passthrough"

    monkeypatch.setenv(config.REASONING_MODE_ENV, "DROP")
    assert config.reasoning_mode() == "drop"


def test_xai_model_default_and_env_override(monkeypatch):
    """XAI_MODEL defaults to grok-4.20 and trims/normalizes blank overrides."""
    import claude_bridge.config as config

    monkeypatch.delenv(config.XAI_MODEL_ENV, raising=False)
    assert config.xai_model() == "grok-4.20"

    monkeypatch.setenv(config.XAI_MODEL_ENV, "grok-3-mini")
    assert config.xai_model() == "grok-3-mini"

    monkeypatch.setenv(config.XAI_MODEL_ENV, "  grok-4.20  ")
    assert config.xai_model() == "grok-4.20"

    monkeypatch.setenv(config.XAI_MODEL_ENV, "   ")
    assert config.xai_model() == "grok-4.20"


def test_xai_client_version_env_override_wins_verbatim(monkeypatch, tmp_path):
    """An explicit XAI_CLIENT_VERSION override is returned verbatim, ignoring bundles."""
    import claude_bridge.config as config

    (tmp_path / "grok-0.2.93-macos-aarch64").mkdir()
    monkeypatch.setenv(config.XAI_CLIENT_VERSION_ENV, "9.9.9")
    assert config.xai_client_version(downloads_dir=tmp_path) == "9.9.9"


def test_xai_client_version_env_override_respected_below_floor(monkeypatch, tmp_path):
    """An explicit override is honored verbatim even below the proxy floor."""
    import claude_bridge.config as config

    monkeypatch.setenv(config.XAI_CLIENT_VERSION_ENV, "0.0.1")
    assert config.xai_client_version(downloads_dir=tmp_path) == "0.0.1"


def test_xai_client_version_missing_dir_returns_floor(monkeypatch, tmp_path):
    """No downloads directory falls back to the proxy minimum floor."""
    import claude_bridge.config as config

    monkeypatch.delenv(config.XAI_CLIENT_VERSION_ENV, raising=False)
    assert config.xai_client_version(downloads_dir=tmp_path / "absent") == "0.1.202"


def test_xai_client_version_empty_dir_returns_floor(monkeypatch, tmp_path):
    """An empty downloads directory falls back to the floor."""
    import claude_bridge.config as config

    monkeypatch.delenv(config.XAI_CLIENT_VERSION_ENV, raising=False)
    assert config.xai_client_version(downloads_dir=tmp_path) == "0.1.202"


def test_xai_client_version_picks_highest_bundle(monkeypatch, tmp_path):
    """The highest installed bundle version wins over lower ones."""
    import claude_bridge.config as config

    for name in (
        "grok-0.2.82-macos-aarch64",
        "grok-0.2.93-macos-aarch64",
        "grok-0.1.500-macos-aarch64",
    ):
        (tmp_path / name).mkdir()
    monkeypatch.delenv(config.XAI_CLIENT_VERSION_ENV, raising=False)
    assert config.xai_client_version(downloads_dir=tmp_path) == "0.2.93"


def test_xai_client_version_orders_numerically_not_lexically(monkeypatch, tmp_path):
    """0.2.100 outranks 0.2.9 numerically; a lexical string sort would invert this."""
    import claude_bridge.config as config

    (tmp_path / "grok-0.2.9-macos-aarch64").mkdir()
    (tmp_path / "grok-0.2.100-macos-aarch64").mkdir()
    monkeypatch.delenv(config.XAI_CLIENT_VERSION_ENV, raising=False)
    assert config.xai_client_version(downloads_dir=tmp_path) == "0.2.100"


def test_xai_client_version_floors_installed_below_minimum(monkeypatch, tmp_path):
    """A single installed bundle below the floor still resolves to the floor (avoids HTTP 426)."""
    import claude_bridge.config as config

    (tmp_path / "grok-0.1.100-macos-aarch64").mkdir()
    monkeypatch.delenv(config.XAI_CLIENT_VERSION_ENV, raising=False)
    assert config.xai_client_version(downloads_dir=tmp_path) == "0.1.202"


def test_xai_client_version_ignores_non_version_entries(monkeypatch, tmp_path):
    """The unversioned symlink target and stray files never parse as a version."""
    import claude_bridge.config as config

    (tmp_path / "grok-macos-aarch64").mkdir()
    (tmp_path / "grok-notaversion-macos").mkdir()
    (tmp_path / "grok-0.2.93-macos-aarch64").mkdir()
    (tmp_path / "README.txt").write_text("x")
    monkeypatch.delenv(config.XAI_CLIENT_VERSION_ENV, raising=False)
    assert config.xai_client_version(downloads_dir=tmp_path) == "0.2.93"


def test_xai_client_version_all_non_version_entries_returns_floor(monkeypatch, tmp_path):
    """A directory with only unversioned entries falls back to the floor."""
    import claude_bridge.config as config

    (tmp_path / "grok-macos-aarch64").mkdir()
    (tmp_path / "other-file").write_text("x")
    monkeypatch.delenv(config.XAI_CLIENT_VERSION_ENV, raising=False)
    assert config.xai_client_version(downloads_dir=tmp_path) == "0.1.202"


def test_xai_client_version_blank_env_falls_through_to_bundle(monkeypatch, tmp_path):
    """A blank override is not treated as a version; bundle resolution proceeds."""
    import claude_bridge.config as config

    (tmp_path / "grok-0.2.93-macos-aarch64").mkdir()
    monkeypatch.setenv(config.XAI_CLIENT_VERSION_ENV, "   ")
    assert config.xai_client_version(downloads_dir=tmp_path) == "0.2.93"
