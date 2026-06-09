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

    monkeypatch.setenv(config.LLM_BRIDGE_FALLBACK_ENV, "gemini, openai,,xai")
    assert config.fallback_chain() == ["gemini", "openai", "xai"]

    monkeypatch.setenv(config.LLM_BRIDGE_FALLBACK_ENV, "")
    assert config.fallback_chain() == []


def test_optional_path_and_api_key_accessors_trim_empty_values(monkeypatch):
    """Trace path and provider API keys normalize missing/blank values to None."""
    import claude_bridge.config as config

    monkeypatch.delenv(config.CLAUDE_BRIDGE_TRACE_PATH_ENV, raising=False)
    monkeypatch.delenv(config.OPENAI_API_KEY_ENV, raising=False)
    monkeypatch.delenv(config.GEMINI_API_KEY_ENV, raising=False)
    assert config.trace_path() is None
    assert config.openai_api_key() is None
    assert config.gemini_api_key() is None

    monkeypatch.setenv(config.CLAUDE_BRIDGE_TRACE_PATH_ENV, "/tmp/trace.jsonl")
    monkeypatch.setenv(config.OPENAI_API_KEY_ENV, "  sk-test-placeholder  ")
    monkeypatch.setenv(config.GEMINI_API_KEY_ENV, "  gemini-test-placeholder  ")
    assert config.trace_path() == "/tmp/trace.jsonl"
    assert config.openai_api_key() == "sk-test-placeholder"
    assert config.gemini_api_key() == "gemini-test-placeholder"

    monkeypatch.setenv(config.CLAUDE_BRIDGE_TRACE_PATH_ENV, "")
    monkeypatch.setenv(config.OPENAI_API_KEY_ENV, "   ")
    monkeypatch.setenv(config.GEMINI_API_KEY_ENV, "   ")
    assert config.trace_path() is None
    assert config.openai_api_key() is None
    assert config.gemini_api_key() is None


def test_reasoning_mode_default_and_lowercase_override(monkeypatch):
    """REASONING_MODE preserves the OpenAI import-time lowercase behavior."""
    import claude_bridge.config as config

    monkeypatch.delenv(config.REASONING_MODE_ENV, raising=False)
    assert config.reasoning_mode() == "passthrough"

    monkeypatch.setenv(config.REASONING_MODE_ENV, "DROP")
    assert config.reasoning_mode() == "drop"


def test_gemini_model_defaults_share_one_override(monkeypatch):
    """Gemini API-key and OAuth model defaults are explicit and share GEMINI_MODEL."""
    import claude_bridge.config as config

    monkeypatch.delenv(config.GEMINI_MODEL_ENV, raising=False)
    assert config.gemini_api_key_model() == "gemini-2.5-pro"
    assert config.gemini_oauth_model() == "gemini-3-flash-preview"

    monkeypatch.setenv(config.GEMINI_MODEL_ENV, "gemini-3.1-pro-preview")
    assert config.gemini_api_key_model() == "gemini-3.1-pro-preview"
    assert config.gemini_oauth_model() == "gemini-3.1-pro-preview"
