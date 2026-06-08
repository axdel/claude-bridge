"""Runtime configuration owner for Claude Bridge.

Centralizes environment variable names, defaults, and small stdlib-only accessors.
Shell-launcher environment handling remains owned by the launcher scripts.
"""

from __future__ import annotations

import os
from collections.abc import Callable

OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
GEMINI_MODEL_ENV = "GEMINI_MODEL"
REASONING_MODE_ENV = "REASONING_MODE"
LOG_LEVEL_ENV = "LOG_LEVEL"
UPSTREAM_TIMEOUT_ENV = "UPSTREAM_TIMEOUT"
MAX_REQUEST_BODY_ENV = "MAX_REQUEST_BODY"
LLM_BRIDGE_FALLBACK_ENV = "LLM_BRIDGE_FALLBACK"
ANTHROPIC_REAL_URL_ENV = "ANTHROPIC_REAL_URL"
CLAUDE_BRIDGE_TRACE_PATH_ENV = "CLAUDE_BRIDGE_TRACE_PATH"

DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_ANTHROPIC_REAL_URL = "https://api.anthropic.com"
DEFAULT_MAX_REQUEST_BODY = 10_485_760
DEFAULT_FALLBACK_CHAIN = ("openai",)
DEFAULT_REASONING_MODE = "passthrough"
GEMINI_API_KEY_DEFAULT_MODEL = "gemini-2.5-pro"
GEMINI_OAUTH_DEFAULT_MODEL = "gemini-3-flash-preview"


def _non_empty_stripped_env(name: str) -> str | None:
    """Return a stripped env value, normalizing missing and blank values to None."""
    value = os.environ.get(name, "").strip()
    return value or None


def openai_api_key() -> str | None:
    """Return the configured OpenAI API key, or None when unset or blank."""
    return _non_empty_stripped_env(OPENAI_API_KEY_ENV)


def gemini_api_key() -> str | None:
    """Return the configured Gemini API key, or None when unset or blank."""
    return _non_empty_stripped_env(GEMINI_API_KEY_ENV)


def reasoning_mode() -> str:
    """Return the OpenAI reasoning mode, lowercased like the legacy provider read."""
    return os.environ.get(REASONING_MODE_ENV, DEFAULT_REASONING_MODE).lower()


def log_level(explicit_level: str | None = None) -> str:
    """Return the explicit log level or the LOG_LEVEL env value with INFO fallback."""
    return explicit_level or os.environ.get(LOG_LEVEL_ENV, DEFAULT_LOG_LEVEL)


def upstream_timeout(
    default: int,
    *,
    on_invalid: Callable[[str], None] | None = None,
) -> int:
    """Return the positive UPSTREAM_TIMEOUT override or the caller's default."""
    raw = os.environ.get(UPSTREAM_TIMEOUT_ENV)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (ValueError, TypeError):
        if on_invalid is not None:
            on_invalid(raw)
        return default
    if value <= 0:
        return default
    return value


def max_request_body() -> int:
    """Return the import-time request body limit in bytes."""
    return int(os.environ.get(MAX_REQUEST_BODY_ENV, DEFAULT_MAX_REQUEST_BODY))


def fallback_chain() -> list[str]:
    """Return the ordered fallback provider names from LLM_BRIDGE_FALLBACK."""
    raw = os.environ.get(LLM_BRIDGE_FALLBACK_ENV)
    if raw is None:
        return list(DEFAULT_FALLBACK_CHAIN)
    return [name.strip() for name in raw.split(",") if name.strip()]


def anthropic_real_url() -> str:
    """Return the passthrough Anthropic upstream URL."""
    return os.environ.get(ANTHROPIC_REAL_URL_ENV, DEFAULT_ANTHROPIC_REAL_URL)


def trace_path() -> str | None:
    """Return the redacted structural trace path, or None when tracing is disabled."""
    return os.environ.get(CLAUDE_BRIDGE_TRACE_PATH_ENV) or None


def gemini_api_key_model() -> str:
    """Return the Gemini model for API-key mode, honoring GEMINI_MODEL override."""
    return os.environ.get(GEMINI_MODEL_ENV, GEMINI_API_KEY_DEFAULT_MODEL)


def gemini_oauth_model() -> str:
    """Return the Gemini model for OAuth mode, honoring GEMINI_MODEL override."""
    return os.environ.get(GEMINI_MODEL_ENV, GEMINI_OAUTH_DEFAULT_MODEL)
