"""Runtime configuration owner for Claude Bridge.

Centralizes environment variable names, defaults, and small stdlib-only accessors.
Shell-launcher environment handling remains owned by the launcher scripts.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from pathlib import Path

OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
REASONING_MODE_ENV = "REASONING_MODE"
LOG_LEVEL_ENV = "LOG_LEVEL"
UPSTREAM_TIMEOUT_ENV = "UPSTREAM_TIMEOUT"
MAX_REQUEST_BODY_ENV = "MAX_REQUEST_BODY"
LLM_BRIDGE_FALLBACK_ENV = "LLM_BRIDGE_FALLBACK"
ANTHROPIC_REAL_URL_ENV = "ANTHROPIC_REAL_URL"
CLAUDE_BRIDGE_TRACE_PATH_ENV = "CLAUDE_BRIDGE_TRACE_PATH"
XAI_MODEL_ENV = "XAI_MODEL"
XAI_CLIENT_VERSION_ENV = "XAI_CLIENT_VERSION"

DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_ANTHROPIC_REAL_URL = "https://api.anthropic.com"
DEFAULT_MAX_REQUEST_BODY = 10_485_760
DEFAULT_FALLBACK_CHAIN = ("openai",)
DEFAULT_REASONING_MODE = "passthrough"
# Pinned to Grok 4.6 (verified accepted by cli-chat-proxy.grok.com's /v1/responses), not the
# rolling `grok-build` alias. Pinning trades the alias's auto-tracking of upstream releases for
# a known model version; if the id is later deprecated (as `grok-4.20` was), reverting to
# `grok-build` is a one-line change. Supersedes D-XAI-008. Override per run with XAI_MODEL.
DEFAULT_XAI_MODEL = "grok-4.6"

# cli-chat-proxy.grok.com answers HTTP 426 below this x-grok-client-version; the
# resolver never sends a header older than this floor. Bundle dir layout:
# ``~/.grok/downloads/grok-<ver>-<platform>`` (the unversioned symlink target
# ``grok-<platform>`` must not parse as a version).
_XAI_CLIENT_VERSION_FLOOR = "0.1.202"
_GROK_DOWNLOADS_DIR = Path.home() / ".grok" / "downloads"
_GROK_BUNDLE_VERSION_RE = re.compile(r"^grok-(\d+\.\d+\.\d+)-")


def _non_empty_stripped_env(name: str) -> str | None:
    """Return a stripped env value, normalizing missing and blank values to None."""
    value = os.environ.get(name, "").strip()
    return value or None


def openai_api_key() -> str | None:
    """Return the configured OpenAI API key, or None when unset or blank."""
    return _non_empty_stripped_env(OPENAI_API_KEY_ENV)


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
        if on_invalid is not None:
            on_invalid(raw)
        return default
    return value


def max_request_body(
    *,
    on_invalid: Callable[[str], None] | None = None,
) -> int:
    """Return the positive request body limit in bytes, or the default when invalid."""
    raw = os.environ.get(MAX_REQUEST_BODY_ENV)
    if raw is None:
        return DEFAULT_MAX_REQUEST_BODY
    try:
        value = int(raw)
    except (ValueError, TypeError):
        if on_invalid is not None:
            on_invalid(raw)
        return DEFAULT_MAX_REQUEST_BODY
    if value <= 0:
        if on_invalid is not None:
            on_invalid(raw)
        return DEFAULT_MAX_REQUEST_BODY
    return value


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


def xai_model() -> str:
    """Return the xAI Grok model id from XAI_MODEL, defaulting to grok-4.6."""
    return _non_empty_stripped_env(XAI_MODEL_ENV) or DEFAULT_XAI_MODEL


def _version_tuple(version: str) -> tuple[int, ...]:
    """Split a dotted numeric version into an int tuple for correct ordering."""
    return tuple(int(part) for part in version.split("."))


def _installed_grok_versions(downloads_dir: Path) -> list[str]:
    """Return version strings parsed from ``~/.grok/downloads/grok-<ver>-*`` bundles."""
    if not downloads_dir.is_dir():
        return []
    versions: list[str] = []
    for entry in downloads_dir.iterdir():
        match = _GROK_BUNDLE_VERSION_RE.match(entry.name)
        if match:
            versions.append(match.group(1))
    return versions


def xai_client_version(downloads_dir: Path | None = None) -> str:
    """Resolve the ``x-grok-client-version`` header value.

    Precedence: an explicit ``XAI_CLIENT_VERSION`` override wins (verbatim);
    otherwise the highest installed grok CLI bundle version, floored at the
    proxy's minimum (below which cli-chat-proxy answers HTTP 426); otherwise the
    floor itself. Self-healing: a newer grok CLI bumps the header automatically.
    """
    override = _non_empty_stripped_env(XAI_CLIENT_VERSION_ENV)
    if override:
        return override
    resolved = _XAI_CLIENT_VERSION_FLOOR
    for version in _installed_grok_versions(downloads_dir or _GROK_DOWNLOADS_DIR):
        if _version_tuple(version) > _version_tuple(resolved):
            resolved = version
    return resolved
