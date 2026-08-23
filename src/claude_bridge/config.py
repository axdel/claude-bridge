"""Runtime configuration owner for Claude Bridge.

Centralizes environment variable names, defaults, and small stdlib-only accessors.
Shell-launcher environment handling remains owned by the launcher scripts.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Callable
from pathlib import Path

OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
REASONING_MODE_ENV = "REASONING_MODE"
LOG_LEVEL_ENV = "LOG_LEVEL"
MAX_REQUEST_BODY_ENV = "MAX_REQUEST_BODY"
LLM_BRIDGE_FALLBACK_ENV = "LLM_BRIDGE_FALLBACK"
ANTHROPIC_REAL_URL_ENV = "ANTHROPIC_REAL_URL"
CLAUDE_BRIDGE_TRACE_PATH_ENV = "CLAUDE_BRIDGE_TRACE_PATH"
XAI_MODEL_ENV = "XAI_MODEL"
XAI_CLIENT_VERSION_ENV = "XAI_CLIENT_VERSION"
XAI_REASONING_EFFORT_ENV = "XAI_REASONING_EFFORT"
CONNECT_TIMEOUT_ENV = "CONNECT_TIMEOUT"
STREAM_IDLE_TIMEOUT_ENV = "STREAM_IDLE_TIMEOUT"
POOL_IDLE_ENV = "POOL_IDLE"

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

# xAI reasoning effort. grok-4.6 accepts omit/low/medium/high (all HTTP 200); `low` is a
# latency choice (63 vs 146 reasoning tokens at omit), not native-high parity. Sent only to
# models that accept the field (see anthropic_to_xai's version gate). Override with
# XAI_REASONING_EFFORT.
DEFAULT_XAI_REASONING_EFFORT = "low"

# The reasoning-effort values grok-4.6 accepts (all verified HTTP 200). Canonical owner of the
# allowed set: xai_reasoning_effort() validates against it so a typo falls back to the default
# rather than shipping an unrecognized effort to every upstream request.
_XAI_REASONING_EFFORTS = ("low", "medium", "high")

# HTTP/2 transport split timeouts (seconds). A single urllib socket timeout fired on
# every recv and killed long-thinking grok-4.6 streams at ~120s; these separate
# connection setup from per-chunk stream idle, so a healthy stream runs as long as
# chunks keep arriving within DEFAULT_STREAM_IDLE_TIMEOUT. DEFAULT_POOL_IDLE overrides
# httpx's 5s keepalive_expiry (grok-build holds an idle connection ~90s).
DEFAULT_CONNECT_TIMEOUT = 10.0
DEFAULT_STREAM_IDLE_TIMEOUT = 300.0
DEFAULT_POOL_IDLE = 90.0

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


def _positive_env_number[NumT: (int, float)](
    name: str,
    default: NumT,
    *,
    cast: Callable[[str], NumT],
    on_invalid: Callable[[str], None] | None = None,
) -> NumT:
    """Return a positive numeric env override parsed by *cast*, else *default*.

    Shared validation for every positive-number setting: a missing var yields the
    default silently; a present-but-invalid value (unparseable, or not > 0) invokes
    *on_invalid* and falls back to the default.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = cast(raw)
    except (ValueError, TypeError):
        if on_invalid is not None:
            on_invalid(raw)
        return default
    # ``float("nan")``/``float("inf")`` parse cleanly but are not usable timeouts, and NaN
    # slips past ``value <= 0`` (every NaN comparison is False). Require a finite, positive
    # value so a non-finite override falls back to the default instead of building a
    # ``Timeout(connect=nan, ...)`` that never fires. ``math.isfinite`` accepts any int.
    if not math.isfinite(value) or value <= 0:
        if on_invalid is not None:
            on_invalid(raw)
        return default
    return value


def max_request_body(
    *,
    on_invalid: Callable[[str], None] | None = None,
) -> int:
    """Return the positive request body limit in bytes, or the default when invalid."""
    return _positive_env_number(
        MAX_REQUEST_BODY_ENV, DEFAULT_MAX_REQUEST_BODY, cast=int, on_invalid=on_invalid
    )


def connect_timeout(*, on_invalid: Callable[[str], None] | None = None) -> float:
    """Return the positive CONNECT_TIMEOUT override (seconds) or the default."""
    return _positive_env_number(
        CONNECT_TIMEOUT_ENV, DEFAULT_CONNECT_TIMEOUT, cast=float, on_invalid=on_invalid
    )


def stream_idle_timeout(*, on_invalid: Callable[[str], None] | None = None) -> float:
    """Return the positive STREAM_IDLE_TIMEOUT override (seconds) or the default.

    httpx's per-read (per-chunk) timeout on a streaming response — the gap a stream
    may sit idle *between* chunks, NOT a cap on total stream duration.
    """
    return _positive_env_number(
        STREAM_IDLE_TIMEOUT_ENV, DEFAULT_STREAM_IDLE_TIMEOUT, cast=float, on_invalid=on_invalid
    )


def pool_idle(*, on_invalid: Callable[[str], None] | None = None) -> float:
    """Return the positive POOL_IDLE override (seconds) or the default.

    Maps to httpx ``Limits(keepalive_expiry=...)``: how long an idle keep-alive
    connection is reused before being dropped. httpx defaults to 5s; upstreams like
    grok-build hold ~90s, so the default is raised to match — a 5s expiry would
    reconnect constantly and forfeit the HTTP/2 multiplex benefit.
    """
    return _positive_env_number(
        POOL_IDLE_ENV, DEFAULT_POOL_IDLE, cast=float, on_invalid=on_invalid
    )


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


def xai_reasoning_effort(*, on_invalid: Callable[[str], None] | None = None) -> str:
    """Return the validated xAI reasoning effort from XAI_REASONING_EFFORT, defaulting to low.

    grok-4.6 accepts low/medium/high; the value is only stamped onto requests for models
    that accept the field (the version gate lives in ``anthropic_to_xai``). ``low`` is the
    latency default, not xAI's native ``high``.

    The override is validated against ``_XAI_REASONING_EFFORTS`` (case-insensitively). An
    unrecognized value invokes *on_invalid* with the raw string and falls back to the
    default, so a typo like ``XAI_REASONING_EFFORT=hihg`` never ships ``{'effort': 'hihg'}``
    to every upstream request; a valid override is normalized to lowercase for the wire.
    """
    raw = _non_empty_stripped_env(XAI_REASONING_EFFORT_ENV)
    if raw is None:
        return DEFAULT_XAI_REASONING_EFFORT
    normalized = raw.lower()
    if normalized not in _XAI_REASONING_EFFORTS:
        if on_invalid is not None:
            on_invalid(raw)
        return DEFAULT_XAI_REASONING_EFFORT
    return normalized


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
