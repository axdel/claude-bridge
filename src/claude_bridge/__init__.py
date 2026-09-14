"""Claude Bridge — use your Claude Code setup with any LLM provider.

An async proxy that sits between Claude Code and LLM providers. Its one runtime
dependency is ``httpx[http2]`` (the HTTP/2 data plane); the rest is Python stdlib.
Intercepts Anthropic Messages API traffic and can either pass it through to
the real Anthropic endpoint or translate it to another provider's format
(e.g., OpenAI Responses API).

Architecture::

    Claude Code  ->  proxy.py  ->  Anthropic (passthrough)
                        |
                    router.py (circuit breaker)
                        |
                    provider.py (protocol)
                        |
          providers/openai/   providers/xai/   (sub-packages)

Adding a new provider: create a ``providers/<name>/`` sub-package (see
``providers/openai/`` for the reference layout) and follow the extension steps in
``provider.py``. Once registered, select it with ``LLM_BRIDGE_FALLBACK=<name>`` or
``--provider <name>``.
"""

import importlib.metadata

# Derived from the distribution metadata, which the build takes from pyproject.toml --
# the single owner of the version. Restating the literal here made this a second writer:
# releases bump pyproject and nothing bumped this file, so each one needed a manual
# re-sync commit, and the release that missed it shipped a launcher banner naming the
# previous version while running the new one. The banner is the operator's only
# in-terminal statement of which bridge is on the wire, so a stale one is a lie.
#
# The fallback is a sentinel, not a version: importing from a source tree with no install
# has no version to report, and inventing a plausible number there would reintroduce the
# very drift this removes. stdlib only -- httpx stays the one runtime dependency.
try:
    __version__ = importlib.metadata.version("claude-bridge")
except importlib.metadata.PackageNotFoundError:  # source tree, not installed
    __version__ = "0.0.0+unknown"
