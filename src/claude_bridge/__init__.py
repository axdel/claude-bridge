"""Claude Bridge — use your Claude Code setup with any LLM provider.

A stdlib-only async proxy that sits between Claude Code and LLM providers.
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
          providers/openai.py / providers/gemini.py

Adding a new provider:
    1. Create ``providers/<name>.py`` implementing the ``Provider`` protocol
    2. Declare provider ``capabilities`` for stream and sync response behavior
    3. Register only implemented providers: ``PROVIDERS["<name>"] = YourProvider``
    4. Import it in ``__main__.py`` so it auto-registers
    5. Set ``LLM_BRIDGE_FALLBACK=<name>`` or ``--provider <name>``
"""

__version__ = "0.7.0"
