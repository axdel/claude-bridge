"""LLM provider adapters — auth and request/response translation.

Each module in this package implements the ``Provider`` protocol from
``claude_bridge.provider`` and registers itself in the ``PROVIDERS`` dict
on import. The proxy never imports from here directly — registration
is triggered by ``__main__.py`` importing each provider module.
"""
