"""LLM provider adapters — auth and request/response translation.

Implemented provider modules declare the ``Provider`` protocol from
``claude_bridge.provider`` and register themselves in the ``PROVIDERS`` dict
on import. Placeholder modules stay unregistered until implemented. The proxy
never imports from here directly — registration is triggered by ``__main__.py``
importing each implemented provider module.
"""
