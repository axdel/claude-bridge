# Derivation Map

> Every derived artifact has exactly one source of truth and a one-way regeneration.
> A derived artifact is never hand-edited; it is regenerated from its source. This
> stdlib proxy generates no SDK or OpenAPI spec, so the chain is short — the honest
> set of mechanical derivations, not a padded one.

This file carries a single table; every row below the header is a derivation.

| Artifact | Source of Truth | Derives From | Regeneration |
|-|-|-|-|
| `uv.lock` | `pyproject.toml` dependency + build declarations | `pyproject.toml` | `uv lock` |
| Editable install of `claude_bridge` in `.venv` (makes the package importable to venv tooling) | `pyproject.toml` `[build-system]` + `[tool.hatch.build.targets.wheel]` | `pyproject.toml` | `uv sync` |
| The `x-grok-client-version` request header value | The highest installed grok CLI bundle version, floored at the proxy minimum | The local grok CLI installation (grok downloads dir) | Dynamic — resolved per request by `config.xai_client_version()`; self-healing when the grok CLI updates; `XAI_CLIENT_VERSION` overrides verbatim |
