# Derivation Map

## Derivations

| Artifact | Source of Truth | Derives From | Regeneration | Status | Superseded By |
|-|-|-|-|-|-|
| editable_install | pyproject.toml [build-system] + [tool.hatch.build.targets.wheel] — the editable install of claude_bridge in .venv makes the package importable to venv tooling | pyproject.toml | uv sync | active |  |
| uv.lock | pyproject.toml dependency + build declarations | pyproject.toml | uv lock | active |  |
| x-grok-client-version | the local grok CLI installation (grok downloads dir) — the highest installed grok CLI bundle version, floored at the proxy minimum | the installed grok CLI bundle | Dynamic — resolved per request by config.xai_client_version(); self-healing when the grok CLI updates; XAI_CLIENT_VERSION overrides verbatim | active |  |
