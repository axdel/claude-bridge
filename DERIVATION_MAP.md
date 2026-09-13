# Derivation Map

## Derivations

| Artifact | Source of Truth | Derives From | Regeneration | Status | Superseded By |
|-|-|-|-|-|-|
| REASONING_SUMMARY_WIRE | the live OpenAI Responses streaming endpoint — a golden capture of the upstream reasoning-summary event shape, captured 2026-09-13 | the OpenAI Responses SSE contract | uv run python scripts/capture_reasoning_summary_wire.py — rebuild the fixture from its printed stream, never from recollection | active |  |
| editable_install | pyproject.toml [build-system] + [tool.hatch.build.targets.wheel] — the editable install of claude_bridge in .venv makes the package importable to venv tooling | pyproject.toml | uv sync | active |  |
| uv.lock | pyproject.toml dependency + build declarations | pyproject.toml | uv lock | active |  |
| x-grok-client-version | the local grok CLI installation (grok downloads dir) — the highest installed grok CLI bundle version, floored at the proxy minimum | the installed grok CLI bundle | Dynamic — resolved per request by config.xai_client_version(); self-healing when the grok CLI updates; XAI_CLIENT_VERSION overrides verbatim | active |  |
