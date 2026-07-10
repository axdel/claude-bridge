# Boundary Map

> The allowed import directions for `claude_bridge`. Dependencies flow one way:
> entry → orchestration → provider abstraction → leaf utilities. Concrete providers
> are self-contained and mutually independent. Enforced mechanically by import-linter
> (`[tool.importlinter]` in `pyproject.toml`); the boundary-map primitive gate runs
> `lint-imports` and is the merge blocker — this file is the human-readable statement
> of the same contract.

## Dependency DAG

```
__main__            (CLI entry — the composition root; wires a provider into the proxy)
   │
   ├── proxy        (HTTP server: binds 127.0.0.1, dispatches via the Provider protocol)
   ├── router       (circuit-breaker failover state: Anthropic ↔ fallback provider)
   │      │
   │      ▼
   │   provider     (abstraction: the Provider protocol + PROVIDERS registry + ProviderCapabilities)
   │      ▲
   │      │  (concrete providers register into PROVIDERS; the proxy never names them)
   │   providers/openai   providers/xai      (self-contained, mutually independent)
   │      │                   │
   ▼      ▼                   ▼
leaf utilities:  config · content · auth · stream · stats · log
   (pure helpers — never import orchestration or a concrete provider)
```

Arrows point from consumer to dependency; there is no reverse edge. A leaf utility
that imported `proxy` or a concrete provider would invert the DAG.

## Import Contracts

These three contracts are the exact `[tool.importlinter]` contracts in `pyproject.toml`;
`lint-imports` enforces them and the boundary-map gate reports its verdict.

| # | Contract | Type | Rule |
|-|-|-|-|
| 1 | Providers are mutually independent | independence | `providers.openai` and `providers.xai` never import each other — each provider is a self-contained translation path (D-XAI-002) |
| 2 | Proxy dispatches via the abstraction | forbidden | `proxy` must not import `providers.openai` or `providers.xai` — it dispatches only through the `Provider` protocol resolved from the `PROVIDERS` registry |
| 3 | Leaf utilities never import orchestration | forbidden | `config`, `content`, `auth`, `stream`, `provider`, `stats`, `log` must not import `proxy`, `router`, `__main__`, or either concrete provider |

## Layer Purity

| Layer | Modules | Owns | Must NOT import |
|-|-|-|-|
| Entry | `__main__` | Arg parsing, provider selection, proxy startup | — (top of the DAG) |
| Orchestration | `proxy`, `router` | HTTP serving, request dispatch, failover state | concrete providers (`providers.openai`, `providers.xai`) |
| Abstraction | `provider` | `Provider` protocol, `PROVIDERS` registry, `ProviderCapabilities` | orchestration, concrete providers |
| Concrete providers | `providers.openai`, `providers.xai` | One provider's full Anthropic↔upstream translation | the other provider |
| Leaf utilities | `config`, `content`, `auth`, `stream`, `stats`, `log` | Pure, reusable helpers | orchestration, concrete providers |

## Error Ownership

| Layer | Raises | Translates |
|-|-|-|
| Leaf utilities (`auth`, `content`) | Value/credential errors on malformed input (bad bearer, unreadable `auth.json`) | raw I/O errors into typed local errors |
| Concrete providers | Upstream/translation errors | HTTP/transport errors into Anthropic-shaped error responses |
| Orchestration (`proxy`, `router`) | HTTP status responses; opens the circuit on repeated failure | provider errors into client responses |

## Enforcement

Mechanical: `lint-imports` (import-linter, `cli_name = lint-imports`) run by the
boundary-map primitive gate. The `claude_bridge` package is editable-installed
(`[build-system]` → `uv sync`) so grimp can resolve `root_package = "claude_bridge"`
on `sys.path`. A new module joins the DAG by being placed in one of the layers above;
a new provider registers into `PROVIDERS` and stays independent of every sibling.
