# Canonical Glossary

> One name per domain concept, used identically across `__main__`, `config`, the
> `Provider` abstraction, both concrete providers, the launchers, tests, and docs.
> Before introducing a name, search this table; a concept that already exists under a
> different name is reused, never re-encoded. Column 2 is the canonical name in bold;
> column 3 lists rejected aliases that must not reappear in code.

## Core Domain

| Concept | Canonical Name | NOT (rejected aliases) | Notes |
|-|-|-|-|
| The translation adapter that converts Anthropic Messages ↔ an upstream API | **Provider** | adapter, backend, engine, connector | The `Provider` protocol in `provider.py`; each concrete one lives in `providers/` |
| The running proxy process that fronts Claude Code | **bridge** | proxy-server, gateway, shim | The product itself; `proxy.py` serves it, bound to loopback |
| The registry mapping a provider name to its `Provider` class | **PROVIDERS** | provider map, registry dict, handlers | `PROVIDERS: dict[str, type[Provider]]` in `provider.py`; the single dispatch table |
| The per-provider feature/flag descriptor | **ProviderCapabilities** | features, provider config, options | Frozen descriptor in `provider.py` (image/document input, tool round-trip) |
| The opaque subscription JWT sent upstream | **bearer** | token, api key, access token, secret | Rides only in the `Authorization` header; validated as RFC 7235 token68 before use |
| The grok CLI credential file reused for auth | **credential file** | keyfile, token store, secrets file | `~/.grok/auth.json`; read-only except on refresh |
| OIDC token renewal that rewrites the credential file | **refresh** | reauth, token renewal, rotate | Rewrites only the selected entry, atomically at mode 0600 (D-XAI-003) |
| The name of the credential mechanism, logged in place of any secret value | **auth mode** | auth type, credential kind, scheme | Only the mode name (`grok_oauth`) is ever logged; never the bearer or refresh-token value |
| The in-memory map from Anthropic call id to encrypted reasoning | **reasoning cache** | reasoning store, thought cache, memo | Per-provider, LRU-bounded, lock-guarded; stays in memory, never logged or persisted |
| The circuit-breaker that fails Anthropic traffic over to a fallback | **Router** | failover controller, load balancer, switch | `Router` in `router.py`; `CLOSED`/`OPEN` state |
| The provider used when Anthropic is unavailable | **fallback provider** | secondary, backup model, standby | Selected by the `Router` on an open circuit |
| The running usage/latency/error counters | **stats** | metrics, telemetry, tally | `BridgeStats` in `stats.py`; request/response/token/failover counts |
| A parsed base64 image or document input | **media source** | attachment, blob, file input | `content.parse_media_source`; base64 payload never leaks into text, warnings, or logs |

## Temporal Conventions

This project persists no timestamped records of its own; the only durable state it
writes is the credential file's refreshed token, whose expiry is read from the JWT
`exp` claim (`decode_jwt_exp`), not stored under a project field. No `_at`/`_date`
convention applies.
