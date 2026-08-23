# Canonical Glossary

## Concepts

| Canonical Name | Domain | Concept | Rejected Aliases | Notes | Status | Superseded By |
|-|-|-|-|-|-|-|
| PROVIDERS | provider | The registry mapping a provider name to its `Provider` class | provider_map, registry_dict, handlers | `PROVIDERS: dict[str, type[Provider]]` in `provider.py`; the single dispatch table | active |  |
| Provider | provider | The translation adapter that converts Anthropic Messages ↔ an upstream API | adapter, backend, engine, connector | The `Provider` protocol in `provider.py`; each concrete one lives in `providers/` | active |  |
| ProviderCapabilities | provider | The per-provider feature/flag descriptor | features, provider_config, options | Frozen descriptor in `provider.py` (image/document input, tool round-trip) | active |  |
| Router | routing | The circuit-breaker that fails Anthropic traffic over to a fallback | failover_controller, load_balancer, switch | `Router` in `router.py`; `CLOSED`/`OPEN` state | active |  |
| auth_mode | auth | The name of the credential mechanism, logged in place of any secret value | auth_type, credential_kind, scheme | Only the mode name (`grok_oauth`) is ever logged; never the bearer or refresh-token value | active |  |
| bearer | auth | The opaque subscription JWT sent upstream | token, api_key, access_token, secret | Rides only in the `Authorization` header; validated as RFC 7235 token68 before use | active |  |
| bridge | proxy | The running proxy process that fronts Claude Code | proxy-server, gateway, shim | The product itself; `proxy.py` serves it, bound to loopback | active |  |
| credential_file | auth | The grok CLI credential file reused for auth | keyfile, token_store, secrets_file | `~/.grok/auth.json`; read-only except on refresh | active |  |
| fallback_provider | routing | The provider used when Anthropic is unavailable | secondary, backup_model, standby | Selected by the `Router` on an open circuit | active |  |
| media_source | media | A parsed base64 image or document input | attachment, blob, file_input | `content.parse_media_source`; base64 payload never leaks into text, warnings, or logs | active |  |
| prompt_cache_key | caching | The sticky per-instance identity that pins a session's requests to the upstream provider's cached prompt prefix | conv_id, session_key, cache_token, instruction_hash | A process-stable `str(uuid.uuid4())` minted once per provider instance (D-CACHE-002); sent as the `prompt_cache_key` body field and, for xAI, the `x-grok-conv-id` header. Never logged, never derived from prompt content, embeds no secret (INV-SEC-01, INV-SEC-06) | active |  |
| reasoning_cache | reasoning | The in-memory map from Anthropic call id to encrypted reasoning | reasoning_store, thought_cache, memo | Per-provider, LRU-bounded, lock-guarded; stays in memory, never logged or persisted | active |  |
| reasoning_effort | reasoning | The model-gated depth hint sent to grok-4.6+ that trades latency for reasoning quality | effort, thinking_level, reasoning_depth | Config-owned via `XAI_REASONING_EFFORT` (default low), sent as `reasoning.effort`; `_model_accepts_reasoning_effort` gates on provably-old, so it reaches grok-4.6+ and rolling aliases but is omitted for pre-4.6 models that 400 on the field (D-XAI-010) | active |  |
| refresh | auth | OIDC token renewal that rewrites the credential file | reauth, token_renewal, rotate | Rewrites only the selected entry, atomically at mode 0600 (D-XAI-003) | active |  |
| stats | observability | The running usage/latency/error counters | metrics, telemetry, tally | `BridgeStats` in `stats.py`; request/response/token/failover counts | active |  |
| token_expiry | auth | The bearer's expiry instant, read from the JWT exp claim rather than persisted under a project field | expires_at, ttl, valid_until, expiry_date | `decode_jwt_exp` reads it from the token; the bridge persists no timestamped records of its own, so no `_at`/`_date` convention applies. Drives proactive refresh | active |  |

## Track Prefixes

| Track | Branch Prefix | Notes | Status | Superseded By |
|-|-|-|-|-|

## State Locations

| State | Path Pattern | Status | Superseded By |
|-|-|-|-|
