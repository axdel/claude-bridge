# Resource Ownership

> Every shared or mutable resource has exactly one writer; everyone else reads through
> a contract. A second writer to any row below is a single-writer violation. This file
> carries a single table; every row below the header is a resource.

| Resource | Owner (single writer) | Consumers (read-only) | Enforcement |
|-|-|-|-|
| Environment configuration values | `config.py` | every module, via `config.get_*()` accessors | `os.environ` is read only inside `config.py`; no other module reads env (Boundary Rule 4) |
| The `PROVIDERS` registry (dispatch table + contract) | `provider.py` | `proxy.py`, `__main__.py` | Declared once in `provider.py`; each provider self-registers its own key at import; the proxy dispatches by lookup, never by naming a concrete provider |
| The grok credential file `~/.grok/auth.json` | `providers/xai.py` | none — the file is private to the xAI provider | Refresh rewrites only the selected entry via `mkstemp` O_EXCL at mode 0600, atomic rename; never widens bits (D-XAI-003) |
| The xAI reasoning cache (`_reasoning_by_call_id`) | `providers/xai.py` | same provider instance only | Per-instance dict, `threading.Lock`-guarded, LRU-evicted at a fixed bound; in-memory only — never logged or persisted |
| The OpenAI reasoning cache | `providers/openai.py` | same provider instance only | Separate per-instance cache; the two providers never share reasoning state (contract 1) |
| Router failover state (circuit `CLOSED`/`OPEN`) | `router.py` (`Router`) | `proxy.py` | State transitions only through `record_success` / `record_failure`; consumers read `state` |
| Bridge usage counters (requests, tokens, failovers) | `stats.py` (`BridgeStats`) | `proxy.py`, `__main__.py` | Mutated only through `record_*` methods under the stats lock |
| The listening socket and bind address | `proxy.py` | `__main__.py` passes the host/port | Binds `127.0.0.1` by default; never `0.0.0.0` unless the operator sets an explicit host |
| Logging configuration | `log.py` (`configure_logging`) | every module, via `get_logger` | Configured once at startup; helpers redact — the auth mode name is logged, never a secret value |
