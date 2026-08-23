# Resource Ownership

## Ownership

| Resource | Area | Owner | Consumers | Enforcement | Status | Superseded By |
|-|-|-|-|-|-|-|
| PROVIDERS | provider | provider.py | proxy.py, __main__.py | Declared once in provider.py; each provider self-registers its own key at import; the proxy dispatches by lookup, never by naming a concrete provider | active |  |
| bridge_stats | observability | stats.py (BridgeStats) | proxy.py, __main__.py | Mutated only through record_* methods under the stats lock | active |  |
| config_values | config | config.py | every module, via config.get_* accessors | os.environ is read only inside config.py; no other module reads env (Boundary Rule 4) | active |  |
| credential_file | auth | providers/xai/auth.py | none — the file is private to the xAI provider | Refresh rewrites only the selected entry via mkstemp O_EXCL at mode 0600, atomic rename; never widens bits (D-XAI-003) | active |  |
| http2_client | transport | proxy.py (create_client / aclose) | providers via post_provider / open_stream / forward_request — they receive the client, never import it | http_client.py builds it (http2=True, split connect/idle Timeout, keepalive Limits); proxy sets owns_client = http_client_instance is None and is the only site that closes it — an injected client is closed by its injector, never by the proxy (D-STRUCT-002) | active |  |
| listen_address | network | proxy.py | __main__.py passes the host/port | Binds 127.0.0.1 by default; never 0.0.0.0 unless the operator sets an explicit host | active |  |
| logging_config | observability | log.py (configure_logging) | every module, via get_logger | Configured once at startup; helpers redact — the auth mode name is logged, never a secret value | active |  |
| openai_reasoning_cache | reasoning | providers/openai/provider.py | same provider instance only | Separate per-instance cache; the two providers never share reasoning state (independence contract) | active |  |
| prompt_cache_key | caching | providers/xai/provider.py, providers/openai/provider.py (each __init__) | the same provider instance's request builder | Minted once as str(uuid.uuid4()) per provider instance, never mutated; sent as the prompt_cache_key body field and, for xAI, the x-grok-conv-id header; never logged, embeds no secret (INV-SEC-01, INV-SEC-06) | active |  |
| router_state | routing | router.py (Router) | proxy.py | State transitions only through record_success / record_failure; consumers read state | active |  |
| xai_reasoning_cache | reasoning | providers/xai/provider.py | same provider instance only | Per-instance dict, threading.Lock-guarded, LRU-evicted at a fixed bound; in-memory only — never logged or persisted | active |  |
