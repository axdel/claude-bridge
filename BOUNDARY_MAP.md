# Boundary Map

## Import Rules

| Module | Target | Rule | Notes | Status | Superseded By |
|-|-|-|-|-|-|
| concrete_provider | sibling_provider | must-not-import | Neither concrete provider imports the other — each is a self-contained Anthropic↔upstream translation, auth, and streaming path. import-linter independence contract "Providers are mutually independent (no cross-provider imports)"; D-XAI-002, INV-ARCH-01 | active |  |
| leaf_utility | orchestration | must-not-import | Leaf utilities (config, content, auth, stream, provider, stats, log, http_client, request_view, wire) never import proxy, proxy_streaming, router, __main__, or either concrete provider. import-linter forbidden contract "Leaf utilities never import orchestration"; source_modules includes the http_client transport, request_view, and the wire protocol-translation leaf | active |  |
| proxy | concrete_provider | must-not-import | The proxy dispatches only through the Provider protocol resolved from the PROVIDERS registry, never importing providers.openai or providers.xai. import-linter forbidden contract "Proxy dispatches via the provider abstraction, never concrete providers"; INV-ARCH-02 | active |  |
| proxy_streaming | concrete_provider | must-not-import | proxy_streaming, the SSE streaming data-plane split out of proxy (D-STRUCT-004), dispatches only through the Provider protocol resolved from the PROVIDERS registry, never importing providers.openai or providers.xai. import-linter forbidden contract "Proxy dispatches via the provider abstraction, never concrete providers" (source_modules now proxy and proxy_streaming); INV-ARCH-02 | active |  |

## Error Ownership

| Layer | Raises | Catches and Translates | Status | Superseded By |
|-|-|-|-|-|
| concrete_provider | Translation/validation errors on an unsupported request shape or an unparseable provider payload (e.g. ValueError); and credential errors raised by its own auth — a missing OPENAI_API_KEY, an unreadable or unsupported auth.json (FileNotFoundError/ValueError in providers/openai/auth.py, providers/xai/auth.py), or a non-printable bearer rejected before header construction. All propagate to orchestration, which catches them generically | None — a provider performs pure translation of an already-200 provider payload and never touches the transport, so it catches no HTTP or transport error (D-STRUCT-002) | active |  |
| leaf_utility | Malformed-JWT value errors (a bad bearer) from the generic auth leaf (auth.py); an httpx.TransportError re-raised by http_client.open_stream when the stream-connect phase exhausts its single retry — a typed transport error propagated to orchestration for translation; and wire.ClientDisconnected, raised by wire.safe_write when the client drops the SSE connection mid-stream (from ConnectionResetError/BrokenPipeError/OSError), also propagated to orchestration | Only in its buffered-POST helpers: http_client.forward_request and post_provider catch transport-retry exhaustion and return the final Anthropic-shaped 502 envelope directly (never an intermediate typed error). The streaming path does not — http_client.open_stream re-raises the transport error to orchestration | active |  |
| orchestration | HTTP status responses; opens the circuit on repeated failure | Provider translation and credential errors, and the httpx.TransportError re-raised by http_client.open_stream on the streaming path, into client responses — proxy_streaming writes the Anthropic-shaped 502 for a streaming-connect failure; and wire.ClientDisconnected on the streaming path, translated by proxy_streaming into StreamOutcome(499) — a client-initiated disconnect, not an upstream error | active |  |

## Layer Purity

| Layer | Owns | Must NOT Contain | Status | Superseded By |
|-|-|-|-|-|
| abstraction | the Provider protocol, PROVIDERS registry, ProviderCapabilities (provider) | orchestration or concrete-provider imports | active |  |
| concrete_provider | one provider's full Anthropic↔upstream translation, auth, and streaming (providers/openai, providers/xai) | the sibling provider; and the transport — providers receive the client via post_provider/open_stream, never importing http_client (D-STRUCT-002) | active |  |
| entry | Arg parsing, provider selection, proxy startup (__main__) | business or translation logic — it only wires a selected provider into the proxy | active |  |
| leaf_utility | Pure, reusable helpers — config, content, auth, stream, stats, log, plus the http_client data-plane transport, request_view, and the wire protocol-translation leaf | orchestration or concrete-provider imports | active |  |
| orchestration | HTTP serving, request dispatch, failover state (proxy, proxy_streaming, router) | concrete-provider imports or translation logic; dispatches only via the Provider protocol | active |  |
