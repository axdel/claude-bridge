# Boundary Map

## Import Rules

| Module | Target | Rule | Notes | Status | Superseded By |
|-|-|-|-|-|-|
| concrete_provider | sibling_provider | must-not-import | Neither concrete provider imports the other — each is a self-contained Anthropic↔upstream translation, auth, and streaming path. import-linter independence contract "Providers are mutually independent (no cross-provider imports)"; D-XAI-002, INV-ARCH-01 | active |  |
| leaf_utility | orchestration | must-not-import | Leaf utilities (config, content, auth, stream, provider, stats, log, http_client, request_view) never import proxy, router, __main__, or either concrete provider. import-linter forbidden contract "Leaf utilities never import orchestration"; the source_modules set now includes the http_client transport and request_view leaves | active |  |
| proxy | concrete_provider | must-not-import | The proxy dispatches only through the Provider protocol resolved from the PROVIDERS registry, never importing providers.openai or providers.xai. import-linter forbidden contract "Proxy dispatches via the provider abstraction, never concrete providers"; INV-ARCH-02 | active |  |

## Error Ownership

| Layer | Raises | Catches and Translates | Status | Superseded By |
|-|-|-|-|-|
| concrete_provider | Upstream/translation errors | HTTP/transport errors into Anthropic-shaped error responses | active |  |
| leaf_utility | Value/credential errors on malformed input (bad bearer, unreadable auth.json) | raw I/O errors into typed local errors | active |  |
| orchestration | HTTP status responses; opens the circuit on repeated failure | provider errors into client responses | active |  |

## Layer Purity

| Layer | Owns | Must NOT Contain | Status | Superseded By |
|-|-|-|-|-|
| abstraction | the Provider protocol, PROVIDERS registry, ProviderCapabilities (provider) | orchestration or concrete-provider imports | active |  |
| concrete_provider | one provider's full Anthropic↔upstream translation, auth, and streaming (providers/openai, providers/xai) | the sibling provider; and the transport — providers receive the client via post_provider/open_stream, never importing http_client (D-STRUCT-002) | active |  |
| entry | Arg parsing, provider selection, proxy startup (__main__) | business or translation logic — it only wires a selected provider into the proxy | active |  |
| leaf_utility | Pure, reusable helpers — config, content, auth, stream, stats, log, plus the http_client data-plane transport and request_view | orchestration or concrete-provider imports | active |  |
| orchestration | HTTP serving, request dispatch, failover state (proxy, proxy_streaming, router) | concrete-provider imports or translation logic; dispatches only via the Provider protocol | active |  |
