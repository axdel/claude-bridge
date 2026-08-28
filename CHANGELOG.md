# Changelog

Reverse-chronological log of all branches, fixes, and hotfixes.

## 2026-08-28

### fix: retire pooled HTTP/2 connections on StreamReset PROTOCOL_ERROR ([PR #20](https://github.com/axdel/claude-bridge/pull/20))
Classify Cloudflare RST_STREAM PROTOCOL_ERROR as a RemoteProtocolError, aclose the httpcore pool (not the AsyncClient), and retry once on a fresh HTTP/2 handshake so the poisoned multiplexed session is not reused.

- [`3f80eaf`](https://github.com/axdel/claude-bridge/commit/3f80eaf) Retire pooled HTTP/2 connections on StreamReset PROTOCOL_ERROR

## 2026-08-27

### fix: recover from Darwin EAI_NONAME DNS wedge and aclose leaked streams ([PR #19](https://github.com/axdel/claude-bridge/pull/19))
Classify Darwin EAI_NONAME (errno 8) as a DNS-resolver failure, retry with a 2s backoff, log that restarting the bridge recovers it, and aclose streamed httpx responses even if SSE-header write fails.

- [`afc71ce`](https://github.com/axdel/claude-bridge/commit/afc71ce) Recover from Darwin EAI_NONAME DNS wedge and aclose leaked streams

## 2026-08-24

### v0.10.0 — HTTP/2 Client Parity
Release v0.10.0.

### fix: demote only the output_config.format drop to DEBUG, keep other subkey drops loud ([PR #18](https://github.com/axdel/claude-bridge/pull/18))
Stops the per-request output_config.format WARNING from flooding the Claude Code TUI (demoted to DEBUG by exact match), while keeping meaningful subkey drops like task_budget loud at WARNING.

- [`12c492a`](https://github.com/axdel/claude-bridge/commit/12c492a) Apply ruff format to output_config test files
- [`21a5321`](https://github.com/axdel/claude-bridge/commit/21a5321) Correct output_config subkey drop log level in Known Limitations
- [`5181f9e`](https://github.com/axdel/claude-bridge/commit/5181f9e) Drive output_config subkey guards through the real translators for wire fidelity
- [`0ebf25f`](https://github.com/axdel/claude-bridge/commit/0ebf25f) Demote only the proven output_config.format drop, keep other subkey drops loud
- [`bdac8ad`](https://github.com/axdel/claude-bridge/commit/bdac8ad) Record decision to log output_config subkey drops at DEBUG
- [`7ec5c94`](https://github.com/axdel/claude-bridge/commit/7ec5c94) Log dropped output_config subkeys at DEBUG, not WARNING

### fix: honor output_config.effort and quiet routine translation notices ([PR #17](https://github.com/axdel/claude-bridge/pull/17))
The bridge now honors the per-request output_config.effort that Claude Code sends, mapping it to each provider reasoning.effort (OpenAI 1:1 default max; xAI clamps max and xhigh to high, precedence env-override then caller then low, model-gated to grok-4.6+) instead of discarding it for a static value that silently pinned grok to low. Separately, routine translation notices are demoted to DEBUG while genuinely-lossy ones stay WARNING, so the shared bridge stderr no longer floods the Claude Code TUI on every request.

- [`581aab4`](https://github.com/axdel/claude-bridge/commit/581aab4) Record D-ROUTER-002 — retain blanket output_config failover block
- [`8f53026`](https://github.com/axdel/claude-bridge/commit/8f53026) Type-tag only container tokens in _safe_token, coerce scalars literally
- [`b8becc4`](https://github.com/axdel/claude-bridge/commit/b8becc4) Always diagnose dropped output_config subkeys on xAI, independent of effort
- [`1dbc4f2`](https://github.com/axdel/claude-bridge/commit/1dbc4f2) Classify translation notices by anchored prefix, not substring
- [`379a00e`](https://github.com/axdel/claude-bridge/commit/379a00e) Correct cache_control note — caching rides prompt_cache_key, not dropped
- [`5772e98`](https://github.com/axdel/claude-bridge/commit/5772e98) Correct README effort/output_config claims + record D-EFFORT-001
- [`2d7c923`](https://github.com/axdel/claude-bridge/commit/2d7c923) Honor output_config.effort and quiet routine translation notices

## 2026-08-23

### feat: unify grok and codex on the native provider client protocol ([PR #16](https://github.com/axdel/claude-bridge/pull/16))
claude-grok and claude-codex now speak the native provider client protocol: httpx HTTP/2 data plane with split timeouts and jittered retry, sticky per-instance prompt-cache identity, per-request x-grok-req-id, model-gated reasoning.effort, cross-process OAuth refresh with reactive-401 retry, and decomposition of the provider god-files into openai/ and xai/ sub-packages. Control-plane token refresh stays on SSRF-pinned stdlib urllib.

- [`fea99d0`](https://github.com/axdel/claude-bridge/commit/fea99d0) Canonicalize http_client name + record token-refresh redirect-refusal control
- [`96658be`](https://github.com/axdel/claude-bridge/commit/96658be) Resolve architecture-primitive gate findings
- [`af09de9`](https://github.com/axdel/claude-bridge/commit/af09de9) Conform to architecture primitives; promote feature decisions
- [`bd54c5d`](https://github.com/axdel/claude-bridge/commit/bd54c5d) Record D-TEST-001 (Codex wire grounded by live-probe and shared translation)
- [`2e21c54`](https://github.com/axdel/claude-bridge/commit/2e21c54) Correct request_view helper-sharing note in docstring and D-STRUCT-003
- [`f67fa0d`](https://github.com/axdel/claude-bridge/commit/f67fa0d) Correct launcher run-model comment in pyproject
- [`6b7fbb0`](https://github.com/axdel/claude-bridge/commit/6b7fbb0) Return Anthropic error envelope for bridge-level 400 and 404
- [`6624547`](https://github.com/axdel/claude-bridge/commit/6624547) Accept provider SSE stream with absent content-type
- [`5472614`](https://github.com/axdel/claude-bridge/commit/5472614) Bump pip 26.1.2 -> 26.2.1 in the lockfile to clear PYSEC-2026-3721
- [`b82c12a`](https://github.com/axdel/claude-bridge/commit/b82c12a) Satisfy the vulture dead-code gate with FP config + stub rewrites
- [`51b7202`](https://github.com/axdel/claude-bridge/commit/51b7202) Property-based bearer validation + enable hypothesis/deptry
- [`62e1d3f`](https://github.com/axdel/claude-bridge/commit/62e1d3f) Harden start.sh launcher via project-venv interpreter (CWE-427, D-RUNTIME-003)
- [`e50e13e`](https://github.com/axdel/claude-bridge/commit/e50e13e) Record D-STRUCT-003 — keep request_view.py cohesion (token estimation + tracing)
- [`b6bf6d3`](https://github.com/axdel/claude-bridge/commit/b6bf6d3) Correct http2_client ownership row to match the real client lifecycle
- [`20f172b`](https://github.com/axdel/claude-bridge/commit/20f172b) Warn operator when a stream is served with the circuit breaker OPEN
- [`11568da`](https://github.com/axdel/claude-bridge/commit/11568da) Record D-SEC-002 — inbound request-size ceilings and accepted response-buffering residual
- [`342f181`](https://github.com/axdel/claude-bridge/commit/342f181) Bound inbound request headers by count and aggregate bytes
- [`4ffe300`](https://github.com/axdel/claude-bridge/commit/4ffe300) Correct provider-extension guidance to the sub-package layout
- [`fe133dc`](https://github.com/axdel/claude-bridge/commit/fe133dc) Distinguish provider parse failures from translation failures
- [`a7d8db2`](https://github.com/axdel/claude-bridge/commit/a7d8db2) Serialize Router.should_use_fallback through the state lock
- [`681c6dc`](https://github.com/axdel/claude-bridge/commit/681c6dc) Bound provider error text relayed to the client
- [`3e24a24`](https://github.com/axdel/claude-bridge/commit/3e24a24) Sanitize request model before it reaches logs and stats
- [`5718837`](https://github.com/axdel/claude-bridge/commit/5718837) Validate the passthrough upstream URL at startup (https-or-loopback, no userinfo)
- [`3ea0041`](https://github.com/axdel/claude-bridge/commit/3ea0041) Lock-free fast path for a fresh token in both providers' bearer refresh
- [`1a6a3fb`](https://github.com/axdel/claude-bridge/commit/1a6a3fb) Start the bridge in Python isolated mode (-I), closing CWE-427
- [`501554e`](https://github.com/axdel/claude-bridge/commit/501554e) Harden token-refresh and bearer handling in both providers
- [`f0003a8`](https://github.com/axdel/claude-bridge/commit/f0003a8) Return the Anthropic error envelope on transport failure, with jittered retry
- [`9cda65f`](https://github.com/axdel/claude-bridge/commit/9cda65f) Validate XAI_REASONING_EFFORT and reject non-finite timeout envs
- [`768d1b0`](https://github.com/axdel/claude-bridge/commit/768d1b0) Run the bridge via the project venv so its httpx dependency loads
- [`be48032`](https://github.com/axdel/claude-bridge/commit/be48032) Migrate the remaining architecture primitives to the writer format
- [`33af929`](https://github.com/axdel/claude-bridge/commit/33af929) Migrate CANONICAL_GLOSSARY.md to the writer format and add client-parity concepts
- [`4c87f3e`](https://github.com/axdel/claude-bridge/commit/4c87f3e) Migrate INVARIANTS.md to the writer table format and update for the httpx runtime
- [`aad37bc`](https://github.com/axdel/claude-bridge/commit/aad37bc) Migrate DECISIONS.md to the writer table format and record client-parity decisions
- [`df6b6b8`](https://github.com/axdel/claude-bridge/commit/df6b6b8) Surface bridge failures on the launcher terminal and harden claude-codex
- [`034fe18`](https://github.com/axdel/claude-bridge/commit/034fe18) Refresh the credential and retry once when a provider returns 401
- [`b956fb2`](https://github.com/axdel/claude-bridge/commit/b956fb2) Send a per-request x-grok-req-id header on grok requests
- [`d173101`](https://github.com/axdel/claude-bridge/commit/d173101) Let providers force a credential refresh past the proactive expiry check
- [`ea4e53a`](https://github.com/axdel/claude-bridge/commit/ea4e53a) Serialize xAI OAuth refresh across processes with flock + double-check
- [`1b5caad`](https://github.com/axdel/claude-bridge/commit/1b5caad) Send model-gated reasoning.effort and max_output_tokens on xAI requests
- [`4ab8621`](https://github.com/axdel/claude-bridge/commit/4ab8621) Stamp a sticky per-instance prompt cache key on provider requests
- [`d84902a`](https://github.com/axdel/claude-bridge/commit/d84902a) Replace urllib data-plane transport with httpx HTTP/2 (split timeouts)
- [`96bfc71`](https://github.com/axdel/claude-bridge/commit/96bfc71) Split provider god-files into cohesive sub-packages

## 2026-08-20

### feat: grok-4.6 default, 300k launcher context, GPT multiplier 1.1 ([PR #15](https://github.com/axdel/claude-bridge/pull/15))
Default claude-grok to grok-4.6 (D-XAI-009, superseding the grok-build alias in D-XAI-008); advertise a 300k context window to Claude Code from both launchers via CLAUDE_CODE_MAX_CONTEXT_TOKENS (D-CONTEXT-001); lower the OpenAI/GPT token-count multiplier from 1.2 to 1.1 (D-USAGE-004).

- [`d9d2b32`](https://github.com/axdel/claude-bridge/commit/d9d2b32) Correct advertised Grok window to grok-4.6's 500K
- [`645913c`](https://github.com/axdel/claude-bridge/commit/645913c) Reference GPT_TOKEN_COUNT_MULTIPLIER in class-capability test
- [`11f5c72`](https://github.com/axdel/claude-bridge/commit/11f5c72) Lower OpenAI/GPT token-count multiplier from 1.2 to 1.1
- [`63f3bfb`](https://github.com/axdel/claude-bridge/commit/63f3bfb) Advertise a 300k context window to Claude Code from both launchers
- [`0ad3fc2`](https://github.com/axdel/claude-bridge/commit/0ad3fc2) Default xAI provider to grok-4.6, superseding the grok-build alias

## 2026-07-13

### v0.9.0 — xAI Grok Provider
Swap the Gemini provider for a Grok (xAI) provider: grok CLI subscription-OAuth via ~/.grok (no API key), self-contained Anthropic<->Responses translation with image/document media and encrypted-reasoning continuity, defaulting to grok-build (xAI's rolling latest-coding alias); provision the six architecture-primitive registries with import-linter enforcement.

### Hotfixes
- [`29431dd`](https://github.com/axdel/claude-bridge/commit/29431dd) Default the xAI Grok provider to grok-build (xAI's rolling alias for the latest coding model, currently Grok 4.3) instead of the pinned, now-stale grok-4.20; the claude-grok launcher banner now derives its model from config

## 2026-07-10

### feat: swap Gemini provider for Grok (xAI) ([PR #14](https://github.com/axdel/claude-bridge/pull/14))
Remove the Gemini provider and add a Grok (xAI) provider mirroring the Codex subscription-OAuth model (~/.grok bearer + OIDC refresh, no API key); provision the six architecture-primitive registries with import-linter enforcement.

- [`9c08410`](https://github.com/axdel/claude-bridge/commit/9c08410) Record D-XAI-007 pinning the xAI OIDC issuer against SSRF
- [`713d52d`](https://github.com/axdel/claude-bridge/commit/713d52d) Harden xAI credential handling and sanitize reasoning fixture
- [`2b5be1d`](https://github.com/axdel/claude-bridge/commit/2b5be1d) Harden claude-grok launcher startup path and liveness
- [`e7203d7`](https://github.com/axdel/claude-bridge/commit/e7203d7) Document xAI Grok provider, remove Gemini from docs, bump to 0.9.0
- [`f1f8d35`](https://github.com/axdel/claude-bridge/commit/f1f8d35) Record the xAI provider and Gemini-removal decision set
- [`8bb5f24`](https://github.com/axdel/claude-bridge/commit/8bb5f24) Add cross-layer contract coverage for the xAI provider
- [`403ca8f`](https://github.com/axdel/claude-bridge/commit/403ca8f) Wire xAI Grok provider — capabilities, registration, launcher
- [`3682472`](https://github.com/axdel/claude-bridge/commit/3682472) Carry xAI encrypted reasoning across tool turns
- [`6225444`](https://github.com/axdel/claude-bridge/commit/6225444) Translate xAI Responses SSE streams to Anthropic Messages events
- [`81edc38`](https://github.com/axdel/claude-bridge/commit/81edc38) Translate xAI Responses objects to Anthropic Messages responses
- [`039fe78`](https://github.com/axdel/claude-bridge/commit/039fe78) Forward image and document media to xAI Responses parts
- [`c7928fa`](https://github.com/axdel/claude-bridge/commit/c7928fa) Translate Anthropic requests to xAI Responses format
- [`11178bb`](https://github.com/axdel/claude-bridge/commit/11178bb) Add xAI model + client-version config resolvers
- [`2e4d836`](https://github.com/axdel/claude-bridge/commit/2e4d836) Add xAI subscription-OAuth token module
- [`071ccdc`](https://github.com/axdel/claude-bridge/commit/071ccdc) Capture xAI Grok wire fixtures (live characterization)
- [`9e32101`](https://github.com/axdel/claude-bridge/commit/9e32101) Remove Gemini provider

## 2026-06-19

### v0.8.1 — Lockfile Version Sync
Release v0.8.1.

### v0.8.0 — Media Translation Reliability
Release v0.8.0.

## 2026-06-11

### fix: terminate Anthropic stream on every OpenAI Responses terminal event ([PR #13](https://github.com/axdel/claude-bridge/pull/13))
GPT-5.5 turns ending in response.incomplete, response.failed, or a top-level error now emit a stream terminator, so Claude Code finalizes the turn instead of halting mid-work; a termination invariant guarantees every started stream is closed.

- [`9d1b29b`](https://github.com/axdel/claude-bridge/commit/9d1b29b) Terminate Anthropic stream on every OpenAI Responses terminal event

## 2026-06-10

### fix: add OpenAI token count multiplier ([PR #12](https://github.com/axdel/claude-bridge/pull/12))
Provider-declared token_count_multiplier (additive ProviderCapabilities field, neutral 1.0 default) scales reported usage totals; OpenAI/GPT set to 1.2 to tune Claude Code auto-compact for GPT-5.5 token-count divergence, applied at the capability boundary over the D-USAGE-001 flat mapping.

- [`e6a83c0`](https://github.com/axdel/claude-bridge/commit/e6a83c0) Add OpenAI token count multiplier

### fix: single-pass token estimation ([PR #11](https://github.com/axdel/claude-bridge/pull/11))
Fold the oversized-media scan into the single token-estimation walk; behavior byte-identical (differential-verified). Resolves SCL-001.

- [`064d098`](https://github.com/axdel/claude-bridge/commit/064d098) Single-pass token estimation, drop oversized media re-walk

## 2026-06-09

### feat: systemic non-text content translation ([PR #10](https://github.com/axdel/claude-bridge/pull/10))
Forward image, document/PDF, and tool_result media from Anthropic input to the OpenAI Responses API via a shared media-source parser and auth-mode-aware capabilities, with base64-safe token estimation and observable degradation.

- [`fa29d71`](https://github.com/axdel/claude-bridge/commit/fa29d71) Document media-forwarding capability in README (D-CONTENT-001/D-MODALITY-001/D-SCOPE-001)
- [`2a76b33`](https://github.com/axdel/claude-bridge/commit/2a76b33) Record D-MEDIA-001 (media bounded by request-body cap) from /review ADV-003
- [`2bc0ff9`](https://github.com/axdel/claude-bridge/commit/2bc0ff9) Warn when tool_result media degrades to string fallback
- [`6edabc0`](https://github.com/axdel/claude-bridge/commit/6edabc0) Validate document media_type and sanitize forwarded filename
- [`e747352`](https://github.com/axdel/claude-bridge/commit/e747352) Finalize media-translation decision records
- [`e79917f`](https://github.com/axdel/claude-bridge/commit/e79917f) Make token estimation media-aware and base64-safe in traces
- [`7e0bd5b`](https://github.com/axdel/claude-bridge/commit/7e0bd5b) Declare OpenAI input-content capabilities per auth-mode backend
- [`f9f47ae`](https://github.com/axdel/claude-bridge/commit/f9f47ae) Emit tool_result media as real Responses content parts
- [`73df069`](https://github.com/axdel/claude-bridge/commit/73df069) Forward top-level image and document blocks to OpenAI Responses
- [`e4f2277`](https://github.com/axdel/claude-bridge/commit/e4f2277) Add additive input-modality capabilities to ProviderCapabilities
- [`88d6370`](https://github.com/axdel/claude-bridge/commit/88d6370) Add shared policy-free media-source parser (content.py)

### feat: consolidated audit drift remediation ([PR #9](https://github.com/axdel/claude-bridge/pull/9))
Remediate provider, config, logging, decision-record, and security-tooling drift from the deep audit.

- [`6ffccad`](https://github.com/axdel/claude-bridge/commit/6ffccad) Record streaming failure outcomes
- [`cc840ab`](https://github.com/axdel/claude-bridge/commit/cc840ab) Harden provider preflight validation
- [`9914235`](https://github.com/axdel/claude-bridge/commit/9914235) Align compatibility documentation
- [`b0fbc1c`](https://github.com/axdel/claude-bridge/commit/b0fbc1c) Use provider sync response capabilities
- [`c15445c`](https://github.com/axdel/claude-bridge/commit/c15445c) Use provider stream capabilities
- [`9405c91`](https://github.com/axdel/claude-bridge/commit/9405c91) Declare provider capabilities
- [`93a317f`](https://github.com/axdel/claude-bridge/commit/93a317f) Centralize runtime config ownership
- [`bd8d43e`](https://github.com/axdel/claude-bridge/commit/bd8d43e) Keep xAI placeholder non-routable
- [`2a0208a`](https://github.com/axdel/claude-bridge/commit/2a0208a) Redact provider error logs
- [`4927bc5`](https://github.com/axdel/claude-bridge/commit/4927bc5) Redact Gemini unsupported blocks
- [`3df8d21`](https://github.com/axdel/claude-bridge/commit/3df8d21) Record autocompact usage decision
- [`2b27876`](https://github.com/axdel/claude-bridge/commit/2b27876) Add decision registry

## 2026-06-08

### v0.7.0 — Claude Code Wire Compatibility
Release v0.7.0.

### feat: claude Code wire compatibility for the OpenAI Responses provider ([PR #8](https://github.com/axdel/claude-bridge/pull/8))
Faithful Claude Code wire translation for GPT-5.5: serialized tool loops, encrypted-reasoning continuity, stop/usage disambiguation, non-streaming aggregation, and Anthropic error envelopes.

- [`98dddb7`](https://github.com/axdel/claude-bridge/commit/98dddb7) Translate provider errors to Anthropic error envelopes
- [`65c81df`](https://github.com/axdel/claude-bridge/commit/65c81df) Aggregate Codex SSE stream for non-streaming requests
- [`28aed6f`](https://github.com/axdel/claude-bridge/commit/28aed6f) Split oversized test files (QAL3 + QAL4)
- [`2ebf018`](https://github.com/axdel/claude-bridge/commit/2ebf018) Harden OpenAIProvider tests (QAL2 construction + REQ2 concurrency)
- [`1370998`](https://github.com/axdel/claude-bridge/commit/1370998) Make trace failures visible and refuse non-regular trace targets (OPS1/OPS2)
- [`db15cc7`](https://github.com/axdel/claude-bridge/commit/db15cc7) Bound the SSE buffer to abort malformed provider streams (SCL-2)
- [`aa1e0b8`](https://github.com/axdel/claude-bridge/commit/aa1e0b8) Sanitize translation-warning tokens (CWE-117) and trace warning strings
- [`d12830f`](https://github.com/axdel/claude-bridge/commit/d12830f) Fix test_proxy type errors; doc: mutation flag + verification
- [`d75298c`](https://github.com/axdel/claude-bridge/commit/d75298c) Add optional Moonshot/Kimi oracle verification workflow
- [`260e0af`](https://github.com/axdel/claude-bridge/commit/260e0af) Redact unsupported server-tool/MCP content blocks instead of stringifying
- [`bb41af1`](https://github.com/axdel/claude-bridge/commit/bb41af1) Disambiguate content_filter from token exhaustion in OpenAI stop/usage
- [`570dfae`](https://github.com/axdel/claude-bridge/commit/570dfae) Provider-local reasoning continuity across tool turns
- [`cf86cf4`](https://github.com/axdel/claude-bridge/commit/cf86cf4) Add redacted compatibility trace mode
- [`c97cc28`](https://github.com/axdel/claude-bridge/commit/c97cc28) Map Anthropic tool_choice and parallel controls (T-002)
- [`5849506`](https://github.com/axdel/claude-bridge/commit/5849506) Add Claude Code wire-contract fixtures (T-001)

## 2026-05-05

### v0.6.4 — Gemini Stability and Tool-Use Fixes
Release v0.6.4.

### v0.6.3 — gpt-5.5 + xhigh reasoning
Release v0.6.3.

### v0.6.2 — gpt-5.5 + xhigh reasoning
Update OpenAI provider to gpt-5.5 with reasoning effort xhigh for maximum code quality

### Hotfixes
- [`626f148`](https://github.com/axdel/claude-bridge/commit/626f148) update codex model to gpt-5.5 with xhigh reasoning effort

## 2026-03-24

### Gemini OAuth — Use Gemini CLI Subscription ([PR #7](https://github.com/axdel/claude-bridge/pull/7))
Add gemini_oauth auth mode using Gemini CLI subscription (Google One AI Premium) — no API key needed. Default model gemini-3-pro-preview. 186→202 tests

- [`c62f222`](https://github.com/axdel/claude-bridge/commit/c62f222) Update README for Gemini OAuth dual auth mode
- [`56b5773`](https://github.com/axdel/claude-bridge/commit/56b5773) Add Gemini OAuth using Gemini CLI subscription
Tasks: 5/5

### Gemini Provider — Second Fallback for Resilience ([PR #6](https://github.com/axdel/claude-bridge/pull/6))
Add Google Gemini as second fallback provider (auth, translation, streaming, launcher) — 156→186 tests, 87% coverage

- [`69deb6c`](https://github.com/axdel/claude-bridge/commit/69deb6c) Add claude-gemini launcher and update README for Gemini provider
- [`f7d78ca`](https://github.com/axdel/claude-bridge/commit/f7d78ca) Gemini SSE stream translation to Anthropic events
- [`3546735`](https://github.com/axdel/claude-bridge/commit/3546735) Gemini response translation (gemini_to_anthropic)
- [`e30c58b`](https://github.com/axdel/claude-bridge/commit/e30c58b) Gemini request translation (anthropic_to_gemini)
- [`244c390`](https://github.com/axdel/claude-bridge/commit/244c390) Add Gemini provider skeleton with API key auth
Tasks: 5/5


### Hotfixes
- [`81583b0`](https://github.com/axdel/claude-bridge/commit/81583b0) Bump version to 0.5.0, update README test count 156→186
- [`e262612`](https://github.com/axdel/claude-bridge/commit/e262612) Bump version to 0.6.0, update README test count 186→202

## 2026-03-23

### v0.4.0 — Developer Tooling + Reliability Fixes ([PR #5](https://github.com/axdel/claude-bridge/pull/5))
Add ruff, basedpyright, pre-commit hooks, retry/backoff, /health endpoint, dead code cleanup, coverage enforcement (153→156 tests, 87% coverage)

- [`62e71a2`](https://github.com/axdel/claude-bridge/commit/62e71a2) Add pytest-cov coverage enforcement at 85% threshold
- [`e1b7245`](https://github.com/axdel/claude-bridge/commit/e1b7245) Wire record_failover into failover path, remove dead record_error
- [`c79ed4b`](https://github.com/axdel/claude-bridge/commit/c79ed4b) Add /health endpoint for liveness probes
- [`0a897a5`](https://github.com/axdel/claude-bridge/commit/0a897a5) Add retry with backoff on sync HTTP calls
- [`2db3b3f`](https://github.com/axdel/claude-bridge/commit/2db3b3f) Add pre-commit hooks for gitleaks and ruff
- [`6d16143`](https://github.com/axdel/claude-bridge/commit/6d16143) Add basedpyright type checking with standard mode
- [`473ccd8`](https://github.com/axdel/claude-bridge/commit/473ccd8) Add ruff linting and formatting with initial codebase cleanup
Tasks: 7/7


### Consistent cache_control hint handling ([PR #4](https://github.com/axdel/claude-bridge/pull/4))
Strip cache_control hints consistently from content blocks, system blocks, and tool definitions with a single summary warning per request

- [`57285fa`](https://github.com/axdel/claude-bridge/commit/57285fa) Consistent cache_control handling across all request locations
Tasks: 1/1


### v0.3.0 — Auth Hardening + Test Coverage ([PR #3](https://github.com/axdel/claude-bridge/pull/3))
Harden auth error paths (JWT decode, OAuth refresh), add streaming and token refresh failure tests, bump to v0.3.0 (137→150 tests)

- [`3e88852`](https://github.com/axdel/claude-bridge/commit/3e88852) Bump version to 0.3.0 and update README test counts
- [`b1991d6`](https://github.com/axdel/claude-bridge/commit/b1991d6) Add end-to-end token refresh failure coverage
- [`7508a2f`](https://github.com/axdel/claude-bridge/commit/7508a2f) Add streaming error path coverage for provider and passthrough
- [`35b4b96`](https://github.com/axdel/claude-bridge/commit/35b4b96) Harden OAuth token refresh error handling
- [`ca6a20a`](https://github.com/axdel/claude-bridge/commit/ca6a20a) Harden JWT decode and token expiry error handling
Tasks: 5/5 | P1: 2/2 | P3: 3/3


### P0 Reliability Hardening + P2 Production Improvements ([PR #2](https://github.com/axdel/claude-bridge/pull/2))
Configurable timeouts, body size limits, translation validation, rate limit header forwarding, image block preservation, and 23 new tests (114→137)

- [`a73bde5`](https://github.com/axdel/claude-bridge/commit/a73bde5) Add streaming integration tests and SSE round-trip coverage
- [`d9c6da4`](https://github.com/axdel/claude-bridge/commit/d9c6da4) Add coverage for /v1/messages/count_tokens endpoint
- [`acfeead`](https://github.com/axdel/claude-bridge/commit/acfeead) Preserve image blocks in tool_result content translation
- [`feddc55`](https://github.com/axdel/claude-bridge/commit/feddc55) Forward rate limit headers from upstream to client
- [`67966b0`](https://github.com/axdel/claude-bridge/commit/67966b0) Validate translate_request() return type before use
- [`be58e35`](https://github.com/axdel/claude-bridge/commit/be58e35) Request body size limit via MAX_REQUEST_BODY env var
- [`6d441dd`](https://github.com/axdel/claude-bridge/commit/6d441dd) Configurable upstream timeouts via UPSTREAM_TIMEOUT env var
Tasks: 7/7 | P0: 5/5 | P2: 2/2


### Hotfixes
- [`387b550`](https://github.com/axdel/claude-bridge/commit/387b550) Update README — test count 114→137, add UPSTREAM_TIMEOUT/MAX_REQUEST_BODY env vars, document streaming rate limit limitation
- [`28699b4`](https://github.com/axdel/claude-bridge/commit/28699b4) Expand stream exception handler to catch unexpected errors, add pytest-cov to dev deps
- [`a480021`](https://github.com/axdel/claude-bridge/commit/a480021) Show bridge version in claude-codex launcher banner
- [`385efe7`](https://github.com/axdel/claude-bridge/commit/385efe7) Add .coverage to .gitignore (generated by pytest-cov)
- [`9b0ddf5`](https://github.com/axdel/claude-bridge/commit/9b0ddf5) Bump version to 0.4.0, update README test count 150→156
- [`de64183`](https://github.com/axdel/claude-bridge/commit/de64183) Update README — add retry/health/coverage features, provider_name/model in /stats example, streaming retry limitation

## 2026-03-20

### v0.2.0 — API key auth, reasoning passthrough, failover guard, session identity ([PR #1](https://github.com/axdel/claude-bridge/pull/1))
Harden claude-bridge: standard OpenAI API key auth, thinking block passthrough, mid-turn failover guard, per-session identity logging

- [`6f3ff21`](https://github.com/axdel/claude-bridge/commit/6f3ff21) Bump version to 0.2.0 and update README with new features
- [`763832b`](https://github.com/axdel/claude-bridge/commit/763832b) Per-session identity logging and provider info in /stats
- [`5588d38`](https://github.com/axdel/claude-bridge/commit/5588d38) Mid-turn failover guard blocks provider switch during tool-use
- [`8cb8471`](https://github.com/axdel/claude-bridge/commit/8cb8471) Reasoning/thinking passthrough with REASONING_MODE config
- [`1dd202c`](https://github.com/axdel/claude-bridge/commit/1dd202c) Add standard OpenAI API key auth alongside Codex OAuth
Tasks: 5/5 | P0: 5/5

