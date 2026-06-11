# Changelog

Reverse-chronological log of all branches, fixes, and hotfixes.

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

