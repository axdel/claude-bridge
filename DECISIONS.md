# Decisions

> Every irreversible or non-obvious design choice gets an ID, a rationale, and a tombstone.
> "Why this, not that" is as important as the code itself.

## Registry

### D-RUNTIME-001 — Keep runtime stdlib-only with zero runtime dependencies
- **Status:** accepted
- **Date:** 2026-03-20
- **Context:** initial project conventions
- **Decision:** Keep runtime stdlib-only with zero runtime dependencies.
- **Rationale:** Portability and supply-chain risk matter more than convenience helpers for this local bridge.
- **Invalidates:** —

### D-API-001 — Target OpenAI Responses API instead of Chat Completions
- **Status:** accepted
- **Date:** 2026-03-20
- **Context:** initial OpenAI provider
- **Decision:** Target OpenAI Responses API instead of Chat Completions.
- **Rationale:** Responses has richer tool-call semantics, including `call_id` and `id` separation, that better match Anthropic Messages.
- **Invalidates:** —

### D-AUTH-001 — Prefer API-key auth by default, with Codex OAuth as experimental fallback
- **Status:** accepted
- **Date:** 2026-03-20
- **Context:** initial OpenAI provider
- **Decision:** Prefer API-key auth by default, with Codex OAuth as experimental fallback.
- **Rationale:** Standard API-key mode is the supported path while subscription OAuth remains policy-gray and local-only.
- **Invalidates:** —

### D-THINK-001 — Preserve native reasoning formats rather than cross-mapping semantics
- **Status:** accepted
- **Date:** 2026-03-20
- **Context:** reasoning-mode design
- **Decision:** Preserve native reasoning formats rather than cross-mapping semantics.
- **Rationale:** Anthropic thinking and provider reasoning are not equivalent, so opaque passthrough or dropping is safer than pretending they translate.
- **Invalidates:** —

### D-REASON-001 — Stateful reasoning continuity qualifies the translation purity rule
- **Status:** accepted
- **Date:** 2026-06-08
- **Context:** feature/claude-code-compatibility
- **Decision:** Stateful reasoning continuity qualifies the translation purity rule.
- **Rationale:** GPT-5.5 with `store:false` requires clients to echo prior encrypted reasoning for multi-turn tool use, while the pure translation functions remain side-effect-free.
- **Invalidates:** —

### D-CACHE-001 — Bound the provider-local reasoning cache with LRU eviction
- **Status:** accepted
- **Date:** 2026-06-08
- **Context:** feature/claude-code-compatibility
- **Decision:** Bound the provider-local reasoning cache with LRU eviction.
- **Rationale:** A 256-entry oldest-first cache prevents hidden unbounded provider state while preserving tool-loop continuity.
- **Invalidates:** —

### D-USAGE-001 — Report OpenAI usage as flat Anthropic totals
- **Status:** accepted
- **Date:** 2026-06-08
- **Context:** feature/claude-code-compatibility
- **Decision:** Report OpenAI usage as flat Anthropic totals.
- **Rationale:** OpenAI `input_tokens` and `output_tokens` already include cached and reasoning details, so splitting them into Anthropic cache fields would double-count.
- **Invalidates:** —

### D-USAGE-002 — Do not add a bridge-side usage floor for Claude Code autocompact without provider-response evidence
- **Status:** superseded
- **Date:** 2026-06-08
- **Context:** feature/audit-drift-remediation
- **Decision:** Do not add a bridge-side usage floor for Claude Code autocompact without provider-response evidence.
- **Rationale:** Claude Code `/context` category boxes are diagnostic estimates, while auto-compact is driven by API usage plus local deltas; changing reported usage from local estimates could over-compact or double-count.
- **Invalidates:** —
- **Superseded by:** D-USAGE-003

### D-USAGE-003 — Apply a provider token-count multiplier and set OpenAI/GPT to 1.2
- **Status:** accepted
- **Date:** 2026-06-09
- **Context:** bugfix/token-count-multiplier
- **Decision:** Apply a provider token-count multiplier and set OpenAI/GPT to 1.2.
- **Rationale:** The multiplier is an explicit compatibility knob for empirical Claude Code auto-compact tuning, kept at the provider capability boundary instead of re-estimating prompt structure in the proxy.
- **Invalidates:** —
- **Replaces:** D-USAGE-002

### D-SRVTOOL-001 — Treat unsupported server-tool blocks as redacted unsupported content
- **Status:** accepted
- **Date:** 2026-06-08
- **Context:** feature/claude-code-compatibility
- **Decision:** Treat unsupported server-tool blocks as redacted unsupported content.
- **Rationale:** Unknown server-tool blocks have no proven target-provider equivalent, so safe degradation beats speculative translation.
- **Invalidates:** —

### D-SSE-001 — Cap malformed SSE buffers instead of rewriting normal buffering
- **Status:** accepted
- **Date:** 2026-06-08
- **Context:** feature/claude-code-compatibility
- **Decision:** Cap malformed SSE buffers instead of rewriting normal buffering.
- **Rationale:** A 4 MiB cap bounds malformed streams while well-formed SSE drains quickly, making a bytearray rewrite unnecessary.
- **Invalidates:** —

### D-TRACE-001 — Use open-append-close JSONL trace writes for compatibility diagnostics
- **Status:** accepted
- **Date:** 2026-06-08
- **Context:** feature/claude-code-compatibility
- **Decision:** Use open-append-close JSONL trace writes for compatibility diagnostics.
- **Rationale:** The opt-in fail-open trace is simpler and more rotation-friendly without a long-lived handle.
- **Invalidates:** —

### D-GOV-001 — Keep decision records in tracked `DECISIONS.md`, not ignored local `CLAUDE.md`
- **Status:** accepted
- **Date:** 2026-06-08
- **Context:** feature/audit-drift-remediation
- **Decision:** Keep decision records in tracked `DECISIONS.md`, not ignored local `CLAUDE.md`.
- **Rationale:** The registry must be committed and diffable while project-local memory remains ignored and outside branch scope.
- **Invalidates:** Ignored local `CLAUDE.md` §Key Decisions should point here out-of-band

### D-CONFIG-001 — Centralize runtime env ownership in a minimal stdlib `config.py`
- **Status:** accepted
- **Date:** 2026-06-08
- **Context:** feature/audit-drift-remediation
- **Decision:** Centralize runtime env ownership in a minimal stdlib `config.py`.
- **Rationale:** A small config owner fixes scattered env reads without adding dependencies or broad settings injection.
- **Invalidates:** README.md §Configuration

### D-PROVIDER-001 — Declare provider stream and sync-response behavior via `ProviderCapabilities`
- **Status:** accepted
- **Date:** 2026-06-08
- **Context:** feature/audit-drift-remediation
- **Decision:** Declare provider stream and sync-response behavior via `ProviderCapabilities`.
- **Rationale:** Provider-specific behavior should be explicit in `provider.py` while `proxy.py` continues to own HTTP transport, retries, stats, and errors.
- **Invalidates:** README.md §Adding a New Provider, `provider.py` provider contract

### D-PROVIDER-002 — Keep xAI unregistered until a real provider implementation exists
- **Status:** superseded
- **Date:** 2026-06-08
- **Context:** feature/audit-drift-remediation
- **Decision:** Keep xAI unregistered until a real provider implementation exists.
- **Rationale:** A routable provider whose methods raise `NotImplementedError` breaks the provider contract and failover semantics.
- **Invalidates:** README.md provider list and limitations
- **Superseded by:** D-XAI-001

### D-LOG-001 — Log provider error status and extracted summaries instead of unredacted upstream payloads
- **Status:** accepted
- **Date:** 2026-06-08
- **Context:** feature/audit-drift-remediation
- **Decision:** Log provider error status and extracted summaries instead of unredacted upstream payloads.
- **Rationale:** Upstream error payloads can include request-derived content or secrets; bounded summaries preserve operator diagnostics without leaking provider data.
- **Invalidates:** README.md feature list

### D-DEPS-001 — Add Bandit and pip-audit as dev-only audit tools
- **Status:** accepted
- **Date:** 2026-06-08
- **Context:** feature/audit-drift-remediation
- **Decision:** Add Bandit and pip-audit as dev-only audit tools.
- **Rationale:** Security gates require reproducible local source/dependency checks, and dev-only dependencies preserve the stdlib-only runtime contract.
- **Invalidates:** README.md §Security audits

### D-CONTENT-001 — Add a shared policy-free `content.py` media-source parser that both providers DERIVE from
- **Status:** accepted
- **Date:** 2026-06-09
- **Context:** feature/nontext-content-translation
- **Decision:** Add a shared policy-free `content.py` media-source parser that both providers DERIVE from.
- **Rationale:** Anthropic media-source parsing (base64/url/file → media_type/data/url/filename) is one fact; a single leaf parser stops the OpenAI and Gemini providers re-encoding it divergently (Single Owner). Kept strictly policy-free — no encoding, no warnings, no size policy — per the [grok] YAGNI caveat; if it ever needs provider-specific parameters, that is the trigger to inline it back per-provider.
- **Invalidates:** CLAUDE.md Boundary Map (adds `content` leaf row + allows `providers/* → content`), Canonical Glossary (adds media block, media source, input modality), Derivation Map (content.py is a non-derived leaf)

### D-PROVIDER-003 — Widen `ProviderCapabilities` (transport-only per D-PROVIDER-001) with additive input-content fields
- **Status:** accepted
- **Date:** 2026-06-09
- **Context:** feature/nontext-content-translation
- **Decision:** Widen `ProviderCapabilities` (transport-only per D-PROVIDER-001) with additive input-content fields.
- **Rationale:** The mapper needs a declared, testable seam — `input_modalities` and `supports_tool_output_content_parts` — to decide forward-vs-degrade per provider/backend without runtime guessing. Both fields default to the conservative pre-feature behavior (text-only, string tool output) so every existing provider and the sibling `token_count_multiplier` branch construct unchanged. Rejected a separate content-capability dataclass: for a 3-provider world a second value object is more surface than the one widened docstring costs (KISS).
- **Invalidates:** `provider.py` ProviderCapabilities docstring (now transport + input-content), README.md §Adding a New Provider

### D-MODALITY-001 — Declare OpenAI input-content capabilities per auth-mode backend on the instance, keeping the class attribute conservative
- **Status:** accepted
- **Date:** 2026-06-09
- **Context:** feature/nontext-content-translation
- **Decision:** Declare OpenAI input-content capabilities per auth-mode backend on the instance, keeping the class attribute conservative.
- **Rationale:** api.openai.com and chatgpt.com (Codex) are one provider class with two backends of differing modality support, so a single class-level `capabilities` is the wrong runtime granularity. `__init__` sets `self.capabilities` per `auth_mode`: api_key → `{text,image,document}` + tool-output arrays True (documented public Responses support); codex_oauth → `{text,image,document}` + tool-output arrays **False**. The codex defaults come from the live probe (input_image and input_file returned HTTP 200; array-form `function_call_output.output` was NOT probed, so it stays disabled and tool-result media degrades observably until a real tool-loop probe). The class attribute stays the conservative text-only default for Protocol conformance and instance-less callers; instances shadow it, and the proxy + `translate_request` both read the instance.
- **Invalidates:** `openai.py` OpenAIProvider docstring; README.md §Auth modes / limitations; follow-up: codex tool-output-array tool-loop probe

### D-SCOPE-001 — Drop Gemini media translation from this feature and leave `gemini.py` text-only as-is
- **Status:** superseded
- **Date:** 2026-06-09
- **Context:** feature/nontext-content-translation
- **Decision:** Drop Gemini media translation from this feature and leave `gemini.py` text-only as-is.
- **Rationale:** The user narrowed scope mid-branch — Gemini is no longer under active development for this bridge (no media support added; Grok will receive the canonical media treatment later), so building Gemini media support is unrequested scope; removing `gemini.py` was also not requested, so its existing text/tool path stays unchanged and media degrades observably there until Gemini is removed or revived.
- **Invalidates:** CLAUDE.md Modules (`providers/gemini.py` documented media-unsupported, OpenAI is the supported media path)
- **Superseded by:** D-GEMINI-001

### D-TOKEN-001 — Estimate media input tokens with a flat per-modality budget (image 1200, document 3000) computed inline in `proxy.py`
- **Status:** accepted
- **Date:** 2026-06-09
- **Context:** feature/nontext-content-translation
- **Decision:** Estimate media input tokens with a flat per-modality budget (image 1200, document 3000) computed inline in `proxy.py`, independent of base64 payload size.
- **Rationale:** `estimate_input_tokens` previously counted a base64 payload as text bytes (a ~300 KiB pasted image read as ~114k phantom tokens), corrupting Claude Code's auto-compact signal at `/v1/messages/count_tokens`; model media cost is dominated by fixed per-item processing (vision tiles, per-page render/extract) rather than base64 length, so a flat budget tracks reality better than byte-counting and keeps text-only estimates byte-identical (int media tokens add after rounding). Kept inline in `proxy.py` rather than in the policy-free `content.py` leaf because pre-routing token estimation is a proxy-owned concern with no provider, so `content.py` stays parse-only per D-CONTENT-001.
- **Invalidates:** CLAUDE.md Known Tech Debt (flat document budget ignores page count → token_count_multiplier follow-up)

### D-MEDIA-001 — Bound media input by the whole-request-body cap, not a second per-media hard cap
- **Status:** accepted
- **Date:** 2026-06-09
- **Context:** feature/nontext-content-translation
- **Decision:** Bound media input by the whole-request-body cap, not a second per-media hard cap.
- **Rationale:** `_MAX_REQUEST_BODY` (config; enforced at `proxy.py:198` before any media handling, rejecting with HTTP 413) already bounds total request size including base64 media; `_OVERSIZED_MEDIA_BYTES` (5 MiB, `proxy.py:223`) drives a diagnostic warning only, and `_approx_decoded_bytes` (`proxy.py:226`) is pure integer arithmetic (`len(data) * 3 // 4`, no base64 decode) so an oversized pasted image inflates neither memory nor the token estimate. A second per-media hard cap would duplicate the body bound and would wrongly reject valid multi-image requests that sit under the body limit — the body cap is the single source of size truth.
- **Invalidates:** —

### D-STREAM-001 — Map the full OpenAI Responses terminal-event taxonomy to an Anthropic stream terminator
- **Status:** accepted
- **Date:** 2026-06-11
- **Context:** bugfix/stream-terminal-events (silent mid-work halt on GPT-5.5 incomplete/failed turns)
- **Decision:** Map the full OpenAI Responses terminal-event taxonomy to an Anthropic stream terminator.
- **Rationale:** The Responses streaming API ends a turn with one of four DISTINCT top-level event types — `response.completed`, `response.incomplete` (`max_output_tokens`/`content_filter`), `response.failed` (`server_error`/...), or a bare top-level `error` — never a `completed` event with a non-completed status nested inside. The original dispatcher routed only `response.completed` to the terminating handler and dropped the rest to `return []`, so a GPT-5.5 turn ending incomplete (the common case under hardcoded `xhigh` reasoning) or failed produced no `message_stop`, and Claude Code halted mid-work with no error (HTTP 200). Route `response.incomplete` to the existing terminal handler (it already maps status/`incomplete_details` to `max_tokens`/`end_turn` + a refusal block on content_filter); translate `response.failed` and the top-level `error` to an Anthropic `error` event carrying the verbatim, length-bounded upstream reason (an upstream failure is an API error, not assistant output). Also capture reasoning continuity (D-REASON-001) on `response.incomplete`, not only `completed`.
- **Invalidates:** —

### D-STREAM-002 — Enforce a termination invariant in `translate_stream` independent of the upstream event taxonomy
- **Status:** accepted
- **Date:** 2026-06-11
- **Context:** bugfix/stream-terminal-events (silent mid-work halt on GPT-5.5 incomplete/failed turns)
- **Decision:** Enforce a termination invariant in `translate_stream` independent of the upstream event taxonomy.
- **Rationale:** Mapping each known terminal event (D-STREAM-001) fixes the observed cases but not the class of bug: any future unhandled terminal, or a dropped upstream connection, would again leave a started turn without a closer. `translate_stream` tracks whether a `message_start` was emitted and whether a terminator (`message_stop`/`error`) followed; if a started stream ends without one, it synthesizes `message_delta` + `message_stop` (`stop_reason = tool_use` when tool calls were already emitted so Claude Code runs them, else `end_turn` — never `max_tokens`, which would trigger an auto-compact retry loop). The invariant fires ONLY when a `message_start` was sent (a stream that produced no output stays empty) and never when a real terminator already arrived (no duplicate terminator). This makes "started stream with no closer" structurally impossible regardless of what the upstream sends.
- **Invalidates:** —

### D-XAI-001 — Back the xAI provider with the subscription-metered cli-chat-proxy, not api.x.ai
- **Status:** accepted
- **Date:** 2026-07-10
- **Context:** feature/swap-gemini-for-grok
- **Decision:** Route the xAI provider to `https://cli-chat-proxy.grok.com/v1/responses`, authenticated with the grok CLI subscription bearer (OIDC refresh) plus the `x-grok-client-version` / `x-grok-client-identifier` gate headers.
- **Rationale:** The subscription proxy meters against the existing Grok subscription and speaks the Responses wire verbatim, so it mirrors the Codex OAuth model with no API key; rejected `api.x.ai/v1/responses` because it bills as a separate metered API per xAI docs.
- **Invalidates:** README.md provider list, CLAUDE.md Modules (`providers/xai.py`), CLAUDE.md Resource Ownership (`~/.grok/auth.json`)
- **Replaces:** D-PROVIDER-002

### D-XAI-002 — Implement `xai.py` as a self-contained duplicate of the Responses translation (time-bounded debt)
- **Status:** accepted
- **Date:** 2026-07-10
- **Context:** feature/swap-gemini-for-grok
- **Decision:** Duplicate the OpenAI Responses request/response/stream translation into a self-contained `xai.py` with no cross-provider imports, rather than extracting a shared Responses core.
- **Rationale:** The user refused to re-touch the proven `openai.py` path, and the one-file-per-provider boundary preserves failure isolation; accepted as time-bounded duplication debt whose extraction trigger is the 3rd Responses-family provider (Rule of Three).
- **Invalidates:** CLAUDE.md Known Tech Debt (record ~800 LOC Responses duplication + `openai.py` snapshot provenance)

### D-XAI-003 — Resolve the grok credential from `~/.grok/auth.json` with re-read-before-write atomic replacement
- **Status:** accepted
- **Date:** 2026-07-10
- **Context:** feature/swap-gemini-for-grok
- **Decision:** Read the subscription bearer/refresh from `~/.grok/auth.json` (test-overridable `auth_path`), refresh via OIDC on expiry, and persist a rotated token by re-reading then atomically replacing the file at owner-only (0600) perms.
- **Rationale:** The grok CLI may refresh the same file concurrently, so re-read-before-write plus atomic rename plus 0600 perms reduce (do not eliminate) lost-update risk without a long-lived handle; no credential value is ever logged or placed in error messages.
- **Invalidates:** CLAUDE.md Resource Ownership (`~/.grok/auth.json` reader/rotator)

### D-XAI-004 — Key encrypted-reasoning continuity by the exact upstream `call_id`, characterized from live fixtures
- **Status:** accepted
- **Date:** 2026-07-10
- **Context:** feature/swap-gemini-for-grok
- **Decision:** Re-inject each turn's encrypted reasoning item keyed by the VERBATIM upstream `call_id` (no `_to_openai_id` rewrite), holding the blobs in memory only.
- **Rationale:** Captured cli-chat-proxy fixtures show the proxy echoes the client's exact `call_id`, so rewriting it would break multi-turn tool continuity; the encrypted reasoning is opaque and must never be persisted, logged, or returned to Claude Code (qualifies the translation-purity rule per D-REASON-001).
- **Invalidates:** —

### D-XAI-005 — Report xAI usage as flat Anthropic totals with an identity token-count multiplier
- **Status:** accepted
- **Date:** 2026-07-10
- **Context:** feature/swap-gemini-for-grok
- **Decision:** Flat-map xAI Responses `input_tokens`/`output_tokens` to Anthropic totals and set `token_count_multiplier = 1.0` (identity).
- **Rationale:** The subscription proxy is metered by xAI and not re-tokenized by the bridge, so no OpenAI-compat 1.2 scaling applies; Grok's 1M context does not change Claude Code's local auto-compact math, which keys off the reported totals (parallels D-USAGE-001).
- **Invalidates:** —

### D-XAI-006 — Source the `x-grok-client-version` gate dynamically from the installed CLI, with a pinned floor and env override
- **Status:** accepted
- **Date:** 2026-07-10
- **Context:** feature/swap-gemini-for-grok
- **Decision:** Resolve `x-grok-client-version` at runtime from the installed grok CLI (glob), fall back to a pinned floor `0.1.202`, and allow an `XAI_CLIENT_VERSION` env override.
- **Rationale:** cli-chat-proxy rejects requests below its required client version and may raise the floor over time; sourcing from the auto-updating CLI tracks the gate, the pinned floor still clears today's gate when the CLI is absent, and the env override is the manual escape hatch — an accepted maintenance dependency.
- **Invalidates:** CLAUDE.md Known Tech Debt (record the client-version gate as a maintenance dependency)

### D-GEMINI-001 — Remove the Gemini provider entirely, preserving only immutable historical references
- **Status:** accepted
- **Date:** 2026-07-10
- **Context:** feature/swap-gemini-for-grok
- **Decision:** Delete `providers/gemini.py` with its tests and fixtures and unregister the provider, leaving Gemini named only in immutable history (CHANGELOG entries and superseded-decision tombstones).
- **Rationale:** The user requested a complete removal of Gemini alongside the Grok addition, and a residual text-only provider nobody maintains is dead weight; CHANGELOG history and reversed-decision tombstones stay verbatim per the never-delete rule.
- **Invalidates:** CLAUDE.md Modules (remove `providers/gemini.py`), README.md provider list
- **Replaces:** D-SCOPE-001

### D-XAI-007 — Pin the xAI OIDC issuer and refuse cross-host or non-HTTPS token refresh
- **Status:** accepted
- **Date:** 2026-07-10
- **Context:** feature/swap-gemini-for-grok
- **Decision:** Pin the refresh issuer to the hardcoded HTTPS constant `https://auth.x.ai`, ignore any `oidc_issuer` field carried in `~/.grok/auth.json`, and refuse to POST the refresh_token to a non-HTTPS scheme or a cross-host endpoint.
- **Rationale:** A poisoned `auth.json` could otherwise redirect the refresh_token and client_id to an attacker-controlled or internal host (SSRF / credential exfiltration, CWE-918/200); the issuer is a fixed xAI endpoint, so trusting a file-provided `oidc_issuer` buys nothing and only opens the exfil path — rejected honoring it.
- **Invalidates:** —

### D-BUILD-001 — Make `claude_bridge` editable-installable via a `[build-system]`, not a PYTHONPATH hack
- **Status:** accepted
- **Date:** 2026-07-10
- **Context:** feature/swap-gemini-for-grok
- **Decision:** Add a hatchling `[build-system]` and `[tool.hatch.build.targets.wheel]` targeting `src/claude_bridge`, alongside an empty `[project] dependencies`, so `uv sync` editable-installs the package and venv tooling can import `claude_bridge` from `sys.path`.
- **Rationale:** The boundary-map gate runs import-linter/grimp, which needs `root_package` importable; the alternative — teaching the protocol's `tool_env` to prepend `PYTHONPATH=src` — was rejected as a per-tool hack the exemplar (claude-protocol itself) avoids by shipping a build backend, and hatchling is build-time only so the stdlib-only runtime (D-RUNTIME-001) is untouched.
- **Invalidates:** —

### D-DEPS-002 — Add import-linter as a dev-only architecture gate
- **Status:** accepted
- **Date:** 2026-07-10
- **Context:** feature/swap-gemini-for-grok
- **Decision:** Add `import-linter` to `[dependency-groups] dev` and declare three `[tool.importlinter]` contracts — providers mutually independent, proxy dispatches via the abstraction, leaf utilities never import orchestration — enforcing the module import DAG.
- **Rationale:** The boundary-map primitive delegates to `lint-imports`, so the DAG documented in BOUNDARY_MAP.md needs a mechanical enforcer; import-linter is the Python standard for this and, like Bandit and pip-audit (D-DEPS-001), is dev-only — the runtime stays zero-dependency.
- **Invalidates:** —

### D-GOV-002 — Provision the six mandatory architecture-primitive registries on this branch
- **Status:** accepted
- **Date:** 2026-07-10
- **Context:** feature/swap-gemini-for-grok
- **Decision:** Author CANONICAL_GLOSSARY, BOUNDARY_MAP, DERIVATION_MAP, RESOURCE_OWNERSHIP, INVARIANTS, and MEMORY_GOVERNANCE at the repo root, grounded in the shipped code, rather than merging with the primitive gap waived.
- **Rationale:** The review merge gate reported the six mandatory primitives as blocking STUB; provisioning them here — versus deferring to a separate prerequisite track or merging under a waiver — closes the gap in the same branch that first exercised the gate, and the registries derive from the existing architecture so they carry no invented facts.
- **Invalidates:** —

### D-XAI-008 — Default xAI model = grok-build (rolling latest-coding alias), not pinned grok-4.20
- **Status:** superseded
- **Date:** 2026-07-13
- **Context:** 2026-07-13 (hotfix)
- **Decision:** Set the config-owned `DEFAULT_XAI_MODEL` to `grok-build` and derive the `claude-grok` launcher banner model from `config.xai_model()` instead of hardcoding a model literal.
- **Rationale:** `grok-build` is xAI's rolling alias for the latest Grok coding model (currently Grok 4.3, 512K context) and the grok CLI's own default — verified against `~/.grok/models_cache.json` and the `cli-chat-proxy.grok.com` `/v1/models` origin — whereas the prior `grok-4.20` was a pinned, now-stale version (resolved upstream to `grok-4.20-0309-reasoning`), so pinning a version number was rejected for the alias that rolls forward with the subscription automatically.
- **Invalidates:** —
- **Superseded by:** D-XAI-009

### D-XAI-009 — Pin default xAI model to grok-4.6 (explicit request), superseding the grok-build alias
- **Status:** accepted
- **Date:** 2026-08-20
- **Context:** feature/model-defaults-300k
- **Decision:** Set the config-owned `DEFAULT_XAI_MODEL` to the pinned `grok-4.6` instead of the rolling `grok-build` alias. The banner still derives from `config.xai_model()`, so it tracks the pin; `XAI_MODEL` still overrides per run.
- **Rationale:** Requested explicitly to run `claude-grok` on Grok 4.6, newer than `grok-build`'s current Grok 4.3 target. `grok-4.6` was verified accepted by `cli-chat-proxy.grok.com`'s `/v1/responses` endpoint (the sibling `grok-4.6-build` 404s, and `grok models` lists only `grok-build`, but the backend accepts specific version ids too). Tradeoff vs D-XAI-008: pinning forfeits the alias's automatic roll-forward, so a future upstream deprecation (as befell `grok-4.20`) surfaces as a model error until the pin is bumped — a one-line, `XAI_MODEL`-overridable revert to `grok-build`.
- **Invalidates:** —

### D-CONTEXT-001 — Advertise a 300k context window to Claude Code from both launchers
- **Status:** accepted
- **Date:** 2026-08-20
- **Context:** feature/model-defaults-300k
- **Decision:** Export `CLAUDE_CODE_MAX_CONTEXT_TOKENS=300000` (overridable) in the `claude` subshell of both `claude-codex` and `claude-grok`.
- **Rationale:** Claude Code sizes its auto-compact threshold from the assumed model context window; the default assumption is smaller than the Codex/Grok backends actually offer. 300k is a deliberate under-estimate of both real windows (GPT-5.6 larger; Grok 512K per D-XAI-008), so Claude Code defers compaction and uses more context per session without risking a real overflow. Set on the launcher (a Claude Code env var) rather than in the bridge, which never reads it. Kept `:-300000` so an operator can still override per run.
- **Invalidates:** —

## Archive
