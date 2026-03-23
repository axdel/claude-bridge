# Changelog

Reverse-chronological log of all branches, fixes, and hotfixes.

## 2026-03-23

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

## 2026-03-20

### v0.2.0 — API key auth, reasoning passthrough, failover guard, session identity ([PR #1](https://github.com/axdel/claude-bridge/pull/1))
Harden claude-bridge: standard OpenAI API key auth, thinking block passthrough, mid-turn failover guard, per-session identity logging

- [`6f3ff21`](https://github.com/axdel/claude-bridge/commit/6f3ff21) Bump version to 0.2.0 and update README with new features
- [`763832b`](https://github.com/axdel/claude-bridge/commit/763832b) Per-session identity logging and provider info in /stats
- [`5588d38`](https://github.com/axdel/claude-bridge/commit/5588d38) Mid-turn failover guard blocks provider switch during tool-use
- [`8cb8471`](https://github.com/axdel/claude-bridge/commit/8cb8471) Reasoning/thinking passthrough with REASONING_MODE config
- [`1dd202c`](https://github.com/axdel/claude-bridge/commit/1dd202c) Add standard OpenAI API key auth alongside Codex OAuth
Tasks: 5/5 | P0: 5/5

