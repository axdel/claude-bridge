# Changelog

Reverse-chronological log of all branches, fixes, and hotfixes.

## 2026-03-20

### v0.2.0 — API key auth, reasoning passthrough, failover guard, session identity ([PR #1](https://github.com/axdel/claude-bridge/pull/1))
Harden claude-bridge: standard OpenAI API key auth, thinking block passthrough, mid-turn failover guard, per-session identity logging

- [`6f3ff21`](https://github.com/axdel/claude-bridge/commit/6f3ff21) Bump version to 0.2.0 and update README with new features
- [`763832b`](https://github.com/axdel/claude-bridge/commit/763832b) Per-session identity logging and provider info in /stats
- [`5588d38`](https://github.com/axdel/claude-bridge/commit/5588d38) Mid-turn failover guard blocks provider switch during tool-use
- [`8cb8471`](https://github.com/axdel/claude-bridge/commit/8cb8471) Reasoning/thinking passthrough with REASONING_MODE config
- [`1dd202c`](https://github.com/axdel/claude-bridge/commit/1dd202c) Add standard OpenAI API key auth alongside Codex OAuth
Tasks: 5/5 | P0: 5/5

