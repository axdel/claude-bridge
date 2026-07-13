# Claude Bridge

**Use your Claude Code setup with any LLM provider.**

```
      _                 _            _          _     _
  ___| | __ _ _   _  __| | ___      | |__  _ __(_) __| | __ _  ___
 / __| |/ _` | | | |/ _` |/ _ \ ___ | '_ \| '__| |/ _` |/ _` |/ _ \
| (__| | (_| | |_| | (_| |  __/|___|| |_) | |  | | (_| | (_| |  __/
 \___|_|\__,_|\__,_|\__,_|\___|     |_.__/|_|  |_|\__,_|\__, |\___|
                                                         |___/
```

## Why This Exists

We built a full development protocol on top of Claude Code — a finite state machine
driving multi-agent workflows. Hundreds of hours of investment in `.claude/` configuration, hooks, CLAUDE.md conventions,
and muscle memory.

Then one day Claude is overloaded. Or rate-limited. Or you just want to see how GPT-5.6
handles the same task with the same tools.

**Without Claude Bridge:** your entire setup is useless. Claude Code only talks to Anthropic.

**With Claude Bridge:** one command, same setup, different model.

```bash
claude-codex    # your Claude Code + GPT-5.6
```

Core Claude Code flows work — tools, hooks, skills, streaming, multi-turn tool
conversations — because the bridge translates the Anthropic Messages API to the
provider's native format on-the-fly.

## How It Works

```
Claude Code  -->  Claude Bridge (localhost:9999)  -->  Anthropic (passthrough)
                           |
                     circuit breaker
                           |
                     Provider adapter  -->  OpenAI / xAI / ...
```

1. Claude Code sends an Anthropic Messages API request to `localhost:9999`
2. The bridge translates it to the target provider's format (e.g., OpenAI Responses API)
3. The provider responds with SSE; non-streaming Claude clients get an aggregated message
4. The bridge translates the response back to Anthropic format
5. Claude Code receives it as if Anthropic answered — tools, streaming, and normal text flows work

Core fidelity: `tool_use` ↔ `function_call`, `tool_result` ↔ `function_call_output`,
streaming SSE events mapped to Anthropic SSE, and tool IDs translated (`toolu_` ↔ `fc_`).
Unsupported server-tool/MCP/code-execution blocks degrade to redacted placeholders instead
of leaking provider-incompatible content.

## Features

- **Zero dependencies** — stdlib-only Python, no `pip install`
- **Direct API key auth** — set `OPENAI_API_KEY` for the official OpenAI Responses API
- **Subscription OAuth** — no API key? Uses Codex OAuth (OpenAI, `~/.codex/auth.json`) or Grok CLI OAuth (xAI, `~/.grok/auth.json`) automatically — xAI is subscription-only, no API key
- **Reasoning continuity** — encrypted reasoning blobs are cached in memory and echoed across tool turns (OpenAI and xAI)
- **Reasoning passthrough controls** — `REASONING_MODE` preserves or drops thinking blocks (OpenAI and xAI)
- **Non-text input** — image and document (PDF) blocks forward to OpenAI and xAI as `input_image`/`input_file`, including media nested in `tool_result`; unsupported media degrades to a redacted placeholder with a warning (never echoes the payload)
- **Auto-failover** — circuit breaker routes Anthropic 429/500/502/503 to the first available fallback provider
- **Retry with backoff** — transient HTTP errors retried once with 0.5s exponential backoff
- **Mid-turn failover guard** — blocks provider switch during active tool-use turns
- **Direct mode** — skip Anthropic entirely, always use a specific provider
- **Health check** — `/health` endpoint for liveness probes and process managers
- **Structured logging** — request IDs, provider/model identity, log levels (`LOG_LEVEL=DEBUG`)
- **Metrics** — `/stats` endpoint: request count, errors, latency, tokens, provider, uptime
- **Token count multiplier** — OpenAI/GPT usage totals report with a 1.2 compatibility multiplier for Claude Code auto-compact tuning
- **Token estimation** — structure-aware byte counting for context window management
- **Compatibility trace** — optional redacted structural trace for wire-contract debugging
- **Provider error redaction** — logs status and extracted summaries, never raw upstream error bodies
- **Multi-provider** — adding a provider = one provider file with declared capabilities plus registration import
- **570 tests** — coverage enforced, type-checked with basedpyright, linted with ruff

## Prerequisites

### Accounts

- **Anthropic account** — for Claude Code ([console.anthropic.com](https://console.anthropic.com/))
- **OpenAI access** — either `OPENAI_API_KEY` for the standard Responses API or ChatGPT Plus for Codex OAuth
- **xAI access** — a Grok subscription, reached through xAI's grok CLI (`grok login` → `~/.grok/auth.json`); no API key

### Software (macOS)

```bash
brew install python claude-code codex

# xAI's grok CLI self-installs its own binary under ~/.grok/bin (not via brew).
# Install it per xAI's grok CLI instructions, then it is on your PATH as `grok`.

# Verify
python3 --version    # 3.12+
claude --version
codex --version
grok --version

# Optional: authenticate with subscriptions when not using API keys
codex login                        # ChatGPT Plus OAuth path
grok login                         # xAI Grok subscription OAuth path
cat ~/.codex/auth.json             # should show access_token
cat ~/.grok/auth.json              # should show an xAI OAuth entry (keyed by issuer)
```

> **macOS only** for now (brew dependencies). Linux support is untested.
> **No `pip install` needed** — the bridge is stdlib-only Python.
> Codex CLI is for the OpenAI OAuth path, grok CLI is for the xAI OAuth path.
> If you set `OPENAI_API_KEY`, direct OpenAI mode uses the official API instead. xAI has no
> API-key mode — it always reuses the grok CLI subscription credentials.

## Install

```bash
git clone https://github.com/axdel/claude-bridge.git
cd claude-bridge

# Make the launchers available system-wide
mkdir -p ~/.local/bin
ln -sf "$(pwd)/claude-codex" ~/.local/bin/claude-codex
ln -sf "$(pwd)/claude-grok" ~/.local/bin/claude-grok

# Verify
which claude-codex claude-grok || echo 'Add to PATH: echo "export PATH=\$HOME/.local/bin:\$PATH" >> ~/.zshrc && source ~/.zshrc'
```

## Usage

### One command (recommended)

```bash
claude-codex     # use OpenAI GPT-5.6 (ChatGPT Plus subscription or OPENAI_API_KEY)
claude-grok      # use xAI Grok (grok-build) via your Grok subscription
```

You'll see:
```
      _                 _                         _
  ___| | __ _ _   _  __| | ___       ___ ___   __| | _____  __
 / __| |/ _` | | | |/ _` |/ _ \ ___ / __/ _ \ / _` |/ _ \ \/ /
| (__| | (_| | |_| | (_| |  __/|___||(_| (_) | (_| |  __/>  <
 \___|_|\__,_|\__,_|\__,_|\___|     \___\___/ \__,_|\___/_/\_\

 port:9472  pid:12345  model:gpt-5.6-sol  version:0.9.0
 by axdel  github.com/axdel/claude-bridge
```

or

```
      _                 _                            _
  ___| | __ _ _   _  __| | ___        __ _ _ __ ___ | | __
 / __| |/ _` | | | |/ _` |/ _ \_____ / _` | '__/ _ \| |/ /
| (__| | (_| | |_| | (_| |  __/_____| (_| | | | (_) |   <
 \___|_|\__,_|\__,_|\__,_|\___|      \__, |_|  \___/|_|\_\
                                     |___/
 port:9738  pid:59952  model:grok-build  version:0.9.0
 by axdel  github.com/axdel/claude-bridge
```

> Claude Code's banner still says "Sonnet 4.6" — it doesn't know about the bridge.
> For `claude-codex`, the actual model is the `gpt-5.6-sol` model shown in the bridge banner.
> For `claude-grok`, the model is `grok-build` — xAI's rolling alias for the latest
> Grok coding model (currently Grok 4.3), the grok CLI's own default (override with `XAI_MODEL`).

The bridge starts on a random port, launches Claude Code through it, and cleans up on exit.

### Options

```bash
claude-codex              # OpenAI/Codex (GPT-5.6)
claude-grok               # xAI Grok subscription (grok-build)
claude-codex --debug      # show bridge translation logs
claude-grok --debug       # same for Grok
claude-codex -- -p opus   # pass flags through to claude
```

Override the Grok model:
```bash
XAI_MODEL=grok-build claude-grok   # default; set to any model your Grok subscription exposes
```

### Verify it works

After launching with `claude-codex` or `claude-grok`, paste this into Claude Code:

> Verify Claude Code uses the local bridge: check ANTHROPIC_BASE_URL, find the bridge port, hit /stats, send one test request, compare stats.

Claude Code will confirm it's routing through the bridge by hitting the `/stats` endpoint
and seeing the request counters increment.

### Manual launch (two terminals)

```bash
# Terminal 1 — start the bridge
./start.sh --provider openai
```

```bash
# Terminal 2 — point Claude Code at it
export ANTHROPIC_BASE_URL=http://127.0.0.1:9999
export ANTHROPIC_AUTH_TOKEN=bridge-placeholder
unset ANTHROPIC_API_KEY
claude
```

## Modes

**Direct mode** — always use a specific provider, never contact Anthropic:
```bash
./start.sh --provider openai
```

**Auto mode** (default) — passthrough to Anthropic, failover on error:
```bash
./start.sh    # 429/500/502/503 -> circuit breaker -> fallback provider
```

**Fallback selection** — choose the ordered list of registered fallback providers.
The bridge uses the first registered provider in the list; it does not cascade across
later providers after a provider-side failure.
```bash
LLM_BRIDGE_FALLBACK=xai,openai ./start.sh
```

## Metrics

```bash
curl -s localhost:9999/stats | python3 -m json.tool
```

```json
{
    "requests_total": 42,
    "errors_total": 0,
    "upstream_attempts": 42,
    "failovers": 0,
    "tokens_in": 125000,
    "tokens_out": 48000,
    "latency_total_ms": 62340.5,
    "latency_avg_ms": 1484.3,
    "started_at": "2026-03-20T10:00:00+00:00",
    "uptime_seconds": 3600.0,
    "provider_name": "openai",
    "model": "gpt-5.6-sol"
}
```

## Configuration

| Env Var | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | _(none)_ | OpenAI API key — direct OpenAI mode uses the standard Responses API when set; otherwise it uses Codex OAuth |
| `XAI_MODEL` | `grok-build` | xAI Grok model id used by the `xai` provider (rolling alias for the latest coding model) |
| `XAI_CLIENT_VERSION` | highest installed grok CLI bundle (floor `0.1.202`) | Override for the `x-grok-client-version` header the cli-chat-proxy gates on; when unset, resolved from the newest `~/.grok/downloads/grok-<ver>-*` bundle |
| `REASONING_MODE` | `passthrough` | Thinking-block handling for OpenAI and xAI: `passthrough` preserves tagged thinking text, `drop` strips it |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `UPSTREAM_TIMEOUT` | caller default (`60` sync / `120` streaming) | Upstream request timeout in seconds; invalid, zero, or negative values fall back to the caller default |
| `MAX_REQUEST_BODY` | `10485760` | Maximum request body size in bytes (default 10 MiB) |
| `LLM_BRIDGE_FALLBACK` | `openai` | Comma-separated fallback preference list; the first registered provider is used |
| `LLM_BRIDGE_PORT` | `9999` | Shell launcher default proxy port |
| `ANTHROPIC_REAL_URL` | `https://api.anthropic.com` | Real Anthropic endpoint (passthrough) |
| `CLAUDE_BRIDGE_TRACE_PATH` | _(none)_ | Optional redacted JSONL structural trace path for wire-compatibility debugging |

## Architecture

```
src/claude_bridge/
├── __main__.py       # CLI entry, provider imports, auth mode detection
├── proxy.py          # HTTP server, routing, streaming, /stats, /health, retry
├── provider.py       # Provider protocol and PROVIDERS registry
├── router.py         # Circuit breaker (CLOSED/OPEN/HALF_OPEN)
├── stats.py          # Thread-safe metrics counters
├── log.py            # Structured logging, request IDs, redacted trace sink
├── auth.py           # JWT decode, token expiry
├── stream.py         # SSE parsing/formatting utilities
└── providers/
    ├── __init__.py   # Provider registration notes
    ├── openai.py     # OpenAI: API key + Codex OAuth + Responses API translation
    └── xai.py        # xAI Grok: grok CLI subscription OAuth + cli-chat-proxy Responses translation
```

### Adding a New Provider

1. Create `src/claude_bridge/providers/yourprovider.py`
2. Implement the `Provider` protocol:
   - `capabilities` — declare `ProviderCapabilities(stream_request_mode=..., sync_response_mode=...)`
   - `authenticate()` — return auth headers
   - `translate_request()` — Anthropic -> your format
   - `translate_response()` — your format -> Anthropic
   - `translate_stream()` — raw bytes -> Anthropic SSE events
3. Register only implemented providers: `PROVIDERS["yourprovider"] = YourProvider`
4. Import registered providers in `__main__.py`
5. Use: `./start.sh --provider yourprovider`
6. Optionally copy `claude-codex` -> `claude-yourprovider` (change `--provider` and banner model)

Capability modes are explicit: `stream_request_mode="body_parameter"` means the proxy sets `stream: true` in the provider request body, while `stream_request_mode="url"` means streaming is selected by endpoint URL. `sync_response_mode="sse"` keeps the current SSE aggregation path for non-streaming Anthropic clients; `sync_response_mode="json"` parses provider JSON and calls `translate_response()` directly.

Unimplemented placeholders should stay unregistered and unimported until their translation is built.

### OpenAI Translation Map

| Anthropic | OpenAI Responses API |
|---|---|
| `system` (str/blocks) | `instructions` |
| `messages[].content[type=text]` | `input[].content[type=input_text]` |
| `messages[].content[type=image]` | `input[].content[type=input_image]` (`data:` URL; MIME allowlist) |
| `messages[].content[type=document]` | `input[].content[type=input_file]` (filename + `file_data`; `application/pdf` only) |
| `messages[].content[type=tool_use]` | `input[type=function_call]` (top-level) |
| `messages[].content[type=tool_result]` | `input[type=function_call_output]` (top-level) |
| `tool_result` media (image/document) | `output[]` content-part array where supported, else redacted string |
| `tools[].input_schema` | `tools[].parameters` |
| Tool ID: `toolu_xxx` / `call_xxx` | Tool ID: `fc_xxx` |
| SSE: `content_block_delta` | SSE: `response.output_text.delta` |
| `stop_reason: tool_use` | `status: completed` + function_call in output |

> Uses OpenAI's **Responses API** (not Chat Completions) — richer tool call semantics
> with `call_id`/`id` separation.

### xAI Grok Translation

xAI Grok speaks the same **Responses API** shape as OpenAI — reached through the grok CLI's
own subscription-metered proxy — so the Anthropic ↔ Responses translation is identical to the
OpenAI map above. The provider is deliberately **self-contained**: it duplicates the Responses
translation rather than importing OpenAI's, because cross-provider imports are forbidden (see
[`DECISIONS.md`](DECISIONS.md) D-XAI-002). Grok-specific details:

| Aspect | xAI Grok |
|---|---|
| Endpoint | `https://cli-chat-proxy.grok.com/v1/responses` (subscription-metered) |
| Auth | `~/.grok/auth.json` OIDC bearer + refresh (`grok login`); no API key |
| Client gate | `x-grok-client-version` (auto-resolved from the installed grok CLI, floor `0.1.202`) + `grok-cli` client identifier |
| Reasoning continuity | encrypted reasoning cached in memory, keyed by `call_id`, echoed across tool turns (never persisted or logged) |
| Tool linkage | `call_id` alone — cli-chat-proxy has no separate `id`, and 400s if a `reasoning` key is sent |
| Token multiplier | `1.0` — subscription-metered, so no OpenAI-compat scaling |
| Media | image + document (PDF) input and array-form tool output forwarded as `input_image` / `input_file` |

> The subscription bearer rides **only** in the `Authorization` header — never in the
> version or client-identifier headers. See [`DECISIONS.md`](DECISIONS.md) D-XAI-001,
> D-XAI-003..006 for the backend, credential-handling, reasoning, usage, and
> client-version decisions.

## Decision Records

Architecture and compatibility decisions live in [`DECISIONS.md`](DECISIONS.md).
Ignored local memory files such as `CLAUDE.md`, when present, should point to that
tracked registry instead of duplicating decision rows.

## Known Limitations

- Claude Code's startup banner always shows "Sonnet 4.6" regardless of actual model
- OpenAI and xAI `thinking` blocks are passed through as tagged text by default — set `REASONING_MODE=drop` to strip them
- `output_config` and `cache_control` hints are stripped with a warning
- Token estimation is approximate (~bytes/3.5), not exact tokenization
- Streaming stats don't include token counts (only latency)
- Failover is blocked during active tool-use turns (by design — prevents broken tool state)
- Rate limit headers (`x-ratelimit-*`, `retry-after`) forwarded on sync responses only — streaming responses cannot include HTTP headers after SSE begins
- Retry applies to sync HTTP calls only — streaming connections are not retried (SSE state replay is too complex)
- Image/document input forwards to **OpenAI and xAI**; unsupported media degrades to a redacted text placeholder
- Document forwarding is limited to `application/pdf`; other document MIME types degrade to a redacted placeholder
- On the Codex OAuth (chatgpt.com) backend, tool-returned media (images/PDFs inside `tool_result`) is redacted to a text placeholder with a warning — tool-output content arrays are enabled only for the API-key backend (D-MODALITY-001)
- Media input size is bounded only by the overall request-body cap (`MAX_REQUEST_BODY`), not a separate per-media limit (D-MEDIA-001)

## Running Tests

```bash
pip install uv              # if you don't have uv
cd claude-bridge
uv run pytest tests/ -v     # installs test deps on first run, shows coverage
```

No external services — every test uses mock HTTP servers or pure-function
fixtures. Coverage is enforced at 80%.

### Mutation testing

Mutation testing (dev-only — `pytest-gremlins`, never a runtime dependency)
checks that the tests actually constrain behavior rather than merely execute it.
Run it through the wrapper script, which encodes the correct invocation and
scopes to your changed source files by default (an unscoped run mutates the
whole tree — slower, and noisy with unrelated survivors):

```bash
scripts/mutate.sh                       # mutate changed source vs HEAD
scripts/mutate.sh tests/test_stream.py  # narrow the kill-test universe (faster)
scripts/mutate.sh --all                 # full-tree sweep (scheduled)
```

The script runs:

```bash
uv run pytest --no-cov --gremlins \
  --gremlin-targets="$CHANGED" --gremlin-no-coverage-filter \
  --gremlin-parallel --gremlin-cache
```

`--gremlin-no-coverage-filter` is mandatory. Without it, gremlins' coverage-guided
selection builds a degenerate line→test map — on this Python 3.14 environment each
mutated line resolves to a single covering test that rarely asserts the mutation,
so mutants the suite actually kills are reported as false survivors (observed 50%
where the true rate is 100%). The flag bypasses the map and runs the full test set
per mutant: accurate, slower. (`--no-cov` only avoids redundant coverage overhead —
it does *not* fix the selection; pyproject's addopts forces `--cov` on every pytest
run.)

Target: ≥85% kill rate on changed source files (zero survivors for auth code).

### Security audits

Dev-only security audit tools are available through `uv`:

```bash
uv run bandit -r src
uv run pip-audit
```

Bandit suppressions are applied only at intentional stdlib/OAuth call sites; new
findings should be reviewed rather than globally skipped.

## Verifying Against an Anthropic-Compatible Reference (optional)

The contract tests pin the bridge against the Anthropic Messages and OpenAI
Responses **specifications**. If you want a second opinion from a live
Anthropic-compatible endpoint — one that speaks the same `/v1/messages` wire
format Claude Code expects — you can use one as a **black-box oracle**: send the
same request to both, then compare the response *shape*. Moonshot's Kimi endpoint
is one such reference.

This is a maintainer convenience, **not** a feature and **not** a provider. It
adds no code path, no dependency, and no provider to the bridge.

> **Not a CI requirement.** Oracle checks are manual and opt-in. The test suite
> (`uv run pytest`) runs fully offline against fixtures and is the only gate CI
> enforces. Never wire an oracle endpoint, credential, or network call into CI.

### Credentials stay out of the repo

The reference is reached the same way Claude Code reaches any Anthropic endpoint —
through environment variables, set only in your shell for the duration of a run:

```bash
# Point Claude Code (or a one-off client) at the reference. Credentials come from
# your environment or a private file OUTSIDE this repo — never inline, never committed.
export ANTHROPIC_BASE_URL=https://api.moonshot.ai/anthropic
export ANTHROPIC_AUTH_TOKEN="$(cat ~/.secrets/moonshot-token)"   # private file, git-ignored path
claude
```

- **Never** paste a token into a script, wrapper, README, fixture, or trace file.
- Source it from an environment variable or a private file outside the working tree.
- **If a token was ever stored inline** in a launcher or wrapper, treat it as
  compromised: **rotate it now**, then move it to an environment variable. A key
  that lived in a file on disk should not be trusted again.

### Manual fixture-ratification workflow

The bridge's redacted trace mode captures the *structure* of every request and
response — counts, types, names, ids, and lengths only, never prompt text, file
contents, tool output, or credentials (redaction is enforced by construction; see
`proxy.py`). That structural trace is exactly what you diff against the reference:

1. **Capture** a redacted shape trace from the bridge by pointing it at a file:

   ```bash
   CLAUDE_BRIDGE_TRACE_PATH=/tmp/bridge-trace.jsonl ./start.sh --provider openai
   # drive a representative Claude Code session, then inspect the structural trace:
   cat /tmp/bridge-trace.jsonl    # one JSON shape-summary per line — safe to read, no secrets
   ```

2. **Compare** that shape against the reference. Drive the *same* prompts with
   `ANTHROPIC_BASE_URL` pointed at Moonshot, and check the envelope matches: top-level
   fields (`id`, `type`, `role`, `model`, `content`, `stop_reason`, `usage`), the
   `stop_reason` enum value, `content` block types and ordering, and streaming event
   order. The bridge currently omits the optional `stop_sequence` field, for example —
   an envelope diff is how you'd spot a gap like that.

3. **Ratify** any confirmed difference as a **deterministic offline test** in
   `tests/test_contract.py`. Encode the *expected* shape from the
   specification (and the reference that confirmed it), not from running the bridge —
   so the test bites when the translation drifts. `TestOracleEnvelopeShape` is the
   seed example: it pins the full Anthropic Messages response envelope as the anchor a
   reference diff is measured against. New oracle findings extend that class (or a
   sibling) and then run forever in CI, fully offline.

The loop is: **reference reveals a shape difference → encode it as an offline
spec-derived test → the bridge is held to it without ever needing the reference
again.** No credential, endpoint, or network call ever enters the committed test
suite.

## Comparison

| | Claude Bridge | [1rgs/claude-code-proxy](https://github.com/1rgs/claude-code-proxy) | [fuergaosi233/claude-code-proxy](https://github.com/fuergaosi233/claude-code-proxy) |
|---|---|---|---|
| Target API | **Responses API** | Chat Completions | Chat Completions |
| Dependencies | **stdlib-only** | FastAPI + LiteLLM | FastAPI + openai SDK |
| Tool fidelity | **Proper function_call_output** | Lossy (text flatten) | Proper |
| Auto-failover | Yes (circuit breaker) | No | No |
| Metrics | `/stats` endpoint | No | No |
| Token estimation | Structure-aware | No | No |
| Multi-provider | Pluggable protocol | Via LiteLLM | OpenAI-only |
| Tests | 570 | Minimal | Some |

## Terms of Service Considerations

**This is a research project exploring API interoperability.** Before using it, be aware:

### Anthropic (Claude Code)

Claude Code is a "Beta" product under
[Anthropic's Commercial Terms](https://www.anthropic.com/legal/commercial-terms).
This project does not modify the Claude Code binary — it redirects network traffic to a
local proxy (standard networking practice). Using Claude Code with a non-Anthropic backend
**was likely not anticipated** by these terms. No explicit prohibition, no explicit permission.

### OpenAI (Codex / ChatGPT Plus)

The OpenAI provider uses the Codex OAuth flow (ChatGPT Plus subscription).
Per [OpenAI's Terms](https://openai.com/policies/terms-of-use/) and
[Usage Policies](https://openai.com/policies/usage-policies/), using the Codex endpoint
through a proxy **may fall outside intended use**. This is the same approach taken by
1rgs/claude-code-proxy (3.3k stars) and others — none taken down as of March 2026,
but past tolerance doesn't guarantee future acceptance.

### xAI (grok CLI / Grok subscription)

The xAI provider uses the grok CLI OAuth flow (Grok subscription), reusing the credentials
the grok CLI stores at `~/.grok/auth.json` to reach xAI's `cli-chat-proxy` Responses endpoint.
Per [xAI's Terms](https://x.ai/legal/terms-of-service), using this subscription endpoint through
a proxy **may fall outside intended use** — the same approach as the OpenAI/Codex provider:
reusing subscription credentials through a local proxy. There is no API-key mode; you bring your
own Grok subscription.

### Your Responsibility

**No credentials are embedded in this software.** You bring your own auth. You are responsible
for compliance with each provider's terms, costs incurred, and all consequences of usage.

## Disclaimer

**Research project.** Provided as-is for educational and experimental purposes.

- **Not affiliated** with Anthropic, OpenAI, xAI, or any AI company
- **No liability** for API terms violations, service disruptions, data loss, or costs
- **No proprietary code** — translates between publicly documented APIs
- Claude Code banner is rendered by the Claude Code binary (your install, your agreement)

**By using this software, you accept full responsibility for your use.**

## License

[MIT](LICENSE)

---

Built by [axdel](https://github.com/axdel) (with AI, for AI)
