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

Then one day Claude is overloaded. Or rate-limited. Or you just want to see how GPT-5.5
handles the same task with the same tools.

**Without Claude Bridge:** your entire setup is useless. Claude Code only talks to Anthropic.

**With Claude Bridge:** one command, same setup, different model.

```bash
claude-codex    # your Claude Code + GPT-5.5
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
                     Provider adapter  -->  OpenAI / Gemini / ...
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
- **Direct API key auth** — set `OPENAI_API_KEY` or `GEMINI_API_KEY` for official APIs
- **Subscription OAuth** — no API key? Falls back to Codex OAuth (OpenAI) or Gemini CLI OAuth (Google) automatically
- **OpenAI reasoning continuity** — encrypted reasoning blobs are cached in memory and echoed across tool turns
- **Reasoning passthrough controls** — OpenAI can preserve or drop thinking blocks; Gemini strips them with a warning
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
- **351 tests** — coverage enforced, type-checked with basedpyright, linted with ruff

## Prerequisites

### Accounts

- **Anthropic account** — for Claude Code ([console.anthropic.com](https://console.anthropic.com/))
- **OpenAI access** — either `OPENAI_API_KEY` for the standard Responses API or ChatGPT Plus for Codex OAuth
- **Gemini access** — either `GEMINI_API_KEY` from [Google AI Studio](https://aistudio.google.com/apikey) or Google One AI Premium for Gemini CLI OAuth

### Software (macOS)

```bash
brew install python claude-code codex gemini

# Verify
python3 --version    # 3.12+
claude --version
codex --version
gemini --version

# Optional: authenticate with subscriptions when not using API keys
codex login                        # ChatGPT Plus OAuth path
gemini login                       # Google One AI Premium OAuth path
cat ~/.codex/auth.json             # should show access_token
cat ~/.gemini/oauth_creds.json     # should show access_token
```

> **macOS only** for now (brew dependencies). Linux support is untested.
> **No `pip install` needed** — the bridge is stdlib-only Python.
> Codex CLI is for the OpenAI OAuth path, Gemini CLI is for the Gemini OAuth path.
> If you set `OPENAI_API_KEY` or `GEMINI_API_KEY`, the matching direct provider mode uses
> the official API instead.

## Install

```bash
git clone https://github.com/axdel/claude-bridge.git
cd claude-bridge

# Make the launchers available system-wide
mkdir -p ~/.local/bin
ln -sf "$(pwd)/claude-codex" ~/.local/bin/claude-codex
ln -sf "$(pwd)/claude-gemini" ~/.local/bin/claude-gemini

# Verify
which claude-codex claude-gemini || echo 'Add to PATH: echo "export PATH=\$HOME/.local/bin:\$PATH" >> ~/.zshrc && source ~/.zshrc'
```

## Usage

### One command (recommended)

```bash
claude-codex     # use OpenAI GPT-5.5 (ChatGPT Plus subscription or OPENAI_API_KEY)
claude-gemini    # use Gemini 3 Flash OAuth by default, or API key mode with GEMINI_API_KEY
```

You'll see:
```
      _                 _                         _
  ___| | __ _ _   _  __| | ___       ___ ___   __| | _____  __
 / __| |/ _` | | | |/ _` |/ _ \ ___ / __/ _ \ / _` |/ _ \ \/ /
| (__| | (_| | |_| | (_| |  __/|___||(_| (_) | (_| |  __/>  <
 \___|_|\__,_|\__,_|\__,_|\___|     \___\___/ \__,_|\___/_/\_\

 port:9472  pid:12345  model:gpt-5.5  version:0.7.0
 by axdel  github.com/axdel/claude-bridge
```

or

```
      _                 _                                _       _
  ___| | __ _ _   _  __| | ___       __ _  ___ _ __ ___ (_)_ __ (_)
 / __| |/ _` | | | |/ _` |/ _ \ ___ / _` |/ _ \ '_ ` _ \| | '_ \| |
| (__| | (_| | |_| | (_| |  __/|___| (_| |  __/ | | | | | | | | | |
 \___|_|\__,_|\__,_|\__,_|\___|     \__, |\___|_| |_| |_|_|_| |_|_|
                                    |___/
 port:9738  pid:59952  model:gemini-3-flash-preview  version:0.7.0
 by axdel  github.com/axdel/claude-bridge
```

> Claude Code's banner still says "Sonnet 4.6" — it doesn't know about the bridge.
> For `claude-codex`, the actual model is the GPT-5.5 model shown in the bridge banner.
> For `claude-gemini`, the banner shows the `GEMINI_MODEL` launcher value; API-key mode
> defaults to `gemini-2.5-pro` unless you set `GEMINI_MODEL`.

The bridge starts on a random port, launches Claude Code through it, and cleans up on exit.

### Options

```bash
claude-codex              # OpenAI/Codex (GPT-5.5)
claude-gemini             # Google Gemini OAuth (gemini-3-flash-preview)
claude-codex --debug      # show bridge translation logs
claude-gemini --debug     # same for Gemini
claude-codex -- -p opus   # pass flags through to claude
```

Override the Gemini model:
```bash
GEMINI_MODEL=gemini-2.5-pro claude-gemini      # stable pro model
GEMINI_MODEL=gemini-3.1-pro-preview claude-gemini  # latest pro (may have capacity limits)
```

### Verify it works

After launching with `claude-codex` or `claude-gemini`, paste this into Claude Code:

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
LLM_BRIDGE_FALLBACK=gemini,openai ./start.sh
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
    "model": "gpt-5.5"
}
```

## Configuration

| Env Var | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | _(none)_ | OpenAI API key — direct OpenAI mode uses the standard Responses API when set; otherwise it uses Codex OAuth |
| `GEMINI_API_KEY` | _(none)_ | Google Gemini API key — direct Gemini mode uses the public API when set; otherwise it uses Gemini CLI OAuth (`~/.gemini/oauth_creds.json`) |
| `GEMINI_MODEL` | API-key: `gemini-2.5-pro`; OAuth: `gemini-3-flash-preview` | Gemini model override. Both auth modes honor the same env var; when unset, API-key mode uses the API-key default and OAuth mode uses the OAuth default |
| `REASONING_MODE` | `passthrough` | OpenAI thinking-block handling: `passthrough` preserves tagged thinking text, `drop` strips it. Gemini strips thinking blocks because it has no equivalent |
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
    ├── gemini.py     # Gemini: API key + Gemini CLI OAuth + generateContent translation
    └── xai.py        # xAI Grok: unregistered placeholder, not implemented
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

Unimplemented placeholders should stay unregistered and unimported, like the current xAI stub.

### OpenAI Translation Map

| Anthropic | OpenAI Responses API |
|---|---|
| `system` (str/blocks) | `instructions` |
| `messages[].content[type=text]` | `input[].content[type=input_text]` |
| `messages[].content[type=tool_use]` | `input[type=function_call]` (top-level) |
| `messages[].content[type=tool_result]` | `input[type=function_call_output]` (top-level) |
| `tools[].input_schema` | `tools[].parameters` |
| Tool ID: `toolu_xxx` / `call_xxx` | Tool ID: `fc_xxx` |
| SSE: `content_block_delta` | SSE: `response.output_text.delta` |
| `stop_reason: tool_use` | `status: completed` + function_call in output |

> Uses OpenAI's **Responses API** (not Chat Completions) — richer tool call semantics
> with `call_id`/`id` separation.

### Gemini Translation Map

| Anthropic | Gemini generateContent API |
|---|---|
| `system` (str/blocks) | `system_instruction.parts[].text` |
| `messages[role=user]` | `contents[role=user].parts[].text` |
| `messages[role=assistant]` | `contents[role=model].parts[].text` |
| `messages[].content[type=tool_use]` | `contents[].parts[].functionCall` |
| `messages[].content[type=tool_result]` | `contents[].parts[].functionResponse` |
| `tools[].input_schema` | `tools[0].function_declarations[].parameters` |
| Tool ID: `toolu_xxx` | Tool ID: `toolu_gemini_xxx` (legacy `call_gemini_xxx` accepted; thoughtSignature encoded) |
| SSE: `content_block_delta` | SSE: `data:` chunks (complete JSON per chunk) |
| `stop_reason: tool_use` | `finishReason: STOP` + functionCall in parts |

> Uses Gemini's **generateContent API** via the Code Assist endpoint for OAuth
> (subscription) or the public API for API key auth. `$schema`, `propertyNames`,
> and other unsupported JSON Schema keywords are automatically stripped from tool definitions.

## Decision Records

Architecture and compatibility decisions live in [`DECISIONS.md`](DECISIONS.md).
Ignored local memory files such as `CLAUDE.md`, when present, should point to that
tracked registry instead of duplicating decision rows.

## Known Limitations

- Claude Code's startup banner always shows "Sonnet 4.6" regardless of actual model
- OpenAI `thinking` blocks are passed through as tagged text by default — set `REASONING_MODE=drop` to strip them; Gemini strips thinking blocks with a warning
- `output_config` and `cache_control` hints are stripped with a warning
- Token estimation is approximate (~bytes/3.5), not exact tokenization
- Streaming stats don't include token counts (only latency)
- Failover is blocked during active tool-use turns (by design — prevents broken tool state)
- Rate limit headers (`x-ratelimit-*`, `retry-after`) forwarded on sync responses only — streaming responses cannot include HTTP headers after SSE begins
- Retry applies to sync HTTP calls only — streaming connections are not retried (SSE state replay is too complex)
- xAI remains an unregistered extensibility placeholder; `xai` is not a runtime provider until implemented

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
Always scope it to the source files you changed; an unscoped run mutates the
whole tree — slower, and noisy with unrelated survivors:

```bash
CHANGED=$(git diff --name-only HEAD -- '*.py' | rg -v '(^|/)tests?/' | paste -sd, -)
uv run pytest --no-cov --gremlins \
  --gremlin-targets="$CHANGED" --gremlin-no-coverage-filter \
  --gremlin-parallel --gremlin-cache
```

`--gremlin-no-coverage-filter` is required: without it, gremlins consults a coverage
map that does not yet include freshly added lines and reports them as false survivors.
Disabling the filter runs the full selected test set per mutant — accurate, slightly slower.

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
| Tests | 343 | Minimal | Some |

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

### Google (Gemini CLI / Google One AI Premium)

The Gemini provider uses the Gemini CLI OAuth flow (Google One AI Premium subscription).
Per [Google's Terms](https://policies.google.com/terms), using the Code Assist endpoint
through a proxy **may fall outside intended use**. This is the same approach as the OpenAI/Codex
provider — reusing subscription credentials through a local proxy. The Gemini CLI OAuth
credentials (client ID and secret) are intentionally public per
[Google's OAuth documentation for installed applications](https://developers.google.com/identity/protocols/oauth2#installed).

When using `GEMINI_API_KEY` instead, this is a standard API client within normal API terms.

### xAI

When implemented, uses standard API endpoints with your own API keys — straightforward
client implementation within normal API terms.

### Your Responsibility

**No credentials are embedded in this software.** You bring your own auth. You are responsible
for compliance with each provider's terms, costs incurred, and all consequences of usage.

## Disclaimer

**Research project.** Provided as-is for educational and experimental purposes.

- **Not affiliated** with Anthropic, OpenAI, xAI, Google, or any AI company
- **No liability** for API terms violations, service disruptions, data loss, or costs
- **No proprietary code** — translates between publicly documented APIs
- Claude Code banner is rendered by the Claude Code binary (your install, your agreement)

**By using this software, you accept full responsibility for your use.**

## License

[MIT](LICENSE)

---

Built by [axdel](https://github.com/axdel) (with AI, for AI)
