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

Then one day Claude is overloaded. Or rate-limited. Or you just want to see how GPT-5.4
handles the same task with the same tools.

**Without Claude Bridge:** your entire setup is useless. Claude Code only talks to Anthropic.

**With Claude Bridge:** one command, same setup, different model.

```bash
claude-codex    # your Claude Code + GPT-5.4
```

Everything works — tools, hooks, skills, streaming, multi-turn tool conversations — because
the bridge translates the Anthropic Messages API to the provider's native format on-the-fly.

## How It Works

```
Claude Code  -->  Claude Bridge (localhost:9999)  -->  Anthropic (passthrough)
                           |
                     circuit breaker
                           |
                     Provider adapter  -->  OpenAI / Grok / Gemini / ...
```

1. Claude Code sends an Anthropic Messages API request to `localhost:9999`
2. The bridge translates it to the target provider's format (e.g., OpenAI Responses API)
3. The provider responds (streaming SSE or sync JSON)
4. The bridge translates the response back to Anthropic format
5. Claude Code receives it as if Anthropic answered — tools, streaming, everything works

Full fidelity: `tool_use` ↔ `function_call`, `tool_result` ↔ `function_call_output`,
streaming SSE events mapped one-to-one, tool IDs translated (`toolu_` ↔ `fc_`).

## Features

- **Zero dependencies** — stdlib-only Python, no `pip install`
- **Standard API key auth** — set `OPENAI_API_KEY` for the official OpenAI Responses API
- **Codex OAuth fallback** — no API key? Falls back to Codex OAuth automatically
- **Reasoning passthrough** — thinking blocks preserved by default, not silently stripped
- **Auto-failover** — circuit breaker routes to fallback on Anthropic 429/500/502/503
- **Retry with backoff** — transient HTTP errors retried once with 0.5s exponential backoff
- **Mid-turn failover guard** — blocks provider switch during active tool-use turns
- **Direct mode** — skip Anthropic entirely, always use a specific provider
- **Health check** — `/health` endpoint for liveness probes and process managers
- **Structured logging** — request IDs, provider/model identity, log levels (`LOG_LEVEL=DEBUG`)
- **Metrics** — `/stats` endpoint: request count, errors, latency, tokens, provider, uptime
- **Token estimation** — structure-aware byte counting for context window management
- **Multi-provider** — adding a provider = one file, zero proxy changes
- **186 tests** — 87% coverage enforced, type-checked with basedpyright, linted with ruff

## Prerequisites

### Accounts

- **Anthropic account** — for Claude Code ([console.anthropic.com](https://console.anthropic.com/))
- **ChatGPT Plus** ($20/mo) — for the OpenAI/Codex provider (provides OAuth access)
- **Google AI Studio** (free) — for the Gemini provider ([aistudio.google.com/apikey](https://aistudio.google.com/apikey))

### Software (macOS)

```bash
brew install python claude-code codex

# Verify
python3 --version    # 3.12+
claude --version
codex --version

# Authenticate Codex with your ChatGPT Plus account
codex login
cat ~/.codex/auth.json   # should show access_token
```

> **macOS only** for now (brew dependencies). Linux support is untested.
> **No `pip install` needed** — the bridge is stdlib-only Python.
> Codex CLI is only for the OpenAI provider — future providers (Grok, Gemini)
> use standard API keys.

## Install

```bash
git clone https://github.com/axdel/claude-bridge.git
cd claude-bridge

# Make the launcher available system-wide
mkdir -p ~/.local/bin
ln -sf "$(pwd)/claude-codex" ~/.local/bin/claude-codex

# Verify
which claude-codex || echo 'Add to PATH: echo "export PATH=\$HOME/.local/bin:\$PATH" >> ~/.zshrc && source ~/.zshrc'
```

## Usage

### One command (recommended)

```bash
claude-codex
```

You'll see:
```
      _                 _                         _
  ___| | __ _ _   _  __| | ___       ___ ___   __| | _____  __
 / __| |/ _` | | | |/ _` |/ _ \ ___ / __/ _ \ / _` |/ _ \ \/ /
| (__| | (_| | |_| | (_| |  __/|___||(_| (_) | (_| |  __/>  <
 \___|_|\__,_|\__,_|\__,_|\___|     \___\___/ \__,_|\___/_/\_\

 port:9472  pid:12345  model:gpt-5.4
 by axdel  github.com/axdel/claude-bridge

 ▐▛███▜▌   Claude Code v2.1.80
▝▜█████▛▘  Sonnet 4.6 with high effort
  ▘▘ ▝▝    ~/your-project

❯ ready for work
```

> Claude Code's banner still says "Sonnet 4.6" — it doesn't know about the bridge.
> The actual model answering is GPT-5.4.

The bridge starts on a random port, launches Claude Code through it, and cleans up on exit.

### Options

```bash
claude-codex              # OpenAI/Codex (GPT-5.4)
claude-codex --debug      # show bridge translation logs
claude-codex -- -p opus   # pass flags through to claude
```

Each provider gets its own launcher:
```bash
claude-codex       # OpenAI GPT-5.4 via Codex
claude-gemini      # Google Gemini (gemini-2.5-pro)
# claude-xai      # xAI Grok (coming soon)
```

### Verify it works

After launching with `claude-codex`, paste this into Claude Code:

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

**Fallback chain** — control the failover order:
```bash
LLM_BRIDGE_FALLBACK=openai,xai ./start.sh
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
    "model": "gpt-5.4"
}
```

## Configuration

| Env Var | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | _(none)_ | OpenAI API key — uses standard Responses API when set |
| `GEMINI_API_KEY` | _(none)_ | Google Gemini API key — get from [AI Studio](https://aistudio.google.com/apikey) |
| `GEMINI_MODEL` | `gemini-2.5-pro` | Default Gemini model (override to use gemini-2.5-flash, gemini-3.1-pro-preview, etc.) |
| `REASONING_MODE` | `passthrough` | `passthrough` preserves thinking blocks, `drop` strips them |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `UPSTREAM_TIMEOUT` | `60`/`120` | Upstream request timeout in seconds (60s sync, 120s streaming) |
| `MAX_REQUEST_BODY` | `10485760` | Maximum request body size in bytes (default 10MB) |
| `LLM_BRIDGE_FALLBACK` | `openai` | Comma-separated fallback provider chain |
| `LLM_BRIDGE_PORT` | `9999` | Default proxy port |
| `ANTHROPIC_REAL_URL` | `https://api.anthropic.com` | Real Anthropic endpoint (passthrough) |

## Architecture

```
src/claude_bridge/
├── proxy.py          # HTTP server, routing, streaming, /stats, /health, retry
├── provider.py       # Provider protocol (abstract interface)
├── router.py         # Circuit breaker (CLOSED/OPEN/HALF_OPEN)
├── stats.py          # Thread-safe metrics counters
├── log.py            # Structured logging with request IDs
├── auth.py           # JWT decode, token expiry
├── stream.py         # SSE parsing/formatting utilities
└── providers/
    ├── openai.py     # OpenAI Codex: OAuth + Anthropic <-> Responses API
    ├── gemini.py     # Google Gemini: API key + Anthropic <-> generateContent API
    └── xai.py        # xAI Grok: stub
```

### Adding a New Provider

1. Create `src/claude_bridge/providers/yourprovider.py`
2. Implement the `Provider` protocol:
   - `authenticate()` — return auth headers
   - `translate_request()` — Anthropic -> your format
   - `translate_response()` — your format -> Anthropic
   - `translate_stream()` — raw bytes -> Anthropic SSE events
3. Register: `PROVIDERS["yourprovider"] = YourProvider`
4. Import in `__main__.py`
5. Use: `./start.sh --provider yourprovider`
6. Optionally copy `claude-codex` -> `claude-yourprovider` (change `--provider`)

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
> with `call_id`/`id` separation. Other providers will have their own translation maps.

## Known Limitations

- Claude Code's startup banner always shows "Sonnet 4.6" regardless of actual model
- `thinking` blocks are passed through as tagged text (no native OpenAI equivalent) — set `REASONING_MODE=drop` to strip
- `output_config` and `cache_control` hints are stripped with a warning
- Token estimation is approximate (~bytes/3.5), not exact tokenization
- Streaming stats don't include token counts (only latency)
- Failover is blocked during active tool-use turns (by design — prevents broken tool state)
- Rate limit headers (`x-ratelimit-*`, `retry-after`) forwarded on sync responses only — streaming responses cannot include HTTP headers after SSE begins
- Retry applies to sync HTTP calls only — streaming connections are not retried (SSE state replay is too complex)

## Running Tests

```bash
pip install uv              # if you don't have uv
cd claude-bridge
uv run pytest tests/ -v     # installs test deps on first run, shows coverage
```

No external services — all 186 tests use mock HTTP servers. Coverage is enforced at 85% (currently 87%).

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
| Tests | 186 | Minimal | Some |

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

### xAI / Google

When implemented, these use standard API endpoints with your own API keys — straightforward
client implementations within normal API terms.

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
