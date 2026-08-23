# xAI (Grok) wire fixtures — golden captures

**Provenance:** captured live from `POST https://cli-chat-proxy.grok.com/v1/responses`
(the grok CLI's subscription-metered proxy) on 2026-07-10, model alias `grok-4.20`
(resolves upstream to `grok-4.20-0309-reasoning`), request header
`x-grok-client-version: 0.2.93`. These are **real response bytes**, not hand-authored —
the xAI provider's translation is asserted against the captured wire shape, never a
mental model of it (Boundary Fixture Fidelity).

**Sanitization:** request `Authorization` headers were never saved. Response bodies are
model output to synthetic prompts and carry no credentials or PII (scanned). Response
`id` / `system_fingerprint` values are upstream request identifiers, not secrets.

**Oracle discipline:** tests assert the wire **contract** — object/field names, id
formats, event sequence, `usage` key set, terminal shapes — not the model's generated
text (which is non-deterministic). The fixtures pin structure.

| Fixture | Captures | Used by |
|-|-|-|
| `text_nonstream.json` | Base `response` object: `output[]` = reasoning (`rs_`) + message (`msg_`), `usage` shape, no `cost_in_usd` (subscription-metered) | response translation |
| `text_stream.txt` | Full SSE lifecycle (25 events): `response.created` → reasoning deltas → message deltas → `response.completed` (carries `usage`) | stream translation |
| `single_tool_call.json` | One `function_call`: `call_id` = `call-<uuid>-<idx>`, item `id` = `fc_<resp>_<idx>`, `arguments` JSON string | request/response tool translation |
| `parallel_tool_calls.json` | Two parallel `function_call`s sharing a `call-<uuid>` base, differing by `-0`/`-1` index suffix; item ids `fc_<resp>_0/_1` | reasoning/tool continuity (ordering) |
| `reasoning_encrypted.json` | `include:["reasoning.encrypted_content"]` + `store:false` → reasoning item carries `encrypted_content` blob | reasoning continuity |
| `incomplete_max_tokens.json` | `status:"incomplete"`, `incomplete_details:{"reason":"max_output_tokens"}`; still carries completed reasoning + message | stream/response incomplete terminal |
| `tool_result_replay_exact.json` | Multi-turn: replaying `function_call_output` with the **exact** `call_id` → model consumes the result ("18°C and sunny") | request continuity |
| `image_input.json` | `input_image` via `data:image/png;base64,…` accepted (≥8px/dim AND ≥512 total px required) → model describes it | media forwarding |
| `field_effort_low.json` | **Negative:** `reasoning.effort` → `400 "Model grok-4.20 does not support parameter reasoningEffort."` — the oracle for the version gate: pre-4.6 models 400, so effort is sent only to grok-4.6+ and omitted below | request translation (model-gated `reasoning.effort`) |
