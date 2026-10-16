# 03 — Gateway: Jev Enforcement

**Priority:** P1 · **Depends:** 02 · **Status:** Implemented · **Phase:** J2

## Summary

When the active model has a Jev schema attached, the gateway rewrites `POST /v1/chat/completions` bodies so vLLM
constrains decoding to that schema (`response_format: json_schema`, `strict: true`). With Jev inactive, behaviour is
**byte-for-byte unchanged** (ADR-6, JEV-X.8). The gateway never reads the schema store; it learns the active schema
from the controller (`Status.jev` + `GetActiveJev`).

## Goals / Non-goals

**Goals**
- Streaming and non-streaming both work (injection happens before the request is sent; SSE passthrough unchanged).
- Clear, typed rejection of requests that can't coexist with a forced schema.
- Jev visible to clients in `/v1/models` and `/health`.

**Non-goals**
- Validating responses in the gateway (streamed SSE makes it costly; deferred, README open question).
- Injecting instructions into `messages` (follow-up `JEV-FU.1`).
- `/v1/completions` support (needs a different vLLM param family; out of scope for v1).

## Request semantics in Jev mode

Applies only when **all** hold: path is `/v1/chat/completions`, the routing decision is `Proxy` (not `Swap`), and
`model == status.active_alias` with `status.HasField("jev")`.

| Client sent | Gateway does |
| --- | --- |
| nothing special | sets `response_format = {type:"json_schema", json_schema:{name, schema: wire_schema, strict:true}}` |
| its own `response_format` | **overwrites** it (Jev is authoritative); log `jev_override` at info; add response header `X-Vibe-Jev-Overrode: response_format` |
| `guided_json` / `guided_choice` / `guided_regex` / `guided_grammar` / `structured_outputs` | **removed** (vLLM rejects multiple constraints) with the same log + header |
| `tools` non-empty, or `tool_choice` not in (absent, `"none"`) | **400** `jev_incompatible_request`: "Jev is active (<name>); tool calling is disabled. Detach with `vibe jev off`." |
| path `/v1/completions` | **400** `jev_incompatible_request` (same hint) |
| path `/v1/embeddings` | pass through untouched |
| `stream: true` | unchanged; SSE relayed as today |
| `max_tokens` absent | untouched (vLLM defaults to remaining context) |

Always add response header `X-Vibe-Jev: <name>; sha=<sha256[:12]>` on proxied responses in Jev mode so clients and
tests can tell which schema produced the output. The header is added **only** in Jev mode (JEV-X.8).

## Requirements

| ID | Requirement | Sev |
| --- | --- | --- |
| JEV-03.1 | `gateway/jev.py` (pure, no IO): `apply_jev(data: dict, schema: dict, name: str) -> JevRewrite(data, overrode: list[str])` and `check_compatible(path, data) -> ErrorSpec | None`. Unit-testable without HTTP. Uses `vibe.jev.wire_schema` / `schema_name_for_openai` (import of the pure package is allowed; no store access). | P1 |
| JEV-03.2 | `_SnapshotCache` additionally holds `jev_body: (sha256, dict) | None`. When `status.jev.sha256` changes (or becomes set), fetch `GetActiveJev` once under the snapshot lock; cache by sha. If the fetch fails or the returned sha differs from `Status.jev.sha256`, **fail closed**: respond 503 `jev_unavailable` + `Retry-After: 5` rather than proxy unconstrained (a Jev client expects JSON). | P1 |
| JEV-03.3 | `post_endpoint`: after `decide(...)` returns `Proxy` and Jev applies, build the new body with `json.dumps(data, separators=(",", ":"))` and pass it to `proxy_or_502`. Everything else (decision log, 502 handling, swap flow) unchanged. Swap path: after a swap the controller has cleared Jev; call `cache.invalidate()` (already done) so the next request sees plain mode. | P1 |
| JEV-03.4 | `stream_proxy` must drop any inbound `content-length` header (and let httpx recompute from `content=body`) — today the client's value is forwarded verbatim, which is wrong once the body is re-serialised. Add a unit test that fails if a stale `Content-Length` is sent. This fix is correct for plain mode as well (identical length there), so it does not violate JEV-X.8. | P1 |
| JEV-03.5 | Response header injection: `X-Vibe-Jev`/`X-Vibe-Jev-Overrode` set on the `StreamingResponse` built by `stream_proxy` via an optional `extra_headers` argument (default `None` ⇒ no change). | P2 |
| JEV-03.6 | `/v1/models`: the active entry gains `"vibe_jev": {"name", "sha256", "source", "stale"}`; every other entry (and the active one when Jev is off) has `"vibe_jev": null`. `build_models_payload` takes the `Status` it already receives. | P2 |
| JEV-03.7 | `/health` gains `"jev": null | {"name","sha256"}`; `null` when the controller is down. Existing keys unchanged. | P2 |
| JEV-03.8 | Structured logs (`vibe.gateway`): `jev_applied` (alias, schema, sha, elapsed_ms), `jev_override`, `jev_rejected` (reason), `jev_unavailable`. No schema body or prompts in logs. | P2 |
| JEV-03.9 | If vLLM answers 400 to an injected request (backend rejects schema at request time even though the probe passed), relay it verbatim **and** invalidate the snapshot (so a concurrent detach is noticed). No automatic retry without the schema (that would silently violate Jev). | P2 |
| JEV-03.10 | Body-size limit (`BODY_LIMIT`) is applied **before** injection; the injected schema (≤ 64 KiB) may push a near-limit body over the limit — acceptable, upstream enforces its own. Note in code comment. | P2 |
| JEV-03.11 | `vibe zed-settings` and the Zed docs gain one sentence: with Jev active the Zed agent panel's tool calls will receive `jev_incompatible_request`; detach first. No settings change. | P2 |

## Design notes

- **Why overwrite rather than respect the client's `response_format`?** A Jev-mode deployment's contract is "every
  answer is schema-valid". Respecting client formats would make the guarantee depend on each client. The override
  is surfaced (header + log) so it is never silent.
- **Why fail closed?** Proxying unconstrained would return free text to a caller that will `json.loads` it. A 503
  with `Retry-After` follows the existing retry conventions of the gateway.
- **Re-serialisation** changes whitespace/key order of the request only; vLLM doesn't care. JSON numbers are
  re-emitted by Python — floats such as `1e-7`/large ints round-trip, but note `json.loads` → `dumps` of very large
  integers is lossless in Python; NaN/Infinity would be rejected upstream either way.
- **Snapshot staleness**: ≤ 5 s between a detach/attach and the gateway noticing. Acceptable for an operator
  action; the CLI/admin print "gateway picks this up within 5 s". If sub-second matters later, add a server-streaming
  `WatchStatus` RPC (not in v1).
- **Interaction with swap-on-request**: a `Swap` decision implies a fresh container ⇒ plain mode after the swap.
  Jev is **never** carried across swaps.

## Task breakdown

- [x] `gateway/jev.py` pure helpers + tests.
- [x] `ControllerClient.get_active_jev()` consumer; cache-by-sha in `_SnapshotCache`; fail-closed path.
- [x] `post_endpoint` wiring; compatibility checks; completions rejection.
- [x] `stream_proxy`: drop inbound `content-length`; `extra_headers`.
- [x] `/v1/models` + `/health` markers; update existing payload tests (`vibe_jev: null` everywhere).
- [x] Logging.
- [x] Docs: `features/03-openai-gateway.md` gets a "Jev mode" pointer; README Troubleshooting row for
      `jev_incompatible_request`.

## Acceptance criteria

- Jev off: golden-file test shows the proxied body bytes **identical** to the client's (existing tests unchanged).
- Jev on (`default`): a chat request without `response_format` reaches a fake upstream with the expected
  `response_format`; a request with its own `response_format` + `guided_json` arrives with only the Jev one; the
  response carries `X-Vibe-Jev`.
- `tools` present ⇒ 400 `jev_incompatible_request`, upstream not called.
- Controller returns a mismatched sha for `GetActiveJev` ⇒ 503 `jev_unavailable`, upstream not called.
- Streaming request in Jev mode relays SSE chunks unmodified.

## Risks

| Risk | Mitigation |
| --- | --- |
| Zed (or other client) sends `tools` by default → every request 400s in Jev mode | Clear error message + hint; documented; Jev is meant for programmatic callers |
| Backend honours `response_format` but not `strict` | vLLM ignores `strict`; harmless; keep for OpenAI-client parity |
| Truncation at `max_tokens` yields invalid JSON | Not detectable pre-flight; probe warns on unbounded schemas; docs recommend bounds. Follow-up: gateway adds `X-Vibe-Jev-Truncated` when `finish_reason=length` (non-stream only) |
| Re-serialisation edge cases | Only the dict path; fall back to **400 invalid JSON** exactly as today if parsing fails (parsing already happens) |
