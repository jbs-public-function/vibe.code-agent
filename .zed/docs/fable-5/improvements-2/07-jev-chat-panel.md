# 07 — Admin: Jev Chat Panel

**Priority:** P1 (user-requested feature) · **Depends:** 03 (gateway enforcement), 05 (admin Jev pages) · **Status:** Implemented — unit suite green (`make check`: 734 passed, 1 skipped, pyright 0 errors) and verified live against `qwen2.5-coder-7b` with Jev `default` attached (schema-valid verdict, usage tokens, `X-Vibe-Jev` header, `stream_options.include_usage` honoured by vLLM) · **Phase:** J6

## Summary

A new zero-JS admin panel at `/jev/chat` that sends a prompt **through the gateway** and streams the model's
answer back, so the active Jev schema is enforced exactly as it is for any OpenAI-compatible client (ADR-J2:
one enforcement point). When the stream ends the panel renders an **inspected result**: the assembled JSON
pretty-printed, a schema-validity verdict (computed with the same `jsonschema` dialect mapping the activation
probe uses), the `finish_reason` (with a truncation warning for `length`), token usage, and the
`X-Vibe-Jev` header the gateway stamped on the response.

The panel is the natural companion to the `/jev` schema pages: edit a schema, launch a model with it, then
exercise it interactively and see whether (and why) each answer validates.

## Goals / Non-goals

**Goals**

- Prompt the **active model** and stream its Jev-constrained JSON output using the existing chunked-HTML
  streaming machinery (two-phase render, heartbeat padding, first-token/idle budgets, no `yield` in
  `finally`).
- After the stream: show the assembled JSON, validity against the active schema, `finish_reason`, usage, and
  which schema was enforced (`X-Vibe-Jev`).
- History-carrying conversations with a session **independent** of the existing `/chat` panel.
- Graceful degradation: no model, no Jev attached, controller down, gateway down, gateway rejection — each has
  a distinct, non-500 response.

**Non-goals**

- Changing the existing `/chat` panel to route through the gateway (it keeps the controller `Chat` RPC; noted
  as a follow-up).
- Making the controller `Chat` RPC Jev-aware (enforcement stays gateway-only, ADR-J2).
- Attaching/launching schemas from the panel (the `/jev` page owns that; the panel links to it).
- Inline schema editing in the panel.
- Gateway-side non-stream validation (`X-Vibe-Jev-Valid`, follow-up `JEV-FU.2`) — the panel validates
  client-side for display only.

## Requirements

| ID | Requirement | Sev |
| --- | --- | --- |
| JEV-07.1 | `GET /jev/chat` renders: active-Jev panel (name/source/sha/stale, link to the editor), a "no schema attached — unconstrained" note when Jev is off, the gateway-down banner, the history, and the send/reset forms. Defaults: `temperature 0.7`, `max_tokens 512` (clamped server-side). | P1 |
| JEV-07.2 | `POST /jev/chat` sends `POST {gateway}/v1/chat/completions` with `model = active alias`, `messages` (history + the new user turn), `stream: true`, `stream_options: {"include_usage": true}`, `temperature`, `max_tokens`. **No `response_format`** — the gateway injects the active schema (single enforcement point, JEV-03.1). | P1 |
| JEV-07.3 | The SSE stream is relayed to chunked HTML exactly like `/chat`: header flushed first (with ~1 KiB padding), deltas HTML-escaped, 120 s first-token budget, 30 s idle budget, 5 s heartbeat, controller/gateway death mid-stream renders an inline error block. | P1 |
| JEV-07.4 | Verdict block after the stream: assembled reply pretty-printed in `<pre>` (escaped); when a Jev schema was attached at send time, validate with `instance_validator_for` (same dialect mapping as `probe_jev`) and show valid/invalid plus up to 3 errors; `finish_reason == "length"` shows a truncation warning; usage tokens and the `X-Vibe-Jev` header are displayed. | P1 |
| JEV-07.5 | A non-200 gateway answer (400 `jev_incompatible_request`, 503 `jev_unavailable`, 502 `upstream_error`, controller-down) flashes the gateway's message and redirects back to `/jev/chat`; the user turn is **not** persisted (IMP-04.6 pattern). | P1 |
| JEV-07.6 | Sessions reuse `ChatSession`/`SessionStore` under a distinct cookie `vibe_jev_session`; `POST /jev/chat/reset` clears only that session. History truncation at 75 % of context reuses `truncate_history`. | P1 |
| JEV-07.7 | Zero-JS (JEV-X.5): plain forms, POST-redirect-GET except the stream, all dynamic text HTML-escaped; the shared CSP/nosniff headers and the `SecurityMiddleware` same-origin checks apply (cross-origin POST → 403). | P1 |
| JEV-07.8 | Controller unreachable → the standard banner + POST flash (NFR-01.5); gateway unreachable (probe or connect failure) → banner on GET, flash on POST, never a 500. | P1 |
| JEV-07.9 | Unit tests (no GPU/Docker): render, stream concatenation + validity verdict, truncation warning, gateway error/unreachable flashes, controller-down, HTML escaping, CSP/zero-JS guard, cross-origin 403, reset, session-cookie isolation from `/chat`, payload shape (no `response_format`, `stream_options` present, clamps). | P1 |
| JEV-07.10 | Navigation: a "Jev chat" link in the base nav and a link from the `/jev` page. | P2 |

## Design notes

- **Why through the gateway, not the controller `Chat` RPC?** The controller's `Chat` proxies vLLM directly
  and is Jev-unaware by design; enforcement lives only in the gateway (`gateway/app.py` post_endpoint,
  ADR-J2). Routing the panel through the gateway means it observes exactly what Zed or any CLI client
  observes — same injection, same rejections, same `X-Vibe-Jev` header — with zero controller changes. The
  admin is allowed to talk HTTP to the gateway (it already probes `/health` on the dashboard); XR-5/JEV-X.4
  only forbid touching Docker/NVML or the schema store directly.
- **Streaming vs one-shot**: streamed, because a cold deepseek first token can take tens of seconds and the
  heartbeat padding keeps the browser painting (NFR-01.6). The SSE lines are parsed with the same
  `data:`/`[DONE]` rules as `servicer.Chat`.
- **Verdict placement**: validity is computed only after the stream ends (the schema that was active at send
  time is fetched up front via `GetActiveJev` + `GetJevSchema`; a mid-stream re-attach is noted, not chased).
  The gateway already re-serialises the injected request, so `stream_options` passes through untouched.
- **Gateway URL**: derived from `app.state.gateway_health_url` (bind-aware, IPv6-safe) with `/health` replaced
  by `/v1/chat/completions`; tests inject `app.state.gateway_transport` (the dashboard's hook) to fake it.
- **JSON extraction**: the reply is parsed directly; if that fails and the text is fenced
  (```` ```json … ```` ```` ``` ````), the fence body is retried — plain-mode answers are often wrapped, Jev-mode answers
  are not.
- **Failure ordering** (IMP-04.6): status checks → open the gateway stream → only then persist the user turn
  and render. A gateway rejection therefore never leaves a dangling user turn.

## Task breakdown

- [x] `src/vibe/admin/routes/jev_chat.py` — `GET /jev/chat`, `POST /jev/chat`, `POST /jev/chat/reset`; SSE
      parse + chunked render; verdict computation; flash paths.
- [x] `src/vibe/admin/templates/jev_chat.html` and `jev_chat_stream.html` (two-phase, verdict block).
- [x] Register the router in `admin/app.py`; nav link in `base.html`; link from `jev.html`.
- [x] `tests/test_admin_jev_chat.py` (JEV-07.9 list).
- [x] `make check` green; record outcomes in the README status line and this doc's Status.

## Acceptance criteria

- With a model READY and Jev `default` attached: POSTing a prompt streams escaped deltas and ends with a
  "schema-valid" verdict, pretty-printed JSON, `finish_reason`, usage, and `X-Vibe-Jev: default; sha=…`.
- A deliberately invalid reply renders "INVALID" with up to 3 schema errors; `finish_reason=length` shows the
  truncation warning.
- Gateway answering 400/503 (or unreachable) → 303 back to `/jev/chat` with the gateway's message; the user
  turn is not persisted.
- No Jev attached → banner "unconstrained", verdict shows no validity badge but still pretty-prints parseable
  JSON.
- Cross-origin POST → 403; every page/stream carries the unchanged CSP; no `<script>` anywhere.
- `POST /jev/chat/reset` clears only the `vibe_jev_session` conversation; the `/chat` panel's
  `vibe_session` is untouched.

## Risks

| Risk | Mitigation |
| --- | --- |
| `vibe serve --only admin` has no gateway — the panel can't work | Banner + flash "gateway unreachable"; the page stays rendered (degradation, never 500) |
| Long generations stall the browser | Same budgets as `/chat`: 120 s first token, 30 s idle, 5 s heartbeat padding |
| A mid-stream schema re-attach makes the verdict stale | Verdict labels the schema name + sha that was active at send time |
| Fenced/plain-mode answers | Fence-strip fallback; validity badge absent when no schema was attached |
| Model emits invalid JSON despite the schema (truncation, backend quirk) | That is exactly what the verdict surfaces; `length` gets an explicit warning |
| Panel requests trigger `VIBE_SWAP_ON_REQUEST` swaps | Impossible for the active alias — `decide()` returns `Proxy` for the model that is already loaded |
