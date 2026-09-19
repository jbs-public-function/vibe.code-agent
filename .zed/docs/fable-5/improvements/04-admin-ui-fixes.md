# 04 — Admin UI Fixes

> Priority: P1 · Depends on: 01-proto-additions (IMP-04.9, 04.13), 02-controller-hardening (IMP-02.25) · Status: Implemented

## Summary

The zero-JS/CSP posture is exactly as specified: no inline `style=`/`<script>` in any of the 11 templates,
exact CSP header, `parse_qsl` form parsing, vendored 96-line CSS. The material issues are security and
robustness edges: state-changing POSTs have no CSRF/Origin defence (loopback binding does not stop a browser
being used as the vector); the chat and `/logs/stream` routes apply a 3 s deadline to the **first** chunk and
therefore abort on slow prefill or a quiet container; `url_for` renders absolute URLs that break behind the
documented Caddy TLS path; and the register flow has no way to recover when the write succeeds but the reload
fails.

## Goals / Non-goals

**Goals**
- Cross-site POSTs are rejected; `Host` is validated.
- Chat streams tolerate long prefill and report upstream errors truthfully; history is truncated server-side at 75 % of `max_model_len`.
- `/logs/stream` follow works on a quiet container.
- Links and form actions are root-relative (proxy-safe).
- Admin-side state duplicating controller knowledge (`known_jobs`, `download_live`) is removed once doc 01 lands.
- Basic ARIA on progress bars, flashes, and per-row buttons (NFR-01.7).

**Non-goals**
- Any JavaScript, WebSockets, or SSE in the admin (ADR-7).
- Authentication (v1 is loopback-only).

## Requirements

### Security (`app.py`, `flash.py`, `sessions.py`)

| ID | Requirement | Sev | Ref |
| --- | --- | --- | --- |
| IMP-04.1 | CSRF/Origin defence in `_SecurityHeaders` (or a sibling middleware): for non-GET/HEAD requests reject unless `Sec-Fetch-Site ∈ {same-origin, none}`; fall back to `Origin`/`Referer` host == `Host`. Validate `Host` ∈ {`127.0.0.1`, `localhost`, `[::1]`, `VIBE_BIND`} to defeat DNS rebinding. Return `403` with an HTML page. | P1 | `app.py:39-63` |
| IMP-04.2 | Extend CSP with `form-action 'self'; frame-ancestors 'none'; base-uri 'none'`. Update the `CSP` constant in `tests/test_admin.py:26`. | P2 | `app.py:36` |
| IMP-04.3 | Cookies set `samesite="lax"` (or `strict`) explicitly; flash text truncated to ~500 chars (4 KiB cookie limit; long `RuntimeError`s are flashed verbatim). | P2 | `flash.py:29-35`; `chat.py:53-59`; `models.py:87` |

### Chat (`routes/chat.py`, `sessions.py`)

| ID | Requirement | Sev | Ref |
| --- | --- | --- | --- |
| IMP-04.4 | Deadlines: first-token budget ≥ 120 s, inter-chunk idle 30 s. The current per-chunk `wait_for(anext(agen), 3.0)` kills any reply whose prefill exceeds 3 s (routine for 6–8 K-token histories on a 14B AWQ model) and mislabels it "controller unreachable". | P1 | `chat.py:128-133` |
| IMP-04.5 | Distinguish error classes mid-stream: `ControllerUnavailable` → "controller unreachable"; `NoModelLoaded`/`UpstreamError`/other `RuntimeError` (e.g. vLLM 400 context-length) → render `html.escape(str(exc))`. Depends on IMP-02.25. | P1 | `chat.py:131-132` |
| IMP-04.6 | Append the user turn **after** the READY/controller checks pass. Today a failed send leaves a dangling user turn; the next send produces consecutive `user` messages, which some chat templates reject. | P1 | `chat.py:176-185` |
| IMP-04.7 | Server-side truncation: before building `proto_messages`, drop oldest non-system turns until `token_estimate() ≤ 0.75 × max_model_len`; note "N older turns omitted" in the footer. The spec's risk item asks for this; only the GET-time warning exists. | P2 | `chat.py:42-44, 78-81` |
| IMP-04.8 | No `yield` inside `finally`. Track `stream_error` in a normal `except`, persist the partial turn in `finally` without yielding, emit the footer after the `try`. Yielding while `GeneratorExit` is pending raises on `aclose()`. | P2 | `chat.py:137-151` |
| IMP-04.9 | Session store bounds: LRU cap (e.g. 256 sessions), TTL from `last_seen` not `created_at`, cap turns per session (e.g. 200). Every cookie-less `GET /chat` mints a session today. | P2 | `sessions.py:37-64` |
| IMP-04.10 | Clamp `temperature` (0–2) and `max_tokens` (1–`max_model_len`) server-side; render inputs as `type="number" min max step`. | P2 | `chat.py:168-175`; `chat.html:15-17` |

### Jobs / logs (`routes/jobs.py`, `routes/logs.py`, `routes/__init__.py`)

| ID | Requirement | Sev | Ref |
| --- | --- | --- | --- |
| IMP-04.11 | `/logs/stream` follow on a quiet container: default `tail=50` so the first chunk is a replay, or remove the first-line deadline and verify channel readiness with `client.status()` under `grpc()` before iterating. Today `tail=0` + 3 s first-line deadline prints "controller unreachable" and closes — the "follow stream" link in `logs.html:16` uses exactly this path (FR-01.7 ❌). | P1 | `logs.py:56-72` |
| IMP-04.12 | `tail` accepted as `str` and coerced manually so `?tail=abc` renders an HTML error instead of FastAPI's 422 JSON. | P2 | `logs.py:32` |
| IMP-04.13 | Delete `app.state.known_jobs`; read kind/alias from `JobEvent.kind/alias` or `GetJob` (IMP-01.1/01.2). Until then, fall back to the first event's message. | P2 | `jobs.py:67-70`; `__init__.py:137-138` |
| IMP-04.14 | Keep meta-refresh on while the controller is down (`refresh = (is_current or controller_down) and not terminal`); render an "outcome unknown" note for a job that is non-terminal but no longer `current_job`. | P2 | `jobs.py:53-56, 95` |
| IMP-04.15 | `grpc()`/`collect()` catch only `ControllerUnavailable`/`TimeoutError` as "unreachable"; other `RuntimeError`s surface as a distinct error banner with the message. `aclose()` the generator on first-`anext` timeout (also `developer.py:166`, `chat.py:128`, `logs.py:66`). Catch `StopAsyncIteration` in `_start_job`/`models.py:69-87` (empty stream → 500 today). | P2 | `__init__.py:46, 64, 120` |
| IMP-04.16 | Parallelise independent controller calls with `asyncio.gather` (`/models`: 2 calls → 6 s worst case; `/developer`: 4 calls → 12 s) to honour the 3 s NFR-01.5 budget. | P2 | `models.py:37-38`; `developer.py:195-198` |

### Developer / register (`routes/developer.py`, templates)

| ID | Requirement | Sev | Ref |
| --- | --- | --- | --- |
| IMP-04.17 | Standalone **Reload registry** button: `POST /developer/reload` on `/developer` and on the register error page. FR-04.7d asks for it; today write-succeeds-then-reload-fails leaves `models.yaml` modified with no in-UI recovery. | P1 | `developer.py:462-476` |
| IMP-04.18 | Form parse errors (`"80a0"`, `"0.7x"`) re-render the form with messages instead of coercing to `0` ("auto"). | P2 | `developer.py:86-96` |
| IMP-04.19 | Print the controller's `yaml_preview` (IMP-01.5) verbatim; delete `_preview_yaml` (duplicated serialisation, XR-3). | P2 | `developer.py:118-144`; `register_preview.html:23` |
| IMP-04.20 | Delete `download_live`/`download_started` once `DownloadInfo` carries `current_file`/timestamps (IMP-01.4). Meanwhile evict entries absent from the controller's recent list and initialise all side-channel dicts in `build_app`. | P2 | `developer.py:168-175, 281-288`; `app.py:104-105` |
| IMP-04.21 | `cancel_download`/`remove_cached` catch `KeyError` (NOT_FOUND) → flash "unknown download/repo" instead of 500. | P2 | `developer.py:315, 350` |
| IMP-04.22 | `register_confirm` re-proposes and diffs against the previewed values (or compares `write.model`) so alias/port cannot drift between preview and write. | P2 | `developer.py:441-443` |
| IMP-04.23 | Pass `_QUANTIZATION_OPTIONS`/`_TOOL_PARSER_OPTIONS` into the template instead of hardcoding the lists twice. | P2 | `developer.py:44-45`; `register_preview.html:38, 44` |

### Dashboard (`routes/dashboard.py`, `app.py`)

| ID | Requirement | Sev | Ref |
| --- | --- | --- | --- |
| IMP-04.24 | Gateway probe URL: map wildcard binds (`0.0.0.0`, `::`) to loopback, bracket IPv6, read the port from `GATEWAY_PORT`; catch `Exception` (`httpx.InvalidURL` is not an `HTTPError`). Make URL/transport injectable via `app.state` so tests stop hitting the real `127.0.0.1:8080`. Probe timeout 0.5 s. | P2 | `app.py:107`; `dashboard.py:27-35` |
| IMP-04.25 | Label the bar "memory" (it renders `mem_pct`, not utilization) or render both; add `.badge-unknown` CSS. | P2 | `dashboard.py:59-70`; `vibe.css:57-60` |

### Templates / a11y

| ID | Requirement | Sev | Ref |
| --- | --- | --- | --- |
| IMP-04.26 | Root-relative URLs everywhere: register a `url_path_for`-style Jinja global (`request.app.url_path_for`) and replace `url_for`. Starlette `url_for` yields absolute URLs from the request scope; behind Caddy TLS without trusted `X-Forwarded-*` every link/form becomes `http://127.0.0.1:8081/...`. Alternatively document `uvicorn --proxy-headers --forwarded-allow-ips`. | P1 | `base.html:7, 13-21`; all templates |
| IMP-04.27 | `developer.html:58` uses `url_path_for('developer_register_form')` + `{{ entry.repo_id|urlencode }}` (raw href, unencoded query today). | P2 | `developer.html:58` |
| IMP-04.28 | `HEAD` support: middleware rewriting HEAD→GET with body dropped, or `methods=["GET","HEAD"]` on page routes (`curl -I /` returns 405 today). | P2 | `app.py:111-116` |
| IMP-04.29 | ARIA: `role="progressbar" aria-valuenow/min/max` on bars; `role="status"`/`role="alert"` on flash/banner; `aria-label="Download {{ alias }}"` on per-row buttons; warnings use a `--warn` class rather than `p.dim`. | P2 | `dashboard.html:26`; `download.html:17`; `models.html:45-48`; `register_preview.html:11` |
| IMP-04.30 | Rename the `mib` filter to `thousands` (it only adds separators) and use `_fmt_size` for byte counts; interpolate `GATEWAY_PORT` into the footer instead of the literal `http://localhost:8080/v1`. | P2 | `download.html:18`; `models.html:39`; `base.html:29` |

## Design notes

- **CSRF middleware** is ~30 lines of ASGI; keep it in `admin/security.py` next to `_SecurityHeaders`. Tests: same-origin POST passes; POST with `Origin: http://evil.example` → 403; missing all headers (curl) → allowed only when `Sec-Fetch-Site` absent **and** `Origin` absent (CLI/test convenience) — decide and document.
- **Chat deadlines**: two constants `CHAT_FIRST_TOKEN_S = 120`, `CHAT_IDLE_S = 30` in `chat.py`; the padding heartbeat keeps the browser connection alive during prefill.
- **Truncation**: `token_estimate` already exists; truncation is a pure function `truncate_history(messages, budget) -> (messages, dropped)` — unit-testable.
- **Root-relative URLs**: a single Jinja global `path_for = request.app.url_path_for` and a test asserting no `href="http` in rendered pages except the gateway footer.

## Task breakdown

- [x] IMP-04.1–04.3 — security middleware, CSP, cookie attrs.
- [x] IMP-04.4–04.10 — chat route + sessions.
- [x] IMP-04.11, 04.12 — logs route.
- [x] IMP-04.13, 04.14 — jobs page without `known_jobs`; refresh/unknown rules.
- [x] IMP-04.15, 04.16 — helpers + gather.
- [x] IMP-04.17, 04.18, 04.21, 04.22, 04.23 — developer/register flow.
- [x] IMP-04.24, 04.25 — dashboard.
- [x] IMP-04.26–04.30 — templates, HEAD, a11y.
- [x] After doc 01: IMP-04.13, 04.19, 04.20 — remove side channels and `_preview_yaml`.
- [x] Tests per `07-testing-and-acceptance.md` §Admin.

## Acceptance criteria

- `curl -X POST -H 'Origin: http://evil.example' http://127.0.0.1:8081/models/unload` → 403; the same POST from the UI succeeds.
- Chat with a 7 K-token history on `deepseek-r1-14b` streams to completion (first token > 3 s).
- `/logs/stream?follow=1` on an idle READY container stays open and prints the next log line when it arrives.
- Rendered pages contain no absolute `http://` self-links; the UI navigates correctly behind `make tls` (`https://localhost:8444`).
- Register flow: simulate `reload_registry` failure → error page offers "Reload registry" which succeeds on retry.
- `curl -I http://127.0.0.1:8081/` → 200.
- `test_pages_are_script_free_and_carry_security_headers` extended to every route and asserts `style=` absent.

## Risks

- Strict `Host` validation can break access via a LAN hostname when `VIBE_BIND` is widened; document the allow-list and derive it from settings.
- Longer chat deadlines mean a truly dead controller is detected later; mitigated by the `ControllerUnavailable` fast path (gRPC UNAVAILABLE is immediate).
- Root-relative URLs assume the admin is served at `/` (true for Caddy `:8444 → :8081`; a sub-path mount would need `root_path`).
