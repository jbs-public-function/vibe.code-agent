# 06 — Jev Testing Plan and GPU Acceptance Runbook

**Priority:** P1 · **Depends:** 01–05 · **Status:** Planned · **Phase:** J5

Baseline before starting: `uv run --offline pytest -q` → 475 passed, 1 skipped; `uv run pyright` clean;
`make check` green. Always `export UV_CACHE_DIR=/home/jamesb/projects/vibe.code-agent/.uv-cache`
(see `../dev-workflow.md`). Each phase must leave all of these green (JEV-X.7).

## Part 1 — Unit / integration tests (no GPU, no Docker)

### A. `vibe.jev` (doc 01) — `tests/test_jev_schema.py`, `tests/test_jev_store.py`

| # | Test |
| --- | --- |
| A1 | built-in `default` validates, 0 errors; choice enum is exactly `["YES","NO"]`; three branches |
| A2 | every presets file in `builtin/` validates |
| A3 | the user's original example validates; `{"decision_type":"score","result":true}` is invalid under `default` and **valid** under the original (documents the coupling fix) |
| A4 | errors: not JSON (line/col in message), root array, bad `$schema` dialect, `type: "strng"`, empty `enum`, depth 17, > 64 KiB, remote `$ref`, `$ref` cycle |
| A5 | warnings: missing `additionalProperties:false`, risky keyword, unbounded string; warnings never flip `ok` |
| A6 | `canonical_hash`: key-order / whitespace invariant; content-sensitive |
| A7 | `wire_schema` strips `$schema` only; input not mutated |
| A8 | `schema_name_for_openai` sanitisation (spaces, unicode, > 64 chars, empty) |
| A9 | store: put/get/list/delete round-trip; atomic replace (simulate crash between temp write and replace ⇒ old file intact); `.active.json`, `../x`, `a/b`, `A`, reserved names (`new`, `launch`, `detach`, `build`) rejected; 257th schema rejected |
| A10 | shadowing: user `default` ⇒ source `override`; delete ⇒ reverts to `builtin`; deleting builtin-only ⇒ `StoreError` |
| A11 | wheel/`importlib.resources` can load builtins (guards packaging) |
| A12 | `vibe.jev` imports no `docker`/`grpc`/`httpx`/`fastapi` (import-graph test, like the existing XR-5 checks) |

### B. Controller (doc 02) — `tests/test_jev_controller.py`, extend `test_jobs.py`, `test_servicer.py`, `test_registry.py`

| # | Test |
| --- | --- |
| B1 | `resolve_jev` precedence matrix: inline > name > model `jev_schema` > override default > builtin default (table-driven, 8 cases) |
| B2 | fresh load + `JEV_ON` (fake `DockerCtl`, fake probe): purge → start → READY → probe → `JevState` set; `Status.jev` populated |
| B3 | fast path: READY same alias, `JEV_ON`, no purge call, container untouched; different hash replaces; same hash reports "already attached" |
| B4 | `JEV_OFF` clears, never touches Docker; `JEV_OFF` with nothing attached = info event, exit-OK semantics |
| B5 | unload / purge / alias change clears Jev; failed fresh load leaves Jev cleared |
| B6 | invalid inline schema ⇒ `error_code=jev_schema_invalid`, **no `purge_all` call** (spy), active model unaffected |
| B7 | probe failure (400 / invalid output / `finish_reason=length`) ⇒ `jev_probe_failed`, model READY, Jev unset, terminal event state READY with `level=error` |
| B8 | `save_as` persists only after a successful probe |
| B9 | persistence: `.active.json` round-trip; `reconcile` keeps with running container, drops without, drops on alias mismatch; corrupt file ⇒ dropped + warning, no crash |
| B10 | `stale` flag after store edit of the active named schema |
| B11 | proto round-trip + old-client compatibility: `LoadModelRequest` without new fields behaves as before (golden event sequence unchanged) |
| B12 | servicer: `NOT_FOUND`, `FAILED_PRECONDITION`, validation-as-data for `Put/Validate`; message size limit |
| B13 | `ModelSpec.jev_schema` / `jev` parsing; missing named schema ⇒ warning only; `jev: false` ⇒ `jev_unsupported_model`; reasoning tag ⇒ warn event |
| B14 | single-job guard: attach while a load runs ⇒ `JobInProgress` |
| B15 | XR-2/XR-5 guard tests still pass (purge only in `jobs.py`; `docker` import only in `docker_ctl.py`) |

### C. Gateway (doc 03) — `tests/test_gateway_jev.py`

| # | Test |
| --- | --- |
| C1 | Jev off: proxied body bytes identical to client's (golden); no `X-Vibe-*` headers; `vibe_jev: null` in `/v1/models` |
| C2 | Jev on: upstream receives `response_format.json_schema.schema` == wire schema (no `$schema`), `strict: true`, name from title |
| C3 | client `response_format` and `guided_*`/`structured_outputs` removed; `X-Vibe-Jev-Overrode` header; log line emitted |
| C4 | `tools` / `tool_choice:"auto"` ⇒ 400 `jev_incompatible_request`, upstream not called; `tool_choice:"none"` allowed |
| C5 | `/v1/completions` ⇒ 400; `/v1/embeddings` passthrough |
| C6 | stale `Content-Length` not forwarded after re-serialisation (fake upstream asserts header equals actual body length) |
| C7 | `GetActiveJev` failure or sha mismatch ⇒ 503 `jev_unavailable`, upstream not called |
| C8 | cache-by-sha: `GetActiveJev` called once across N requests; refetched when sha changes |
| C9 | SSE stream relayed chunk-for-chunk in Jev mode |
| C10 | swap-on-request after Jev: next snapshot plain; no injection |
| C11 | upstream 400 relayed verbatim; snapshot invalidated; **no** unconstrained retry |
| C12 | `/health` and `/v1/models` markers; controller-down ⇒ `jev: null` |

### D. CLI (doc 04) — `tests/test_cli_jev.py` (Typer `CliRunner`, fake `ControllerClient`)

| # | Test |
| --- | --- |
| D1 | option conflicts (`--include-json`+`--schema`, `--no-jev`+`--jev`) ⇒ exit 2 |
| D2 | `--include-json` reads file; missing file / > 64 KiB / bad JSON ⇒ exit 2 **without** an RPC |
| D3 | `--include-json -` reads stdin; TTY stdin ⇒ error |
| D4 | exit codes: `jev_schema_invalid`→2, unknown `--schema`→3, `jev_probe_failed`→1, error event with final state READY→1 |
| D5 | plain `vibe load X` sends `JEV_MODE_UNSPECIFIED` (no regression) |
| D6 | `status --json` has `jev: null` / populated, snake_case, ints not strings |
| D7 | `jev schema validate FILE` works with controller down; `--name` needs it |
| D8 | `schema list/show/put/rm/export` output shapes; `rm` builtin ⇒ exit 1; override ⇒ "reverted" |

### E. Admin (doc 05) — `tests/test_admin_jev.py` (Starlette `TestClient`, fake client)

| # | Test |
| --- | --- |
| E1 | GET `/jev` renders list, active panel, stale badge; controller-down ⇒ banner, no 500 |
| E2 | save invalid ⇒ 200, text preserved byte-for-byte in textarea, errors listed |
| E3 | validate does not call `Put`; save calls `Put` once; `save_launch` calls `Put` then `LoadModel(JEV_ON, name)` |
| E4 | `launch_unsaved` sends inline `schema_json`; `jev_schema_invalid` from first event ⇒ flash, no 500 |
| E5 | escaping: `</textarea><script>` and `<img onerror>` in schema and in error text inert |
| E6 | body > 128 KiB ⇒ 413 before parse; other routes unaffected |
| E7 | cross-origin POST (`Origin: http://evil.example`, `Sec-Fetch-Site: cross-site`) ⇒ 403 on every POST route |
| E8 | reserved/invalid names ⇒ inline error; path param invalid ⇒ 404 |
| E9 | delete confirm page; override-revert copy; builtin delete refused |
| E10 | zero-JS guard: no `<script>`, no `on*=` attributes, CSP header unchanged on every `/jev*` response |
| E11 | quick builder output validates and, with the default options, equals the built-in `default` semantically |
| E12 | optimistic concurrency: hidden `base_sha` mismatch ⇒ re-render with conflict notice (P2) |
| E13 | `HEAD /jev` ⇒ 200 empty body |

**Target:** ≥ 90 new tests; no change to existing assertions except `vibe_jev: null` in `/v1/models` payloads and the
`Status`/`JobEvent` golden fixtures.

## Part 2 — Manual acceptance runbook (target machine: RTX 4070 Ti SUPER 16 GB)

Prereqs: Docker + `nvidia-ctk` OK, `.env` with a pinned `VLLM_IMAGE_TAG` (suggested `v0.11.2`), image already
pulled (or accept the silent first-pull wait — improvements plan 08), `uv sync` done, `vibe serve` running in a
second terminal. Use `qwen2.5-coder-7b` unless noted.

### Pre-flight: confirm the pinned vLLM supports what the plan assumes (decide before J2 merges)

| # | Command | Expected / decision |
| --- | --- | --- |
| P1 | `docker run --rm vllm/vllm-openai:v0.11.2 --help=all \| grep -i -E "structured|guided"` | Identify the structured-output backend option name (`--structured-outputs-config…` vs `--guided-decoding-backend`). Record. We do **not** pass it (default `auto`), but note it for the "backend rejects schema" troubleshooting row. |
| P2 | Against a plain loaded model: `curl localhost:8030/v1/chat/completions` with `response_format` = built-in default (root `oneOf`) | HTTP 200 and valid JSON ⇒ keep root-`oneOf` default. HTTP 400 ⇒ switch `default` to the single-object fallback form (doc 01 note) before J0 ships. |
| P3 | Same with the user's original `anyOf` schema | Record accept/reject. |
| P4 | Same with `$schema` present vs stripped | Confirms need for `wire_schema`. |
| P5 | Same with a `{"if":…,"then":…}` schema | Records whether `RISKY_KEYWORDS` should keep `if/then`. |

### A. Launch paths

| # | Command | Expected |
| --- | --- | --- |
| A1 | `uv run vibe load qwen2.5-coder-7b --jev` | cold load → READY → `jev: probing` → `jev: attached (default, builtin, sha …)`; exit 0; record cold-start seconds |
| A2 | `uv run vibe status` / `--json` | Jev populated; `docker ps` shows one `vibe-qwen2.5-coder-7b`, port `127.0.0.1:8030->8000` |
| A3 | `uv run vibe load qwen2.5-coder-7b --include-json scratch/yesno.json` (file with a custom enum) | **fast path**: same container id, < 5 s, new sha |
| A4 | `uv run vibe load qwen2.5-coder-7b --include-json bad.json` | exit 2 locally with line/col; container untouched |
| A5 | `uv run vibe load qwen2.5-coder-7b --include-json rejected-by-backend.json` | exit 1 `jev_probe_failed`; model still READY plain |
| A6 | `uv run vibe jev off` ×2 | both exit 0; Jev cleared; container id unchanged |
| A7 | `uv run vibe load llama-3.1-8b --jev --schema yes-no` | single container at all times; Jev bound to llama |
| A8 | `uv run vibe unload`; `vibe status` | Jev cleared |
| A9 | `kill` the controller (not the container) and restart `vibe serve` | Jev survives while container runs; after `docker rm -f` + restart it is cleared |

### B. Data-plane correctness (the real point of the feature)

| # | Command | Expected |
| --- | --- | --- |
| B1 | `curl -s localhost:8080/v1/chat/completions -d '{"model":"qwen2.5-coder-7b","messages":[{"role":"user","content":"Is the sky blue? Answer as a decision."}]}'` (default Jev) | `choices[0].message.content` parses as JSON; **validates** against the stored schema (`uv run vibe jev schema validate` style check or `python -c` with `jsonschema`); `X-Vibe-Jev` header present |
| B2 | 50 varied prompts × default schema, script in `scratch/` (not committed) | 50/50 schema-valid; record % where `finish_reason=length`; record p50 latency vs plain |
| B3 | Same with `stream:true` | concatenated deltas parse and validate |
| B4 | Request containing its own `response_format` and `guided_json` | overridden; header `X-Vibe-Jev-Overrode` |
| B5 | Request with `tools` | 400 `jev_incompatible_request` |
| B6 | `/v1/completions` | 400 |
| B7 | `vibe jev off`, repeat B1 | plain free text; no `X-Vibe-Jev`; identical to pre-Jev behaviour (compare with `:8030` directly) |
| B8 | `deepseek-r1-14b` + Jev | warn event emitted; output still schema-valid; note quality |
| B9 | Semantic spot check: 10 clear YES/NO questions | ≥ 9/10 sensible `result` values (sanity, not a benchmark) |
| B10 | Peak VRAM during B2 vs plain | no meaningful increase (grammar compile is CPU-side); record |

### C. Admin UI (real browser, JS disabled via devtools/policy)

| # | Action | Expected |
| --- | --- | --- |
| C1 | open `/jev` | built-ins listed with `built-in` badge |
| C2 | "Create editable copy" of `default`, change enum to `["YES","NO","MAYBE"]`, **Validate** then **Save** | list shows `overrides built-in` |
| C3 | **Save & launch** on qwen | redirected to job page; Jev events visible; dashboard badge shows `default` + sha |
| C4 | paste invalid JSON, **Save** | text preserved, line/col error |
| C5 | **Launch without saving** with a valid inline schema | attaches; not in list afterward |
| C6 | edit the active schema, save | **stale** badge; live behaviour unchanged until re-launch |
| C7 | delete the override | built-in `default` back; notice shown |
| C8 | **Detach** | dashboard badge gone |
| C9 | quick builder → generate → launch | works end-to-end |
| C10 | check devtools Network: no script/CDN requests; CSP header intact | pass |

Record outcomes (cold start, fast-path latency, B2 validity rate, any backend rejections) in the Status line of
docs 01–05 and fold any `RISKY_KEYWORDS` findings back into doc 01.

## Definition of done for the whole feature

- Part 1 green with `make check`; Part 2 pre-flight decisions recorded; A1–A9, B1–B7, C1–C10 pass.
- README: "Jev" section; `.env.example`: `VIBE_JEV_DIR`; `features/03-openai-gateway.md` and
  `features/01-admin-ui.md` link to these docs.
- No change to plain-mode behaviour (B7, C1 of the gateway tests).

## Follow-ups (not in scope)

| ID | Item |
| --- | --- |
| JEV-FU.1 | Optional instruction message injected by the gateway ("respond only with JSON matching the schema") — decide after B9/B2 results |
| JEV-FU.2 | Non-stream response validation + `X-Vibe-Jev-Valid` / `Truncated` headers |
| JEV-FU.3 | `WatchStatus` streaming RPC to drop the 5 s gateway staleness window |
| JEV-FU.4 | Schema export/import bundle, committing `jev/` for team sharing |
| JEV-FU.5 | `/v1/completions` support via vLLM's completions structured-output params |
| JEV-FU.6 | Per-schema generation defaults (temperature, max_tokens) as an envelope around the schema |
