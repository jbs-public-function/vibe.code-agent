# 02 — Controller: Proto, Jev State, Jobs

**Priority:** P1 · **Depends:** 01 · **Status:** Planned · **Phase:** J1

## Summary

Make the controller the owner of "which Jev schema (if any) is attached to the active model". Extend
`control.proto` additively, add a `JevState` holder, teach `JobRunner.load` to **attach / replace / detach** a schema
(with a fast path that does **not** restart the container), probe the schema against the live vLLM, and expose the
schema store over gRPC. This is the only phase that changes `control.proto`; docs 03–05 consume it.

## Goals / Non-goals

**Goals**
- `LoadModel` with a Jev spec = normal load **or** (same alias already READY) a seconds-long attach job.
- Typed errors on failure (`error_code`), no silent downgrade to plain mode.
- Jev state survives a controller restart and never outlives its container.
- Schema CRUD / validate RPCs for the CLI and admin.

**Non-goals**
- Request rewriting (doc 03).
- Changing vLLM launch arguments for Jev (none needed — see ADR-J2).

## Requirements

### Proto (additive; `make proto`; never hand-edit `src/vibe/gen/**`)

| ID | Requirement | Sev |
| --- | --- | --- |
| JEV-02.1 | `enum JevMode { JEV_MODE_UNSPECIFIED = 0; JEV_ON = 1; JEV_OFF = 2; }` — `UNSPECIFIED` keeps today's behaviour exactly. | P1 |
| JEV-02.2 | `message JevSpec { string schema_name = 1; string schema_json = 2; string save_as = 3; }` — `schema_json` (inline text, from `--include-json` or the admin textarea) wins over `schema_name`; both empty ⇒ resolve per precedence J4. `save_as` optionally persists an inline schema to the store atomically with the launch. | P1 |
| JEV-02.3 | `LoadModelRequest { … JevMode jev_mode = 5; JevSpec jev = 6; }` (existing fields 1–4 untouched). | P1 |
| JEV-02.4 | `message JevInfo { string schema_name = 1; string sha256 = 2; string source = 3; int64 applied_at_unix_ms = 4; bool stale = 5; string title = 6; }` where `source ∈ {"include-json","store","model-default","default"}` and `stale` = a named store schema was edited after activation. **No schema body** (keeps the 5 s status poll small). `Status { … JevInfo jev = 8; }` (unset message ⇒ Jev inactive; use `HasField`). | P1 |
| JEV-02.5 | `rpc GetActiveJev(GetActiveJevRequest) returns (ActiveJev)`; `message ActiveJev { JevInfo info = 1; string schema_json = 2; }` — the gateway fetches the body only when `sha256` changes. Empty response (no `info`) when inactive. | P1 |
| JEV-02.6 | `JobEvent { … string error_code = 10; }` set on terminal error events (see taxonomy in README). | P1 |
| JEV-02.7 | Schema store RPCs: `ListJevSchemas`, `GetJevSchema{name}`, `PutJevSchema{name, schema_json}`, `DeleteJevSchema{name}`, `ValidateJevSchema{schema_json}`. Messages: `JevSchemaEntry {name, source, sha256, size_bytes, mtime_unix_ms, title, active}`; `JevValidation {ok, errors[], warnings[], sha256, size_bytes}`. `Put`/`Delete` return `JevValidation`/`{ok, error}` — validation failures are **data**, not gRPC errors (mirrors `ProposeRegistration`). | P1 |
| JEV-02.8 | gRPC status mapping: unknown name → `NOT_FOUND`; `Delete` of a builtin-only name → `FAILED_PRECONDITION`; oversized message → `RESOURCE_EXHAUSTED` (server `max_receive_message_length` ≥ 256 KiB, explicit). | P2 |

### Registry (XR-3)

| ID | Requirement | Sev |
| --- | --- | --- |
| JEV-02.9 | `ModelSpec.jev_schema: str | None = None` (optional per-model default schema name, same name regex). `Registry` validation: if set, the name must exist in the store **at load time** → otherwise a *warning*, not a config error (store contents change at runtime; `vibe validate` prints it, exit 0). | P2 |
| JEV-02.10 | Optional `ModelSpec.jev: bool | None` capability flag (`false` ⇒ launch refused with `jev_unsupported_model`). Default `None` = allowed; reasoning-tagged models (`tags` contains `reasoning`) emit a **warn** event ("structured output suppresses `<think>`; quality may drop"). Keep to these two fields; no other Jev knobs in `models.yaml`. | P2 |

### Controller logic

| ID | Requirement | Sev |
| --- | --- | --- |
| JEV-02.11 | `vibe/controller/jev.py`: `JevState` — `current() -> ActiveJev | None`, `set(alias, name, source, schema)`, `clear(reason)`, persisted to `${VIBE_JEV_DIR}/.active.json` (`{alias, name, source, sha256, applied_at_unix_ms, schema}`; atomic write). On controller start `reconcile(find_managed())`: keep only if a managed container for `alias` is **running**, else delete the file. | P1 |
| JEV-02.12 | `JobRunner.load(alias, …, jev: ResolvedJev | None, jev_mode)`: (a) **fresh load** (container purged/started): after READY, if `JEV_ON` → probe → `set`; any purge → `clear` first. (b) **fast path** — alias already READY, `not force`, `JEV_ON`: no purge; validate → probe → `set` (replace if different hash; info event "already attached" if identical hash). (c) `JEV_OFF` on READY same alias: `clear`, info event, terminal READY. (d) `JEV_OFF` otherwise: no-op info event. (e) `JEV_UNSPECIFIED`: unchanged behaviour **plus** a purge clears Jev; a no-op "already READY" load with Jev active adds the hint "Jev '<name>' active — `--no-jev` / `vibe jev off` to detach". | P1 |
| JEV-02.13 | `unload()` and every `purge_all` path call `JevState.clear` (XR-2: purge stays only in the runner; Jev clearing rides along). A failed fresh load leaves Jev cleared. | P1 |
| JEV-02.14 | Resolution (`resolve_jev(spec, spec_model, store)`) implements J4: inline `schema_json` → `schema_name` → `ModelSpec.jev_schema` → `default` (store override else builtin). Returns `(schema, name, source, sha256)`; raises typed `JevError(code, message)`. `save_as` writes to the store **only after** validation + successful probe (a schema that vLLM rejects is not persisted by a launch). | P1 |
| JEV-02.15 | **Probe** (`probe_jev(host_port, alias, schema)`): one `POST http://127.0.0.1:<port>/v1/chat/completions` with the wire schema as `response_format`, `max_tokens=128`, `temperature=0`, a fixed neutral prompt ("Return a valid example."), 20 s timeout (separate from `VIBE_LOAD_TIMEOUT`). Success = HTTP 200 **and** parsed `choices[0].message.content` validates against the schema. Failure ⇒ `jev_probe_failed` with vLLM's error body (truncated to 500 chars) or the validation error; **model stays loaded in plain mode**, Jev not set. Uses `httpx` from the controller (it already speaks HTTP to vLLM for readiness/Chat). | P1 |
| JEV-02.16 | `finish_reason == "length"` during the probe ⇒ treated as failure with a hint ("schema may permit unbounded output; add `maxLength`/bounds"). | P2 |
| JEV-02.17 | Events: `info "jev: resolved schema '<name>' (<source>, sha <12>)"`, `info "jev: probing…"`, `info "jev: attached"`, `warn` for risky-keyword/reasoning warnings, `error` + `error_code` on failure. Terminal state semantic unchanged (READY/ERROR). On a **Jev-only** failure the terminal event state is **READY** with `level="error"` and `error_code` set (model is healthy; Jev isn't) — clients must key off `error_code`, not state, to exit non-zero. | P1 |
| JEV-02.18 | `Status.jev.stale`: computed per `GetStatus` by comparing the active hash to `store.get(name).sha256` when `source in {store, model-default, default}`. Admin/CLI show a "re-apply" hint; gateway ignores `stale` (live behaviour is the *activated* schema). | P2 |
| JEV-02.19 | `ListModels`/`Status` consumers unaffected when `jev` is absent (additive). `ModelInfo` gets no Jev field. | P2 |
| JEV-02.20 | Structured logs (`vibe.controller`): `jev attach/replace/detach/probe_failed` with `alias, schema, sha, source, elapsed_ms`. | P2 |

## Design notes

- **Why a fast path?** Cold start is 30–120 s; Jev is a gateway overlay so attaching to an already-READY model
  should be ~1–2 s (the probe). This is also what makes "edit schema in admin → relaunch" pleasant.
- **Why not labels on the container?** Docker labels are immutable after `run`, so they can't express attach/detach
  without a restart. Optionally *also* set `vibe.jev=<name>` at fresh-load time for `docker ps` visibility only —
  never read back as truth (`.active.json` is truth).
- **Single-job guard** (`JobRunner._acquire`) already serialises everything; an attach while a load is running gets
  `JobInProgress` like any other job. Attach holds the job slot for the probe duration only.
- **Proto3 presence**: `Status.jev` is a message field ⇒ `HasField("jev")` is the "active" test. CLI `--json`
  output must use `preserving_proto_field_name=True` and explicit int casts (protobuf 7.x int64 → string).
- **Model-state race**: if the container dies after attach, `reconcile`/status already degrade to ERROR; Jev state is
  cleared lazily at the next `GetStatus` that sees no running container for the alias.
- **Job kind**: reuse `LOAD`. Do not add a `JobKind` for attach (clients render by kind; a second kind forces UI
  changes for no benefit). The admin job page already works for any LOAD job.

## Task breakdown

- [ ] Proto edits (JEV-02.1–02.8); `make proto`; commit regenerated `src/vibe/gen/**`; run pyright.
- [ ] `ControllerClient`: `load(..., jev_mode, jev)`, `get_active_jev()`, schema CRUD/validate wrappers; map
      `NOT_FOUND`/`FAILED_PRECONDITION` to typed Python errors (`JevSchemaNotFound`, `StoreError`).
- [ ] `controller/jev.py`: `JevState`, `resolve_jev`, `probe_jev`; persistence + reconcile on startup.
- [ ] `JobRunner.load` branches (a)–(e); `unload` clears; plumb `JobEvent.error_code` through `jobs.py`
      dataclass and `_pump_job`/`_event_to_proto`.
- [ ] Servicer: `LoadModel` passes Jev args; `GetStatus` fills `jev`; new RPC handlers; message-size options in
      `controller/server.py`.
- [ ] `ModelSpec.jev_schema` / `jev` fields + registry tests; `vibe validate` prints store-missing warning.
- [ ] Tests (doc 06 §B).

## Acceptance criteria

- `LoadModel(alias, JEV_ON)` on a stopped system: purge → start → READY → probe → `Status.jev` populated.
- Same call again with a different schema, container id unchanged (`docker ps`), completes in < 5 s, `sha256` updated.
- `JEV_OFF` clears `Status.jev`; container id unchanged.
- `vibe unload`, or a load of a different alias, clears Jev; controller restart with the container still running
  keeps Jev; controller restart after `docker rm` clears it.
- Invalid inline schema ⇒ terminal event `error_code=jev_schema_invalid`, **no purge occurred**, active model
  unaffected. Backend-rejected schema ⇒ `jev_probe_failed`, model still READY, Jev unset.
- Old clients (no Jev fields) behave exactly as before.

## Risks

| Risk | Mitigation |
| --- | --- |
| Probe false-negative (model emits valid JSON slowly / hits 128 tokens) | `finish_reason` check + hint; probe uses `temperature=0`; failure leaves the system usable |
| Probe adds ~1–2 s to every launch | acceptable; skip via internal flag in tests only |
| `.active.json` drift vs reality | `reconcile` on start + lazy clear in `GetStatus`; file is advisory, container is authoritative |
| Large `schema_json` over gRPC | 64 KiB cap enforced before the call (client) and after (server); explicit `max_receive_message_length` |
| Race: gateway caches `jev` for ≤5 s after detach | gateway invalidates on any `jev_incompatible`/400 from upstream; acceptable window documented in doc 03 |
