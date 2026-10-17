# 01 — Proto Additions (additive-only)

> Priority: P1 · Depends on: — · Status: Implemented

## Summary

Several UI and CLI shortcomings trace back to information the controller has but the proto does not carry.
The admin keeps two side-channel dicts (`app.state.known_jobs`, `app.state.download_live/download_started`)
purely to work around this; the CLI cannot label a replayed job; `GetStatus` never reports `LOADING`/`UNLOADING`.
This plan lists the **additive** fields and one new RPC that remove those workarounds, per NFR-02.1
("additive-only proto").

## Goals / Non-goals

**Goals**
- Every job stream / replay is self-describing (kind + alias).
- `DownloadInfo` is sufficient to render the download page without admin-side state.
- `RegistrationSpec`/`ModelInfo` can round-trip a full registry entry.
- `GetStatus.state` reflects transitional states while a job is in flight.

**Non-goals**
- Breaking changes, field renumbering, or removing `known_jobs` before the proto lands (that is in doc 04).
- Auth, TLS, or multi-model routing.

## Requirements

| ID | Requirement | Sev |
| --- | --- | --- |
| IMP-01.1 | `JobEvent` gains `JobKind kind = 8; string alias = 9;` populated on every event (`servicer.py:_pump_job`). `GetJobEvents` replay therefore identifies the job. | P1 |
| IMP-01.2 | New RPC `GetJob(GetJobRequest{job_id}) returns (Job)` where `Job` is the existing message used by `Status.current_job`, extended with `bool terminal`, `string last_error`, `int64 started_at_unix_ms`, `int64 finished_at_unix_ms`. Served from the per-job history (`JOB_HISTORY_JOBS`). `NOT_FOUND` when evicted. | P1 |
| IMP-01.3 | `GetStatus.state` returns `LOADING` when `current_job.kind == LOAD` and `UNLOADING` when `UNLOAD`, otherwise the container-derived state (`servicer.py:311`). Aligns with `ListModels` (`servicer.py:231-236`) and FR-02.2. | P1 |
| IMP-01.4 | `DownloadInfo` gains `string current_file`, `string revision`, `int64 started_at_unix_ms`, `int64 finished_at_unix_ms`. `DownloadProgress` gains `bool ok` so success is not inferred from `error == ""`. | P2 |
| IMP-01.5 | `RegistrationSpec` gains `string container_name`, `repeated string extra_args`; `ModelInfo` gains `string container_name`, `repeated string extra_args`, `string yaml_preview` (the exact document `write_entry` would append). Lets the admin print the controller's YAML instead of re-implementing serialisation (`developer.py:118-144`, XR-3). | P2 |
| IMP-01.6 | `Status.current_job` may reference a `DOWNLOAD` job (`JobKind.DOWNLOAD = 3` is declared but never emitted). Downloads surface as jobs so XR-4 holds uniformly. Optional; `ListDownloads` remains the primary API. | P2 |
| IMP-01.7 | `GetJobEvents` gains `bool follow` (default false = replay-only, current behaviour). When true, the server tails live events after the replay so the admin need not poll every 2 s. | P2 |

## Design notes

- All fields are new numbers; no `reserved` needed. Re-run `make proto`; `tests/test_proto.py::test_codegen_matches_commit` guards the committed `src/vibe/gen/`.
- `client.py` gets `get_job(job_id) -> Job` and passes `follow=` through `job_events`.
- `JobRunner` already knows kind/alias when it yields events; thread them through `_pump_job` rather than re-deriving in `GetJobEvents`.
- For IMP-01.3, compute once in a `_effective_state(status, job)` helper used by both `GetStatus` and `ListModels` so the two RPCs cannot disagree again.

## Task breakdown

- [x] Add fields/RPC to `proto/vibe/v1/control.proto`; regenerate; commit `gen/`.
- [x] `servicer.py`: populate `JobEvent.kind/alias`; implement `GetJob`; `_effective_state` helper; `DownloadInfo` timestamps + `current_file`; `DownloadProgress.ok`.
- [x] `registry.py`: expose `render_entry(entry) -> str` so `ProposeRegistration` can fill `yaml_preview` and `write_entry` uses the same function.
- [x] `client.py`: `get_job`, `follow` param; exception translation unchanged.
- [x] `cli.py`: `vibe jobs <id>` (or `vibe status --job <id>`) prints kind/alias/terminal; `vibe download` shows `current_file` from `DownloadInfo` when attaching.
- [x] Tests: replayed events carry kind/alias; `GetJob` NOT_FOUND after eviction; `GetStatus` LOADING while a fake job is in flight; `DownloadProgress.ok` true exactly once on success.
- [x] Update `features/02-grpc-server.md` and `04-model-scripts.md` message tables.

## Implementation notes

- **IMP-01.6 (downloads-as-jobs) skipped** — it is marked optional in the plan. `ListDownloads` remains the primary API and `DownloadInfo` now carries `started_at_unix_ms`/`finished_at_unix_ms`, so the elapsed-time need it addressed is already met. Surfacing downloads as `Status.current_job` would also make the admin job page point at a job with no `GetJobEvents` history (downloads do not flow through `_pump_job`) — worth doing properly together with doc 04's job-page work if ever needed.
- `client.job_events(job_id, *, follow=False)` adds a keyword-only parameter. The admin worker owns `tests/test_admin.py`, whose `FakeControllerClient.job_events(self, job_id)` override must gain the `follow` parameter when it adopts the follow stream (pyright flags the override until then).
- `GetJob` serves `Job` from the bounded per-job history (`JOB_HISTORY_JOBS`, `_job_meta` in lockstep with `_job_events`); a still-running current job is served even before its first event arrives. `NOT_FOUND` otherwise.

## Acceptance criteria

- `uv run vibe status --json` shows `"state": "LOADING"` during `vibe load` from another terminal.
- Admin `/jobs/<id>` for a **CLI-started** job shows kind and alias after completion, with `app.state.known_jobs` deleted (doc 04).
- Admin `/developer/downloads/<id>` shows `current_file` and elapsed time with `app.state.download_live` deleted (doc 04).
- `test_codegen_matches_commit` passes; all existing clients still work against the new server (additive).

## Risks

- `yaml_preview` on `ModelInfo` slightly couples the proto to YAML; acceptable because `models.yaml` is the only registry format (ADR-2). Alternative: a dedicated `RenderRegistration` RPC.
- Live `follow` on `GetJobEvents` duplicates `LoadModel`'s stream mechanics; reuse the same fan-out queue to avoid a second code path.
