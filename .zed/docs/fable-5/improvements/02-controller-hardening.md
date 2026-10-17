# 02 — Controller, Jobs, Docker, Registry Hardening

> Priority: P1 · Depends on: 01-proto-additions (for IMP-02.2 only) · Status: Implemented

## Summary

The controller stack matches the architecture (detached job task + fan-out queue, `to_thread` for Docker,
thread→loop handoff via `call_soon_threadsafe`, validate-before-swap registry). The review found no P0 but a
cluster of edge-case races, un-translated SDK exceptions, blocking calls on the event loop, and a registry
writer with a TOCTOU window. This plan fixes them without changing the public gRPC surface (except where
doc 01 already adds fields).

## Goals / Non-goals

**Goals**
- Every long-running op terminates with a typed terminal event (XR-4), including the no-op unload.
- No blocking Docker call on the asyncio loop (NFR-02.3).
- Docker SDK / requests exceptions never escape as `UNKNOWN` gRPC errors.
- `models.yaml` writes are serialised and re-validated before `os.replace`.
- `cached_locally` computed one way everywhere (FR-04.9).
- Client exceptions describe what actually happened (FR-02.6/02.10).

**Non-goals**
- Replacing the multi-document YAML append with ruamel round-trip (works; keep).
- Auth, multi-model, or job persistence across controller restarts.

## Requirements

### `servicer.py`

| ID | Requirement | Sev | Ref |
| --- | --- | --- | --- |
| IMP-02.1 | Retain strong references to pump tasks: `self._tasks: set[asyncio.Task]`, `add_done_callback(discard)`. A client disconnect must never let the load task be garbage-collected mid-run (FR-02.11). | P1 | `servicer.py:486` |
| IMP-02.2 | `GetStatus.state` reports `LOADING`/`UNLOADING` while a job is in flight (see IMP-01.3). | P1 | `servicer.py:311` |
| IMP-02.3 | `ListModels.cached_locally` uses `self.cache.is_cached(spec.hf_repo)` (handles local paths, case-insensitive), same as `ProposeRegistration`. `is_cached` lowercases the compare. | P1 | `servicer.py:248, 823`; `cache.py:65-111` |
| IMP-02.4 | `RemoveCached` refuses (`removed=False, error="download in flight"`) when the repo key is in `downloads._inflight`. Protects FR-04.2 "partial blobs stay resumable". | P2 | `servicer.py:783-786` |
| IMP-02.5 | In `_pump_job`, treat `KeyError` as "unknown alias" only before the first event; afterwards fall through to the generic handler and record a terminal `ERROR` event. Suppress the `job stream failed` stderr print when a terminal event was already seen. | P2 | `servicer.py:420-428` |
| IMP-02.6 | Remove or correct the misleading "cancellation detected on heartbeat timeout" comment; grpc.aio cancels the coroutine directly. | P2 | `servicer.py:513` |

### `jobs.py`

| ID | Requirement | Sev | Ref |
| --- | --- | --- | --- |
| IMP-02.7 | Idempotent unload with nothing to purge emits a terminal event with `state=AVAILABLE` (single `READY→AVAILABLE` no-op per FR-02.4). Today `terminal` is never true and clients cannot tell "done" from "disconnected". | P1 | `jobs.py:239-245` |
| IMP-02.8 | Cleanup path after `ContainerExited`/`ReadinessTimeout` catches `Exception` (not just `DockerError`) so a `docker.errors.APIError` during `stop` cannot mask the real cause. | P2 | `jobs.py:199-203` |

### `docker_ctl.py`

| ID | Requirement | Sev | Ref |
| --- | --- | --- | --- |
| IMP-02.9 | `wait_ready` wraps every SDK call (`containers.get`, `.attrs`, `_last_logs`) in `asyncio.to_thread`. Currently a 2 s poll for up to 600 s blocks every other RPC for a daemon round-trip each tick. | P1 | `docker_ctl.py:296-327` |
| IMP-02.10 | A `_call(fn, *args)` helper maps `docker.errors.DockerException` **and** `requests.RequestException` to `DockerError`; used by `find_managed`, `container_info`, `logs`, `_ensure_image`, `purge_all`. A mid-session daemon restart must degrade per spec, not surface `UNKNOWN`. | P1 | `docker_ctl.py:110, 236, 250` |
| IMP-02.11 | Local-path detection consistent across `docker_ctl._is_local` (`startswith("/")`), `registry._is_local_path` and `cache.is_cached` (accept `~`). Decide: require absolute paths in `propose` (reject `~`) — vLLM cannot see an unexpanded `~` and there is no mount for it. | P2 | `docker_ctl.py:74`; `registry.py:571`; `cache.py:109` |

### `controller/downloads.py`

| ID | Requirement | Sev | Ref |
| --- | --- | --- | --- |
| IMP-02.12 | `_fail` sets `error`/`ok` **before** `done`. `events()` polls `handle.done` on the loop thread; the current order can publish a terminal event with `error==""` that looks like success. (Success path at L300-303 already orders correctly.) | P1 | `downloads.py:307-310` |
| IMP-02.13 | Attach-to-in-flight keys on `(repo, revision, sorted(include))`; a mismatch returns `INVALID_ARGUMENT` rather than silently attaching to a download of different files. | P2 | `downloads.py:177-179` |
| IMP-02.14 | Add `DownloadManager.close()` (`executor.shutdown(wait=False, cancel_futures=True)`) called from `server.run`; wrap `call_soon_threadsafe` in `try/except RuntimeError` for loop teardown. | P2 | `downloads.py:156, 357-368` |
| IMP-02.15 | Document that cancel is cooperative (observed only inside `tqdm.update`); consider `max_workers=1` on `snapshot_download` so cancellation is prompt. Optionally delete partial `.incomplete` blobs on explicit cancel behind a flag. | P2 | `downloads.py:419-421` |

### `controller/cache.py`

| ID | Requirement | Sev | Ref |
| --- | --- | --- | --- |
| IMP-02.16 | `_ensure_fresh` guarded by a `threading.Lock` (three `to_thread` callers run concurrently). | P2 | `cache.py:80-84` |
| IMP-02.17 | `weights_bytes` measures the **newest snapshot only**; summing across revisions double-counts and can trip the >10 GiB refusal. Update `test_weights_bytes_sums_across_snapshots` accordingly. | P2 | `cache.py:128` |
| IMP-02.18 | `datetime.fromtimestamp(ts, UTC)` — the rest of the codebase emits UTC ISO strings. | P2 | `cache.py:97-99` |

### `controller/server.py`

| ID | Requirement | Sev | Ref |
| --- | --- | --- | --- |
| IMP-02.19 | Install `loop.add_signal_handler(SIGINT/SIGTERM)` that calls `server.stop(grace)` so `KeyboardInterrupt` from `asyncio.run` does not hard-cancel pump tasks and downloads. (Coordinated with IMP-03.1 for all-in-one mode.) | P2 | `server.py:113-122` |

### `registry.py`

| ID | Requirement | Sev | Ref |
| --- | --- | --- | --- |
| IMP-02.20 | `write_entry` holds a module-level `threading.Lock` around load→check→append→replace, and re-validates the merged text via `Registry.from_dict(safe_load_all(...))` **before** `os.replace`. Two concurrent `WriteRegistration` calls today can both pass duplicate checks (same auto-port from `_next_free_port`) and produce a file that fails `reload`. | P1 | `registry.py:465-490` |
| IMP-02.21 | GGUF-only detection: `has_gguf and not any(p.suffix in {".safetensors", ".bin"})`. The current `all(p.suffix == ".gguf")` is false for every real GGUF repo (README, config.json present), so such repos fall to `weights=0` → small-model heuristic → unservable vLLM entry. | P1 | `registry.py:343-347` |
| IMP-02.22 | `weights == 0` for a cached repo returns an error (`no *.safetensors found`) rather than silently taking the ≤3 GiB bucket. | P2 | `registry.py:355-357` |
| IMP-02.23 | Atomic write preserves mode (`shutil.copymode`) and `fsync`s the temp file before `os.replace`. | P2 | `registry.py:487-490` |
| IMP-02.24 | `propose` computes the auto-port from the on-disk registry when `self.path` is set, so the port checked in `write_entry` is the one proposed. | P2 | `registry.py:300, 473` |

### `client.py`

| ID | Requirement | Sev | Ref |
| --- | --- | --- | --- |
| IMP-02.25 | Error translation is per-RPC, not global-by-code. `Chat` uses `FAILED_PRECONDITION` for "no model loaded" (→ currently `JobInProgress`) and `UNAVAILABLE` for "model server returned 500" (→ currently `ControllerUnavailable: start it with vibe serve`, which is false). Add `NoModelLoaded` and `UpstreamError`; pass a mapping into `_raise_translated`. Admin `chat.py` and CLI `chat` display the right message. | P1 | `client.py:65-76`; `servicer.py:607-610, 646-648, 693-696` |
| IMP-02.26 | Remove dead `try/except` around synchronous `stub.X(request)` stream creation (never raises); dedupe docstring line. | P2 | `client.py:9-10, 135-148` |

### `config.py` / `gpu.py`

| ID | Requirement | Sev | Ref |
| --- | --- | --- | --- |
| IMP-02.27 | `_check_image_tag` rejects case variants (`Latest`) and whitespace. | P2 | `config.py:43` |
| IMP-02.28 | `GpuStats` nvidia-smi fallback gets a 1 s TTL cache; the admin dashboard refresh otherwise spawns ~4 processes/s. | P2 | `gpu.py` |

## Design notes

- **Task retention** (IMP-02.1) and **signal handling** (IMP-02.19) are the two changes with the largest blast radius on shutdown behaviour; land them together with a test that cancels the `LoadModel` stream mid-job and asserts the job still reaches a terminal state.
- **`_call` helper** (IMP-02.10): keep `DockerError` the only exception type callers see; include the original class name in the message for debugging.
- **Registry lock** (IMP-02.20): a process-local lock suffices — the controller is the only writer (XR-2/XR-3). Add a comment saying so.
- **Per-RPC translation** (IMP-02.25): simplest form is `_raise_translated(exc, *, failed_precondition=JobInProgress)` with `Chat` passing `NoModelLoaded`, and a `details`-prefix check for the upstream-500 case (`"model server returned"`), keeping the server side unchanged.

## Task breakdown

- [x] IMP-02.1, 02.5, 02.6 — `servicer.py` pump lifecycle.
- [x] IMP-02.2, 02.3 — `_effective_state`, unified `cached_locally`.
- [x] IMP-02.4 — `RemoveCached` in-flight guard (+ test).
- [x] IMP-02.7, 02.8 — `jobs.py` terminal no-op unload, broad cleanup catch.
- [x] IMP-02.9, 02.10, 02.11 — `docker_ctl.py` `to_thread` in `wait_ready`, `_call` helper, local-path policy.
- [x] IMP-02.12–02.15 — `downloads.py` ordering, attach key, close(), cancel notes.
- [x] IMP-02.16–02.18 — `cache.py` lock, newest-snapshot size, UTC.
- [x] IMP-02.19 — `server.py` signal handlers (`install_signals` param; IMP-03.1 composition root passes False).
- [x] IMP-02.20–02.24 — `registry.py` writer lock + re-validate, GGUF detection, zero-weights error, fsync/mode, on-disk auto-port.
- [x] IMP-02.25, 02.26 — `client.py` exceptions; update admin `chat.py` and CLI `chat` handlers.
- [x] IMP-02.27, 02.28 — config/gpu polish.
- [x] Tests listed in `07-testing-and-acceptance.md` §Controller.

## Implementation notes

- IMP-02.27 was already implemented (`_check_image_tag` compares `tag.lower() == "latest"` after `.strip()`); verified, no change.
- IMP-02.25: the new `NoModelLoaded`/`UpstreamError` classes are `RuntimeError` subclasses with readable messages, so any generic `except Exception` handler (CLI, admin) displays them correctly. The CLI has no `chat` command; the admin `chat.py` UI is doc 04's scope and needs no change beyond its existing `except` path. `tests/test_admin.py::FakeControllerClient.job_events` must gain the new `follow` keyword when the admin worker adopts follow polling (see doc 01's note).
- IMP-02.15: `max_workers=1` on `snapshot_download` + a comment explaining cooperative cancel; deleting partial `.incomplete` blobs on cancel was left out (they stay resumable, per FR-04.2).
- IMP-02.20: lock/re-validate/copymode/fsync all in `write_entry`; the process-local `threading.Lock` comment cites XR-2/XR-3 (the controller is the only writer).
- `wait_ready` now accepts `timeout_s: int | float` (sub-second timeouts are useful to tests; the deadline math was float already).

## Acceptance criteria

- `vibe unload` with nothing running prints a single terminal `AVAILABLE` event and exits 0.
- Killing the CLI mid-`vibe load` leaves the controller job running to READY (verified via `vibe status`).
- Stopping the Docker daemon mid-session turns `vibe list`/`vibe logs` into the documented degrade paths, not `UNKNOWN`.
- Two concurrent `WriteRegistration` calls yield a valid `models.yaml` with one entry rejected as duplicate.
- `vibe register bartowski/<repo>-GGUF` is rejected with the GGUF hint.
- `vibe chat` with no model loaded prints "no model loaded" (not "controller unreachable").
- `uv run --offline pytest -q` green; `pyright` 0 errors.

## Risks

- Changing `weights_bytes` semantics (IMP-02.17) alters `vibe cache list` sizes; document in the CLI help.
- `to_thread` in `wait_ready` adds thread hops on every poll; negligible at a 2 s cadence.
- New client exception classes are a minor API change for any external gRPC clients (none today).
