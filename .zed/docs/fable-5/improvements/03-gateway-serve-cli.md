# 03 — Gateway, `vibe serve` Lifecycle, and CLI Fixes

> Priority: P1 · Depends on: 02-controller-hardening (IMP-02.25 for error classes) · Status: Planned

## Summary

The gateway's decision table, byte-for-byte relay, hop-by-hop stripping and swap coordinator are solid and
well-tested. The gaps are around the edges: `vibe serve` does not survive SIGTERM gracefully and never notices
a dead HTTP component; the swap coordinator can wedge forever on a hung stream; some `httpx` transport errors
bypass the 502/invalidation path; there is **no structured logging** anywhere in the gateway (ADR-10); and
`vibe status --json` emits proto3-JSON (camelCase, int64-as-string), which is also what broke the lab's
serving guard (doc 05).

## Goals / Non-goals

**Goals**
- `vibe serve` shuts down all three components cleanly on SIGINT **and** SIGTERM; a crashed component takes the process down with a non-zero exit.
- Swap-on-request cannot hang indefinitely; CLI-initiated loads are attached to rather than rejected.
- Every upstream transport failure maps to `502` + snapshot invalidation (FR-03.9).
- Structured key=value/JSON logs at the gateway's decision points.
- CLI `--json` output is stable, snake_case, with real integers.

**Non-goals**
- HTTP→gRPC transcoding, auth, metrics endpoints (v2 candidates).
- Changing the 32 MiB bounded-buffer approach (accepted v1 risk).

## Requirements

### `serve.py`

| ID | Requirement | Sev | Ref |
| --- | --- | --- | --- |
| IMP-03.1 | Single signal handler for the composite process. Two `uvicorn.Server.serve()` calls each install `signal.signal` handlers; on SIGTERM the chain ends by re-raising the signal with the **default** handler → process killed while still parked on `wait_for_termination()`; the `finally` and `grpc_srv.stop(grace=2)` never run. Use `uvc._serve()` (or `Config` with signal capture disabled) plus one `loop.add_signal_handler` for SIGINT/SIGTERM that sets both `should_exit=True` and calls `grpc_srv.stop(grace)`. | P1 | `serve.py:110-122` |
| IMP-03.2 | Observe component death: `asyncio.wait({uvc_task, admin_task, term_task}, return_when=FIRST_COMPLETED)`; if an HTTP task finished with an exception (bind error, crash), log it, shut down the others, exit `EXIT_RUNTIME`. Today the process keeps running with a dead gateway and the exception is only surfaced at GC. | P1 | `serve.py:110-113` |
| IMP-03.3 | Drain instead of cancel: set `should_exit`, `await asyncio.wait_for(task, timeout=grace)`, cancel only on timeout so in-flight SSE responses finish. | P2 | `serve.py:115-121` |
| IMP-03.4 | Move `_load_config` (duplicated in `cli.py:70-93`) into `vibe/config.py` returning `(settings, registry)` and raising a typed error; `cli.py` translates to `typer.Exit`. | P2 | `serve.py:26-47` |

### `gateway/swap.py`

| ID | Requirement | Sev | Ref |
| --- | --- | --- | --- |
| IMP-03.5 | Bound the in-flight job: wrap the `async for` over the load stream in `asyncio.timeout(settings.vibe_load_timeout + 30)`; expiry → `SwapFailed`, slot cleared by the existing `finally`. Without it a stalled stream leaves `self._job` set forever: same-alias callers time out repeatedly, every other alias gets `SwapConflict`. | P1 | `swap.py:98-146` |
| IMP-03.6 | Store the `_run` task on `_Job` (`job.task = asyncio.create_task(...)`) so it cannot be GC'd and can be cancelled on shutdown. | P2 | `swap.py:82` |
| IMP-03.7 | Attach to CLI/admin-initiated loads: when `load()` raises `JobInProgress` for the same alias, poll `status()` (or use `GetJob` from doc 01) until READY/ERROR instead of returning `503 model_load_failed`. For a different alias return `503 swap_in_progress`. | P2 | `swap.py:104-114`; `router.py:123-124` |

### `gateway/app.py`, `proxy.py`, `router.py`

| ID | Requirement | Sev | Ref |
| --- | --- | --- | --- |
| IMP-03.8 | Structured logging (ADR-10): `logging.getLogger("vibe.gateway")` with `extra={alias, upstream, status, elapsed_ms, reason}` at: decision outcome, 502 + invalidation, swap start/finish/fail, controller-down transitions, 413 rejections. `serve.py` configures a key=value (or JSON) formatter to stdout and replaces its `print` calls. Same logger family for `vibe.controller` and `vibe.admin`. | P1 | `app.py` (no `logging` import); `serve.py:62-101` |
| IMP-03.9 | Catch `httpx.TransportError` (superclass) in the proxy, not the four-tuple. `ReadError`/`WriteError`/`CloseError`/`ProxyError` during the header exchange currently surface as unhandled 500 **without** snapshot invalidation, violating FR-03.9. | P1 | `proxy.py:73-79` |
| IMP-03.10 | `/health` returns `"state": null` (or `"UNKNOWN"`) when `controller == "down"` rather than `"AVAILABLE"`. Update `tests/test_gateway.py:879-884`. | P2 | `app.py:176-183` |
| IMP-03.11 | Body accumulation via `bytearray`/list+join (not `bytes +=`); reject immediately when `Content-Length` header exceeds `BODY_LIMIT` (413) before reading. | P2 | `app.py:194-208` |
| IMP-03.12 | Preserve duplicate upstream response headers via `response.headers.multi_items()` (comment claims this already; `items()` merges). Drop client `Host` on the request direction or add `X-Forwarded-Host`. Log mid-stream relay exceptions before re-raising. | P2 | `proxy.py:66-93` |
| IMP-03.13 | Active alias in `ERROR`: message includes `last_error` and the `vibe load X` hint, `Retry-After: 30`; when swap-on-request is on, treat as swappable (`Swap(model)`). Retrying a terminal error every 10 s is pointless. | P2 | `router.py:91-104` |
| IMP-03.14 | Document the controller-down `503 controller_unavailable` + `Retry-After: 5` shape in `features/03-openai-gateway.md` (implemented, unspecified). | P2 | `app.py:61-70` |

### `cli.py`

| ID | Requirement | Sev | Ref |
| --- | --- | --- | --- |
| IMP-03.15 | `--json` output uses `preserving_proto_field_name=True` and converts 64-bit fields to `int` (or builds dicts explicitly). Today `status --json` yields `"startedAtUnix": "1726…"` and `cache list --json` yields string byte counts; `laboratory/common.py:209` reads `active_alias` and silently gets `None`. Treat the JSON shape as a contract and add a schema test. | P1 | `cli.py:314-323, 490-500` |
| IMP-03.16 | `typer.Exit` raised inside a `try` is caught by `except Exception` (it subclasses `RuntimeError`) and prints a spurious `error:` line. Add `except typer.Exit: raise` before the generic handler, or set a flag and exit after the block. | P2 | `cli.py:429-437` |
| IMP-03.17 | `--only` becomes a `Component(str, Enum)` so typos are usage errors (exit 2) with choices in `--help`, like `--quantization`. | P2 | `cli.py:383-391` |
| IMP-03.18 | Close `ControllerClient` in every command (`try/finally: await client.close()` or an `async with` helper). | P2 | `cli.py:188, 238, 274, 304, 370` |
| IMP-03.19 | One controller-down message style (`_exit_unavailable` everywhere: "controller unreachable at … — start it with `vibe serve`"). | P2 | `cli.py:135-138, 191-193` |
| IMP-03.20 | Help text derives the port range from `REGISTER_PORT_MIN/MAX` instead of repeating `8010-8099` (XR-3). | P2 | `cli.py:575` |

### `zedsettings.py` / README

| ID | Requirement | Sev | Ref |
| --- | --- | --- | --- |
| IMP-03.21 | README's hand-maintained Zed snippet drifts from `vibe zed-settings` (`display_name` differs; README never mentions the command, FR-04.8). Replace the snippet with "run `uv run vibe zed-settings`" + one abbreviated example, **and** add a test that parses the README JSON block and compares it to `build_zed_settings(registry)`. | P1 | `zedsettings.py:53-70`; `README.md:128-167` |

## Design notes

- **Signals**: uvicorn's `Server.serve()` = `capture_signals()` + `_serve()`. Calling `_serve()` is private-API; the alternative is subclassing `Server` and overriding `capture_signals` to a no-op. Either way, one place owns SIGINT/SIGTERM for the composite process. In `--only` mode the single component may keep its own handling.
- **Logging format**: a small `vibe/logging.py` with `configure(level)` producing `ts=… level=… logger=… msg=… k=v…` (grep-friendly, no deps). JSON behind `VIBE_LOG_JSON=1` optional.
- **JSON contract**: define `to_json_dict(msg)` in `cli.py` (or `vibe/jsonout.py`) that walks descriptors and casts `INT64/UINT64` fields to `int`; test it against `Status`, `CacheEntry`, `DownloadInfo`.

## Task breakdown

- [ ] IMP-03.1–03.3 — `serve.py` signal ownership, `FIRST_COMPLETED` wait, drain.
- [ ] IMP-03.4 — shared `load_config`.
- [ ] IMP-03.5–03.7 — swap coordinator bound + attach.
- [ ] IMP-03.8 — logging module + call sites in gateway/serve (controller/admin follow in their docs).
- [ ] IMP-03.9, 03.12 — proxy error class + header handling.
- [ ] IMP-03.10, 03.11, 03.13, 03.14 — app/router polish + spec doc update.
- [ ] IMP-03.15–03.20 — CLI fixes; regenerate `laboratory/common.py` expectations (doc 05 IMP-05.7).
- [ ] IMP-03.21 — README snippet + parity test.
- [ ] Tests per `07-testing-and-acceptance.md` §Gateway/serve/CLI.

## Acceptance criteria

- `systemctl --user stop vibe` (SIGTERM) logs `gateway stopped`, `admin stopped`, `controller stopped` and exits 0 within the grace period.
- Starting `vibe serve` with 8081 already bound exits non-zero with a clear error instead of running headless.
- A `LoadModel` stream that never terminates (fake client) causes `SwapFailed` after `VIBE_LOAD_TIMEOUT + 30` s and subsequent requests are not stuck on `SwapConflict`.
- `httpx.ReadError` from the upstream yields `502` and a fresh `GetStatus` on the next request.
- `vibe status --json | jq .started_at_unix` is a number; `jq .active_alias` exists.
- README Zed block equals `vibe zed-settings` output (test).

## Risks

- Relying on uvicorn private API (`_serve`) may break on upgrade; pin `uvicorn` minor and add a test that imports the symbol.
- Changing `--json` key casing is a breaking change for any scripts written against the current output; only the lab consumes it today (doc 05 updates it).
