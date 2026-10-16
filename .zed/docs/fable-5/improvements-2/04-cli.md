# 04 — CLI: `vibe load --jev` and `vibe jev …`

**Priority:** P1 · **Depends:** 02 (01 for offline validation) · **Status:** Implemented · **Phase:** J3

## Summary

Add Jev to the CLI in two places: flags on the existing `vibe load`, and a `vibe jev` command group for
schema management and detach. All state goes through gRPC; the only local file access is reading the user's
`--include-json` file (and, for offline `validate`, a file/stdin) — never the schema store.

## Surface

```text
vibe load <alias> [--force] [--keep-failed] [--timeout S]
                  [--jev]                       # attach Jev, schema by precedence J4
                  [--include-json FILE|-]       # schema file; implies --jev; "-" = stdin
                  [--schema NAME]               # stored/built-in schema; implies --jev
                  [--save-as NAME]              # persist the --include-json schema after a successful launch
                  [--no-jev]                    # explicit detach (same alias READY) / plain load

vibe jev status [--json]                        # active schema: name, source, sha, stale, applied-at
vibe jev off                                    # detach (idempotent)
vibe jev schema list [--json]
vibe jev schema show NAME [--raw]               # pretty JSON (or exact stored bytes with --raw)
vibe jev schema validate (FILE|-|--name NAME)   # offline when FILE; exit 0/2; prints errors + warnings
vibe jev schema put NAME FILE|-                 # create/update user schema (override of builtin allowed)
vibe jev schema rm NAME                         # delete user schema / remove override
vibe jev schema export NAME [-o FILE]           # write stored JSON to a file (e.g. to start an --include-json)
```

`--include-json` takes a **file path** (or `-` for stdin) and is deliberately distinct from `vibe download --include`
(a glob), so the two never get confused.

## Requirements

| ID | Requirement | Sev |
| --- | --- | --- |
| JEV-04.1 | `--include-json`, `--schema` are mutually exclusive (exit 2 with usage error). `--jev` with neither ⇒ precedence J4 (model's `jev_schema` → `default`). `--include-json`/`--schema` imply `--jev`. `--no-jev` conflicts with all of them (exit 2). | P1 |
| JEV-04.2 | `--include-json FILE`: read as UTF-8 text client-side; reject > 64 KiB **before** the RPC (exit 2); run `validate_schema_text` locally first and print errors/warnings so the common mistake (bad JSON) fails in < 100 ms without a controller. The controller re-validates (authoritative). File-not-found/permission ⇒ exit 2 with the path. `-` reads stdin (error if stdin is a TTY). | P1 |
| JEV-04.3 | `vibe load X --jev` on a READY same alias without `--force` takes the **fast path**; output states "attached without restart". With `--force` ⇒ full reload then attach. | P1 |
| JEV-04.4 | Exit codes: invalid schema ⇒ **2** (`EXIT_CONFIG`); unknown `--schema NAME` ⇒ **3** (`EXIT_ALIAS` reused, message names the schema); probe failure or any terminal `error_code` ⇒ **1**. Because a Jev-only failure terminates with state READY (doc 02 JEV-02.17) the CLI must exit non-zero when it sees an event with `error_code != ""`, regardless of final state. | P1 |
| JEV-04.5 | `_print_proto_event` prints `error_code` in brackets for error events; warnings (e.g. reasoning model) print as `[WARN]`. Output stays line-oriented. | P2 |
| JEV-04.6 | `vibe status` text/`--json` gain a `jev` field (`null` or `{schema_name, sha256, source, stale, applied_at_unix_ms}`), snake_case, int64 cast to `int` (protobuf 7.x quirk: use `preserving_proto_field_name=True`). | P1 |
| JEV-04.7 | `vibe jev schema list` columns: NAME, SOURCE (`builtin`/`user`/`override`), SHA12, SIZE, ACTIVE (`*`). `--json` for scripts. | P2 |
| JEV-04.8 | `vibe jev schema validate FILE` works **without a running controller** (pure `vibe.jev`); `--name NAME` requires the controller. Exit 0 with warnings still prints them. | P1 |
| JEV-04.9 | `vibe jev schema rm` on a builtin-only name ⇒ exit 1 "built-in schemas are read-only"; on an override prints "reverted to built-in". Deleting the **currently active** schema is allowed (active state holds its own copy) but prints a notice. | P2 |
| JEV-04.10 | `Makefile`: `make jev ALIAS=… [SCHEMA=file.json]` convenience (`uv run vibe load $(ALIAS) --jev …`). Optional. README: new "Jev" section (what it is, example, limitations: no tools, chat completions only). | P2 |
| JEV-04.11 | Help text for `vibe load` lists Jev flags in their own group (`rich_help_panel="Jev"` if Typer 0.27.2 supports it; otherwise ordering + prefix). | P2 |
| JEV-04.12 | All new commands are sync Typer wrappers using `_run(asyncio.run(...))`; keep `except typer.Exit: raise` pattern (Typer 0.27.2 does not await async commands). | P1 |

## Design notes

- **Client-side read of `--include-json`**: the controller may run in another cwd/user (systemd); shipping the text avoids
  path semantics entirely and keeps JEV-X.4.
- **Why validate locally *and* remotely?** Local = instant feedback and works offline; remote = authoritative
  (controller's dependency versions / limits could differ in theory).
- **`--save-as`** persists only after validation **and** probe succeed (doc 02 JEV-02.14), so the store never
  accumulates schemas vLLM rejects.
- **`vibe jev off` vs `vibe load X --no-jev`**: the first never touches the container; the second is the same RPC
  (`JEV_OFF`) and exists so scripts can express "load X plain" idempotently.
- **Backward compat**: `vibe load X` with no Jev flags sends `JEV_MODE_UNSPECIFIED` — identical to today.

## Task breakdown

- [x] `cli.py`: new options on `load`; `jev_app = typer.Typer()` group registered via `app.add_typer`; helper
      `_read_schema_arg(include)`; `_load` plumbs `jev_mode`/`JevSpec`.
- [x] `errors.py`: no new constants unless a distinct code is wanted (plan: reuse 1/2/3).
- [x] `status` JSON/text additions; `_print_proto_event` tweaks.
- [x] `schema list/show/validate/put/rm/export` commands.
- [x] Makefile target + README section + `.env.example` (`VIBE_JEV_DIR`).
- [x] Tests (doc 06 §C): option conflicts, exit codes, stdin handling, size cap, offline validate, JSON output shape.

## Acceptance criteria

- `vibe load qwen2.5-coder-7b --jev` → LOADING → READY → "jev: attached (default, builtin, sha …)"; `vibe status`
  shows Jev; exit 0.
- `vibe load qwen2.5-coder-7b --include-json ./my.json` (already READY) completes in seconds, no container restart.
- Malformed JSON file → exit 2, message with line/column, no RPC attempted (controller may be down).
- `--include-json a.json --schema yes-no` → usage error, exit 2.
- `vibe jev schema list` shows `default` as `builtin` until an override is `put`, then as `override`.
- `vibe jev off` twice → both exit 0.

## Risks

| Risk | Mitigation |
| --- | --- |

| Windows/odd encodings of schema files | require UTF-8; error names the byte offset |
| Users expect `--include-json` to *merge* with the default | documented as **replace** ("overrides default"); a merge semantics would need a patch format — out of scope |
