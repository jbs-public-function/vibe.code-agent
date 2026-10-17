# 06 — Ops, Deploy, and Developer Experience

> Priority: P2 · Depends on: — · Status: Implemented (a4b3630…; controller logger wiring in `bda05ce` follow-up)

## Summary

Small, mostly independent items that make the repo easier to operate and contribute to: tooling declared in
`pyproject.toml` so `uv sync` stops pruning it, README/Makefile drift, systemd units and Caddyfile that
work outside the author's home directory, structured logging configuration, and a handful of DX papercuts.
Nothing here changes runtime behaviour of the control plane beyond logging.

## Goals / Non-goals

**Goals**
- `uv sync` + `uv run pytest` + `uv run pyright` work from a fresh clone with no manual installs.
- README, Makefile help, `.env.example`, and `vibe zed-settings` agree with each other and with `models.yaml`.
- `deploy/` units are portable and pull in their dependencies.
- One logging configuration shared by controller, gateway, admin (ADR-10).

**Non-goals**
- CI pipelines (no GPU runner available); packaging/publishing `vibe` to PyPI.

## Requirements

| ID | Requirement | Sev | Ref |
| --- | --- | --- | --- |
| IMP-06.1 | Add `pyright>=1.1` to `[dependency-groups] dev` and a `[tool.pyright]` block (`include = ["src", "tests"]`, `pythonVersion = "3.12"`). `uv sync` currently uninstalls an ad-hoc `uv pip install pyright`. Add `make typecheck`. | P1 | `pyproject.toml:24-30` |
| IMP-06.2 | Declare `protobuf>=5.26` as a direct dependency (`MessageToDict(always_print_fields_with_no_presence=…)` needs it; today it arrives transitively via `grpcio-health-checking`). | P2 | `pyproject.toml:7-22`; `cli.py:14` |
| IMP-06.3 | Makefile: `proto` uses `$(PY) -m grpc_tools.protoc` (not `.venv/bin/python`) so `UV_PROJECT_ENVIRONMENT` works; `chat` drops the hardcoded `--model qwen2.5-coder-7b` (script resolves the active model, FR-03.11); help text stops listing ports 8010/8030/8020 (XR-3) — print them from `vibe list` instead; `tls-stop` target and a persistent `-v vibe-caddy-data:/data` volume so `tls internal` does not mint a new root CA each run. | P2 | `Makefile:29-31, 59, 60, 76-79` |
| IMP-06.4 | README: add missing Makefile rows (`serve`, `admin`, `chat`, `tls`, `proto`, `lab-*`); an `.env` key table covering all ADR-10 keys with defaults (links to `.env.example`); replace the hand-maintained Zed snippet with `vibe zed-settings` (IMP-03.21); a "Troubleshooting" section (controller down, 503 model_not_loaded, ghost containers → `make kill-ghosts`). | P2 | `README.md:83-112, 128-167` |
| IMP-06.5 | `deploy/*.service`: replace `/home/jamesb/...` with `%h/projects/vibe.code-agent` or `EnvironmentFile=%h/.config/vibe/env` + `WorkingDirectory=`; add `Wants=vibe-controller.service` (not only `After=`) to gateway/admin; `Restart=always`, `RestartSec=2`; document `systemctl --user enable --now vibe-controller vibe-gateway vibe-admin` and `loginctl enable-linger`. | P2 | `deploy/vibe-*.service:15-20` |
| IMP-06.6 | `deploy/Caddyfile`: site addresses `:8443`/`:8444` (not `localhost:8443`) so a widened `VIBE_BIND` actually serves remote hosts; document trusting Caddy's internal CA (`caddy trust`) and that Caddy must forward `X-Forwarded-Proto/Host` for the admin (IMP-04.26). | P2 | `deploy/Caddyfile:3, 7` |
| IMP-06.7 | `scripts/test_chat.py`: streaming call uses `timeout=None` (or `max(30, VIBE_SWAP_TIMEOUT)`) — a per-read 30 s timeout fails on slow first tokens and swap-on-request; when no model is active, print the alias list and exit 1 instead of guessing `DEFAULT_MODEL` (XR-3). | P2 | `scripts/test_chat.py:17, 53` |
| IMP-06.8 | Structured logging config in one module (`vibe/logging.py`): key=value line format to stdout, level from `VIBE_LOG_LEVEL`, optional JSON via `VIBE_LOG_JSON=1`; used by `serve.py`, `controller/server.py`, and the admin/gateway loggers (IMP-03.8). Replace remaining `print(..., file=sys.stderr)` in the controller with logger calls. Document the new env keys in `.env.example` and ADR-10. | P2 | `serve.py:62-101`; `servicer.py:425-428` |
| IMP-06.9 | `HEAD` support on the admin (see IMP-04.28) and a test for `HEAD /health` on the gateway (already 200; untested). | P2 | — |
| IMP-06.10 | `make check` = `uv lock --check && uv run pytest -q && uv run pyright && uv sync --dry-run` (asserts the root env stays torch-free, NFR-05.1). Document as the pre-commit gate for every phase (XR-8). | P2 | `Makefile` |
| IMP-06.11 | Record the sandbox/offline workflow used during development in `.zed/docs/` (e.g. `UV_CACHE_DIR` inside the repo, `uv run --offline`, `no_proxy` for local smoke tests) so future agents do not rediscover it. | P2 | `.zed/docs/` |

## Task breakdown

- [x] IMP-06.1, 06.2, 06.10 — `pyproject.toml`, `make typecheck`, `make check`.
- [x] IMP-06.3 — Makefile fixes (proto via `$(PY)`, chat without hardcoded model, help without port literals, `tls-stop` + persistent volume).
- [x] IMP-06.4 — README refresh (Makefile rows, `.env` key table, Troubleshooting; Zed snippet replaced per IMP-03.21).
- [x] IMP-06.5, 06.6 — deploy files (`%h`, `Wants=`, `Restart=`, Caddyfile bare ports + notes) + docs.
- [x] IMP-06.7 — `test_chat.py` (no read timeout; lists aliases and exits 1 when none active).
- [x] IMP-06.8 — `vibe/logging.py` + call sites in serve/gateway/admin/controller; `.env.example` keys.
- [x] IMP-06.9 — HEAD tests (admin `HEAD /` → 200; gateway `HEAD /` 404 / `HEAD /health` 200).
- [x] IMP-06.11 — `.zed/docs/fable-5/dev-workflow.md`.

## Implementation notes

- IMP-06.1: `pyright>=1.1` in the `dev` group + `[tool.pyright]` (`include = ["src", "tests"]`, `pythonVersion = "3.12"`); `uv sync` no longer prunes an ad-hoc pyright install.
- IMP-06.8: `vibe.logging.configure(level)` reads `VIBE_LOG_LEVEL`/`VIBE_LOG_JSON`; key=value lines to stdout with `extra={...}` k=v passthrough. Gateway/admin/serve/controller loggers wired in docs 03/04/02 respectively.
- IMP-06.5: units use `%h/projects/vibe.code-agent`, `EnvironmentFile=…/.env`, `Wants=vibe-controller.service`, `Restart=always`/`RestartSec=2`; `loginctl enable-linger` documented.
- IMP-06.6: Caddyfile site addresses are bare `:8443`/`:8444`; `caddy trust` + `X-Forwarded-*` notes added.
- IMP-06.10: `make check` = `uv lock --check && uv run --offline pytest -q && uv run pyright && uv sync --dry-run --offline` (torch-free root assertion, NFR-05.1).

## Acceptance criteria

- Fresh clone: `uv sync && make check` passes with no extra installs.
- `make tls && make tls-stop && make tls` succeeds twice in a row; the browser trusts the cert after `caddy trust`.
- `systemctl --user start vibe-gateway` on a machine without the repo at `/home/jamesb` starts the controller too.
- `vibe serve` output is one key=value line per event, greppable by `logger=vibe.gateway`.

## Risks

- Adding `protobuf` as a direct dependency can conflict with future `grpcio-tools` pins; keep the lower bound loose.
- Changing log output format affects anyone tailing `vibe serve` today (only humans so far).
