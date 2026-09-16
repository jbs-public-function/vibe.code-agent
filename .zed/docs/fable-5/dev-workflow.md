# Dev workflow — sandbox / offline notes (IMP-06.11)

Practical environment notes for working on this repo from a sandboxed or
air-gapped machine. Read this before running `uv` or pytest in an agent sandbox.

## UV cache

The sandbox cannot write `~/.cache` (and `/tmp` is wiped between terminal
calls), so point uv at a cache inside the repo (gitignored):

```sh
export UV_CACHE_DIR=/home/jamesb/projects/vibe.code-agent/.uv-cache
```

After the first `uv sync` / `uv lock`, everything runs offline:

```sh
uv run --offline pytest -q
uv lock --check
```

## Network

Outbound network is blocked by default and flaky when granted. Always prefer
the offline commands above. When a lock change is genuinely needed
(`uv lock`), retry up to three times before giving up.

## Local smoke tests

HTTP clients in tests and scripts talk to `127.0.0.1`; if the environment
sets a proxy, export `no_proxy=127.0.0.1,localhost` (or `NO_PROXY`) so local
smoke tests are not routed through it.

## Tooling

- `make proto` regenerates `src/vibe/gen/` (grpcio-tools pinned to the
  grpcio version). Never hand-edit generated files; the import alias lives in
  `src/vibe/gen/__init__.py` only.
- `uv run pyright` (dev dependency) type-checks `src/` and `tests/`.
- `make check` is the pre-commit gate: `uv lock --check && uv run --offline
  pytest -q && uv run pyright && uv sync --dry-run --offline`.
- The root environment must stay **torch-free** (NFR-05.1): `laboratory/` is
  a workspace member whose heavy deps are not installed by the root
  `uv sync`. Verify with `uv sync --dry-run` after any pyproject change.

## Git

The agent sandbox protects `.git` metadata. If a command must write it
(commit, branch), run it with `unsandboxed: true` and a reason.
