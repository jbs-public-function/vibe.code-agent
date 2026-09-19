# 07 — Testing Gaps and GPU/Docker Acceptance Runbook

> Priority: P1 · Depends on: 05-laboratory-fixes P0s (for §Laboratory runbook) · Status: Planned

## Summary

323 unit tests pass and pyright is clean, but the suite was written in an environment with **no Docker daemon
and no GPU**, and several areas rely on monkeypatched heavy functions or hand-written fixtures that do not
match real outputs (the lab's `active_alias` fixture vs. the real `activeAlias` is the canonical example).
This document lists (1) unit/integration tests to add per area — each one pinned to a finding in docs 02–06 —
and (2) the **manual acceptance runbook** to execute on the target machine (RTX 4070 Ti SUPER, CachyOS,
Docker + nvidia-ctk) the first time the system runs for real.

## Part 1 — Unit / integration test gaps

### Controller (`tests/test_servicer.py`, `test_docker_ctl.py`, `test_registry.py`, `test_downloads.py`, `test_client.py`)

> Implemented in `tests/test_controller.py`, `test_docker_ctl.py`, `test_registry.py`, `test_downloads.py`,
> `test_cache.py`, `test_client.py`, `test_jobs.py`, `test_job_events.py` (there is no `test_servicer.py`).

- [x] Heartbeat: patch `HEARTBEAT_SECONDS` small, assert `level="debug"` heartbeat events (NFR-02.5).
- [x] `wait_ready`: `ContainerExited`, `ReadinessTimeout`, gated-repo hint, `last_logs` truncation, `/health`→`/v1/models` alias check (currently `FakeDocker.wait_ready` is a stub).
- [x] Cancel `LoadModel` stream mid-job → job still reaches terminal; `GetStatus` shows `LOADING` meanwhile (IMP-02.1/02.2).
- [x] Idempotent unload emits terminal `AVAILABLE` (IMP-02.7).
- [x] Concurrent `write_entry` ×2 on one file → one succeeds, one gets `duplicate …`, file re-`load`s (IMP-02.20).
- [x] `Registry.load` with duplicate alias/port **across** documents; `write_entry` output re-loads with special-char `use_case`/`display_name`.
- [x] Realistic GGUF repo (README + config.json + `.gguf`) is rejected (IMP-02.21).
- [x] `RemoveCached` while a download for the same repo is in flight → refused (IMP-02.4).
- [x] `DownloadManager.start` with a different `include` for an in-flight repo (IMP-02.13).
- [x] Stress: every terminal `DownloadProgress` with `ok=False` has non-empty `error` (IMP-02.12).
- [x] `Chat` RPC: SSE parsing, `FAILED_PRECONDITION` when no container, upstream 500, cancel → `aclose()`.
- [x] Client translation: `FAILED_PRECONDITION` from `Chat` → `NoModelLoaded`; `INVALID_ARGUMENT` from `DownloadModel` (IMP-02.25).
- [x] `ListModels.cached_locally` for a local-path `hf_repo` (IMP-02.3).
- [x] Daemon-gone simulation: `containers.list` raising `requests.ConnectionError` → `DockerError` (IMP-02.10).
- [x] (doc 01) Replayed events carry kind/alias; `GetJob` NOT_FOUND after eviction; `DownloadProgress.ok` true exactly once on success; `GetJobEvents(follow=true)` tails to the terminal event.

### Gateway / serve / CLI (`tests/test_gateway.py`, new `test_serve.py`, `test_cli.py`)

- [x] `HEAD /`, `HEAD /health` (404 / 200, empty body).
- [x] HTTP-level `Retry-After: 10` for active-but-LOADING (only `decide` is tested).
- [x] Snapshot TTL: patch `time.monotonic` → refresh after 5 s, reuse before.
- [x] `httpx.ReadError`/`WriteError` during `send` → 502 + invalidation (IMP-03.9).
- [x] Duplicate upstream headers preserved (IMP-03.12).
- [x] Client disconnect mid-SSE → upstream `aclose()`.
- [x] Swap coordinator: `JobInProgress` from a CLI-initiated load (IMP-03.7); hung stream → `SwapFailed` after bound (IMP-03.5).
- [x] `serve.py`: component crash propagates non-zero (IMP-03.2); SIGTERM runs `grpc_srv.stop` (IMP-03.1); `--only` validation.
- [x] CLI: `list`, `load`, `unload`, `status [--json]`, `logs`, `validate`, `serve --only` (only Phase-4 commands are tested today); `status --json` fields are `int` and snake_case (IMP-03.15); no spurious `error:` line on `typer.Exit` (IMP-03.16).
- [x] README Zed block == `build_zed_settings(registry)` (IMP-03.21).
- [x] 413 when `Content-Length` exceeds the limit up front (IMP-03.11).

### Admin (`tests/test_admin.py`)

- [x] Dashboard gateway probe uses an injected `MockTransport` (tests currently hit the real `127.0.0.1:8080`).
- [x] `/logs/stream?tail=0&follow=1` with a quiet stream stays open (IMP-04.11).
- [x] Chat: first token after 5 s still streams (IMP-04.4); `RuntimeError` mid-stream renders its message (IMP-04.5); not-READY `POST /chat` does **not** persist the user turn (IMP-04.6); truncation drops oldest turns at 75 % (IMP-04.7).
- [x] Cross-origin POST → 403; same-origin passes; `Host` outside allow-list → 403 (IMP-04.1).
- [x] `HEAD /` → 200 (IMP-04.28).
- [x] Controller-down on every POST route (`/models/load`, `/models/unload`, `/chat`, `/developer/downloads`, `/developer/register[/confirm]`, cancel/remove).
- [x] `register_confirm` failure paths (write `ok=False`; `reload_registry` errors → "Reload registry" button present, IMP-04.17); `GET /developer/register?repo=` prefill + `_suggest_alias`.
- [x] Job page for a CLI-started job shows kind/alias after IMP-01.1 (and `—` fallback before).
- [x] `SessionStore` TTL sweep, LRU cap, turn cap (IMP-04.9).
- [x] Script/style-free assertion extended to every route incl. `/developer/register`, `/developer/downloads/<id>`, `/developer/cache/<repo>/confirm`; assert `"style="` absent.
- [x] Flash round-trip with quotes/semicolons/unicode; >4 KiB truncated.
- [x] No absolute self-links in rendered HTML (IMP-04.26).
- [x] Lifespan runs under tests (`asgi_lifespan`) so `app.state.http` closes without resource warnings.

### Laboratory (`tests/test_lab_common.py`, `test_lab_scripts.py`, new `tests/test_lab_compat.py`)

- [ ] `serving_guard` fed genuine `MessageToDict(Status(...))` output; `LOADING`/`UNLOADING` treated as busy (IMP-05.8).
- [ ] `parse_seed(None)` actually seeds RNGs — assert `torch.initial_seed()`/`random.getstate()` deterministic (IMP-05.9).
- [ ] Compat (marker `lab`, skipped when `trl` absent): `inspect.signature(SFTTrainer.__init__)` has `processing_class`; `SFTConfig(max_length=…)` accepted; `AutoModelForCausalLM.from_pretrained` accepts `dtype=` (IMP-05.1, 05.16).
- [ ] `download.py` honours `HF_HOME/hub/` layout given `VIBE_DATA_DIR` (IMP-05.5).
- [ ] Resume: `out` exists without `checkpoint-*` → starts fresh, no `ValueError` (IMP-05.12).
- [ ] `inference_gguf` estimate per backend; `_guess_params_from_name` on `Qwen2.5-7B-Instruct-Q4_K_M.gguf`-style names (IMP-05.27).
- [ ] `distill`: `_run_cli` `FileNotFoundError` surfaces as a clean error; teacher timeout retry; `--train-only` with a corrupt cache line; flag passthrough (IMP-05.17–05.20).
- [ ] `export`: `--awq` leaves no bf16 shards; calibration from `messages` rows; `--quantization awq` passed to `vibe register`; `--base` used for merge (IMP-05.22–05.26).
- [ ] Shell-level check (`make lab-check`): `docker run … vibe --help` works inside the lab image and `--network host` reaches `127.0.0.1:50051` (IMP-05.2).

## Part 2 — Manual acceptance runbook (target machine)

Run in order; each step's expected result is the acceptance check. Record results (and peak VRAM where noted)
in the corresponding feature doc's Status line. Prerequisites: Docker running, `nvidia-ctk` configured,
`.env` with `HF_TOKEN`, `VLLM_IMAGE_TAG` pinned, `uv sync` done, no legacy `vllm-*` containers.

### A. Control plane (Phases 0–2)

| # | Command | Expected |
| --- | --- | --- |
| A1 | `uv run vibe validate` | exit 0; three aliases; rejects `VLLM_IMAGE_TAG=latest` if set. |
| A2 | `uv run vibe serve` (separate terminal) | three `listening` lines: `:50051`, `:8080`, `:8081`, all `127.0.0.1`. |
| A3 | `uv run vibe status` | `AVAILABLE`, no active model; GPU stats show ~830 MiB used (desktop). |
| A4 | `make qwen` (= `vibe load qwen2.5-coder-7b`) | job events stream: `LOADING` → readiness → `READY` within `VIBE_LOAD_TIMEOUT`; `vibe status --json` shows `LOADING` mid-way (IMP-02.2). Record cold-start seconds and VRAM. |
| A5 | `docker ps --filter label=vibe.managed=true` | exactly one container `vibe-qwen2.5-coder-7b`, port `127.0.0.1:8030->8000`. |
| A6 | `curl -s localhost:8080/v1/models \| jq '.data[].id'` | all three aliases; `vibe_active` true only for qwen. |
| A7 | `uv run python scripts/test_chat.py` | streamed completion via `:8080`; first-token latency noted (NFR-03.1: compare with direct `:8030`). |
| A8 | `curl -s -o /dev/null -w '%{http_code} %{header_json}' -d '{"model":"llama-3.1-8b","messages":[]}' localhost:8080/v1/chat/completions` | `503`, `Retry-After: 30`, body `model_not_loaded` with `vibe load llama-3.1-8b` hint. |
| A9 | `VIBE_SWAP_ON_REQUEST=true` restart; repeat A8 with `stream:true` | request held, model swaps (single container at all times — watch `docker ps`), completion returned within `VIBE_SWAP_TIMEOUT`. Concurrent request for `deepseek-r1-14b` during the swap → `503 swap_in_progress`. |
| A10 | `docker kill vibe-llama-3.1-8b`; then a chat request | `502`; `vibe status` shows `ERROR`; gateway re-reads status within 5 s. |
| A11 | `make stop` | container gone; `vibe unload` again → single terminal `AVAILABLE` event (IMP-02.7). |
| A12 | `docker run -d --name vllm-legacy busybox sleep 999`; `make qwen` | legacy container purged before load (XR-2). |
| A13 | `kill -TERM <vibe serve pid>` | all three components log a clean stop; exit 0 (IMP-03.1). Also Ctrl-C. |
| A14 | Zed: provider `openai_compatible`, URL `http://localhost:8080/v1`, models from `vibe zed-settings` | agent panel lists the aliases, streams from the active one; a non-active alias shows the 503 message. |

### B. Admin UI (Phase 3)

| # | Command / action | Expected |
| --- | --- | --- |
| B1 | Open `http://127.0.0.1:8081/` | dashboard: GPU bar, active model, gateway `ok`; page refreshes every 5 s; devtools shows **zero** JS requests and the CSP header. |
| B2 | `/models` → Load `deepseek-r1-14b` | redirect to `/jobs/<id>`; events appear; page stops refreshing at terminal state; kind/alias visible. |
| B3 | `/chat` with a long prompt on the 14B model | first token > 3 s still streams (after IMP-04.4); reply rendered escaped; 75 % warning appears at high context. |
| B4 | `/logs?follow=1` on an idle container | stays connected; new vLLM log lines appear when a request is sent (IMP-04.11). |
| B5 | `curl -I http://127.0.0.1:8081/` | `200` (IMP-04.28). |
| B6 | `make tls`; open `https://localhost:8444` | navigation works (no `http://127.0.0.1:8081` links, IMP-04.26); certificate from Caddy internal CA. |
| B7 | Cross-site POST from a local HTML file (`<form action="http://127.0.0.1:8081/models/unload">`) | `403` (IMP-04.1). |

### C. Downloads / registry (Phase 4)

| # | Command | Expected |
| --- | --- | --- |
| C1 | `uv run vibe download Qwen/Qwen2.5-Coder-1.5B-Instruct` | progress with `current_file` and bytes; resumable after Ctrl-C + rerun; blobs under `$HF_HOME/hub/`. |
| C2 | `uv run vibe cache list --json \| jq '.[0].size_bytes'` | number (not string). |
| C3 | `uv run vibe register Qwen/Qwen2.5-Coder-1.5B-Instruct --alias qwen2.5-coder-1.5b` | preview shows next free port (8040), `gpu_mem_util`/`max_model_len` heuristic; `--yes` appends a `---` document; `vibe validate` passes; alias appears in `vibe list` and `/v1/models`. |
| C4 | `uv run vibe register bartowski/Qwen2.5-Coder-7B-Instruct-GGUF` | rejected with GGUF hint (IMP-02.21). |
| C5 | `vibe load qwen2.5-coder-1.5b` | READY; chat works — full download→register→serve loop. |
| C6 | `/developer` page | shows cache entries with sizes/disk free; Remove requires confirm; Register prefilled from `?repo=`. |
| C7 | `vibe cache remove <repo>` while `vibe download <repo>` runs | refused (IMP-02.4). |

### D. Laboratory (Phase 5) — run after doc 05 P0s land

| # | Command | Expected |
| --- | --- | --- |
| D1 | `uv pip list \| grep -i torch` (root venv) | nothing (NFR-05.1). |
| D2 | `make lab-build`; `docker image ls vibe-lab` | builds; record size (target < 12 GB after IMP-05.4). |
| D3 | `make lab-run CMD='python -c "import torch,sys;print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_capability(), sys.version)"'` | one torch, `True`, `(8, 9)`, Python 3.12. |
| D4 | `make lab-run CMD='python -m bitsandbytes'` | SUCCESS, CUDA lib found for the installed torch. |
| D5 | `make lab-run CMD='python -c "import awq, awq_ext"'` | no `undefined symbol` → keep autoawq; else execute IMP-05.3 fallback. |
| D6 | `make lab-run CMD='python -c "from trl import SFTTrainer, SFTConfig; import transformers, peft; print(transformers.__version__, peft.__version__)"'` | imports succeed. |
| D7 | `make lab-run CMD='vibe status --json'` | JSON from the host controller (proves `--network host` + package present, IMP-05.2). |
| D8 | `make lab-run CMD='python download.py Qwen/Qwen2.5-Coder-0.5B-Instruct'` | path under `/root/.cache/huggingface/hub/models--Qwen--…`; host `vibe cache list` shows it (IMP-05.5). |
| D9 | `make lab-run CMD='python inference_hf.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct --prompt "def fib(n):" --seed 1'` | streamed text; `tokens/s`; `peak_vram_gib` < 5; `runs/*/{args,metrics}.json` written. Repeat with the same seed → identical output. |
| D10 | `… inference_hf.py --model Qwen/Qwen2.5-Coder-7B-Instruct --dtype fp16 --prompt hi` then without `--dtype` | first refused (>10 % over free VRAM); second auto-selects 4-bit, peak ≈ 5–6 GiB. |
| D11 | `… inference_gguf.py --gguf bartowski/Qwen2.5-Coder-7B-Instruct-GGUF:Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf --backend llamacpp --prompt hi` | ~5 GiB peak; `--backend transformers` is refused by the corrected estimate (IMP-05.27). |
| D12 | Create `laboratory/data/tiny.jsonl` (20 `messages` rows); `python finetune_lora.py --base Qwen/Qwen2.5-Coder-0.5B-Instruct --data data/tiny.jsonl --epochs 1 --seed 1` | adapter dir with `adapter_config.json`, `adapter_model.safetensors`, `checkpoint-*`; rerun same `--out` → resumes (IMP-05.12). |
| D13 | `python finetune_lora.py --qlora --base Qwen/Qwen2.5-Coder-7B-Instruct --data data/tiny.jsonl --epochs 1` | completes; `peak_vram_gib ≤ 12.5`; record vs. estimate (IMP-05.10). |
| D14 | With `qwen2.5-coder-7b` READY: `python inference_hf.py --model Qwen/Qwen2.5-Coder-0.5B-Instruct --prompt hi` | exit 1 "run `vibe unload` first"; `--allow-coexist` proceeds (NFR-05.4). |
| D15 | `python distill.py --teacher-alias qwen2.5-coder-7b --student Qwen/Qwen2.5-Coder-0.5B-Instruct --prompts data/prompts.jsonl --n 10` | teacher cache with 10 rows; Ctrl-C midway + rerun fetches only missing rows; `--auto-swap` unloads, waits for `AVAILABLE`, then trains (IMP-05.19). |
| D16 | `python export.py --adapter models/…-lora --base Qwen/Qwen2.5-Coder-0.5B-Instruct --out models/lab-0.5b --register --alias lab-0.5b` | dir has `model*.safetensors`, `tokenizer_config.json` with `chat_template`, `generation_config.json`; `vibe load lab-0.5b` → READY; chat via `:8080`. |
| D17 | `… --awq --calibration data/tiny.jsonl --out models/lab-0.5b-awq` (if D5 passed) | only AWQ shards + `quantization_config`; registered with `--quantization awq`; loads in vLLM. |

### E. Sign-off

- [ ] All A–D rows pass or have a linked IMP item.
- [ ] Peak VRAM / cold-start numbers recorded in `features/00-foundation.md` (load times) and `features/05-laboratory-workflow.md` (estimator table).
- [ ] Feature docs' `Status: Implemented` lines amended with `· Verified on GPU <date>`.

## Risks

- Several runbook steps download multi-GB models; run C1/D8 first so later steps reuse the cache.
- Steps A9/A10 intentionally break things; run them last within section A.
- The lab section depends on doc 05 P0 fixes; running it earlier will fail at D6/D7/D12 by design.
