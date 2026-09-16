# 05 — Laboratory Fixes and GPU Hardening

> Priority: **P0** · Depends on: 03-gateway-serve-cli (IMP-03.15 for the `--json` contract) · Status: Planned

## Summary

Phase 5 was implemented without a GPU or Docker daemon, with every heavy function monkeypatched in tests.
The review found three defects that would stop the documented happy path on the **first real run**:
the `SFTTrainer` call uses the TRL ≤0.11 API while the lock resolves **trl 1.13 / transformers 5.17**;
lab containers run without `--network host` and without the `vibe` package, so the serving guard is silently
skipped and `distill`/`export` crash when they shell out; and `autoawq-kernels 0.0.9` predates torch 2.6 and
is likely ABI-incompatible. Beyond those, the container installs a **second torch**, downloads land outside
`HF_HOME/hub/`, the guard reads a JSON key that does not exist, and an omitted `--seed` is printed but never
applied.

## Goals / Non-goals

**Goals**
- Every script in `laboratory/` imports and runs against the **locked** versions (import-signature tests, not just monkeypatched calls).
- One torch in the image; CUDA userland matches the image; image size sane.
- Lab containers can reach the controller/gateway and the `vibe` CLI; registry paths written by `export.py --register` are valid on the host.
- Guards (VRAM, serving) are correct and conservative for training, not only inference.
- `distill.py` exposes the same training flags as `finetune_lora.py`; `export.py` produces a clean directory that `vibe register`/vLLM accepts.
- Reproducibility: printed seed is the applied seed.

**Non-goals**
- Multi-GPU, DeepSpeed, full fine-tuning, RLHF.
- Supporting Python 3.13 in the lab (autoawq-kernels are cp312-only).

## Requirements

### P0 — blockers

| ID | Requirement | Ref |
| --- | --- | --- |
| IMP-05.1 | Port `_sft_train` to TRL ≥1.x: `SFTConfig(output_dir, max_length=max_seq_len, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}, …)`; `SFTTrainer(model, args=cfg, train_dataset=Dataset.from_list(samples), processing_class=tokenizer, peft_config=…)`. `tokenizer=`, `max_seq_length=`, `TrainingArguments` and a `list` dataset all fail on the locked versions. Either pin `trl<1, transformers<5` **or** port — porting is preferred (newer bnb/peft paths). | `finetune_lora.py:198-205` |
| IMP-05.2 | `make lab`/`lab-run`: add `--network host`; mount the repo at the **same absolute path** (`-v $(pwd):$(pwd) -w $(pwd)/laboratory`) so `vibe register <out>` paths resolve on the host (`registry.py:337`, `docker_ctl.py:60-78`); make the `vibe` CLI available in the container (`uv sync --all-packages` in the Dockerfile, or a thin gRPC/HTTP client in `common.py`). Without this: `serving_guard` returns "vibe CLI not found — guard skipped" (NFR-05.4 silently void), `distill.py:236`/`export.py:204` raise `FileNotFoundError`, `http://localhost:8080` is unreachable. | `Makefile:70-71`; `common.py:198-199` |
| IMP-05.3 | Verify `import awq, awq_ext` on the GPU host (runbook step). If it fails (expected: `undefined symbol` — kernels built Nov 2024, torch 2.6.0 Jan 2025), replace `autoawq` with `llmcompressor` (vLLM's supported successor) for `export.py --awq`, or drop AWQ export and document "serve ≤3B bf16 unquantized" as the v1 path. | `laboratory/pyproject.toml:14-15`; `uv.lock:200-228` |

### Container / dependencies

| ID | Requirement | Sev | Ref |
| --- | --- | --- | --- |
| IMP-05.4 | One torch. Either (a) base on `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` and keep the `cu124` lock, or (b) move the lock to `torch==2.7.1+cu126` (index `…/whl/cu126`) and keep `pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel`. Today the image ships torch 2.7.1/py3.13 **and** the venv installs 2.6.0+cu124 + all `nvidia-*-cu12` wheels (~+5 GB). Option (b) also lets bitsandbytes/autoawq re-resolve against a current torch. | P1 | `Dockerfile:9-10, 28` |
| IMP-05.5 | `ENV HF_HOME=/root/.cache/huggingface` in the image; `download.py` stops passing `cache_dir=$VIBE_DATA_DIR` (or passes `<VIBE_DATA_DIR>/hub`). Today snapshots land in `…/huggingface/models--*` instead of `…/huggingface/hub/models--*`, so `from_pretrained` and the Phase-4 vLLM container never see them (FR-05.4, ADR-8). Align `src/vibe/config.py` (`hf_home`) and the lab on a single rule: **`VIBE_DATA_DIR` is `HF_HOME`**. | P1 | `Dockerfile:36`; `download.py:58-60` |
| IMP-05.6 | `ENV PATH=/workspace/.venv/bin:$PATH` baked into the image (plain `docker run` currently gets the base Python 3.13). llama.cpp extra: `CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=89"` + `cmake ninja` in the image (default `native` arch fails in a GPU-less build). | P1 | `Dockerfile:32, 36-38` |
| IMP-05.7 | Add `nvidia-ml-py` to lab deps (the `pynvml` fallback in `common.py:129` can never fire today) or delete the fallback. Add upper bounds or explicit pins for `transformers`/`trl`/`peft` so the lock cannot silently jump majors again. | P2 | `laboratory/pyproject.toml:6-19` |

### `common.py`

| ID | Requirement | Sev | Ref |
| --- | --- | --- | --- |
| IMP-05.8 | `serving_guard` reads the real `vibe status --json` shape. `MessageToDict` emits **`activeAlias`**, not `active_alias`, so the alias branch is dead and `LOADING`/`UNLOADING` pass the guard. Fix: after IMP-03.15 the key is `active_alias`; accept both during transition; treat `state ∈ {READY, LOADING, UNLOADING}` as busy. Add a test that feeds genuine `MessageToDict(Status(...))` output. | P1 | `common.py:193, 209` |
| IMP-05.9 | `parse_seed(None)` applies the generated seed (`random`, `numpy`, `torch`) — today it is only printed, so the recorded seed does not reproduce the run (FR-05.10). Print to stdout/run log, not stderr. | P1 | `common.py:277-280` |
| IMP-05.10 | VRAM estimator for training: derive adapter+optimizer cost from `r × Σ(in+out) × 12 B` and activations from `layers × seq × hidden × k` with a grad-checkpoint factor, instead of the back-fitted `1.5 + 0.36·params`. Keep current constants as a documented fallback when `config.json` is unavailable. Record `estimate_vs_peak` in `metrics.json` so the model can be tuned from real runs. | P1 | `common.py:238-245`; `finetune_lora.py:118-134` |
| IMP-05.11 | `_vibe_status_json` timeout 15 s (cold typer+grpc import can exceed 5 s); prefer a direct gRPC `GetStatus` via `ControllerClient` once IMP-05.2 puts the package in the venv. `download.py` skips `serving_guard` (network-only op). | P2 | `common.py:171-187`; `download.py:80-87` |

### `finetune_lora.py`

| ID | Requirement | Sev | Ref |
| --- | --- | --- | --- |
| IMP-05.12 | Resume: `save_strategy="steps", save_steps=N` (`--save-steps`), `resume_from_checkpoint=get_last_checkpoint(out)` (returns `None` when absent). Today `save_strategy="epoch"` with 1 epoch never checkpoints mid-run, and `resume_from_checkpoint=out.is_dir()` raises `ValueError` whenever `out` exists without `checkpoint-*` (pre-created dir, completed run, distill re-run). | P1 | `finetune_lora.py:194, 206` |
| IMP-05.13 | Gradient checkpointing for **both** LoRA and QLoRA (via `SFTConfig`), `model.config.use_cache = False`. Plain bf16 LoRA on 3B at seq 2048 without it exceeds the 6.6 GiB estimate. | P1 | `finetune_lora.py:178-179` |
| IMP-05.14 | `LoraConfig(target_modules="all-linear")` default + `--target-modules`; `optim="paged_adamw_8bit"` for QLoRA, `--lr` (default 2e-4), `--warmup-ratio`, `--lr-scheduler`. PEFT's default for Qwen2 is `q_proj,v_proj` only and adamw_torch@5e-5 is a poor LoRA default. | P2 | `finetune_lora.py:180-197` |
| IMP-05.15 | Catch `torch.cuda.OutOfMemoryError` explicitly with a hint (`--max-seq-len`, `--qlora`, `--grad-accum`); call `start_run` after argument/data validation so usage errors do not litter `runs/`. Same ordering in `download.py:68`, `export.py:145`. | P2 | `finetune_lora.py:217-221, 273-275` |
| IMP-05.16 | `torch_dtype=` → `dtype=` (deprecated in transformers ≥4.56; confirm on 5.17). Also `inference_hf.py:125`, `inference_gguf.py:132`, `export.py:75`. | P2 | `finetune_lora.py:175-176` |

### `distill.py`

| ID | Requirement | Sev | Ref |
| --- | --- | --- | --- |
| IMP-05.17 | Expose the training flags (`--qlora --r --alpha --epochs --max-seq-len --grad-accum --lr`) via a shared `add_training_args(parser)` helper used by both scripts; forward to `_sft_train`. Hardcoded `r=16, alpha=32, epochs=1, seq 2048, grad_accum=4` contradicts FR-05.7. | P1 | `distill.py:293-298` |
| IMP-05.18 | One `httpx.AsyncClient` per run; `--timeout` (default ≥300 s; swap-on-request can take `VIBE_LOAD_TIMEOUT`), `--max-tokens` in the payload, `--concurrency`. A 14B reasoning teacher routinely exceeds the fixed 60 s. | P1 | `distill.py:143-146` |
| IMP-05.19 | After `--auto-swap` unload, poll `vibe status --json` until `AVAILABLE` and `free_vram_gib()` stabilises before phase 2 (state may be `UNLOADING` with VRAM still held → false refusal or OOM). | P1 | `distill.py:235-240, 282` |
| IMP-05.20 | `_run_cli` inherits stdio (no `capture_output`) and has a timeout; `parse_jsonl(teacher_cache)` inside `try` with a line-number error; use `asyncio.TaskGroup` so one failure cancels siblings. | P2 | `distill.py:175, 181, 249, 263` |
| IMP-05.21 | Cache rows keep the full `messages` the teacher saw (system + multi-turn) plus the assistant completion; `format_samples` consumes the chat form. Today only the last user turn is kept as `prompt`. | P2 | `distill.py:92, 157-161` |

### `export.py`

| ID | Requirement | Sev | Ref |
| --- | --- | --- | --- |
| IMP-05.22 | Use `--base` for merging: `AutoModelForCausalLM.from_pretrained(base)` + `PeftModel.from_pretrained(model, adapter)` + `merge_and_unload()`. `AutoPeftModelForCausalLM` resolves the base from `adapter_config.json` and silently ignores `--base`. | P1 | `export.py:61-77` |
| IMP-05.23 | Merge on CPU (`device_map="cpu"`, 32 GB RAM) and skip the VRAM guard for the merge step; GPU merge makes 7B exports impossible without `--force` (17 GiB estimate). | P1 | `export.py:72-76, 170-171` |
| IMP-05.24 | `--awq` writes to a clean directory: merge to `<out>.bf16/` (or temp), quantize into `--out`; delete bf16 shards after `save_quantized`. In-place quantization currently leaves bf16 shards next to the AWQ `model.safetensors`, which vLLM/`vibe register` will misread. | P1 | `export.py:79, 187` |
| IMP-05.25 | Calibration flattens `messages` rows (the finetune data format); error if < N samples. Today `calib_data` is empty for chat data and autoawq silently downloads `pileval`, contradicting FR-05.9. | P1 | `export.py:104-120` |
| IMP-05.26 | `--register` passes `--quantization awq` when `--awq`, and a `--tool-parser` passthrough; assert `tokenizer_config.json` has `chat_template` after export; copy `generation_config.json` from the resolved **base** snapshot (the adapter dir never has one). | P1 | `export.py:82-85, 204` |

### Inference scripts

| ID | Requirement | Sev | Ref |
| --- | --- | --- | --- |
| IMP-05.27 | `inference_gguf.py --backend transformers` estimates with `BYTES_PER_PARAM["fp16"]` (transformers **dequantizes** GGUF on load — 7B Q4 needs ~14.5 GB, not 0.55 GiB/B). Default backend `llamacpp`; `--n-ctx` flag; hint at the `llamacpp` extra when `llama_cpp` is missing. | P1 | `inference_gguf.py:21, 118, 126-133, 202-216` |
| IMP-05.28 | `inference_hf.py`: `bnb_4bit_quant_type="nf4"`; apply the chat template when present (`--raw` to opt out); tokens/s from `len(output_ids)` decode-only, not streamer chunks; reject `--load-in-4bit` + `--dtype`. | P2 | `inference_hf.py:100-103, 119-123, 150, 173-178` |
| IMP-05.29 | `laboratory/README.md`: remove the "wheels prebuilt in image path, known-good" claim until the runbook confirms it; document `--network host`, the same-path mount, and the AWQ decision from IMP-05.3. | P2 | `laboratory/README.md:16-23` |

## Design notes

- **Version alignment first**: decide IMP-05.4 (a) vs (b) before porting `_sft_train`; (b) means re-locking `trl/peft/bitsandbytes/autoawq` against torch 2.7.1 and re-checking `autoawq-kernels` availability (there is no cp312 wheel newer than 0.0.9 — this strongly favours `llmcompressor` for IMP-05.3).
- **Signature tests**: add `tests/test_lab_compat.py` (marker `lab`, skipped unless `trl` importable) asserting `inspect.signature(SFTTrainer.__init__)` contains `processing_class` and that `SFTConfig` accepts `max_length`. It runs only in the lab venv/container but would have caught IMP-05.1 before any GPU time.
- **`vibe` in the container**: `uv sync --all-packages --frozen --no-dev --python 3.12` installs the root package into the lab venv; root deps are light (grpcio, typer, pydantic, httpx). Then `serving_guard` can use `ControllerClient.status()` directly and IMP-03.15's JSON contract stops mattering for the lab.
- **Guard semantics**: busy = `state ∈ {READY, LOADING, UNLOADING}` or `current_job` set; `--allow-coexist` stays as the override.
- **Shared arg groups**: `common.add_training_args(parser)` and `common.add_model_args(parser)`; `distill.py` builds the finetune call from the parsed namespace.

## Task breakdown

- [ ] **P0** IMP-05.1 — port `_sft_train`; add signature compat test.
- [ ] **P0** IMP-05.2 — Makefile `--network host` + same-path mount; Dockerfile `--all-packages`; `serving_guard` hard-fails (not skips) when the CLI/controller is unreachable **and** a `--allow-coexist` override is absent? → decide; at minimum log loudly.
- [ ] **P0** IMP-05.3 — runbook step 5; branch: keep autoawq / switch to llmcompressor / drop AWQ.
- [ ] IMP-05.4–05.7 — Dockerfile/lock alignment, `HF_HOME`, PATH, llama.cpp arch, deps.
- [ ] IMP-05.8–05.11 — `common.py` guard key, seed, estimator, timeouts.
- [ ] IMP-05.12–05.16 — `finetune_lora.py`.
- [ ] IMP-05.17–05.21 — `distill.py`.
- [ ] IMP-05.22–05.26 — `export.py`.
- [ ] IMP-05.27–05.29 — inference scripts, README.
- [ ] Run the GPU runbook (`07-testing-and-acceptance.md` §Laboratory) and record peak VRAM per step in `features/05-laboratory-workflow.md`.

## Acceptance criteria

- `make lab-build` succeeds; `make lab-run CMD='python -c "import torch;print(torch.__version__, torch.version.cuda)"'` prints exactly one torch whose CUDA minor matches the image.
- `make lab-run CMD='python -c "import trl, peft, bitsandbytes; from trl import SFTTrainer, SFTConfig"'` succeeds; `python -m bitsandbytes` reports SUCCESS on sm_89.
- With `vibe serve` running on the host and `qwen2.5-coder-7b` READY, `make lab-run CMD='python inference_hf.py --model Qwen/Qwen2.5-Coder-0.5B-Instruct --prompt hi'` exits 1 with "run `vibe unload` first"; with `--allow-coexist` it runs.
- `python download.py Qwen/Qwen2.5-Coder-0.5B-Instruct` lands under `$HF_HOME/hub/models--Qwen--…` and `vibe cache list` shows it.
- `finetune_lora.py --qlora --base Qwen/Qwen2.5-Coder-7B-Instruct` on a 20-row dataset completes with `peak_vram_gib ≤ 12.5`; re-running with the same `--out` resumes from `checkpoint-*`.
- `export.py --register --alias lab-0.5b` → `vibe load lab-0.5b` reaches READY and answers via `:8080`; `--awq` output directory contains only quantized shards and a `quantization_config`.
- Two runs with the same printed seed produce identical `metrics.json` loss curves.

## Risks

- Re-locking torch/cu126 can pull `triton`/`bitsandbytes` versions with their own sm_89 quirks — keep the runbook's smoke imports as the gate.
- `llmcompressor` has a heavier dependency tree than autoawq; if it bloats the lab venv, gate it behind an `awq` extra.
- `--network host` widens the container's network view; acceptable on a single-user workstation, document it.
- The 4070 Ti SUPER has 16 GB minus ~830 MiB for the desktop; QLoRA 7B at seq 2048 is near the edge — the estimator (IMP-05.10) must err conservative and the runbook must record real peaks.
