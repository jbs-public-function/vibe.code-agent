# vibe laboratory

Hands-on PyTorch experiments for vibe: download models, run inference, QLoRA
fine-tune a 7B model, distill from a served teacher into a small student, and
export the result back into `models.yaml` so vLLM can serve it. The control
plane (`src/vibe`) stays torch-free — this workspace is a separate uv project
member (ADR-1, FR-05.1).

Six scripts, `common.py` shared helpers, and per-run logging under
`laboratory/runs/<timestamp>-<script>/` (args, metrics, console log, peak VRAM;
FR-05.10). All artifacts live under `laboratory/models/` (gitignored;
FR-05.11).

## Where it runs

The host has CUDA 13.3 and a rolling toolchain; the lab pins a CUDA 12.6
userland (torch 2.7.1+cu126, matching the `pytorch/pytorch:2.7.1-cuda12.6`
base image) to keep `bitsandbytes`/`autoawq` kernels consistent with the
image.

| | Container (`make lab`) | Host venv (`uv sync --project laboratory`) |
| --- | --- | --- |
| CUDA userland | 12.6 from image; driver 13.3 is backward compatible | cu126 wheels; same driver compat |
| Risk on rolling OS | low — pinned image | medium — system python/glibc updates |
| bitsandbytes/autoawq kernels | **pending GPU verification** (doc 07 runbook D4/D5) — not "known-good" until that run passes | may need rebuild after system updates |
| Convenience | one extra shell | direct editor integration |
| **Recommendation** | **default** | opt-in for quick iteration |

- **Container (default, FR-05.2):** `make lab` drops into a shell in the
  pytorch cuda12.6 image with `--gpus all`. `make lab-run CMD="python
  inference_hf.py ..."` runs a single command.
- **Host venv (FR-05.3):** `uv sync --project laboratory` — works while the
  driver stays CUDA-12-compatible; the container is the supported path.
  Root `uv sync` never installs torch (NFR-05.1).

### Container networking and paths (IMP-05.2)

The `lab`/`lab-run` targets run the container with `--network host`, so
`127.0.0.1:50051` (controller) and `localhost:8080` (gateway) inside the
container are the **host's** services — the serving guard and `distill.py`'s
teacher generation depend on this. They also mount the repository at the
**same absolute path** the host sees (`-v $(pwd):$(pwd)`, workdir
`$(pwd)/laboratory`), so absolute paths written by `export.py --register` /
`vibe register` inside the container resolve identically on the host
(`registry.py` validates `hf_repo` paths in the host controller process).

The image bakes `PATH=/workspace/.venv/bin:$PATH` and installs the root
`vibe` package (`uv sync --all-packages`), so `vibe status --json`,
`vibe unload`, and `vibe register` work with a plain `docker run` — no
manual `-e PATH` needed.

### HF cache layout (IMP-05.5)

The single rule: **`VIBE_DATA_DIR` is `HF_HOME`** (both default to
`/root/.cache/huggingface` in the image). Snapshots land under
`$HF_HOME/hub/models--<repo>` — `download.py` passes
`<VIBE_DATA_DIR>/hub` when the variable is set. This is the exact layout the
controller's cache scanner (`HF_HOME/hub`) and the Phase-4 vLLM containers
read, so `vibe cache list` and `from_pretrained` see everything the lab
downloads.

## VRAM budget (16 GB card, ~830 MiB desktop overhead, ~15 GB usable)

| Task | Model size | Precision | Est. VRAM | Fits? |
| --- | --- | --- | --- | --- |
| Inference | 1.5–3B | bf16/fp16 | 3–7 GB | yes |
| Inference | 7B | fp16 | ~14.5 GB + KV | **no** (use 4-bit) |
| Inference | 7B | 4-bit (bnb/AWQ) | 4.5–5.5 GB + KV | yes |
| Inference | 14B | 4-bit AWQ | ~9 GB + KV | yes (short ctx) |
| GGUF inference | 7B Q4_K_M | llama.cpp | ~5 GB | yes |
| LoRA (no quant) | ≤3B | bf16 | 8–12 GB | yes (bs 1–2, seq 2048) |
| QLoRA | 7B | 4-bit base + bf16 adapters | 10–12 GB | yes (bs 1, seq 2048, grad ckpt) |
| QLoRA | 14B | 4-bit | 16+ GB | **no** |
| AWQ quantization | 7B | fp16 load required | ~14.5 GB | borderline — CPU offload; slow |
| Distill teacher | 14B AWQ via vLLM | — | 9 GB (in vLLM) | yes, but not concurrently with student training |

Training VRAM is estimated from the model's `config.json`: weights +
`r × Σ(in+out) × 12 B` (adapter param, grad, two Adam moments) +
`layers × seq × hidden × 68 B × 0.4` activations (gradient-checkpoint
recompute factor) + 2.5 GiB framework (IMP-05.10). This lands 7B QLoRA at
~12.3 GiB and refuses 14B QLoRA at ~22.5 GiB; when `config.json` is
unavailable the older back-fitted constants are the fallback. Every training
run records `estimate_vram_gib` and `estimate_vs_peak` in `metrics.json` so
the constants can be tuned from real peaks.

Scripts refuse to start when the estimate exceeds free VRAM by more than 10 %
unless `--force`, and abort when a model is busy on the GPU (READY/LOADING/
UNLOADING or a job in flight) unless `--allow-coexist` (NFR-05.3, NFR-05.4).
If the controller is unreachable the serving guard **fails closed** rather
than silently skipping (IMP-05.2) — pass `--allow-coexist` to proceed anyway.

## AWQ decision tree (IMP-05.3)

`autoawq-kernels 0.0.9` predates the image's torch and may be ABI-broken on
the GPU host; the decision is made by doc 07 runbook step **D5**:

```sh
make lab-run CMD='python -c "import awq, awq_ext"'
```

1. **Imports succeed (no `undefined symbol`)** → keep autoawq as-is; run
   runbook D17 to validate `export.py --awq` end to end.
2. **`undefined symbol` / other ABI failure** → switch `export.py --awq` to
   **llmcompressor** (vLLM's supported successor; requires a re-lock against
   torch 2.7.1). Gate it behind an `awq` extra if its dependency tree bloats
   the venv (doc 05 risk note).
3. **llmcompressor infeasible** → drop AWQ export from v1, remove `autoawq`
   from `laboratory/pyproject.toml`, and document "serve ≤3B bf16
   unquantized" as the v1 export path.

Until D5 passes, `--awq` must be treated as experimental.

## Scripts

```sh
# Download a model into the shared HF cache (HF_HOME); prints the local path
python download.py Qwen/Qwen2.5-Coder-1.5B-Instruct --include "*.safetensors" "config.json"

# One-shot streaming inference; auto-selects 4-bit above 3B
python inference_hf.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct --prompt "def fib(n):"

# 7B fp16 is refused by the VRAM guard; 4-bit fits
python inference_hf.py --model Qwen/Qwen2.5-Coder-7B-Instruct --prompt "hi"   # auto 4-bit
python inference_hf.py --model Qwen/Qwen2.5-Coder-7B-Instruct --dtype fp16 --force  # dangerous

# GGUF inference (path or repo:file; llamacpp is the default backend, and the
# transformers backend dequantizes to fp16 on load, so 7B is refused there)
python inference_gguf.py --gguf models/qwen2.5-coder-7b-instruct-q4_k_m.gguf --prompt "hi"
python inference_gguf.py --gguf bartowski/Qwen2.5-Coder-7B-Instruct-GGUF:qwen2.5-coder-7b-instruct-q4_k_m.gguf --n-ctx 4096
# llama.cpp needs the optional extra (CUDA build, sm_89): CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=89" uv sync --project laboratory --extra llamacpp

# QLoRA fine-tune on a JSONL of chat or prompt/completion rows
python finetune_lora.py --base Qwen/Qwen2.5-Coder-7B-Instruct --data data/code.jsonl --qlora --epochs 1

# Distill: phase 1 (teacher must be loaded), then phase 2 (teacher must be unloaded)
python distill.py --teacher-alias deepseek-r1-14b --student Qwen/Qwen2.5-Coder-1.5B-Instruct --prompts data/prompts.jsonl
# ...then: vibe unload; python distill.py ... --train-only
# or in one go: python distill.py ... --auto-swap   (unloads, waits for AVAILABLE, trains)

# Merge the adapter, optionally AWQ-quantize, and register it
python export.py --adapter models/qwen2.5-coder-7b-instruct-lora --base Qwen/Qwen2.5-Coder-7B-Instruct --out models/qwen2.5-coder-lab
python export.py --adapter ... --base ... --out models/qwen2.5-coder-lab-awq --awq --calibration data/code.jsonl --register --alias qwen-coder-lab
```

## Round trip

```sh
python download.py Qwen/Qwen2.5-Coder-1.5B-Instruct                # → HF cache
python finetune_lora.py --base Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --data data/code.jsonl --out models/qwen-coder-1.5b-lora       # → adapter
python export.py --adapter models/qwen-coder-1.5b-lora \
    --base Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --out models/qwen-coder-1.5b-lab --register --alias qwen-coder-lab
vibe load qwen-coder-lab                                          # → READY
# ...and chat against it from Zed via the gateway on :8080
```

## Notes

- GGUF stays lab-side: vLLM does not serve GGUF; use `llama-cpp-python`
  locally (FR-05.6, non-goal).
- Distillation is sequential: phase 1 needs the teacher loaded, phase 2 needs
  it unloaded — `vibe unload` before training (the spec's VRAM table).
- Run `vibe unload` before any training/export; both on one card will OOM.
- `laboratory/models/`, `laboratory/data/`, and `laboratory/runs/` are
  gitignored and dockerignored; keep artifacts in `models/` (FR-05.11).
- `HF_TOKEN` is read from the environment for gated repos; the cache is
  `$HF_HOME` (`$VIBE_DATA_DIR`, which **is** `HF_HOME` — snapshots live in
  its `hub/` subdirectory, ADR-8, IMP-05.5).
