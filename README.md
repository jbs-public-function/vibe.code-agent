# vibe.code-agent

A personal sandbox repository for exploring **vLLM** and experimenting with smaller, non-SOTA (State-of-the-Art) Large Language Models. This project is built for self-edification and to serve as a local inference backend for personal development projects.

## Project Goals
*   **Learn vLLM**: Master prompt caching, continuous batching, and model quantization.
*   **Small Model Optimization**: Experiment with highly efficient, smaller models (e.g., Qwen, Llama-3-8B, or Phi-3) that run comfortably on consumer hardware.
*   **Local Tooling**: Integrate a local inference setup with custom coding assistants and personal automation scripts.

## Environment & System Hardware

### NVIDIA System Management Interface (`nvidia-smi`)

| GPU | Name | Persistence | Total Memory | Compute Mode |
| :---: | :--- | :---: | :---: | :---: |
| 0 | NVIDIA GeForce RTX 4070 Ti SUPER | On | 16GB (16376MiB) | Default |

*   **Driver Version**: 610.43.02
*   **CUDA UMD Version**: 13.3
*   **Max Power Cap**: 285W

### CUDA Compiler (`nvcc`)
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Tue_Jun_09_02:43:40_PM_PDT_2026
Cuda compilation tools, release 13.3, V13.3.73
Build cuda_13.3.r13.3/compiler.38244171_0
```

### NVIDIA Container Toolkit (`nvidia-ctk`)
```text
NVIDIA Container Toolkit CLI version 1.19.1
```

### OS Environment
*   **OS**: CachyOS (Arch-based Linux optimized for performance)

### Initial models and choices

| Model Name | Weight Size (VRAM) | Safe Context Ceiling on Your Setup | Best Primary Use Case |
| :--- | :---: | :---: | :--- |
| **Qwen2.5-Coder-7B-Instruct-AWQ** | ~4.5 GB | **32,000 tokens** (Maxed) | Code generation & repository parsing |
| **Llama-3.1-8B-Instruct-AWQ** | ~5.5 GB | **32,000 – 48,000 tokens** | Resume writing & brainstorming |
| **DeepSeek-R1-Distill-Qwen-14B-AWQ** | ~9.0 GB | **16,000 – 32,000 tokens** | Deep reasoning & logic problems |

## Docker Deployment (vLLM)

Because CachyOS runs on a rolling release cycle with cutting-edge CUDA versions (currently CUDA 13.3), running vLLM inside a Docker container isolates fragile deep-learning dependencies while mapping the host's GPU cleanly via the NVIDIA Container Toolkit.

### 1. Deploying a Model

Model parameters (HF repo, quantization, context length, VRAM budget, host port, container name) are defined once in [`models.yaml`](models.yaml), the single source of truth. Deploy any registered alias with the `vibe` CLI:

```bash
vibe load deepseek-r1-14b    # or: uv run vibe load deepseek-r1-14b
```

| Alias | Container Name | Host Port | Use Case |
| :--- | :--- | :---: | :--- |
| `deepseek-r1-14b` | `vibe-deepseek-r1-14b` | `8010` | Deep reasoning & logic problems |
| `llama-3.1-8b` | `vibe-llama-3.1-8b` | `8020` | Writing & brainstorming |
| `qwen2.5-coder-7b` | `vibe-qwen2.5-coder-7b` | `8030` | Code generation & repository parsing |

### 2. VRAM Optimization Choices for RTX 4070 Ti SUPER (16GB)
These knobs are configured per model in [`models.yaml`](models.yaml) instead of raw `docker run` flags:
*   **`quantization: awq`**: Ingests compressed 4-bit weights, cutting model baseline footprints to ~4.5–9 GB.
*   **`max_model_len`**: Budgets the context window per model (`7984` DeepSeek-R1-14B, `16384` Llama-3.1-8B, `32768` Qwen2.5-Coder-7B).
*   **`gpu_memory_utilization`**: Restricts engine VRAM allocation (0.70–0.85 across the registry). This prevents conflicts with the active CachyOS/Wayland desktop interface (~830MiB overhead).
*   **HF cache bind-mount**: `vibe` bind-mounts the host HF cache into each container, so model weights persist across container restarts without redownloads.

### 3. Monitoring the Server
To watch the model initialization, download progress, and API launch logs, run:
```bash
vibe logs --follow    # or: make logs
```
The server is fully initialized and listening once the logs output: `Uvicorn running on http://0.0.0.0`.

## Environment Configuration & Automation

This project uses a centralized environment file for credentials and a `Makefile` that delegates to the `vibe` CLI for container orchestration, VRAM purging, and multi-model hot-swapping.

### 1. Local Environment Setup (`.env`)

To manage private repository access (such as downloading gated weights for Llama 3.1), create a local `.env` file in the project root. This file is ignored by Git to protect your credentials.

Copy the example template to get started:
```bash
cp .env.example .env
```

Open `.env` and populate it with your Hugging Face read token:
```text
HF_TOKEN=hf_your_actual_token_string_here
```

### 2. The Automation Dashboard (`Makefile`)

The `Makefile` loads your `.env` variables and exports them, then delegates every action to the `vibe` CLI (invoked as `uv run vibe` by default). Because multiple large models cannot occupy your 16GB VRAM simultaneously, `vibe load` purges any previously managed container before starting a new one.

| Command | Action / Description |
| :--- | :--- |
| `make` / `make help` | Displays the help menu and usage instructions. |
| `make deepseek` | Loads **DeepSeek-R1 (14B AWQ)** (`vibe load deepseek-r1-14b`). |
| `make qwen` | Loads **Qwen-2.5-Coder (7B AWQ)** (`vibe load qwen2.5-coder-7b`). |
| `make llama` | Loads **Llama-3.1 (8B AWQ)** (`vibe load llama-3.1-8b`). |
| `make stop` | Stops and removes all managed vLLM containers, freeing GPU memory (`vibe unload`). |
| `make status` | Prints the active model state and current GPU usage (`vibe status`). |
| `make logs` | Follows the active container's logs (`vibe logs --follow`). |
| `make validate` | Validates `.env` + `models.yaml` (exits 0 on success, 2 on error). |
| `make list` | Lists registered models with their aliases and ports (`vibe list`). |
| `make serve` | Starts the control plane — controller + gateway + admin (`vibe serve`). |
| `make admin` | Opens the admin UI at `http://localhost:8081` in a browser. |
| `make chat` | Streams a test chat through the gateway (active model; lists aliases when none is loaded). |
| `make tls` / `make tls-stop` | Start / stop the optional Caddy TLS front (requires Docker; persistent `vibe-caddy-data` volume). |
| `make proto` | Regenerates committed gRPC code from `proto/vibe/v1/control.proto`. |
| `make typecheck` | Runs pyright over `src/` and `tests/`. |
| `make check` | Full pre-commit gate: lock check, tests, pyright, torch-free sync (XR-8). |
| `make lab-build` | Builds the pinned pytorch laboratory image (`vibe-lab`). |
| `make lab` | Opens a shell in the laboratory container (torch + CUDA). |
| `make lab-run CMD="…"` | Runs one laboratory script in the container. |
| `make kill-ghosts` | Alias for `make stop`. |

### 3. `.env` keys (ADR-10)

Every key is optional; defaults are in parentheses. See [`.env.example`](.env.example) for a copy-paste template.

| Key | Default | Purpose |
| :--- | :--- | :--- |
| `HF_TOKEN` | _(empty)_ | Hugging Face read token for gated repos (Llama 3.1, …). |
| `VIBE_DATA_DIR` | `~/.cache/huggingface` | Shared HF cache for vLLM containers and `laboratory/` — also `HF_HOME`. |
| `VIBE_BIND` | `127.0.0.1` | Bind address for controller/gateway/admin (XR-1). |
| `VIBE_SWAP_ON_REQUEST` | `false` | Gateway loads a non-active model on request instead of returning 503. |
| `VIBE_SWAP_TIMEOUT` | `180` | How long (s) the gateway waits for a swap-on-request. |
| `VIBE_LOAD_TIMEOUT` | `600` | Load readiness timeout (s). |
| `VIBE_LOG_LEVEL` | `info` | Level for the shared structured logger. |
| `VIBE_LOG_JSON` | _(unset)_ | `1` for JSON log lines instead of key=value. |
| `VLLM_IMAGE_TAG` | _(required)_ | Pinned vLLM image tag — never `latest` (XR-6). |
| `LAB_IMAGE_TAG` | `2.7.1-cuda12.6-cudnn9-devel` | pytorch base image for `make lab-build` (NFR-05.2). |

### 3. Workflow Examples

To hot-swap from your default reasoning model to your code assistant and monitor its health, execute:
```bash
make qwen
make status
```

### 4. Troubleshooting

| Symptom | Fix |
| :--- | :--- |
| `vibe status` says "controller unreachable" | Start it: `make serve` (or `vibe serve`). |
| Chat returns `503 model_not_loaded` | `vibe load <alias>` first, or set `VIBE_SWAP_ON_REQUEST=true`. |
| A load fails instantly / ports collide | `make kill-ghosts` purges stale managed and legacy containers, then retry. |
| Admin UI links break behind `make tls` | Ensure Caddy forwards `X-Forwarded-Proto/Host` (default `reverse_proxy` does) and trust the CA with `caddy trust`. |
| `uv sync` removes `pyright` | It is a dev dependency now — run `uv sync` once from a clean clone; `make check` is the gate. |

## Gateway & Zed integration

The Phase 2 OpenAI-compatible gateway serves every registry alias at a single endpoint, `http://localhost:8080/v1`. Start it with `make serve` (or `vibe serve`), load a model with `vibe load <alias>`, and run `make chat` to stream a test chat through the gateway.

To use it from Zed, add the following to `.zed/settings.json` (project) or your global `settings.json`:

```json
{
  "language_models": {
    "openai_compatible": {
      "vibe-local": {
        "api_url": "http://localhost:8080/v1",
        "available_models": [
          {
            "name": "qwen2.5-coder-7b",
            "display_name": "Qwen2.5 Coder 7B (local vLLM)",
            "max_tokens": 32768,
            "max_output_tokens": 4096,
            "max_completion_tokens": 4096,
            "capabilities": { "tools": true, "images": false, "parallel_tool_calls": false, "prompt_cache_key": false }
          },
          {
            "name": "llama-3.1-8b",
            "display_name": "Llama 3.1 8B (local vLLM)",
            "max_tokens": 16384,
            "max_output_tokens": 4096,
            "max_completion_tokens": 4096,
            "capabilities": { "tools": true, "images": false, "parallel_tool_calls": false, "prompt_cache_key": false }
          },
          {
            "name": "deepseek-r1-14b",
            "display_name": "DeepSeek R1 14B (local vLLM)",
            "max_tokens": 7984,
            "max_output_tokens": 2048,
            "max_completion_tokens": 2048,
            "capabilities": { "tools": false, "images": false, "parallel_tool_calls": false, "prompt_cache_key": false }
          }
        ]
      }
    }
  },
  "agent": {
    "default_model": { "provider": "vibe-local", "model": "qwen2.5-coder-7b" }
  }
}
```

Notes:
*   The API key can be any non-empty string (e.g. `local`); the gateway does not check it.
*   `VIBE_SWAP_ON_REQUEST=false` is the default, so swap to the model you want with `vibe load <alias>` (or `make qwen` / `make llama` / `make deepseek`) before selecting it in Zed.
*   `make tls` + [`deploy/Caddyfile`](deploy/Caddyfile) provide an optional TLS front and only matter if `VIBE_BIND` is widened beyond loopback.

## Admin UI

The Phase 3 admin UI runs at `http://localhost:8081`, served by `vibe serve` (or `vibe serve --only admin`; `make admin` opens it in a browser). It is deliberately **zero-JS**: live views auto-refresh with `<meta http-equiv="refresh">`, actions are plain HTML forms with POST-redirect-GET, and chat streams as chunked HTML, so everything works with JavaScript disabled.

*   **Dashboard** — active alias + state, GPU usage, controller/gateway health, and a link to the current job.
*   **Models** — registry table with Load / Force reload / Unload buttons.
*   **Job pages** — live progress for loads and unloads, auto-refreshing every 2 s until the job reaches a terminal state.
*   **Chat** — a quick sanity check against the active model, streamed token-by-token.
*   **Logs** — the container log tail in the browser, plus `curl -N http://localhost:8081/logs/stream` for a plain-text follow stream.
*   **Developer** — placeholder in Phase 3 (downloads/registry management land in Phase 4).

All state comes from the controller over gRPC; when the controller is stopped, pages degrade gracefully to a "controller unreachable" banner instead of erroring.
