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

### 1. Running DeepSeek R1 (14B AWQ)
Create and run a deployment script (e.g., `start_deepseek.sh`) to launch the local OpenAI-compatible inference server:

```bash
docker run -d --gpus all \
  --name vllm-deepseek \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-openai:latest \
  --model casperhansen/DeepSeek-R1-Distill-Qwen-14B-AWQ \
  --quantization awq \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.70
```

### 2. VRAM Optimization Choices for RTX 4070 Ti SUPER (16GB)
*   **`--quantization awq`**: Ingests compressed 4-bit weights, reducing model baseline footprint to ~9GB.
*   **`--max-model-len 16384`**: Strikes a sweet spot for learning/personal scripts, budgeting a generous **16K context window** without exhausting memory.
*   **`--gpu-memory-utilization 0.70`**: Restricts the engine to allocating ~11.4 GB of VRAM. This prevents conflicts with the active CachyOS/Wayland desktop interface (~830MiB overhead).
*   **`-v ~/.cache/huggingface...`**: Caches model weights permanently on the host system to prevent redownloads across container restarts.

### 3. Monitoring the Server
To watch the model initialization, download progress, and API launch logs, run:
```bash
docker logs -f vllm-deepseek
```
The server is fully initialized and listening once the logs output: `Uvicorn running on http://0.0.0.0`.

## Environment Configuration & Automation

This project uses a centralized environment file for credentials and a `Makefile` to handle VRAM purging, container orchestration, and multi-model hot-swapping.

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

The `Makefile` automatically loads your `.env` variables and securely exports them to any downstream container initialization shell scripts. 

Because multiple large models cannot occupy your 16GB VRAM simultaneously, targets that launch an engine depend on the `clean-vram` sequence. This ensures conflicting processes are completely purged before a new server instantiates.

| Command | Action / Description |
| :--- | :--- |
| `make` / `make help` | Displays the help menu and usage instructions. |
| `make deepseek` | Purges active vLLM containers and deploys **DeepSeek-R1 (14B AWQ)** on port `8000`. |
| `make qwen` | Purges active vLLM containers and deploys **Qwen-2.5-Coder (7B AWQ)** on port `8002`. |
| `make llama` | Purges active vLLM containers and deploys **Llama-3.1 (8B AWQ)** on port `8001`. |
| `make stop` | Gracefully tears down all sandbox container instances to completely free up GPU memory. |
| `make kill-ghosts` | Checks for any active or stopped runaway vLLM containers and aggressively drops their footprints. |
| `make status` | Prints a live dashboard of active sandbox containers alongside current GPU compute process memory. |

### 3. Workflow Examples

To hot-swap from your default reasoning model to your code assistant and monitor its health, execute:
```bash
make qwen
make status
```
