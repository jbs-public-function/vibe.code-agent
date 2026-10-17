# vibe.code-agent — vLLM model controller Makefile.
#
# Every target delegates to the `vibe` CLI (see README and models.yaml).
# `.env` is loaded into make and all variables are exported to child
# processes, so `vibe load` sees HF_TOKEN / VLLM_IMAGE_TAG / etc.

-include .env
export

# CLI entry point: `uv run vibe` resolves the editable `vibe` package in
# .venv. Override per-invocation, e.g. `make UV=".venv/bin/vibe" list`.
UV ?= uv run vibe
# Python entry point for scripts (e.g. scripts/test_chat.py).
PY ?= uv run python

# Laboratory container (FR-05.2): image name and pinned pytorch base tag
# (NFR-05.2), both overridable via .env. Host HF cache dir (ADR-8) follows
# VIBE_DATA_DIR from .env and falls back to ~/.cache/huggingface.
LAB_IMAGE ?= vibe-lab
LAB_IMAGE_TAG ?= 2.7.1-cuda12.6-cudnn9-devel
LAB_HF ?= $(if $(VIBE_DATA_DIR),$(VIBE_DATA_DIR),$(HOME)/.cache/huggingface)

# Default target
.PHONY: help
help:
	@echo "vibe.code-agent - vLLM Model Controller (vibe CLI)"
	@echo "=================================================="
	@echo "Usage:"
	@echo "  make deepseek    - Load the deepseek-r1-14b alias"
	@echo "  make qwen        - Load the qwen2.5-coder-7b alias"
	@echo "  make llama       - Load the llama-3.1-8b alias"
	@echo "  make stop        - Stop and remove all managed vLLM containers"
	@echo "  make status      - Show active model state and GPU usage"
	@echo "  make logs        - Follow the active container's logs"
	@echo "  make validate    - Validate .env + models.yaml (exit 0 ok, 2 on error)"
	@echo "  make list        - List registered models (aliases and ports)"
	@echo "  make kill-ghosts - Alias for 'make stop'"
	@echo "  make serve       - Start the control plane (controller + gateway + admin)"
	@echo "  make admin       - Open the admin UI at http://localhost:8081"
	@echo "  make chat        - Stream a test chat through the gateway (8080)"
	@echo "  make tls         - Optional TLS front; requires Docker"
	@echo "  make tls-stop    - Stop the optional TLS front"
	@echo "  make proto       - Regenerate committed gRPC code from proto/vibe/v1/control.proto"
	@echo "  make jev         - Load a model in Jev mode: make jev ALIAS=… [SCHEMA=file.json] (P2, JEV-04.10)"
	@echo "  make typecheck   - Run pyright over src/ and tests/"
	@echo "  make check       - Full pre-commit gate: lock, tests, pyright, torch-free sync"
	@echo "  make lab         - Open a shell in the laboratory container (torch + CUDA)"
	@echo "  make lab-build   - Build the laboratory image (pinned pytorch base)"
	@echo "  make lab-run CMD=\"python inference_hf.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct --prompt hi\" - Run a lab script"

# Targets delegate to the vibe CLI (chat/tls run a test script / Caddy).
.PHONY: qwen llama deepseek stop status logs validate list kill-ghosts serve chat tls admin
qwen:      ; $(UV) load qwen2.5-coder-7b
llama:     ; $(UV) load llama-3.1-8b
deepseek:  ; $(UV) load deepseek-r1-14b
stop:      ; $(UV) unload
status:    ; $(UV) status
logs:      ; $(UV) logs --follow
validate:  ; $(UV) validate
list:      ; $(UV) list
serve:     ; $(UV) serve
admin:     ; xdg-open http://localhost:8081
chat:      ; $(PY) scripts/test_chat.py
tls:       ; docker run -d --name vibe-caddy --network host -v vibe-caddy-data:/data -v $$(pwd)/deploy/Caddyfile:/etc/caddy/Caddyfile caddy:2-alpine
tls-stop:  ; docker rm -f vibe-caddy
kill-ghosts: stop

# Jev (improvements-2, JEV-04.10): load with the default schema, or a custom
# one via SCHEMA (maps to --include-json).
#   make jev ALIAS=qwen2.5-coder-7b
#   make jev ALIAS=qwen2.5-coder-7b SCHEMA=./jev/mine.json
.PHONY: jev
jev:      ; $(UV) load $(ALIAS) --jev $(if $(SCHEMA),--include-json $(SCHEMA))

# Laboratory (FR-05.2/05.3): build the pinned pytorch image, drop into a shell
# or run one lab script. IMP-05.2: --network host lets the lab reach the
# controller/gateway on the host's loopback, and the repo is mounted at the
# SAME absolute path the host sees (`-v $$(pwd):$$(pwd)`,
# `-w $$(pwd)/laboratory`) so paths `vibe register`/`export.py` write inside
# the container resolve identically on the host. The shared HF cache is
# mounted at /root/.cache/huggingface (ADR-8, IMP-05.5). `-e HF_TOKEN` and
# `-e VLLM_IMAGE_TAG` forward the values only when set in make's environment
# (.env exports them); unset is fine for non-gated repos / non-register CLI
# use.
.PHONY: lab-build lab lab-run
lab-build: ; docker build -f laboratory/Dockerfile --build-arg LAB_IMAGE_TAG=$(LAB_IMAGE_TAG) -t $(LAB_IMAGE) .
lab:      ; docker run --rm -it --gpus all --ipc=host --network host -v $$(pwd):$$(pwd) -w $$(pwd)/laboratory -e HF_TOKEN -e VLLM_IMAGE_TAG -v $(LAB_HF):/root/.cache/huggingface $(LAB_IMAGE)
lab-run:  ; docker run --rm --gpus all --ipc=host --network host -v $$(pwd):$$(pwd) -w $$(pwd)/laboratory -e HF_TOKEN -e VLLM_IMAGE_TAG -v $(LAB_HF):/root/.cache/huggingface $(LAB_IMAGE) $(CMD)

# Regenerate the committed codegen (NFR-02.2). Run this after editing
# proto/vibe/v1/control.proto; `make proto` must leave the tree clean.
.PHONY: proto
proto:
	$(PY) -m grpc_tools.protoc -I proto \
		--python_out=src/vibe/gen --grpc_python_out=src/vibe/gen --pyi_out=src/vibe/gen \
		proto/vibe/v1/control.proto

# Static type check over src/ and tests/ (pyright is a dev dependency, IMP-06.1).
.PHONY: typecheck
typecheck:
	uv run pyright

# Pre-commit gate (XR-8, IMP-06.10): lock is in sync, tests pass, pyright is
# clean, and the root environment stays torch-free (NFR-05.1).
.PHONY: check
check:
	uv lock --check
	uv run --offline pytest -q
	uv run pyright
	uv sync --dry-run --offline
