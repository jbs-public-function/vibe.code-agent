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

# Default target
.PHONY: help
help:
	@echo "vibe.code-agent - vLLM Model Controller (vibe CLI)"
	@echo "=================================================="
	@echo "Usage:"
	@echo "  make deepseek    - Load DeepSeek-R1-Distill-Qwen-14B on port 8010"
	@echo "  make qwen        - Load Qwen2.5-Coder-7B on port 8030"
	@echo "  make llama       - Load Llama-3.1-8B on port 8020"
	@echo "  make stop        - Stop and remove all managed vLLM containers"
	@echo "  make status      - Show active model state and GPU usage"
	@echo "  make logs        - Follow the active container's logs"
	@echo "  make validate    - Validate .env + models.yaml (exit 0 ok, 2 on error)"
	@echo "  make list        - List registered models and their ports"
	@echo "  make kill-ghosts - Alias for 'make stop'"
	@echo "  make serve       - Start the control plane (controller + gateway; admin later)"
	@echo "  make chat        - Stream a test chat through the gateway (8080)"
	@echo "  make tls         - Optional TLS front; requires Docker"
	@echo "  make proto       - Regenerate committed gRPC code from proto/vibe/v1/control.proto"

# Targets delegate to the vibe CLI (chat/tls run a test script / Caddy).
.PHONY: qwen llama deepseek stop status logs validate list kill-ghosts serve chat tls
qwen:      ; $(UV) load qwen2.5-coder-7b
llama:     ; $(UV) load llama-3.1-8b
deepseek:  ; $(UV) load deepseek-r1-14b
stop:      ; $(UV) unload
status:    ; $(UV) status
logs:      ; $(UV) logs --follow
validate:  ; $(UV) validate
list:      ; $(UV) list
serve:     ; $(UV) serve
chat:      ; $(PY) scripts/test_chat.py --model qwen2.5-coder-7b
tls:       ; docker run -d --name vibe-caddy --network host -v $$(pwd)/deploy/Caddyfile:/etc/caddy/Caddyfile caddy:2-alpine
kill-ghosts: stop

# Regenerate the committed codegen (NFR-02.2). Run this after editing
# proto/vibe/v1/control.proto; `make proto` must leave the tree clean.
.PHONY: proto
proto:
	.venv/bin/python -m grpc_tools.protoc -I proto \
		--python_out=src/vibe/gen --grpc_python_out=src/vibe/gen --pyi_out=src/vibe/gen \
		proto/vibe/v1/control.proto
