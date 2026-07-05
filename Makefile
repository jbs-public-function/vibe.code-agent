-include .env
export

# Default target
.PHONY: help
help:
	@echo "vibe.code-agent - vLLM Model Controller"
	@echo "========================================"
	@echo "Usage:"
	@echo "  make deepseek    - Clear VRAM and launch DeepSeek-R1 (14B)"
	@echo "  make qwen        - Clear VRAM and launch Qwen-2.5-Coder (7B)"
	@echo "  make llama       - Clear VRAM and launch Llama-3.1 (8B)"
	@echo "  make stop        - Stop and wipe all running vLLM sandbox engines"
	@echo "  make status      - Display active containers and GPU footprint"

# Shared teardown sequence to completely purge conflicting models from VRAM
.PHONY: clean-vram
clean-vram:
	@echo "Cleaning GPU VRAM allocations..."
	-docker rm -f vllm-deepseek vllm-qwen vllm-llama 2>/dev/null
	@sleep 1

.PHONY: deepseek
deepseek: clean-vram
	@echo "Deploying DeepSeek-R1-Distill-Qwen-14B-AWQ..."
	./scripts/start_deepseek.sh

.PHONY: qwen
qwen: clean-vram
	@echo "Deploying Qwen2.5-Coder-7B-Instruct-AWQ..."
	./scripts/start_qwen.sh

.PHONY: llama
llama: clean-vram
	@echo "Deploying Llama-3.1-8B-Instruct-AWQ ..."
	./scripts/start_llama.sh

.PHONY: kill-ghosts
kill-ghosts:
	@echo "Hunting for runaway vLLM ghost containers..."
	@if [ -n "$$(docker ps -aq --filter name=vllm)" ]; then \
		echo "Found active/stopped vLLM containers. Purging now..."; \
		docker rm -f $$(docker ps -aq --filter name=vllm); \
		echo "GPU VRAM cleared of sandbox container footprints."; \
	else \
		echo "Pruning complete. No vLLM ghost containers found."; \
	fi

.PHONY: stop
stop: clean-vram
	@echo "All models deactivated."

.PHONY: status
status:
	@echo "=== Active Sandbox Containers ==="
	-docker ps --filter "name=vllm" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
	@echo ""
	@echo "=== Active GPU Allocation Layer ==="
	-nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
