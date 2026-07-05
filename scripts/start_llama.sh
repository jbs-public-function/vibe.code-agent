#!/bin/bash

# Stop and remove existing container if it exists
docker rm -f vllm-llama 2>/dev/null

docker run --gpus all \
  --name vllm-llama \
  -p 8020:8000 \
  --ipc=host \
  -e HF_TOKEN="$HF_TOKEN" \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
  --quantization awq \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.75

echo "Llama-3.1 container launched. Monitor logs with: docker logs -f vllm-llama"
