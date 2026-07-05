#!/bin/bash

# Stop and remove existing container if it exists
docker rm -f vllm-qwen 2>/dev/null

docker run --gpus all \
  --name vllm-qwen \
  -p 8030:8000 \
  --ipc=host \
  -e HF_TOKEN="$HF_TOKEN" \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-Coder-7B-Instruct-AWQ \
  --quantization awq \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.7

echo "Qwen-Coder container launched. Monitor logs with: docker logs -f vllm-qwen"
