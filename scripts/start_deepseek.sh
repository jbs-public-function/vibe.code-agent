#!/bin/bash

# Stop and remove existing container if it exists
docker rm -f vllm-deepseek 2>/dev/null

docker run --gpus all \
  --name vllm-deepseek \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8010:8000 \
  --ipc=host \
  -e HF_TOKEN="$HF_TOKEN" \
  vllm/vllm-openai:latest \
  --model casperhansen/DeepSeek-R1-Distill-Qwen-14B-AWQ \
  --quantization awq \
  --max-model-len 7984 \
  --gpu-memory-utilization 0.85

echo "DeepSeek-R1 container launched. Monitor logs with: docker logs -f vllm-deepseek"
