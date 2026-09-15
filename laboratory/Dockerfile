# vibe laboratory — torch/trl environment for QLoRA, distillation and export (FR-05.2).
#
# Host-venv alternative (FR-05.3): skip this image entirely and run
# `uv sync --project laboratory` on the host; the project's uv index config
# (laboratory/pyproject.toml) pins torch 2.6.0+cu124 wheels, so the venv gets
# CUDA support without a container.
#
# Base tag pinned per NFR-05.2; override with --build-arg LAB_IMAGE_TAG=...
ARG LAB_IMAGE_TAG=2.7.1-cuda12.6-cudnn9-devel
FROM pytorch/pytorch:${LAB_IMAGE_TAG}

# uv for fast, reproducible sync (FR-05.2)
COPY --from=ghcr.io/astral-sh/uv:0.9.0 /uv /uvx /bin/

# The base image ships Python 3.13, but the lab deps are locked for 3.12
# (autoawq-kernels publishes cp312 wheels only). Pin the lab venv to the same
# CPython 3.12 the host venv uses, so the lock applies bit-for-bit in both.
RUN uv python install 3.12

# Workspace metadata only — scripts are bind-mounted at /lab at runtime (FR-05.2)
COPY pyproject.toml /workspace/pyproject.toml
COPY uv.lock /workspace/uv.lock
COPY laboratory/pyproject.toml /workspace/laboratory/pyproject.toml
WORKDIR /workspace
# --frozen uses the committed lock, so the container venv matches the host
# venv bit-for-bit. The base image's own torch is deliberately replaced here
# by the lock's torch 2.6.0+cu124 (venv/container parity).
RUN uv sync --project laboratory --frozen --no-dev --python 3.12
# Optional: llama.cpp with CUDA (GGML_CUDA=on) for GGUF inference; uncomment
# to build the container with the `llamacpp` extra. The image is larger and
# the build is slower (spec risks), hence commented out.
# RUN CMAKE_ARGS="-DGGML_CUDA=on" uv sync --project laboratory --frozen --extra llamacpp --python 3.12

# HF_HOME and laboratory/ arrive via `make lab` runtime mounts, not baked in.
# Scripts default to this path so the mounted cache is shared by default.
ENV VIBE_DATA_DIR=/root/.cache/huggingface

CMD ["/bin/bash"]
