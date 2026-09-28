# vibe laboratory — torch/trl environment for QLoRA, distillation and export (FR-05.2).
#
# Host-venv alternative (FR-05.3): skip this image entirely and run
# `uv sync --project laboratory` on the host; the project's uv index config
# (laboratory/pyproject.toml) pins torch 2.7.1+cu126 wheels, so the venv gets
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

# Workspace metadata + the root package source. Scripts are bind-mounted at
# the repo's host path at runtime (IMP-05.2), so only what `uv sync` needs is
# baked in.
COPY pyproject.toml /workspace/pyproject.toml
COPY uv.lock /workspace/uv.lock
COPY laboratory/pyproject.toml /workspace/laboratory/pyproject.toml
COPY src/vibe /workspace/src/vibe
WORKDIR /workspace
# --frozen uses the committed lock, so the container venv matches the host
# venv bit-for-bit. --all-packages (IMP-05.2) also installs the root `vibe`
# package, giving the container the `vibe` CLI (serving guard, `vibe
# register`). The base image's own torch (2.7.1, cp313) is deliberately
# replaced by the lock's torch 2.7.1+cu126 for CPython 3.12 — same CUDA
# userland as the image (IMP-05.4), so there is exactly one torch in the venv.
RUN uv sync --all-packages --frozen --no-dev --python 3.12
# cmake/ninja serve the optional llama.cpp CUDA build below (IMP-05.6).
RUN apt-get update \
    && apt-get install -y --no-install-recommends cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*
# Optional: llama.cpp with CUDA for GGUF inference; uncomment to build the
# container with the `llamacpp` extra. The image is larger and the build is
# slower (spec risks), hence commented out. GGML_CUDA_ARCHITECTURES=89 pins
# the RTX 4070 Ti SUPER (sm_89): the `native` default fails in a GPU-less
# build (IMP-05.6).
# RUN CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=89" uv sync --project laboratory --frozen --extra llamacpp --python 3.12

# IMP-05.5 (single rule): VIBE_DATA_DIR **is** HF_HOME; snapshots land under
# $HF_HOME/hub/models--* and laboratory/ scripts read the same location the
# controller's cache scanner does. The cache arrives via the `make lab`
# runtime mount, not baked in.
ENV HF_HOME=/root/.cache/huggingface
ENV VIBE_DATA_DIR=/root/.cache/huggingface
# IMP-05.6: the venv's python/CLI by default — a plain `docker run` would
# otherwise get the base image's Python 3.13.
ENV PATH=/workspace/.venv/bin:$PATH

CMD ["/bin/bash"]
