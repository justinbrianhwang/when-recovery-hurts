# Optional: use only on a machine where Docker builds are supported.
# A Vast instance is often already a container; do not assume nested Docker works.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    PYTHONUNBUFFERED=1 \
    WANDB_MODE=disabled
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip git ca-certificates build-essential unzip \
    libvulkan1 vulkan-tools libegl1 libgl1 libglib2.0-0 libx11-6 libxrender1 libsm6 ffmpeg \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace/hci
COPY . /workspace/hci
# Install interactively after inspecting the host. This image has not been built by the package author.
CMD ["/bin/bash"]
