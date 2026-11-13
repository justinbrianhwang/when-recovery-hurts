#!/usr/bin/env bash
set -euo pipefail
if [[ $(id -u) -ne 0 ]]; then
  echo "Run with sudo/root only for these user-space system packages. No driver installation is performed." >&2
  exit 2
fi
apt-get update
apt-get install -y --no-install-recommends git unzip ca-certificates build-essential python3-venv \
  libvulkan1 vulkan-tools libegl1 libgl1 libglib2.0-0 libx11-6 libxrender1 libsm6 ffmpeg tmux
# Never install NVIDIA kernel drivers or restart a rented host from this script.
