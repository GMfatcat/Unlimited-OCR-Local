#!/usr/bin/env bash
set -euo pipefail
# 系統層級前置（需 root：wsl -d Ubuntu-24.04 -u root bash ...）。
# SGLang dev wheel 在 Blackwell sm_120 會 runtime JIT 編譯 CUDA kernel，需要完整工具鏈。

echo "== apt base build deps =="
apt-get update -qq
apt-get install -y \
  libnuma1 libnuma-dev \
  build-essential \
  python3-dev python3.12-dev \
  wget ca-certificates

echo "== NVIDIA CUDA Toolkit 12.8 (WSL-Ubuntu repo) =="
if [ ! -x /usr/local/cuda-12.8/bin/nvcc ]; then
  cd /tmp
  KEYRING="cuda-keyring_1.1-1_all.deb"
  wget -q "https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/${KEYRING}"
  dpkg -i "${KEYRING}"
  apt-get update -qq
  # 只裝 nvcc + 開發 headers（cudart-dev / cccl），避免整包 ~3GB 的 driver/profiler。
  apt-get install -y cuda-nvcc-12-8 cuda-cudart-dev-12-8 cuda-crt-12-8 libcublas-dev-12-8 || \
    apt-get install -y cuda-toolkit-12-8
fi

# 確保 /usr/local/cuda 指向 12.8
if [ -d /usr/local/cuda-12.8 ] && [ ! -e /usr/local/cuda ]; then
  ln -s /usr/local/cuda-12.8 /usr/local/cuda
fi

echo "== nvcc version =="
/usr/local/cuda-12.8/bin/nvcc --version | tail -2 || echo "nvcc not found!"
echo "OK_SYSTEM_PREREQS_DONE"
