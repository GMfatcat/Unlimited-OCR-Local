#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
# 大型 CUDA/kernel wheel 下載慢，拉高 timeout、降低並行。
export UV_HTTP_TIMEOUT=600
export UV_CONCURRENT_DOWNLOADS=2

REPO="/mnt/c/Users/User/Desktop/project/unlimited-ocr"
WHEEL="$REPO/wheel/sglang-0.0.0.dev11416+g92e8bb79e-py3-none-any.whl"
VENV="$HOME/uocr/.venv-sglang"
PY="$VENV/bin/python"

echo "== create venv (python 3.12) =="
[ -x "$PY" ] || uv venv --python 3.12 "$VENV"

# 先裝 torch 2.9.1 (cu128, 含 Blackwell sm_120 kernel)，避免之後從 PyPI 抓到不含 sm_120 的版本。
echo "== install torch stack (cu128) =="
uv pip install --python "$PY" \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.9.1 torchvision torchaudio==2.9.1

# 安裝本機 sglang wheel + README 指定的 kernels + pymupdf。
# torch 已滿足 ==2.9.1，不會被 PyPI 重裝。其餘依賴(flashinfer/flash-attn-4/sglang-kernel...)從 PyPI 取。
echo "== install sglang wheel + deps =="
uv pip install --python "$PY" \
  "$WHEEL" \
  kernels==0.11.7 \
  pymupdf==1.27.2.2

echo "== sanity: import sglang + torch GPU =="
"$PY" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "cap", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)
import sglang
print("sglang", getattr(sglang, "__version__", "?"))
PY

echo "OK_SGLANG_VENV_DONE"
