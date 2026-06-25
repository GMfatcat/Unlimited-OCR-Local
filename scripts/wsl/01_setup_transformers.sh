#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
# pypi.nvidia.com is slow for the large CUDA wheels; raise timeout, fewer parallel downloads.
export UV_HTTP_TIMEOUT=600
export UV_CONCURRENT_DOWNLOADS=2

VENV="$HOME/uocr/.venv-transformers"

echo "== create venv (python 3.12) =="
[ -x "$VENV/bin/python" ] || uv venv --python 3.12 "$VENV"

# uv pip install into the target venv via VIRTUAL_ENV
export VIRTUAL_ENV="$VENV"

echo "== install torch/torchvision (Blackwell sm_120 -> cu129) =="
uv pip install --python "$VENV/bin/python" \
  --index-url https://download.pytorch.org/whl/cu129 \
  torch==2.10.0 torchvision==0.25.0

echo "== install transformers + model deps (PyPI) =="
uv pip install --python "$VENV/bin/python" \
  transformers==4.57.1 \
  Pillow==12.1.1 \
  matplotlib==3.10.8 \
  einops==0.8.2 \
  addict==2.4.0 \
  easydict==1.13 \
  pymupdf==1.27.2.2 \
  psutil==7.2.2

echo "== sanity: torch sees GPU =="
"$VENV/bin/python" - <<'PY'
import torch
print("torch", torch.__version__, "cuda_avail", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
PY

echo "OK_TRANSFORMERS_VENV_DONE"
