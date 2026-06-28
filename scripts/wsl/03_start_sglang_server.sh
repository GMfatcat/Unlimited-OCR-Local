#!/usr/bin/env bash
set -euo pipefail
# 啟動 SGLang server。可用環境變數覆寫 attention backend（D4 fallback 策略）：
#   ATTN_BACKEND=fa3|flashinfer|triton|torch_native  (預設 fa3)
VENV="$HOME/uocr/.venv-sglang"
PY="$VENV/bin/python"
# 這個 sglang dev build 在 Blackwell(sm_120) 會 runtime JIT 編譯多個 CUDA kernel(rope/decode...)，
# 需要完整工具鏈：nvcc(>=12.8 才支援 sm_120) + ninja + python headers + gcc。
# - venv/bin 進 PATH：pip 安裝的 ninja 可執行檔。
# - /usr/local/cuda/bin 進 PATH 並設 CUDA_HOME：CUDA Toolkit 的 nvcc。
CUDA_ROOT="/usr/local/cuda"
[ -x "$CUDA_ROOT/bin/nvcc" ] || CUDA_ROOT="/usr"   # 尚未裝 toolkit 時退回 /usr（至少讓 deep_gemm import 過）
export PATH="$VENV/bin:/usr/local/cuda/bin:$HOME/.local/bin:$PATH"
export CUDA_HOME="${CUDA_HOME:-$CUDA_ROOT}"
export SGLANG_ENABLE_JIT_DEEPGEMM=0
# 協助各家 kernel 的 Blackwell arch 偵測（RTX 5070 Ti = sm_120 = 12.0）。
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"

REPO="/mnt/c/Users/User/Desktop/project/unlimited-ocr"
MODEL_DIR="$REPO/unlimited-ocr-hf"
ATTN_BACKEND="${ATTN_BACKEND:-fa3}"
# KV cache 靜態預留比例。單頁 base 服務不需大池：0.5 在 16GB 卡 ≈ 10GB，足夠跑單頁 base。
# 換 GPU 請覆寫 MEM_FRACTION（例：80GB H100 設 0.18 ≈ 14GB；不要沿用 0.8，會吃掉 64GB）。
MEM_FRACTION="${MEM_FRACTION:-0.5}"

echo "Starting SGLang server with attention-backend=$ATTN_BACKEND"
exec "$PY" -m sglang.launch_server \
    --model "$MODEL_DIR" \
    --served-model-name Unlimited-OCR \
    --attention-backend "$ATTN_BACKEND" \
    --page-size 1 \
    --mem-fraction-static "$MEM_FRACTION" \
    --context-length 32768 \
    --enable-custom-logit-processor \
    --disable-overlap-schedule \
    --skip-server-warmup \
    --host 0.0.0.0 \
    --port 10000
