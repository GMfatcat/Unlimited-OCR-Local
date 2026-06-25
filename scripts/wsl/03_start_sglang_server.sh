#!/usr/bin/env bash
set -euo pipefail
# 啟動 SGLang server。可用環境變數覆寫 attention backend（D4 fallback 策略）：
#   ATTN_BACKEND=fa3|flashinfer|triton|torch_native  (預設 fa3)
export PATH="$HOME/.local/bin:$PATH"

REPO="/mnt/c/Users/User/Desktop/project/unlimited-ocr"
MODEL_DIR="$REPO/unlimited-ocr-hf"
PY="$HOME/uocr/.venv-sglang/bin/python"
ATTN_BACKEND="${ATTN_BACKEND:-fa3}"

echo "Starting SGLang server with attention-backend=$ATTN_BACKEND"
exec "$PY" -m sglang.launch_server \
    --model "$MODEL_DIR" \
    --served-model-name Unlimited-OCR \
    --attention-backend "$ATTN_BACKEND" \
    --page-size 1 \
    --mem-fraction-static 0.8 \
    --context-length 32768 \
    --enable-custom-logit-processor \
    --disable-overlap-schedule \
    --skip-server-warmup \
    --host 0.0.0.0 \
    --port 10000
