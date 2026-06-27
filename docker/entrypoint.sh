#!/usr/bin/env bash
# =============================================================================
# H100 容器 entrypoint：啟動 SGLang server → 等 /health → 啟動 Streamlit UI（前景）
# 草稿（DRAFT）。所有參數可由 docker run -e 覆寫。
# =============================================================================
set -euo pipefail

VENV=/opt/uocr/venv
export PATH="$VENV/bin:/usr/local/cuda/bin:$PATH"

# --- CUDA forward-compat：讓映像內的 R570 compat libs 取代主機 R550 driver libs，
#     使 cu128 stack 能跑在 12.4 主機上。（主機端零下載；不依賴 container-toolkit 的 hook）
export LD_LIBRARY_PATH="/usr/local/cuda/compat:${LD_LIBRARY_PATH:-}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export SGLANG_ENABLE_JIT_DEEPGEMM="${SGLANG_ENABLE_JIT_DEEPGEMM:-0}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"   # H100 = sm_90

MODEL_DIR="${MODEL_DIR:-/models/unlimited-ocr}"
ATTN_BACKEND="${ATTN_BACKEND:-fa3}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
SERVER_PORT="${SERVER_PORT:-10000}"
UI_PORT="${UI_PORT:-8501}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-}"   # 留空＝SGLang 自動；獨佔 H100 可設較大值
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-360}"            # 等 server 就緒最長秒數

if [ ! -d "$MODEL_DIR" ]; then
  echo "[entrypoint] ERROR: 權重目錄不存在：$MODEL_DIR （請用 -v 掛載）" >&2
  exit 1
fi

echo "[entrypoint] starting SGLang server (backend=$ATTN_BACKEND, model=$MODEL_DIR) ..."
extra_args=()
[ -n "$MAX_RUNNING_REQUESTS" ] && extra_args+=(--max-running-requests "$MAX_RUNNING_REQUESTS")

python -m sglang.launch_server \
    --model "$MODEL_DIR" \
    --served-model-name Unlimited-OCR \
    --attention-backend "$ATTN_BACKEND" \
    --page-size 1 \
    --mem-fraction-static "$MEM_FRACTION" \
    --context-length "$CONTEXT_LENGTH" \
    --enable-custom-logit-processor \
    --disable-overlap-schedule \
    --skip-server-warmup \
    --host 0.0.0.0 \
    --port "$SERVER_PORT" \
    "${extra_args[@]}" &
SERVER_PID=$!

# 容器停止時一起收掉 server
cleanup() { echo "[entrypoint] stopping ..."; kill "$SERVER_PID" 2>/dev/null || true; }
trap cleanup TERM INT EXIT

echo "[entrypoint] waiting for /health (timeout ${HEALTH_TIMEOUT}s) ..."
deadline=$((SECONDS + HEALTH_TIMEOUT))
until curl -sf "http://127.0.0.1:${SERVER_PORT}/health" >/dev/null; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[entrypoint] ERROR: SGLang server 在就緒前結束。" >&2
        exit 1
    fi
    if [ "$SECONDS" -ge "$deadline" ]; then
        echo "[entrypoint] ERROR: 等待 server 就緒逾時。" >&2
        exit 1
    fi
    sleep 3
done
echo "[entrypoint] server ready. launching Streamlit UI on :${UI_PORT}"

# UI 走前景（PID 1 角色），讓容器隨 UI 生命週期存活
export SGLANG_SERVER_URL="http://127.0.0.1:${SERVER_PORT}"
exec streamlit run /opt/uocr/app.py \
    --server.headless true \
    --server.port "$UI_PORT" \
    --server.address 0.0.0.0
