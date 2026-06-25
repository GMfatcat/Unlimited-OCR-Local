#!/usr/bin/env bash
set -euo pipefail

REPO="/mnt/c/Users/User/Desktop/project/unlimited-ocr"
WEIGHTS="$REPO/unlimited-ocr-hf"
PDF="$REPO/Unlimited-OCR.pdf"

echo "== install uv (if missing) =="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

echo "== work dir =="
mkdir -p "$HOME/uocr"

echo "== verify weights + pdf readable =="
ls -la "$WEIGHTS/model-00001-of-000001.safetensors"
ls -la "$WEIGHTS/config.json"
ls -la "$PDF"

echo "OK_SETUP_DONE"
