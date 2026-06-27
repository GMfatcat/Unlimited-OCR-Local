# H100 容器（草稿）— build / 轉移 / 執行

> 狀態：**草稿**。版本（CUDA tag、cuda-compat、torch…）標在 `Dockerfile.h100` 的 `ARG`，待實機校正。
> 規劃依據：[`../docs/DOCKER-DEPLOYMENT-PROPOSAL.md`](../docs/DOCKER-DEPLOYMENT-PROPOSAL.md)

## 內容
- 單容器：**SGLang server（:10000）+ Streamlit UI（:8501）**，只走 SGLang（不含 Transformers）。
- H100 = Hopper **sm_90** → attention-backend `fa3`。
- 客製 wheel 是 cu128；H100 驅動 ~R550（CUDA 12.4）→ 用 **CUDA forward-compat**：映像內含 R570 compat libs，
  entrypoint 以 `LD_LIBRARY_PATH=/usr/local/cuda/compat` 啟用，`NVIDIA_DISABLE_REQUIRE=1` 跳過版本硬檢查。
  **離線主機端零下載。**

## 1) 在「有網路的 Linux build 機」建置
```bash
# build context = repo 根目錄；Dockerfile 在 docker/ 下
docker build -f docker/Dockerfile.h100 -t uocr-h100:draft .

# 版本要改時用 --build-arg（例：對齊不同 CUDA tag）
# docker build -f docker/Dockerfile.h100 \
#   --build-arg CUDA_TAG=12.8.1-cudnn-devel-ubuntu24.04 \
#   --build-arg CUDA_COMPAT_PKG=cuda-compat-12-8 \
#   -t uocr-h100:draft .
```

## 2) 打包成 tar、拷貝到 H100（無 registry）
```bash
docker save uocr-h100:draft | gzip > uocr-h100-draft.tar.gz
# scp / 隨身碟 → H100 主機
```

## 3) 在 H100（air-gapped）載入並執行
```bash
docker load < uocr-h100-draft.tar.gz

# 權重以 -v 掛載（不在映像內）；GPU 用 --gpus
docker run --rm --gpus all \
  -p 8501:8501 -p 10000:10000 \
  -v /path/on/host/unlimited-ocr-hf:/models/unlimited-ocr:ro \
  uocr-h100:draft
```
瀏覽器開 `http://<H100>:8501`。

### 常用可覆寫環境變數（`-e`）
| 變數 | 預設 | 說明 |
|---|---|---|
| `ATTN_BACKEND` | `fa3` | H100 用 fa3 |
| `MEM_FRACTION` | `0.85` | `--mem-fraction-static` |
| `MAX_RUNNING_REQUESTS` | （空） | 獨佔 H100 可設較大併發 |
| `CONTEXT_LENGTH` | `32768` | |
| `MODEL_DIR` | `/models/unlimited-ocr` | 掛載點 |
| `TORCH_CUDA_ARCH_LIST` | `9.0` | H100 = sm_90 |
| `HEALTH_TIMEOUT` | `360` | 等 server 就緒秒數 |

## ⚠️ 動工前待校 / 待確認
1. **`CUDA_TAG` / `cuda-compat` 版本** 是否與 H100 主機驅動相容（R550 → cu128 forward-compat OK，但確切 tag 待定）。
2. H100 主機 **nvidia-container-toolkit 版本**：若 ≥ 1.17.5，可改用其 `enable-cuda-compat` hook 取代手動 `LD_LIBRARY_PATH`（二擇一）。
3. **尚未實際 build/run 驗證**（本機是 sm_120，可先驗證映像/啟動腳本/UI 串接邏輯，但 fa3 與 forward-compat 必須在 H100 上才能真正驗證）。
4. 映像體積：目前用 `devel` base（含 nvcc 備援）。若 sm_90 路徑確認不需 runtime JIT，可改 `runtime` base 縮小體積。
