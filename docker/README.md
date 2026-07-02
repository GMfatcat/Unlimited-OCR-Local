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
| `MEM_FRACTION` | `0.18` | `--mem-fraction-static`；單頁 base 只需小 KV 池，0.18 在 80GB H100 ≈ 14GB。獨佔且要高併發可調高 |
| `MAX_RUNNING_REQUESTS` | （空） | 獨佔 H100 可設較大併發 |
| `CONTEXT_LENGTH` | `32768` | |
| `MODEL_DIR` | `/models/unlimited-ocr` | 掛載點 |
| `TORCH_CUDA_ARCH_LIST` | `9.0` | H100 = sm_90 |
| `HEALTH_TIMEOUT` | `360` | 等 server 就緒秒數 |

## ✅ Build 驗證結果（本機 sm_120，2026-06-27）

- `docker build` **成功**（exit 0）；torch cu128 + 客製 sglang wheel + flashinfer/sgl-kernel + streamlit 全數安裝。
- 容器內 import 正常：`torch 2.9.1+cu128`、`streamlit 1.58.0`、`sglang dev11416`（slim、draft 皆驗過）。
- **已改為多階段瘦身版**（builder=devel → runtime base，複製 uv 可攜 python + venv）。

| 版本 | base | 映像大小 |
|---|---|---|
| 單階段（舊，已淘汰） | `*-devel` | 33.5 GB |
| **多階段 slim（現行 `Dockerfile.h100`）** | `*-runtime` | **24.1 GB** |

- 省下的主要是 devel 工具鏈（nvcc/headers）；剩餘體積大頭是 **torch 自帶的 cu128 libs + flashinfer/sgl-kernel（~15–18GB，在 venv 內）**，難再大砍。
- **轉移檔大小（實測）**：`docker images` 顯示 24.1GB 是「展開後」大小；但 **`docker save` 出來的 tar 只有 ~8.1GB**（OCI tar 內 layer 是已壓縮 blob）。額外 `gzip` 幫助很小（已壓縮）。→ **要 scp 的檔案 ≈ 8GB**。
  > 小坑：本機 Docker Desktop(經 WSL) 下 `docker save -o <WSL路徑>` 與 `docker save | gzip` 會出問題（docker.exe 解析不到 WSL 路徑 / 管線吐空）；**在你的原生 Linux build 機上 `docker save uocr-h100:slim | gzip > x.tar.gz` 正常**。
- 注意：本機是 **sm_120**，**未**驗證 `fa3` 與 forward-compat（需 H100）。entrypoint 邏輯（server→health→UI）以 `bash -n` 驗過。
- 進一步瘦身（如真要）：torch wheel 自帶的 nvidia-* libs 與 base 的 CUDA libs 有重疊，可嘗試去重，但風險高、效益有限。

## H100 主機環境（已確認 2026-07-02）
- driver **550.127.08** / CUDA **12.4** 上限 → cu128 stack 需 forward-compat；H100 是資料中心卡，`cuda-compat-12-8` 支援良好（R550 遠高於 CUDA 12.8 compat 的 R525 下限）。→ `CUDA_TAG=12.8.1` + `cuda-compat-12-8` **確認相容**。
- nvidia-container-toolkit **1.17.2-1**（< 1.17.5）→ **無** `enable-cuda-compat` 自動 hook → 沿用映像內**手動 `LD_LIBRARY_PATH=/usr/local/cuda/compat`**（entrypoint `USE_CUDA_COMPAT=1`，預設值即正確）。

## ⚠️ 仍待實機驗證
1. **尚未在真正的 H100 上 build/run**：本機是 sm_120，只驗過 build/import 與 entrypoint `bash -n`。`fa3` 與 forward-compat **必須在 H100 上才能真正驗證**。
2. **air-gap 下的 HF Hub 抓取**：`kernels` / transformers 首次使用**可能**嘗試連 HF Hub。H100 完全離線，若觸發會卡在網路 timeout。建議在映像加 `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1`（快速失敗、暴露是否有東西需預先 bake）。smoke test 時確認。
3. 映像體積：現為多階段 slim（`runtime` base，24.1GB / tar ≈8GB）。sm_90 走 `fa3` 不需 runtime JIT nvcc，故 runtime 段免 devel，已是合理下限。
