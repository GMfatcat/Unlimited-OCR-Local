# Docker 部署規劃提案（Unlimited-OCR）

> 狀態：**H100 版 Dockerfile 草稿已建立並通過 build 驗證**（見 `docker/`）。
> **DGX Spark 暫不進行** —— 以 H100 為主，H100 成功即收尾；Spark（arm64 / sm_121）視情況再議。
> 目的：為兩個目標平台規劃容器化策略、釐清相依限制與風險，並列出動工前必須先確認的事項。
> 日期：2026-06-27

---

## 1. 目標平台

| 平台 | 架構 | GPU / 算力 | 最高 CUDA | 備註 |
|---|---|---|---|---|
| **A. amd64 主機** | x86_64 | **NVIDIA H100（Hopper, sm_90）** | 12.4（驅動上限） | **此主機無對外網路（air-gapped）**。H100 是資料中心卡、sm_90，`fa3` 原生可用。 |
| **B. DGX Spark** | **arm64 (aarch64)** | **GB10 Grace-Blackwell，sm_121(a)**，128GB 統一記憶體 | 13.0 | sm_121 只在 **CUDA 13.0+** 支援。**同樣 air-gapped**；目前只驗證過 vLLM，SGLang 尚未驗證。 |

> **兩台目標主機都無對外網路**。統一交付流程：在**有網路的 Linux build 機** `docker build` → `docker save` 成 tar → 拷貝到目標主機 → `docker load`（**無 registry**）。

> 兩者**架構、CUDA、GPU 代次全不同**，無法用單一映像；必須**分平台各自一套**。
>
> **使用情境定調**：目標是用 **SGLang 的 continuous batching 提供併發 OCR 服務**。Transformers 無批次/併發、需自製 queue、效率差，因此**只當 DGX Spark 萬一 SGLang 不通時的保底**，不是主要交付。→ **H100 上的 SGLang 服務是首要目標**（可行性高）；DGX Spark 的 SGLang 是高風險研究項。

## 2. 核心限制：客製 SGLang wheel 是關鍵變數

本專案 SGLang 路線用的是 **Baidu 提供的客製 dev wheel**
`sglang-0.0.0.dev11416+g92e8bb79e`，它**內含 `unlimited_ocr` 模型支援與自訂 `DeepseekOCRNoRepeatNGramLogitProcessor`**，這是上游 stock SGLang 沒有的。因此：

- **不能直接換成官方 `lmsysorg/sglang` 映像** —— 官方映像不一定有 Unlimited-OCR 模型與該 logit processor。
- 這顆 wheel 的 `Requires-Dist` **釘死**了一組 kernel 相依：
  `torch==2.9.1`、`flashinfer_python==0.6.7.post3`、`flashinfer_cubin==0.6.7.post3`、`sglang-kernel==0.4.1`、`cuda-python==12.9`、`flash-attn-4`、`quack-kernels`…
  這些預編譯 kernel 的**目標是 x86_64 + CUDA 12.8/12.9 + sm_80…sm_120**。

→ **兩個目標都不在這顆 wheel 的「原生支援範圍」**：A 是 cu124（低於 12.8）、B 是 arm64 + cu130 + sm_121。
這正是 Docker 化最大的不確定點（詳見 §6 風險）。

## 3. 兩條路線的風險分級（重要決策）

| 路線 | 相依 | amd64 H100 (12.4) | arm64 (DGX Spark) |
|---|---|---|---|
| ⚡ **SGLang**（**主要目標**，有 continuous batching/併發） | 客製 wheel + flashinfer/sgl-kernel/triton JIT | 🟡 中風險（cu128 stack vs 12.4 驅動 → forward-compat；但 H100/sm_90/`fa3` 本身很成熟） | 🔴 高風險（sm_121a kernel 缺位，連官方/vLLM 都在補；見 §4B） |
| 🤗 **Transformers**（無併發，**僅保底**） | 只要 `torch + transformers`，**eager attention** | 🟢 低風險 | 🟢 低風險（torch 有 cu130 aarch64；eager 不依賴 sm_121 kernel） |

> **定調（已確認）**：H100 **只做 SGLang 單路線**（SGLang 在 H100/sm_90 可行性高）→ **映像不塞 Transformers / 第二個 venv**，更小更單純。Transformers 只在 **DGX Spark 萬一 SGLang 不通**時才當保底。

## 4. 分平台策略

### A. amd64 / H100（驅動上限 CUDA 12.4，且**無網路**）

H100 = Hopper **sm_90**，是這兩個目標裡**最可行**的：`fa3` backend 成熟、bf16 模型不需 fp8/fp4 exotic kernel。唯一要解的是「**客製 wheel 是 cu128，但主機驅動只到 12.4（≈ R550）**」。

**為什麼需要 forward-compat**：cu128 的 torch/kernel 需要 R570+ 驅動；H100 主機是 12.4/R550，**版本不夠** → 直接跑會報 *"CUDA driver version is insufficient"*。
**CUDA Forward Compatibility**（`cuda-compat-12-8` 套件）內含 R570 的 userspace driver libs，載入後就能讓 cu128 app 跑在 R550 驅動上。

> ✅ **回答你的問題（air-gap）**：`cuda-compat-12-8` 是**在「有網路的建置機」build image 時就裝進映像**的，
> **離線主機完全不需要再下載任何東西**。離線部署流程是：
> 1. 在有網路的機器 `docker build`（把客製 wheel、cu128 stack、`cuda-compat-12-8` 全 bake 進去）；
> 2. `docker save` 成 tar → 拷貝到 H100 主機 → `docker load`；
> 3. 直接 `docker run`。
> 離線主機端**唯一**的前提是：本來就要有的 **NVIDIA 驅動 + nvidia-container-toolkit**（跑任何 GPU 容器都需要）。
> forward-compat 的自動掛載需 container-toolkit ≥ 1.17.5 的 `enable-cuda-compat` hook。
> **已確認（2026-07-02）：主機 driver 550.127.08、nvidia-container-toolkit 1.17.2-1（< 1.17.5，無自動 hook）→ 採用「映像內手動 `LD_LIBRARY_PATH=/usr/local/cuda/compat:$LD_LIBRARY_PATH`」**（不依賴主機 toolkit 版本、最穩、一樣不需連網）。R550 遠高於 CUDA 12.8 compat 的 R525 下限，forward-compat 相容性確認。

**替代方案（避開 forward-compat）**：把整套 stack 對齊 cu124。但客製 wheel 釘 `torch==2.9.1`，而 PyTorch 約在 2.7 之後**不再出 cu124 wheel**，且 flashinfer 0.6.7/sgl-kernel 0.4.1 也是 cu128 build → 對齊 cu124 等於要動 torch 版本與重編 kernel，**比 forward-compat 麻煩很多**。故 **建議走 forward-compat**。

**attention backend**：H100 sm_90 → 直接用 **`fa3`**（README 原本的設定），不需 Blackwell 那套 triton-JIT/nvcc 黑魔法。

**Base image 建議**：`nvidia/cuda:12.8.x-cudnn-devel-ubuntu22.04`（cu128 對齊客製 wheel；**devel** 版含 nvcc 備用）＋ `cuda-compat-12-8`。

### B. DGX Spark（arm64 / CUDA 13.0 / sm_121a）

> 你目前在 Spark 上**只跑過 vLLM、尚未驗證 SGLang**。下面是動工前的現況評估。

**現況（2026/06 調查）—— 生態仍在補洞，需有心理準備：**
- 官方有 **`lmsysorg/sglang:spark`** 映像與 **GB10/sm_121a 支援追蹤 issue（sgl #11658）**。
- 但：**`sgl-kernel` 需為 sm_121a 重編**；PyTorch 官方 binary **只內建到 sm_120**（缺 sm_121 kernel）；Triton 在 sm_121a 有 **PTXAS 不支援的指令**；FP8 CUTLASS 在 sm_121a 會 fallback。
- **flashinfer 的 aarch64 wheel 有限**（PyPI 僅 x86_64；aarch64 多靠 nightly/原始碼），且 **sm_121 未被列為獨立 build target**（SM12x 以 `120f` 編譯）。

→ **直接把 Baidu 客製 wheel（釘 flashinfer 0.6.7 / sgl-kernel 0.4.1）搬上 DGX Spark，極可能無法開機**（kernel 非 arm64/sm_121a）。
可行路徑有二，皆需驗證：
1. **移植法**：以官方 `lmsysorg/sglang:spark`（或 cu130-aarch64）為基底，把 **Unlimited-OCR 的模型檔與自訂 logit processor 疊上去**（前提：該版本的 SGLang 介面相容、且能掛入自訂模型）。
2. **重編法**：在 arm64 + cu130 環境**重新編譯** sgl-kernel/flashinfer（指定 sm_121a），再裝客製 wheel —— 成本與不確定性最高。

**Base image 建議**：`nvidia/cuda:13.0.x-cudnn-devel-ubuntu24.04`（arm64）或官方 `lmsysorg/sglang:spark` 作參考。

**保底**：DGX Spark 上**先把 Transformers 路線跑起來**（torch cu130 aarch64 + eager attention），確保有可用的 OCR；SGLang 當研究專案推進。

## 5. 共通設計（兩平台一致）

- **不要把 6.7GB 權重烤進映像**：以 **volume / bind-mount** 掛入（`-v /host/unlimited-ocr-hf:/models/unlimited-ocr`），映像保持輕量、權重可換。
  （air-gap 主機：權重檔一併拷貝過去掛載即可。）
- **服務形態：UI + 推論 server 同一容器**（依你的決定，較簡單）。
  - 容器內**單一 `.venv-sglang`**（含 streamlit）：entrypoint 先 `sglang.launch_server`（10000）→ 等 `/health` → 再 `streamlit run app.py`（8501）。
  - **不含 Transformers / 第二個 venv**（H100 只走 SGLang）。`app.py` 的「Transformers 批次」分支在無該 venv 時會回錯誤訊息；可在映像版把該選項隱藏。
  - `docker run -p 8501:8501 -p 10000:10000`（10000 視需要對外）。
- **建置 / 離線轉移流程（兩台目標通用）**：
  1. 有網路的 Linux build 機：`docker build -t uocr:h100 .`（把客製 wheel、cu128 stack、`cuda-compat-12-8` 全 bake 進去）。
  2. `docker save uocr:h100 | gzip > uocr-h100.tar.gz` → 拷貝到目標主機。
  3. 目標主機：`docker load < uocr-h100.tar.gz` → `docker run ...`（權重以 `-v` 掛入）。
- **VRAM**：H100（80GB）資源充裕，SGLang continuous batching 可開較大併發；DGX Spark 128GB 統一記憶體更寬鬆。
- **沿用我們已知的系統前置**（見 `docs/DEVELOPMENT-NOTES.md`）：`-devel` CUDA base 多半已含 nvcc/headers；仍需確認 `libnuma`、`ninja`(PATH)、`TORCH_CUDA_ARCH_LIST`、`CUDA_HOME`、`SGLANG_ENABLE_JIT_DEEPGEMM=0` 等設定。

## 6. 風險彙整

| 風險 | 平台 | 影響 | 緩解 |
|---|---|---|---|
| cu128 stack 跑在 12.4/R550 驅動上 | A (H100) | 不處理會報 driver insufficient | bake `cuda-compat-12-8`（H100 為資料中心卡，forward-compat 支援良好）；離線不需額外下載 |
| container-toolkit 太舊、無 cuda-compat 自動 hook | A (H100) | forward-compat 不自動生效 | 映像內手動 `LD_LIBRARY_PATH=/usr/local/cuda/compat` |
| sm_121a kernel 缺位（sgl-kernel/flashinfer/PyTorch 只到 sm_120） | B | SGLang 無法在 DGX Spark 開機 | 移植到官方 spark 映像 or 重編 kernel；先用 Transformers 保底 |
| flashinfer/sgl-kernel 無 arm64 穩定 wheel | B | 需 nightly/自編 | 鎖官方 spark 映像版本；或純 Transformers |
| Triton/flashinfer runtime JIT 需 nvcc + 工具鏈 | A、B | 映像需 devel base、體積大 | 用 `*-devel` base，預先 bake 工具鏈 |
| 客製 wheel 與「能跑 sm_121 的新版 SGLang」介面不相容 | B | 移植法失敗 | 動工前先驗證 Unlimited-OCR 模型能否掛上新版 SGLang |

## 7. 動工前必須先確認（Open Questions / 驗證清單）

已確認：H100/sm_90、air-gapped、build 機+tar+`docker load`（無 registry）、**H100 單 SGLang（不留 Transformers）**、UI+server 同容器。剩下：

1. ~~（待補）H100 主機的驅動版本與 nvidia-container-toolkit 版本~~ **已確認（2026-07-02）：driver 550.127.08 / CUDA 12.4 上限、nvidia-container-toolkit 1.17.2-1**。1.17.2 < 1.17.5 → 無自動 hook → 定案用**映像內手動 `LD_LIBRARY_PATH=/usr/local/cuda/compat`**。cu128 + cuda-compat-12-8 forward-compat 相容性確認。
2. H100 是否**獨佔**給此服務？（決定 `--mem-fraction-static`、`--max-running-requests` 等併發調參）
3. 權重在目標主機的**擺放路徑**（給 `-v` 掛載）？
4. （DGX Spark）先做**可行性驗證**：官方 `lmsysorg/sglang:spark`（別台 build、tar/load）能否在 Spark 跑起一般模型？
5. （DGX Spark，最關鍵）**Unlimited-OCR 模型檔 + 自訂 logit processor 能否掛到「支援 sm_121 的較新 SGLang」**？還是非得用這顆 dev11416 wheel？→ 決定「移植」還是「重編 kernel」。

## 8. 建議推進順序（phasing）

1. **Phase 0｜驗證**：回答 §7；在 WSL 這台先用 **cu128 base image + forward-compat** 把「H100 版」映像**建出來並本機 smoke test**（RTX 5070 Ti 雖是 sm_120，可先驗證映像/啟動腳本/UI 串接是否正確，再到 H100 驗證 forward-compat 與 `fa3`）。
2. **Phase 1｜H100 SGLang + UI 單容器（主要交付）**：cu128 `-devel` base + `cuda-compat-12-8` + 客製 wheel；backend `fa3`；entrypoint 串 server→UI；`docker save/load` 上線。
3. **Phase 2｜DGX Spark 可行性 PoC**：先確認 SGLang 能否在 sm_121a 起 server（官方 spark 映像 + 掛模型）。
4. **Phase 3｜DGX Spark 正式化**：成功則容器化；若 SGLang 短期不通，**退而用 Transformers 保底**先提供服務。

---

### 參考來源
- DGX Spark 硬體 / sm_121 / CUDA 13：<https://docs.nvidia.com/dgx/dgx-spark/hardware.html>、<https://github.com/natolambert/dgx-spark-setup>
- SGLang Docker / aarch64 / blackwell build：<https://github.com/sgl-project/sglang/blob/main/docker/Dockerfile>、<https://docs.sglang.io/get_started/install.html>
- SGLang DGX Spark(sm_121a) 支援追蹤：<https://github.com/sgl-project/sglang/issues/11658>
- flashinfer aarch64 / CUDA 13 / SM121 audit：<https://github.com/flashinfer-ai/flashinfer/issues/3170>、<https://docs.flashinfer.ai/installation.html>
- CUDA Forward Compatibility：<https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html>
- vLLM sm_121 aarch64 議題（生態現況佐證）：<https://github.com/vllm-project/vllm/issues/36821>
