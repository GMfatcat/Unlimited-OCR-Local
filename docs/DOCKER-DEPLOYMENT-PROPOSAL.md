# Docker 部署規劃提案（Unlimited-OCR）

> 狀態：**僅調查與提案，尚未建立任何 Docker 檔案或映像**。
> 目的：為兩個目標平台規劃容器化策略、釐清相依限制與風險，並列出動工前必須先確認的事項。
> 日期：2026-06-27

---

## 1. 目標平台

| 平台 | 架構 | GPU / 算力 | 最高 CUDA | 備註 |
|---|---|---|---|---|
| **A. amd64 主機** | x86_64 | 由使用者指定「**最多支援 CUDA 12.4**」 | 12.4 | 12.4 上限代表**不是** Blackwell（sm_120 需 12.8、sm_121 需 13.0）。應為 Ampere/Ada/Hopper（sm_80/86/89/90）。 |
| **B. DGX Spark** | **arm64 (aarch64)** | **GB10 Grace-Blackwell，sm_121(a)**，128GB 統一記憶體 | 13.0 | sm_121 只在 **CUDA 13.0+** 支援。 |

> 兩者**架構、CUDA、GPU 代次全不同**，無法用單一映像；必須**分平台各自一套**。

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

| 路線 | 相依 | amd64 (12.4) | arm64 (DGX Spark) |
|---|---|---|---|
| 🤗 **Transformers** | 只要 `torch + transformers`，**eager attention**，無 exotic kernel | 🟢 低風險（torch 有 cu124 x86_64 build） | 🟢 低風險（torch 有 **cu130 aarch64** build；eager 不依賴 sm_121 kernel） |
| ⚡ **SGLang** | 客製 wheel + flashinfer/sgl-kernel/triton JIT（需 nvcc） | 🟡 中風險（cu124 vs wheel 的 cu128；見 §4A） | 🔴 高風險（sm_121a 連官方都在補；見 §4B） |

> **建議定調**：把 **Transformers 路線當成兩個平台「保證能跑」的 baseline**（先容器化它、確保 OCR 服務可用）；
> **SGLang 路線分平台逐步攻堅**，並把它當成「效能/吞吐增強」而非首發必備。

## 4. 分平台策略

### A. amd64（最高 CUDA 12.4）

關鍵子問題：**這張卡是資料中心卡還是消費卡？** 影響做法：

- **若是資料中心卡（A100/H100/L40…）**：可用 **CUDA Forward Compatibility**（`cuda-compat-12-8` 套件）讓 **cu128 的容器跑在 12.4 驅動**上（需 NVIDIA Container Toolkit ≥1.17.5 的 `enable-cuda-compat` hook、正確設定 `LD_LIBRARY_PATH`）。如此可**沿用既有 cu128 stack 與客製 wheel**，最省事。
- **若是消費卡（RTX 40 系等）**：forward-compat **不支援**，必須讓整個 stack 對齊 cu124 → 客製 wheel 釘死的 flashinfer 0.6.7/sgl-kernel 0.4.1（cu128 build）需找 **cu124 對應版本或自行重編**，難度高。

**attention backend**：
- Hopper **sm_90** → `fa3` 可用（**不需** Blackwell 上那套 triton-JIT 黑魔法），最單純。
- Ampere/Ada **sm_80/86/89** → `flashinfer` 或 `triton`。

**Base image 建議**：`nvidia/cuda:12.4.x-cudnn-devel-ubuntu22.04`（用 **devel** 版，含 nvcc，供 triton/flashinfer 的 runtime JIT）。

### B. DGX Spark（arm64 / CUDA 13.0 / sm_121a）

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
- **雙 venv 對應**：沿用本機的雙環境設計，可做成
  (a) **兩個映像**（`uocr-transformers`、`uocr-sglang`），或
  (b) **一個映像內兩個 venv**。建議 (a)，邊界清楚、相依不互汙染。
- **服務形態**：
  - SGLang 映像 = `python -m sglang.launch_server`（暴露 10000）。
  - UI 映像（Streamlit，暴露 8501）可獨立，透過網路打 SGLang server；或與 SGLang 同容器。
- **VRAM 互斥**：amd64/DGX 若單卡，仍是「SGLang 與 Transformers 一次跑一個」的限制（DGX Spark 128GB 統一記憶體則寬鬆很多）。
- **沿用我們已知的系統前置**（見 `docs/DEVELOPMENT-NOTES.md`）：`-devel` CUDA base 多半已含 nvcc/headers；仍需確認 `libnuma`、`ninja`(PATH)、`TORCH_CUDA_ARCH_LIST`、`CUDA_HOME`、`SGLANG_ENABLE_JIT_DEEPGEMM=0` 等設定。

## 6. 風險彙整

| 風險 | 平台 | 影響 | 緩解 |
|---|---|---|---|
| 客製 wheel 釘 cu128 kernel，與 cu124 不合 | A | SGLang 起不來 | 資料中心卡用 forward-compat；否則重編對齊 cu124 |
| sm_121a kernel 缺位（sgl-kernel/flashinfer/PyTorch 只到 sm_120） | B | SGLang 無法在 DGX Spark 開機 | 移植到官方 spark 映像 or 重編 kernel；先用 Transformers 保底 |
| flashinfer/sgl-kernel 無 arm64 穩定 wheel | B | 需 nightly/自編 | 鎖官方 spark 映像版本；或純 Transformers |
| Triton/flashinfer runtime JIT 需 nvcc + 工具鏈 | A、B | 映像需 devel base、體積大 | 用 `*-devel` base，預先 bake 工具鏈 |
| 客製 wheel 與「能跑 sm_121 的新版 SGLang」介面不相容 | B | 移植法失敗 | 動工前先驗證 Unlimited-OCR 模型能否掛上新版 SGLang |

## 7. 動工前必須先確認（Open Questions / 驗證清單）

1. **amd64 那張卡的型號與算力**（sm_?）、以及**是資料中心卡還是消費卡**？（決定 forward-compat 是否可用、用哪個 attention backend）
2. amd64 主機的**驅動版本**（決定 forward-compat 到哪個 cu 版可行）。
3. DGX Spark 是否能取得 / 已驗證 **官方 `lmsysorg/sglang:spark` 跑一般模型**？（作為移植基準）
4. **Unlimited-OCR 的模型檔（`modeling_*` + 自訂 logit processor）能否掛到「支援 sm_121 的較新 SGLang」**？還是非得用這顆 dev11416 wheel？（最關鍵，決定 B 走移植還是重編）
5. 是否接受「**DGX Spark 首發只上 Transformers 路線**」？
6. 部署形態：UI 與 SGLang **同容器**還是**分容器**（compose）？權重 mount 路徑慣例？

## 8. 建議推進順序（phasing）

1. **Phase 0｜驗證**：回答 §7 的開放問題；在各目標主機上手動確認「Transformers 路線可跑」「SGLang 在該平台能否起 server」。
2. **Phase 1｜Transformers 容器化（兩平台）**：低風險、先讓 OCR 服務在 amd64 與 DGX Spark 都可用（多階段 build、權重 mount、UI 可選）。
3. **Phase 2｜amd64 SGLang 容器化**：依 §4A 決定 forward-compat 或重編；跑通串流 server + UI。
4. **Phase 3｜DGX Spark SGLang 攻堅**：依 §4B 走移植或重編，風險最高，最後做。

---

### 參考來源
- DGX Spark 硬體 / sm_121 / CUDA 13：<https://docs.nvidia.com/dgx/dgx-spark/hardware.html>、<https://github.com/natolambert/dgx-spark-setup>
- SGLang Docker / aarch64 / blackwell build：<https://github.com/sgl-project/sglang/blob/main/docker/Dockerfile>、<https://docs.sglang.io/get_started/install.html>
- SGLang DGX Spark(sm_121a) 支援追蹤：<https://github.com/sgl-project/sglang/issues/11658>
- flashinfer aarch64 / CUDA 13 / SM121 audit：<https://github.com/flashinfer-ai/flashinfer/issues/3170>、<https://docs.flashinfer.ai/installation.html>
- CUDA Forward Compatibility：<https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html>
- vLLM sm_121 aarch64 議題（生態現況佐證）：<https://github.com/vllm-project/vllm/issues/36821>
