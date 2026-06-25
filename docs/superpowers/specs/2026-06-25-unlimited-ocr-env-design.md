# Unlimited-OCR 本地環境建置 — 決策文件 (Design & Decisions)

- 日期: 2026-06-25
- 作者: Claude (自主決策，使用者已授權「先照你的意思做，留下決策文件即可」)
- 目標: 用 uv 建立本地可執行 Unlimited-OCR 的環境，**Transformers** 與 **SGLang** 兩條路線都要跑通；用 repo 內附的測試 PDF 驗證；兩路線都通過後，做一個 **Streamlit UI** 即時查看 OCR 效果。

---

## 1. 環境探勘結果 (Facts)

| 項目 | 值 | 影響 |
|---|---|---|
| 主機 OS | Windows 11 Pro (win32) | SGLang 不支援原生 Windows |
| GPU | NVIDIA RTX 5070 Ti, 16 GB VRAM, **Blackwell sm_120** | FlashAttention-3 (fa3, 針對 Hopper sm_90) 在 Blackwell 上相容性有風險 |
| GPU 驅動 | 610.47, CUDA UMD 13.3 | 支援 cu128/cu129 PyTorch wheel |
| WSL2 | **Ubuntu-24.04 (Running)**，GPU 直通正常 (`nvidia-smi` 在 WSL 內看得到卡) | SGLang 的可行落腳處 |
| WSL Python | 3.12.3 | 符合 README 要求 (3.12) |
| WSL uv | 未安裝 | 需先安裝 |
| WSL 資源 | 16 vCPU / 15 GB RAM / 根目錄 (ext4) 954 GB free | 載入 6.7GB 權重 RAM 偏緊但可行；磁碟充足 |
| 模型權重 | `unlimited-ocr-hf/` 已下載，bf16 單檔 6.7 GB | VRAM 16GB 可容納 |
| 測試 PDF | `unlimited-ocr-hf/Unlimited-OCR.pdf` (亦在 repo 根目錄)，460 KB | 作為兩路線測試輸入 |
| SGLang wheel | `wheel/sglang-0.0.0.dev11416+g92e8bb79e-py3-none-any.whl` (純 python，runtime 依賴 sgl-kernel/flashinfer) | 需在 Linux 安裝 |
| 模型程式碼 flash_attn | 條件式 import (`is_flash_attn_2_available()`)，未安裝會 fallback 到 sdpa/eager；vision encoder `use_flash_attn` 預設 False | **Transformers 路線不強制 flash_attn** |

## 2. 核心決策 (Decisions)

### D1 — 兩條路線都跑在 WSL2 Ubuntu-24.04 內
**理由**: SGLang 本質上只支援 Linux，WSL2 是這台機器上唯一可行路徑，且 GPU 直通已驗證可用。Transformers 也放同一個 WSL 環境，可：(a) 與 README 的 Linux 指令一字不差對應；(b) 避免在原生 Windows 上編譯 `flash_attn` 的痛苦；(c) 單一一致工具鏈。
**否決方案**: 原生 Windows 跑 SGLang（不支援，直接淘汰）。

### D2 — uv 管理，**兩個獨立 venv**
- `.venv-transformers`：torch + transformers 4.57.1 + pymupdf 等（README Transformers 區塊清單）。
- `.venv-sglang`：本機 sglang wheel + kernels + pymupdf。
**理由**: 兩路線的 torch / kernel / flashinfer 依賴版本不同且可能衝突；README 本身也把兩者分開處理。獨立 venv 最乾淨、互不汙染。

### D3 — 權重路徑
先用 WSL 直接讀 Windows 路徑 `/mnt/c/.../unlimited-ocr-hf`（不複製、不佔額外空間）。若 `/mnt/c` 載入過慢或不穩，再複製到 WSL 原生 `~/models/unlimited-ocr`（磁碟有 954GB，可承受）。**實測後在文件補記實際採用哪個。**

### D4 — SGLang attention backend (Blackwell 風險)
README 用 `--attention-backend fa3`。fa3 針對 Hopper，Blackwell sm_120 相容性不確定。
**策略**: 依序嘗試 `fa3` → `flashinfer` → `triton` → `torch_native`，以第一個能成功啟動並產生正確輸出者為準，並在文件記錄最終採用的 backend。

### D5 — Transformers attention 實作
不安裝 `flash_attn`（Windows/WSL 編譯成本高、且非必要）。讓模型 fallback 到 `sdpa`/`eager`。若 `infer()` 路徑硬性要求 flash_attention_2 才能跑，則顯式傳入 `attn_implementation="eager"` 或 `"sdpa"`。

### D6 — Streamlit UI 設計
- 跑在 WSL，Windows 瀏覽器經 `localhost:8501` 存取（WSL2 localhost 轉發）。
- **即時查看 = 串流輸出**，這是 SGLang OpenAI 相容 API 的原生能力，故 UI 主要後端為 **SGLang server 串流**。
- 提供路線切換：另含 `transformers` 模式（非串流，跑完顯示結果）。因 16GB VRAM 無法同時常駐 SGLang(mem-fraction 0.8≈13GB) 與獨立 transformers 模型，UI 明示「一次只跑一個後端」。
- 功能：上傳圖片或 PDF → 選 image_mode(gundam/base) → 即時顯示 OCR markdown，並渲染預覽。

### D7 — 「測試通過」的定義 (驗收標準)
- **Transformers 路線通過**: 對測試 PDF（或其單頁）執行 `infer`/`infer_multi`，產出非空且結構合理的 OCR markdown 檔。
- **SGLang 路線通過**: 啟動 server 成功，`infer.py --pdf Unlimited-OCR.pdf` 回傳每頁非空 OCR，並印出 TPS 統計。
- 兩者皆通過後才動工 Streamlit。

## 3. 架構與資料流

```
Windows 瀏覽器  ──localhost:8501──▶  Streamlit (WSL, .venv-sglang)
                                          │  上傳 image/pdf, 選 mode
                                          ▼
                       ┌─────────────────────────────────────┐
                       │ 後端 A: SGLang server (port 10000)    │  ← 串流, 即時
                       │   OpenAI /v1/chat/completions (stream)│
                       └─────────────────────────────────────┘
                       ┌─────────────────────────────────────┐
                       │ 後端 B: transformers in-process       │  ← 非串流, 跑完顯示
                       │   model.infer / infer_multi           │
                       └─────────────────────────────────────┘
                 PDF → fitz(PyMuPDF) 轉每頁 PNG(300dpi) → 模型
```

## 4. 分階段執行計畫 (Plan)

1. **WSL 前置**: 安裝 uv；確認 `/mnt/c` 可讀權重與 PDF。
2. **Transformers venv**: 建 `.venv-transformers`，依 README 清單安裝（torch cu128/cu129、transformers 4.57.1、pymupdf 等）。
3. **Transformers 冒煙測試**: 寫 `scripts/test_transformers.py`，對測試 PDF 首頁跑 `infer`(gundam) 與 PDF 跑 `infer_multi`(base)，確認輸出非空 → 達成 D7。
4. **SGLang venv**: 建 `.venv-sglang`，裝本機 wheel + kernels + pymupdf（必要時補 flashinfer/torch）。
5. **SGLang 冒煙測試**: 以 D4 策略啟動 server，跑 `infer.py --pdf`，確認每頁輸出非空 → 達成 D7。
6. **Streamlit UI**: 依 D6 實作 `app.py`，串接 SGLang 串流 + transformers 模式。
7. **驗證與收尾**: 端到端跑一次，更新本文件「實際結果」段，列出最終採用的 backend / 權重路徑 / 啟動指令；交付 `Makefile` 或 `run.md` 快速啟動說明。

## 5. 風險與緩解

| 風險 | 緩解 |
|---|---|
| Blackwell 上 fa3/flashinfer 無 sm_120 kernel | D4 backend 逐級 fallback；最差用 `torch_native` |
| sglang dev wheel 依賴版本與 torch 2.10/cu129 不合 | 以 wheel metadata 為準鎖定 torch；必要時降版 |
| `/mnt/c` 載入 6.7GB 過慢 | D3 fallback：複製到 WSL 原生磁碟 |
| WSL 15GB RAM 載入尖峰不足 | 必要時調 `.wslconfig` 提高記憶體；或用 `low_cpu_mem_usage`/直接 device_map |
| SGLang 與 transformers 同時佔 VRAM | UI 一次只跑一個後端（D6） |

## 6. 實際結果 (Results) — 實作後回填
- 採用權重路徑: _待填_
- Transformers 安裝/結果: _待填_
- SGLang 採用 backend / 結果: _待填_
- Streamlit 啟動方式: _待填_
