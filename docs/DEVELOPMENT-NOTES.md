# 開發踩坑記錄（Unlimited-OCR 本地化 + UI）

> 把這次「在 Windows + WSL2 + RTX 5070 Ti（Blackwell sm_120）」跑通 Transformers / SGLang 兩條路線、
> 並做出 Streamlit 即時 UI 的過程中遇到的坑與解法整理在此，供後續維護與 Docker 化參考。

---

## 1. 平台與環境

| 坑 | 解法 |
|---|---|
| **SGLang 不支援原生 Windows** | 全部跑在 **WSL2（Ubuntu-24.04）**，GPU 直通已驗證（WSL 內 `nvidia-smi` 看得到卡）。 |
| **兩條路線依賴衝突** | 開**兩個獨立 uv venv**：`transformers`（torch 2.10/cu129、transformers 4.57.1）與 `sglang`（本機 wheel、torch 2.9.1/cu128、transformers **5.3.0**）。同一個 venv 必爆。 |
| **PowerShell / Git Bash 巢狀引號地獄**（`$VENV`、`2>/dev/null`、`$(...)` 常被外層 shell 吃掉或解析錯） | 一律把指令寫成 **shell 腳本檔**，用 `wsl bash -lc "tr -d '\r' < script.sh \| bash"` 執行。 |
| **Windows 換行 CRLF 讓 bash 腳本爆掉** | 執行前一律 `tr -d '\r'` 去除 CR。 |
| **uv 從 `pypi.nvidia.com` 抓大型 CUDA wheel 常逾時** | `export UV_HTTP_TIMEOUT=600`、`UV_CONCURRENT_DOWNLOADS=2` 降低並行、拉長逾時。 |
| 模型權重 6.7GB 放在 Windows 端 | 從 WSL 直接走 `/mnt/c/...` 讀，載入僅約 45s，**不需複製**到 WSL 原生磁碟。 |

## 2. Transformers 路線

- **必須 `attn_implementation="eager"`**：`config.use_mla=False` → 模型只註冊 `mha_eager`（`SlidingWindowLlamaAttention`），
  沒有 `mha_sdpa` / `mha_flash_attention_2`。若用 HF 預設（sdpa）會直接 `KeyError`。
- 不需安裝 `flash_attn`（vision encoder `use_flash_attn` 預設 False；語言模型走 eager）。
- `infer` / `infer_multi` 在 `save_results=True` 時輸出到 `{output_path}/result.md`。

## 3. SGLang 路線在 Blackwell（sm_120）— 最深的坑

啟動 server 是一連串「缺東缺西」，依序撞牆並解掉：

1. **`deep_gemm` import 失敗**（`AssertionError`，`_find_cuda_home()` 找不到）
   → 設 `CUDA_HOME`（指向 `/usr` 或 CUDA Toolkit 即可通過；bf16 模型不會真的呼叫 fp8 kernel）。另設 `SGLANG_ENABLE_JIT_DEEPGEMM=0`。
2. **`sgl_kernel` 載入失敗：`libnuma.so.1: cannot open shared object file`**
   → `apt install libnuma1 libnuma-dev`。
3. **`--attention-backend fa3` 直接拒絕**：`FlashAttention v3 requires SM>=80 and SM<=90`（Blackwell sm_120 不在範圍）
   → 換 backend。
4. **`flashinfer` backend**：cuda graph capture 時要 **JIT 編譯** sliding-window decode kernel，`check_cuda_arch` / 需 nvcc 失敗
   → 再換 backend。
5. **`triton` backend**：可用，但 JIT 過程接連缺工具：
   - Triton 編譯 stub 找不到 **C compiler** → `apt install build-essential`（gcc）。
   - 接著缺 **`Python.h`** → `apt install python3-dev python3.12-dev`。
   - sglang `jit_kernel`（fused RoPE，走 tvm_ffi）需 **`ninja`** → pip 的 ninja 在 `venv/bin`，把它加進 `PATH`。
   - 以及需要 **nvcc** 來編譯 CUDA kernel，且 **nvcc ≥ 12.8 才支援 sm_120** → 裝 **CUDA Toolkit 12.8**（`cuda-nvcc-12-8` 等）。
6. `Failed to get device capability: SM 12.x requires CUDA >= 12.9` 警告：來自 cu128 的 torch 對 sm_120 的查詢；有 flashinfer cubin 撐著，**非致命**。設 `TORCH_CUDA_ARCH_LIST=12.0` 協助 arch 偵測。

**最終可用組合**：`--attention-backend triton` + `CUDA_HOME=/usr/local/cuda`(Toolkit 12.8) + `PATH` 含 `venv/bin`(ninja) 與 nvcc + `SGLANG_ENABLE_JIT_DEEPGEMM=0` + `TORCH_CUDA_ARCH_LIST=12.0`。
全部寫進 `scripts/wsl/03_start_sglang_server.sh` 與 `scripts/wsl/04_system_prereqs.sh`。

> 補充：MoE 沒有 RTX 5070 Ti 的 tuned config（只是效能警告）；mem-fraction 0.8 載入後 avail ≈ 1.9GB，可服務。

## 4. 無限重複生成（鬼打牆）與效能退化

- 某些頁（實測 DeepSeek-OCR.pdf 第 8、17 頁）模型會**重複生成到 `max_length`（32768）**，卡死數分鐘。
- **關鍵發現**：跑到 7000+ token 的異常長度會把 **GPU 記憶體一路吃到接近滿（15952/16303）**，導致 **server 整體降速約 4 倍**（正常頁從 ~250 → ~80 tok/s）。重啟 server 即復原。
- **中止請求本身不洩漏記憶體**（連續中止多次，GPU 紋風不動）；問題出在「真的生成到超長」。
- **解法（兩道防線）**：
  1. **主要：`max_tokens` 上限**（OpenAI 參數，預設 4096）→ 問題頁在上限處**乾淨結束、正常釋放**；正常頁（1–3k）不受影響。
  2. **後備：每頁逾時**（client 端，預設 30s）→ 超時 `break` 關連線，SGLang 偵測斷線 abort 該請求。
- 兩種情況都會在 **狀態列 / 純文字結尾 / 指標列 / 回看標記 / ZIP** 標註，提示「結果可能不完整」。

## 5. Streamlit UI 的坑

| 坑 | 解法 |
|---|---|
| 每頁往下**堆疊**，新頁與即時 tail 跑到視窗外 | 改成**固定一組兩欄、原地替換**；多頁用頁碼導覽回看。 |
| 圖片左對齊留白 / 想加寬文字 | 欄比例 **2:3** + 圖片 `use_container_width=True` 填滿欄寬。 |
| 指標列壓縮圖片高度 | 標題移到 sidebar、指標改**單行**。 |
| 串流中每 token 做 `strip_markup`（O(n)）拖慢消化 | 文字/指標**節流 0.15s**。 |
| **達上限/逾時頁文字「閃一下就沒」** | 真因是**疊框圖太大（~0.5MB）每秒重送多次塞爆瀏覽器、餓死文字更新**。把**圖片獨立慢節流（0.7s）**、文字維持 0.15s。 |
| `draw_dets` 崩潰 `y1 must be >= y0` | 模型在壞頁吐出**反向座標**的框；`draw_dets` 改成 `sorted` 正規化 + clamp + 跳過退化框。 |
| `max_tokens` 到了卻無提示就跳頁 | 以 `page_units >= max_tokens` 判定「達上限」並標註。 |
| 回看頁多時拖拉滑桿不便 | 頁數 ≥ 16 改**輸入框 + 前往/上頁/下頁**（按鈕用 `on_click` callback 確保穩定）。 |
| det 座標系 | `<\|det\|>label [x1,y1,x2,y2]<\|/det\|>` 為 **0–999 正規化**，還原像素 `x*W/999`、`y*H/999`。 |

## 6. 一句話總結

> 在消費級 Blackwell（sm_120）上跑這個 SGLang dev build，本質是「**湊齊一整套 runtime JIT 工具鏈（nvcc≥12.8 + ninja + gcc + python headers + libnuma）並改用 triton backend**」；
> 推論層面則要用 **max_tokens + 逾時** 擋住重複生成造成的卡死與記憶體退化；UI 層面最大的雷是**大圖高頻重送**。
