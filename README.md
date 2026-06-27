<p align="center">
  <img src="assets/baidu.png" width="40%" alt="Baidu Inc." />
</p>

<h1 align="center">Unlimited-OCR · 本地執行 + 即時 OCR UI</h1>

<p align="center">
  在 Windows + WSL2 上以 <b>uv</b> 跑通 <b>Transformers</b> 與 <b>SGLang</b> 兩條推論路線，並提供一個 <b>Streamlit 即時 OCR 介面</b>。
  <br>上游官方說明（英文）見 <a href="./README-en.md">README-en.md</a>。
</p>

---

## 📌 這個專案是什麼

以 [baidu/Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR) 權重為基礎，建立一套**本地可重現**的環境與工具：

- 🤗 **Transformers 路線**：直接用 `transformers` 載模型做 OCR（單張 / 多頁 / PDF）。
- ⚡ **SGLang 路線**：以本機 SGLang wheel 起 OpenAI 相容 server，**逐 token 串流**。
- 🖥️ **Streamlit UI**：上傳圖片 / PDF，**邊掃邊呈現** —— 左欄即時長出版面偵測框、右欄純文字跟著掃描往下捲，含速度指標、逾時 / 上限防護、頁碼回看與 ZIP 匯出。

> 完整的環境決策與實測數據見 [`docs/superpowers/specs/2026-06-25-unlimited-ocr-env-design.md`](docs/superpowers/specs/2026-06-25-unlimited-ocr-env-design.md)；
> 一頁式的指令速查見 [`RUN.md`](RUN.md)；
> **開發踩坑記錄**（Blackwell/SGLang/UI 各種雷與解法）見 [`docs/DEVELOPMENT-NOTES.md`](docs/DEVELOPMENT-NOTES.md)。

## 🧩 環境與架構

| 項目 | 內容 |
|---|---|
| 執行平台 | **WSL2（Ubuntu-24.04）**。SGLang 不支援原生 Windows；WSL2 的 GPU 直通已驗證 |
| GPU | NVIDIA RTX 5070 Ti（**Blackwell sm_120**）/ 16GB |
| 套件管理 | **uv**，兩個獨立 venv（依賴會衝突，必須分開） |
| `~/uocr/.venv-transformers` | torch 2.10 (cu129) / transformers 4.57.1 |
| `~/uocr/.venv-sglang` | 本機 sglang wheel / torch 2.9.1 (cu128) / transformers 5.3.0 / streamlit |
| 權重 | `unlimited-ocr-hf/`（從 WSL 走 `/mnt/c/...` 直讀，不複製） |

> **Blackwell 重點**：SGLang 的 `fa3`、`flashinfer` attention backend 在 sm_120 上不可用，實測採 **`triton`**；
> 並需系統層級的 `libnuma / build-essential / python3-dev / CUDA Toolkit 12.8`（runtime JIT 需要 nvcc≥12.8）。
> 這些都寫進 `scripts/wsl/04_system_prereqs.sh`。

## 🚀 安裝（一次性）

在 Windows 終端機執行（會呼叫 WSL）。`R = /mnt/c/Users/User/Desktop/project/unlimited-ocr`：

```powershell
# 0) 安裝 uv、確認權重可讀
wsl -d Ubuntu-24.04 bash -lc "tr -d '\r' < $R/scripts/wsl/00_setup_uv.sh | bash"
# 1) Transformers venv
wsl -d Ubuntu-24.04 bash -lc "tr -d '\r' < $R/scripts/wsl/01_setup_transformers.sh | bash"
# 2) SGLang venv
wsl -d Ubuntu-24.04 bash -lc "tr -d '\r' < $R/scripts/wsl/02_setup_sglang.sh | bash"
# 3) 系統前置（需 root）：libnuma / gcc / python headers / CUDA Toolkit 12.8
wsl -d Ubuntu-24.04 -u root bash -lc "tr -d '\r' < $R/scripts/wsl/04_system_prereqs.sh | bash"
# 4) UI 用的 streamlit（裝進 sglang venv）
wsl -d Ubuntu-24.04 bash -lc "~/.local/bin/uv pip install --python ~/uocr/.venv-sglang/bin/python streamlit"
```

> 上面把 `$R` 當變數示意；實際請用完整路徑 `/mnt/c/Users/User/Desktop/project/unlimited-ocr`。

## ✅ 測試兩條路線

```powershell
# Transformers（單頁 gundam + 多頁 base）
wsl -d Ubuntu-24.04 bash -lc "~/uocr/.venv-transformers/bin/python /mnt/c/Users/User/Desktop/project/unlimited-ocr/scripts/test_transformers.py"

# SGLang：先起 server（attention-backend 固定 triton），再用 infer.py 跑 PDF
wsl -d Ubuntu-24.04 bash -lc "tr -d '\r' < /mnt/c/Users/User/Desktop/project/unlimited-ocr/scripts/wsl/03_start_sglang_server.sh | ATTN_BACKEND=triton bash"
wsl -d Ubuntu-24.04 bash -lc "cd /mnt/c/Users/User/Desktop/project/unlimited-ocr && ~/uocr/.venv-sglang/bin/python infer.py --pdf ./Unlimited-OCR.pdf --output_dir ./outputs/sglang --image_mode base"
```

實測（RTX 5070 Ti，repo 內 `Unlimited-OCR.pdf`）：Transformers 單頁 gundam ≈ 13.6s；SGLang 串流正常頁 ≈ **200–250 tok/s**。

## 🖥️ 啟動 Streamlit UI

> ⚠️ **SGLang server 與 Transformers 一次只能跑一個**（16GB VRAM 不足以同時常駐）。
> UI 的「SGLang 串流」需要 server；「Transformers 批次」請先停掉 server。

```powershell
# 終端 A：SGLang server
wsl -d Ubuntu-24.04 bash -lc "tr -d '\r' < /mnt/c/Users/User/Desktop/project/unlimited-ocr/scripts/wsl/03_start_sglang_server.sh | ATTN_BACKEND=triton bash"

# 終端 B：UI
wsl -d Ubuntu-24.04 bash -lc "cd /mnt/c/Users/User/Desktop/project/unlimited-ocr && ~/uocr/.venv-sglang/bin/streamlit run app.py"
```

瀏覽器開 **http://localhost:8501**。

### UI 功能
- 🟦 **即時偵測框**：每解析出一個 `<|det|>` 框就畫到左欄圖片上（`title` 紅粗框，其餘依類別配色）。
- 📝 **純文字 + 自動捲動**：右欄只顯示去標記的純文字，終端機式 tail 永遠停在最新、跟著掃描走。
- 📊 **即時指標**：頁數 / 累計時間 / 總 tokens / 平均速度（串流中節流跳動，每頁完成定版）。
- 🧱 **生成上限 `max_tokens`（預設 4096）**：某些頁會「鬼打牆」無限重複生成，此上限讓它**乾淨提早結束**，避免長到異常 token 數吃滿 GPU 拖垮後續頁。
- ⏳ **每頁逾時（預設 30s）**：後備防線，超時即中止跳下一頁。
- ⚠️ 逾時 / 達上限的頁會在**狀態列、純文字結尾、指標列、回看標記、ZIP 內**都標註，方便辨識「結果可能不完整」。
- 🧭 **頁碼回看**：頁數 < 16 用滑桿；≥ 16 用輸入框 + 前往 / 上頁 / 下頁。
- ⬇️ **下載 ZIP**：檔名 `unlimited_ocr_{原檔名}_{到秒時間戳}.zip`；每頁一資料夾，含 `overlay.png`（疊框圖）、`raw.txt`（原始輸出）、`text.txt`（純文字）。
- 📖 **使用說明**：sidebar 的按鈕會跳出寬版說明對話框。

## 🛡️ 防護機制（避免無限重複迴圈）

部分頁面模型會重複生成到 `max_length`（32768），造成卡死數分鐘、甚至把 GPU 記憶體吃滿導致 **server 整體降速**。本專案以兩道防線處理：

1. **主要：`max_tokens` 上限**（server 端）。讓問題頁在上限處**乾淨結束並正常釋放**；正常頁（約 1–3k token）不受影響。實測：迴圈頁停在 4096 token、GPU 記憶體全程穩定。
2. **後備：每頁逾時**（client 端）。即使到不了上限，超時也會中止、關連線（SGLang 會 abort 該請求並釋放），跳下一頁。

> 若長時間高強度使用後仍想完全歸零：`wsl -d Ubuntu-24.04 bash -lc "pkill -f sglang.launch_server"`，再重跑 `03_start_sglang_server.sh`。

## 🗂️ 專案結構

```
app.py                       # Streamlit UI（只管介面）
ocr_backends.py              # 後端邏輯：串流 / 子行程 / 解析 det / 畫框 / ZIP 打包
infer.py                     # 上游：SGLang 併發批次推論
scripts/
  test_transformers.py       # Transformers 路線冒煙測試
  test_ui_stream.py          # UI SGLang 串流測試
  test_ui_transformers.py    # UI Transformers 批次測試
  test_overlay.py            # det 解析 / 純文字 / 畫框 離線測試
  test_timeout.py            # 逾時中止 + server 復原測試
  test_deepseek.py           # 用多頁 PDF 驗證逾時 / 迴圈頁
  ocr_once.py                # 單張圖 Transformers OCR（給 UI 子行程呼叫）
  wsl/                       # WSL 安裝 / 啟動腳本（00–04 + wait_health）
docs/superpowers/specs/      # 環境決策文件
unlimited-ocr-hf/            # 模型權重（已 gitignore）
README-en.md                 # 上游官方說明（英文）
```

## 🙏 致謝 / 引用

模型與方法來自百度 Unlimited-OCR，並感謝 DeepSeek-OCR、PaddleOCR。引用資訊與原始說明見 [README-en.md](./README-en.md)。
