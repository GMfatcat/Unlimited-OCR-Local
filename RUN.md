# Unlimited-OCR 本地執行說明

> 環境跑在 **WSL2 (Ubuntu-24.04)**，GPU = RTX 5070 Ti。用 `uv` 管理兩個獨立 venv。
> 詳細決策見 `docs/superpowers/specs/2026-06-25-unlimited-ocr-env-design.md`。

## 目錄
- `~/uocr/.venv-transformers` — Transformers 路線 (torch 2.10 / transformers 4.57.1)
- `~/uocr/.venv-sglang` — SGLang 路線 (sglang dev wheel / torch 2.9.1)
- 權重: `unlimited-ocr-hf/`（從 WSL 走 `/mnt/c/...` 讀取）
- 腳本: `scripts/`，UI: `app.py`

## 一、建立環境（一次性）
在 Windows 終端機執行（會呼叫 WSL）。`R` 代表 repo 路徑 `/mnt/c/Users/User/Desktop/project/unlimited-ocr`：
```powershell
# 0) uv + 確認權重
wsl -d Ubuntu-24.04 bash -lc "tr -d '\r' < /mnt/c/Users/User/Desktop/project/unlimited-ocr/scripts/wsl/00_setup_uv.sh | bash"
# 1) Transformers venv（torch 2.10+cu129 / transformers 4.57.1）
wsl -d Ubuntu-24.04 bash -lc "tr -d '\r' < /mnt/c/Users/User/Desktop/project/unlimited-ocr/scripts/wsl/01_setup_transformers.sh | bash"
# 2) SGLang venv（本機 wheel / torch 2.9.1+cu128）
wsl -d Ubuntu-24.04 bash -lc "tr -d '\r' < /mnt/c/Users/User/Desktop/project/unlimited-ocr/scripts/wsl/02_setup_sglang.sh | bash"
# 3) 系統前置（需 root）：libnuma / gcc / python headers / CUDA Toolkit 12.8（SGLang 在 Blackwell 需 runtime JIT）
wsl -d Ubuntu-24.04 -u root bash -lc "tr -d '\r' < /mnt/c/Users/User/Desktop/project/unlimited-ocr/scripts/wsl/04_system_prereqs.sh | bash"
# 4) UI 需要 streamlit（裝進 sglang venv）
wsl -d Ubuntu-24.04 bash -lc "~/.local/bin/uv pip install --python ~/uocr/.venv-sglang/bin/python streamlit"
```

## 二、測試兩條路線
```powershell
# Transformers
wsl -d Ubuntu-24.04 bash -lc "~/uocr/.venv-transformers/bin/python /mnt/c/Users/User/Desktop/project/unlimited-ocr/scripts/test_transformers.py"

# SGLang（自動起 server 並對測試 PDF 推論）
wsl -d Ubuntu-24.04 bash -lc "cd /mnt/c/Users/User/Desktop/project/unlimited-ocr && ~/uocr/.venv-sglang/bin/python infer.py --pdf ./Unlimited-OCR.pdf --output_dir ./outputs/sglang --image_mode base"
```

## 三、啟動 Streamlit UI（即時 OCR）
> **重要：SGLang server 與 Transformers 一次只能跑一個**（16GB VRAM 不足以同時常駐）。
> UI 的「SGLang 串流即時」模式需 server；「Transformers 批次」模式請先停掉 server。

```powershell
# 終端 A：啟動 SGLang server（attention-backend 已固定為 triton；fa3/flashinfer 不支援 Blackwell sm_120）
wsl -d Ubuntu-24.04 bash -lc "tr -d '\r' < /mnt/c/Users/User/Desktop/project/unlimited-ocr/scripts/wsl/03_start_sglang_server.sh | ATTN_BACKEND=triton bash"

# 終端 B：啟動 UI
wsl -d Ubuntu-24.04 bash -lc "cd /mnt/c/Users/User/Desktop/project/unlimited-ocr && ~/uocr/.venv-sglang/bin/streamlit run app.py"
```
瀏覽器開 http://localhost:8501 。側欄選後端、image_mode（單張建議 gundam、PDF 建議 base），上傳圖片/PDF 後即時看 OCR。

> 停止 server：`wsl -d Ubuntu-24.04 bash -lc "pkill -f sglang.launch_server"`
> 最終採用的 backend、系統前置、實測數據見決策文件「實際結果」段：
> `docs/superpowers/specs/2026-06-25-unlimited-ocr-env-design.md`
