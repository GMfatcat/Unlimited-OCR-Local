# Unlimited-OCR 本地執行說明

> 環境跑在 **WSL2 (Ubuntu-24.04)**，GPU = RTX 5070 Ti。用 `uv` 管理兩個獨立 venv。
> 詳細決策見 `docs/superpowers/specs/2026-06-25-unlimited-ocr-env-design.md`。

## 目錄
- `~/uocr/.venv-transformers` — Transformers 路線 (torch 2.10 / transformers 4.57.1)
- `~/uocr/.venv-sglang` — SGLang 路線 (sglang dev wheel / torch 2.9.1)
- 權重: `unlimited-ocr-hf/`（從 WSL 走 `/mnt/c/...` 讀取）
- 腳本: `scripts/`，UI: `app.py`

## 一、建立環境（一次性）
在 Windows 終端機執行（會呼叫 WSL）：
```powershell
wsl -d Ubuntu-24.04 bash -lc "tr -d '\r' < /mnt/c/Users/User/Desktop/project/unlimited-ocr/scripts/wsl/00_setup_uv.sh | bash"
wsl -d Ubuntu-24.04 bash -lc "tr -d '\r' < /mnt/c/Users/User/Desktop/project/unlimited-ocr/scripts/wsl/01_setup_transformers.sh | bash"
wsl -d Ubuntu-24.04 bash -lc "tr -d '\r' < /mnt/c/Users/User/Desktop/project/unlimited-ocr/scripts/wsl/02_setup_sglang.sh | bash"
```

## 二、測試兩條路線
```powershell
# Transformers
wsl -d Ubuntu-24.04 bash -lc "~/uocr/.venv-transformers/bin/python /mnt/c/Users/User/Desktop/project/unlimited-ocr/scripts/test_transformers.py"

# SGLang（自動起 server 並對測試 PDF 推論）
wsl -d Ubuntu-24.04 bash -lc "cd /mnt/c/Users/User/Desktop/project/unlimited-ocr && ~/uocr/.venv-sglang/bin/python infer.py --pdf ./Unlimited-OCR.pdf --output_dir ./outputs/sglang --image_mode base"
```

## 三、啟動 Streamlit UI（即時 OCR）
SGLang 串流模式需先啟動 server（見 `scripts/wsl/03_start_sglang_server.sh`），再開 UI：
```powershell
# 終端 A：啟動 SGLang server（背景）
wsl -d Ubuntu-24.04 bash -lc "tr -d '\r' < /mnt/c/Users/User/Desktop/project/unlimited-ocr/scripts/wsl/03_start_sglang_server.sh | bash"

# 終端 B：啟動 UI
wsl -d Ubuntu-24.04 bash -lc "cd /mnt/c/Users/User/Desktop/project/unlimited-ocr && ~/uocr/.venv-sglang/bin/streamlit run app.py"
```
瀏覽器開 http://localhost:8501 。Transformers 批次模式不需 server。

> 最終採用的 SGLang attention backend 與權重路徑等實測結果，見決策文件「實際結果」段。
