"""自動錄製 Streamlit UI 的即時 OCR 畫面 → .webm 影片。

前置（兩個都要先跑起來）：
  1. SGLang server：scripts/wsl/03_start_sglang_server.sh（ATTN_BACKEND=triton）
  2. UI：~/uocr/.venv-sglang/bin/streamlit run app.py   → http://localhost:8501

執行（用 playwright 專用 venv）：
  ~/uocr/.venv-playwright/bin/python scripts/record_demo.py [pdf] [out_dir]
  - pdf     ：要 OCR 的檔案，預設 assets/demo_3pages.pdf（3 頁，demo 用）
  - out_dir ：輸出資料夾，預設 outputs/demo

環境變數：
  UI_URL  （預設 http://localhost:8501）
  HEADED=1（顯示瀏覽器視窗，需 WSLg；預設 headless）

輸出為 .webm。轉 mp4/gif：
  ffmpeg -i demo.webm demo.mp4
  ffmpeg -i demo.webm -vf "fps=12,scale=900:-1:flags=lanczos" demo.gif
"""
import os
import sys
import time

from playwright.sync_api import sync_playwright

UI_URL = os.environ.get("UI_URL", "http://localhost:8501")
HEADED = os.environ.get("HEADED") == "1"
PDF = sys.argv[1] if len(sys.argv) > 1 else "assets/demo_3pages.pdf"
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "outputs/demo"
W, H = 1500, 870
DONE_TIMEOUT_MS = 600_000   # 等串流跑完（出現「下載 ZIP」）最長 10 分鐘


def main():
    pdf = os.path.abspath(PDF)
    if not os.path.exists(pdf):
        sys.exit(f"找不到 PDF：{pdf}")
    os.makedirs(OUT_DIR, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not HEADED)
        ctx = browser.new_context(
            viewport={"width": W, "height": H},
            record_video_dir=OUT_DIR,
            record_video_size={"width": W, "height": H},
        )
        page = ctx.new_page()

        print(f"[1/5] 開啟 UI：{UI_URL}")
        try:
            page.goto(UI_URL, wait_until="domcontentloaded", timeout=60_000)
        except Exception:
            ctx.close()
            browser.close()
            sys.exit(f"連不到 UI（{UI_URL}）。請先啟動 SGLang server 與 `streamlit run app.py`。")

        # 等 Streamlit 前端就緒：「開始 OCR」按鈕出現
        start_btn = page.get_by_role("button", name="開始 OCR")
        start_btn.wait_for(state="visible", timeout=60_000)
        time.sleep(1.0)   # 多錄一點初始畫面

        print(f"[2/5] 上傳 PDF：{pdf}")
        page.set_input_files('input[type="file"]', pdf)
        # 等上傳註冊（Streamlit 會 rerun 並顯示檔名）
        try:
            page.get_by_text(os.path.basename(pdf), exact=False).wait_for(timeout=30_000)
        except Exception:
            time.sleep(2.0)
        time.sleep(1.0)

        print("[3/5] 按「開始 OCR」，開始串流錄製…")
        page.get_by_role("button", name="開始 OCR").click()

        print("[4/5] 錄到串流完成（等「下載 ZIP」出現）…")
        try:
            page.get_by_role("button", name="下載 ZIP").wait_for(timeout=DONE_TIMEOUT_MS)
            print("      ✔ 偵測到完成。")
        except Exception:
            print("      ⚠ 未在時限內偵測到完成，仍儲存目前錄影。")
        time.sleep(2.0)   # 收尾多錄一點

        video_src = page.video.path()
        print("[5/5] 收尾，flush 影片…")
        ctx.close()        # 必須關閉 context 才會寫完影片
        browser.close()

    ts = time.strftime("%Y%m%d_%H%M%S")
    final = os.path.join(OUT_DIR, f"ui_demo_{ts}.webm")
    try:
        os.replace(video_src, final)
    except OSError:
        final = video_src
    print(f"\n✅ 影片已存：{final}")
    print("   轉 mp4 ：ffmpeg -i '%s' demo.mp4" % final)
    print("   轉 gif ：ffmpeg -i '%s' -vf \"fps=12,scale=900:-1:flags=lanczos\" demo.gif" % final)


if __name__ == "__main__":
    main()
