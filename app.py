"""
Unlimited-OCR Streamlit UI — 即時查看 OCR 效果。

版面：固定一組「圖片 | 純文字」兩欄（2:3，文字欄較寬），各自固定高度，不隨內容往下延伸。
  - 標題放 sidebar 最上方；主畫面把高度留給顯示區。
  - 圖片填滿欄寬（整頁可見、不留白）；串流時左欄框逐塊長出、右欄純文字 tail 跟著掃描往下捲。
  - 上方一行指標：頁數 / 累計時間 / 總量 / 平均速度；串流中節流（~0.3s）跳動，每頁完成定版。
  - 每頁逾時保護：超過設定秒數即中止該頁（保留已辨識部分並標註），跳下一頁。
  - 回看：頁少用滑桿；頁多（≥ NAV_BUTTONS_FROM）改輸入框 + 前往/上頁/下頁。
  - 全部完成後可下載 ZIP（每頁一資料夾：疊框圖 / 原始輸出 / 純文字）。

在 .venv-sglang 內執行：streamlit run app.py
"""
import subprocess
import time

import streamlit as st

from ocr_backends import (
    DEFAULT_MODEL_DIR,
    draw_dets,
    load_display_image,
    parse_dets,
    results_to_zip,
    server_healthy,
    sglang_stream,
    strip_markup,
    to_image_paths,
    transformers_run,
)

TAIL_LINES = 40
PANE_H = 980
NAV_BUTTONS_FROM = 16
RENDER_THROTTLE = 0.15   # 串流中渲染（文字/框/指標）最小間隔（秒）；避免每 token O(n) 重繪拖慢消化串流

st.set_page_config(page_title="Unlimited-OCR", layout="wide")

st.session_state.setdefault("results", [])
st.session_state.setdefault("metrics", None)
st.session_state.setdefault("zip_bytes", None)
st.session_state.setdefault("pagenum", 1)


def _tail(text, n=TAIL_LINES):
    lines = text.splitlines()
    return "\n".join(lines[-n:]) if lines else ""


def show_metrics(ph, m):
    if not m:
        return
    ph.markdown(
        f"📄 **{m['pages']}** 頁　·　⏱ **{m['elapsed']:.1f}s**　·　"
        f"🔤 **{m['total']:,}** {m['unit']}　·　⚡ **{m['speed']:.0f}** {m['sunit']}"
        + (f"　·　⏳ 逾時 **{m['timeouts']}** 頁" if m.get("timeouts") else "")
    )


def _nav_step(delta, n):
    st.session_state.pagenum = min(n, max(1, st.session_state.pagenum + delta))


@st.dialog("📖 使用說明")
def help_dialog():
    st.markdown(
        """
### 🚀 快速開始
1. 左側選 **後端** → 選 **image_mode** → **上傳** 圖片或 PDF → 按 **開始 OCR**。
2. 左欄圖片會即時長出偵測框、右欄純文字跟著掃描往下捲。
3. 全部跑完可用 **頁碼** 回看、按 **⬇️ 下載 ZIP** 匯出。

### ⚙️ 欄位說明
- 🧠 **後端**
  - *SGLang (串流即時)*：呼叫本機 server，逐 token 串流、即時動畫（需先啟動 server）。
  - *Transformers (批次)*：每頁整段跑完才顯示（不需 server）。
  - ⚠️ 兩者一次只能用一個（16GB VRAM 不足以同時常駐）。
- 🖼️ **image_mode**
  - *gundam*：切塊細掃，**單張圖**品質較佳、token 較多。
  - *base*：整頁單視圖，**多頁/PDF** 較快。
- 🔎 **PDF DPI**：PDF 轉圖解析度，越高越清楚但越慢（一般 300）。
- 🔁 **no_repeat_ngram_size / ngram_window**：抑制重複生成的參數；數值越嚴格越能避免鬼打牆，但可能影響正常重複文字。
- 🧱 **每頁最多生成 tokens**：server 端硬上限。某些頁會無限重複生成（鬼打牆），此上限讓它**乾淨提早結束**、不會長到異常 token 數吃滿 GPU 拖垮後續頁。正常頁約 1–3k，預設 4096 有餘裕。
- 🟦 **在圖片上畫偵測框**：把 `<|det|>` 版面框畫到圖上（title 紅粗框、其餘依類別配色）。
- ⏳ **每頁逾時 (秒)**：後備防線。即使到不了 max_tokens 也會在此秒數中止跳下一頁（保留已辨識部分並標註）。

### 🧭 回看與匯出
- 📄 頁數少 → 滑桿；頁數多（≥ 16）→ 輸入框 + 前往/上頁/下頁。
- ⬇️ **下載 ZIP**：每頁一個資料夾，內含 `overlay.png`（疊框圖）、`raw.txt`（原始輸出）、`text.txt`（純文字）。
        """
    )


# ---------- sidebar ----------
with st.sidebar:
    st.title("📄 Unlimited-OCR")
    if st.button("📖 使用說明", use_container_width=True):
        help_dialog()
    st.header("設定")
    backend = st.radio("後端", ["SGLang (串流即時)", "Transformers (批次)"])
    image_mode = st.selectbox("image_mode", ["gundam", "base"],
                              help="單張建議 gundam；多頁/PDF 建議 base")
    dpi = st.slider("PDF DPI", 150, 400, 300, 50)
    ngram_size = st.number_input("no_repeat_ngram_size", 0, 100, 35)
    ngram_window = st.number_input("ngram_window", 0, 4096, 128 if image_mode == "gundam" else 1024)
    max_tokens = st.number_input("每頁最多生成 tokens", 256, 32768, 4096,
                                 help="server 端上限：讓無限重複的頁乾淨提早結束，避免長到異常 token 數吃滿 GPU、拖垮後續頁。正常頁約 1–3k。")
    page_timeout = st.number_input("每頁逾時 (秒)", 5, 600, 60,
                                   help="後備：單頁推理超過此秒數即中止並跳下一頁（max_tokens 是主要防線）")
    show_boxes = st.checkbox("在圖片上畫偵測框", value=True)
    if backend.startswith("SGLang"):
        server_url = st.text_input("SGLang server", "http://127.0.0.1:10000")
        st.caption("✅ server 線上" if server_healthy(server_url)
                   else "⚠️ 未偵測到 server（先啟動 03_start_sglang_server.sh）")
        model_dir = DEFAULT_MODEL_DIR
    else:
        server_url = "http://127.0.0.1:10000"
        model_dir = st.text_input("模型路徑", DEFAULT_MODEL_DIR)
    uploaded = st.file_uploader("上傳圖片或 PDF",
                                type=["png", "jpg", "jpeg", "webp", "bmp", "pdf"])
    run = st.button("開始 OCR", type="primary", use_container_width=True)

# ---------- 固定顯示區（建立一次） ----------
status = st.empty()
metrics_ph = st.empty()
nav = st.empty()
col_img, col_txt = st.columns([2, 3])
col_img.caption("來源圖片 / 偵測框")
col_txt.caption("純文字")
img_ph = col_img.container(height=PANE_H).empty()
txt_ph = col_txt.container(height=PANE_H).empty()


def run_ocr():
    is_sglang = backend.startswith("SGLang")
    unit, sunit = ("tokens", "tok/s") if is_sglang else ("字元", "字/s")
    images = to_image_paths(uploaded.getvalue(), uploaded.name, dpi)
    st.session_state.results = []
    st.session_state.pagenum = 1
    t_all = time.time()
    total_units = 0
    timeouts = 0

    for idx, path in enumerate(images, 1):
        status.info(f"處理中：第 {idx} / {len(images)} 頁")
        base = load_display_image(path, max_width=1000)
        img_ph.image(base, use_container_width=True)
        txt_ph.code("", language=None)
        timed_out = False

        if is_sglang:
            raw, drawn, page_units = "", 0, 0
            p_start = time.time()
            last_r = 0.0

            def render(force=False):
                """節流渲染：文字 tail + 新框 + 指標。per-token 只累加，這裡才做 O(n) 的解析/重繪。"""
                nonlocal drawn
                txt_ph.code(_tail(strip_markup(raw)), language=None)
                if show_boxes:
                    dets = parse_dets(raw)
                    if len(dets) > drawn or force:
                        try:
                            img_ph.image(draw_dets(base, dets), use_container_width=True)
                        except Exception:
                            pass   # 壞框不應中斷串流
                        drawn = len(dets)
                el = time.time() - t_all
                tot = total_units + page_units
                show_metrics(metrics_ph, {"pages": f"{idx}/{len(images)}", "total": tot,
                                          "elapsed": el, "speed": tot / el if el else 0,
                                          "unit": unit, "sunit": sunit, "timeouts": timeouts})

            try:
                for delta in sglang_stream(server_url, path, image_mode,
                                           int(ngram_window), int(ngram_size),
                                           max_tokens=int(max_tokens)):
                    raw += delta
                    page_units += 1
                    now = time.time()
                    if now - p_start > page_timeout:        # 逾時 → 中止本頁（break 會關連線、abort 請求）
                        timed_out = True
                        break
                    if now - last_r >= RENDER_THROTTLE:
                        render()
                        last_r = now
                render(force=True)   # 收尾定版
            except Exception as e:
                status.error(f"SGLang 串流失敗：{e}")
        else:
            raw, page_units = "", 0
            try:
                raw = transformers_run(path, image_mode, model_dir, timeout=int(page_timeout))
            except subprocess.TimeoutExpired:
                timed_out = True
            page_units = len(strip_markup(raw))
            txt_ph.code(_tail(strip_markup(raw)), language=None)

        clean = strip_markup(raw)
        if timed_out:
            timeouts += 1
            warn = f"\n\n⚠️ 本頁推理超時（>{int(page_timeout)}s），已中止並跳下一頁。辨識結果可能不完整或異常。"
            clean += warn
            raw += warn
            txt_ph.code(_tail(clean), language=None)

        overlay = draw_dets(base, parse_dets(raw)) if show_boxes else base
        img_ph.image(overlay, use_container_width=True)
        st.session_state.results.append({"overlay": overlay, "clean": clean, "raw": raw})

        total_units += page_units
        elapsed = time.time() - t_all
        m = {"pages": f"{idx}/{len(images)}", "total": total_units, "elapsed": elapsed,
             "speed": total_units / elapsed if elapsed > 0 else 0,
             "unit": unit, "sunit": sunit, "timeouts": timeouts}
        st.session_state.metrics = m
        show_metrics(metrics_ph, m)

    st.session_state.zip_bytes = results_to_zip(st.session_state.results)
    msg = f"完成，共 {len(images)} 頁。"
    if timeouts:
        msg += f"（其中 {timeouts} 頁逾時）"
    status.success(msg + " 可用下方控制回看、下載 ZIP。")
    st.rerun()


def navigator(n):
    if st.session_state.pagenum > n:
        st.session_state.pagenum = n
    with nav.container():
        top = st.columns([3, 1])
        with top[0]:
            if n < NAV_BUTTONS_FROM:
                st.session_state.pagenum = st.slider("頁碼", 1, n, min(st.session_state.pagenum, n))
            else:
                c = st.columns([2, 1, 1, 1])
                c[0].number_input("頁碼", 1, n, key="pagenum", label_visibility="collapsed")
                c[1].button("前往", use_container_width=True)
                c[2].button("◀ 上頁", use_container_width=True, on_click=_nav_step, args=(-1, n))
                c[3].button("下頁 ▶", use_container_width=True, on_click=_nav_step, args=(1, n))
        with top[1]:
            if st.session_state.zip_bytes:
                st.download_button("⬇️ 下載 ZIP", data=st.session_state.zip_bytes,
                                   file_name="ocr_results.zip", mime="application/zip",
                                   use_container_width=True)
    return st.session_state.pagenum


def review():
    show_metrics(metrics_ph, st.session_state.metrics)
    results = st.session_state.results
    page = navigator(len(results))
    r = results[page - 1]
    img_ph.image(r["overlay"], use_container_width=True)
    txt_ph.code(r["clean"], language=None)
    with st.expander("原始輸出（含 <|det|>/<|ref|> 標記）"):
        st.code(r["raw"])


# ---------- 流程 ----------
if run and uploaded:
    run_ocr()
elif run and not uploaded:
    status.warning("請先在左側上傳圖片或 PDF。")
elif st.session_state.results:
    review()
else:
    img_ph.info("上傳圖片或 PDF，按左側「開始 OCR」。")
    txt_ph.write("純文字會顯示在這裡。")
