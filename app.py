"""
Unlimited-OCR Streamlit UI — 即時查看 OCR 效果。

版面：固定一組「圖片 | 純文字」兩欄（2:3，文字欄較寬），各自固定高度，不隨內容往下延伸。
  - 圖片填滿欄寬（整頁可見、不留白）；串流時左欄框逐塊長出、右欄純文字 tail 跟著掃描往下捲。
  - 頁與頁之間是「替換」而非堆疊，不用手動往下滑就看得到即時動態。
  - 上方一排指標：頁數 / 累計時間 / 總量 / 平均速度，每頁完成後更新。
  - 跑完多頁後：頁碼選擇器可在同一組固定欄位回看任一頁。

兩種後端（一次只用一個；16GB VRAM 不足以同時常駐）：
  1. SGLang (streaming)：本機 SGLang server，token 逐字串流、即時動畫（速度以實際 token 計）。
  2. Transformers (batch)：子行程在 .venv-transformers 跑 model.infer，每頁跑完即原地替換（以字元計）。

在 .venv-sglang 內執行：streamlit run app.py
"""
import time

import streamlit as st

from ocr_backends import (
    DEFAULT_MODEL_DIR,
    draw_dets,
    load_display_image,
    parse_dets,
    server_healthy,
    sglang_stream,
    strip_markup,
    to_image_paths,
    transformers_run,
)

TAIL_LINES = 40      # 串流時終端機面板顯示的最新行數
PANE_H = 980         # 兩欄固定高度（px）

st.set_page_config(page_title="Unlimited-OCR", layout="wide")
st.title("📄 Unlimited-OCR — 即時 OCR 檢視")

if "results" not in st.session_state:
    st.session_state.results = []     # 每頁 {overlay: PIL, clean: str, raw: str}
if "metrics" not in st.session_state:
    st.session_state.metrics = None   # {pages,total,elapsed,speed,unit,sunit}


def _tail(text, n=TAIL_LINES):
    lines = text.splitlines()
    return "\n".join(lines[-n:]) if lines else ""


def show_metrics(ph, m):
    if not m:
        return
    with ph.container():
        c = st.columns(4)
        c[0].metric("頁數", m["pages"])
        c[1].metric("累計時間", f"{m['elapsed']:.1f}s")
        c[2].metric(f"總{m['unit']}", f"{m['total']:,}")
        c[3].metric("平均速度", f"{m['speed']:.0f} {m['sunit']}")


# ---------- sidebar ----------
with st.sidebar:
    st.header("設定")
    backend = st.radio("後端", ["SGLang (串流即時)", "Transformers (批次)"])
    image_mode = st.selectbox("image_mode", ["gundam", "base"],
                              help="單張建議 gundam；多頁/PDF 建議 base")
    dpi = st.slider("PDF DPI", 150, 400, 300, 50)
    ngram_size = st.number_input("no_repeat_ngram_size", 0, 100, 35)
    ngram_window = st.number_input("ngram_window", 0, 4096, 128 if image_mode == "gundam" else 1024)
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
col_img, col_txt = st.columns([2, 3])     # 文字欄較寬
col_img.caption("來源圖片 / 偵測框")
col_txt.caption("純文字")
img_ph = col_img.container(height=PANE_H).empty()
txt_ph = col_txt.container(height=PANE_H).empty()


def run_ocr():
    """處理上傳檔案，逐頁在固定欄位原地更新；每頁完成後更新指標；結果存入 session_state。"""
    is_sglang = backend.startswith("SGLang")
    unit, sunit = ("tokens", "tok/s") if is_sglang else ("字元", "字/s")
    images = to_image_paths(uploaded.getvalue(), uploaded.name, dpi)
    st.session_state.results = []
    t_all = time.time()
    total_units = 0

    for idx, path in enumerate(images, 1):
        status.info(f"處理中：第 {idx} / {len(images)} 頁")
        base = load_display_image(path, max_width=1000)
        img_ph.image(base, use_container_width=True)
        txt_ph.code("", language=None)

        if is_sglang:
            raw, drawn, page_units = "", 0, 0
            try:
                for delta in sglang_stream(server_url, path, image_mode,
                                           int(ngram_window), int(ngram_size)):
                    raw += delta
                    page_units += 1
                    txt_ph.code(_tail(strip_markup(raw)), language=None)
                    if show_boxes:
                        dets = parse_dets(raw)
                        if len(dets) > drawn:
                            img_ph.image(draw_dets(base, dets), use_container_width=True)
                            drawn = len(dets)
            except Exception as e:
                status.error(f"SGLang 串流失敗：{e}")
        else:
            raw = transformers_run(path, image_mode, model_dir)
            txt_ph.code(_tail(strip_markup(raw)), language=None)
            page_units = len(strip_markup(raw))

        clean = strip_markup(raw)
        overlay = draw_dets(base, parse_dets(raw)) if show_boxes else base
        img_ph.image(overlay, use_container_width=True)
        st.session_state.results.append({"overlay": overlay, "clean": clean, "raw": raw})

        total_units += page_units
        elapsed = time.time() - t_all
        m = {"pages": f"{idx}/{len(images)}", "total": total_units, "elapsed": elapsed,
             "speed": total_units / elapsed if elapsed > 0 else 0, "unit": unit, "sunit": sunit}
        st.session_state.metrics = m
        show_metrics(metrics_ph, m)

    status.success(f"完成，共 {len(images)} 頁。可用下方頁碼回看。")
    st.rerun()   # 進入回看模式


def review():
    """回看模式：指標 + 頁碼選擇器 + 固定欄位顯示選定頁。"""
    show_metrics(metrics_ph, st.session_state.metrics)
    results = st.session_state.results
    n = len(results)
    page = nav.slider("頁碼", 1, n, 1) if n > 1 else 1
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
