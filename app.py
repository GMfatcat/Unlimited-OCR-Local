"""
Unlimited-OCR Streamlit UI — 即時查看 OCR 效果。

呈現方式：
  - 左：來源圖片，串流時「掃到一個 <|det|> 框就即時畫上去」。
  - 右：純文字（去掉 <|det|>/<|ref|> 等標記），固定高度終端機 tail，永遠顯示最新、跟著掃描走。

兩種後端（一次只用一個；16GB VRAM 不足以同時常駐）：
  1. SGLang (streaming)：呼叫本機 SGLang server，token 逐字串流、即時動畫。
  2. Transformers (batch)：以子行程在 .venv-transformers 跑 model.infer，跑完一次畫好。

在 .venv-sglang 內執行：streamlit run app.py
"""
import os

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

TAIL_LINES = 30  # 終端機面板顯示的最新行數

st.set_page_config(page_title="Unlimited-OCR", layout="wide")
st.title("📄 Unlimited-OCR — 即時 OCR 檢視")


def _tail(clean_text, n=TAIL_LINES):
    lines = clean_text.splitlines()
    return "\n".join(lines[-n:]) if lines else ""


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
    else:
        model_dir = st.text_input("模型路徑", DEFAULT_MODEL_DIR)

uploaded = st.file_uploader("上傳圖片或 PDF", type=["png", "jpg", "jpeg", "webp", "bmp", "pdf"])

if uploaded and st.button("開始 OCR", type="primary"):
    images = to_image_paths(uploaded.getvalue(), uploaded.name, dpi)
    st.info(f"共 {len(images)} 頁 / 張")

    for idx, img_path in enumerate(images, 1):
        st.divider()
        st.subheader(f"第 {idx} 頁 / 張")
        base_img = load_display_image(img_path)
        col_img, col_txt = st.columns([1, 1])
        img_placeholder = col_img.empty()
        img_placeholder.image(base_img, use_container_width=True)

        with col_txt:
            st.caption("純文字（即時，跟著掃描）")
            txt_box = st.container(height=460)
            txt_placeholder = txt_box.empty()

        if backend.startswith("SGLang"):
            raw = ""
            drawn = 0
            try:
                for delta in sglang_stream(server_url, img_path, image_mode,
                                           int(ngram_window), int(ngram_size)):
                    raw += delta
                    # 右：純文字 tail
                    txt_placeholder.code(_tail(strip_markup(raw)), language=None)
                    # 左：有新框才重畫（避免每 token 重畫）
                    if show_boxes:
                        dets = parse_dets(raw)
                        if len(dets) > drawn:
                            img_placeholder.image(draw_dets(base_img, dets),
                                                  use_container_width=True)
                            drawn = len(dets)
            except Exception as e:
                st.error(f"SGLang 串流失敗：{e}")
            clean = strip_markup(raw)
            txt_placeholder.code(_tail(clean), language=None)
            with st.expander("完整純文字 / 原始輸出"):
                st.text(clean)
                st.caption("— 原始（含標記）—")
                st.code(raw)
        else:
            with st.spinner("Transformers 推論中…"):
                raw = transformers_run(img_path, image_mode, model_dir)
            clean = strip_markup(raw)
            if show_boxes:
                img_placeholder.image(draw_dets(base_img, parse_dets(raw)),
                                      use_container_width=True)
            txt_placeholder.code(_tail(clean), language=None)
            with st.expander("完整純文字 / 原始輸出"):
                st.text(clean)
                st.caption("— 原始（含標記）—")
                st.code(raw)
