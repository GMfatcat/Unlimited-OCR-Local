"""
Unlimited-OCR Streamlit UI — 即時查看 OCR 效果。

兩種後端（一次只用一個；16GB VRAM 不足以同時常駐）：
  1. SGLang (streaming)：呼叫本機 SGLang server 的 OpenAI 相容 API，token 逐字串流，即時顯示。
  2. Transformers (batch)：以子行程在 .venv-transformers 跑 model.infer，跑完一次顯示。

在 .venv-sglang 內執行：
  streamlit run app.py
"""
import os

import streamlit as st

from ocr_backends import (
    DEFAULT_MODEL_DIR,
    server_healthy,
    sglang_stream,
    to_image_paths,
    transformers_run,
)

st.set_page_config(page_title="Unlimited-OCR", layout="wide")
st.title("📄 Unlimited-OCR — 即時 OCR 檢視")

with st.sidebar:
    st.header("設定")
    backend = st.radio("後端", ["SGLang (串流即時)", "Transformers (批次)"])
    image_mode = st.selectbox("image_mode", ["gundam", "base"],
                              help="單張建議 gundam；多頁/PDF 建議 base")
    dpi = st.slider("PDF DPI", 150, 400, 300, 50)
    ngram_size = st.number_input("no_repeat_ngram_size", 0, 100, 35)
    ngram_window = st.number_input("ngram_window", 0, 4096, 128 if image_mode == "gundam" else 1024)
    if backend.startswith("SGLang"):
        server_url = st.text_input("SGLang server", "http://127.0.0.1:10000")
        st.caption("✅ server 線上" if server_healthy(server_url) else "⚠️ 未偵測到 server（先啟動 03_start_sglang_server.sh）")
    else:
        model_dir = st.text_input("模型路徑", DEFAULT_MODEL_DIR)

uploaded = st.file_uploader("上傳圖片或 PDF", type=["png", "jpg", "jpeg", "webp", "bmp", "pdf"])

if uploaded and st.button("開始 OCR", type="primary"):
    images = to_image_paths(uploaded.getvalue(), uploaded.name, dpi)
    st.info(f"共 {len(images)} 頁 / 張")

    for idx, img in enumerate(images, 1):
        st.divider()
        st.subheader(f"第 {idx} 頁 / 張")
        col_img, col_txt = st.columns([1, 1])
        with col_img:
            st.image(img, use_container_width=True)
        with col_txt:
            if backend.startswith("SGLang"):
                placeholder = st.empty()
                acc = ""
                try:
                    for delta in sglang_stream(server_url, img, image_mode,
                                               int(ngram_window), int(ngram_size)):
                        acc += delta
                        placeholder.markdown(acc)
                except Exception as e:
                    st.error(f"SGLang 串流失敗：{e}")
                with st.expander("原始輸出"):
                    st.code(acc)
            else:
                with st.spinner("Transformers 推論中…"):
                    text = transformers_run(img, image_mode, model_dir)
                st.markdown(text)
                with st.expander("原始輸出"):
                    st.code(text)
