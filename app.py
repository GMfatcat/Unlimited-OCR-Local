"""
Unlimited-OCR Streamlit UI — 即時查看 OCR 效果。

兩種後端：
  1. SGLang (streaming)：呼叫本機 SGLang server 的 OpenAI 相容 API，token 逐字串流，即時顯示。
  2. Transformers (batch)：以子行程在 .venv-transformers 跑 model.infer，跑完一次顯示。

因 16GB VRAM 無法同時常駐兩個後端，一次只用一個。

在 .venv-sglang 內執行：
  streamlit run app.py
"""
import base64
import json
import os
import subprocess
import tempfile

import fitz  # PyMuPDF
import requests
import streamlit as st

REPO = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_DIR = os.path.join(REPO, "unlimited-ocr-hf")
TRANSFORMERS_PY = os.path.expanduser("~/uocr/.venv-transformers/bin/python")
OCR_ONCE = os.path.join(REPO, "scripts", "ocr_once.py")

st.set_page_config(page_title="Unlimited-OCR", layout="wide")
st.title("📄 Unlimited-OCR — 即時 OCR 檢視")


# ---------- 共用：PDF / 圖片 轉成 PNG 路徑 ----------
def to_image_paths(uploaded, dpi):
    suffix = os.path.splitext(uploaded.name)[1].lower()
    tmp_dir = tempfile.mkdtemp(prefix="uocr_ui_")
    raw = os.path.join(tmp_dir, uploaded.name)
    with open(raw, "wb") as f:
        f.write(uploaded.getbuffer())
    if suffix == ".pdf":
        doc = fitz.open(raw)
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        paths = []
        for i, page in enumerate(doc):
            out = os.path.join(tmp_dir, f"page_{i + 1:04d}.png")
            page.get_pixmap(matrix=mat).save(out)
            paths.append(out)
        doc.close()
        return paths
    return [raw]


# ---------- SGLang 串流 ----------
def sglang_stream(server_url, image_path, image_mode, ngram_window, ngram_size):
    from sglang.srt.sampling.custom_logit_processor import (
        DeepseekOCRNoRepeatNGramLogitProcessor,
    )

    ext = os.path.splitext(image_path)[1].lower()
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else f"image/{ext.lstrip('.')}"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "model": "Unlimited-OCR",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "document parsing."},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ],
        }],
        "temperature": 0,
        "skip_special_tokens": False,
        "images_config": {"image_mode": image_mode},
        "stream": True,
    }
    if ngram_size > 0 and ngram_window > 0:
        payload["custom_logit_processor"] = DeepseekOCRNoRepeatNGramLogitProcessor.to_str()
        payload["custom_params"] = {"ngram_size": ngram_size, "window_size": ngram_window}

    session = requests.Session()
    session.trust_env = False
    resp = session.post(
        f"{server_url}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=1200,
        stream=True,
    )
    resp.raise_for_status()
    for raw in resp.iter_lines(decode_unicode=True):
        if not raw or not raw.startswith("data: "):
            continue
        data = raw[len("data: "):]
        if data == "[DONE]":
            break
        try:
            event = json.loads(data)
            delta = event["choices"][0].get("delta", {}).get("content", "")
        except (json.JSONDecodeError, KeyError):
            continue
        if delta:
            yield delta


def server_healthy(url):
    try:
        return requests.get(f"{url}/health", timeout=3).status_code == 200
    except requests.RequestException:
        return False


# ---------- Transformers 子行程 ----------
def transformers_run(image_path, image_mode, model_dir):
    env = dict(os.environ, MODEL_DIR=model_dir)
    proc = subprocess.run(
        [TRANSFORMERS_PY, OCR_ONCE, "--image", image_path, "--mode", image_mode],
        capture_output=True, text=True, env=env, timeout=1800,
    )
    out = proc.stdout
    start, end = "<<<OCR_RESULT_START>>>", "<<<OCR_RESULT_END>>>"
    if start in out and end in out:
        return out.split(start, 1)[1].split(end, 1)[0].strip()
    return f"(無法解析輸出)\nstdout:\n{out[-2000:]}\nstderr:\n{proc.stderr[-2000:]}"


# ---------- Sidebar ----------
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
        st.caption("✅ 線上" if server_healthy(server_url) else "⚠️ 未偵測到 server")
    else:
        model_dir = st.text_input("模型路徑", DEFAULT_MODEL_DIR)


uploaded = st.file_uploader("上傳圖片或 PDF", type=["png", "jpg", "jpeg", "webp", "bmp", "pdf"])

if uploaded and st.button("開始 OCR", type="primary"):
    images = to_image_paths(uploaded, dpi)
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
                    for delta in sglang_stream(server_url, img, image_mode, int(ngram_window), int(ngram_size)):
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
