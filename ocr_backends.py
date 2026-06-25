"""
Unlimited-OCR 後端邏輯（與 UI 分離，方便測試）。

- to_image_paths：圖片/PDF -> PNG 路徑清單
- server_healthy：檢查 SGLang server
- sglang_stream：呼叫 SGLang OpenAI 串流 API，yield 每個 token delta（即時）
- transformers_run：以子行程在 .venv-transformers 跑 model.infer（批次）
"""
import base64
import json
import os
import subprocess
import tempfile

import fitz  # PyMuPDF
import requests

REPO = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_DIR = os.path.join(REPO, "unlimited-ocr-hf")
TRANSFORMERS_PY = os.path.expanduser("~/uocr/.venv-transformers/bin/python")
OCR_ONCE = os.path.join(REPO, "scripts", "ocr_once.py")


def to_image_paths(file_bytes, filename, dpi=300):
    """把上傳的位元組（圖片或 PDF）轉成 PNG 路徑清單。"""
    suffix = os.path.splitext(filename)[1].lower()
    tmp_dir = tempfile.mkdtemp(prefix="uocr_ui_")
    raw = os.path.join(tmp_dir, filename)
    with open(raw, "wb") as f:
        f.write(file_bytes)
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


def server_healthy(url):
    try:
        return requests.get(f"{url}/health", timeout=3).status_code == 200
    except requests.RequestException:
        return False


def _encode_image(image_path):
    ext = os.path.splitext(image_path)[1].lower()
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else f"image/{ext.lstrip('.')}"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def sglang_stream(server_url, image_path, image_mode, ngram_window, ngram_size, prompt="document parsing."):
    """對 SGLang server 串流請求，逐 token yield delta 文字。"""
    from sglang.srt.sampling.custom_logit_processor import (
        DeepseekOCRNoRepeatNGramLogitProcessor,
    )

    payload = {
        "model": "Unlimited-OCR",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": _encode_image(image_path)}},
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


def transformers_run(image_path, image_mode, model_dir=DEFAULT_MODEL_DIR, timeout=1800):
    """以子行程在 .venv-transformers 跑單張 OCR，回傳 markdown 文字。"""
    env = dict(os.environ, MODEL_DIR=model_dir)
    proc = subprocess.run(
        [TRANSFORMERS_PY, OCR_ONCE, "--image", image_path, "--mode", image_mode],
        capture_output=True, text=True, env=env, timeout=timeout,
    )
    out = proc.stdout
    start, end = "<<<OCR_RESULT_START>>>", "<<<OCR_RESULT_END>>>"
    if start in out and end in out:
        return out.split(start, 1)[1].split(end, 1)[0].strip()
    return f"(無法解析輸出)\nstdout tail:\n{out[-2000:]}\nstderr tail:\n{proc.stderr[-2000:]}"
