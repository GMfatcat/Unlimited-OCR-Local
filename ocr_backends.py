"""
Unlimited-OCR 後端邏輯（與 UI 分離，方便測試）。

- to_image_paths：圖片/PDF -> PNG 路徑清單
- server_healthy：檢查 SGLang server
- sglang_stream：呼叫 SGLang OpenAI 串流 API，yield 每個 token delta（即時）
- transformers_run：以子行程在 .venv-transformers 跑 model.infer（批次）
"""
import ast
import base64
import io
import json
import os
import re
import subprocess
import tempfile
import zipfile

import fitz  # PyMuPDF
import requests
from PIL import Image, ImageDraw, ImageFont

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


def sglang_stream(server_url, image_path, image_mode, ngram_window, ngram_size,
                  prompt="document parsing.", max_tokens=None):
    """對 SGLang server 串流請求，逐 token yield delta 文字。
    max_tokens：server 端上限，讓無限重複的頁面乾淨提早結束（避免長到異常 token 數拖垮/吃滿 GPU）。"""
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
    if max_tokens:
        payload["max_tokens"] = int(max_tokens)
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
    # 用 try/finally：消費端若提早 break（例如逾時），產生器關閉時會走到 finally，
    # 關閉連線 → SGLang 偵測 client 斷線 → abort 該請求，釋放 GPU 給下一頁。
    try:
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
    finally:
        resp.close()


# ---------- det 解析 / 純文字 / 畫框 ----------
# 模型輸出座標為 0–999 正規化；label 與 box 都包在 <|det|>…<|/det|> 內，
# 例如：<|det|>header [123, 29, 322, 75]<|/det|>Baidu百度
_DET_RE = re.compile(r"<\|det\|>\s*([A-Za-z_][\w-]*)\s*(\[[0-9,\s\.\[\]]+\])\s*<\|/det\|>")
# 另一變體：<|ref|>label<|/ref|><|det|>[box]<|/det|>
_REFDET_RE = re.compile(r"<\|ref\|>\s*([^<]+?)\s*<\|/ref\|>\s*<\|det\|>\s*(\[[0-9,\s\.\[\]]+\])\s*<\|/det\|>")

# 依 label 給固定配色（比模型原本的隨機色乾淨）。RGB。
LABEL_COLORS = {
    "title": (220, 40, 40),
    "header": (230, 140, 20),
    "text": (40, 110, 220),
    "image": (30, 170, 90),
    "image_caption": (20, 160, 160),
    "table": (150, 60, 200),
    "table_caption": (120, 80, 200),
    "list": (160, 100, 40),
    "figure": (30, 170, 90),
    "formula": (200, 60, 160),
    "page_number": (130, 130, 130),
    "footer": (130, 130, 130),
}
_DEFAULT_COLOR = (200, 60, 160)


def _boxes_from_str(box_str):
    """把 '[x1,y1,x2,y2]' 或 '[[..],[..]]' 解析成 [(x1,y1,x2,y2), ...]（0–999）。"""
    try:
        val = ast.literal_eval(box_str)
    except Exception:
        return []
    if not val:
        return []
    if isinstance(val[0], (int, float)):
        val = [val]
    out = []
    for b in val:
        if isinstance(b, (list, tuple)) and len(b) >= 4:
            out.append(tuple(float(x) for x in b[:4]))
    return out


def parse_dets(text):
    """從（可能仍在串流中的）文字解析出已完整的偵測框。
    回傳 [(label, [(x1,y1,x2,y2,)...]), ...]，座標仍是 0–999。"""
    dets = []
    for label, box_str in _DET_RE.findall(text):
        boxes = _boxes_from_str(box_str)
        if boxes:
            dets.append((label.strip(), boxes))
    for label, box_str in _REFDET_RE.findall(text):
        boxes = _boxes_from_str(box_str)
        if boxes:
            dets.append((label.strip(), boxes))
    return dets


def strip_markup(text):
    """移除所有 <|det|>…<|/det|>、<|ref|>…<|/ref|>、<PAGE>、其餘 <|…|> 特殊 token，
    留下乾淨純文字。也會清掉串流尾端尚未閉合的標記碎片。"""
    t = _REFDET_RE.sub("", text)
    t = re.sub(r"<\|ref\|>.*?<\|/ref\|>", "", t, flags=re.DOTALL)
    t = re.sub(r"<\|det\|>.*?<\|/det\|>", "", t, flags=re.DOTALL)
    # 尾端未閉合的 <|det| / <|ref| 片段（串流中）
    t = re.sub(r"<\|(?:det|ref)\|>.*\Z", "", t, flags=re.DOTALL)
    t = t.replace("<PAGE>", "\n")
    t = re.sub(r"<\|[^|]*\|>", "", t)        # 其餘特殊 token
    t = re.sub(r"<\|[^|]*\Z", "", t)         # 尾端未閉合的 <|…
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def load_display_image(image_path, max_width=900, max_height=None):
    """開圖並等比縮放，使其同時不超過 max_width / max_height（給 UI 顯示與畫框用）。
    框座標是相對影像尺寸（0–999），所以縮放後再畫框仍然正確。"""
    img = Image.open(image_path).convert("RGB")
    scale = 1.0
    if max_width:
        scale = min(scale, max_width / img.width)
    if max_height:
        scale = min(scale, max_height / img.height)
    if scale < 1.0:
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))
    return img


def draw_dets(base_img, dets, font=None):
    """在 base_img（PIL RGB）上畫出 dets 的框與 label，回傳新圖。座標 0–999 → 像素。"""
    img = base_img.copy()
    draw = ImageDraw.Draw(img, "RGBA")
    W, H = img.size
    if font is None:
        font = ImageFont.load_default()
    for label, boxes in dets:
        color = LABEL_COLORS.get(label, _DEFAULT_COLOR)
        width = 4 if label == "title" else 2
        for (x1, y1, x2, y2) in boxes:
            px1, px2 = sorted((int(x1 / 999 * W), int(x2 / 999 * W)))  # 容錯：座標可能反向
            py1, py2 = sorted((int(y1 / 999 * H), int(y2 / 999 * H)))
            px1, px2 = max(0, px1), min(W - 1, px2)
            py1, py2 = max(0, py1), min(H - 1, py2)
            if px2 <= px1 or py2 <= py1:   # 退化/零面積框，跳過
                continue
            draw.rectangle([px1, py1, px2, py2], outline=color, width=width)
            draw.rectangle([px1, py1, px2, py2], fill=color + (28,))
            # label 小標
            ty = max(0, py1 - 13)
            tb = draw.textbbox((0, 0), label, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            draw.rectangle([px1, ty, px1 + tw + 4, ty + th + 2], fill=(255, 255, 255, 220))
            draw.text((px1 + 2, ty), label, font=font, fill=color)
    return img


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


def results_to_zip(results):
    """把每頁結果打包成 ZIP（bytes）。每頁一個資料夾，含疊框圖、原始輸出、純文字。
    results: [{overlay: PIL, raw: str, clean: str}, ...]"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for i, r in enumerate(results, 1):
            folder = f"page_{i:02d}"
            png = io.BytesIO()
            r["overlay"].save(png, format="PNG")
            z.writestr(f"{folder}/overlay.png", png.getvalue())
            z.writestr(f"{folder}/raw.txt", r.get("raw", ""))
            z.writestr(f"{folder}/text.txt", r.get("clean", ""))
    return buf.getvalue()
