"""把單頁/單文件丟 SGLang server 並存產出（品質軌與穩定軌共用）。"""
import os
import subprocess
import time
from dataclasses import dataclass

import fitz

from bench.corpus_io import parse_pages
from ocr_backends import draw_dets, load_display_image, parse_dets, sglang_stream, strip_markup


def gpu_mem_mib():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    return int(out.stdout.strip().splitlines()[0])


def render_pages(pdf_path, pages, dpi=300):
    with fitz.open(pdf_path) as doc:
        idxs = parse_pages(pages, doc.page_count)
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        out_dir = os.path.join(os.path.dirname(pdf_path), "_pages_" + os.path.basename(pdf_path))
        os.makedirs(out_dir, exist_ok=True)
        paths = []
        for i in idxs:
            p = os.path.join(out_dir, f"page_{i + 1:04d}.png")
            doc[i].get_pixmap(matrix=mat).save(p)
            paths.append(p)
    return paths


@dataclass
class PageResult:
    idx: int
    tokens: int
    decode_s: float
    clean: str
    raw: str
    overlay_path: str
    flag: str


def run_page(image_path, server_url, image_mode, max_tokens, page_timeout,
             ngram_window, ngram_size, out_dir, idx):
    raw, tokens, timed_out = "", 0, False
    t0 = time.time()
    gen = sglang_stream(server_url, image_path, image_mode, ngram_window, ngram_size,
                        max_tokens=max_tokens)
    for delta in gen:
        raw += delta
        tokens += 1
        if time.time() - t0 > page_timeout:
            timed_out = True
            gen.close()
            break
    decode_s = time.time() - t0
    capped = (not timed_out) and tokens >= max_tokens
    flag = "timeout" if timed_out else ("capped" if capped else "")

    clean = strip_markup(raw)
    note = {"timeout": f"\n\n[TIMEOUT >{page_timeout}s]",
            "capped": f"\n\n[CAPPED max_tokens={max_tokens}]"}.get(flag)
    if note:
        clean += note
        raw += note

    os.makedirs(out_dir, exist_ok=True)
    base = load_display_image(image_path, max_width=1000)
    overlay = draw_dets(base, parse_dets(raw))
    overlay_path = os.path.join(out_dir, f"overlay_p{idx:04d}.png")
    overlay.save(overlay_path)
    return PageResult(idx, tokens, decode_s, clean, raw, overlay_path, flag)
