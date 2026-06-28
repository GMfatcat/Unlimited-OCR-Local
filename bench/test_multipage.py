"""Multipage 最小驗證（測試 1）：逐頁 base vs 多頁一次 base。
比較：文字一致性、總耗時/吞吐(tok/s)、峰值 VRAM。多圖請求若不被 server 支援會明確報錯。"""
import json
import os
import sys
import threading
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ocr_backends import _encode_image, strip_markup  # noqa: E402
from bench.runner import render_pages, gpu_mem_mib  # noqa: E402
from bench.corpus_io import pdf_pseudo_gt  # noqa: E402
from bench import metrics  # noqa: E402
from sglang.srt.sampling.custom_logit_processor import (  # noqa: E402
    DeepseekOCRNoRepeatNGramLogitProcessor,
)

SERVER = "http://127.0.0.1:10000"
# CLI: test_multipage.py [pages] [pdf] [mp_max_tokens]
PAGES = sys.argv[1] if len(sys.argv) > 1 else "3-6"
PDF = sys.argv[2] if len(sys.argv) > 2 else "bench/corpus/academic/unlimited-ocr.pdf"
MP_MAX = int(sys.argv[3]) if len(sys.argv) > 3 else 12000
OUT = "bench/reports/_multipage"
# base 模式標準防重複參數（同 app.py / infer.py）
NGRAM_SIZE = 35
NGRAM_WINDOW = 1024


class Peak(threading.Thread):
    """背景每 0.2s 取一次 VRAM，記錄峰值。"""

    def __init__(self):
        super().__init__(daemon=True)
        self.peak = 0
        self.on = True

    def run(self):
        while self.on:
            self.peak = max(self.peak, gpu_mem_mib())
            time.sleep(0.2)

    def stop(self):
        self.on = False
        self.join()
        return self.peak


def stream(payload):
    """送串流請求，回傳 (text, n_tokens)。"""
    s = requests.Session()
    s.trust_env = False
    r = s.post(f"{SERVER}/v1/chat/completions", headers={"Content-Type": "application/json"},
               data=json.dumps(payload), timeout=3600, stream=True)
    r.raise_for_status()
    buf, n = [], 0
    try:
        for raw in r.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data: "):
                continue
            d = raw[len("data: "):]
            if d == "[DONE]":
                break
            try:
                ev = json.loads(d)
                delta = ev["choices"][0].get("delta", {}).get("content", "")
            except (json.JSONDecodeError, KeyError):
                continue
            if delta:
                buf.append(delta)
                n += 1
    finally:
        r.close()
    return "".join(buf), n


def payload_for(content, max_tokens):
    return {"model": "Unlimited-OCR", "messages": [{"role": "user", "content": content}],
            "temperature": 0, "skip_special_tokens": False,
            "images_config": {"image_mode": "base"}, "stream": True, "max_tokens": max_tokens,
            "custom_logit_processor": DeepseekOCRNoRepeatNGramLogitProcessor.to_str(),
            "custom_params": {"ngram_size": NGRAM_SIZE, "window_size": NGRAM_WINDOW}}


def page_by_page(imgs, max_tokens=8192):
    raws, n_tot, t0 = [], 0, time.time()
    for p in imgs:
        content = [{"type": "text", "text": "document parsing."},
                   {"type": "image_url", "image_url": {"url": _encode_image(p)}}]
        txt, n = stream(payload_for(content, max_tokens))
        raws.append(txt)
        n_tot += n
    return "\n".join(raws), n_tot, time.time() - t0


def multipage(imgs, max_tokens=12000):
    content = [{"type": "text", "text": "Multi page parsing."}]
    for p in imgs:
        content.append({"type": "image_url", "image_url": {"url": _encode_image(p)}})
    t0 = time.time()
    txt, n = stream(payload_for(content, max_tokens))
    return txt, n, time.time() - t0


def main():
    os.makedirs(OUT, exist_ok=True)
    imgs = render_pages(PDF, PAGES)
    gt = pdf_pseudo_gt(PDF, PAGES)
    base = gpu_mem_mib()
    print(f"pages={PAGES} ({len(imgs)} imgs)  base VRAM={base} MiB")

    pk = Peak()
    pk.start()
    pp_raw, pp_n, pp_t = page_by_page(imgs)
    pp_peak = pk.stop()
    pp_clean = strip_markup(pp_raw)
    open(f"{OUT}/page_by_page.txt", "w", encoding="utf-8").write(pp_clean)

    print("\n[1] page-by-page base done; running multipage...")
    try:
        pk2 = Peak()
        pk2.start()
        mp_raw, mp_n, mp_t = multipage(imgs, max_tokens=MP_MAX)
        mp_peak = pk2.stop()
    except requests.HTTPError as e:
        pk2.stop()
        print(f"\n!! MULTIPAGE REQUEST FAILED (multi-image likely unsupported): {e}")
        print(f"   response: {getattr(e.response, 'text', '')[:500]}")
        return
    mp_clean = strip_markup(mp_raw)
    open(f"{OUT}/multipage.txt", "w", encoding="utf-8").write(mp_clean)

    def row(label, raw, clean, n, t, peak):
        print(f"\n[{label}]")
        print(f"  vs GT   : CER={metrics.cer(gt, raw):.4f}  cov={metrics.coverage(gt, raw):.4f}  "
              f"order={metrics.order_insensitive_sim(gt, raw):.4f}")
        print(f"  output  : {len(clean)} chars / {n} tokens")
        print(f"  time    : {t:.1f}s  ({n / t:.1f} tok/s)")
        print(f"  peakVRAM: {peak} MiB (+{peak - base} over base)")

    row("page-by-page base", pp_raw, pp_clean, pp_n, pp_t, pp_peak)
    row("multipage base", mp_raw, mp_clean, mp_n, mp_t, mp_peak)
    print("\n[consistency: multipage vs page-by-page]")
    print(f"  CER={metrics.cer(pp_raw, mp_raw):.4f}  cov={metrics.coverage(pp_raw, mp_raw):.4f}  "
          f"order={metrics.order_insensitive_sim(pp_raw, mp_raw):.4f}")
    print(f"\nspeed: page-by-page {pp_t:.1f}s  vs  multipage {mp_t:.1f}s  "
          f"({pp_t / mp_t:.2f}x)")


if __name__ == "__main__":
    main()
