"""
Transformers 路線冒煙測試。

- 從本地權重資料夾載入 Unlimited-OCR（attn_implementation="eager"，因為 config.use_mla=False
  且模型只提供 mha_eager 注意力實作）。
- 用 PyMuPDF 把測試 PDF 轉成 PNG。
- 單頁跑 infer (gundam: base_size=1024, image_size=640, crop_mode=True)。
- 多頁跑 infer_multi (base: image_size=1024)。
- 驗收：兩種輸出都必須非空。

環境變數：
  MODEL_DIR  本地權重路徑
  PDF        測試 PDF
  OUT        輸出根目錄
  MAX_PAGES  infer_multi 取前幾頁（預設 3，加快測試）
"""
import os
import sys
import tempfile
import time

import torch
import fitz  # PyMuPDF
from transformers import AutoModel, AutoTokenizer

MODEL_DIR = os.environ.get("MODEL_DIR", "/mnt/c/Users/User/Desktop/project/unlimited-ocr/unlimited-ocr-hf")
PDF = os.environ.get("PDF", "/mnt/c/Users/User/Desktop/project/unlimited-ocr/Unlimited-OCR.pdf")
OUT = os.environ.get("OUT", "/mnt/c/Users/User/Desktop/project/unlimited-ocr/outputs/transformers")
MAX_PAGES = int(os.environ.get("MAX_PAGES", "3"))


def pdf_to_images(pdf_path, dpi=300, max_pages=None):
    doc = fitz.open(pdf_path)
    tmp_dir = tempfile.mkdtemp(prefix="pdf_ocr_tf_")
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    paths = []
    for i, page in enumerate(doc):
        if max_pages is not None and i >= max_pages:
            break
        out = os.path.join(tmp_dir, f"page_{i + 1:04d}.png")
        page.get_pixmap(matrix=mat).save(out)
        paths.append(out)
    doc.close()
    return paths


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"[load] model from {MODEL_DIR}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model = model.eval().cuda()
    print(f"[load] done in {time.time() - t0:.1f}s; device={next(model.parameters()).device}")

    # 轉所有頁，單頁測試用第一頁；多頁測試用前 MAX_PAGES 頁
    all_imgs = pdf_to_images(PDF, dpi=300)
    print(f"[pdf] total pages={len(all_imgs)}")

    # ── 單頁 infer (gundam) ──
    single_out = os.path.join(OUT, "single_gundam")
    t0 = time.time()
    model.infer(
        tokenizer,
        prompt="<image>document parsing.",
        image_file=all_imgs[0],
        output_path=single_out,
        base_size=1024, image_size=640, crop_mode=True,
        max_length=32768,
        no_repeat_ngram_size=35, ngram_window=128,
        save_results=True,
    )
    single_md = _read_result(single_out)
    print(f"[single] gundam done in {time.time() - t0:.1f}s, chars={len(single_md)}")

    # ── 多頁 infer_multi (base) ──
    multi_imgs = all_imgs[:MAX_PAGES]
    multi_out = os.path.join(OUT, "multi_base")
    t0 = time.time()
    model.infer_multi(
        tokenizer,
        prompt="<image>Multi page parsing.",
        image_files=multi_imgs,
        output_path=multi_out,
        image_size=1024,
        max_length=32768,
        no_repeat_ngram_size=35, ngram_window=1024,
        save_results=True,
    )
    multi_md = _read_result(multi_out)
    print(f"[multi] base done ({len(multi_imgs)} pages), chars={len(multi_md)}")

    # ── 驗收 ──
    ok = len(single_md.strip()) > 0 and len(multi_md.strip()) > 0
    print("\n===== SINGLE (gundam) preview =====")
    print(single_md[:500])
    print("\n===== MULTI (base) preview =====")
    print(multi_md[:500])
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'} (single={len(single_md)} chars, multi={len(multi_md)} chars)")
    sys.exit(0 if ok else 1)


def _read_result(out_dir):
    """讀 output_path 下所有 .md/.txt 合併。"""
    texts = []
    for root, _, files in os.walk(out_dir):
        for name in sorted(files):
            if name.lower().endswith((".md", ".txt", ".mmd")):
                with open(os.path.join(root, name), encoding="utf-8") as f:
                    texts.append(f.read())
    return "\n".join(texts)


if __name__ == "__main__":
    main()
