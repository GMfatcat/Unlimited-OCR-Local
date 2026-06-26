"""離線驗證 det 解析 / 純文字 / 畫框（不需 server）。"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fitz  # noqa: E402
from ocr_backends import draw_dets, load_display_image, parse_dets, strip_markup  # noqa: E402

REPO = "/mnt/c/Users/User/Desktop/project/unlimited-ocr"
MD = f"{REPO}/outputs/sglang/Unlimited-OCR_page_0001.md"
PDF = f"{REPO}/Unlimited-OCR.pdf"


def main():
    with open(MD, encoding="utf-8") as f:
        raw = f.read()

    dets = parse_dets(raw)
    clean = strip_markup(raw)
    print(f"raw chars={len(raw)}  dets={len(dets)}  clean chars={len(clean)}")
    print("labels:", [d[0] for d in dets])
    assert "<|det|>" not in clean and "<|ref|>" not in clean and "<PAGE>" not in clean, "markup leaked!"
    print("----- clean preview -----")
    print(clean[:300])

    # 渲染首頁並畫框
    doc = fitz.open(PDF)
    tmp = tempfile.mkdtemp(prefix="overlay_")
    pg = os.path.join(tmp, "p1.png")
    doc[0].get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72)).save(pg)
    doc.close()
    base = load_display_image(pg)
    out_img = draw_dets(base, dets)
    out_path = f"{REPO}/outputs/overlay_page1.png"
    out_img.save(out_path)
    print(f"overlay saved: {out_path}  size={out_img.size}")

    ok = len(dets) > 3 and len(clean) > 100 and "<|" not in clean
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
