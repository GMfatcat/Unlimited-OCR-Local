"""
驗證 UI 的 Transformers 批次後端：渲染測試 PDF 首頁，呼叫 ocr_backends.transformers_run
（會以子行程在 .venv-transformers 跑 model.infer），確認輸出非空。
可在任一 venv 執行（只需 PyMuPDF）；transformers 推論在子行程的 .venv-transformers 內進行。
注意：執行前請確保 SGLang server 已停（VRAM 不足以同時常駐）。
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fitz  # noqa: E402
from ocr_backends import transformers_run  # noqa: E402

PDF = os.environ.get("PDF", "/mnt/c/Users/User/Desktop/project/unlimited-ocr/Unlimited-OCR.pdf")


def main():
    doc = fitz.open(PDF)
    tmp = tempfile.mkdtemp(prefix="ui_tf_")
    img = os.path.join(tmp, "page_0001.png")
    doc[0].get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72)).save(img)
    doc.close()

    print("[ui-tf] running transformers_run(gundam) via subprocess ...")
    text = transformers_run(img, "gundam")
    print(f"[ui-tf] chars={len(text)}")
    print("----- preview -----")
    print(text[:400])
    ok = len(text.strip()) > 50 and "無法解析輸出" not in text
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
