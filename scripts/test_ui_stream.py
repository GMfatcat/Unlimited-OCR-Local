"""
驗證 UI 的 SGLang 串流後端：對運行中的 server 跑測試 PDF 首頁，
確認 sglang_stream 會逐 token yield 且輸出非空。需 server 已啟動。
在 .venv-sglang 內執行。
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr_backends import server_healthy, sglang_stream, to_image_paths  # noqa: E402

SERVER = os.environ.get("SERVER_URL", "http://127.0.0.1:10000")
PDF = os.environ.get("PDF", "/mnt/c/Users/User/Desktop/project/unlimited-ocr/Unlimited-OCR.pdf")


def main():
    assert server_healthy(SERVER), f"server not healthy at {SERVER}"
    with open(PDF, "rb") as f:
        images = to_image_paths(f.read(), "Unlimited-OCR.pdf", dpi=300)
    print(f"[ui-stream] pages={len(images)}, streaming page 1 (base) ...")

    deltas = 0
    chars = 0
    t0 = time.time()
    first_token_t = None
    acc = []
    for delta in sglang_stream(SERVER, images[0], "base", ngram_window=1024, ngram_size=35):
        if first_token_t is None:
            first_token_t = time.time() - t0
        deltas += 1
        chars += len(delta)
        acc.append(delta)

    text = "".join(acc)
    print(f"[ui-stream] deltas={deltas}, chars={chars}, ttft={first_token_t:.2f}s, total={time.time()-t0:.2f}s")
    print("----- preview -----")
    print(text[:400])
    ok = deltas > 5 and chars > 50
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
